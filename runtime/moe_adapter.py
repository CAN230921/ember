"""SM121 W4A4 MoE backend adapter: routes target MoE layers to the up-first kernels."""
from collections import Counter
import importlib.util
import os
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location('model_upfirst',os.path.join(os.environ.get('SPARK_LAB_DIR','/lab'),'moe_upfirst.py'))
leaf=importlib.util.module_from_spec(spec);spec.loader.exec_module(leaf)
leaf.ALIGN_ROUNDING=True
MODE='baseline'
DECODE=False
SHADOW=False
CALLS=Counter()
CHECKS=Counter()

def views(m):
    if not hasattr(m,'_experiment_views'):
        m._experiment_views=(m.resident_w13.view(torch.uint8).reshape(512,1280,1280),
            m.resident_sf13.view(torch.uint8).reshape(512,1280,160).view(torch.float8_e4m3fn),
            m.resident_alpha1.flatten(),m.resident_w2.view(torch.uint8).reshape(512,2560,320),
            m.resident_sf2.view(torch.uint8).reshape(512,2560,40).view(torch.float8_e4m3fn),
            m.resident_alpha2.flatten(),m.a1_inverse.reshape(1),m.a2_inverse.reshape(1))
    return m._experiment_views

def candidate(m,x,weights,ids):
    rows=x.shape[0]
    work=m.__dict__.setdefault('_experiment_work',{})
    if rows not in work:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('new MoE workspace must be warmed before capture')
        work[rows]=leaf.allocate_workspace(rows,2560,640,512,10,True,x.device)
    out=torch.empty_like(x)
    return leaf.run_moe_w4a4(x,ids.int().contiguous(),weights.float().contiguous(),
                           *views(m),out,work[rows])

def install():
    from engine.ops.moe import MoERef
    from engine.model import GreedyEngineRef
    from engine.batch_mtp_model import BatchMTPModel
    original=MoERef.packed_forward
    original_forward=GreedyEngineRef.forward
    original_deferred=GreedyEngineRef.verify_deferred
    original_tree=GreedyEngineRef.verify_tree
    original_batch=BatchMTPModel.verify
    def packed(m,x,weights,ids,active_experts=None,activation_quant=None):
        eligible=(DECODE and MODE=='candidate' and 1<=x.shape[0]<=12 and x.shape[1]==2560
            and m.resident_experts and m.top_k==10 and m.num_experts==512
            and active_experts is None and activation_quant is None
            and m.prefix.startswith('model.language_model.layers.'))
        if not eligible:
            CALLS[('fallback',MODE,x.shape[0])]+=1
            return original(m,x,weights,ids,active_experts,activation_quant)
        key=(m.prefix,x.shape[0]);CALLS[('candidate',x.shape[0])]+=1
        check=SHADOW and not torch.cuda.is_current_stream_capturing() and CHECKS[key]<2
        if check: ref=original(m,x,weights,ids)
        out=candidate(m,x,weights,ids)
        if check:
            if not torch.equal(ref.contiguous().view(torch.uint8),out.contiguous().view(torch.uint8)):
                raise RuntimeError('real hidden MoE mismatch '+str(key)+', maxabs='+str(float((ref.float()-out.float()).abs().max())))
            CHECKS[key]+=1
        return out
    def forward(m,input_ids,state=None,*args,**kwargs):
        global DECODE
        before=DECODE
        DECODE=bool(state is not None and state.tokens and not kwargs.get('prefill_chunk',False))
        try:return original_forward(m,input_ids,state,*args,**kwargs)
        finally:DECODE=before
    def batch(m,*args,**kwargs):
        global DECODE
        before=DECODE;DECODE=True
        try:return original_batch(m,*args,**kwargs)
        finally:DECODE=before
    def deferred(m,*args,**kwargs):
        global DECODE
        before=DECODE;DECODE=True
        try:return original_deferred(m,*args,**kwargs)
        finally:DECODE=before
    def tree(m,*args,**kwargs):
        global DECODE
        before=DECODE;DECODE=True
        try:return original_tree(m,*args,**kwargs)
        finally:DECODE=before
    MoERef.packed_forward=packed
    GreedyEngineRef.forward=forward
    GreedyEngineRef.verify_deferred=deferred
    GreedyEngineRef.verify_tree=tree
    BatchMTPModel.verify=batch

class ModeBanks:
    """Separate captured target graphs and pools; one unchanged resident weight bank."""
    def __init__(self,runtime):
        self.runtime=runtime;self.current='baseline';self.banks={}
        self.fields=('_decode_graphs','_graph_pool','_verify_graphs','_verify_graph_pools')
        self.batchfields=('graphs','pools','layouts')
    def switch(self,name):
        global MODE
        import engine.graph_capture as capture
        if name!=self.current:
            engine=self.runtime.engine;batch=self.runtime.batch_runner.batch_target
            self.banks[self.current]=(tuple(getattr(engine,f) for f in self.fields),
                tuple(getattr(batch,f) for f in self.batchfields),capture._shared)
            e,b,p=self.banks.get(name,(({},None,{},{}),({},{},{}),[]))
            for f,v in zip(self.fields,e):setattr(engine,f,v)
            for f,v in zip(self.batchfields,b):setattr(batch,f,v)
            capture._shared=p;self.current=name
        MODE=name
    def snapshot(self):
        self.banks[self.current]=(tuple(getattr(self.runtime.engine,f) for f in self.fields),
            tuple(getattr(self.runtime.batch_runner.batch_target,f) for f in self.batchfields),
            __import__('engine.graph_capture',fromlist=['_shared'])._shared)
        return {name:dict(decode_blocks=len(e[0]),verify_blocks=len(e[2]),batch_blocks=len(b[0]),
            batch_layouts=[list(k) for k in b[2]],private_pool_count=len(p)) for name,(e,b,p) in self.banks.items()}

def snapshot():
    return dict(mode=MODE,calls={str(k):v for k,v in CALLS.items()},
                real_hidden_checks={str(k):v for k,v in CHECKS.items()})
