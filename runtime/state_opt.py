"""GDN state-copy optimizations, installed after the MoE adapter.

A omits persistent final-cache writeback only for non-borrowed lazy verification.
B materializes accepted graph views into a request-private, reusable arena.
All copies use uint16 payloads; recurrence and projection arithmetic is unchanged.
The arena is internal mutable request state; exported checkpoints still clone it.
"""
import ast
from collections import Counter,OrderedDict
import functools
import inspect
from pathlib import Path
import os
import textwrap

import torch
import triton as tr
import triton.language as tl

MODE='base'
COUNTS=Counter()
BLOCK=4096

@tr.jit
def _gather_bits(PTRS,CONV,REC,CN:tl.constexpr,RN:tl.constexpr,B:tl.constexpr):
    layer=tl.program_id(0);tile=tl.program_id(1)
    rec_tiles:tl.constexpr=(RN+B-1)//B
    if tile<rec_tiles:
        raw=tl.multiple_of(tl.load(PTRS+layer*2+1),16)
        src=raw.to(tl.pointer_type(tl.uint16))
        off=tile*B+tl.arange(0,B)
        value=tl.load(src+off,off<RN,0)
        tl.store(REC+layer*RN+off,value,off<RN)
    else:
        raw=tl.multiple_of(tl.load(PTRS+layer*2),16)
        src=raw.to(tl.pointer_type(tl.uint16))
        off=(tile-rec_tiles)*B+tl.arange(0,B)
        value=tl.load(src+off,off<CN,0)
        tl.store(CONV+layer*CN+off,value,off<CN)

class Arena:
    def __init__(self,items):
        self.indices=tuple(i for i,_ in items)
        sample=items[0][1];self.conv_shape=sample.conv.shape;self.rec_shape=sample.recurrent.shape
        self.cn=sample.conv.numel();self.rn=sample.recurrent.numel();n=len(items)
        self.conv=torch.empty((n,self.cn),dtype=sample.conv.dtype,device=sample.conv.device)
        self.recurrent=torch.empty((n,self.rn),dtype=sample.recurrent.dtype,device=sample.recurrent.device)
        self.conv_views=tuple(self.conv[j].view(self.conv_shape) for j in range(n))
        self.rec_views=tuple(self.recurrent[j].view(self.rec_shape) for j in range(n))
        self.pointer_tables=OrderedDict()
        COUNTS['arena_allocations']+=1
    def compatible(self,items):
        return (tuple(i for i,_ in items)==self.indices and all(
            a.conv.shape==self.conv_shape and a.recurrent.shape==self.rec_shape
            and a.conv.device==self.conv.device and a.recurrent.device==self.recurrent.device
            and a.conv.dtype==self.conv.dtype and a.recurrent.dtype==self.recurrent.dtype for _,a in items))
    def copy(self,items):
        key=tuple(p for _,a in items for p in (a.conv.data_ptr(),a.recurrent.data_ptr()))
        if key not in self.pointer_tables:
            # A blocking tiny upload on a cache miss avoids overwriting an in-flight
            # pinned host descriptor. Graph source banks remain owned by the engine.
            self.pointer_tables[key]=torch.tensor(key,dtype=torch.int64,device=self.conv.device)
            COUNTS['pointer_table_misses']+=1
            if len(self.pointer_tables)>16:self.pointer_tables.popitem(last=False)
        else:
            self.pointer_tables.move_to_end(key);COUNTS['pointer_table_hits']+=1
        device_ptrs=self.pointer_tables[key]
        _gather_bits[(len(items),tr.cdiv(self.rn,BLOCK)+tr.cdiv(self.cn,BLOCK))](
            device_ptrs,self.conv.view(torch.uint16),self.recurrent.view(torch.uint16),
            self.cn,self.rn,BLOCK,num_warps=4)
        COUNTS['gather_launches']+=1
        COUNTS['payload_bytes']+=len(items)*(self.cn+self.rn)*2

_original_materialize=None
def commit(state,owner):
    if MODE not in ('b','ab','abc'):return _original_materialize(state)
    items=[(i,l.attention) for i,l in enumerate(getattr(state,'layers',())) if getattr(getattr(l,'attention',None),'graph_view',False)]
    if not items:return state
    if (owner is None or not all(
        a.conv.is_cuda and a.recurrent.is_cuda and a.conv.dtype==torch.bfloat16
        and a.recurrent.dtype==torch.bfloat16 and a.conv.is_contiguous() and a.recurrent.is_contiguous()
        and a.conv.data_ptr()%16==0 and a.recurrent.data_ptr()%16==0
        and a.conv.shape==items[0][1].conv.shape and a.recurrent.shape==items[0][1].recurrent.shape
        and a.conv.device==items[0][1].conv.device and a.recurrent.device==items[0][1].recurrent.device for _,a in items)
        or torch.cuda.is_current_stream_capturing()):
        COUNTS['unsupported_materialize_fallback']+=1
        return _original_materialize(state)
    arena=getattr(owner,'_state_opt_arena',None)
    if arena is None or not arena.compatible(items):
        arena=owner._state_opt_arena=Arena(items)
    arena.copy(items)
    from engine.ops.gdn import GDNState
    for j,(i,_) in enumerate(items):
        state.layers[i].attention=GDNState(arena.conv_views[j],arena.rec_views[j])
    COUNTS['committed_layers']+=len(items)
    return state

def install():
    global _original_materialize
    import engine.decode_graph as DG
    import engine.batched_mtp as BM
    import engine.batch_mtp_model as BMM
    import engine.continuous_mtp as CM
    _original_materialize=BMM.materialize_views
    original_run=DG.DecodeGraphBlock._run;original_forward=DG.DecodeGraphBlock.forward
    def run(block):
        if not getattr(block,'_state_opt_skip_writeback',False):return original_run(block)
        assert block.return_prefixes
        value=block.pending;intermediate=[]
        for layer,state in zip(block.layers,block.states):
            value,updated=layer(value[0],None,value[1],value[2],state=state,prefill=False,return_prefixes=True)
            intermediate.append(value);state.prefixes=updated.prefixes
        block.intermediates=intermediate
        return value
    @functools.wraps(original_forward)
    def forward(block,pending,states,borrowed=False,owned=False,lazy=False):
        want=MODE in ('a','ab','abc') and block.return_prefixes and lazy and not borrowed
        if block.graph is None:block._state_opt_skip_writeback=want
        elif getattr(block,'_state_opt_skip_writeback',False)!=want:
            # A distinct graph contract, never silently replay a graph captured
            # without writeback for borrowed/ordinary semantics.
            variants=block.__dict__.setdefault('_state_opt_variants',{})
            if want not in variants:
                variants[want]=DG.DecodeGraphBlock(block.layers,torch.cuda.graph_pool_handle(),block.return_prefixes)
                COUNTS['contract_variants']+=1
            other=variants[want]
            result=forward(other,pending,states,borrowed,owned,lazy)
            block.intermediates=other.intermediates
            return result
        if want:COUNTS['skip_writeback_calls']+=1
        return original_forward(block,pending,states,borrowed,owned,lazy)
    DG.DecodeGraphBlock._run=run;DG.DecodeGraphBlock.forward=forward

    fn=inspect.unwrap(BM.BatchedMTPRunner.decode_batch)
    assert not fn.__code__.co_freevars
    tree=ast.parse(textwrap.dedent(inspect.getsource(fn)))
    class Rewrite(ast.NodeTransformer):
        count=0
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='materialize_views':
                assert len(node.args)==1 and not node.keywords
                node.func=ast.Name(id='_state_opt_commit',ctx=ast.Load())
                node.args.append(ast.Name(id='session',ctx=ast.Load()));self.count+=1
            return node
    rewrite=Rewrite();rewrite.visit(tree);assert rewrite.count==1
    ast.fix_missing_locations(tree)
    generated=Path(os.environ.get('SPARK_LAB_DIR','/lab'))/'generated_decode_batch.py'
    source=ast.unparse(tree)+'\n'
    if not generated.exists():generated.write_text(source)
    else:assert generated.read_text()==source
    BM.__dict__['_state_opt_commit']=commit
    namespace={};exec(compile(source,str(generated),'exec'),BM.__dict__,namespace)
    BM.BatchedMTPRunner.decode_batch=namespace['decode_batch']
    original_close=CM.ContinuousMTPRunner.close
    def close(runner,session):
        try:return original_close(runner,session)
        finally:session.__dict__.pop('_state_opt_arena',None)
    CM.ContinuousMTPRunner.close=close
    import indirect
    indirect.install()

class ModeBanks:
    def __init__(self,runtime):
        self.runtime=runtime;self.current='base';self.banks={}
        self.fields=('_decode_graphs','_graph_pool','_verify_graphs','_verify_graph_pools')
        self.batchfields=('graphs','pools','layouts')
    def switch(self,name):
        global MODE
        import engine.graph_capture as capture
        if name!=self.current:
            engine=self.runtime.engine;batch=self.runtime.batch_runner.batch_target
            self.banks[self.current]=(tuple(getattr(engine,f) for f in self.fields),tuple(getattr(batch,f) for f in self.batchfields),capture._shared)
            e,b,p=self.banks.get(name,(({},None,{},{}),({},{},{}),[]))
            for f,v in zip(self.fields,e):setattr(engine,f,v)
            for f,v in zip(self.batchfields,b):setattr(batch,f,v)
            capture._shared=p;self.current=name
        MODE=name
    def snapshot(self):
        self.banks[self.current]=(tuple(getattr(self.runtime.engine,f) for f in self.fields),tuple(getattr(self.runtime.batch_runner.batch_target,f) for f in self.batchfields),__import__('engine.graph_capture',fromlist=['_shared'])._shared)
        return {name:dict(decode_blocks=len(e[0]),verify_blocks=len(e[2]),batch_blocks=len(b[0]),batch_layouts=[list(k) for k in b[2]],private_pool_count=len(p)) for name,(e,b,p) in self.banks.items()}
