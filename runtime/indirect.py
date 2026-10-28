"""Graph inputs use fixed GPU descriptors for request-owned GDN state addresses.

Only input loads change. The generated kernels retain every original operation,
store and BF16 recurrence boundary. Native prefill and non-lazy contracts keep
the original implementation.
"""
import ast
from collections import OrderedDict
import functools
import inspect
from pathlib import Path
import os
import weakref

import torch
import state_opt as opt

REGISTRY={}

def indexed_kernel(original,name,word):
    # Generate from the exact original kernel and fail closed if its load structure changes.
    tree=ast.parse(original.src)
    fn=tree.body[0];assert isinstance(fn,ast.FunctionDef)
    fn.name=name;fn.decorator_list=[]
    fn.args.args.append(ast.arg(arg='STATE_INPUT'))
    fn.args.defaults.append(ast.Constant(value=None))
    class Rewrite(ast.NodeTransformer):
        count=0
        def visit_Call(self,node):
            self.generic_visit(node)
            if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                and node.func.value.id=='tl' and node.func.attr=='load' and node.args):
                class InputPointer(ast.NodeTransformer):
                    count=0
                    def visit_Name(self,value):
                        if value.id=='S':
                            self.count+=1
                            return ast.Name(id='INPUT',ctx=ast.Load())
                        return value
                pointer=InputPointer();node.args[0]=pointer.visit(node.args[0]);self.count+=pointer.count
            return node
    rewrite=Rewrite();rewrite.visit(fn)
    assert rewrite.count==(3 if word==0 else 1),rewrite.count
    setup=ast.parse('INPUT = tl.multiple_of(tl.load(STATE_INPUT + '+str(word)+'), 16).to(tl.pointer_type(S.dtype.element_ty))').body
    fn.body=setup+fn.body;ast.fix_missing_locations(tree)
    source='@tr.jit\n'+ast.unparse(tree)+'\n'
    path=Path(os.environ.get('SPARK_LAB_DIR','/lab'))/(name+'.py')
    if path.exists():assert path.read_text()==source
    else:path.write_text(source)
    namespace=dict(original.fn.__globals__)
    exec(compile(source,str(path),'exec'),namespace)
    return namespace[name]

class Dispatch:
    def __init__(self,original,indexed,argument):
        self.original=original;self.indexed=indexed;self.argument=argument
    def __getitem__(self,grid):
        ordinary=self.original[grid];indexed=self.indexed[grid]
        def launch(*args,**kwargs):
            entry=REGISTRY.get(args[self.argument].data_ptr())
            if entry is None or entry[0]() is None:return ordinary(*args,**kwargs)
            return indexed(*args,STATE_INPUT=entry[1],**kwargs)
        return launch

def bind(graph,pairs):
    """Pairs are in fixed layer/lane order, independent of request scheduling order."""
    for source,target in pairs:
        for name in ('conv','recurrent'):
            a,b=getattr(source,name),getattr(target,name)
            if (not a.is_cuda or a.shape!=b.shape or a.dtype!=b.dtype or a.device!=b.device
                or not a.is_contiguous() or a.data_ptr()%16):
                raise ValueError('Indirect graph state contract changed')
    key=tuple(p for source,_ in pairs for p in (source.conv.data_ptr(),source.recurrent.data_ptr()))
    if not hasattr(graph,'_state_descriptors'):
        graph._state_descriptors=torch.empty((len(pairs),2),device=pairs[0][0].conv.device,dtype=torch.int64)
        graph._state_pointer_cache=OrderedDict();graph._state_pointer_key=None
        # Keep the original dummy tensors alive as shape/dtype anchors. Their
        # contents are never read or written by the optimized lazy graph.
        for row,(_,target) in enumerate(pairs):
            descriptor=graph._state_descriptors[row]
            for t in (target.conv,target.recurrent):
                ptr=t.data_ptr();entry=REGISTRY.get(ptr)
                assert entry is None or entry[0]() is None
                # A freed graph must never redirect a later allocation reusing
                # its address. The weak callback also drops the small descriptor.
                def expired(ref,key=ptr):
                    if key in REGISTRY and REGISTRY[key][0] is ref:REGISTRY.pop(key)
                REGISTRY[ptr]=(weakref.ref(t,expired),descriptor)
        opt.COUNTS['indirect_descriptor_banks']+=1
    if key!=graph._state_pointer_key:
        cache=graph._state_pointer_cache
        if key not in cache:
            cache[key]=torch.tensor(key,device=graph._state_descriptors.device,dtype=torch.int64).reshape(-1,2)
            opt.COUNTS['indirect_table_misses']+=1
            if len(cache)>16:cache.popitem(last=False)
        else:cache.move_to_end(key)
        graph._state_descriptors.copy_(cache[key]);graph._state_pointer_key=key
        opt.COUNTS['indirect_descriptor_updates']+=1
    opt.COUNTS['indirect_input_layers']+=len(pairs)

def install():
    import engine.ops.gdn_triton as G
    import engine.decode_graph as DG
    import engine.batch_mtp_model as BM
    G._conv=Dispatch(G._conv,indexed_kernel(G._conv,'_conv_indexed',0),1)
    G._decode=Dispatch(G._decode,indexed_kernel(G._decode,'_decode_indexed',1),5)
    old_copy=DG.DecodeGraphBlock._copy_inputs;old_forward=DG.DecodeGraphBlock.forward
    def copy(block,pending,states):
        if not getattr(block,'_state_indirect',False):return old_copy(block,pending,states)
        if len(pending)!=3 or len(states)!=len(block.states):raise ValueError('Graph input structure changed')
        for source,target in zip(pending,block.pending):
            if (source is None)!=(target is None):raise ValueError('Graph pending structure changed')
            if source is not None:
                if source.shape!=target.shape or source.dtype!=target.dtype or source.device!=target.device:
                    raise ValueError('Graph pending tensor layout changed')
                target.copy_(source)
        if any(s.ple is not None for s in states):raise ValueError('PLE cannot enter a GDN graph')
        bind(block,[(s.attention,t.attention) for s,t in zip(states,block.states)])
    @functools.wraps(old_forward)
    def forward(block,pending,states,borrowed=False,owned=False,lazy=False):
        want=(opt.MODE=='abc' and block.return_prefixes and lazy and not borrowed
              and not any(getattr(s.attention,'graph_view',False) for s in states))
        if block.graph is None:block._state_indirect=want
        elif getattr(block,'_state_indirect',False)!=want:
            variants=block.__dict__.setdefault('_indirect_variants',{})
            if want not in variants:
                variants[want]=DG.DecodeGraphBlock(block.layers,torch.cuda.graph_pool_handle(),block.return_prefixes)
            other=variants[want];result=forward(other,pending,states,borrowed,owned,lazy)
            block.intermediates=other.intermediates
            return result
        return old_forward(block,pending,states,borrowed,owned,lazy)
    DG.DecodeGraphBlock._copy_inputs=copy;DG.DecodeGraphBlock.forward=forward

    old_batch_copy=BM.BatchMTPGraph.copy_inputs
    def batch_copy(graph,pending,states):
        want=getattr(graph,'_state_indirect',False)
        if not want:return old_batch_copy(graph,pending,states)
        if len(pending)!=3 or len(states)!=len(graph.states):raise ValueError('Batch graph input structure changed')
        for value,destination in zip(pending,graph.pending):
            if (value is None)!=(destination is None):raise ValueError('Graph pending structure changed')
            if value is not None:
                if value.shape!=destination.shape or value.dtype!=destination.dtype or value.device!=destination.device:
                    raise ValueError('Graph pending tensor layout changed')
                destination.copy_(value)
        if any(len(s)!=len(graph.indices) or any(x.ple is not None for x in s) for s in states):
            raise ValueError('Batch graph state structure changed')
        bind(graph,[(s[i].attention,graph.states[lane][i].attention)
            for i in range(len(graph.indices)) for lane,s in enumerate(states)])
    BM.BatchMTPGraph.copy_inputs=batch_copy
    old_batch_call=BM.BatchMTPGraph.__call__
    def batch_call(graph,pending,states):
        want=(opt.MODE=='abc' and not any(getattr(s.attention,'graph_view',False)
              for lane in states for s in lane))
        if graph.graph is None:graph._state_indirect=want
        elif getattr(graph,'_state_indirect',False)!=want:
            variants=graph.__dict__.setdefault('_indirect_variants',{})
            if want not in variants:
                variants[want]=BM.BatchMTPGraph(graph.batch,graph.indices,graph.sections,torch.cuda.graph_pool_handle())
                opt.COUNTS['indirect_batch_contract_variants']+=1
            other=variants[want];result=batch_call(other,pending,states)
            graph.intermediates=other.intermediates
            return result
        return old_batch_call(graph,pending,states)
    BM.BatchMTPGraph.__call__=batch_call
