"""Bounded tree diagnosis. Instruments this process only; no engine changes."""
from collections import Counter,defaultdict
from contextlib import nullcontext
import functools
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from tree_generation import generate_tree


class Instrumentation:
    def __init__(self):self.saved=[];self.depth=Counter()
    def wrap(self,obj,name,label):
        original=getattr(obj,name)
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            key=label(*args,**kwargs) if callable(label) else label
            active=self.depth[key]>0;self.depth[key]+=1
            try:
                with nullcontext() if active else torch.profiler.record_function(key):
                    return original(*args,**kwargs)
            finally:self.depth[key]-=1
        self.saved.append((obj,name,original));setattr(obj,name,wrapped)
    def install(self,engine,draft):
        import engine.tree_model as model
        import engine.candidate_tree as candidates
        from engine.tree_verification import TreeVerification
        from engine.ops.hyperconnection import GatedResidualRef
        from engine.ops.moe import MoERef
        self.wrap(model,'attention_nodes',lambda layer,*a,**k:
            'component.gdn' if layer.layer_type=='linear_attention' else 'component.qsa')
        self.wrap(model,'ple_nodes','component.ple')
        self.wrap(MoERef,'forward','component.moe')
        for name in ('mix','combine','combine_and_mix'):self.wrap(GatedResidualRef,name,'component.hc')
        self.wrap(engine,'logits','component.head')
        self.wrap(TreeVerification,'__init__','tree.acceptance_io')
        self.wrap(candidates,'tree_from_lattice','draft.cpu_beam')
        self.wrap(draft,'_propose_core','draft.network')
    def close(self):
        for obj,name,original in reversed(self.saved):setattr(obj,name,original)
        self.saved=[]


def main(runtime,out):
    from engine.dflash_draft import LAYERS
    from engine.tokenization import chat_token_ids
    import moe_adapter
    result=dict(status='running',release_approved=False,scope='single lane; current tree prototype; no HTTP',
        weight_index_sha256=hashlib.sha256(Path('/model/model.safetensors.index.json').read_bytes()).hexdigest())
    assert result['weight_index_sha256']=='4e01d7e4582e9f8238df392504bacfce399340b675693c35a4feeed62b043176'
    def save():
        p=out/'profile.tmp';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(out/'profile.json')
    save()
    prompt=chat_token_ids(runtime.tokenizer,[dict(role='user',content=
        'Implement a Python module for an agent: read JSON tool requests, dispatch read_file and apply_patch, '
        'and return JSON results. Include type hints, input validation, and three unit tests. Output only full code.')],False)
    expected=None;samples={k:[] for k in ('baseline','kernel_tree')};last=None
    def generate(kind):
        nonlocal expected,last
        torch.cuda.synchronize();start=time.perf_counter()
        if kind=='baseline':row,_=runtime.generate(prompt,128,cache=False)
        else:row=generate_tree(runtime,prompt,128);last=row
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        if expected is None:expected=row['generated_ids']
        assert expected==row['generated_ids'],'Generation output changed'
        return elapsed
    for kind in samples:generate(kind)
    for repeat in range(3):
        for kind in (tuple(samples) if repeat%2==0 else tuple(reversed(samples))):
            samples[kind].append(generate(kind))
    result['unprofiled_request']=dict(prompt_tokens=len(prompt),output_tokens=len(expected),samples_seconds=samples,
        median_seconds={k:statistics.median(v) for k,v in samples.items()},all_output_ids_identical=True)
    rounds=last['rounds'];nodes=sum(r['nodes'] for r in rounds);consumed=sum(len(r['consumed_nodes']) for r in rounds)
    result['tree_work']=dict(rounds=len(rounds),nodes=nodes,consumed_nodes=consumed,
        consumed_node_fraction=consumed/nodes,emitted_per_round=sum(r['emitted'] for r in rounds)/len(rounds),
        reasons=dict(Counter(r['reason'] for r in rounds)),rows=rounds)
    save()
    engine=runtime.engine;draft=runtime.batch_runner.dflash
    def prepare():
        taps={}
        def observe(index,value):
            if isinstance(index,int) and index in LAYERS:taps[index]=draft.tap(*value)
        logits,state=engine(prompt,observer=observe,prefill_chunk=True)
        ctx=draft.new_context();draft.extend(ctx,torch.cat([taps[i] for i in LAYERS],-1))
        return state,ctx,int(logits[-1].argmax())
    def two_rounds(state,ctx,anchor,profiled=False):
        output=[anchor];details=[]
        scope=torch.profiler.record_function if profiled else lambda name:nullcontext()
        for _ in range(2):
            with scope('phase.draft'):tree=draft.propose_tree(ctx,output[-1])
            taps={}
            def observe(index,value):
                if isinstance(index,int) and index in LAYERS:taps[index]=draft.tap(*value)
            with scope('phase.target'):tx=engine.verify_tree(tree,state,observe)
            try:
                with scope('phase.select'):selection=tx.select(32,runtime.eos)
                with scope('phase.commit'):state=tx.commit(selection)
            finally:
                if tx.status=='open':tx.discard()
            with scope('phase.features'):
                indices=torch.tensor(selection.consumed_nodes,device=engine.embedding.device,dtype=torch.long)
                draft.push(ctx,torch.cat([taps[i].index_select(0,indices) for i in LAYERS],-1))
            output.extend(selection.emitted)
            details.append(dict(nodes=len(tree.tokens),leaves=len(tree.paths()),emitted=len(selection.emitted)))
        return output,details
    controls=[];control_ids=None
    for _ in range(3):
        state,ctx,anchor=prepare();torch.cuda.synchronize();start=time.perf_counter()
        ids,details=two_rounds(state,ctx,anchor);torch.cuda.synchronize();controls.append(time.perf_counter()-start)
        if control_ids is None:control_ids=ids
        assert control_ids==ids
        del state,ctx
    state,ctx,anchor=prepare();torch.cuda.synchronize()
    instrumentation=Instrumentation();instrumentation.install(engine,draft)
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=False,profile_memory=False,with_stack=False) as profile:
            start=time.perf_counter();ids,details=two_rounds(state,ctx,anchor,True)
            torch.cuda.synchronize();profiled_seconds=time.perf_counter()-start
        assert ids==control_ids,'Profiling changed outputs'
        profile.export_chrome_trace(str(out/'tree-trace.json'))
        scopes=[]
        for event in profile.key_averages():
            if event.key.startswith(('component.','phase.','tree.','draft.')):
                scopes.append(dict(name=event.key,device_type=str(event.device_type),count=event.count,cpu_total_us=event.cpu_time_total,
                    self_cpu_us=event.self_cpu_time_total,device_total_us=event.device_time_total))
        scopes.sort(key=lambda r:r['device_total_us'],reverse=True)
        device=[];intervals=[]
        for event in profile.events():
            if event.device_type==torch.autograd.DeviceType.CUDA:
                # CUDA user annotations project a host scope onto the stream;
                # their spans include gaps and are not active device work.
                if event.name.startswith(('component.','phase.','tree.','draft.')):continue
                duration=event.time_range.elapsed_us();device.append((event.name,duration))
                intervals.append((event.time_range.start,event.time_range.end))
        aggregate=defaultdict(lambda:dict(count=0,total_us=0.))
        for name,duration in device:aggregate[name]['count']+=1;aggregate[name]['total_us']+=duration
        union=0.;end=-1.
        for start,stop in sorted(intervals):
            union+=max(0.,stop-max(end,start));end=max(end,stop)
        result['profile']=dict(unprofiled_two_round_seconds=controls,profiled_seconds=profiled_seconds,
            profiler_slowdown=profiled_seconds/statistics.median(controls),outputs_identical=True,rounds=details,
            scopes=scopes,device_event_count=len(device),device_work_union_us=union,
            top_device_events=sorted([dict(name=n,**d) for n,d in aggregate.items()],key=lambda x:x['total_us'],reverse=True)[:25],
            caution='Scope times are inclusive; do not sum overlapping scopes or treat profiler dilation as production idle time.')
    finally:instrumentation.close()
    save()
    # Complete the previously interrupted check with the repaired tool adapter.
    import validate_tree_agent
    agent_out=out/'agent';agent_out.mkdir()
    try:
        validate_tree_agent.benchmark(runtime,agent_out)
        result['agent_status']=json.loads((agent_out/'results.json').read_text())['status']
    except Exception as error:
        result['agent_status']='failed';result['agent_error']=repr(error)
    result.update(status='passed',quality_suite_run=False,http_suite_run=False,
        moe_mode=moe_adapter.MODE,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    save();print(json.dumps({k:v for k,v in result.items() if k not in ('tree_work','profile')}),flush=True)


if __name__=='__main__':
    from serve import install
    install()
    from api_server import Runtime
    runtime=Runtime();out=Path('/out')
    try:
        with torch.inference_mode():runtime.load();main(runtime,out)
    except BaseException as error:
        p=out/'profile.json';value=json.loads(p.read_text()) if p.exists() else {}
        value.update(status='failed',error=repr(error),release_approved=False)
        p.write_text(json.dumps(value,indent=2)+'\n');raise
    finally:runtime.close()
