"""Exact-current-weight model comparison; isolated process, no HTTP listener."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

_runtime_for_cleanup=None

def equal(a,b,path='state'):
    from engine.ops.kv_cache import PagedKV
    if isinstance(a,PagedKV) or isinstance(b,PagedKV):
        equal(a.to_dense() if isinstance(a,PagedKV) else a,
              b.to_dense() if isinstance(b,PagedKV) else b,path+'.packed')
    elif isinstance(a,torch.Tensor):
        if a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)):
            raise AssertionError('Tensor differs: '+path)
    elif dataclasses.is_dataclass(a):
        for f in dataclasses.fields(a):equal(getattr(a,f.name),getattr(b,f.name),path+'.'+f.name)
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+f'[{i}]')
    else:
        assert a==b,path


def main():
    global _runtime_for_cleanup
    out=Path('/out');result=dict(status='loading',release_approved=False,checks=[],timings=[],tree_checks=[])
    def save():
        temp=out/'results.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(out/'results.json')
    save()
    from serve import install
    install()
    from api_server import Runtime
    from engine.prefix_cache import clone_state
    from engine.tokenization import chat_token_ids
    from engine.candidate_tree import CandidateTree
    import state_opt
    import moe_adapter
    runtime=Runtime();_runtime_for_cleanup=runtime;runtime.load();engine=runtime.engine
    result['weight_index_sha256']=hashlib.sha256(Path('/model/model.safetensors.index.json').read_bytes()).hexdigest()
    result['status']='equivalence';save()
    def check_tree(tree,base,label):
        expected_logits={};expected_states={};expected_hidden={}
        for path in tree.paths():
            current=clone_state(base)
            for node in path:
                logits,current,hidden=engine([tree.tokens[node]],current,owned_graph_cache=True,return_hidden=True)
                expected_logits[node]=logits[-1].clone()
                expected_hidden[node]=hidden.multi[-1].clone()
                expected_states[node]=clone_state(current)
        for implementation,verify in (('expanded-path reference',engine.verify_tree_reference),
                                      ('unique nodes, shared projections',engine.verify_tree)):
            tx=verify(tree,base)
            for node in range(len(tree.tokens)):
                equal(tx.logits[node],expected_logits[node],'tree node logits')
                equal(tx.hidden.multi[node],expected_hidden[node],'tree node hidden')
            selected=tx.select(8,runtime.eos);updated=tx.commit(selected)
            equal(updated,expected_states[selected.consumed_nodes[-1]],'tree committed state')
            result['tree_checks'].append(dict(label=label,context=len(base.tokens),nodes=len(tree.tokens),
                leaves=len(tree.paths()),consumed_nodes=selected.consumed_nodes,emitted=selected.emitted,
                all_node_logits_bit_equal=True,all_node_hidden_bit_equal=True,state_bit_equal=True,
                implementation=getattr(tx,'implementation',implementation)))
            save()
    messages=[dict(role='user',content='Return a Python function that adds two integers. Include only code.')]
    seed=chat_token_ids(runtime.tokenizer,messages,False)
    filler=runtime.tokenizer.encode('The agent reads files and checks a patch. ',add_special_tokens=False)
    for length in (len(seed),2081):
        ids=seed if length==len(seed) else (seed+filler*((length-len(seed))//len(filler)+1))[:length]
        logits,base=engine.prefill(ids)
        baseline=clone_state(base)
        candidates=[];token=int(logits[-1].argmax());s=clone_state(base)
        for _ in range(8):
            candidates.append(token)
            logits,s=engine([token],s,owned_graph_cache=True)
            token=int(logits[-1].argmax())
        del s
        owner=SimpleNamespace()
        for rows in (() if os.environ.get('SPARK_VALIDATE_TREE_ONLY')=='1' else (3,8)):
            proposed=candidates[:rows]
            for count in range(1,rows+1):
                logits,states,hidden=engine.verify_tokens(proposed,base,lazy_prefixes=True)
                old=state_opt.commit(states.prefixes[count-1],owner)
                expected_logits=logits.clone();expected_hidden=hidden.multi.clone()
                calls_before=sum(v for k,v in moe_adapter.CALLS.items() if k[0]=='candidate')
                tx=engine.verify_deferred(proposed,base)
                calls_after=sum(v for k,v in moe_adapter.CALLS.items() if k[0]=='candidate')
                assert calls_after>calls_before,'Deferred entry missed the candidate MoE route'
                equal(tx.logits,expected_logits,'logits')
                equal(tx.hidden.multi,expected_hidden,'hidden')
                retained=tx.retained_gdn_bytes
                new=tx.commit(count)
                equal(new,old)
                equal(base,baseline,'base')
                a,sa=engine([candidates[count] if count<8 else token],old,owned_graph_cache=True)
                a=a.clone();sa=clone_state(sa)
                b,sb=engine([candidates[count] if count<8 else token],new,owned_graph_cache=True)
                equal(a,b,'continued logits');equal(sa,sb,'continued state')
                result['checks'].append(dict(context=len(ids),rows=rows,accepted=count,
                    logits_bit_equal=True,state_bit_equal=True,continuation_bit_equal=True,
                    retained_gdn_bytes=retained))
                save()
            # Discard and supersession must not poison subsequent ordinary calls.
            tx=engine.verify_deferred(proposed,base);tx.discard()
            stale=engine.verify_deferred(proposed,base)
            engine([candidates[0]],clone_state(base),owned_graph_cache=True)
            assert stale.status=='discarded'
            try:stale.commit(1)
            except RuntimeError:pass
            else:raise AssertionError('Superseded commit was allowed')
            # Same inputs/accept count, warmed graphs, alternating order.
            count=min(3,rows);samples={'baseline':[],'deferred':[]}
            def step(kind):
                if kind=='baseline':
                    _,states,_=engine.verify_tokens(proposed,base,lazy_prefixes=True)
                    return state_opt.commit(states.prefixes[count-1],owner)
                tx=engine.verify_deferred(proposed,base)
                return tx.commit(count)
            for _ in range(2):step('baseline');step('deferred')
            for repeat in range(6):
                for kind in (('baseline','deferred') if repeat%2==0 else ('deferred','baseline')):
                    torch.cuda.synchronize();begin=time.perf_counter();value=step(kind)
                    torch.cuda.synchronize();samples[kind].append(time.perf_counter()-begin);del value
            result['timings'].append(dict(context=len(ids),rows=rows,accepted=count,
                samples_seconds=samples,median_seconds={k:statistics.median(v) for k,v in samples.items()},
                scope='target verification plus accepted-state commit; excludes draft and HTTP'))
            save()
        equal(base,baseline,'base after equivalence')
        tree=CandidateTree((candidates[0],candidates[1],(candidates[1]+1)%engine.config['vocab_size'],
                            candidates[2],(candidates[2]+1)%engine.config['vocab_size']),(-1,0,0,1,2))
        check_tree(tree,base,'two-branch fixture')
        reversed_tree=CandidateTree((candidates[0],(candidates[1]+1)%engine.config['vocab_size'],
            candidates[1],(candidates[2]+1)%engine.config['vocab_size'],candidates[2]),(-1,0,0,1,2))
        check_tree(reversed_tree,base,'target follows second branch')
        del base,baseline;torch.cuda.empty_cache()
    # The same retained DFlash2 weights produce the proposal lattice.
    from engine.dflash_draft import LAYERS
    drafter=runtime.batch_runner.dflash;taps={}
    def observe(index,value):
        if isinstance(index,int) and index in LAYERS:taps[index]=drafter.tap(*value)
    logits,base=engine(seed,observer=observe,prefill_chunk=True)
    ctx=drafter.new_context()
    drafter.extend(ctx,torch.cat([taps[i] for i in LAYERS],-1))
    tree=drafter.propose_tree(ctx,int(logits[-1].argmax()))
    check_tree(tree,base,'current DFlash2 proposal lattice')
    if os.environ.get('SPARK_VALIDATE_TREE_ONLY')=='1':
        from tree_generation import generate_tree
        torch.cuda.synchronize();begin=time.perf_counter()
        baseline,_=runtime.generate(seed,128,cache=False)
        torch.cuda.synchronize();baseline_seconds=time.perf_counter()-begin
        torch.cuda.synchronize();begin=time.perf_counter()
        generated=generate_tree(runtime,seed,128)
        torch.cuda.synchronize();tree_seconds=time.perf_counter()-begin
        assert baseline['generated_ids']==generated['generated_ids'],'Tree generation changed target output'
        result['generation_smoke']=dict(tokens=len(generated['generated_ids']),identical_target_tokens=True,
            finish_reason=generated['finish_reason'],baseline_seconds=baseline_seconds,tree_seconds=tree_seconds,
            tree_rounds=len(generated['rounds']),scope='one code-generation response; cold experimental tree; not Agent TPS')
        if os.environ.get('SPARK_TREE_TIMING')=='1':
            extended=chat_token_ids(runtime.tokenizer,[dict(role='user',content=
                'Implement a Python module for an agent: read JSON tool requests, dispatch read_file and apply_patch, '
                'and return JSON results. Include type hints, input validation, and three unit tests. Output only full code.')],False)
            result['generation_timings']=[]
            for label,prompt in (('short function',seed),('agent tool-dispatch module',extended)):
                samples={kind:[] for kind in ('baseline','eager_tree','kernel_tree')}
                expected=None;rounds={}
                def run(kind):
                    nonlocal expected
                    os.environ['SPARK_TREE_GDN_KERNEL']='1' if kind=='kernel_tree' else '0'
                    torch.cuda.synchronize();start=time.perf_counter()
                    if kind=='baseline':row,_=runtime.generate(prompt,128,cache=False)
                    else:row=generate_tree(runtime,prompt,128)
                    torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    if expected is None:expected=row['generated_ids']
                    assert row['generated_ids']==expected,'Timed tree generation changed output'
                    rounds[kind]=len(row.get('rounds',[]))
                    return elapsed
                for kind in samples:run(kind)
                for repeat in range(4):
                    for kind in (tuple(samples) if repeat%2==0 else tuple(reversed(samples))):
                        samples[kind].append(run(kind))
                medians={k:statistics.median(v) for k,v in samples.items()}
                result['generation_timings'].append(dict(label=label,prompt_tokens=len(prompt),output_tokens=len(expected),
                    samples_seconds=samples,median_seconds=medians,rounds=rounds,identical_target_tokens=True,
                    median_output_tokens_per_wall_second={k:len(expected)/v for k,v in medians.items()},
                    scope='one lane, full request including prefill and draft; no HTTP, cache disabled; not multi-step Agent'))
                save()
    result.update(status='passed',peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  quality_suite_run=False,agent_suite_run=False)
    save();print(json.dumps(result),flush=True)


if __name__=='__main__':
    try:
        with torch.inference_mode():main()
    except BaseException as error:
        p=Path('/out/results.json')
        x=json.loads(p.read_text()) if p.exists() else {}
        x.update(status='failed',error=repr(error),release_approved=False)
        p.write_text(json.dumps(x,indent=2)+'\n')
        raise
    finally:
        if _runtime_for_cleanup is not None:_runtime_for_cleanup.close()
