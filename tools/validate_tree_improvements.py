"""Exact-weight proof and matched timing for bounded history/best-first trees."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from tree_generation import generate_tree
from validate_linear_verification import equal


def main(runtime,out):
    from engine.candidate_tree import CandidateTree
    from engine.dflash_draft import LAYERS
    from engine.prefix_cache import clone_state
    from engine.tokenization import chat_token_ids
    import moe_adapter
    engine=runtime.engine;draft=runtime.batch_runner.dflash
    result=dict(status='equivalence',checks=[],request_timings=[],release_approved=False,
        weight_index_sha256=hashlib.sha256(Path('/model/model.safetensors.index.json').read_bytes()).hexdigest())
    assert result['weight_index_sha256']=='4e01d7e4582e9f8238df392504bacfce399340b675693c35a4feeed62b043176'
    def save():
        p=out/'results.tmp';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(out/'results.json')
    save()
    seed=chat_token_ids(runtime.tokenizer,[dict(role='user',content=
        'Return a Python function that adds two integers. Include only code.')],False)
    filler=runtime.tokenizer.encode('The agent reads files and checks a patch. ',add_special_tokens=False)
    def prefill(ids):
        state=None;ctx=draft.new_context()
        for start in range(0,len(ids),2048):
            taps={}
            def observe(index,value):
                if isinstance(index,int) and index in LAYERS:taps[index]=draft.tap(*value)
            logits,state=engine(ids[start:start+2048],state,observer=observe,prefill_chunk=True)
            draft.extend(ctx,torch.cat([taps[i] for i in LAYERS],-1))
        assert ctx.length==len(state.tokens)==len(ids)
        return logits,state,ctx
    def check(tree,base,label):
        frozen=clone_state(base);expected={};features={};states={}
        for path in tree.paths():
            state=clone_state(base)
            for node in path:
                logits,state,hidden=engine([tree.tokens[node]],state,owned_graph_cache=True,return_hidden=True)
                expected[node]=logits[-1].clone();features[node]=hidden.multi[-1].clone();states[node]=clone_state(state)
        for budget in (1,2,8):
            before=sum(v for k,v in moe_adapter.CALLS.items() if k[0]=='candidate')
            tx=engine.verify_tree(tree,base)
            after=sum(v for k,v in moe_adapter.CALLS.items() if k[0]=='candidate')
            assert after>before,'Tree verifier missed the current MoE route'
            for node in range(len(tree.tokens)):
                equal(tx.logits[node],expected[node],'node logits')
                equal(tx.hidden.multi[node],features[node],'node hidden')
            selected=tx.select(budget,runtime.eos);actual=tx.commit(selected)
            wanted=states[selected.consumed_nodes[-1]]
            equal(actual,wanted,'committed state');equal(base,frozen,'base state')
            a,sa=engine([selected.emitted[-1]],clone_state(wanted),owned_graph_cache=True)
            a=a.clone();sa=clone_state(sa)
            b,sb=engine([selected.emitted[-1]],clone_state(actual),owned_graph_cache=True)
            equal(a,b,'continued logits');equal(sa,sb,'continued state')
            result['checks'].append(dict(label=label,context=len(base.tokens),budget=budget,nodes=len(tree.tokens),
                parents=tree.parents,leaves=len(tree.paths()),max_depth=max(map(len,tree.paths()))-1,
                all_node_logits_and_hidden_bit_equal=True,full_state_bit_equal=True,continuation_bit_equal=True,
                consumed_nodes=selected.consumed_nodes))
            save()
    for length in (len(seed),2081):
        ids=seed if length==len(seed) else (seed+filler*((length-len(seed))//len(filler)+1))[:length]
        logits,base,ctx=prefill(ids);anchor=int(logits[-1].argmax())
        candidates=[];state=clone_state(base);token=anchor
        for _ in range(8):
            candidates.append(token);rows,state=engine([token],state,owned_graph_cache=True);token=int(rows[-1].argmax())
        del state
        for policy in ('beam','best_first'):
            check(draft.propose_tree(ctx,anchor,policy=policy),base,'actual DFlash '+policy)
        c=candidates;v=engine.config['vocab_size']
        check(CandidateTree(c,(-1,0,1,2,3,4,5,6)),base,'eight-node deep chain')
        check(CandidateTree((c[0],c[1],(c[1]+1)%v,c[2],(c[2]+1)%v,(c[2]+2)%v,c[3],(c[3]+1)%v),
            (-1,0,0,1,1,2,3,5)),base,'eight-node asymmetric branches')
        check(CandidateTree((c[0],c[1],(c[1]+1)%v,c[2]),(-1,0,0,1)),base,'four-node boundary')
        check(CandidateTree((c[0],c[1],(c[1]+1)%v,c[2],(c[2]+1)%v,c[3]),(-1,0,0,1,2,3)),base,'six-node boundary')
        del base,ctx;torch.cuda.empty_cache()
    result['status']='request_timing';save()
    extended=chat_token_ids(runtime.tokenizer,[dict(role='user',content=
        'Implement a Python module for an agent: read JSON tool requests, dispatch read_file and apply_patch, '
        'and return JSON results. Include type hints, input validation, and three unit tests. Output only full code.')],False)
    backends=('baseline','beam_tree','best_first_tree')
    for label,prompt in (('short function',seed),('agent tool-dispatch module',extended)):
        samples={k:[] for k in backends};work={};expected=None
        def run(kind):
            nonlocal expected
            torch.cuda.synchronize();start=time.perf_counter()
            if kind=='baseline':row,_=runtime.generate(prompt,128,cache=False)
            else:row=generate_tree(runtime,prompt,128,policy='best_first' if kind=='best_first_tree' else 'beam')
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            if expected is None:expected=row['generated_ids']
            assert expected==row['generated_ids'],'Request generation changed token IDs'
            rounds=row.get('rounds',[])
            work[kind]=dict(rounds=len(rounds),nodes=sum(r['nodes'] for r in rounds),
                consumed_nodes=sum(len(r['consumed_nodes']) for r in rounds),
                max_depth=max((r['depth'] for r in rounds),default=0),reasons=dict(Counter(r['reason'] for r in rounds)))
            return seconds
        for kind in backends:run(kind)
        for repeat in range(4):
            for kind in (backends if repeat%2==0 else tuple(reversed(backends))):samples[kind].append(run(kind))
        medians={k:statistics.median(v) for k,v in samples.items()}
        result['request_timings'].append(dict(label=label,prompt_tokens=len(prompt),output_tokens=len(expected),
            samples_seconds=samples,median_seconds=medians,output_tokens_per_wall_second={k:len(expected)/v for k,v in medians.items()},
            tree_work=work,all_output_ids_identical=True,scope='single lane; includes prefill/draft; no HTTP; cache disabled'))
        save()
    result['status']='agent';save()
    from validate_tree_agent import benchmark
    agent_out=out/'agent';agent_out.mkdir()
    benchmark(runtime,agent_out,backends=backends,repeats=4)
    result['agent_status']=json.loads((agent_out/'results.json').read_text())['status']
    assert result['agent_status']=='passed'
    result.update(status='passed',peak_allocated_bytes=torch.cuda.max_memory_allocated(),quality_suite_run=False,http_suite_run=False)
    save();print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)


if __name__=='__main__':
    from serve import install
    install()
    from api_server import Runtime
    runtime=Runtime();out=Path('/out')
    try:
        with torch.inference_mode():runtime.load();main(runtime,out)
    except BaseException as error:
        p=out/'results.json';data=json.loads(p.read_text()) if p.exists() else {}
        data.update(status='failed',error=repr(error),release_approved=False)
        p.write_text(json.dumps(data,indent=2)+'\n');raise
    finally:runtime.close()
