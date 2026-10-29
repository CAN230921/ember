"""Same virtual Agent fixtures through direct baseline/tree generation.

No network server or host tools: reuses the published in-memory fixture and
the real request normalization, tokenizer, output decoder and tool parser.
This does not validate HTTP/SSE, queueing, prefix caching or concurrency.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time
from collections import Counter

import torch

from tree_generation import generate_tree


def benchmark(runtime,out,*,backends=('baseline','kernel_tree'),repeats=4,cases=('patch_loop','inspect')):
    from api_protocol import OutputFramer,normalize_request
    from api_server import request_token_ids
    from engine.tokenization import prompt_starts_in_reasoning
    from output_decoder import IncrementalOutputDecoder
    spec=importlib.util.spec_from_file_location('tree_agent_fixture','/benchmarks/agent_suite.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    result=dict(status='running',release_approved=False,rows=[],comparison=[],
        transport='direct synchronous generation; no HTTP/SSE',lanes=1,prefix_cache=False,
        weight_index_sha256=hashlib.sha256(Path('/model/model.safetensors.index.json').read_bytes()).hexdigest())
    reference={};kind=None;name=None;turn=0
    def save():
        temp=out/'results.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(out/'results.json')
    def digest(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
    def chat(port,model,key,messages,with_tools):
        nonlocal turn
        body=dict(model=model,messages=messages,max_tokens=1024,temperature=0,
            chat_template_kwargs=dict(enable_thinking=False),tools=fixture.TOOLS if with_tools else [])
        options=normalize_request(body,model=model,max_output=0);ids=request_token_ids(runtime.tokenizer,options)
        torch.cuda.synchronize();start=time.perf_counter()
        if kind=='baseline':row,_=runtime.generate(ids,1024,cache=False)
        else:row=generate_tree(runtime,ids,1024,policy='best_first' if kind=='best_first_tree' else 'beam')
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        signatures=(digest(ids),digest(row['generated_ids']))
        index=(name,turn)
        if index not in reference:reference[index]=signatures
        assert reference[index]==signatures,'Agent turn prompt or generated token IDs changed'
        turn+=1
        decoder=IncrementalOutputDecoder(runtime.tokenizer,runtime.eos)
        framer=OutputFramer(options['tools'],starts_in_reasoning=prompt_starts_in_reasoning(runtime.tokenizer,ids))
        framer.feed(decoder.finish(row['generated_ids']),final=True)
        message=framer.message();calls=[]
        for i,call in enumerate(message.get('tool_calls',[])):
            calls.append(dict(id=f'tree_fixture_{len(messages)}_{i}',name=call['function']['name'],
                              arguments=call['function']['arguments']))
        rounds=row.get('rounds',[])
        work=dict(rounds=len(rounds),nodes=sum(r['nodes'] for r in rounds),
            consumed_nodes=sum(len(r['consumed_nodes']) for r in rounds),
            reasons=dict(Counter(r['reason'] for r in rounds)),max_depth=max((r['depth'] for r in rounds),default=0))
        return dict(first_visible_s=None,total_s=seconds,done=True,tree_work=work,
            finish_reason='tool_calls' if calls else row['finish_reason'],errors=[],
            usage=dict(prompt_tokens=len(ids),completion_tokens=len(row['generated_ids'])),
            content=message.get('content') or '',tool_calls=calls,
            prompt_ids_sha256=signatures[0],generated_ids_sha256=signatures[1])
    fixture.chat=chat
    save()
    for name in cases:
        samples={k:[] for k in backends};counts={}
        for repeat in range(-1,repeats):
            for kind in (tuple(samples) if repeat%2==0 or repeat==-1 else tuple(reversed(samples))):
                turn=0
                row=fixture.scenario(0,'current-checkpoint','',name)
                row.update(backend=kind,phase='warmup' if repeat==-1 else 'measure',repeat=repeat)
                result['rows'].append(row);save()
                assert row['passed'] and not row['tool_errors'],f'Agent fixture failed: {name} {kind}'
                if repeat>=0:samples[kind].append(row['total_s'])
                counts[kind]=sum(t['usage']['completion_tokens'] for t in row['turns'])
        medians={k:statistics.median(v) for k,v in samples.items()}
        result['comparison'].append(dict(case=name,samples_seconds=samples,median_seconds=medians,
            output_tokens=counts,output_tokens_per_wall_second={k:counts[k]/v for k,v in medians.items()},
            all_turn_prompts_and_outputs_identical=True,all_fixture_runs_passed=True))
        save()
    result['status']='passed';result['peak_allocated_bytes']=torch.cuda.max_memory_allocated();save()
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    from serve import install
    install()
    from api_server import Runtime
    runtime=Runtime();out=Path('/out')
    try:
        with torch.inference_mode():runtime.load();benchmark(runtime,out)
    except BaseException as error:
        p=out/'results.json';data=json.loads(p.read_text()) if p.exists() else {}
        data.update(status='failed',error=repr(error),release_approved=False)
        p.write_text(json.dumps(data,indent=2)+'\n');raise
    finally:runtime.close()
