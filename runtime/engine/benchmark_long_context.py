"""Actual long-text generation with genuine sparse QSA and chunked prefill.

Stores exact input IDs for a separately loaded original-runner comparison.
This is a small retrieval smoke test, not a long-context quality benchmark.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from transformers import AutoTokenizer

from engine.model import GreedyEngineRef
from engine.tokenization import chat_token_ids, generation_eos_ids
from engine.weights import WeightStore


def case_for_length(tokenizer,minimum):
    count=1
    while True:
        ledger='\n'.join(f'Record {i:04d}: warehouse item {i*17%997:03d}, shelf {i%23:02d}, tag amber.'
                          for i in range(count))
        prompt=('Read the archive and answer the final question. The permanent code ALPHA is quartz731.\n'
                +ledger+'\nThe permanent code OMEGA is maple284.\n'
                'Return only a JSON object with the two permanent codes under keys ALPHA and OMEGA. '
                'Do not include any record numbers or explanation.')
        ids=chat_token_ids(tokenizer,[dict(role='user',content=prompt)],False)
        if len(ids)>=minimum:
            return dict(name=f'long{minimum}',prompt=prompt,input_ids=ids,records=count,
                        expected=dict(ALPHA='quartz731',OMEGA='maple284'))
        count+=max(1,(minimum-len(ids))//24)


@torch.no_grad()
def generate(engine,tokenizer,ids,max_tokens,eos,chunk_size,prefill_progress=False):
    generated=[]; timings=[]
    begin_request=time.perf_counter()
    torch.cuda.synchronize(); begin=time.perf_counter()
    progress=(lambda start,end,total:print('PREFILL_CHUNK',end,total,flush=True)) if prefill_progress else None
    logits,state=engine.prefill(ids,chunk_size,progress=progress)
    token=int(logits[0].argmax()); torch.cuda.synchronize()
    timings.append(time.perf_counter()-begin)
    for step in range(max_tokens):
        if not bool(torch.isfinite(logits).all()):raise RuntimeError('Nonfinite logits')
        generated.append(token)
        if token in eos or len(generated)==max_tokens:break
        torch.cuda.synchronize(); begin=time.perf_counter()
        logits,state=engine([token],state)
        token=int(logits[0].argmax()); torch.cuda.synchronize()
        timings.append(time.perf_counter()-begin)
    text=tokenizer.decode(generated,skip_special_tokens=True)
    return dict(generated_ids=generated,visible_text=text,output_tokens=len(generated),
        finish_reason='stop' if generated[-1] in eos else 'length',first_token_seconds=timings[0],
        decode_seconds=timings[1:],decode_tokens_per_second=(len(generated)-1)/sum(timings[1:])
        if len(generated)>1 else None,median_decode_ms=statistics.median(timings[1:])*1000
        if len(generated)>1 else None,request_seconds=time.perf_counter()-begin_request),state


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='/model')
    p.add_argument('--lengths',default='4096,6144')
    p.add_argument('--chunk-size',type=int,default=2048)
    p.add_argument('--max-tokens',type=int,default=96)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--output',default='logs/real-long-context.json')
    args=p.parse_args()
    if not 1<=args.repeats<=4 or not 1<=args.max_tokens<=256:raise ValueError('Invalid generation limits')
    cfg=json.loads((Path(args.model)/'config.json').read_text())['text_config']
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    eos=generation_eos_ids(tokenizer,cfg,args.model)
    cases=[case_for_length(tokenizer,int(length)) for length in args.lengths.split(',')]
    if any(len(c['input_ids'])+args.max_tokens>cfg['max_position_embeddings'] for c in cases):
        raise ValueError('Context exceeds the actual checkpoint override')
    torch.set_num_threads(4); begin=time.perf_counter()
    engine=GreedyEngineRef(WeightStore(args.model,'cuda'),cfg,activation_backend='sm121-native',
        router_backend='native',moe_finalize='deterministic',expert_backend='flashinfer',
        head_backend='sm121',qsa_backend='triton',resident_experts=True,graph_decode=True)
    torch.cuda.synchronize()
    result=dict(scope='Actual long-text retrieval smoke test; no broad quality score',
                chunk_size=args.chunk_size,load_seconds=time.perf_counter()-begin,requests=[])
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    for case in cases:
        for repeat in range(args.repeats):
            row,state=generate(engine,tokenizer,case['input_ids'],args.max_tokens,eos,args.chunk_size)
            try:answer=json.loads(row['visible_text'].split('</think>')[-1].strip())
            except ValueError:answer=None
            histories={str(i):dict(packed_tokens=s.attention.length,
                                  compressed_groups=len(s.attention.compressed))
                       for i,s in enumerate(state.layers) if engine.layers[i].layer_type=='full_attention'}
            row.update(case=case['name'],repeat=repeat,prompt=case['prompt'],input_ids=list(case['input_ids']),
                       input_tokens=len(case['input_ids']),max_tokens=args.max_tokens,expected=case['expected'],
                       answer_checks=dict(expected_codes=answer==case['expected']),qsa_histories=histories)
            result['requests'].append(row)
            result['graph_stats']=engine.graph_stats()
            result['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
            result['runtime_vllm_modules']=[n for n in sys.modules if n=='vllm' or n.startswith('vllm.')]
            if result['runtime_vllm_modules']:raise RuntimeError('Independent runtime imported vLLM')
            path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
            print('LONG_REQUEST',case['name'],repeat,len(case['input_ids']),row['decode_tokens_per_second'],
                  row['answer_checks'],row['finish_reason'],flush=True)
            print('TEXT',row['visible_text'],flush=True)
            del state
    result['repeat_consistency']={c['name']:all(r['generated_ids']==next(
        x['generated_ids'] for x in result['requests'] if x['case']==c['name'])
        for r in result['requests'] if r['case']==c['name']) for c in cases}
    result['passed']=all(result['repeat_consistency'].values()) and all(
        r['answer_checks']['expected_codes'] for r in result['requests'])
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('RESULT',result['passed'],result['peak_allocated_gib'],flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
