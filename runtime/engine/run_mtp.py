"""Interactive CLI entry for the same learned MTP1 path as the real A/B."""
import argparse
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from engine.model import GreedyEngineRef
from engine.mtp import MTPDraft
from engine.speculative import generate_mtp
from engine.tokenization import chat_token_ids,generation_eos_ids
from engine.weights import WeightStore


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='/model')
    p.add_argument('--prompt',required=True)
    p.add_argument('--max-tokens',type=int,default=1536)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--thinking',action='store_true')
    p.add_argument('--top-k',type=int)
    p.add_argument('--graph-verify',action='store_true')
    p.add_argument('--hc-projection-backend',choices=('torch','sm121-two'),default='torch')
    p.add_argument('--prefill-chunk-size',type=int,default=2048)
    p.add_argument('--kv-backend',choices=('contiguous','paged','auto'),default='auto')
    p.add_argument('--output',default='logs/mtp-chat.json')
    args=p.parse_args()
    if not 1<=args.max_tokens<=1536 or not 1<=args.repeats<=10:raise ValueError('Invalid token budget or repeats')
    torch.set_num_threads(4)
    cfg=json.loads((Path(args.model)/'config.json').read_text())['text_config']
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    ids=chat_token_ids(tokenizer,[dict(role='user',content=args.prompt)],args.thinking)
    if len(ids)+args.max_tokens>cfg['max_position_embeddings']:raise ValueError('Prompt and output budget exceed actual checkpoint context')
    eos=generation_eos_ids(tokenizer,cfg,args.model)
    store=WeightStore(args.model,'cuda'); begin=time.perf_counter()
    engine=GreedyEngineRef(store,cfg,activation_backend='sm121-native',router_backend='native',
        moe_finalize='deterministic',expert_backend='flashinfer',head_backend='sm121',qsa_backend='triton',
        resident_experts=True,graph_decode=True,graph_verify=args.graph_verify,hc_projection_backend=args.hc_projection_backend,kv_backend=args.kv_backend)
    draft=MTPDraft(store,cfg,engine.embedding,engine.logits,top_k=args.top_k,kv_backend=args.kv_backend)
    torch.cuda.synchronize()
    result=dict(scope='Dedicated GB10 Qwen learned MTP1; same path as benchmark_mtp',
        load_seconds=time.perf_counter()-begin,draft_top_k=draft.moe.top_k,thinking=args.thinking,
        graph_verify=args.graph_verify,hc_projection_backend=args.hc_projection_backend,
        prefill_chunk_size=args.prefill_chunk_size,kv_backend=args.kv_backend,requests=[])
    for repeat in range(args.repeats):
        request,state=generate_mtp(engine,draft,ids,args.max_tokens,eos,args.prefill_chunk_size)
        request['repeat']=repeat
        request['visible_text']=tokenizer.decode(request['generated_ids'],skip_special_tokens=True)
        result['requests'].append(request)
        result['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
        result['graph_stats']=engine.graph_stats()
        path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print('TEXT',request['visible_text'],flush=True)
        print('RESULT',repeat,request['decode_tokens_per_second'],request['finish_reason'],request['speculative'],flush=True)
        del state
    result['repeat_consistency']=all(x['generated_ids']==result['requests'][0]['generated_ids'] for x in result['requests'])
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['repeat_consistency'] else 1


if __name__=='__main__':raise SystemExit(main())
