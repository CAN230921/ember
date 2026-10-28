"""Actual single-sequence text generation and separate latency diagnostics.

This uses the independent 48-layer engine. Timings include CPU dispatch,
expert transfers, PLE lookup and the output head. Profiling is a separate
extra decode and is excluded from the generation throughput measurement.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time
import torch
from transformers import AutoTokenizer
from engine.weights import WeightStore
from engine.model import GreedyEngineRef
from engine.tokenization import chat_token_ids,generation_eos_ids


class Timeline:
    """CUDA intervals include host gaps; they are not GPU active time."""
    def __init__(self):self.records=[]; self.restore=[]

    def wrap(self,obj,name,category):
        original=getattr(obj,name)
        def measured(*args,**kwargs):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record(); begin=time.perf_counter()
            value=original(*args,**kwargs)
            cpu_ms=(time.perf_counter()-begin)*1000; end.record()
            self.records.append((category,start,end,cpu_ms))
            return value
        self.restore.append((obj,name,original)); setattr(obj,name,measured)

    def install(self,engine):
        for layer in engine.layers:
            self.wrap(layer.attention,'forward','gdn_attention' if layer.layer_type=='linear_attention' else 'qsa_attention')
            self.wrap(layer.moe,'forward','moe')
            if layer.ple is not None:self.wrap(layer.ple,'forward','ple')
            for hc in (layer.attn_hc,layer.mlp_hc):
                for method in ('mix','combine_and_mix','combine'):self.wrap(hc,method,'hyperconnection')
        self.wrap(engine.final_mixer,'combine_and_mix','final_mixer')
        self.wrap(engine,'logits','output_head')
        for block in engine._decode_graphs.values():self.wrap(block,'forward','gdn_graph')

    def finish(self):
        torch.cuda.synchronize()
        categories=defaultdict(lambda:dict(calls=0,cuda_interval_ms=0.,cpu_submission_ms=0.))
        for category,start,end,cpu_ms in self.records:
            entry=categories[category]; entry['calls']+=1
            entry['cuda_interval_ms']+=start.elapsed_time(end)
            entry['cpu_submission_ms']+=cpu_ms
        for obj,name,original in self.restore:setattr(obj,name,original)
        return dict(categories)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='/model')
    parser.add_argument('--prompt',default='请用一句话解释为什么天空通常是蓝色的。')
    parser.add_argument('--trace-prompt',help='Use captured token IDs for a comparable experiment')
    parser.add_argument('--chat',action='store_true')
    parser.add_argument('--thinking',action='store_true',help='Enable the checkpoint chat template thinking mode (default off)')
    parser.add_argument('--max-tokens',type=int,default=64)
    parser.add_argument('--activation-backend',choices=('torch','sm121','sm121-cutlass','sm121-native'),default='sm121-cutlass')
    parser.add_argument('--router-backend',choices=('torch','native'),default='torch')
    parser.add_argument('--moe-finalize',choices=('fused','deterministic'),default='fused')
    parser.add_argument('--expert-backend',choices=('torch','flashinfer'),default='flashinfer')
    parser.add_argument('--head-backend',choices=('torch','sm121'),default='torch')
    parser.add_argument('--qsa-backend',choices=('torch','triton'),default='torch')
    parser.add_argument('--resident-experts',action='store_true')
    parser.add_argument('--graph-decode',action='store_true')
    parser.add_argument('--hc-projection-backend',choices=('torch','sm121'),default='torch')
    parser.add_argument('--autotune-cache')
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--output',default='logs/greedy-generation.json')
    args=parser.parse_args()
    if args.autotune_cache:
        from flashinfer.autotuner import AutoTuner
        AutoTuner.get().load_configs(args.autotune_cache)
    if args.max_tokens<1:raise ValueError('max-tokens must be positive')
    torch.set_num_threads(4)
    cfg=json.load(open(Path(args.model)/'config.json'))['text_config']
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    if args.trace_prompt:
        metadata=json.loads(Path(args.trace_prompt).read_text())
        ids=metadata['input_token_ids']; prompt=metadata['prompt']
    elif args.chat:
        prompt=args.prompt
        ids=chat_token_ids(tokenizer,[dict(role='user',content=prompt)],args.thinking)
    else:
        prompt=args.prompt; ids=tokenizer.encode(prompt,add_special_tokens=False)
    torch.cuda.synchronize(); begin=time.perf_counter()
    engine=GreedyEngineRef(WeightStore(args.model,'cuda'),cfg,expert_backend=args.expert_backend,activation_backend=args.activation_backend,router_backend=args.router_backend,moe_finalize=args.moe_finalize,
        head_backend=args.head_backend,qsa_backend=args.qsa_backend,resident_experts=args.resident_experts,graph_decode=args.graph_decode,hc_projection_backend=args.hc_projection_backend)
    torch.cuda.synchronize(); load_seconds=time.perf_counter()-begin
    state=None; generated=[]; steps=[]; ple=[]
    eos=generation_eos_ids(tokenizer,cfg,args.model)
    for step in range(args.max_tokens):
        torch.cuda.synchronize(); begin=time.perf_counter()
        logits,state=engine(ids,state)
        token=int(logits[0].argmax()); torch.cuda.synchronize()
        elapsed=time.perf_counter()-begin
        if not bool(torch.isfinite(logits).all()):raise RuntimeError('Non-finite output logits')
        generated.append(token); steps.append(elapsed); ids=[token]
        for layer in engine.layers:
            if layer.ple is not None:ple.append(dict(step=step,**layer.ple.table.last_stats))
        print('GENERATED',step,token,elapsed,repr(tokenizer.decode([token])),flush=True)
        if token in eos:break
    result=dict(scope='one sequence on GB10, this checkpoint, no serving/batching/MTP acceptance',
        activation_backend=args.activation_backend,router_backend=args.router_backend,
        moe_finalize=args.moe_finalize,hc_projection_backend=args.hc_projection_backend,
        expert_backend=args.expert_backend,head_backend=args.head_backend,qsa_backend=args.qsa_backend,
        resident_experts=args.resident_experts,eos_token_ids=sorted(eos),graph_stats=engine.graph_stats(),autotune_cache=args.autotune_cache,chat_template=args.chat,thinking=args.thinking,prompt=prompt,
        load_seconds=load_seconds,generated_ids=generated,text=tokenizer.decode(generated),
        visible_text=tokenizer.decode(generated,skip_special_tokens=True),
        prefill_and_first_token_seconds=steps[0],decode_seconds=steps[1:],
        decode_tokens_per_second=(len(steps)-1)/sum(steps[1:]) if len(steps)>1 else None,
        median_decode_ms=statistics.median(steps[1:])*1000 if len(steps)>1 else None,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,ple=ple,
        timing_includes='CPU dispatch, disk-backed expert loading, PLE, all 48 layers and output head',
        comparison='absolute latency only; no old-service speedup is established by this run')
    if args.profile and generated[-1] not in eos:
        timeline=Timeline(); timeline.install(engine)
        engine([generated[-1]],state)
        result['profile_extra_decode']=timeline.finish()
        result['profile_interpretation']='instrumented CUDA intervals include host gaps; excluded from throughput'
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('TEXT',result['visible_text'],flush=True)
    print('RESULT',result['decode_tokens_per_second'],result['peak_allocated_gib'],flush=True)


if __name__=='__main__':main()
