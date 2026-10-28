"""Screened GB10 NVFP4 deterministic expert strategy namespace."""
import hashlib
from pathlib import Path
import torch
from flashinfer.autotuner import AutoTuner,autotune

NAMESPACE='gb10_sm121_nvfp4_top5_deterministic_v1'
CACHE_SHA256='9ac337e549010c25c6111543af4ff78dec283d837b27dc4c6d86c209c616f9c3'
TACTICS={1:(18,37),2:(16,37),3:(16,36)}


class DeterministicMoEDomain:
    def __init__(self,cache):
        if torch.cuda.get_device_capability()!=(12,1):raise RuntimeError('GB10 SM121 only')
        path=Path(cache)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=CACHE_SHA256:raise RuntimeError('Unreviewed deterministic strategy cache')
        self.tuner=AutoTuner.get();self.original=self.tuner.choose_one;self.enabled=False;self.observed={}
        if not self.tuner.load_configs(str(path)):raise RuntimeError('Strategy cache metadata does not match this image/device')
        self.tuner.choose_one=self.choose

    def choose(self,op,runners,config,inputs,**kwargs):
        if (not self.enabled or op not in ('trtllm::fused_moe::gemm1','trtllm::fused_moe::gemm2')
                or len(runners)!=1 or runners[0].weight_dtype!=torch.int64 or inputs[0].shape[0] not in TACTICS):
            return self.original(op,runners,config,inputs,**kwargs)
        runner=runners[0];m=inputs[0].shape[0]
        if (runner.use_fused_finalize is not False or runner.top_k!=5 or runner.output_dtype!=torch.bfloat16
                or tuple(inputs[0].shape)!=(m,1280) or tuple(inputs[1].shape)!=(512,1280,160)
                or tuple(inputs[3].shape)!=(512,2560,40)):
            raise RuntimeError('A different MoE configuration entered the deterministic strategy domain')
        name=NAMESPACE+'::'+op.rsplit('::',1)[1];stage=0 if op.endswith('gemm1') else 1
        with autotune(False,tuning_buckets=(1,2,3)):
            selected=self.original(name,runners,config,inputs,**kwargs)
        if selected[1]!=TACTICS[m][stage]:raise RuntimeError('Deterministic tactic changed or fell back: '+str((m,stage,selected[1])))
        key=f'm{m}_gemm{stage+1}_tactic{selected[1]}'
        self.observed[key]=self.observed.get(key,0)+1
        return selected

    def close(self):self.tuner.choose_one=self.original


# Official-checkpoint top10 (micro_moe_tactics_top10_20261020): per token count, the fastest gemm1/gemm2
# tactics; each matched the vendor-heuristic output exactly (max abs diff 0). Other counts keep the heuristic.
# 5/7/10/11 rows (deep single-lane / ragged multi-lane verification) from micro_moe_tactics_top10_v2_20261021:
# +2.4/+3.6/+2.5/+4.3% at 205-219 GB/s, diff 0; the other counts up to 32 gained < 2% and keep the heuristic.
TACTICS_TOP10={1:(16,36),2:(16,37),3:(16,37),4:(16,36),5:(16,36),6:(0,36),7:(16,38),8:(16,39),10:(18,38),11:(16,38),
               12:(10,38)}


class TunedTop10MoEDomain:
    def __init__(self):
        if torch.cuda.get_device_capability()!=(12,1):raise RuntimeError('GB10 SM121 only')
        self.tuner=AutoTuner.get();self.original=self.tuner.choose_one;self.observed={}
        self.tuner.choose_one=self.choose

    def choose(self,op,runners,config,inputs,**kwargs):
        if (op not in ('trtllm::fused_moe::gemm1','trtllm::fused_moe::gemm2') or len(runners)!=1
                or runners[0].weight_dtype!=torch.int64 or inputs[0].shape[0] not in TACTICS_TOP10):
            return self.original(op,runners,config,inputs,**kwargs)
        runner=runners[0];m=inputs[0].shape[0]
        if (runner.use_fused_finalize is not False or runner.top_k!=10 or runner.output_dtype!=torch.bfloat16
                or tuple(inputs[0].shape)!=(m,1280) or tuple(inputs[1].shape)!=(512,1280,160)
                or tuple(inputs[3].shape)!=(512,2560,40)):
            return self.original(op,runners,config,inputs,**kwargs)
        stage=0 if op.endswith('gemm1') else 1;tactic=TACTICS_TOP10[m][stage]
        key=f'm{m}_gemm{stage+1}_tactic{tactic}'
        self.observed[key]=self.observed.get(key,0)+1
        return runner,tactic

    def close(self):self.tuner.choose_one=self.original
