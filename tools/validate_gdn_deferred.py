"""Bounded SM121 kernel equivalence; synthetic inputs, no model-load claim."""
import json
import time
from types import SimpleNamespace

import torch

from engine.ops.gdn import GDNState
from engine.ops.gdn_deferred import verify_gdn
from engine.ops.gdn_triton import run_core
from engine.ops.gdn_commit import commit_many


def same(actual, expected, name):
    if not torch.equal(actual.view(torch.uint16), expected.view(torch.uint16)):
        count = int((actual.view(torch.uint16) != expected.view(torch.uint16)).sum())
        raise AssertionError(f'{name}: {count} differing BF16 words')


def main():
    torch.manual_seed(20261029)
    torch.set_num_threads(2)
    device = 'cuda'
    model = SimpleNamespace(conv=torch.randn(10240, 4, device=device, dtype=torch.bfloat16)*.1,
        A_log=torch.randn(48, device=device)-2, dt_bias=torch.randn(48, device=device)*.1,
        norm=torch.ones(128, device=device, dtype=torch.bfloat16), eps=1e-6, gate_activation='silu')
    checks = []
    for inference in (False, True):
        with torch.inference_mode(inference):
            for rows in (1, 2, 3, 4, 8):
                state = GDNState(torch.randn(1,10240,3,device=device,dtype=torch.bfloat16)*.1,
                    torch.randn(1,48,128,128,device=device,dtype=torch.bfloat16)*.01)
                original = GDNState(state.conv.clone(), state.recurrent.clone())
                x = torch.randn(rows,16384,device=device,dtype=torch.bfloat16)*.2
                ba = torch.randn(rows,96,device=device,dtype=torch.bfloat16)*.2
                output, final, prefixes = run_core(model,x,ba,state,False,True)
                first = verify_gdn(model,x,ba,state)
                same(first.output, output, 'output')
                retained = first.retained_bytes
                first.discard()
                for count in range(1,rows+1):
                    tape = verify_gdn(model,x,ba,state)
                    committed = tape.commit(count)
                    same(committed.conv,prefixes[count-1].conv,'conv')
                    same(committed.recurrent,prefixes[count-1].recurrent,'recurrent')
                    y1,s1 = run_core(model,x[:1],ba[:1],committed,False)
                    y2,s2 = run_core(model,x[:1],ba[:1],prefixes[count-1],False)
                    same(y1,y2,'continuation output')
                    same(s1.recurrent,s2.recurrent,'continuation state')
                    try:tape.commit(count)
                    except RuntimeError:pass
                    else:raise AssertionError('double commit allowed')
                same(state.conv,original.conv,'base conv')
                same(state.recurrent,original.recurrent,'base recurrence')
                for count in range(1,rows+1):
                    group=[verify_gdn(model,x,ba,state) for _ in range(3)]
                    for committed in commit_many(group,count):
                        same(committed.conv,prefixes[count-1].conv,'multi-layer conv')
                        same(committed.recurrent,prefixes[count-1].recurrent,'multi-layer recurrence')
                checks.append(dict(rows=rows,inference_mode=inference,prefixes_checked=rows,
                    retained_bytes=retained,baseline_prefix_bytes=rows*(state.conv.numel()+state.recurrent.numel())*2))
    torch.cuda.synchronize()
    print(json.dumps(dict(passed=True,checks=checks,weight_source='synthetic kernel inputs',commit_backend='original_per_layer_decode',
        gpu=torch.cuda.get_device_name(),release_approved=False)))


if __name__ == '__main__':
    main()
