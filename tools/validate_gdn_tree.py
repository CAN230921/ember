"""Compare every tree node and continuation against original BF16 decode."""
import json
import statistics
import time
from types import SimpleNamespace

import torch

from engine.candidate_tree import CandidateTree
from engine.ops.gdn import GDNState
from engine.ops.gdn_tree import run_tree
from engine.ops.gdn_triton import run_core
from validate_gdn_deferred import same


def main():
    torch.manual_seed(20261029);torch.set_num_threads(2)
    model=SimpleNamespace(conv=torch.randn(10240,4,device='cuda',dtype=torch.bfloat16)*.1,
        A_log=torch.randn(48,device='cuda')-2,dt_bias=torch.randn(48,device='cuda')*.1,
        norm=torch.ones(128,device='cuda',dtype=torch.bfloat16),eps=1e-6,gate_activation='silu')
    checks=[]
    topologies=((-1,),(-1,0),(-1,0,0),(-1,0,0,1,2),(-1,0,0,1,1,3,3),
                (-1,0,1,2,3,4,5,6),(-1,0,0,0,0,0,0,0))
    for inference in (False,True):
        with torch.inference_mode(inference):
            for parents in topologies:
                tree=CandidateTree(tuple(range(len(parents))),parents);n=len(parents)
                initial=GDNState(torch.randn(1,10240,3,device='cuda',dtype=torch.bfloat16)*.1,
                    torch.randn(1,48,128,128,device='cuda',dtype=torch.bfloat16)*.01)
                frozen=GDNState(initial.conv.clone(),initial.recurrent.clone())
                x=torch.randn(n,16384,device='cuda',dtype=torch.bfloat16)*.2
                ba=torch.randn(n,96,device='cuda',dtype=torch.bfloat16)*.2
                def reference():
                    states=[];cores=[]
                    for node,parent in enumerate(parents):
                        core,state=run_core(model,x[node:node+1],ba[node:node+1],
                            initial if parent==-1 else states[parent],False)
                        states.append(state);cores.append(core)
                    return torch.cat(cores),states
                expected,states=reference();output,actual=run_tree(model,x,ba,initial,tree)
                same(output,expected,'tree core')
                for node,(a,b) in enumerate(zip(actual,states)):
                    same(a.conv,b.conv,'tree conv');same(a.recurrent,b.recurrent,'tree recurrence')
                    y1,s1=run_core(model,x[:1],ba[:1],a,False)
                    y2,s2=run_core(model,x[:1],ba[:1],b,False)
                    same(y1,y2,'tree continuation');same(s1.recurrent,s2.recurrent,'continuation recurrence')
                same(initial.conv,frozen.conv,'base conv');same(initial.recurrent,frozen.recurrent,'base recurrence')
                times={'separate':[],'tree':[]}
                p=torch.tensor(parents,device='cuda',dtype=torch.int32)
                for repeat in range(10):
                    for kind in (('separate','tree') if repeat%2==0 else ('tree','separate')):
                        torch.cuda.synchronize();start=time.perf_counter()
                        value=reference() if kind=='separate' else run_tree(model,x,ba,initial,tree,p)
                        torch.cuda.synchronize();times[kind].append(time.perf_counter()-start)
                checks.append(dict(parents=parents,inference_mode=inference,nodes=n,
                    all_states_and_outputs_bit_equal=True,median_seconds={k:statistics.median(v) for k,v in times.items()}))
    print(json.dumps(dict(passed=True,checks=checks,release_approved=False,scope='synthetic layer only')))


if __name__=='__main__':main()
