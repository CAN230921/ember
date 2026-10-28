"""GB10 greedy selection: finite check and first argmax in one GPU scan.

The two target rows are transferred together in one small host copy. The
finite-logit gate remains enabled. Workspaces belong to this selector instance.
"""
import torch
from ..limits import VERIFY_ROWS


class GreedySelector:
    def __init__(self):self.workspace={}

    def __call__(self,logits):
        if logits.ndim!=2 or not 1<=logits.shape[0]<=VERIFY_ROWS or not logits.shape[1]:
            raise ValueError('Expected one to three vocabulary rows')
        if not logits.is_cuda:
            if not bool(torch.isfinite(logits).all()):raise RuntimeError('Nonfinite speculative logits')
            return logits.argmax(-1).tolist()
        if logits.dtype not in (torch.float32,torch.bfloat16):raise ValueError('Unsupported logit dtype')
        from .greedy_triton import select
        key=(logits.device,logits.dtype,logits.shape)
        if key not in self.workspace:
            rows,vocab=logits.shape; tiles=(vocab+1023)//1024
            self.workspace[key]=(torch.empty(rows,tiles,device=logits.device,dtype=torch.float32),
                torch.empty(rows,tiles,device=logits.device,dtype=torch.int32),
                torch.empty(rows,tiles,device=logits.device,dtype=torch.int32),
                torch.empty(rows,2,device=logits.device,dtype=torch.int64))
        output=select(logits.contiguous(),self.workspace[key])
        host=output.cpu().tolist()
        if any(row[1] for row in host):raise RuntimeError('Nonfinite speculative logits')
        return [row[0] for row in host]
