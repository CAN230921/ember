"""Row-preserving projection dispatch; CPU references do not require Triton."""
import torch
from torch.nn import functional as F


def per_row_linear(x,weight):
    if not x.is_cuda:
        return torch.cat([F.linear(row,weight) for row in x.split(1)])
    from .row_gemv import per_row_linear as cuda_rows
    return cuda_rows(x,weight)
