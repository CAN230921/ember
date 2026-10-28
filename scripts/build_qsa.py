#!/usr/bin/env python3
"""Build the bundled deterministic selector against the active PyTorch/CUDA ABI."""
import os
from pathlib import Path
import shutil
import torch
from torch.utils.cpp_extension import load

root=Path(__file__).resolve().parents[1]/'runtime'/'native'
build=root/'build';build.mkdir(exist_ok=True)
os.environ.setdefault('MAX_JOBS','2')
library=load(name='_C_det',sources=[str(root/'topk_det.cu'),str(root/'bindings_det.cpp')],
    extra_include_paths=[str(root)],extra_cflags=['-O3','-std=c++17','-DUSE_CUDA'],
    extra_cuda_cflags=['-O3','-std=c++17','-gencode=arch=compute_121a,code=sm_121a',
        '--expt-relaxed-constexpr','-DTORCH_STABLE_ONLY','-DUSE_CUDA'],
    build_directory=str(build),verbose=True,is_python_module=False)
shutil.copyfile(library,root/'_C_det.so')
print('Built deterministic persistent_topk:',torch.ops._C_det.persistent_topk)
