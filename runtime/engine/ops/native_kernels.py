"""Reuse the pinned image's native GEMM/router without a vLLM Python runner.

These compiled operators are third-party kernels, not authored optimizations.
The engine still owns quantization, weights, execution and sequence state.
"""
from functools import lru_cache
import os
from pathlib import Path
import sys
import torch


_LIBRARIES = {
    'gemm': ('_C_stable_libtorch.abi3.so', '_C', 'cutlass_scaled_mm'),
    'router': ('_moe_C_stable_libtorch.abi3.so', '_moe_C', 'topk_softmax'),
    'qsa_topk': ('_C_det.so', '_C_det', 'persistent_topk'),
}
_loaded_paths = {}


def library_candidates(kind):
    """Configured paths take precedence, including for the QSA extension."""
    filename = _LIBRARIES[kind][0]
    configured = os.environ.get('ENGINE_NATIVE_KERNEL_DIR')
    directories = ([Path(configured)] if configured else [])
    directories += [Path(entry) / 'vllm' for entry in sys.path if entry]
    if kind == 'qsa_topk':
        directories += [Path(__file__).resolve().parents[2] / 'native',
                        Path('/opt/llm/kernel-det')]
    return [directory / filename for directory in directories]


@lru_cache(maxsize=3)
def load_operator(kind):
    filename, namespace, name = _LIBRARIES[kind]
    ops = getattr(torch.ops, namespace)
    try:
        operator = getattr(ops, name)
    except AttributeError:
        library = next((path for path in library_candidates(kind) if path.is_file()), None)
        if library is None:
            raise RuntimeError(f'Native kernel library {filename} not found; '
                               'build the documented native operators or set ENGINE_NATIVE_KERNEL_DIR')
        torch.ops.load_library(str(library))
        operator = getattr(ops, name)
        _loaded_paths[kind] = str(library.resolve())
    else:
        _loaded_paths[kind] = 'operator already registered'
    return operator


def provenance():
    return dict(_loaded_paths)


def scaled_fp8_mm(a, b, scale_a, scale_b, output_dtype):
    if not a.is_cuda or b.device != a.device:
        raise ValueError('Native GEMM requires tensors on the same CUDA device')
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError('Expected A[M,K] and B[K,N]')
    if a.stride(1) != 1 or b.stride(0) != 1:
        raise ValueError('Expected row-major A and column-major B')
    output = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=output_dtype)
    load_operator('gemm')(output, a, b, scale_a, scale_b, None)
    return output


def topk_softmax(logits, top_k, renormalize=True):
    if not logits.is_cuda or logits.ndim != 2 or not 0 < top_k <= logits.shape[1]:
        raise ValueError('Native router requires CUDA logits[M,E] and valid top_k')
    shape = (logits.shape[0], top_k)
    weights = torch.empty(shape, device=logits.device, dtype=torch.float32)
    ids = torch.empty(shape, device=logits.device, dtype=torch.int32)
    token_expert_indices = torch.empty_like(ids)
    load_operator('router')(weights, ids, token_expert_indices, logits.contiguous(),
                            renormalize, None, None)
    return weights, ids


def qsa_topk(logits, visible, blocks, workspace, top_k, columns):
    """Pinned deterministic selector, including the incumbent block order."""
    load_operator('qsa_topk')(logits, visible, blocks, workspace, top_k, columns)
