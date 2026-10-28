"""Fixed shared-expert activation and final sum; all BF16 boundaries retained."""
import ctypes
from functools import lru_cache
from pathlib import Path
import torch
from ..limits import VERIFY_WIDTHS


@lru_cache(maxsize=1)
def _library():
    lib = ctypes.CDLL(str(Path(__file__).with_name('libshared_epilogue_sm121.so')))
    lib.sm121_shared_silu.argtypes = [ctypes.c_void_p]*2+[ctypes.c_int, ctypes.c_void_p]
    lib.sm121_shared_silu.restype = ctypes.c_int
    lib.sm121_shared_finalize.argtypes = [ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_void_p]
    lib.sm121_shared_finalize.restype = ctypes.c_int
    return lib


def _check(*tensors):
    first = tensors[0]
    if (not first.is_cuda or torch.cuda.get_device_capability(first.device) != (12, 1) or
            any(t.dtype != torch.bfloat16 or t.device != first.device or not t.is_contiguous()
                for t in tensors)):
        raise ValueError('Expected contiguous BF16 buffers on GB10')


def activation(gate_up):
    _check(gate_up)
    if gate_up.ndim != 2 or gate_up.shape[0] not in VERIFY_WIDTHS or gate_up.shape[1] != 1280:
        raise ValueError('Expected fixed one/ two/ three shared-expert rows')
    result = torch.empty((gate_up.shape[0], 640), device=gate_up.device, dtype=gate_up.dtype)
    with torch.cuda.device(gate_up.device):
        rc = _library().sm121_shared_silu(gate_up.data_ptr(), result.data_ptr(), gate_up.shape[0],
                                        torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f'Shared activation launch failed: {rc}')
    return result


def finalize(routed, shared, gate):
    _check(routed, shared, gate)
    if (routed.ndim != 2 or routed.shape[0] not in VERIFY_WIDTHS or routed.shape[1] != 2560 or
            shared.shape != routed.shape or tuple(gate.shape) not in ((routed.shape[0], 1), tuple(routed.shape))):
        raise ValueError('Expected fixed hidden rows and scalar or elementwise BF16 gates')
    result = torch.empty_like(routed)
    with torch.cuda.device(routed.device):
        rc = _library().sm121_shared_finalize(routed.data_ptr(), shared.data_ptr(), gate.data_ptr(),
            result.data_ptr(), routed.shape[0], gate.shape[1], torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f'Shared finalize launch failed: {rc}')
    return result
