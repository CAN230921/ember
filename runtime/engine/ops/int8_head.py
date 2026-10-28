"""Direct packed GPTQ INT8 one to three token heads for the GB10 checkpoint."""
import ctypes
import os
from pathlib import Path
import torch


def _load(rows=1):
    if rows>3:
        # v2: x staged in shared memory, four columns per thread; bit-identical to the
        # per-row kernels and ~2.4x faster at 12 rows (GPU micro check 2026-10-19).
        if not hasattr(_load,'rows_library'):
            _load.rows_library=ctypes.CDLL(str(Path(__file__).with_name('libint8_head_rows_v2_sm121.so')))
            fn=_load.rows_library.gptq_head_gemv_rows_v2
            fn.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*4+[ctypes.c_void_p]
            fn.restype=ctypes.c_int
        fn=_load.rows_library.gptq_head_gemv_rows_v2
        return lambda x,w,s,o,k,n,stream:fn(x,w,s,o,k,n,rows,4,stream)
    if rows==3:
        if not hasattr(_load,'three_library'):
            _load.three_library=ctypes.CDLL(str(Path(__file__).with_name('libint8_head_three_sm121.so')))
        fn=_load.three_library.gptq_head_gemv_three
        fn.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_void_p]
        fn.restype=ctypes.c_int
        return fn
    if not hasattr(_load,'library'):
        _load.library=ctypes.CDLL(str(Path(__file__).with_name('libint8_head_sm121.so')))
    fn=getattr(_load.library,'gptq_head_gemv_two' if rows==2 else 'gptq_head_gemv')
    fn.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_void_p]
    fn.restype=ctypes.c_int
    return fn


# Opt-in: rows 4..12 (all batch lanes) share one head traversal; rows 1..3 keep their kernels.
ROW_LIMIT=12 if os.environ.get('SPARK_HEAD_ROWS','0')=='1' else 3


def packed_head(hidden,qweight,scales,max_rows=None):
    max_rows=ROW_LIMIT if max_rows is None else max_rows
    if not hidden.is_cuda or torch.cuda.get_device_capability(hidden.device)!=(12,1):
        raise ValueError('This output head is validated only for CUDA sm_121')
    if max_rows not in (3,12):raise ValueError('Unverified head row limit')
    if hidden.ndim!=2 or not 1<=hidden.shape[0]<=max_rows or hidden.shape[1]!=2560 or hidden.dtype!=torch.bfloat16:
        raise ValueError('Expected one to max_rows BF16 hidden rows of width 2560')
    if qweight.dtype!=torch.int32 or qweight.shape[0]!=640 or qweight.shape[1]%32:
        raise ValueError('Expected checkpoint GPTQ packing [640,N]')
    if scales.dtype!=torch.float16 or scales.shape!=(20,qweight.shape[1]):
        raise ValueError('Expected checkpoint FP16 group128 scales [20,N]')
    if any(not t.is_contiguous() for t in (hidden,qweight,scales)):
        raise ValueError('Expected contiguous checkpoint buffers')
    if any(t.device!=hidden.device for t in (qweight,scales)):
        raise ValueError('Head buffers must share the hidden-state device')
    out=torch.empty(hidden.shape[0],qweight.shape[1],device=hidden.device,dtype=hidden.dtype)
    with torch.cuda.device(hidden.device):
        result=_load(hidden.shape[0])(hidden.data_ptr(),qweight.data_ptr(),scales.data_ptr(),out.data_ptr(),
            hidden.shape[1],qweight.shape[1],torch.cuda.current_stream().cuda_stream)
    if result:raise RuntimeError(f'GPTQ head launch failed: CUDA code {result}')
    return out
