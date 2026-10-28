"""Blockwise W8A8 quantization and explicit reference/native GEMM backends."""
import os

import torch
from .m0_refs import fp8_blockwise_dequant

W8A16 = os.environ.get('SPARK_W8A16', '0') == '1'


def quantize_activation_cuda(x):
    """Custom sm_121 kernel, with column-major scales for scaled GEMM.

    Build the adjacent .cu into libfp8_quant_sm121.so on Spark first. This
    explicit entry point keeps the CPU correctness reference available.
    """
    if not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError('Expected contiguous CUDA BF16 input')
    if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] % 128:
        raise ValueError('Expected nonempty [tokens, K] with K divisible by 128')
    if torch.cuda.get_device_capability(x.device) != (12,1):
        raise ValueError('This kernel has been validated only on sm_121')
    fn = _load_cuda_quantizer()
    rows,cols = x.shape
    q = torch.empty_like(x,dtype=torch.float8_e4m3fn)
    scales = torch.empty_strided((rows,cols//128),(1,rows),device=x.device,dtype=torch.float32)
    with torch.cuda.device(x.device):
        rc = fn(x.data_ptr(),q.data_ptr(),scales.data_ptr(),rows,cols,4,
                torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f'FP8 quantizer launch failed: CUDA code {rc}')
    return q,scales


def _load_cuda_quantizer():
    import ctypes
    from pathlib import Path
    # Keep the CDLL alive for as long as function pointers can be used.
    if not hasattr(_load_cuda_quantizer,'library'):
        lib = ctypes.CDLL(str(Path(__file__).with_name('libfp8_quant_sm121.so')))
        fn = lib.fp8_quantize
        fn.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
                       ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
        fn.restype = ctypes.c_int
        _load_cuda_quantizer.library = lib
    return _load_cuda_quantizer.library.fp8_quantize


def quantize_activation(x, group_size=128, eps=1e-10):
    if x.ndim != 2 or x.shape[-1] % group_size:
        raise ValueError("Expected token rows divisible into 128-wide groups")
    grouped = x.float().reshape(x.shape[0],-1,group_size)
    # CUDA tensor /448 may fold into multiplication by a rounded reciprocal,
    # shifting a few exact FP8 midpoints. Round the true quotient to FP32.
    scales = (grouped.abs().amax(-1).clamp_min(eps).double() / 448.0).float()
    q = (grouped / scales.unsqueeze(-1)).clamp(-448,448).to(torch.float8_e4m3fn)
    return q.reshape_as(x),scales


def linear(x,weight,scale_inv,activation_backend='torch'):
    if activation_backend=='torch':q,scales=quantize_activation(x)
    elif activation_backend=='sm121':q,scales=quantize_activation_cuda(x)
    elif activation_backend=='sm121-native':
        if W8A16 and x.shape[0] <= 16 and weight.shape[0] % 128 == 0:
            # Opt-in tensor-core W8A16: BF16 activations, FP8 weights widened in registers.
            from .fp8_gemv import linear as w8a16
            return w8a16(x.contiguous(), weight, scale_inv)
        q,scales=quantize_activation_cuda(x.contiguous())
        from .native_kernels import scaled_fp8_mm
        return scaled_fp8_mm(q,weight.T,scales,scale_inv.T,x.dtype)
    elif activation_backend=='sm121-cutlass':
        q,scales=quantize_activation_cuda(x)
        from flashinfer.gemm import gemm_fp8_nt_groupwise
        rows=x.shape[0]
        padded=(rows+3)//4*4
        if rows!=padded:
            q_padded=torch.zeros(padded,x.shape[1],device=x.device,dtype=q.dtype)
            q_padded[:rows].copy_(q)
            scales_padded=torch.empty_strided((padded,scales.shape[1]),(1,padded),device=x.device,dtype=scales.dtype)
            scales_padded.fill_(1); scales_padded[:rows].copy_(scales)
            q,scales=q_padded,scales_padded
        # FI's SM120 K-major entry point requires contiguous [M,K/128]
        # scales; the custom quantizer deliberately emits the incumbent
        # column-major layout, so this experimental adapter repacks it.
        return gemm_fp8_nt_groupwise(q,weight,scales.contiguous(),scale_inv,
            scale_major_mode='K',scale_granularity_mnk=(1,128,128),out_dtype=x.dtype,backend='cutlass')[:rows]
    else:raise ValueError('Unknown FP8 activation backend')
    activation = (q.float().reshape(x.shape[0],-1,128) * scales.unsqueeze(-1)).reshape_as(x)
    decoded = fp8_blockwise_dequant(weight,scale_inv)
    return (activation @ decoded.T).to(x.dtype)
