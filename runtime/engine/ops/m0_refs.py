"""M0 operator references: blockwise-FP8 linear, GPTQ-int8 head, NVFP4 expert dequant.

Pure torch references for correctness bring-up (fp32/fp64 accumulate).
Packing-order facts marked TODO-ORACLE must be confirmed against the vLLM
oracle dump before any layer-level alignment is claimed.
"""
import torch

# ---------------------------------------------------------------- FP8 blockwise
def fp8_blockwise_dequant(w_fp8: torch.Tensor, w_scale_inv: torch.Tensor) -> torch.Tensor:
    """w [N,K] fp8e4m3, scale_inv [N/128, K/128] fp32 (128x128 blocks).
    Returns dequantized fp32 weights: W = fp8(w) * scale_inv (block-broadcast)."""
    N, K = w_fp8.shape
    assert N % 128 == 0 and K % 128 == 0 and w_scale_inv.shape == (N // 128, K // 128)
    return (w_fp8.to(torch.float32).view(N // 128, 128, K // 128, 128) *
            w_scale_inv[:, None, :, None]).view(N, K)


def fp8_linear_ref(x_bf16: torch.Tensor, w_fp8: torch.Tensor, w_scale_inv: torch.Tensor) -> torch.Tensor:
    """T=1/small-T blockwise-FP8 linear, fp32 accumulate, bf16 out (semantics of
    Fp8LinearMethod + CutlassFp8BlockScaledMMKernel incl. dynamic per-token-group
    activation quant is intentionally NOT applied here — that is the W4A8 oracle
    difference to be aligned in M1; this reference is weights-only dequant)."""
    w = fp8_blockwise_dequant(w_fp8, w_scale_inv)
    return (x_bf16.to(torch.float32) @ w.T).to(torch.bfloat16)


# ---------------------------------------------------------------- GPTQ int8 head
def gptq_int8_dequant(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor,
                      bits: int = 8) -> torch.Tensor:
    """This checkpoint's symmetric GPTQ W8A16, group128, desc_act=False.

    head_dispatch.py selects sym=True -> scalar_types.uint8b128. Four
    unsigned bytes are packed along K, least-significant byte first. Marlin
    ignores qzeros on this symmetric path (even though the checkpoint stores
    0x7f7f7f7f). W[k,n] = (unsigned_byte - 128) * scales[k//128,n].
    Other GPTQ conventions are deliberately unsupported.
    """
    if bits != 8:
        raise ValueError("Only symmetric GPTQ int8 is supported")
    K4, N = qweight.shape
    K = K4 * 4
    G = scales.shape[0]
    assert G * 128 == K and qzeros.shape == (G, N // 4)
    # little-endian int32: byte j of qweight[row, n] holds weight k=row*4+j
    shifts = torch.arange(4, device=qweight.device, dtype=torch.int64) * 8
    qw = ((qweight.to(torch.int64).unsqueeze(1) >> shifts[None, :, None]) & 255)
    qw = qw.reshape(K, N).to(torch.float32) - 128.0
    kidx = torch.arange(K, device=qweight.device) // 128
    w = qw * scales.float()[kidx]
    return w


def gptq_head_gemv_ref(x_bf16: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor,
                       scales: torch.Tensor) -> torch.Tensor:
    w = gptq_int8_dequant(qweight, qzeros, scales)  # (K, N) = (2560, 248320)
    return (x_bf16.to(torch.float32) @ w).to(torch.bfloat16)


# ---------------------------------------------------------------- NVFP4 experts
_E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)


def _fp8e4m3_to_float(bits: torch.Tensor) -> torch.Tensor:
    return bits.view(torch.uint8).to(torch.int16).float()


def nvfp4_dequant(w_packed_u8: torch.Tensor, w_scale_f8: torch.Tensor,
                  w_scale2_f32: torch.Tensor) -> torch.Tensor:
    """NVFP4 weights-only dequant reference.
    w_packed_u8 [N, K/2] (2 fp4 per byte, LOW nibble first — TODO-ORACLE),
    w_scale_f8 [N, K/16] fp8e4m3 per-16-group scale, w_scale2_f32 scalar.
    W = e2m1(nibble) * scale(group) * scale2.
    Activation-side quant (W4A4 input_scale) is NOT applied here (M1 oracle)."""
    N, K2 = w_packed_u8.shape
    K = K2 * 2
    lo = (w_packed_u8 & 0xF).to(torch.int64)
    hi = (w_packed_u8 >> 4).to(torch.int64)
    vals = torch.empty(N, K, device=w_packed_u8.device, dtype=torch.float32)
    vals[:, 0::2] = _E2M1_LUT.to(w_packed_u8.device)[lo]
    vals[:, 1::2] = _E2M1_LUT.to(w_packed_u8.device)[hi]
    # group scale: [N, K/16] fp8 codes -> float via bit reinterpret (e4m3)
    s = w_scale_f8.view(torch.uint8).to(torch.int16)
    # e4m3 decode: sign(1) exp(4) mantissa(3), bias 7; subnormals included
    sign = torch.where(s >= 128, -1.0, 1.0)
    e = ((s & 0x7F) >> 3).float()
    m = (s & 0x7).float()
    s_val = sign * torch.where(e > 0, 2.0 ** (e - 7) * (1 + m / 8), m / 8 * 2.0 ** -6)
    gidx = torch.arange(K, device=w_packed_u8.device) // 16
    return vals * s_val.float()[:, gidx] * float(w_scale2_f32)
