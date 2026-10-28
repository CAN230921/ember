"""Small-M NVFP4 W4A16 projection (draft-only arithmetic, not bit-equal to any BF16 path).

Weights: E2M1 codes packed two per byte (low nibble = even K) [N, K/2] uint8, FP8-E4M3 per-16 block
scales [N, K/16] (stored as uint8), one FP32 global scale per matrix. Each tile is decoded to BF16 in
registers (code * block scale is exact in BF16), multiplied on tensor cores against BF16 activations
with FP32 accumulation, and the global scale is applied once to the accumulator. Activations are not
quantized. Pure Triton, so launches can be captured in CUDA graphs. Optional split-K adds FP32
partials with a second tiny reduction launch (same reduction as fp8_gemv).
"""
import torch
import triton as tr
import triton.language as tl

from .fp8_gemv import _reduce
from .moe_w4a16 import _decode

# E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}; rounding boundaries are the midpoints (ties to the lower code).
_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_VALUES = (0., .5, 1., 1.5, 2., 3., 4., 6.)


@tr.jit
def _w4a16(X, W, S, Y, P, M, N: tl.constexpr, K: tl.constexpr, SX, GS,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    pid_n, pid_k = tl.program_id(0), tl.program_id(1)
    n = pid_n*BN + tl.arange(0, BN)
    m = tl.arange(0, BM)
    acc = tl.zeros((BM, BN), tl.float32)
    PER: tl.constexpr = K // BK // SPLIT
    for i in range(PER):
        kb = (pid_k*PER + i)*BK
        k = kb + tl.arange(0, BK)
        x = tl.load(X + m[:, None]*SX + k[None, :], m[:, None] < M, 0.0)
        acc += tl.dot(x, tl.trans(_decode(W, S, n, kb, BN, BK, K // 2, K // 16)))
    acc = acc*GS
    if SPLIT == 1:
        tl.store(Y + m[:, None]*N + n[None, :], acc.to(tl.bfloat16), m[:, None] < M)
    else:
        tl.store(P + (pid_k*BM + m[:, None])*N + n[None, :], acc, m[:, None] < M)


def quantize(w):
    """BF16/FP32 [N, K] -> (packed uint8 [N, K/2], scales uint8 [N, K/16], global scale float).
    Per-tensor global = amax/(6*448), block scale = E4M3(amax16/6/global), codes = nearest E2M1
    (the fake-quant of diagnostics/teacher_force_dflash_q_20261021.py)."""
    n, k = w.shape
    if k % 16 or n % 16:
        raise ValueError('NVFP4 weights need N and K divisible by 16')
    bounds = torch.tensor(_BOUNDS, device=w.device)
    gs = float(w.float().abs().amax().clamp(min=1e-12))/(6*448)
    packed = torch.empty(n, k//2, dtype=torch.uint8, device=w.device)
    scales = torch.empty(n, k//16, dtype=torch.uint8, device=w.device)
    for r0 in range(0, n, 1024):                       # bounded temporaries
        blk = w[r0:r0+1024].float().unflatten(-1, (-1, 16))
        sc = (blk.abs().amax(-1, keepdim=True)/6/gs).to(torch.float8_e4m3fn)
        x = (blk/(sc.float()*gs).clamp(min=1e-30)).clamp(-6, 6)
        code = torch.bucketize(x.abs(), bounds).to(torch.uint8) | ((x < 0).to(torch.uint8) << 3)
        code = code.flatten(-2)
        packed[r0:r0+1024] = code[:, 0::2] | (code[:, 1::2] << 4)
        scales[r0:r0+1024] = sc[..., 0].view(torch.uint8)
    return packed, scales, gs


def dequantize(packed, scales, gs, dtype=torch.bfloat16):
    """Inverse of quantize -> [N, K] (the exact weights the kernel multiplies by, before rounding to dtype)."""
    values = torch.tensor(_VALUES, device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    w = values[codes & 7]*torch.where((codes & 8) != 0, -1.0, 1.0)
    s = scales.view(torch.float8_e4m3fn).float()
    return (w.unflatten(-1, (-1, 16))*s[..., None]).flatten(-2).mul(gs).to(dtype)


DEFAULT = dict(block_n=32, block_k=128, split=1, warps=4, stages=3)
_P = lambda bn, bk, sp, wp, st: dict(block_n=bn, block_k=bk, split=sp, warps=wp, stages=st)
# Fastest profiles per (N, K) for 1..8 rows (GB10, graph replay over L2-cold weight copies,
# micro_dflash_fp4_20261021 r1: ~160-210 GB/s of packed weight + scales).
PROFILES = {
    (2560, 12800): _P(32, 256, 1, 4, 3),   # DFlash fc
    (6144, 2560): _P(32, 256, 2, 4, 3),    # q|k|v
    (2048, 2560): _P(32, 256, 2, 4, 3),    # k|v (context extend)
    (15360, 2560): _P(32, 256, 2, 4, 3),   # gate|up
    (2560, 4096): _P(32, 256, 8, 4, 3),    # o
    (2560, 7680): _P(32, 256, 1, 4, 3),    # down
}


def profile(n, k):
    return PROFILES.get((n, k), DEFAULT)


def linear(x, packed, scales, gs, *, block_n=None, block_k=None, split=None, warps=None, stages=None):
    """x [M, K] BF16 (M <= 16, unit stride along K) -> [M, N] BF16 = x @ W.T for NVFP4 W."""
    rows, k = x.shape
    n = packed.shape[0]
    if not 1 <= rows <= 16 or x.dtype != torch.bfloat16 or packed.dtype != torch.uint8:
        raise ValueError('W4A16 projection requires 1..16 BF16 rows and packed uint8 weights')
    if packed.shape[1]*2 != k or tuple(scales.shape) != (n, k//16) or x.stride(1) != 1:
        raise ValueError('W4A16 projection shape mismatch')
    if packed.stride(1) != 1 or packed.stride(0) != k//2 or scales.stride(0) != k//16:
        raise ValueError('W4A16 projection requires row-contiguous weights')
    prof = profile(n, k)
    block_n = block_n or prof['block_n']; block_k = block_k or prof['block_k']; split = split or prof['split']
    warps = warps or prof['warps']; stages = stages or prof['stages']
    if n % block_n or k % (block_k*split):
        raise ValueError('Unsupported W4A16 launch profile')
    y = torch.empty(rows, n, device=x.device, dtype=torch.bfloat16)
    p = torch.empty(split*16, n, device=x.device, dtype=torch.float32) if split > 1 else y
    _w4a16[(n // block_n, split)](x, packed, scales, y, p, rows, n, k, x.stride(0), gs,
                                  16, block_n, block_k, split, num_warps=warps, num_stages=stages)
    if split > 1:
        _reduce[(tr.cdiv(n, 256), rows)](p, y, rows, n, 16, split, 256, num_warps=4)
    return y
