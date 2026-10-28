"""Small-M W8A16 projection for the checkpoint's FP8 block-128 weights (experimental, not bit-equal).

Replaces activation FP8 quantization + cutlass W8A8 with one launch: FP8 weight tiles are widened to
BF16 in registers (exact), multiplied on tensor cores against BF16 activations with FP32 accumulation,
and each 128x128 block's FP32 weight scale is applied to the block's partial dot product. Activations
are not quantized, so the result is closer to the BF16 reference than the W8A8 path; it is not
bit-identical to it. Optional split-K adds FP32 partials with a second tiny reduction launch.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _w8a16(X, W, S, Y, P, M, N: tl.constexpr, K: tl.constexpr, SX, KB: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, SPLIT: tl.constexpr):
    pid_n, pid_k = tl.program_id(0), tl.program_id(1)
    n = pid_n*BN + tl.arange(0, BN)
    m = tl.arange(0, BM)
    k = tl.arange(0, 128)
    acc = tl.zeros((BM, BN), tl.float32)
    blocks = K // 128
    per = blocks // SPLIT
    for b in range(pid_k*per, pid_k*per + per):
        kk = b*128 + k
        x = tl.load(X + m[:, None]*SX + kk[None, :], m[:, None] < M, 0.0)
        # N is a multiple of 128 and BN divides 128: no N mask (an FP8 masked load cannot take an int fill).
        w = tl.load(W + n[:, None]*K + kk[None, :]).to(tl.bfloat16)
        s = tl.load(S + (n // 128)*KB + b)
        acc += tl.dot(x, tl.trans(w)) * s[None, :]
    if SPLIT == 1:
        tl.store(Y + m[:, None]*N + n[None, :], acc.to(tl.bfloat16), (m[:, None] < M) & (n[None, :] < N))
    else:
        tl.store(P + (pid_k*BM + m[:, None])*N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


@tr.jit
def _reduce(P, Y, M, N: tl.constexpr, BM: tl.constexpr, SPLIT: tl.constexpr, BN: tl.constexpr):
    n = tl.program_id(0)*BN + tl.arange(0, BN)
    m = tl.program_id(1)
    acc = tl.zeros((BN,), tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(P + (s*BM + m)*N + n, n < N, 0)
    tl.store(Y + m*N + n, acc.to(tl.bfloat16), n < N)


DEFAULT = dict(block_n=32, split=1, warps=4, stages=3)
# Fastest profiles per (N, K) and row bucket (1, <=4, more) from micro_w8a16_20261020 (DRAM-cold, graph replay).
_P = lambda bn, sp, wp, st: dict(block_n=bn, split=sp, warps=wp, stages=st)
PROFILES = {
    (16384, 2560): (_P(16, 4, 2, 4), _P(16, 1, 2, 4), _P(32, 1, 4, 4)),
    (2560, 6144): (_P(16, 8, 2, 3), _P(16, 8, 2, 3), _P(32, 1, 4, 4)),
    (12288, 2560): (_P(16, 2, 2, 3), _P(16, 1, 2, 3), _P(32, 1, 4, 3)),
    (13312, 2560): (_P(16, 2, 2, 3), _P(16, 1, 2, 3), _P(32, 1, 4, 3)),  # QSA fused q|k|v (as q)
    (512, 2560): (_P(64, 4, 8, 3), _P(32, 4, 2, 3), _P(32, 4, 2, 4)),
    (1280, 2560): (_P(16, 4, 2, 3), _P(16, 2, 2, 4), _P(32, 4, 4, 3)),
    (2560, 640): (_P(32, 1, 4, 3), _P(64, 1, 8, 3), _P(64, 1, 8, 3)),
}


def profile(n, k, rows):
    table = PROFILES.get((n, k))
    if table is None: return DEFAULT
    return table[0 if rows == 1 else 1 if rows <= 4 else 2]


def linear(x, weight, scale_inv, *, block_n=None, split=None, warps=None, stages=None):
    """x[M,K] BF16 (M<=16), weight[N,K] float8_e4m3fn, scale_inv[N/128,K/128] FP32 -> [M,N] BF16."""
    rows, k = x.shape
    n = weight.shape[0]
    if not 1 <= rows <= 16 or x.dtype != torch.bfloat16 or weight.dtype != torch.float8_e4m3fn:
        raise ValueError('W8A16 projection requires 1..16 BF16 rows and FP8 weights')
    if k % 128 or n % 128 or weight.shape[1] != k or tuple(scale_inv.shape) != (n//128, k//128):
        raise ValueError('W8A16 projection requires block-128 weights')
    if x.stride(1) != 1 or not weight.is_contiguous() or not scale_inv.is_contiguous():
        raise ValueError('W8A16 projection requires contiguous operands')
    prof = profile(n, k, rows)
    block_n = block_n or prof['block_n']; split = split or prof['split']
    warps = warps or prof['warps']; stages = stages or prof['stages']
    if (k // 128) % split or 128 % block_n:
        raise ValueError('Unsupported W8A16 launch profile')
    y = torch.empty(rows, n, device=x.device, dtype=torch.bfloat16)
    p = torch.empty(split*16, n, device=x.device, dtype=torch.float32) if split > 1 else y
    _w8a16[(tr.cdiv(n, block_n), split)](x, weight, scale_inv, y, p, rows, n, k, x.stride(0), k//128,
                                          16, block_n, split, num_warps=warps, num_stages=stages)
    if split > 1:
        _reduce[(tr.cdiv(n, 256), rows)](p, y, rows, n, 16, split, 256, num_warps=4)
    return y
