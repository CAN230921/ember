"""One-launch BF16 row projections that reproduce cuBLAS's M=1 GEMV bits for every row.

cuBLAS (pinned image) computes each M=1 BF16 GEMV output as: 32 lanes, lane t accumulates the FP32
products k = t, t+32, t+64, ... in ascending order, then a halving tree (lane i + lane i+h, h = 16..1),
rounded to BF16 (probe_gemv_order_20261020: exact over heavy-tailed operands for N=512 and N=96,
K=2560). BF16 x BF16 products are exact in FP32, so reproducing that addition order reproduces the
bits. This kernel keeps the order and reads the weight once for all rows (router logits, GDN ba).

2026-10-22 (probe_gemv_order_v2_20261022): the QSA indexer index_qk (640 x 2560) uses the same 32-lane
order; the PLE key (10240 x 2560) and value (2560 x 2560) projections use a 16-lane variant (lane t:
k = t, t+16, ...; halving tree h = 8..1). LANES selects the variant per verified shape.
"""
import os

import torch
import torch.nn.functional as F
import triton as tr
import triton.language as tl


@tr.jit
def _halve(acc, BM: tl.constexpr, BN: tl.constexpr, H: tl.constexpr):
    # [BM, BN, 2H] -> [BM, BN, H]: lane i + lane i+H (first half + second half).
    a, b = tl.split(tl.permute(tl.reshape(acc, (BM, BN, 2, H)), (0, 1, 3, 2)))
    return a + b


@tr.jit
def _rows_gemv(X, W, Y, M, N: tl.constexpr, K: tl.constexpr, SX, BM: tl.constexpr, BN: tl.constexpr,
               UNROLL: tl.constexpr, LANES: tl.constexpr):
    n = tl.program_id(0)*BN + tl.arange(0, BN)
    m = tl.arange(0, BM)
    t = tl.arange(0, LANES)
    acc = tl.zeros((BM, BN, LANES), tl.float32)
    for i0 in range(0, K // LANES, UNROLL):
        for u in tl.static_range(UNROLL):
            # Same ascending per-lane order: block i0+u is added before i0+u+1.
            k = (i0 + u)*LANES + t
            x = tl.load(X + m[:, None]*SX + k[None, :], m[:, None] < M, 0).to(tl.float32)
            w = tl.load(W + n[:, None]*K + k[None, :], n[:, None] < N, 0).to(tl.float32)
            acc = acc + x[:, None, :]*w[None, :, :]
    if LANES == 32:
        acc = _halve(acc, BM, BN, 16)
    acc = _halve(acc, BM, BN, 8)
    acc = _halve(acc, BM, BN, 4)
    acc = _halve(acc, BM, BN, 2)
    acc = _halve(acc, BM, BN, 1)
    out = tl.reshape(acc, (BM, BN))
    tl.store(Y + m[:, None]*N + n[None, :], out.to(tl.bfloat16), (m[:, None] < M) & (n[None, :] < N))


# Verified shape -> lane variant (router, GDN ba, QSA index_qk: 32 lanes; PLE key/value: 16 lanes).
VERIFIED = {(512, 2560): 32, (96, 2560): 32, (640, 2560): 32, (10240, 2560): 16, (2560, 2560): 16}


# Fastest bit-equal profiles from sweep_row_gemv_20261020 (all 300 swept profiles were bit-equal).
# 640/10240/2560 from sweep_row_gemv_v2_20261022 (DRAM-cold graph timing, 3/5/8 rows; all profiles bit-equal).
PROFILE = {512: dict(block_n=2, warps=2, stages=1, unroll=8), 96: dict(block_n=2, warps=2, stages=4, unroll=8),
           640: dict(block_n=1, warps=1, stages=2, unroll=8), 10240: dict(block_n=1, warps=8, stages=3, unroll=8),
           2560: dict(block_n=1, warps=8, stages=2, unroll=8)}
# Below these row counts the per-row cuBLAS calls are faster (DRAM-cold graph timing, verify_row_gemv_v2_20261022:
# router 27 vs 55 us at 4 rows, index_qk 28 vs 48 at 3, PLE key 445 vs 629 at 2, PLE value ~85 vs 166 at 2;
# ba stays per-row below 7).
MIN_ROWS = {512: 4, 96: 7, 640: 3, 10240: 2, 2560: 2}


def rows_linear(x, weight, *, block_n=None, warps=None, stages=None, unroll=None):
    """Equal, row for row, to torch.cat([F.linear(r, weight) for r in x.split(1)]) for verified shapes."""
    rows, k = x.shape
    if (tuple(weight.shape) not in VERIFIED or k != weight.shape[1] or not 1 <= rows <= 16
            or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16):
        raise ValueError('Unverified row GEMV shape')
    if x.stride(1) != 1 or not weight.is_contiguous():
        raise ValueError('Row GEMV requires unit inner strides')
    n = weight.shape[0]
    prof = PROFILE[n]
    block_n = block_n or prof['block_n']; warps = warps or prof['warps']
    stages = stages or prof['stages']; unroll = unroll or prof['unroll']
    if (k // 32) % unroll:
        raise ValueError('Unroll must divide K/32')
    lanes = VERIFIED[tuple(weight.shape)]
    if (k // lanes) % unroll:
        raise ValueError('Unroll must divide K/lanes')
    y = torch.empty(rows, n, device=x.device, dtype=x.dtype)
    _rows_gemv[(tr.cdiv(n, block_n),)](x, weight, y, rows, n, k, x.stride(0), 16, block_n, unroll, lanes,
                                      num_warps=warps, num_stages=stages)
    return y


# Opt-in for the single-session decode/verify path: 2-3 per-row calls there re-read DRAM-cold weights,
# so one bit-equal traversal pays off already at two rows.
SMALL = os.environ.get('SPARK_ROW_GEMV_SMALL', '0') == '1'
# Opt-in for deep single-lane verification (up to VERIFY_ROWS rows): every verified per-row projection
# (router, GDN ba, QSA indexer, PLE key/value) takes one traversal from its MIN_ROWS rows on.
DEEP = os.environ.get('SPARK_ROW_GEMV_DEEP', '0') == '1'


def per_row_linear(x, weight):
    """torch.cat([F.linear(r, weight) for r in x.split(1)]), in one launch when opted in and verified."""
    shape = tuple(weight.shape); rows = x.shape[0]
    if ((SMALL and 2 <= rows or DEEP and shape in VERIFIED and MIN_ROWS[shape[0]] <= rows)
            and shape in VERIFIED and rows <= 16 and x.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16 and weight.is_contiguous() and x.stride(1) == 1):
        return rows_linear(x, weight)
    return torch.cat([F.linear(row, weight) for row in x.split(1)])
