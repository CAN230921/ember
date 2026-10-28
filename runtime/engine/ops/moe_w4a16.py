"""Experimental small-M NVFP4 MoE (W4A16, tensor cores) -- prototype stage 1: gate_up + SwiGLU.

One program per (unique expert, intermediate tile). The tokens routed to that expert form a 16-row
tile; FP4 (E2M1, low nibble first) weights are decoded to BF16 in registers, scaled by their FP8
per-16 group scale (exact in BF16), and multiplied with BF16 activations by tl.dot (FP32 accumulate).
Activations are not FP4-quantized (unlike the W4A4 cutlass path). Not bit-equal to the engine.
Weights use the FlashInfer packing: rows [0, I) = up, [I, 2I) = gate.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _e2m1(c):
    mag = c & 7
    e = mag >> 1
    m = (mag & 1).to(tl.float32)
    pw = tl.where(e == 3, 4.0, tl.where(e == 2, 2.0, 1.0))
    v = tl.where(e == 0, 0.5*m, (1.0 + 0.5*m)*pw)
    return tl.where((c & 8) != 0, -v, v)


@tr.jit
def _decode(W, S, row, kb, BN: tl.constexpr, BK: tl.constexpr, K2: tl.constexpr, KS: tl.constexpr):
    # row: [BN] weight rows; kb: first K element of the block. Returns BF16 [BN, BK].
    j = tl.arange(0, BK // 2)
    b = tl.load(W + row[:, None]*K2 + (kb // 2 + j)[None, :])
    v = tl.reshape(tl.join(_e2m1(b & 0xF), _e2m1(b >> 4)), (BN, BK))
    g = tl.arange(0, BK // 16)
    s = tl.load(S + row[:, None]*KS + (kb // 16 + g)[None, :]).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    w = tl.reshape(tl.reshape(v, (BN, BK // 16, 16)) * s[:, :, None], (BN, BK))
    return w.to(tl.bfloat16)


@tr.jit
def _gate_up(X, W13, S13, G, U, T, H, I: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    u = tl.program_id(0); nt = tl.program_id(1)
    e = tl.load(U + u)
    r = tl.arange(0, 16)
    tok = tl.load(T + u*16 + r)                       # token index or -1
    n = nt*BN + tl.arange(0, BN)
    K2: tl.constexpr = K // 2
    KS: tl.constexpr = K // 16
    Wb = W13 + e.to(tl.int64)*(2*I*K2); Sb = S13 + e.to(tl.int64)*(2*I*KS)
    up = tl.zeros((16, BN), tl.float32); gate = tl.zeros((16, BN), tl.float32)
    for kb in range(0, K, BK):
        k = kb + tl.arange(0, BK)
        x = tl.load(X + tok[:, None]*K + k[None, :], tok[:, None] >= 0, 0.0)
        up += tl.dot(x, tl.trans(_decode(Wb, Sb, n, kb, BN, BK, K2, KS)))
        gate += tl.dot(x, tl.trans(_decode(Wb, Sb, I + n, kb, BN, BK, K2, KS)))
    g = tl.load(G + e)
    up = up*g; gate = gate*g
    h = gate / (1.0 + tl.exp(-gate)) * up
    tl.store(H + (u*16 + r[:, None])*I + n[None, :], h.to(tl.bfloat16), tok[:, None] >= 0)


def plan(ids):
    """ids [M, top_k] int -> (unique experts [nU] int32, token table [nU, 16] int32 with -1 padding)."""
    m, k = ids.shape
    flat = ids.reshape(-1).long()
    uniq, inv = flat.unique(return_inverse=True)
    toks = torch.arange(m, device=ids.device).repeat_interleave(k)
    table = torch.full((uniq.numel(), 16), -1, dtype=torch.int32, device=ids.device)
    # slot = rank of this (token) within its expert
    order = torch.argsort(inv, stable=True)
    inv_sorted = inv[order]
    first = torch.searchsorted(inv_sorted, torch.arange(uniq.numel(), device=ids.device))
    slot = torch.arange(flat.numel(), device=ids.device) - first[inv_sorted]
    table[inv_sorted, slot] = toks[order].to(torch.int32)
    return uniq.to(torch.int32), table


def gate_up(x, w13_u8, s13_u8, g, uniq, table, *, block_n=32, block_k=128, warps=4, stages=3):
    nU = uniq.numel(); E, I2, K2 = w13_u8.shape; I = I2 // 2; K = K2*2
    h = torch.empty(nU*16, I, device=x.device, dtype=torch.bfloat16)
    _gate_up[(nU, I // block_n)](x, w13_u8, s13_u8, g, uniq, table, h, I, K, block_n, block_k,
                                 num_warps=warps, num_stages=stages)
    return h.view(nU, 16, I)
