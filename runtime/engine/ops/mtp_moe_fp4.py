"""Draft-only NVFP4 W4A16 routed experts for the MTP layer (opt-in SPARK_DRAFT_FP4; half the FP8 bank).

Same job structure as mtp_moe_fp8 (one static (token, slot) job each, no unique/sort or host sync,
graph-capturable; fixed-order top-k sum), with each expert matrix stored as NVFP4: E2M1 codes two per
byte (low nibble = even K) [E, N, K/2], E4M3 per-16 block scales [E, N, K/16] and an FP32 global scale
per expert matrix (engine/ops/fp4_gemv.quantize). Tiles are decoded to BF16 in registers and multiplied
on tensor cores against BF16 activations; the global scale multiplies the FP32 accumulator.
Layout follows MTPMoE.w13 (rows [0, I) = up, [I, 2I) = gate) and w2 [E, H, I].
"""
import torch
import triton as tr
import triton.language as tl

from .moe_w4a16 import _decode


def quantize_bank(bank):
    """[E, N, K] BF16 -> (packed uint8 [E, N, K/2], scales uint8 [E, N, K/16], FP32 global scales [E])."""
    from .fp4_gemv import quantize
    e, n, k = bank.shape
    packed = torch.empty(e, n, k//2, device=bank.device, dtype=torch.uint8)
    scales = torch.empty(e, n, k//16, device=bank.device, dtype=torch.uint8)
    gs = torch.empty(e, device=bank.device, dtype=torch.float32)
    for i in range(e):
        packed[i], scales[i], g = quantize(bank[i]); gs[i] = g
    return packed, scales, gs


@tr.jit
def _gate_up(X, IDS, W, S, GS, H, TOPK: tl.constexpr, I: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    j = tl.program_id(0); nt = tl.program_id(1)
    t = j // TOPK
    e = tl.load(IDS + j).to(tl.int64)
    r = tl.arange(0, 16)
    n = nt*BN + tl.arange(0, BN)
    K2: tl.constexpr = K // 2
    KS: tl.constexpr = K // 16
    Wb = W + e*(2*I*K2); Sb = S + e*(2*I*KS)
    up = tl.zeros((16, BN), tl.float32); gate = tl.zeros((16, BN), tl.float32)
    for kb in range(0, K, BK):
        k = kb + tl.arange(0, BK)
        x = tl.load(X + t*K + k[None, :] + r[:, None]*0, (r[:, None] == 0), 0.0)
        up += tl.dot(x, tl.trans(_decode(Wb, Sb, n, kb, BN, BK, K2, KS)))
        gate += tl.dot(x, tl.trans(_decode(Wb, Sb, I + n, kb, BN, BK, K2, KS)))
    g13 = tl.load(GS + e)
    g = (gate*g13).to(tl.bfloat16).to(tl.float32)
    h = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32) * (up*g13).to(tl.bfloat16).to(tl.float32)
    hv = tl.sum(tl.where(r[:, None] == 0, h, 0.0), 0)
    tl.store(H + j*I + n, hv.to(tl.bfloat16))


@tr.jit
def _down(H, IDS, WT, W2, S2, GS2, P, HID: tl.constexpr, I: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    j = tl.program_id(0); nt = tl.program_id(1)
    e = tl.load(IDS + j).to(tl.int64)
    wt = tl.load(WT + j)
    r = tl.arange(0, 16)
    n = nt*BN + tl.arange(0, BN)
    I2: tl.constexpr = I // 2
    IS: tl.constexpr = I // 16
    Wb = W2 + e*(HID*I2); Sb = S2 + e*(HID*IS)
    acc = tl.zeros((16, BN), tl.float32)
    for kb in range(0, I, BK):
        k = kb + tl.arange(0, BK)
        h = tl.load(H + j*I + k[None, :] + r[:, None]*0, (r[:, None] == 0), 0.0)
        acc += tl.dot(h, tl.trans(_decode(Wb, Sb, n, kb, BN, BK, I2, IS)))
    v = tl.sum(tl.where(r[:, None] == 0, acc, 0.0), 0)*tl.load(GS2 + e)
    tl.store(P + j*HID + n, v.to(tl.bfloat16).to(tl.float32)*wt)


@tr.jit
def _combine(P, Y, TOPK: tl.constexpr, HID: tl.constexpr, BN: tl.constexpr):
    t = tl.program_id(0); n = tl.program_id(1)*BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), tl.float32)
    for s in tl.static_range(TOPK):
        acc += tl.load(P + (t*TOPK + s)*HID + n)
    tl.store(Y + t*HID + n, acc.to(tl.bfloat16))


def routed(x, weights, ids, w13, s13, g13, w2, s2, g2, *, bn_up=64, bn_down=64, bk=128, warps=4, stages=3):
    rows, hid = x.shape; topk = ids.shape[1]; inter = w13.shape[1]//2
    jobs = rows*topk
    ids = ids.reshape(-1).contiguous(); wt = weights.reshape(-1).float().contiguous()
    h = torch.empty(jobs, inter, device=x.device, dtype=torch.bfloat16)
    p = torch.empty(jobs, hid, device=x.device, dtype=torch.float32)
    y = torch.empty(rows, hid, device=x.device, dtype=x.dtype)
    _gate_up[(jobs, inter//bn_up)](x.contiguous(), ids, w13, s13, g13, h, topk, inter, hid, bn_up, bk, num_warps=warps, num_stages=stages)
    _down[(jobs, hid//bn_down)](h, ids, wt, w2, s2, g2, p, hid, inter, bn_down, bk, num_warps=warps, num_stages=stages)
    _combine[(rows, hid//512)](p, y, topk, hid, 512, num_warps=4)
    return y
