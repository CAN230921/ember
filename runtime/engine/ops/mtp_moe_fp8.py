"""Draft-only FP8 (block-128) W8A16 routed experts for the MTP layer, on tensor cores.

The engine keeps the MTP expert bank in BF16 (the official checkpoint stores it as FP8 128x128).
Draft proposals only steer verification -- the target model checks every token -- so this bank is
re-quantized to FP8 block-128 (amax/448, as prepare_fp8) and the BF16 bank is released. Each
(token, slot) pair is one static job (no unique/sort, no host sync, graph-capturable):
  1) gate_up + SwiGLU per job, 2) down projection scaled by the routing weight per job,
  3) fixed-order sum of each token's top-k partials (deterministic).
Layout follows MTPMoE.w13 (rows [0, I) = up, [I, 2I) = gate) and w2 [E, H, I].
"""
import torch
import triton as tr
import triton.language as tl


def quantize_bank(bank, chunk=32):
    """[E, N, K] BF16 -> (FP8 [E, N, K], FP32 scales [E, N/128, K/128])."""
    e, n, k = bank.shape
    q = torch.empty(e, n, k, device=bank.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(e, n//128, k//128, device=bank.device, dtype=torch.float32)
    for a in range(0, e, chunk):
        w = bank[a:a+chunk].float().reshape(-1, n//128, 128, k//128, 128)
        scale = w.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-12)/448
        q[a:a+chunk] = (w/scale).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(-1, n, k)
        s[a:a+chunk] = scale.reshape(-1, n//128, k//128)
        del w, scale
    return q, s


@tr.jit
def _gate_up(X, IDS, W, S, H, TOPK: tl.constexpr, I: tl.constexpr, K: tl.constexpr, BN: tl.constexpr):
    j = tl.program_id(0); nt = tl.program_id(1)
    t = j // TOPK
    e = tl.load(IDS + j).to(tl.int64)
    r = tl.arange(0, 16); kk = tl.arange(0, 128)
    n = nt*BN + tl.arange(0, BN)
    KB: tl.constexpr = K // 128
    Wb = W + e*(2*I*K); Sb = S + e*(2*I//128*KB)
    up = tl.zeros((16, BN), tl.float32); gate = tl.zeros((16, BN), tl.float32)
    for b in range(KB):
        k = b*128 + kk
        x = tl.load(X + t*K + k[None, :] + r[:, None]*0, (r[:, None] == 0), 0.0)
        wu = tl.load(Wb + n[:, None]*K + k[None, :]).to(tl.bfloat16)
        wg = tl.load(Wb + (I + n)[:, None]*K + k[None, :]).to(tl.bfloat16)
        su = tl.load(Sb + (n // 128)*KB + b); sg = tl.load(Sb + ((I + n) // 128)*KB + b)
        up += tl.dot(x, tl.trans(wu))*su[None, :]
        gate += tl.dot(x, tl.trans(wg))*sg[None, :]
    g = gate.to(tl.bfloat16).to(tl.float32)
    h = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32) * up.to(tl.bfloat16).to(tl.float32)
    hv = tl.sum(tl.where(r[:, None] == 0, h, 0.0), 0)
    tl.store(H + j*I + n, hv.to(tl.bfloat16))


@tr.jit
def _down(H, IDS, WT, W2, S2, P, HID: tl.constexpr, I: tl.constexpr, BN: tl.constexpr):
    j = tl.program_id(0); nt = tl.program_id(1)
    e = tl.load(IDS + j).to(tl.int64)
    wt = tl.load(WT + j)
    r = tl.arange(0, 16); kk = tl.arange(0, 128)
    n = nt*BN + tl.arange(0, BN)
    IB: tl.constexpr = I // 128
    Wb = W2 + e*(HID*I); Sb = S2 + e*(HID//128*IB)
    acc = tl.zeros((16, BN), tl.float32)
    for b in range(IB):
        k = b*128 + kk
        h = tl.load(H + j*I + k[None, :] + r[:, None]*0, (r[:, None] == 0), 0.0)
        w = tl.load(Wb + n[:, None]*I + k[None, :]).to(tl.bfloat16)
        s = tl.load(Sb + (n // 128)*IB + b)
        acc += tl.dot(h, tl.trans(w))*s[None, :]
    v = tl.sum(tl.where(r[:, None] == 0, acc, 0.0), 0)
    tl.store(P + j*HID + n, v.to(tl.bfloat16).to(tl.float32)*wt)


@tr.jit
def _combine(P, Y, TOPK: tl.constexpr, HID: tl.constexpr, BN: tl.constexpr):
    t = tl.program_id(0); n = tl.program_id(1)*BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), tl.float32)
    for s in tl.static_range(TOPK):
        acc += tl.load(P + (t*TOPK + s)*HID + n)
    tl.store(Y + t*HID + n, acc.to(tl.bfloat16))


def routed(x, weights, ids, w13, s13, w2, s2, *, bn_up=64, bn_down=64, warps=4, stages=3):
    rows, hid = x.shape; topk = ids.shape[1]; inter = w13.shape[1]//2
    jobs = rows*topk
    ids = ids.reshape(-1).contiguous(); wt = weights.reshape(-1).float().contiguous()
    h = torch.empty(jobs, inter, device=x.device, dtype=torch.bfloat16)
    p = torch.empty(jobs, hid, device=x.device, dtype=torch.float32)
    y = torch.empty(rows, hid, device=x.device, dtype=x.dtype)
    _gate_up[(jobs, inter//bn_up)](x.contiguous(), ids, w13, s13, h, topk, inter, hid, bn_up, num_warps=warps, num_stages=stages)
    _down[(jobs, hid//bn_down)](h, ids, wt, w2, s2, p, hid, inter, bn_down, num_warps=warps, num_stages=stages)
    _combine[(rows, hid//512)](p, y, topk, hid, 512, num_warps=4)
    return y
