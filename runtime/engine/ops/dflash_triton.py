"""Fused element-wise kernels for the DFlash2 drafter (draft-only; same math as the torch reference in
engine/dflash_draft.py with the same BF16 rounding points, not guaranteed bit-equal: FP32 reduction order
and cos/sin differ).

  add_rms      r = bf16(h + res); x = bf16(rms(r) * w)            (res optional)
  conv         grouped dynamic two-tap convolution over block rows (coef = base + per-row delta)
  head_rope    per-head RMS norm + NeoX RoPE for q/k heads read from a fused projection output
  silu_mul     bf16(bf16(silu(g)) * u) for a fused [gate | up] row
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _add_rms(Hp, Rp, Wp, Ro, Xo, H: tl.constexpr, BLOCK: tl.constexpr, HAS_RES: tl.constexpr, EPS):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK); ok = c < H
    h = tl.load(Hp + r*H + c, ok, 0.0).to(tl.float32)
    if HAS_RES:
        h = (h + tl.load(Rp + r*H + c, ok, 0.0).to(tl.float32)).to(tl.bfloat16)
        tl.store(Ro + r*H + c, h, ok)
        h = h.to(tl.float32)
    var = tl.sum(h*h, 0)/H
    w = tl.load(Wp + c, ok, 0.0).to(tl.float32)
    tl.store(Xo + r*H + c, (h*tl.rsqrt(var + EPS)*w).to(tl.bfloat16), ok)


def add_rms(h, res, w, eps):
    """-> (new residual, normed); new residual is h itself when res is None."""
    rows, n = h.shape
    x = torch.empty_like(h)
    out = torch.empty_like(h) if res is not None else h
    _add_rms[(rows,)](h, res if res is not None else h, w, out, x, n, tr.next_power_of_2(n), res is not None, eps,
                      num_warps=8)
    return out, x


@tr.jit
def _conv(Xp, Cp, CS, Bp, Op, B: tl.constexpr, H: tl.constexpr, GS: tl.constexpr, TAPS: tl.constexpr,
          G: tl.constexpr, WHICH: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK); ok = c < H
    g = c // GS
    pos = r % B
    acc = tl.zeros((BLOCK,), tl.float32)
    for t in tl.static_range(TAPS):
        coef = tl.load(Bp + t*H + c, ok, 0.0).to(tl.float32) + \
            tl.load(Cp + r*CS + (WHICH*TAPS + t)*G + g, ok, 0.0).to(tl.float32)
        x = tl.load(Xp + (r - t)*H + c, ok & (pos >= t), 0.0).to(tl.float32)
        acc += coef*x
    tl.store(Op + r*H + c, acc.to(tl.bfloat16), ok)


def conv(x, ca, base, which, block, taps, groups):
    """x [R, H] bf16, ca [R, 2*taps*groups] kernel projection (layout [2, taps, groups]), base [taps*H]."""
    rows, n = x.shape
    out = torch.empty_like(x)
    _conv[(rows,)](x, ca, ca.stride(0), base, out, block, n, n//groups, taps, groups, which,
                   tr.next_power_of_2(n), num_warps=4)
    return out


@tr.jit
def _head_rope(Yp, YS, Pp, INV, Wq, Wk, Qo, Ko, QOFF, KOFF, NQ: tl.constexpr, NK: tl.constexpr,
               D: tl.constexpr, EPS):
    r = tl.program_id(0); hd = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    i = tl.arange(0, HALF)
    if hd < NQ:
        src = Yp + r*YS + QOFF + hd*D; wp = Wq; dst = Qo + (r*NQ + hd)*D
    else:
        src = Yp + r*YS + KOFF + (hd - NQ)*D; wp = Wk; dst = Ko + (r*NK + hd - NQ)*D
    a = tl.load(src + i).to(tl.float32); b = tl.load(src + HALF + i).to(tl.float32)
    s = tl.rsqrt((tl.sum(a*a, 0) + tl.sum(b*b, 0))/D + EPS)
    a = (a*s*tl.load(wp + i).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    b = (b*s*tl.load(wp + HALF + i).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    f = tl.load(Pp + r).to(tl.float32)*tl.load(INV + i)
    cs = tl.cos(f); sn = tl.sin(f)
    tl.store(dst + i, (a*cs - b*sn).to(tl.bfloat16))
    tl.store(dst + HALF + i, (b*cs + a*sn).to(tl.bfloat16))


def head_rope(y, pos, inv, eps, *, q_off=None, q_heads=0, wq=None, k_off=0, k_heads=0, wk=None, head_dim=128):
    """y [R, *] fused projection rows; returns (q [R, q_heads, D] or None, k [R, k_heads, D])."""
    rows = y.shape[0]
    q = torch.empty(rows, max(q_heads, 1), head_dim, device=y.device, dtype=torch.bfloat16) if q_heads else None
    k = torch.empty(rows, k_heads, head_dim, device=y.device, dtype=torch.bfloat16)
    _head_rope[(rows, q_heads + k_heads)](y, y.stride(0), pos, inv, wq if wq is not None else wk, wk,
                                          q if q is not None else k, k, q_off or 0, k_off, q_heads, k_heads,
                                          head_dim, eps, num_warps=1)
    return q, k


@tr.jit
def _silu_mul(Gp, Op, I: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0); c = tl.program_id(1)*BLOCK + tl.arange(0, BLOCK); ok = c < I
    g = tl.load(Gp + r*2*I + c, ok, 0.0).to(tl.float32)
    u = tl.load(Gp + r*2*I + I + c, ok, 0.0).to(tl.float32)
    s = (g/(1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(Op + r*I + c, (s*u).to(tl.bfloat16), ok)


def silu_mul(gu):
    rows, two = gu.shape
    i = two//2
    out = torch.empty(rows, i, device=gu.device, dtype=torch.bfloat16)
    _silu_mul[(rows, tr.cdiv(i, 1024))](gu, out, i, 1024, num_warps=4)
    return out


@tr.jit
def _attn_part(Q, K, V, Mk, Mo, Lo, Ao, L, SCALE, B: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr,
               D: tl.constexpr, CHUNK: tl.constexpr, BL: tl.constexpr):
    # One (kv head, key chunk): the GQA group's NQ/NK query heads x B rows form one tensor-core tile.
    kh = tl.program_id(0); part = tl.program_id(1)
    GR: tl.constexpr = NQ // NK
    R: tl.constexpr = GR*B                                # tile row r = (head g, block row b): r = g*B + b
    r = tl.arange(0, R); d = tl.arange(0, D)
    qh = kh*GR + r // B; qb = r % B
    q = tl.load(Q + (qb[:, None]*NQ + qh[:, None])*D + d[None, :])       # [R, D] bf16, q [B, NQ, D]
    m = tl.full((R,), float('-inf'), tl.float32); s_ = tl.zeros((R,), tl.float32); acc = tl.zeros((R, D), tl.float32)
    for j0 in range(part*CHUNK, part*CHUNK + CHUNK, BL):
        j = j0 + tl.arange(0, BL); inside = j < L
        ok = inside & (tl.load(Mk + j, inside, 0) != 0)
        k = tl.load(K + (kh*L + j[:, None])*D + d[None, :], inside[:, None], 0.0)        # [BL, D]
        v = tl.load(V + (kh*L + j[:, None])*D + d[None, :], inside[:, None], 0.0)
        sc = tl.dot(q, tl.trans(k))*SCALE
        sc = tl.where(ok[None, :], sc, float('-inf'))
        mn = tl.maximum(m, tl.max(sc, 1))
        safe = tl.where(mn == float('-inf'), 0.0, mn)
        p = tl.exp(sc - safe[:, None])
        alpha = tl.exp(m - safe)
        s_ = s_*alpha + tl.sum(p, 1)
        acc = acc*alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m = mn
    base = (kh*tl.num_programs(1) + part)*R
    tl.store(Mo + base + r, m); tl.store(Lo + base + r, s_)
    tl.store(Ao + (base + r[:, None])*D + d[None, :], acc)


@tr.jit
def _attn_merge(Mo, Lo, Ao, O, P: tl.constexpr, B: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr):
    kh = tl.program_id(0)
    GR: tl.constexpr = NQ // NK
    R: tl.constexpr = GR*B
    r = tl.arange(0, R); d = tl.arange(0, D)
    mt = tl.full((R,), float('-inf'), tl.float32)
    for p in range(P):
        mt = tl.maximum(mt, tl.load(Mo + (kh*P + p)*R + r))
    s_ = tl.zeros((R,), tl.float32); acc = tl.zeros((R, D), tl.float32)
    for p in range(P):
        mp = tl.load(Mo + (kh*P + p)*R + r)
        w = tl.where(mp == float('-inf'), 0.0, tl.exp(mp - mt))
        s_ += tl.load(Lo + (kh*P + p)*R + r)*w
        acc += tl.load(Ao + ((kh*P + p)*R + r[:, None])*D + d[None, :])*w[:, None]
    qh = kh*GR + r // B; qb = r % B
    tl.store(O + (qb[:, None]*NQ + qh[:, None])*D + d[None, :], (acc/s_[:, None]).to(tl.bfloat16))


def block_attention(q, k, v, mask, chunk=256, bl=64):
    """q [B, NQ, D] (block rows), k/v [NK, L, D] (ring + block slots), mask [L] bool -> [B, NQ, D] bf16.
    Non-causal masked attention (every block row sees every valid key), flash-decoding split over keys."""
    b, nq, d = q.shape; nk, length, _ = k.shape
    parts = tr.cdiv(length, chunk); rows = nq//nk*b
    m = torch.empty(nk*parts*rows, device=q.device, dtype=torch.float32); s = torch.empty_like(m)
    a = torch.empty(nk*parts*rows, d, device=q.device, dtype=torch.float32)
    out = torch.empty(b, nq, d, device=q.device, dtype=torch.bfloat16)
    _attn_part[(nk, parts)](q, k, v, mask, m, s, a, length, d**-0.5, b, nq, nk, d, chunk, bl, num_warps=4)
    _attn_merge[(nk,)](m, s, a, out, parts, b, nq, nk, d, num_warps=4)
    return out
