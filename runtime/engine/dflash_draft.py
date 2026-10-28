"""DFlash2 block drafter (incoai/Qwen3.8-Flash-Next-DFlash2) for single-lane speculative decoding.

Draft-only: proposals are always verified by the target, so this module never changes emitted tokens.

Context: completed outputs of target layers 5/15/25/35/45 (mean over the 4 HC streams), concatenated
[n, 5*2560] -> fc -> hidden_norm -> per drafter layer K (k_norm + RoPE at the token's position) and V.
Only the last `sliding_window` (2048) context positions are attended, so each layer keeps a fixed ring
buffer; RoPE is applied when a row is written, so ring order does not matter (context rows are all
before the anchor and are never masked against each other).

Proposal: block [anchor, MASK x (B-1)] at positions anchor_pos.., 5 layers with non-causal block
attention over (ring context + block), grouped two-tap dynamic convolutions, target LM head top-16 per
slot, then the candidate-selector greedy walk (predecessor/successor codebooks). Math mirrors
diagnostics/teacher_force_dflash_q_20261021.py, whose acceptance was measured against our target.
"""
import json
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F

LAYERS = (5, 15, 25, 35, 45)
# Replay propose/extend as CUDA graphs (static shapes: fixed block, full ring with a validity mask).
DFLASH_GRAPH = os.environ.get('SPARK_DFLASH_GRAPH', '1') == '1'
# Drafter projection weights: 'nvfp4' (E2M1 + E4M3 per-16 + FP32 global on fc and all attention/MLP
# projections, Triton W4A16 for <= 16 rows, dequantized BF16 for wide prefill rows) or 'bf16'.
# Norms, conv kernels, the selector and the shared embedding/head are never quantized.
# (The earlier FP8 W8A16 option faulted inside CUDA graphs on GB10 and was removed.)
DFLASH_WEIGHTS = os.environ.get('SPARK_DFLASH_WEIGHTS', 'nvfp4')
# Fused Triton element-wise kernels (add+RMS, conv, head norm+RoPE, SiLU*up) instead of torch ops.
DFLASH_FUSED = os.environ.get('SPARK_DFLASH_FUSED', '1') == '1'
EXTEND_GRAPH_ROWS = 8


def _capture(fn):
    """Capture fn() into a CUDA graph (thread-local mode: other threads' CUDA calls cannot invalidate it).
    Returns (graph, outputs) or (None, error) after printing the traceback; callers then stay eager."""
    import traceback
    from .graph_capture import graph as capture_graph, shared_pool
    graph = torch.cuda.CUDAGraph()
    try:
        torch.cuda.synchronize()
        pool = shared_pool()
        with capture_graph(graph, capture_error_mode='thread_local', **({} if pool is None else dict(pool=pool))):
            out = fn()
        return graph, out
    except Exception as error:
        print('DFLASH_GRAPH_CAPTURE_FAILED', type(error).__name__, str(error)[:300], flush=True)
        traceback.print_exc()
        torch.cuda.synchronize()
        return None, error


class DFlashContext:
    """Per-request ring buffers of drafter K/V for the last `window` context positions.

    Layout [kv_heads, window + block, head_dim]: slots [0, window) are the ring, the last `block` slots
    hold the proposal block's own K/V (rewritten by every proposal), so attention reads one buffer."""
    def __init__(self, layers, window, kv_heads, head_dim, device, block=0):
        # Zero-filled: masked slots must not hold NaN bit patterns (0 * NaN poisons the softmax sum).
        self.k = [torch.zeros(kv_heads, window+block, head_dim, dtype=torch.bfloat16, device=device) for _ in range(layers)]
        self.v = [torch.zeros_like(self.k[0]) for _ in range(layers)]
        self.mask = torch.zeros(window+block, dtype=torch.bool, device=device)
        self.mask[window:] = True
        self.window = window; self.length = 0      # logical context length (= next position to write)
        self.written = 0                            # rows actually written (a prefix-cache hit starts at its length)
        self.pending = []                           # tapped feature rows not yet written (flushed before a proposal)
        # CUDA graphs bound to this context's buffers (lanes keep their own, so alternating lanes never re-capture)
        self.prop = None; self.prop_out = None; self.ext = {}
        self.anchor = torch.zeros(1, dtype=torch.long, device=device)
        self.start = torch.zeros((), dtype=torch.long, device=device)

    @property
    def valid(self):
        return min(self.written, self.window)


class DFlashDraft:
    def __init__(self, root, engine, block=5, device='cuda', head=None):
        from safetensors.torch import load_file
        root = Path(root)
        cfg = json.loads((root/'config.json').read_text()); dc = cfg['dflash_config']
        if tuple(dc['target_layer_ids']) != LAYERS:
            raise ValueError('Unexpected DFlash target layers')
        self.W = {k: v.to(device) for k, v in load_file(str(root/'model.safetensors')).items()}
        cfg_nl = cfg['num_hidden_layers']
        for l in range(cfg_nl):   # fused projections: one launch for q|k|v and one for gate|up
            a, m = f'layers.{l}.self_attn.', f'layers.{l}.mlp.'
            self.W[f'{l}.qkv'] = torch.cat([self.W.pop(a+n+'_proj.weight') for n in ('q', 'k', 'v')])
            self.W[f'{l}.gu'] = torch.cat([self.W.pop(m+n+'_proj.weight') for n in ('gate', 'up')])
            self.W[f'{l}.o'] = self.W.pop(a+'o_proj.weight'); self.W[f'{l}.down'] = self.W.pop(m+'down_proj.weight')
        self.H, self.NH, self.KVH, self.D = cfg['hidden_size'], cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg['head_dim']
        self.NL, self.EPS = cfg['num_hidden_layers'], cfg['rms_norm_eps']
        self.MASK, self.TOPK, self.WINDOW = dc['mask_token_id'], dc['selector_top_k'], cfg['sliding_window']
        self.G = self.H//dc['conv_group_size']; self.TAPS = dc['conv_kernel_size']
        if not 2 <= block <= 8:
            raise ValueError('DFlash block must be 2..8')
        self.B = block; self.engine = engine; self.device = device
        theta = cfg['rope_parameters']['rope_theta']
        self.inv = 1.0/(theta ** (torch.arange(0, self.D, 2, device=device).float()/self.D))
        # Draft head: the compact draft vocabulary head when given (top-16 over its ids, mapped back to
        # full token ids for the selector codebooks); otherwise the target head.
        self.head = head
        ids = getattr(head, 'vocab_ids', None)
        self.vocab_map = torch.tensor(list(ids), device=device, dtype=torch.long) if ids is not None else None
        self.weights = DFLASH_WEIGHTS; self.fused = DFLASH_FUSED
        rows = self.NH//self.KVH*block   # tensor-core attention tile: GQA group x block rows, a power of two >= 16
        self.tile_attention = rows >= 16 and rows & (rows-1) == 0
        qd, kd = self.NH*self.D, self.KVH*self.D
        self.q4 = {}
        if self.weights == 'nvfp4':
            from .ops.fp4_gemv import quantize
            for name in ['fc.weight']+[f'{l}.{n}' for l in range(self.NL) for n in ('qkv', 'gu', 'o', 'down')]:
                self.q4[name] = quantize(self.W.pop(name))
            for l in range(self.NL):   # k|v rows of the fused q|k|v matrix (extend): row-contiguous views
                pk, sk, gk = self.q4[f'{l}.qkv']; self.q4[f'{l}.kv'] = (pk[qd:], sk[qd:], gk)
            torch.cuda.empty_cache()
        elif self.weights == 'bf16':
            for l in range(self.NL): self.W[f'{l}.kv'] = self.W[f'{l}.qkv'][qd:]
        else:
            raise ValueError('SPARK_DFLASH_WEIGHTS must be nvfp4 or bf16')
        self.deq = {}   # dequantized BF16 copies for calls wider than 16 rows (prefill context)
        self.calls = 0; self.replays = 0
        self.mask_ids = torch.full((block-1,), self.MASK, device=device, dtype=torch.long)
        self.arange_b = torch.arange(block, device=device)
        self.graph_ok = DFLASH_GRAPH

    # ---- shared math (bf16 storage, fp32 reductions; same as the teacher-forcing reference) ----
    def _rms(self, x, w):
        xf = x.float(); return (xf*torch.rsqrt(xf.pow(2).mean(-1, keepdim=True)+self.EPS)*w.float()).to(torch.bfloat16)

    def _rope(self, x, pos):
        f = pos.float()[..., None]*self.inv; cos, sin = f.cos()[..., None, :], f.sin()[..., None, :]
        x1, x2 = x.float().chunk(2, -1)
        return torch.cat((x1*cos-x2*sin, x2*cos+x1*sin), -1).to(torch.bfloat16)

    def _lin(self, x, name):
        q = self.q4.get(name)
        if q is None:
            return F.linear(x, self.W[name])
        if x.shape[0] <= 16:
            from .ops.fp4_gemv import linear as w4a16
            return w4a16(x.contiguous(), *q)
        w = self.deq.get(name)
        if w is None:
            from .ops.fp4_gemv import dequantize
            w = self.deq[name] = dequantize(*q)
        return F.linear(x, w)

    def _add_rms(self, h, res, w):
        """-> (residual, normed): residual = bf16(h + res) (h when res is None)."""
        if self.fused:
            from .ops.dflash_triton import add_rms
            return add_rms(h, res, w, self.EPS)
        if res is not None: h = (h.float()+res.float()).to(torch.bfloat16)
        return h, self._rms(h, w)

    def _conv2(self, x, ca, base_kernel, which):
        """ca: kernel projection rows [R, 2*TAPS*G]; which 0 = pre-op conv, 1 = post-op conv."""
        if self.fused:
            from .ops.dflash_triton import conv
            return conv(x, ca, base_kernel[which], which, self.B, self.TAPS, self.G)
        return self._conv(x, ca.reshape(-1, 2, self.TAPS, self.G)[:, which], base_kernel[which])

    def _heads(self, y, pos, q_heads, wq, k_off, wk):
        """Per-head RMS norm + RoPE of q (columns [0, NH*D)) and k (columns [k_off, k_off+KVH*D)) of y."""
        n = y.shape[0]
        if self.fused:
            from .ops.dflash_triton import head_rope
            return head_rope(y, pos, self.inv, self.EPS, q_off=0, q_heads=q_heads, wq=wq, k_off=k_off,
                             k_heads=self.KVH, wk=wk, head_dim=self.D)
        q = self._rope(self._rms(y[:, :q_heads*self.D].reshape(n, q_heads, self.D), wq), pos) if q_heads else None
        k = self._rope(self._rms(y[:, k_off:k_off+self.KVH*self.D].reshape(n, self.KVH, self.D), wk), pos)
        return q, k

    def _silu_mul(self, gu):
        if self.fused:
            from .ops.dflash_triton import silu_mul
            return silu_mul(gu)
        g, u = gu.chunk(2, -1)
        return F.silu(g.float()).to(torch.bfloat16)*u

    def _conv(self, x, delta, base):
        B, G, H, TAPS = self.B, self.G, self.H, self.TAPS
        blocks = x.float().unflatten(-1, (G, H//G)); coef = base.float().view(1, TAPS, G, H//G)+delta.float().unsqueeze(-1)
        out = coef[:, 0]*blocks; pos = torch.arange(x.shape[0], device=x.device) % B
        for t in range(1, TAPS):
            shifted = F.pad(blocks[:-t], (0, 0, 0, 0, t, 0)); out = out+coef[:, t]*shifted*(pos >= t).view(-1, 1, 1)
        return out.flatten(-2).to(torch.bfloat16)

    # ---- context ----
    def new_context(self, start=0):
        ctx = DFlashContext(self.NL, self.WINDOW, self.KVH, self.D, self.device, self.B); ctx.length = start
        return ctx

    @staticmethod
    def tap(hidden, block, injection, hc_count=4):
        """Completed layer output, mean over HC streams -> [rows, 2560] bf16 (call inside the target observer)."""
        from .ops.hyperconnection import hc_combine
        done = hc_combine(hidden, block, injection, hc_count) if block is not None else hidden
        return done.float().unflatten(-1, (hc_count, done.shape[-1]//hc_count)).mean(-2).to(torch.bfloat16)

    def _extend_core(self, ctx, feats, start):
        n = feats.shape[0]
        positions = start+torch.arange(n, device=feats.device)
        _, h = self._add_rms(self._lin(feats, 'fc.weight'), None, self.W['hidden_norm.weight'])
        slots = positions % ctx.window
        kd = self.KVH*self.D
        for l in range(self.NL):
            kv = self._lin(h, f'{l}.kv')
            _, k = self._heads(kv, positions, 0, None, 0, self.W[f'layers.{l}.self_attn.k_norm.weight'])
            ctx.k[l].index_copy_(1, slots, k.transpose(0, 1))
            ctx.v[l].index_copy_(1, slots, kv[:, kd:].reshape(n, self.KVH, self.D).transpose(0, 1))
        ctx.mask.index_fill_(0, slots, True)

    def push(self, ctx, feats):
        """Queue committed feature rows; they are written (one batched extend) before the next proposal.
        At most a window of rows is ever kept, so MTP-only stretches stay bounded."""
        ctx.pending.append(feats); ctx.length += feats.shape[0]
        while sum(f.shape[0] for f in ctx.pending) > ctx.window+16:
            ctx.pending.pop(0); ctx.trimmed = True   # a gap: every older ring row is out of the window

    def flush(self, ctx):
        if not ctx.pending: return
        feats = torch.cat(ctx.pending) if len(ctx.pending) > 1 else ctx.pending[0]
        ctx.pending = []; ctx.length -= feats.shape[0]
        if getattr(ctx, 'trimmed', False):
            ctx.mask.zero_(); ctx.written = 0; ctx.trimmed = False
        self.extend(ctx, feats)

    def extend(self, ctx, feats):
        """Append context rows (target features of the next consecutive positions), keeping the last window."""
        n = feats.shape[0]
        if n == 0: return
        if n > ctx.window:
            ctx.length += n-ctx.window; ctx.written += n-ctx.window; feats = feats[-ctx.window:]; n = ctx.window
        if self.graph_ok and feats.is_cuda and n <= EXTEND_GRAPH_ROWS:
            g = ctx.ext.get(n)
            if g is None:
                g = dict(feats=feats.clone(), start=torch.zeros((), dtype=torch.long, device=feats.device), graph=None, calls=0)
                ctx.ext[n] = g
            g['feats'].copy_(feats); g['start'].fill_(ctx.length); g['calls'] += 1
            if g['graph'] is None and g['calls'] > 1:
                g['graph'], _ = _capture(lambda: self._extend_core(ctx, g['feats'], g['start']))
                if g['graph'] is None:
                    self.graph_ok = False; self._extend_core(ctx, g['feats'], g['start'])
                else:
                    g['graph'].replay()      # capture does not execute
            elif g['graph'] is not None:
                g['graph'].replay()
            else:
                self._extend_core(ctx, g['feats'], g['start'])
        else:
            self._extend_core(ctx, feats, torch.tensor(ctx.length, device=feats.device))
        ctx.length += n; ctx.written += n

    # ---- proposal ----
    def _propose_core(self, ctx, anchor, start):
        B, H, NH, KVH, D, W = self.B, self.H, self.NH, self.KVH, self.D, self.W
        qd, kd, win = NH*D, KVH*D, ctx.window
        ids = torch.cat((anchor, self.mask_ids))
        pos = start+self.arange_b
        mask = ctx.mask[None, None, None].expand(1, 1, B, -1)
        h = self.engine.embedding[ids]; res = None
        for l in range(self.NL):
            pre = f'layers.{l}.'
            res, x = self._add_rms(h, res, W[pre+'input_layernorm.weight'])
            ca = self._lin(x, pre+'attention_conv.kernel_projection.weight')
            x = self._conv2(x, ca, W[pre+'attention_conv.base_kernel'], 0)
            qkv = self._lin(x, f'{l}.qkv')
            q, k = self._heads(qkv, pos, NH, W[pre+'self_attn.q_norm.weight'], qd, W[pre+'self_attn.k_norm.weight'])
            ctx.k[l][:, win:].copy_(k.transpose(0, 1))
            ctx.v[l][:, win:].copy_(qkv[:, qd+kd:].reshape(B, KVH, D).transpose(0, 1))
            if self.fused and self.tile_attention:
                from .ops.dflash_triton import block_attention
                o = block_attention(q, ctx.k[l], ctx.v[l], ctx.mask).reshape(B, NH*D)
            else:   # (SDPA with a mask runs the FP32 math path here: K/V upcast and expanded per query head)
                o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], ctx.k[l][None], ctx.v[l][None],
                                                   attn_mask=mask, enable_gqa=True)[0].transpose(0, 1).reshape(B, NH*D)
            o = self._lin(o, f'{l}.o')
            o = self._conv2(o, ca, W[pre+'attention_conv.base_kernel'], 1)
            res, x = self._add_rms(o, res, W[pre+'post_attention_layernorm.weight'])
            cm = self._lin(x, pre+'mlp_conv.kernel_projection.weight')
            x = self._conv2(x, cm, W[pre+'mlp_conv.base_kernel'], 0)
            x = self._lin(self._silu_mul(self._lin(x, f'{l}.gu')), f'{l}.down')
            h = self._conv2(x, cm, W[pre+'mlp_conv.base_kernel'], 1)
        final = self._add_rms(h, res, W['norm.weight'])[1][1:]                                     # [B-1, H]
        from .ops.int8_head import ROW_LIMIT
        head = self.head if self.head is not None else self.engine.logits
        step = ROW_LIMIT if getattr(self.engine, 'head_backend', '') == 'sm121' else B-1
        logits = torch.cat([head(final[a:a+step].contiguous()) for a in range(0, B-1, step)]).float()
        vals_, cand = logits.topk(self.TOPK, -1)                                                   # [B-1, K]
        if self.vocab_map is not None: cand = self.vocab_map[cand]
        hp = self._lin(final, 'candidate_selector.hidden_projection.weight').float()
        keys_ = W['candidate_selector.successor_codebook'][cand].float()                           # [s, K, r]
        pred_ids = torch.cat((ids[:1, None].expand(1, self.TOPK), cand[:-1]), 0)
        preds = W['candidate_selector.predecessor_codebook'][pred_ids].float()                     # [s, K, r]
        score = vals_[:, None, :]+torch.einsum('lpr,lcr->lpc', preds*hp[:, None], keys_)
        ix = score[0, 0].argmax(-1); path = [ix]
        for e in range(1, B-1):
            # index_select keeps the walk on the device (a 0-d tensor subscript would sync to the host)
            ix = score[e].index_select(0, ix.view(1))[0].argmax(-1); path.append(ix)
        return cand.gather(1, torch.stack(path)[:, None])[:, 0]

    @torch.no_grad()
    def propose(self, ctx, anchor):
        """anchor: committed token id at position ctx.length (its target features are not context yet).
        Returns B-1 draft token ids for positions ctx.length+1 ..."""
        self.calls += 1
        self.flush(ctx)
        ctx.anchor.fill_(int(anchor)); ctx.start.fill_(ctx.length)
        if not self.graph_ok:
            return self._propose_core(ctx, ctx.anchor, ctx.start).tolist()
        if ctx.prop is None:
            out = self._propose_core(ctx, ctx.anchor, ctx.start).tolist()   # eager warm-up for this request
            graph, value = _capture(lambda: self._propose_core(ctx, ctx.anchor, ctx.start))
            if graph is None:
                self.graph_ok = False
            else:
                ctx.prop, ctx.prop_out = graph, value
            return out
        ctx.prop.replay(); self.replays += 1
        return ctx.prop_out.tolist()
