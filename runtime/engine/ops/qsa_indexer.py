# SPDX-License-Identifier: Apache-2.0
# The norm/RoPE and score arithmetic is adapted from the vLLM QSA kernels.
# Copyright contributors to the vLLM project. See diagnostics/original-qsa-20261001.
"""GB10 text-only QSA indexer with the incumbent rounding and block order.

Contiguous BF16 compressed keys replace the paged indexer side cache. The
compiled deterministic top-k is reused explicitly; it is third-party code.
"""
import math
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _norm_rope_rows(x, positions, CACHE, W, EPS: tl.constexpr,
                    MROPE: tl.constexpr, MH: tl.constexpr, MW: tl.constexpr, KEY: tl.constexpr=False):
    T: tl.constexpr = x.shape[0]
    H: tl.constexpr = x.shape[1]
    ROWS: tl.constexpr = T*H
    p = tl.arange(0, 32)
    if MROPE:
        h_mask = ((p%3) == 1) & (p <= 3*MH)
        w_mask = ((p%3) == 2) & (p <= 3*MW)
        masks = (~(h_mask | w_mask), h_mask, w_mask)
        cos = tl.zeros((T, 32), dtype=CACHE.dtype.element_ty)
        sin = tl.zeros((T, 32), dtype=CACHE.dtype.element_ty)
        for axis in tl.static_range(3):
            cos += tl.load(CACHE+positions[:, None]*64+p[None, :], masks[axis][None, :], 0)
            sin += tl.load(CACHE+positions[:, None]*64+32+p[None, :], masks[axis][None, :], 0)
    else:
        cos = tl.load(CACHE+positions[:, None]*64+p[None, :])
        sin = tl.load(CACHE+positions[:, None]*64+32+p[None, :])
    cos = tl.reshape(tl.broadcast_to(cos[:, None, :], (T, H, 32)), (ROWS, 32))
    sin = tl.reshape(tl.broadcast_to(sin[:, None, :], (T, H, 32)), (ROWS, 32))
    x = tl.reshape(x, (ROWS, 128)).to(tl.float32)
    weight = tl.load(W+tl.arange(0, 128)).to(tl.float32)+1.
    rrms = tl.rsqrt(tl.sum(x*x, axis=1)/128+EPS)
    y = (x*rrms[:, None]*weight[None, :]).to(cos.dtype)
    rotated, passthrough = tl.split(tl.permute(tl.reshape(y, (ROWS, 2, 64)), (0, 2, 1)))
    r0, r1 = tl.split(tl.permute(tl.reshape(rotated, (ROWS, 2, 32)), (0, 2, 1)))
    out0 = r0*cos-r1*sin
    if KEY:
        # The incumbent compression branch stores r1*cos as BF16, then
        # contracts r0*sin with the addition. Pin both boundaries explicitly.
        a=(r1.to(tl.float32)*cos.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        out1=tl.fma(r0.to(tl.float32),sin.to(tl.float32),a).to(out0.dtype)
    else:out1 = r1*cos+r0*sin
    rotated = tl.reshape(tl.permute(tl.join(out0, out1), (0, 2, 1)), (ROWS, 64))
    result = tl.reshape(tl.permute(tl.join(rotated, passthrough), (0, 2, 1)), (ROWS, 128))
    return tl.reshape(result, (T, H, 128))


@tr.jit
def _norm_rope(X, W, CACHE, Y, SX: tl.constexpr, SY: tl.constexpr,
               N: tl.constexpr, HEADS: tl.constexpr, HT: tl.constexpr, EPS: tl.constexpr,
               MROPE: tl.constexpr, MH: tl.constexpr, MW: tl.constexpr):
    rows = tl.program_id(0)*2+tl.arange(0, 2)
    heads = tl.program_id(1)*HT+tl.arange(0, HT)
    dims = tl.arange(0, 128)
    valid = (rows[:, None, None] < N) & (heads[None, :, None] < HEADS)
    x = tl.load(X+rows[:, None, None]*SX+heads[None, :, None]*128+dims[None, None, :], valid, 0.)
    safe_rows = tl.minimum(rows, N-1)
    y = _norm_rope_rows(x, safe_rows, CACHE, W, EPS, MROPE, MH, MW)
    tl.store(Y+rows[:, None, None]*SY+heads[None, :, None]*128+dims[None, None, :], y, valid)


def norm_rope(x, weight, cache_rows, eps, mrope_section=None):
    if x.ndim != 3 or x.shape[-1] != 128 or x.dtype != torch.bfloat16:
        raise ValueError('Indexer requires BF16 [rows, heads, 128]')
    if x.stride(-1) != 1 or x.stride(1) != 128 or cache_rows.shape != (x.shape[0], 64):
        raise ValueError('Invalid indexer input strides or RoPE rows')
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    heads_per_tile = 2 if x.shape[0] <= 4096 else 4
    section = mrope_section or (0, 0, 0)
    _norm_rope[(tr.cdiv(x.shape[0], 2), tr.cdiv(x.shape[1], heads_per_tile))](x, weight, cache_rows, out,
        x.stride(0), out.stride(0), x.shape[0], x.shape[1], heads_per_tile, eps,
        bool(mrope_section), section[1], section[2], num_warps=1)
    return out


@tr.jit
def _compress(RAW, CACHE, W, OUT, SR: tl.constexpr, EPS: tl.constexpr,
              MROPE: tl.constexpr, MH: tl.constexpr, MW: tl.constexpr):
    group = tl.program_id(0)
    d = tl.arange(0, 128)
    source = tl.load(RAW+(group*4+tl.arange(0, 4))[:, None]*SR+d[None, :]).to(tl.float32)
    pooled = (tl.sum(source, axis=0)/4).to(tl.bfloat16).to(tl.float32)
    y = _norm_rope_rows(tl.reshape(pooled, (1, 1, 128)), group+tl.arange(0, 1), CACHE, W, EPS, MROPE, MH, MW, True)
    tl.store(OUT+group*128+d, tl.reshape(y, (128,)))


def compress(raw, weight, cache_rows, eps, mrope_section=None):
    if raw.shape[1:] != (1, 128) or raw.shape[0]%4 or raw.dtype != torch.bfloat16:
        raise ValueError('Compression requires complete groups of four BF16 raw keys')
    groups = raw.shape[0]//4
    if cache_rows.shape != (groups, 64):
        raise ValueError('One RoPE row is required for each completed group')
    out = torch.empty(groups, 1, 128, device=raw.device, dtype=raw.dtype)
    section = mrope_section or (0, 0, 0)
    _compress[(groups,)](raw, cache_rows, weight, out, raw.stride(0), eps,
                       bool(mrope_section), section[1], section[2], num_warps=1)
    return out


@tr.jit(do_not_specialize=['GROUPS','COLUMNS'])
def _score(Q, K, POS, SCORE, VISIBLE, GROUPS,
           SQ: tl.constexpr, SH: tl.constexpr, SK: tl.constexpr,
           COLUMNS, TILES: tl.constexpr, DIVISOR: tl.constexpr):
    row = tl.program_id(0)
    visible = tl.minimum((tl.load(POS+row)+1)//4, GROUPS)
    if tl.program_id(1) == 0:
        tl.store(VISIBLE+row, visible)
    dims = tl.arange(0, 128)
    heads = tl.arange(0, 16)
    query = tl.load(Q+row*SQ+heads[None, :]*SH+dims[:, None],
                    heads[None, :] < 4, 0)
    begin = tl.program_id(1)*TILES
    end = tl.minimum(begin+TILES, tl.cdiv(COLUMNS, 64))
    for tile in tl.range(begin, end, num_stages=2):
        columns = tile*64+tl.arange(0, 64)
        valid = columns < visible
        keys = tl.load(K+columns[:, None]*SK+dims[None, :], valid[:, None], 0,
                       eviction_policy='evict_first')
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.where(heads[None, :] < 4, tl.maximum(scores, 0.), 0.)
        score = tl.sum(scores, axis=1)/DIVISOR
        tl.store(SCORE+row*COLUMNS+columns, tl.where(valid, score, -float('inf')),
                 columns < COLUMNS)


def scores(queries, compressed, positions, columns=None):
    if queries.shape[1:] != (4, 128) or queries.dtype != torch.bfloat16:
        raise ValueError('Sparse indexer requires BF16 four-head queries')
    if compressed.shape[1:] != (1, 128) or compressed.dtype != torch.bfloat16:
        raise ValueError('Sparse indexer requires BF16 compressed keys')
    columns = tr.cdiv(compressed.shape[0], 16)*16 if columns is None else columns
    out = torch.empty(queries.shape[0], columns, device=queries.device, dtype=torch.float32)
    visible = torch.empty(queries.shape[0], device=queries.device, dtype=torch.int32)
    tiles = 1 if queries.shape[0] <= 32 else 8
    _score[(queries.shape[0], tr.cdiv(columns, 64*tiles))](queries, compressed,
        positions, out, visible, compressed.shape[0], queries.stride(0), queries.stride(1),
        compressed.stride(0), columns, tiles, math.sqrt(128), num_warps=2)
    return out, visible


@tr.jit
def _expand(BLOCKS, POS, OUT, WIDTH: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0)
    column = tl.program_id(1)*BN+tl.arange(0, BN)
    position = tl.load(POS+row)
    visible = (position+1)//4
    expanded = tl.minimum(visible, 512)*4
    rank = tl.minimum(column//4, 511)
    block = tl.load(BLOCKS+row*512+rank, column < expanded, -1)
    tail = visible*4+(column-expanded)
    token = tl.where(column < expanded, block*4+column%4, tail)
    valid = (column < expanded+(position+1)%4) & (token >= 0) & (token <= position)
    tl.store(OUT+row*WIDTH+column, tl.where(valid, token, -1), column < WIDTH)


@tr.jit
def _full_prefix(POS,OUT,WIDTH:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0)
    column=tl.arange(0,B)
    position=tl.load(POS+row)
    tl.store(OUT+row*WIDTH+column,tl.where(column<=position,column,-1),column<WIDTH)


def full_prefix(positions):
    """Exact logical-order IDs when the complete prefix fits the QSA budget."""
    out=torch.empty(positions.numel(),2051,device=positions.device,dtype=torch.int32)
    _full_prefix[(positions.numel(),)](positions,out,2051,4096,num_warps=4)
    return out


def select(queries, compressed, positions):
    from .native_kernels import qsa_topk
    rows = queries.shape[0]
    columns = tr.cdiv(compressed.shape[0], 16)*16
    # Bound score workspace even when prefill has hundreds of thousands of rows.
    per_chunk = max(1, (128*1024*1024)//max(4*columns, 1))
    out = torch.empty(rows, 2051, device=queries.device, dtype=torch.int32)
    workspace = torch.empty(1024*1024, device=queries.device, dtype=torch.uint8)
    for start in range(0, rows, per_chunk):
        end = min(rows, start+per_chunk)
        q, pos = queries[start:end], positions[start:end]
        score, visible = scores(q, compressed, pos, columns)
        blocks = torch.empty(end-start, 512, device=queries.device, dtype=torch.int32)
        qsa_topk(score, visible, blocks, workspace, 512, columns)
        _expand[(end-start, tr.cdiv(2051, 256))](blocks, pos, out[start:end], 2051, 256)
    return out
