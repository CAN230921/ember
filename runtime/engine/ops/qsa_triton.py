"""Packed NVFP4 QSA on sm_121, preserving the oracle's split boundaries."""
import math
import torch
from ..limits import VERIFY_ROWS
import triton as tr
import triton.language as tl


@tr.jit
def _project_qkv(P,Q,K,G,QW,KW,CACHE,SP:tl.constexpr,EPS:tl.constexpr):
    row=tl.program_id(0); head=tl.program_id(1)
    is_key=head>=24
    local_head=tl.where(is_key,head-24,head)
    if is_key:
        source=P+row*SP+12288+local_head*256
        weight=KW
        target=K+row*512+local_head*256
    else:
        source=P+row*SP+local_head*512
        weight=QW
        target=Q+row*6144+local_head*256
    d=tl.arange(0,256)
    x=tl.load(source+d).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x)/256+EPS)
    w=tl.load(weight+d).to(tl.float32)+1.
    normalized=(x*inv*w).to(P.dtype.element_ty).to(tl.float32)
    tl.store(target+d,normalized,d>=64)
    # Preserve the BF16 norm materialization before the fused FP32 RoPE
    # multiply-add. Separate Torch multiplications shifted rare BF16 values
    # on the real layer-19/35 inputs.
    r=tl.arange(0,32)
    first=tl.load(source+r).to(tl.float32)
    second=tl.load(source+32+r).to(tl.float32)
    w1=tl.load(weight+r).to(tl.float32)+1.
    w2=tl.load(weight+32+r).to(tl.float32)+1.
    first=(first*inv*w1).to(P.dtype.element_ty).to(tl.float32)
    second=(second*inv*w2).to(P.dtype.element_ty).to(tl.float32)
    cos=tl.load(CACHE+row*64+r).to(tl.float32)
    sin=tl.load(CACHE+row*64+32+r).to(tl.float32)
    tl.store(target+r,first*cos-second*sin)
    tl.store(target+32+r,second*cos+first*sin)
    if not is_key:
        gate=tl.load(source+256+d)
        tl.store(G+row*6144+local_head*256+d,gate)


def project_qkv(projected,q_weight,k_weight,cache_rows,eps):
    if projected.shape[1]!=13312 or projected.dtype!=torch.bfloat16 or cache_rows.shape!=(projected.shape[0],64):
        raise ValueError('QSA projection supports checkpoint BF16 24/2 heads and text RoPE64')
    rows=projected.shape[0]
    q=torch.empty(rows,24,256,device=projected.device,dtype=projected.dtype)
    k=torch.empty(rows,2,256,device=projected.device,dtype=projected.dtype)
    gate=torch.empty(rows,6144,device=projected.device,dtype=projected.dtype)
    _project_qkv[(rows,26)](projected,q,k,gate,q_weight,k_weight,cache_rows,projected.stride(0),eps,
                           num_warps=4,num_stages=2)
    value=projected[:,12800:].reshape(rows,2,256)
    return q,k,value,gate


@tr.jit
def _fp4(packed,dim):
    halves=tl.inline_asm_elementwise(
        '{ .reg .b8 v; cvt.u8.u32 v, $1; cvt.rn.f16x2.e2m1x2 $0, v; }',
        constraints='=r,r',args=[packed.to(tl.uint32)],dtype=tl.uint32,is_pure=True,pack=1)
    bits=(halves >> ((dim%2)*16)).to(tl.uint16)
    return bits.to(tl.float16,bitcast=True)


@tr.jit(do_not_specialize=['N'])
def _partial(Q, KV, TABLE, IDS, PART, LSE, OUT, N, INDEX_WIDTH: tl.constexpr,
             HEADS: tl.constexpr, KV_HEADS: tl.constexpr, DIM: tl.constexpr,
             GROUP: tl.constexpr, PACKED: tl.constexpr, BN: tl.constexpr,
             TILES: tl.constexpr, ORIGINAL_SPLITS: tl.constexpr,
             SINGLE_ACTIVE: tl.constexpr, BM: tl.constexpr, SCALE: tl.constexpr,
             PAGED:tl.constexpr,BLOCK_SIZE:tl.constexpr):
    row, kv_head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    h = tl.arange(0, BM)
    d = tl.arange(0, DIM)
    query = tl.load(Q + (row*HEADS + kv_head*GROUP + h[:, None])*DIM + d[None, :],
        h[:, None] < GROUP, 0).to(tl.float16)
    maximum = tl.full((BM,), -1.e20, tl.float32)
    total = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, DIM), tl.float32)
    for tile in range(split*TILES//ORIGINAL_SPLITS, (split+1)*TILES//ORIGINAL_SPLITS):
        entry = tile*BN + tl.arange(0, BN)
        ids = tl.load(IDS + row*INDEX_WIDTH + entry, entry < INDEX_WIDTH, -1)
        valid = (ids >= 0) & (ids < N) & (entry < INDEX_WIDTH)
        if PAGED:
            pointer=tl.load(TABLE+ids//BLOCK_SIZE,valid,0).to(tl.pointer_type(tl.uint8))
            address=pointer[:,None]+((ids[:,None]%BLOCK_SIZE)*KV_HEADS+kv_head)*PACKED
        else:
            address=KV+(ids[:, None]*KV_HEADS + kv_head)*PACKED
        key_bits = tl.load(address + d[None, :]//2, valid[:, None], 0)
        key_sf = tl.load(address + DIM + d[None, :]//16, valid[:, None], 0)
        key_sf = key_sf.to(tl.float8e4nv, bitcast=True).to(tl.float16)
        key = (_fp4(key_bits,d[None,:])*key_sf).to(tl.float16)
        value_bits = tl.load(address + DIM//2 + d[None, :]//2, valid[:, None], 0)
        value_sf = tl.load(address + DIM + DIM//16 + d[None, :]//16, valid[:, None], 0)
        value_sf = value_sf.to(tl.float8e4nv, bitcast=True).to(tl.float16)
        value = (_fp4(value_bits,d[None,:])*value_sf).to(tl.float16)
        score = tl.dot(query, tl.trans(key))*SCALE
        score = tl.where(valid[None, :], score, -1.e20)
        next_max = tl.maximum(maximum, tl.max(score, 1))
        alpha = tl.exp2(maximum-next_max)
        probability = tl.where(valid[None, :], tl.exp2(score-next_max[:, None]), 0.)
        acc = tl.dot(probability.to(tl.float16), value, acc=acc*alpha[:, None])
        total = total*alpha + tl.sum(probability, 1)
        maximum = next_max
    normalized = acc/tl.maximum(total[:, None], 1.e-20)
    if SINGLE_ACTIVE:
        tl.store(OUT + (row*HEADS + kv_head*GROUP + h[:, None])*DIM + d[None, :],
            normalized, h[:, None] < GROUP)
    else:
        head = kv_head*GROUP + h
        tl.store(PART + ((row*HEADS + head[:, None])*ORIGINAL_SPLITS + split)*DIM + d[None, :],
            normalized, h[:, None] < GROUP)
        lse = tl.where(total > 0, maximum+tl.log2(tl.maximum(total, 1.e-20)), -float('inf'))
        tl.store(LSE + (row*HEADS + head)*ORIGINAL_SPLITS + split, lse, h < GROUP)


@tr.jit(do_not_specialize=['ACTIVE'])
def _merge(PART, LSE, OUT, ACTIVE, DIM: tl.constexpr, SPLITS: tl.constexpr, BS: tl.constexpr):
    rowhead = tl.program_id(0)
    s, d = tl.arange(0, BS), tl.arange(0, DIM)
    lse = tl.load(LSE + rowhead*SPLITS + s, (s < SPLITS) & (s < ACTIVE), -float('inf'))
    maximum=tl.max(lse,0)
    weights=tl.exp2(tl.where(maximum>-float('inf'),lse-maximum,-float('inf')))
    denominator=tl.sum(weights,0)
    parts = tl.load(PART + (rowhead*SPLITS+s[:, None])*DIM + d[None, :],
                    (s[:, None] < SPLITS) & (s[:, None] < ACTIVE), 0)
    # Divide after the weighted reduction, as in the controlled reference.
    # Normalizing each weight first shifts rare BF16 values across splits.
    output = tl.sum(parts*weights[:, None], 0)
    output = tl.where(denominator>0,output/denominator,0.)
    tl.store(OUT + rowhead*DIM + d, output)


def sparse_attention(query, packed, indices, prefill_mode='packed', *, workspace=None, workspace_owner=None, verify=False):
    """Run the incumbent kernel, optionally borrowing request-owned output.

    Workspace output is overwritten by its next use; consume or clone it first.
    This only reuses output/partial/LSE. Predecoded history, KV tables, and
    dynamic launch metadata retain their existing behavior; no full Graph path
    is enabled here.
    """
    from .kv_cache import PagedKV
    if workspace is not None:
        from .qsa_workspace import QSAWorkspace
        if not isinstance(workspace, QSAWorkspace):
            raise TypeError('Expected a request-owned QSAWorkspace')
        workspace.validate_inputs(query, packed, indices, owner=workspace_owner)
    elif workspace_owner is not None:
        raise ValueError('A workspace owner requires an explicit workspace')
    paged=isinstance(packed,PagedKV)
    if query.dtype != torch.bfloat16 or query.shape[-1] != 256:
        raise ValueError('QSA kernel expects checkpoint BF16 heads of width 256')
    if packed.dtype != torch.uint8 or not query.is_contiguous() or not indices.is_contiguous() or not paged and not packed.is_contiguous():
        raise ValueError('QSA kernel expects contiguous query/indices and a valid packed cache')
    rows, heads, dim = query.shape
    kv_heads = packed.shape[1]
    group = heads//kv_heads
    if heads != 24 or kv_heads != 2 or packed.shape[-1] != dim+dim//8:
        raise ValueError('QSA kernel supports checkpoint 24/2 GQA and NVFP4 cache layout')
    if prefill_mode not in ('packed','predecoded'):raise ValueError('Unknown dedicated QSA prefill mode')
    base = rows*kv_heads
    # Three-token verification must use the ordinary decode split boundary.
    # The wider-tail profile merges 64 rather than 32 history rows per split
    # and changes FP16 softmax materialization at a real 31-token boundary.
    if base <= 6 or verify and rows <= VERIFY_ROWS: block, target, warps = 16, 64, 4
    elif base < 32: block, target, warps = 16, 32, 4
    elif base <= 256: block, target, warps = 64, 8, 2
    elif base <= 512: block, target, warps = 64, 4, 2
    else: block, target, warps = 64, 1, 8
    tiles = tr.cdiv(indices.shape[1], block)
    splits = min(1 << (tiles.bit_length()-1), target)
    # Valid IDs form a prefix in this indexer. Bound which original splits can
    # contain data, while retaining their original start/end tile arithmetic.
    possible_tiles = tr.cdiv(min(packed.shape[0], indices.shape[1]), block)
    active = min(splits, tr.cdiv(possible_tiles, max(1, tiles//splits)))
    # Keep workspace strides and merge reduction width fixed. Context length
    # is a runtime bound; generating a new token must not trigger another JIT.
    if workspace is None:
        out = torch.empty_like(query)
        part = torch.empty(rows, heads, splits, dim, device=query.device, dtype=torch.float32)
        lse = torch.empty(rows, heads, splits, device=query.device, dtype=torch.float32)
    else:
        out, part, lse = workspace.buffers(query, indices, splits=splits, owner=workspace_owner)
    storage=packed._pages[0] if paged else packed
    table=packed.block_table if paged else storage
    # SM121's two-warp tail profile spills heavily while decoding FP4 in each
    # attention tile. Expand history once for those shapes and for paged bulk
    # prefill. The eight-warp dense profile is already faster without staging.
    if prefill_mode=='predecoded' and rows>=32 and (paged or rows<=256):
        from .qsa_prefill_sm121 import unpack_history,_partial_predecoded
        decoded=unpack_history(packed)
        # Half-width value accumulators reduce register spills in the
        # two-warp SM121 profile. Each half retains the full score/softmax
        # calculation and original split boundaries; no reductions combine
        # the independently computed output columns.
        value_dim=128 if warps==2 else 256
        _partial_predecoded[(rows,kv_heads,active*(256//value_dim))](query,decoded.view(torch.uint8),indices,part,lse,out,
            packed.shape[0],indices.shape[1],block,tiles,splits,active==1,
            dim**-.5*math.log2(math.e),value_dim,num_warps=warps,num_stages=2)
    else:
        _partial[(rows, kv_heads, active)](query, storage, table, indices, part, lse, out,
            packed.shape[0], indices.shape[1], heads, kv_heads, dim, group, packed.shape[-1],
            block, tiles, splits, active==1, tr.next_power_of_2(group), dim**-.5*math.log2(math.e),
            paged,packed.block_size if paged else 1,
            # Warp layout is part of the numeric boundary. Preserve the
            # reference's profile when keeping its split arithmetic.
            num_warps=warps,num_stages=2)
    if active > 1:
        _merge[(rows*heads,)](part, lse, out, active, dim, splits, tr.next_power_of_2(splits),
                            num_warps=2, num_stages=1)
    return out
