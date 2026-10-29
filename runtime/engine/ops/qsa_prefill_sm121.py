# SPDX-License-Identifier: Apache-2.0
"""Bulk QSA for GB10's fixed 24/2 heads, D256, packed NVFP4 history.

Decode keeps its packed kernel. Bulk queries can unpack history once, retaining
the incumbent FP16 operands, tile boundaries and reduction order. The temporary
FP16 buffer is never stored in request state or prefix checkpoints.
"""
import torch
import triton as tr
import triton.language as tl
from .qsa_triton import _fp4


@tr.jit(do_not_specialize=['N'])
def _unpack_history(KV,TABLE,OUT,N,PAGED:tl.constexpr,BLOCK_SIZE:tl.constexpr,R:tl.constexpr):
    entry=tl.program_id(0)*R+tl.arange(0,R)
    row=entry//2;head=entry%2;d=tl.arange(0,256);valid=row<N
    if PAGED:
        pointer=tl.load(TABLE+row//BLOCK_SIZE,valid,0).to(tl.pointer_type(tl.uint8))
        address=pointer[:,None]+((row[:,None]%BLOCK_SIZE)*2+head[:,None])*288
    else:address=KV+entry[:,None]*288
    key_bits=tl.load(address+d[None,:]//2,valid[:,None],0)
    key_sf=tl.load(address+256+d[None,:]//16,valid[:,None],0)
    key_sf=key_sf.to(tl.float8e4nv,bitcast=True).to(tl.float16)
    key=(_fp4(key_bits,d[None,:])*key_sf).to(tl.float16)
    tl.store(OUT+entry[:,None]*512+d[None,:],key,valid[:,None])
    value_bits=tl.load(address+128+d[None,:]//2,valid[:,None],0)
    value_sf=tl.load(address+272+d[None,:]//16,valid[:,None],0)
    value_sf=value_sf.to(tl.float8e4nv,bitcast=True).to(tl.float16)
    value=(_fp4(value_bits,d[None,:])*value_sf).to(tl.float16)
    tl.store(OUT+entry[:,None]*512+256+d[None,:],value,valid[:,None])


def unpack_history(packed):
    from .kv_cache import PagedKV
    paged=isinstance(packed,PagedKV);n=packed.shape[0]
    out=torch.empty(n,2,512,device=packed.device,dtype=torch.float16)
    storage=packed._pages[0] if paged else packed
    table=packed.block_table if paged else storage
    _unpack_history[(tr.cdiv(n*2,8),)](storage,table,out,n,paged,
        packed.block_size if paged else 1,8,num_warps=4,num_stages=1)
    return out


@tr.jit
def _half_bytes(address,valid):
    # Keep byte-source dot operands, as in the FP4 conversion.
    # Native FP16 loads make Triton choose kWidth=2 instead of kWidth=4,
    # changing MMA accumulation order even with identical operand bits.
    low=tl.load(address,valid,0).to(tl.uint16)
    high=tl.load(address+1,valid,0).to(tl.uint16)
    return (low|(high<<8)).to(tl.float16,bitcast=True)


@tr.jit(do_not_specialize=['N'])
def _partial_predecoded(Q,KV,IDS,PART,LSE,OUT,N,INDEX_WIDTH:tl.constexpr,
        BN:tl.constexpr,TILES:tl.constexpr,ORIGINAL_SPLITS:tl.constexpr,
        SINGLE_ACTIVE:tl.constexpr,SCALE:tl.constexpr,VDIM:tl.constexpr=256):
    row,kv_head,part=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    split=part//(256//VDIM);value_block=part%(256//VDIM)
    h=tl.arange(0,16);d=tl.arange(0,256)
    vd=value_block*VDIM+tl.arange(0,VDIM)
    query=tl.load(Q+(row*24+kv_head*12+h[:,None])*256+d[None,:],h[:,None]<12,0).to(tl.float16)
    maximum=tl.full((16,),-1.e20,tl.float32);total=tl.zeros((16,),tl.float32)
    acc=tl.zeros((16,VDIM),tl.float32)
    for tile in range(split*TILES//ORIGINAL_SPLITS,(split+1)*TILES//ORIGINAL_SPLITS):
        entry=tile*BN+tl.arange(0,BN)
        ids=tl.load(IDS+row*INDEX_WIDTH+entry,entry<INDEX_WIDTH,-1)
        valid=(ids>=0)&(ids<N)&(entry<INDEX_WIDTH)
        address=KV+(ids[:,None]*2+kv_head)*1024
        key=_half_bytes(address+d[None,:]*2,valid[:,None])
        score=tl.dot(query,tl.trans(key))*SCALE
        score=tl.where(valid[None,:],score,-1.e20)
        next_max=tl.maximum(maximum,tl.max(score,1))
        alpha=tl.exp2(maximum-next_max)
        probability=tl.where(valid[None,:],tl.exp2(score-next_max[:,None]),0.)
        # Key operands die before value loads: avoid holding both decoded
        # matrices and their FP4 conversion intermediates in each warp.
        value=_half_bytes(address+512+vd[None,:]*2,valid[:,None])
        acc=tl.dot(probability.to(tl.float16),value,acc=acc*alpha[:,None])
        total=total*alpha+tl.sum(probability,1);maximum=next_max
    normalized=acc/tl.maximum(total[:,None],1.e-20)
    if SINGLE_ACTIVE:
        tl.store(OUT+(row*24+kv_head*12+h[:,None])*256+vd[None,:],normalized,h[:,None]<12)
    else:
        head=kv_head*12+h
        tl.store(PART+((row*24+head[:,None])*ORIGINAL_SPLITS+split)*256+vd[None,:],normalized,h[:,None]<12)
        lse=tl.where(total>0,maximum+tl.log2(tl.maximum(total,1.e-20)),-float('inf'))
        if value_block==0:
            tl.store(LSE+(row*24+head)*ORIGINAL_SPLITS+split,lse,h<12)
