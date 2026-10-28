"""Append KV page pointers without synchronous CPU-to-GPU tensor copies."""
import torch
import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=['P0','P1','KEEP','N'])
def _append_table(OLD,OUT,P0,P1,KEEP,N,B:tl.constexpr):
    index=tl.program_id(0)*B+tl.arange(0,B)
    previous=tl.load(OLD+index,index<KEEP,0)
    pointer=tl.where(index==KEEP,P0,previous)
    pointer=tl.where(index==KEEP+1,P1,pointer)
    tl.store(OUT+index,pointer,index<N)


def append_table(previous,keep,pages):
    if len(pages) not in (1,2):raise ValueError('Decode table appends one or two pages')
    count=keep+len(pages)
    out=torch.empty(count,device=previous.device,dtype=torch.int64)
    _append_table[(tr.cdiv(count,256),)](previous,out,pages[0].data_ptr(),
        pages[1].data_ptr() if len(pages)==2 else 0,keep,count,256,num_warps=4)
    return out
