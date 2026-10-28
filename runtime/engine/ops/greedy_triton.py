"""SM121 vocabulary reduction, exact first-index tie handling."""
import triton as tr
import triton.language as tl


@tr.jit
def _partial(X,MAXIMUM,INDEX,BAD,V:tl.constexpr,TILES:tl.constexpr,BLOCK:tl.constexpr):
    tile=tl.program_id(0);row=tl.program_id(1)
    offsets=tile*BLOCK+tl.arange(0,BLOCK);active=offsets<V
    values=tl.load(X+row*V+offsets,active,other=-float('inf')).to(tl.float32)
    finite=(values==values)&(tl.abs(values)<float('inf'))
    maximum=tl.max(tl.where(active&finite,values,-float('inf')),0)
    index=tl.min(tl.where(active&finite&(values==maximum),offsets,V),0)
    bad=tl.max((active&~finite).to(tl.int32),0)
    tl.store(MAXIMUM+row*TILES+tile,maximum)
    tl.store(INDEX+row*TILES+tile,index)
    tl.store(BAD+row*TILES+tile,bad)


@tr.jit
def _merge(MAXIMUM,INDEX,BAD,OUTPUT,V:tl.constexpr,TILES:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);tiles=tl.arange(0,BLOCK);active=tiles<TILES
    values=tl.load(MAXIMUM+row*TILES+tiles,active,other=-float('inf'))
    indices=tl.load(INDEX+row*TILES+tiles,active,other=V)
    bad=tl.load(BAD+row*TILES+tiles,active,other=0)
    maximum=tl.max(values,0)
    first=tl.min(tl.where(active&(values==maximum),indices,V),0)
    tl.store(OUTPUT+row*2,first)
    tl.store(OUTPUT+row*2+1,tl.max(bad,0))


def select(logits,workspace):
    maximum,index,bad,output=workspace
    rows,vocab=logits.shape;tiles=maximum.shape[1]
    _partial[(tiles,rows)](logits,maximum,index,bad,vocab,tiles,1024,num_warps=4)
    _merge[(rows,)](maximum,index,bad,output,vocab,tiles,tr.next_power_of_2(tiles),num_warps=4)
    return output
