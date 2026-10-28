"""Immutable paged NVFP4 histories with copy-on-write open pages.

Closed pages are shared by prefix/rollback states. Appending never writes an
existing page, including a partly occupied one. Only the open page and the
small pointer table are copied; ordinary decode does not copy the full KV.
Pages are normal Torch allocations, so their owners keep GPU storage alive.
"""
import torch


class PagedKV:
    def __init__(self,pages,table,length,block_size=256):
        self._pages=tuple(pages)
        self.block_table=table
        self.length=length
        self.block_size=block_size
        if len(self._pages)!=(length+block_size-1)//block_size:
            raise ValueError('Page count and logical length differ')
        if not self._pages:raise ValueError('Paged KV requires a nonempty history')

    @property
    def shape(self):return (self.length,2,288)

    @property
    def dtype(self):return self._pages[0].dtype

    @property
    def device(self):return self._pages[0].device

    def is_contiguous(self):return False

    @staticmethod
    def _validate(rows,block_size):
        if rows.ndim!=3 or rows.shape[1:]!=(2,288) or rows.dtype!=torch.uint8 or not rows.is_contiguous():
            raise ValueError('Expected contiguous checkpoint NVFP4 rows [N,2,288]')
        if not len(rows):raise ValueError('Cannot append an empty KV block')
        if block_size<16 or block_size>1024 or block_size&(block_size-1):
            raise ValueError('Page size must be a power of two from 16 to 1024')

    @classmethod
    def from_dense(cls,rows,block_size=256):
        cls._validate(rows,block_size)
        pages=[]
        for start in range(0,len(rows),block_size):
            view=rows[start:start+block_size]
            if len(view)==block_size:
                # A full freshly packed page can stay a view of its producer.
                pages.append(view)
            else:
                page=rows.new_empty(block_size,2,288)
                page[:len(view)].copy_(view)
                pages.append(page)
        table=torch.tensor([p.data_ptr() for p in pages],device=rows.device,dtype=torch.int64)
        return cls(pages,table,len(rows),block_size)

    def append(self,rows):
        self._validate(rows,self.block_size)
        if rows.device!=self.device:raise ValueError('KV devices differ')
        block=self.block_size; keep=self.length//block; offset=self.length%block
        pages=list(self._pages[:keep]); added=[]; consumed=0
        if offset:
            page=rows.new_empty(block,2,288)
            page[:offset].copy_(self._pages[keep][:offset])
            take=min(block-offset,len(rows))
            page[offset:offset+take].copy_(rows[:take])
            added.append(page); consumed=take
        for start in range(consumed,len(rows),block):
            view=rows[start:start+block]
            if len(view)==block:added.append(view)
            else:
                page=rows.new_empty(block,2,288)
                page[:len(view)].copy_(view)
                added.append(page)
        if rows.is_cuda and len(added)<=2:
            from .kv_triton import append_table
            # Scalar kernel arguments go straight into the launch. Creating
            # torch.tensor(pointer_list, device='cuda') waits for the stream
            # on every QSA layer and breaks overlap with preceding graph work.
            table=append_table(self.block_table,keep,added)
        else:
            pointers=torch.tensor([p.data_ptr() for p in added],device=rows.device,dtype=torch.int64)
            table=torch.cat((self.block_table[:keep],pointers)) if keep else pointers
        return type(self)(pages+added,table,self.length+len(rows),block)

    def __getitem__(self,key):
        if not isinstance(key,slice) or key.start not in (None,0) or key.step not in (None,1):
            raise ValueError('Paged KV supports logical prefixes only')
        end=self.length if key.stop is None else key.stop
        if not isinstance(end,int) or not 0<end<=self.length:
            raise ValueError('Invalid nonempty KV prefix')
        count=(end+self.block_size-1)//self.block_size
        return type(self)(self._pages[:count],self.block_table[:count],end,self.block_size)

    def to_dense(self):
        # Used by module oracles/debugging, never by the paged attention core.
        return torch.cat([p[:min(self.block_size,self.length-i*self.block_size)]
                          for i,p in enumerate(self._pages)])


def materialize_kv(history):
    return history.to_dense() if isinstance(history,PagedKV) else history


def append_kv(history,rows,backend='contiguous',prefill=False):
    if backend not in ('contiguous','paged','auto'):raise ValueError('Unknown KV storage backend')
    if isinstance(history,PagedKV):return history.append(rows)
    # Wide prefill rows coalesce better in dense storage. Auto promotes only
    # on the first long decode, retaining existing full pages as zero-copy
    # views of that immutable prefill buffer.
    if backend=='auto' and prefill:backend='contiguous'
    if backend=='paged' or backend=='auto' and history is not None and len(history)+len(rows)>8192:
        if history is None:return PagedKV.from_dense(rows)
        return PagedKV.from_dense(history).append(rows)
    return rows if history is None else torch.cat((history,rows))
