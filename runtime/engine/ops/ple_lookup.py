"""CPU n-gram hashing and bounded-cache disk PLE row lookup.

Hash from tokens already held by the engine on CPU, avoiding a GPU->CPU ID
round trip. Open FP8 (or INT4 g32) table shards read-only; fetch rows, never whole tensors.

INT4 g32 rows (U8, 85 bytes): 80 bytes of two's-complement nibbles for 160 values (low nibble = even index),
then 5 E4M3 group codes. value = q * max(code * gs, 1e-30) with gs = weight_scale / 7, exactly the
fake-quantisation measured in diagnostics/ple_quant_eval_20261021.py (int4_g32_rtn).
This module is synchronous and explicitly lives outside CUDA Graph capture.
"""
from collections import OrderedDict
import json
import mmap
import os
from pathlib import Path
import queue
import re
import struct
import threading
import time
import numpy as np
import torch

INT4_ROW_BYTES, INT4_HEAD, INT4_GROUP = 85, 160, 32


def dequant_int4g32(raw, gs):
    """raw uint8 [..., n*85] -> float32 [..., n*160]; gs = table FP32 global scale (weight_scale / 7)."""
    rows = raw.reshape(*raw.shape[:-1], -1, INT4_ROW_BYTES)
    packed = rows[..., :INT4_HEAD//2]
    codes = rows[..., INT4_HEAD//2:].contiguous().view(torch.float8_e4m3fn).float()
    q = torch.stack(((packed & 15), (packed >> 4)), -1).flatten(-2).to(torch.int16)
    q = torch.where(q >= 8, q-16, q).float()
    scale = (codes*gs).clamp(min=1e-30)
    out = q.unflatten(-1, (-1, INT4_GROUP))*scale[..., None]
    return out.flatten(-2).flatten(-2)


def quantize_int4g32(x, gs):
    """x float32 [N, 160] (FP8 row * weight_scale, rounded through BF16 as PLERef.core does) -> uint8 [N, 85]."""
    g = x.unflatten(-1, (-1, INT4_GROUP))
    codes = (g.abs().amax(-1)/7.0/gs).to(torch.float8_e4m3fn)
    scale = (codes.float()*gs).clamp(min=1e-30)
    q = (g/scale[..., None]).round().clamp(-8, 7).to(torch.int16).flatten(-2)
    nib = (q & 15).to(torch.uint8)
    packed = nib[..., 0::2] | (nib[..., 1::2] << 4)
    return torch.cat((packed, codes.view(torch.uint8)), -1)


TERN_POW = (1, 3, 9, 27, 81)


def _hadamard(n, device):
    h = torch.ones(1, 1)
    while h.shape[0] < n: h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (h/np.sqrt(n)).to(device=device, dtype=torch.float32)   # symmetric orthonormal: H @ H == I


def tern_row_bytes(group):
    return INT4_HEAD//5 + INT4_HEAD//group


def tern_global_scale(weight_scale, group=32, rotate=True):
    """FP32 global so alpha/gs <= 448: |y| <= sqrt(group)*max|x| after the orthonormal rotation."""
    return float(weight_scale)*(float(np.sqrt(group)) if rotate else 1.0)


def quantize_tern(x, gs, group=32, rotate=True):
    """x float32 [N, 160] -> uint8 [N, 32 + 160/group]: per group (optionally Hadamard-rotated) the L2-optimal
    ternary code alpha*t, t in {-1,0,1}: keep the k largest |y| with k maximising (sum of top-k)^2/k, alpha = their
    mean, stored as E4M3 code of alpha/gs (gs = tern_global_scale). 5 trits per byte, base 3, lowest digit first."""
    y = x.unflatten(-1, (-1, group))
    if rotate: y = y@_hadamard(group, x.device)
    a = y.abs(); srt = a.sort(-1, descending=True).values
    csum = srt.cumsum(-1); k = torch.arange(1, group+1, device=x.device, dtype=torch.float32)
    best = (csum*csum/k).argmax(-1, keepdim=True)
    alpha = csum.gather(-1, best)/(best+1).float()
    thresh = srt.gather(-1, best)
    t = torch.where(a >= thresh, y.sign(), torch.zeros_like(y)).to(torch.int16)
    codes = (alpha[..., 0]/gs).clamp(max=448.0).to(torch.float8_e4m3fn)
    d = (t+1).flatten(-2).unflatten(-1, (-1, 5))
    packed = (d*torch.tensor(TERN_POW, device=x.device, dtype=torch.int16)).sum(-1).to(torch.uint8)
    return torch.cat((packed, codes.view(torch.uint8)), -1)


def dequant_tern(raw, gs, group=32, rotate=True):
    """raw uint8 [..., n*rowbytes] -> float32 [..., n*160]."""
    width = tern_row_bytes(group)
    rows = raw.reshape(*raw.shape[:-1], -1, width)
    packed = rows[..., :INT4_HEAD//5].to(torch.int16)
    codes = rows[..., INT4_HEAD//5:].contiguous().view(torch.float8_e4m3fn).float()
    pw = torch.tensor(TERN_POW, device=raw.device, dtype=torch.int16)
    t = ((packed[..., None]//pw) % 3 - 1).flatten(-2).float()
    y = t.unflatten(-1, (-1, group))*(codes*gs)[..., None]
    if rotate: y = y@_hadamard(group, raw.device)
    return y.flatten(-2).flatten(-2)


class NGramHasher:
    def __init__(self, multipliers, sizes, offsets, eos_token_id):
        self.multipliers=np.asarray(multipliers,dtype=np.int64)
        self.sizes,self.offsets=[np.asarray(x,dtype=np.int64) for x in (sizes,offsets)]
        self.ngram_size=len(self.multipliers)
        self.eos=int(eos_token_id)
        if self.ngram_size<2 or len(self.sizes)%(self.ngram_size-1) or self.sizes.shape!=self.offsets.shape:
            raise ValueError('Invalid checkpoint n-gram buffers')
        self.heads_per_ngram=len(self.sizes)//(self.ngram_size-1)

    def __call__(self, input_ids, query_start_loc, context):
        tokens=np.asarray(input_ids,dtype=np.int64).reshape(-1)
        starts=np.asarray(query_start_loc,dtype=np.int64)
        context=np.asarray(context,dtype=np.int64)
        if starts.ndim!=1 or len(starts)<2 or starts[0]!=0 or starts[-1]!=len(tokens) or np.any(np.diff(starts)<0):
            raise ValueError('Offsets must partition all real CPU tokens')
        if context.shape!=(len(starts)-1,self.ngram_size-1):
            raise ValueError('Invalid n-gram context')
        output=np.empty((len(tokens),len(self.sizes)),dtype=np.int64)
        for request,(begin,end) in enumerate(zip(starts[:-1],starts[1:])):
            seq=np.concatenate((context[request],tokens[begin:end]))
            positions=np.arange(len(seq),dtype=np.int64)
            eos_positions=np.where(seq==self.eos,positions,-1)
            previous=np.concatenate(([-1],np.maximum.accumulate(eos_positions)[:-1]))
            positions_in_segment=positions-previous-1
            shifted=[seq]
            for shift in range(1,self.ngram_size):
                source=positions-shift
                valid=(source>=0)&(positions_in_segment>=shift)
                shifted.append(np.where(valid,seq[source.clip(min=0)],self.eos))
            mixed=shifted[0]*self.multipliers[0]
            for ngram in range(2,self.ngram_size+1):
                mixed=np.bitwise_xor(mixed,shifted[ngram-1]*self.multipliers[ngram-1])
                lo=(ngram-2)*self.heads_per_ngram
                hi=lo+self.heads_per_ngram
                output[begin:end,lo:hi]=(mixed[self.ngram_size-1:,None]%self.sizes[lo:hi])+self.offsets[lo:hi]
        return output


class DiskPleTable:
    def __init__(self, model_dir, prefix, cache_rows=65536):
        root=Path(model_dir)
        index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
        pattern=re.compile(re.escape(prefix)+r'\.shard_(\d+)\.weight$')
        headers,shards={},{}
        for name,filename in index.items():
            match=pattern.fullmatch(name)
            if not match:continue
            if filename not in headers:
                with (root/filename).open('rb') as stream:
                    length=struct.unpack('<Q',stream.read(8))[0]
                    headers[filename]=(json.loads(stream.read(length)),8+length)
            header,start=headers[filename]
            item=header[name]
            meta=header.get('__metadata__') or {}
            fmt=('fp8' if item['dtype']=='F8_E4M3' else meta.get('ple_format','int4g32') if item['dtype']=='U8' else None)
            if len(item['shape'])!=2 or fmt not in PLE_FORMATS or PLE_FORMATS[fmt] not in (None,item['shape'][1]):
                raise ValueError('Unsupported PLE table format: '+str((item['dtype'],item['shape'],fmt)))
            if getattr(self,'format',fmt)!=fmt:raise ValueError('Mixed PLE shard formats')
            self.format=fmt
            rows,width=item['shape']
            if item['data_offsets'][1]-item['data_offsets'][0]!=rows*width:
                raise ValueError('Invalid shard byte count')
            shards[int(match.group(1))]=(root/filename,start+item['data_offsets'][0],rows,width)
        if not shards or sorted(shards)!=list(range(len(shards))):
            raise ValueError('Missing or noncontiguous PLE shards')
        self.shard_size=max(s[2] for s in shards.values())
        self.width=shards[0][3]
        self.maps=[]
        self.paths,self.offsets=[],[]
        for i,(path,offset,rows,width) in sorted(shards.items()):
            if width!=self.width or (i<len(shards)-1 and rows!=self.shard_size):
                raise ValueError('Invalid shard layout')
            self.maps.append(np.memmap(path,mode='r',dtype=np.uint8,offset=offset,shape=(rows,width)))
            self.paths.append(str(path)); self.offsets.append(offset)
        # Random 160-byte rows: kernel read-around would pull ~100 KiB per
        # fault. Page-cache hints change only when bytes arrive, never values.
        self.io_enabled=os.environ.get('ENGINE_PLE_IO','prefetch')!='legacy' and hasattr(os,'posix_fadvise') and hasattr(os,'POSIX_FADV_WILLNEED')
        self.page=os.sysconf('SC_PAGE_SIZE')
        self._fds=[None]*len(self.maps)
        self._jobs=None; self._worker=None; self._worker_lock=threading.Lock()
        self.io_stats=dict(willneed_calls=0,willneed_rows=0,async_jobs=0,async_dropped=0,async_errors=0)
        if self.io_enabled and hasattr(mmap,'MADV_RANDOM'):
            for table in self.maps:
                try:table._mmap.madvise(mmap.MADV_RANDOM)
                except (AttributeError,OSError,ValueError):pass
        self.rows_total=sum(len(m) for m in self.maps)
        # ENGINE_PLE_RESIDENT=gpu|cpu: copy the whole table into device memory once and gather there (no page
        # cache, no per-row CPU copy). Only sensible for compressed tables. Byte-identical rows to the disk path.
        self.resident=None
        where=os.environ.get('ENGINE_PLE_RESIDENT','')
        if where:
            dev='cuda' if where=='gpu' else where
            table=torch.empty((self.rows_total,self.width),dtype=torch.uint8,device=dev)
            for i,m in enumerate(self.maps):
                begin=i*self.shard_size
                table[begin:begin+len(m)].copy_(torch.from_numpy(np.array(m)))
                try:os.posix_fadvise(self._fd(i),0,0,os.POSIX_FADV_DONTNEED)
                except (AttributeError,OSError):pass
            self.resident=table; self.io_enabled=False
        self.cache_rows=max(0,int(cache_rows))
        self.cache=OrderedDict()
        self.last_stats={}

    def _fd(self, shard):
        if self._fds[shard] is None:
            self._fds[shard]=os.open(self.paths[shard],os.O_RDONLY|getattr(os,'O_CLOEXEC',0))
        return self._fds[shard]

    def willneed(self, rows):
        """Queue asynchronous page reads for exact rows (no wait, no data copy)."""
        if not self.io_enabled:return 0
        rows=np.unique(np.asarray(rows,dtype=np.int64).reshape(-1))
        rows=rows[(rows>=0)&(rows<self.rows_total)]
        if not rows.size:return 0
        shard=rows//self.shard_size; calls=0
        for sid in np.unique(shard):
            local=rows[shard==sid]-sid*self.shard_size
            begin=self.offsets[sid]+local*self.width
            pages=np.unique(np.concatenate((begin//self.page,(begin+self.width-1)//self.page)))
            breaks=np.flatnonzero(np.diff(pages)>1)
            firsts=np.concatenate(([0],breaks+1)); lasts=np.concatenate((breaks,[len(pages)-1]))
            fd=self._fd(int(sid))
            for first,last in zip(firsts.tolist(),lasts.tolist()):
                os.posix_fadvise(fd,int(pages[first])*self.page,
                    int(pages[last]-pages[first]+1)*self.page,os.POSIX_FADV_WILLNEED)
                calls+=1
        self.io_stats['willneed_calls']+=calls; self.io_stats['willneed_rows']+=int(rows.size)
        return calls

    def submit(self, produce_rows):
        """Best-effort background prefetch of rows returned by produce_rows()."""
        if not self.io_enabled:return False
        with self._worker_lock:
            if self._worker is None:
                self._jobs=queue.Queue(maxsize=2)
                self._worker=threading.Thread(target=self._run,name='ple-prefetch',daemon=True)
                self._worker.start()
        try:self._jobs.put_nowait(produce_rows)
        except queue.Full:
            self.io_stats['async_dropped']+=1
            return False
        self.io_stats['async_jobs']+=1
        return True

    def _run(self):
        while True:
            produce=self._jobs.get()
            try:self.willneed(produce())
            except Exception:self.io_stats['async_errors']+=1

    def gather(self, ids):
        begin=time.perf_counter()
        ids=np.asarray(ids,dtype=np.int64)
        unique,inverse=np.unique(ids.reshape(-1),return_inverse=True)
        if unique.size and (unique[0]<0 or unique[-1]>=self.rows_total):
            raise IndexError('PLE row outside checkpoint table')
        rows=np.empty((len(unique),self.width),dtype=np.uint8)
        missing=[]
        for i,row in enumerate(unique.tolist()):
            if row in self.cache:
                rows[i]=np.frombuffer(self.cache[row],dtype=np.uint8)
                self.cache.move_to_end(row)
            else:missing.append(i)
        missing=np.asarray(missing,dtype=np.int64)
        if len(missing):
            self.willneed(unique[missing])
            shard=unique[missing]//self.shard_size
            for sid in np.unique(shard):
                selected=missing[shard==sid]
                rows[selected]=self.maps[sid][unique[selected]-sid*self.shard_size]
            if self.cache_rows:
                # A long prefill may visit millions of rows; bound insertions
                # and retained payload rather than building a second table.
                for i in missing[-self.cache_rows:]:
                    self.cache[int(unique[i])]=rows[i].tobytes()
                    if len(self.cache)>self.cache_rows:self.cache.popitem(last=False)
        self.last_stats={'requested_rows':int(ids.size),'unique_rows':len(unique),
            'hit_rows':len(unique)-len(missing),'disk_rows':len(missing),
            'cache_payload_bytes':len(self.cache)*self.width,
            'gather_ms':(time.perf_counter()-begin)*1000}
        return rows[inverse].reshape(*ids.shape,self.width)

    def lookup(self, ids, device='cpu'):
        if self.resident is not None:
            begin=time.perf_counter()
            ids=np.asarray(ids,dtype=np.int64)
            if ids.size and (ids.min()<0 or ids.max()>=self.rows_total):raise IndexError('PLE row outside checkpoint table')
            index=torch.from_numpy(ids.reshape(-1)).to(self.resident.device)
            raw=self.resident.index_select(0,index).reshape(*ids.shape,self.width).to(device)
            self.last_stats={'requested_rows':int(ids.size),'resident':True,'gather_ms':(time.perf_counter()-begin)*1000}
        else:
            rows=self.gather(ids)
            raw=torch.from_numpy(rows.copy()).to(device=device)
        if self.format!='fp8':return raw.flatten(-2)   # uint8 rows; PLERef.core dequantises
        return raw.view(torch.float8_e4m3fn).flatten(-2)


PLE_FORMATS = {'fp8': None, 'int4g32': INT4_ROW_BYTES,   # row bytes; FP8 width is free (as before)
               'tern_g32_rot': tern_row_bytes(32), 'tern_g32_none': tern_row_bytes(32),
               'tern_g16_rot': tern_row_bytes(16), 'tern_g16_none': tern_row_bytes(16)}


def dequant_rows(raw, fmt, weight_scale):
    """Any U8 PLE format -> float32 [..., n*160]; weight_scale is the table's float scale."""
    if fmt == 'int4g32': return dequant_int4g32(raw, float(weight_scale)/7.0)
    _, g, how = fmt.split('_'); g = int(g[1:]); rot = how == 'rot'
    return dequant_tern(raw, tern_global_scale(weight_scale, g, rot), g, rot)
