"""Independent PLE BF16 projections, gates and dilated convolution state.

CPU hashing/disk lookup is synchronous outside CUDA Graphs. Core tensor math
can be separated from lookup for future graph scheduling. No vLLM imports.
"""
import math
import os
import numpy as np
import torch
from ..limits import per_row
from torch import nn
from torch.nn import functional as F
from .ple_lookup import NGramHasher,DiskPleTable,dequant_rows

PLE_CONV_MODE=os.environ.get('ENGINE_PLE_CONV','vector')  # 'loop' restores the per-row loop


class PLERef(nn.Module):
    def __init__(self, store, prefix, config):
        super().__init__()
        self.hidden=config['hidden_size']; self.count=config['hc_count']
        self.eps=config['rms_norm_eps']; self.dilation=config['ngram_size']
        for name,suffix in {'key':'key_proj.weight','value':'value_proj.weight',
            'norm_key':'norm_key.weight','norm_query':'norm_query.weight',
            'norm_conv':'norm_conv.weight','conv':'conv1d.weight',
            'embedding_scale':'ple_embedding.ngram_embedding.weight_scale'}.items():
            self.register_buffer(name,store.get(prefix+'.'+suffix))
        self.state_length=(self.conv.shape[-1]-1)*self.dilation
        base=prefix+'.ple_embedding'
        buffers=[store.get(base+'.'+name).cpu().numpy() for name in
            ('layer_multipliers','ngram_heads_vocab_sizes','ngram_heads_offsets')]
        self.hasher=NGramHasher(*buffers,config['eos_token_id'])
        self.table=DiskPleTable(store.model_dir,base+'.ngram_embedding')
        self._hint=None

    def hint_next_chunk(self, next_ids, next_context, expected_rows):
        """Offer the next prefill chunk so its table rows are read during this one.

        Consumed (and always cleared) by the next stateful forward whose row
        count matches; otherwise ignored. Never changes any lookup result.
        """
        self._hint=(np.asarray(next_ids,dtype=np.int64),np.asarray(next_context,dtype=np.int64),int(expected_rows))

    def norm(self,x,weight):
        shape=x.shape
        grouped=x.float().reshape(x.shape[0],self.count,self.hidden)
        grouped=grouped*torch.rsqrt(grouped.square().mean(-1,keepdim=True)+self.eps)
        affine=1+weight.float().reshape(self.count,self.hidden)
        return (grouped*affine).reshape(shape).to(x.dtype)

    def core(self,hidden,embeddings,state=None,return_prefixes=False):
        if hidden.ndim!=2 or hidden.shape[-1]!=self.hidden*self.count:
            raise ValueError('Expected single-sequence HC residual rows')
        if embeddings.dtype==torch.float8_e4m3fn:
            embeddings=embeddings.to(hidden.dtype)*self.embedding_scale.to(hidden.dtype)
        elif embeddings.dtype==torch.uint8:   # INT4 g32 / ternary table rows
            embeddings=dequant_rows(embeddings,self.table.format,float(self.embedding_scale.float())).to(hidden.dtype)
        def project(weight):
            from .row_linear import per_row_linear
            return (per_row_linear(embeddings,weight)
                    if return_prefixes and per_row(embeddings.shape[0]) else F.linear(embeddings,weight))
        key=project(self.key).reshape(-1,self.count,self.hidden)
        query=hidden.reshape(-1,self.count,self.hidden)
        key,query=self.norm(key,self.norm_key),self.norm(query,self.norm_query)
        value=project(self.value)
        # These operations intentionally preserve the reference BF16 gates.
        gate=(key*query).sum(-1,keepdim=True)/math.sqrt(self.hidden)
        gate=torch.sigmoid(gate.sign()*gate.abs().clamp_min(1e-6).sqrt())
        gated=gate*value.unsqueeze(-2)
        normalized=self.norm(gated,self.norm_conv).flatten(-2)
        if state is None:
            state=hidden.new_zeros(1,self.count*self.hidden,self.state_length)
        if state.shape!=(1,self.count*self.hidden,self.state_length):
            raise ValueError('Invalid single-sequence PLE history')
        outputs=[]; prefixes=[]
        weight=self.conv.squeeze(1).float()
        if not return_prefixes and len(normalized)>3 and PLE_CONV_MODE=='vector':
            convolved,state=self.conv_rows(normalized,state,weight,hidden.dtype)
            return gated.flatten(-2)+convolved,state
        for row in normalized:
            history=torch.cat((state,row[None,:,None]),dim=-1)
            sampled=history[:,:,::self.dilation]
            # conv1d BF16 rounds its accumulator before SiLU.
            conv=(sampled.float()*weight[None]).sum(-1).to(hidden.dtype)
            outputs.append(F.silu(conv))
            state=history[:,:,1:].contiguous()
            if return_prefixes:prefixes.append(state)
        result=(gated.flatten(-2)+torch.cat(outputs),state)
        return (*result,prefixes) if return_prefixes else result

    def conv_rows(self,normalized,state,weight,dtype):
        """All rows at once with the per-row loop's exact FP32 arithmetic.

        Each output is silu(bf16((p0+p2)+(p1+p3))) with p_j=float(tap_j)*w_j.
        That pairing is what torch.sum(-1) applies to a contiguous 4-tap FP32
        row on GB10 (measured: zero mismatches over 4.1M elements, while every
        other summation tree of four terms differs); it is stable for 4..2051
        stacked rows. Products and sums are separate elementwise ops (no FMA),
        so values equal the loop's. Needs exactly 4 taps.
        """
        taps=weight.shape[-1]
        if taps!=4:raise ValueError('Vectorized PLE convolution is verified for 4 taps only')
        rows=len(normalized)
        # [history + rows, channels]: tap j of row i is sequence[i + j*dilation].
        sequence=torch.cat((state[0].t(),normalized),dim=0)
        p=[sequence[j*self.dilation:j*self.dilation+rows].float()*weight[:,j] for j in range(taps)]
        total=(p[0]+p[2])+(p[1]+p[3])
        updated=sequence[-self.state_length:].t()[None].contiguous()
        return F.silu(total.to(dtype)),updated

    def forward(self,hidden,input_ids_cpu,query_start_loc_cpu,context_cpu,state=None,return_prefixes=False):
        # Stateful core currently supports one sequence. Hash/lookup supports
        # packed multi-request inputs separately; never silently share history.
        if len(query_start_loc_cpu)!=2:
            raise ValueError('PLE core supports one sequence at this milestone')
        ids=self.hasher(input_ids_cpu,query_start_loc_cpu,context_cpu)
        hint,self._hint=self._hint,None
        if hint is not None and hint[2]==len(ids):
            next_ids,next_context,_=hint
            self.table.submit(lambda:self.hasher(next_ids,[0,len(next_ids)],next_context[None]))
        embeddings=self.table.lookup(ids,hidden.device)
        return self.core(hidden,embeddings,state,return_prefixes)


def hint_next_prefill_chunk(engine, ids, start, end, chunk_size):
    """Tell PLE layers which prompt tokens follow chunk [start, end).

    Pure function of the prompt IDs, so it cannot alter results. Safe for
    engines without PLE (test doubles, CPU references): they are skipped.
    """
    layers=getattr(engine,'layers',None)
    if not layers or end>=len(ids):return
    for layer in layers:
        ple=getattr(layer,'ple',None)
        if ple is None or not hasattr(ple,'hint_next_chunk'):continue
        width=ple.hasher.ngram_size-1
        context=([ple.hasher.eos]*width+list(ids[:end]))[-width:] if width else []
        ple.hint_next_chunk(ids[end:end+chunk_size],context,end-start)
