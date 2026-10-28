"""Two independent request histories with shared decode projections.

This is distinct from the sequential two-token speculative verification
path. Each request owns GDN, QSA, PLE and logical token history.
"""
import numpy as np
import torch

from engine.decoder import DecoderState
from engine.model import EngineState
from engine.ops.fp8 import linear
from engine.ops.gdn import GDNState
from engine.ops.qsa_attention import pack_kv
from engine.ops.kv_cache import append_kv
from engine.graph_capture import graph as capture_graph, shared_pool


class BatchEngine:
    def __init__(self,engine,graph_decode=False):
        self.engine=engine
        if engine.head_backend!='sm121':raise ValueError('Batch head requires the verified packed backend')
        if graph_decode and not engine.graph_decode:raise ValueError('Batch graphs require the verified native configuration')
        self.graph_decode=graph_decode
        self.graphs={}; self.pool=None

    def graph_stats(self):
        return dict(enabled=self.graph_decode,blocks=len(self.graphs),
                    layers=sum(len(g.indices) for g in self.graphs.values()),
                    replays={str(i):g.replays for i,g in self.graphs.items()})

    def layer(self,index,pending,states,positions=None,ids=None,contexts=None):
        layer=self.engine.layers[index]
        hidden,block,injection=pending
        ple=[s.ple for s in states]
        if layer.ple is not None:
            if block is not None:
                hidden=layer.attn_hc.combine(hidden,block,injection)
                block=injection=None
            enhancement=[]
            for row in range(2):
                value,ple[row]=layer.ple(hidden[row:row+1],np.asarray([ids[row]],dtype=np.int64),
                                        [0,1],contexts[row],ple[row])
                enhancement.append(value)
            hidden=hidden+torch.cat(enhancement)
        if block is None:
            residual,mixed,injection=layer.attn_hc.mix(hidden,preserve_rows=True)
        else:
            residual,mixed,injection=layer.attn_hc.combine_and_mix(hidden,block,injection,preserve_rows=True)
        attention=layer.attention
        if layer.layer_type=='linear_attention':
            projected,ba=attention.project(mixed,preserve_rows=True)
            core=[]; histories=[]
            for row in range(2):
                value,state=attention.run_core(projected[row:row+1],ba[row:row+1],states[row].attention,prefill=False)
                core.append(value); histories.append(state)
            attended=(attention.out_linear(torch.cat(core)) if hasattr(attention,'out_linear') else
                linear(torch.cat(core),attention.out,attention.out_scale,attention.activation_backend))
        else:
            from engine.ops.qsa_triton import sparse_attention
            q,k,v,gate=attention.project(mixed,positions)
            packed=pack_kv(k,v)
            core=[]; histories=[]
            for row in range(2):
                previous=states[row].attention
                indices,state=attention.index(mixed[row:row+1],positions[row:row+1],previous,
                                               position_start=previous.length)
                state.packed=append_kv(previous.packed,packed[row:row+1],attention.kv_backend)
                core.append(sparse_attention(q[row:row+1].contiguous(),state.packed,indices))
                histories.append(state)
            gated=torch.cat(core).flatten(1)*torch.sigmoid(gate)
            attended=linear(gated,attention.out,attention.out_scale,attention.activation_backend)
        residual,mixed,injection=layer.mlp_hc.combine_and_mix(residual,attended,injection,preserve_rows=True)
        out=layer.moe(mixed,preserve_rows=True)
        return (residual,out,injection),[DecoderState(histories[row],ple[row]) for row in range(2)]

    def decode(self,ids,states,observer=None):
        if len(ids)!=2 or len(states)!=2 or any(not s.tokens for s in states):
            raise ValueError('Batch decode requires exactly two prefilled requests')
        if any(len(s.layers)!=len(self.engine.layers) for s in states):raise ValueError('Invalid request layer state')
        ids=[int(i) for i in ids]
        starts=[len(s.tokens) for s in states]
        if any(token<0 or token>=self.engine.config['vocab_size'] for token in ids):raise ValueError('Token is outside checkpoint vocabulary')
        if any(start+1>self.engine.config['max_position_embeddings'] for start in starts):raise ValueError('Context exceeds actual checkpoint override')
        device=self.engine.embedding.device
        positions=torch.tensor(starts,device=device,dtype=torch.long)
        tokens=torch.tensor(ids,device=device,dtype=torch.long)
        hidden=self.engine.embedding[tokens].repeat(1,self.engine.config['hc_count'])
        pending=(hidden,None,None)
        preceding=self.engine.config['ngram_size']-1
        contexts=[np.asarray([([self.engine.config['eos_token_id']]*preceding+s.tokens)[-preceding:]],
                             dtype=np.int64) for s in states]
        updated=[EngineState(list(s.layers),s.tokens+[ids[row]]) for row,s in enumerate(states)]
        index=0
        while index<len(self.engine.layers):
            if self.graph_decode and index in self.engine._graph_ranges:
                end=self.engine._graph_ranges[index]
                if index not in self.graphs:
                    if self.pool is None:self.pool=shared_pool() or torch.cuda.graph_pool_handle()
                    self.graphs[index]=BatchDecodeGraph(self,list(range(index,end)),self.pool)
                captured=self.graphs[index]
                pending,next_states=captured(pending,[s.layers[index:end] for s in states])
                for row in range(2):updated[row].layers[index:end]=next_states[row]
                if observer is not None:
                    for offset,value in enumerate(captured.intermediates):observer(index+offset,value)
                index=end
            else:
                pending,next_states=self.layer(index,pending,[s.layers[index] for s in states],positions,ids,contexts)
                for row in range(2):updated[row].layers[index]=next_states[row]
                if observer is not None:observer(index,pending)
                index+=1
        multi,sampled,_=self.engine.final_mixer.combine_and_mix(*pending,preserve_rows=True)
        if observer is not None:observer('final',(multi,sampled,None))
        return self.engine.logits(sampled),updated


class BatchDecodeGraph:
    def __init__(self,batch,indices,pool):
        self.batch=batch; self.indices=indices; self.pool=pool
        self.graph=None; self.replays=0
        if any(batch.engine.layers[i].layer_type!='linear_attention' or batch.engine.layers[i].ple is not None
               for i in indices):raise ValueError('Batch graph contains a non-GDN boundary')

    def copy_inputs(self,pending,states):
        for a,b in zip(pending,self.pending):
            if (a is None)!=(b is None):raise ValueError('Batch graph pending structure changed')
            if a is not None:
                if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Batch graph tensor shape changed')
                b.copy_(a)
        # Most consecutive replays receive two views of the same owned
        # snapshot. Copy each complete cache bank once in that case.
        owner=getattr(states[0][0],'batch_bank',None)
        shared=owner is not None and all(
            getattr(states[row][offset],'batch_bank',None) is owner
            and getattr(states[row][offset],'batch_slot',None)==(offset,row)
            for row in range(2) for offset in range(len(self.indices)))
        if shared:
            self.conv.copy_(owner[0]); self.recurrent.copy_(owner[1])
        else:
            for row in range(2):
                for offset in range(len(self.indices)):
                    source=states[row][offset]
                    if source.ple is not None:raise ValueError('PLE cannot enter the GDN graph')
                    self.conv[offset,row].copy_(source.attention.conv)
                    self.recurrent[offset,row].copy_(source.attention.recurrent)

    def run(self):
        pending=self.pending; self.intermediates=[]
        for offset,index in enumerate(self.indices):
            pending,updated=self.batch.layer(index,pending,[s[offset] for s in self.states])
            for row in range(2):
                self.conv[offset,row].copy_(updated[row].attention.conv)
                self.recurrent[offset,row].copy_(updated[row].attention.recurrent)
            self.intermediates.append(pending)
        return pending

    def capture(self,pending,states):
        self.pending=tuple(x.clone() if x is not None else None for x in pending)
        self.conv=torch.stack([torch.stack([states[row][i].attention.conv for row in range(2)])
                               for i in range(len(self.indices))])
        self.recurrent=torch.stack([torch.stack([states[row][i].attention.recurrent for row in range(2)])
                                    for i in range(len(self.indices))])
        self.states=[[DecoderState(GDNState(self.conv[i,row],self.recurrent[i,row]))
                      for i in range(len(self.indices))] for row in range(2)]
        stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self.copy_inputs(pending,states); self.run()
        torch.cuda.current_stream().wait_stream(stream)
        self.copy_inputs(pending,states); torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with capture_graph(self.graph,pool=self.pool,stream=stream):self.output=self.run()
        self.copy_inputs(pending,states)

    def __call__(self,pending,states):
        if self.graph is None:self.capture(pending,states)
        else:self.copy_inputs(pending,states)
        self.graph.replay(); self.replays+=1
        bank=(self.conv.clone(),self.recurrent.clone())
        snapshots=[]
        for row in range(2):
            values=[]
            for i in range(len(self.indices)):
                state=DecoderState(GDNState(bank[0][i,row],bank[1][i,row]))
                state.batch_bank=bank; state.batch_slot=(i,row)
                values.append(state)
            snapshots.append(values)
        return self.output,snapshots
