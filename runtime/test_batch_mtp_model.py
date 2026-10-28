"""CPU tensor oracle: shared projections, independent GDN/QSA/PLE histories.

Uses real HC, QSA indexing and decoder composition with small CPU weights.
GDN and packed quantization use explicit fixtures; no native GPU claim.
"""
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from torch import nn
from torch.nn import functional as F

import engine.batch_mtp_model as batch_module
import engine.ops.qsa_attention as qsa_module
from engine.batch_mtp_model import BatchMTPModel, BatchMTPDraft, head_rows
from engine.decoder import DecoderLayerRef, DecoderState
from engine.model import GreedyEngineRef, EngineState
from engine.mtp import MTPDraft
from engine.ops.gdn import GDNState
from engine.ops.hyperconnection import GatedResidualRef
from engine.ops.qsa_attention import QSARef


def weight(rows, cols):
    return (torch.randint(-2,3,(rows,cols)).float()/64).bfloat16()


def hc(combine=True):
    return GatedResidualRef(torch.zeros(256,dtype=torch.bfloat16),weight(18 if combine else 16,256),
                            weight(256,16),hc_count=2,use_combine=combine)


class TinyGDN(nn.Module):
    def __init__(self):
        super().__init__(); self.qkvz=weight(128,128);self.ba=weight(8,128);self.out=weight(128,128)
        self.qkvz_scale=self.out_scale=None;self.activation_backend='fixture'
    def project(self,x,preserve_rows=False):
        return F.linear(x,self.qkvz),torch.cat([F.linear(row,self.ba) for row in x.split(1)])
    def run_core(self,qkvz,ba,state=None,prefill=False,return_prefixes=False):
        if state is None:state=GDNState(torch.zeros(1,128,3,dtype=qkvz.dtype),torch.zeros(1,128,dtype=qkvz.dtype))
        out=[];prefixes=[]
        for row, gate in zip(qkvz,ba):
            recurrent=(state.recurrent.float()*.5+row.float()[None]+gate[0].float()/16).to(qkvz.dtype)
            conv=torch.cat((state.conv[:,:,1:],row.reshape(1,128,1)),2)
            state=GDNState(conv,recurrent);prefixes.append(state);out.append(recurrent)
        result=torch.cat(out),state
        return (*result,prefixes) if return_prefixes else result
    def forward(self,x,state=None,prefill=False,return_prefixes=False):
        projected,ba=self.project(x,True);result=self.run_core(projected,ba,state,prefill,return_prefixes)
        out=F.linear(result[0],self.out),result[1]
        return (*out,result[2]) if return_prefixes else out


class TinyPLE(nn.Module):
    def forward(self,hidden,ids,boundaries,context,state=None,return_prefixes=False):
        assert boundaries==[0,len(ids)]
        old=torch.zeros(1,1,dtype=hidden.dtype) if state is None else state
        values=[];prefixes=[];previous=int(context[0,-1])
        for token in ids:
            old=(old.float()*.5+(int(token)+previous)/1024).to(hidden.dtype)
            values.append(old.expand(1,256));prefixes.append(old.clone());previous=int(token)
        result=torch.cat(values),old.clone()
        return (*result,prefixes) if return_prefixes else result


class TinyMoE(nn.Module):
    def __init__(self):
        super().__init__();self.up=weight(128,128);self.down=weight(128,128);self.rows=[]
    def forward(self,x,preserve_rows=False):
        self.rows.append(len(x));return F.linear(F.silu(F.linear(x,self.up).float()).to(x.dtype),self.down)


def qsa(config):
    values={
        'q_proj.weight':weight(32,128),'k_proj.weight':weight(8,128),'v_proj.weight':weight(8,128),
        'o_proj.weight':weight(128,16),'q_norm.weight':torch.zeros(8,dtype=torch.bfloat16),
        'k_norm.weight':torch.zeros(8,dtype=torch.bfloat16),'indexer.index_qk_proj.weight':weight(24,128),
        'indexer.q_layernorm.weight':torch.zeros(8,dtype=torch.bfloat16),
        'indexer.k_layernorm.weight':torch.zeros(8,dtype=torch.bfloat16)}
    return QSARef(SimpleNamespace(
        weight_map={'.'+key:None for key in values},get=lambda key:values[key[1:]]),'',config)


def config():
    return dict(vocab_size=257,hidden_size=128,hc_count=2,rms_norm_eps=1e-6,max_position_embeddings=300,
        ngram_size=3,eos_token_id=256,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        rope_parameters=dict(rope_theta=10000),indexer_n_heads=2,indexer_head_dim=8,
        indexer_compress_ratio=4,indexer_budget=8)


class TinyEngine(GreedyEngineRef):
    def __init__(self):
        nn.Module.__init__(self);self.config=config();self.embedding=weight(257,128)
        self.graph_decode=self.graph_verify=False;self._graph_ranges={};self.head=weight(257,128);self.head_rows=[]
        self.layers=nn.ModuleList()
        for i in range(3):
            layer=DecoderLayerRef.__new__(DecoderLayerRef);nn.Module.__init__(layer)
            layer.attn_hc=hc();layer.mlp_hc=hc();layer.moe=TinyMoE()
            layer.layer_type='full_attention' if i==1 else 'linear_attention'
            layer.attention=qsa(self.config) if i==1 else TinyGDN();layer.ple=TinyPLE() if i==2 else None
            self.layers.append(layer)
        self.final_mixer=hc(False)
    def logits(self,x):
        assert len(x)<=3
        self.head_rows.append(len(x));return F.linear(x,self.head).float()


class TinyDraftMoE(nn.Module):
    def __init__(self):
        super().__init__();self.router=weight(7,128);self.bank=weight(128,128);self.rows=[]
    def route(self,x):
        logits=F.linear(x,self.router);weights,ids=logits.float().softmax(-1).topk(2,-1)
        return weights,ids,logits
    def routed(self,x,weights,ids):
        self.rows.append(len(x));gain=(weights*ids).sum(-1,keepdim=True)
        return (F.linear(x,self.bank).float()*gain).to(x.dtype)
    def shared(self,x):return x/8
    def forward(self,x):
        weights,ids,_=self.route(x);return self.routed(x,weights,ids)+self.shared(x)


def draft(engine):
    d=MTPDraft.__new__(MTPDraft);nn.Module.__init__(d);d.config=engine.config;d.embedding=engine.embedding
    d.fc_embedding=weight(128,128);d.fc_hidden=weight(128,128)
    d.pre_fc_norm_embedding=torch.zeros(128,dtype=torch.bfloat16)
    d.pre_fc_norm_hidden=torch.zeros(256,dtype=torch.bfloat16)
    d.attn_hc=hc();d.mlp_hc=hc();d.final_mixer=hc(False);d.attention=qsa(d.config)
    d.moe=TinyDraftMoE();d.head=engine.logits
    return d


@pytest.fixture
def cpu_attention():
    # Store real BF16 K/V rather than NVFP4 for this independent CPU oracle.
    def pack(k,v):return torch.cat((k,v),-1)
    def unpack(packed,dim):return packed[...,:dim],packed[...,dim:]
    with patch.object(batch_module,'pack_kv',pack),patch.object(qsa_module,'pack_kv',pack),\
         patch.object(qsa_module,'unpack_kv',unpack),\
         patch.object(batch_module,'linear',lambda x,w,s,b:F.linear(x,w)):
        yield


@pytest.mark.parametrize('lengths', [[1,1],[2,2],[3,3],[2,2,2],[3,3,3,3],[1,2,3,2]])
def test_real_decoder_batch_rows_equal_independent_sequence_prefixes(cpu_attention,lengths):
    torch.manual_seed(37);engine=TinyEngine();batch=BatchMTPModel(engine,graph=False)
    prompts=[[3+i,5,8]+[13+i]*i for i in range(len(lengths))]
    states=[engine(prompt,return_hidden=True,prefill_chunk=True)[1] for prompt in prompts]
    old=[s.tokens[:] for s in states]
    ids=[[31+i+j for j in range(n)] for i,n in enumerate(lengths)]
    expected=[engine.verify_tokens(tokens,state) for tokens,state in zip(ids,states)]
    for layer in engine.layers:layer.moe.rows.clear()
    actual=batch.verify(ids,states)
    assert all(layer.moe.rows==[sum(lengths)] for layer in engine.layers)
    assert [s.tokens for s in states]==old
    for (logits,state,hidden),(gold,reference,gold_hidden) in zip(actual,expected):
        torch.testing.assert_close(logits,gold,rtol=0,atol=0)
        torch.testing.assert_close(hidden.multi,gold_hidden.multi,rtol=0,atol=0)
        assert state.tokens==reference.tokens
        for prefix,ref in zip(state.prefixes,reference.prefixes):
            assert prefix.tokens==ref.tokens
            for a,b in zip(prefix.layers,ref.layers):
                if isinstance(a.attention,GDNState):
                    torch.testing.assert_close(a.attention.conv,b.attention.conv,rtol=0,atol=0)
                    torch.testing.assert_close(a.attention.recurrent,b.attention.recurrent,rtol=0,atol=0)
                else:
                    for field in ['packed','raw_tail','compressed','tail_positions']:
                        torch.testing.assert_close(getattr(a.attention,field),getattr(b.attention,field),rtol=0,atol=0)
                if a.ple is not None:torch.testing.assert_close(a.ple,b.ple,rtol=0,atol=0)
    # A later step with different lanes/positions cannot overwrite old prefixes.
    before=[s.prefixes[0].layers[0].attention.recurrent.clone() for _,s,_ in actual]
    next_states=[s.prefixes[-1] for _,s,_ in actual]
    batch.verify([[47] for _ in lengths],next_states)
    for (_,s,_),gold in zip(actual,before):torch.testing.assert_close(s.prefixes[0].layers[0].attention.recurrent,gold)
    assert max(engine.head_rows)<=3


@pytest.mark.parametrize('lengths', [[1,1],[2,3],[3,3,3,3],[1,2,3,1]])
def test_draft_batch_shares_experts_but_preserves_each_qsa_history(cpu_attention,lengths):
    torch.manual_seed(91);engine=TinyEngine();d=draft(engine);batch=BatchMTPDraft(d)
    states=[];multi=[];ids=[];expected=[]
    for lane,n in enumerate(lengths):
        prompt=[3+lane,5,8];_,_,h=engine(prompt,return_hidden=True,prefill_chunk=True)
        initial=d(prompt[1:]+[11],h.multi,prefill_chunk=True)
        values=torch.cat([h.multi[-1:] for _ in range(n)])
        tokens=[22+lane+i for i in range(n)]
        states.append(initial.state);multi.append(values);ids.append(tokens)
        out=d(tokens,values,initial.state)
        expected.append(out)
    d.moe.rows.clear();actual=batch.forward(ids,multi,states)
    assert d.moe.rows==[sum(lengths)] and batch.calls==1
    for result,gold,original,tokens in zip(actual,expected,states,ids):
        assert result.state.length==original.length+len(tokens)
        torch.testing.assert_close(result.logits,gold.logits,rtol=0,atol=0)
        torch.testing.assert_close(result.multi,gold.multi,rtol=0,atol=0)
        for field in ['packed','raw_tail','compressed','tail_positions']:
            torch.testing.assert_close(getattr(result.state,field),getattr(gold.state,field),rtol=0,atol=0)


def test_checkpoint_context_overflow_fails_before_projection(cpu_attention):
    engine=TinyEngine();batch=BatchMTPModel(engine,graph=False)
    states=[EngineState([None]*3,[1]*300) for _ in range(2)]
    with pytest.raises(ValueError,match='Context'):batch.verify([[1],[2]],states)
    assert batch.calls==0


def test_head_twelve_rows_use_unchanged_three_row_abi():
    widths=[]
    def head(x):widths.append(len(x));return x*2
    values=torch.arange(60).reshape(12,5)
    assert torch.equal(head_rows(head,values),values*2)
    assert widths==[3,3,3,3]


def test_graph_replay_prefixes_own_storage_across_lane_permutation():
    graph=batch_module.BatchMTPGraph.__new__(batch_module.BatchMTPGraph)
    graph.indices=[0];graph.replays=0;graph.pending=(torch.zeros(8,4),None,None)
    graph.conv=torch.zeros(1,4,1,2,3);graph.recurrent=torch.zeros(1,4,1,2)
    graph.updated=[[DecoderState(prefixes=[DecoderState(GDNState(torch.zeros(1,2,3),torch.zeros(1,2)))
                                            for _ in range(2)]) for lane in range(4)]]
    graph.output=graph.pending
    class Replay:
        def replay(self):
            for lane in range(4):
                for row,prefix in enumerate(graph.updated[0][lane].prefixes):
                    prefix.attention.conv.copy_(graph.conv[0,lane]+row+1)
                    prefix.attention.recurrent.copy_(graph.recurrent[0,lane]+row+1)
    graph.graph=Replay()
    states=[[DecoderState(GDNState(torch.full((1,2,3),float(lane)),torch.full((1,2),float(lane))))] for lane in range(4)]
    _,first=graph(graph.pending,states)
    values=[lane[0].prefixes[0].attention.recurrent.clone() for lane in first]
    _,second=graph(graph.pending,list(reversed(states)))
    for lane,gold in zip(first,values):torch.testing.assert_close(lane[0].prefixes[0].attention.recurrent,gold)
    assert [s[0].prefixes[0].attention.recurrent[0,0].item() for s in second]==[4,3,2,1]
    stores=[s[0].prefixes[0].attention.recurrent.untyped_storage().data_ptr() for s in first]
    assert len(set(stores))==4 and all(p!=graph.recurrent.untyped_storage().data_ptr() for p in stores)


def test_graph_layout_cache_is_bounded_without_invalidating_captured_pointers(cpu_attention):
    torch.manual_seed(16);engine=TinyEngine();engine.graph_decode=True;engine._graph_ranges={0:1}
    class CpuGraph:
        def __init__(self,batch,indices,sections,pool):
            self.batch=batch;self.indices=indices;self.sections=sections;self.replays=0
        def __call__(self,pending,states):
            self.replays+=1
            result,updated=self.batch.layer(0,pending,[s[0] for s in states],self.sections)
            return result,[[s] for s in updated]
    states=[engine([3+i,5,8],prefill_chunk=True)[1] for i in range(4)]
    with patch.object(batch_module,'BatchMTPGraph',CpuGraph),patch('torch.cuda.graph_pool_handle',side_effect=object):
        batch=BatchMTPModel(engine,max_graph_layouts=2)
        first=batch.verify([[11,12]]*4,states)
        before=first[0][1].prefixes[0].layers[0].attention.recurrent.clone()
        batch.verify([[13]]*2,states[:2]);batch.verify([[14,15,16]]*3,states[:3])
        assert len(batch.layouts)==2 and len(batch.graphs)==2 and len(batch.pools)==2
        assert (2,2,2,2) in batch.layouts and (3,3,3) not in batch.layouts
        assert batch.eager_layout_fallbacks==1
        torch.testing.assert_close(first[0][1].prefixes[0].layers[0].attention.recurrent,before)
