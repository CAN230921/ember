"""Unique-node small-tree verifier with exact parent-state semantics.

Large projections, HC and MoE process the unique node rows once. Stateful
GDN/QSA/PLE work follows parent links with the existing arithmetic. This first
backend is eager; no FP32 closed-form GDN substitution or tree CUDA Graph is
claimed. Each candidate's private states are retained until path selection.
"""
import numpy as np
import os
import torch
from torch.nn import functional as F

from .candidate_tree import CandidateTree
from .decoder import DecoderState
from .model import EngineState, ForwardHidden
from .ops.kv_cache import append_kv
from .ops.qsa_attention import pack_kv
from .tree_verification import TreeVerification


def ple_nodes(layer,hidden,tree,base_tokens,initial):
    ple=layer.ple;width=ple.hasher.ngram_size-1
    # Hashing reads a bounded n-gram context. Parent convolution state still
    # retains its full causal history; the token-list prefix is unnecessary.
    base_tail=list(base_tokens[-width:]) if width else []
    contexts=[]
    for node in range(len(tree.tokens)):
        preceding=base_tail+[tree.tokens[n] for n in tree.path(node)[:-1]]
        contexts.append(([ple.hasher.eos]*width+preceding)[-width:])
    ids=ple.hasher(np.asarray(tree.tokens,dtype=np.int64),np.arange(len(tree.tokens)+1),
        np.asarray(contexts,dtype=np.int64))
    embeddings=ple.table.lookup(ids,hidden.device)
    values=[];states=[]
    for node,parent in enumerate(tree.parents):
        previous=initial if parent==-1 else states[parent]
        value,state=ple.core(hidden[node:node+1],embeddings[node:node+1],previous)
        values.append(value);states.append(state)
    return torch.cat(values),states


def attention_nodes(layer,mixed,positions,tree,initial,start,tree_parents=None):
    attention=layer.attention;states=[];cores=[]
    if layer.layer_type=='linear_attention':
        qkvz,ba=attention.project(mixed,preserve_rows=True)
        if tree_parents is not None:
            from .ops.gdn_tree import run_tree
            core,states=run_tree(attention,qkvz,ba,initial,tree,tree_parents)
            return attention.out_linear(core),states
        for node,parent in enumerate(tree.parents):
            previous=initial if parent==-1 else states[parent]
            core,state=attention.run_core(qkvz[node:node+1],ba[node:node+1],previous,False)
            cores.append(core);states.append(state)
        return attention.out_linear(torch.cat(cores)),states
    q,k,v,gate=attention.project(mixed,positions)
    packed=pack_kv(k,v)
    for node,parent in enumerate(tree.parents):
        previous=initial if parent==-1 else states[parent]
        position=start+len(tree.path(node))-1
        indices,state=attention.index(mixed[node:node+1],positions[node:node+1],previous,
            position_start=position,return_prefixes=False)
        state.packed=append_kv(previous.packed,packed[node:node+1],attention.kv_backend,False)
        if attention.attention_backend=='triton':
            from .ops.qsa_triton import sparse_attention
            core=sparse_attention(q[node:node+1].contiguous(),state.packed,indices,
                prefill_mode=getattr(attention,'prefill_mode','packed'),verify=True)
        else:
            from .ops.qsa_attention import sparse_attention
            core=sparse_attention(q[node:node+1],state.packed,indices)
        cores.append(core);states.append(state)
    gated=torch.cat(cores).flatten(1)*torch.sigmoid(gate)
    from .ops.fp8 import linear
    output=(F.linear(gated,attention.out) if attention.out_scale is None else
        linear(gated,attention.out,attention.out_scale,attention.activation_backend))
    return output,states


@torch.no_grad()
def verify_tree(engine,tree,state,observer=None):
    if not isinstance(tree,CandidateTree):raise TypeError('Expected CandidateTree')
    if len(tree.paths())>4:raise ValueError('Prototype supports at most four leaves')
    if (not state.tokens or len(state.layers)!=len(engine.layers) or
            any(t>=engine.config['vocab_size'] for t in tree.tokens)):
        raise ValueError('Invalid checkpoint tree or base state')
    start=len(state.tokens)
    if max(tree.positions(start))>=engine.config['max_position_embeddings']:
        raise ValueError('Tree depth exceeds context limit')
    engine.discard_verification()
    device=engine.embedding.device
    tokens=torch.tensor(tree.tokens,device=device,dtype=torch.long)
    positions=torch.tensor(tree.positions(start),device=device,dtype=torch.long)
    parents=(torch.tensor(tree.parents,device=device,dtype=torch.int32)
             if os.environ.get('SPARK_TREE_GDN_KERNEL')=='1' else None)
    pending=(engine.embedding[tokens].repeat(1,engine.config['hc_count']),None,None)
    node_layers=[[] for _ in tree.tokens]
    for index,(layer,base) in enumerate(zip(engine.layers,state.layers)):
        hidden,block,injection=pending
        if layer.ple is not None:
            if block is not None:
                hidden=layer.attn_hc.combine(hidden,block,injection)
                block=injection=None
            enhancement,ple=ple_nodes(layer,hidden,tree,state.tokens,base.ple)
            hidden=hidden+enhancement
        else:ple=[base.ple]*len(tree.tokens)
        if block is None:
            residual,mixed,injection=layer.attn_hc.mix(hidden,True)
        else:
            residual,mixed,injection=layer.attn_hc.combine_and_mix(hidden,block,injection,True)
        attention,states=attention_nodes(layer,mixed,positions,tree,base.attention,start,parents)
        residual,mixed,injection=layer.mlp_hc.combine_and_mix(residual,attention,injection,True)
        pending=(residual,layer.moe(mixed,preserve_rows=True),injection)
        if observer is not None:observer(index,pending)
        for node,(att,pl) in enumerate(zip(states,ple)):
            node_layers[node].append(DecoderState(att,pl))
    multi,sampled,_=engine.final_mixer.combine_and_mix(*pending,True)
    if observer is not None:observer('final',(multi,sampled,None))
    logits=engine.logits(sampled)
    nodes=[EngineState(layers,list(state.tokens)+[tree.tokens[n] for n in tree.path(node)])
           for node,layers in enumerate(node_layers)]
    outputs=[];paths=tree.paths()
    for path in paths:
        indices=torch.tensor(path,device=device,dtype=torch.long)
        final=nodes[path[-1]]
        branch=EngineState(final.layers,final.tokens,[nodes[n] for n in path])
        outputs.append((logits.index_select(0,indices),branch,
            ForwardHidden(multi.index_select(0,indices),sampled.index_select(0,indices))))
    tx=TreeVerification(engine,tree,state,paths,outputs)
    tx.implementation=('unique_nodes_tree_gdn_kernel' if parents is not None
                       else 'unique_nodes_parent_states_eager')
    engine._pending_verification=tx
    return tx
