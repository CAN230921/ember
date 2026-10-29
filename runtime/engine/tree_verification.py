"""Correctness reference for bounded trees through existing private lanes.

Root-to-leaf paths are expanded into independent batch lanes. Shared ancestors
are intentionally recomputed and checked for identical logits. This establishes
tree semantics with the existing QSA/PLE implementations; it is NOT the merged
node verifier and makes no tree-speed claim.
"""
import torch

from .batch_mtp_model import BatchMTPModel, materialize_views
from .candidate_tree import CandidateTree, TreeSelection
from .model import EngineState, ForwardHidden


class TreeVerification:
    def __init__(self,engine,tree,base,paths,outputs):
        self.engine=engine;self.tree=tree;self.base=base;self.base_tokens=tuple(base.tokens)
        self.paths=paths;self.outputs=outputs;self.status='open'
        logits={};hidden={};sampled={}
        for path,(rows,_,features) in zip(paths,outputs):
            if not bool(torch.isfinite(rows).all()):raise RuntimeError('Nonfinite tree logits')
            for row,node in enumerate(path):
                if node in logits and not torch.equal(rows[row],logits[node]):
                    raise RuntimeError('Shared ancestor logits differ across reference lanes')
                logits[node]=rows[row];hidden[node]=features.multi[row]
                sampled[node]=features.sampled[row]
        self.logits=torch.stack([logits[n] for n in range(len(tree.tokens))])
        self.hidden=ForwardHidden(torch.stack([hidden[n] for n in range(len(tree.tokens))]),
                                 torch.stack([sampled[n] for n in range(len(tree.tokens))]))
        self.predictions=tuple(self.logits.argmax(-1).tolist())

    def select(self,budget,eos=()):
        return self.tree.accept_argmax(self.predictions,budget,eos)

    def commit(self,selection):
        if self.status!='open' or getattr(self.engine,'_pending_verification',None) is not self:
            raise RuntimeError('Tree verification is closed or superseded')
        if not isinstance(selection,TreeSelection):raise ValueError('Expected TreeSelection')
        nodes=tuple(selection.consumed_nodes)
        if (not nodes or nodes[0]!=0 or any(type(n) is not int or not 0<=n<len(self.predictions) for n in nodes) or
                tuple(selection.emitted)!=tuple(self.predictions[n] for n in nodes)):
            raise ValueError('Invalid tree selection')
        if tuple(self.base.tokens)!=self.base_tokens:raise RuntimeError('Tree base history changed')
        for parent,child in zip(nodes,nodes[1:]):
            if self.tree.parents[child]!=parent or self.tree.tokens[child]!=self.predictions[parent]:
                raise ValueError('Selected nodes are not the verified target path')
        for path,(_,state,_) in zip(self.paths,self.outputs):
            if path[:len(nodes)]==nodes:
                result=materialize_views(state.prefixes[len(nodes)-1])
                # Only the selected state survives. Detach GDN node-bank views
                # even when the serving LAZY_CLONE option is disabled.
                from .decoder import DecoderState
                from .ops.gdn import GDNState
                layers=[]
                for layer in result.layers:
                    attention=layer.attention
                    if getattr(attention,'tree_view',False):
                        attention=GDNState(attention.conv.clone(),attention.recurrent.clone())
                    layers.append(DecoderState(attention,layer.ple))
                result=EngineState(layers,list(result.tokens))
                assert result.tokens==list(self.base_tokens)+[self.tree.tokens[n] for n in nodes]
                self.status='committed';self._release();return result
        raise ValueError('Selected path is absent from reference lanes')

    def discard(self):
        if self.status!='open':raise RuntimeError('Tree verification is already closed')
        self.status='discarded';self._release()

    def _release(self):
        self.outputs=[];self.base=None
        if getattr(self.engine,'_pending_verification',None) is self:
            self.engine._pending_verification=None


@torch.no_grad()
def verify_tree_reference(engine,tree,state):
    if not isinstance(tree,CandidateTree):raise TypeError('Expected CandidateTree')
    paths=tree.paths()
    if len(paths)>4:raise ValueError('Reference verifier supports at most four leaves')
    if (not state.tokens or len(state.layers)!=len(engine.layers) or
            any(t>=engine.config['vocab_size'] for t in tree.tokens)):
        raise ValueError('Invalid checkpoint tree or base state')
    if max(tree.positions(len(state.tokens)))>=engine.config['max_position_embeddings']:
        raise ValueError('Tree depth exceeds context limit')
    engine.discard_verification()
    ids=[[tree.tokens[n] for n in path] for path in paths]
    if len(paths)==1:
        outputs=[engine.verify_tokens(ids[0],state,lazy_prefixes=True)]
    else:
        if not hasattr(engine,'_tree_reference_model'):
            engine._tree_reference_model=BatchMTPModel(engine)
        states=[EngineState(list(state.layers),list(state.tokens)) for _ in paths]
        outputs=engine._tree_reference_model.verify(ids,states)
    tx=TreeVerification(engine,tree,state,paths,outputs)
    engine._pending_verification=tx
    return tx
