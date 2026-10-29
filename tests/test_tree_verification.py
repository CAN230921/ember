from types import SimpleNamespace

import pytest
import torch

from engine.candidate_tree import CandidateTree,TreeSelection
from engine.decoder import DecoderState
from engine.model import EngineState,ForwardHidden,GreedyEngineRef
from engine.tree_verification import verify_tree_reference


class Target:
    discard_verification=GreedyEngineRef.discard_verification
    def __init__(self):
        self.layers=[None];self.config=dict(vocab_size=64,max_position_embeddings=64)
        self._tree_reference_model=SimpleNamespace(verify=lambda ids,states:[self.verify_tokens(i,s) for i,s in zip(ids,states)])
    def verify_tokens(self,ids,state,**kwargs):
        rows=[];prefixes=[]
        for n in range(1,len(ids)+1):
            history=state.tokens+ids[:n];prediction=(sum(history)+1)%64
            row=torch.full((64,),-1.);row[prediction]=1.;rows.append(row)
            prefixes.append(EngineState([DecoderState(tuple(history))],history))
        hidden=torch.tensor([[float(t)] for t in ids])
        updated=EngineState(prefixes[-1].layers,prefixes[-1].tokens,prefixes)
        return torch.stack(rows),updated,ForwardHidden(hidden,hidden)


def fixture():
    # Root predicts 7. Its matching child is the second branch, which predicts 14.
    return Target(),EngineState([DecoderState((2,3))],[2,3]),CandidateTree((1,9,7,20,14),(-1,0,0,1,2))


def test_reference_tree_matches_serial_greedy_and_commits_only_consumed_inputs():
    engine,base,tree=fixture();tx=verify_tree_reference(engine,tree,base)
    selected=tx.select(8);assert selected.consumed_nodes==(0,2,4)
    assert selected.emitted==(7,14,28)
    state=tx.commit(selected)
    assert state.tokens==[2,3,1,7,14]
    assert base.tokens==[2,3]
    assert engine._pending_verification is None
    with pytest.raises(RuntimeError):tx.commit(selected)


@pytest.mark.parametrize('budget',[1,2,3])
def test_reference_tree_budget(budget):
    engine,base,tree=fixture();tx=verify_tree_reference(engine,tree,base)
    selected=tx.select(budget);state=tx.commit(selected)
    assert selected.emitted==(7,14,28)[:budget]
    assert state.tokens==[2,3]+[1,7,14][:budget]


def test_reference_eos_is_output_but_not_consumed():
    engine,base,tree=fixture();tx=verify_tree_reference(engine,tree,base)
    selected=tx.select(8,eos=(14,));state=tx.commit(selected)
    assert selected.emitted==(7,14) and state.tokens==[2,3,1,7]


def test_sibling_state_cannot_be_committed_as_target_path():
    engine,base,tree=fixture();tx=verify_tree_reference(engine,tree,base)
    wrong=TreeSelection((0,1),(7,16),'leaf')
    with pytest.raises(ValueError):tx.commit(wrong)
    tx.discard()


def test_changed_base_and_superseded_transactions_are_rejected():
    engine,base,tree=fixture();old=verify_tree_reference(engine,tree,base)
    new=verify_tree_reference(engine,tree,base)
    with pytest.raises(RuntimeError):old.commit(old.select(1))
    base.tokens.append(0)
    with pytest.raises(RuntimeError,match='history changed'):new.commit(new.select(1))
    new.discard()


@pytest.mark.parametrize('view',['tree_view','graph_view'])
@pytest.mark.parametrize('lazy',[False,True])
def test_committed_tree_bank_state_releases_all_rejected_node_storage(view,lazy,monkeypatch):
    monkeypatch.setattr('engine.batch_mtp_model.LAZY_CLONE',lazy)
    from engine.ops.gdn import GDNState
    engine,base,tree=fixture();tx=verify_tree_reference(engine,tree,base)
    conv=torch.arange(15.).reshape(5,3);rec=torch.arange(20.).reshape(5,4)
    for path,(_,state,_) in zip(tx.paths,tx.outputs):
        for node,prefix in zip(path,state.prefixes):
            value=GDNState(conv[node],rec[node]);setattr(value,view,True)
            prefix.layers=[DecoderState(value)]
    updated=tx.commit(tx.select(8));attention=updated.layers[0].attention
    assert attention.conv.tolist()==[12.,13.,14.]
    assert attention.conv.untyped_storage().nbytes()==12
    assert attention.recurrent.untyped_storage().nbytes()==16
    conv.zero_();rec.zero_()
    assert attention.conv.tolist()==[12.,13.,14.] and not hasattr(attention,view)
