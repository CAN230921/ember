from types import SimpleNamespace

import pytest
import torch

from engine.decoder import DecoderState
from engine.model import EngineState, GreedyEngineRef
from engine.ops.gdn_deferred import DeferredGDN, _versions
from engine.verification import LinearVerification, verify_linear


class AttentionFixture:
    layer_type='full_attention'
    def __call__(self,hidden,positions,block,injection,ids,context,state,**kwargs):
        prefixes=[DecoderState((state.attention,tuple(ids[:n])),('ple',n)) for n in range(1,len(ids)+1)]
        return (hidden+1,None,None),DecoderState(prefixes[-1].attention,prefixes=prefixes)


class EngineFixture:
    discard_verification=GreedyEngineRef.discard_verification
    graph_decode=False;graph_verify=False;_graph_ranges={}
    def __init__(self):
        self.config=dict(vocab_size=12,max_position_embeddings=32,ngram_size=3,eos_token_id=0,hc_count=1)
        self.embedding=torch.arange(24).reshape(12,2).float()
        self.layers=[AttentionFixture()]
        self.final_mixer=SimpleNamespace(combine_and_mix=lambda h,b,i,p:(h,h,None))
    def logits(self,x):return x*2


def fixture():
    return EngineFixture(),EngineState([DecoderState('base')],[1,2,3])


@pytest.mark.parametrize('count',[1,2,3])
def test_explicit_commit_consumes_inputs_and_keeps_base(count):
    engine,base=fixture();tx=verify_linear(engine,[4,5,6],base)
    assert torch.equal(tx.hidden.multi,engine.embedding[[4,5,6]]+1)
    updated=tx.commit(count)
    assert updated.tokens==[1,2,3]+[4,5,6][:count]
    assert updated.layers[0].attention==('base',tuple([4,5,6][:count]))
    assert updated.layers[0].ple==('ple',count)
    assert base.tokens==[1,2,3] and base.layers[0].attention=='base'
    assert engine._pending_verification is None
    with pytest.raises(RuntimeError):tx.commit(count)


@pytest.mark.parametrize('count',[True,False,0,-1,4,1.0,None])
def test_invalid_commit_keeps_transaction_available(count):
    engine,base=fixture();tx=verify_linear(engine,[4,5,6],base)
    with pytest.raises(ValueError):tx.commit(count)
    assert tx.status=='open'
    tx.discard();assert engine._pending_verification is None


def test_new_verification_invalidates_borrowed_previous_tape():
    engine,base=fixture();old=verify_linear(engine,[4,5],base)
    current=verify_linear(engine,[6],base)
    assert old.status=='discarded'
    with pytest.raises(RuntimeError):old.commit(1)
    assert current.commit(1).tokens==[1,2,3,6]


def test_observer_error_releases_lease_and_base():
    engine,base=fixture()
    def fail(*args):raise RuntimeError('observer failed')
    with pytest.raises(RuntimeError,match='observer failed'):verify_linear(engine,[4,5],base,fail)
    assert engine._pending_verification is None
    assert base.layers[0].attention=='base'


def test_base_history_change_is_rejected_before_commit():
    engine,base=fixture();tx=verify_linear(engine,[4],base);base.tokens.append(9)
    with pytest.raises(RuntimeError,match='history changed'):tx.commit(1)
    tx.discard()


def test_inference_state_versions_are_optional():
    with torch.inference_mode():
        state=SimpleNamespace(conv=torch.zeros(1),recurrent=torch.zeros(1))
        assert _versions(state)==(None,None)


class TapeFixture(DeferredGDN):
    def __init__(self,fail=False):self._status='open';self.fail=fail
    def validate(self,count):assert self._status=='open'
    def commit(self,count):
        if self.fail:raise RuntimeError('commit failed')
        self._status='committed';return ('committed',count)
    def discard(self):self._status='discarded'


def test_failed_commit_never_publishes_a_partial_engine_state(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'engine.ops.gdn_commit',
        SimpleNamespace(commit_many=lambda tapes,count:[t.commit(count) for t in tapes]))
    engine,base=fixture();tx=LinearVerification(engine,[4],base);engine._pending_verification=tx
    a,b=TapeFixture(),TapeFixture(True)
    tx.layers=[DecoderState(t,prefixes=[DecoderState(t)]) for t in (a,b)]
    with pytest.raises(RuntimeError,match='commit failed'):tx.commit(1)
    assert tx.status=='failed' and engine._pending_verification is None
    assert a._status=='committed' and b._status=='discarded'
    assert base.tokens==[1,2,3] and base.layers[0].attention=='base'
