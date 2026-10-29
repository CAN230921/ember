"""Tree service state-machine and HTTP contracts, using CPU ownership oracles."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import json
import threading

import pytest
import torch
from fastapi.testclient import TestClient

from api_server import Runtime,create_app,MODEL
from engine.candidate_tree import CandidateTree
from engine.continuous_mtp import ContinuousMTPPrefixCache
from engine.dflash_draft import LAYERS
from engine.request_control import GenerationCancelled
from engine.tree_runner import TreeMTPRunner
from speculation import settings
from test_continuous_mtp import ToyTarget,ToyDraft,greedy_gold,history_hash,next_token


class Target(ToyTarget):
    def __init__(self):super().__init__();self.after_verify=None;self.last_tx=None
    def __call__(self,ids,state=None,observer=None,return_hidden=False,**kwargs):
        logits,state,hidden=super().__call__(ids,state,return_hidden=True,**kwargs)
        if observer is not None:
            for layer in LAYERS:observer(layer,(hidden.multi,None,None))
        return (logits,state,hidden) if return_hidden else (logits,state)
    def verify_tree(self,tree,base,observer=None):
        histories=[base.tokens+[tree.tokens[i] for i in tree.path(n)] for n in range(len(tree.tokens))]
        logits=torch.cat([self._logits(h) for h in histories])
        hidden=torch.tensor([[len(h)-1,history_hash(h),0] for h in histories])
        if observer is not None:
            for layer in LAYERS:observer(layer,(hidden,None,None))
        target=self
        class Transaction:
            status='open'
            def select(self,budget,eos):return tree.accept_argmax(tuple(logits.argmax(-1).tolist()),budget,eos)
            def commit(self,selection):
                self.status='committed';h=histories[selection.consumed_nodes[-1]]
                return SimpleNamespace(tokens=h,bank=torch.tensor([history_hash(h)]),prefixes=None)
            def discard(self):self.status='discarded'
        tx=target.last_tx=Transaction()
        if target.after_verify:target.after_verify()
        return tx


class Draft:
    def __init__(self,reject=False):self.reject=reject
    def new_context(self,start=0):return SimpleNamespace(length=start,hash=7)
    @staticmethod
    def tap(hidden,block,injection):return hidden
    @staticmethod
    def extend(ctx,features):ctx.length+=len(features);ctx.hash=int(features[-1,1])
    push=extend
    def propose_tree(self,ctx,anchor,**kwargs):
        root=history_hash([anchor],ctx.hash);first=next_token(root);other=1+first%200
        if self.reject:return CandidateTree((anchor,other),(-1,0))
        second=next_token(history_hash([first],root))
        other_next=next_token(history_hash([other],root))
        return CandidateTree((anchor,first,other,second,other_next),(-1,0,0,1,2))


def runner(cache=None,reject=False):
    return TreeMTPRunner(Target(),ToyDraft(),cache,dflash=Draft(reject),draft_tokens=2)


def prepare(r,ids,budget=17,eos=(255,),cancel=None):
    s=r.start(ids,budget,set(eos),64,should_cancel=cancel)
    while s.phase=='prefill':r.prefill(s)
    return s


def finish(r,sessions):
    while any(not s.done for s in sessions):
        live=[s for s in sessions if not s.done]
        assert r.decode_batch(live)==[None]*len(live)


@pytest.mark.parametrize('lanes',[1,2,4])
@pytest.mark.parametrize('budget',[2,3,17])
@pytest.mark.parametrize('reject',[False,True])
def test_tree_sessions_match_greedy_and_keep_private_histories(lanes,budget,reject):
    r=runner(reject=reject);ids=[[3+i,5,8] for i in range(lanes)]
    sessions=[prepare(r,p,budget) for p in ids];finish(r,sessions)
    for s,p in zip(sessions,ids):
        assert s.generated==greedy_gold(p,budget)
        assert s.state.tokens==p+s.generated[:-1] and int(s.state.bank[0])==history_hash(s.state.tokens)
        assert s.dflash_ctx.length==len(s.state.tokens) and s.dflash_ctx.hash==history_hash(s.state.tokens)
        stats=s.result(0)['speculative']
        assert stats['strategy']=='tree' and not stats['batched_verification']
        assert stats['proposed']==stats['accepted']+stats['corrections']+stats['discarded']
        r.close(s);assert s.dflash_ctx is None


@pytest.mark.parametrize('position',[0,1,2,5])
def test_tree_eos_is_emitted_but_unconsumed(position):
    r=runner();ids=[3,5,8];eos={greedy_gold(ids,12)[position]}
    s=prepare(r,ids,12,eos);finish(r,[s])
    assert s.generated==greedy_gold(ids,12,eos) and s.state.tokens==ids+s.generated[:-1]
    assert s.result(0)['finish_reason']=='stop';r.close(s)


@pytest.mark.parametrize('after_verify',[False,True])
def test_tree_cancel_discards_disposable_verification_and_preserves_peer(after_verify):
    r=runner();flag=threading.Event();a=prepare(r,[3,5,8],cancel=flag.is_set);b=prepare(r,[4,5,8])
    before=(list(a.state.tokens),int(a.state.bank[0]),a.dflash_ctx.length,a.dflash_ctx.hash)
    if after_verify:r.engine.after_verify=flag.set
    else:flag.set()
    errors=r.decode_batch([a,b]);r.engine.after_verify=None
    assert isinstance(errors[0],GenerationCancelled) and errors[1] is None
    assert (a.state.tokens,int(a.state.bank[0]),a.dflash_ctx.length,a.dflash_ctx.hash)==before
    if after_verify:assert r.engine.last_tx.status=='committed'  # the peer's later transaction
    r.close(a);finish(r,[b]);assert b.generated==greedy_gold([4,5,8],17);r.close(b)


def test_tree_reasoning_limits_forced_tokens_and_continuation():
    r=runner();ids=[3,5,8];s=prepare(r,ids)
    assert r.decode_batch([s],emission_limits=[1])==[None]
    assert len(s.generated)==2
    r.force_next(s,11);prefix=list(s.generated)
    finish(r,[s])
    assert s.generated==prefix+greedy_gold(ids+prefix,17-len(prefix))
    assert s.stats['forced_control_tokens']==1 and s.state.tokens==ids+s.generated[:-1]
    r.close(s)


def test_tree_prefill_cache_keeps_paired_snapshots_and_reuses_exact_prefix():
    cache=ContinuousMTPPrefixCache();r=runner(cache);ids=[3]*145
    a=prepare(r,ids,8);finish(r,[a]);gold=list(a.generated);r.close(a)
    b=prepare(r,ids,8);assert b.reused==128 and b.dflash_ctx.length==len(ids)
    finish(r,[b]);assert b.generated==gold;r.close(b)


def test_startup_switch_defaults_off_and_validates_before_loading(monkeypatch):
    monkeypatch.delenv('SPARK_SPECULATION',raising=False)
    assert settings()==dict(mode='mtp')
    monkeypatch.setenv('SPARK_SPECULATION','bad')
    with pytest.raises(ValueError):settings()
    monkeypatch.setenv('SPARK_SPECULATION','tree');monkeypatch.delenv('SPARK_DFLASH_DIR',raising=False)
    with pytest.raises(ValueError):settings()
    monkeypatch.setenv('SPARK_DFLASH_DIR','/fixture');assert settings()['policy']=='best_first'
    with pytest.raises(ValueError):Runtime(mode='serial_mtp')


def test_tree_real_scheduler_json_sse_status_and_multiple_clients(monkeypatch,tmp_path):
    monkeypatch.setenv('SPARK_API_AUDIT',str(tmp_path/'audit.jsonl'))
    monkeypatch.setenv('SPARK_DFLASH_DIR','/fixture');monkeypatch.setenv('SPARK_MTP_POLICY','acceptance')
    r=Runtime(mode='continuous_mtp',mtp_batch=True,speculation='tree',max_active=4,prepare_workers=0)
    r.ready=True;r.eos={255};r.tokenizer=SimpleNamespace(decode=lambda ids,**kwargs:','.join(map(str,ids)))
    r.batch_runner=runner(r.prefix_cache)
    monkeypatch.setattr('api_server.chat_token_ids',lambda *a,**k:[3,5,8])
    key='tree-service-cpu-fixture';headers={'Authorization':'Bearer '+key}
    client=TestClient(create_app(r,key))
    body=dict(model=MODEL,messages=[dict(role='user',content='fixture')],max_tokens=17,temperature=0)
    try:
        status=client.get('/v1/scheduler',headers=headers).json()
        assert status['speculation']['mode']=='tree' and not status['batched_verification']
        response=client.post('/v1/chat/completions',headers=headers,json=body)
        assert response.status_code==200 and response.json()['usage']['completion_tokens']==17
        streamed=client.post('/v1/chat/completions',headers=headers,json=dict(body,stream=True,stream_options=dict(include_usage=True)))
        assert streamed.status_code==200 and streamed.text.endswith('data: [DONE]\n\n')
        chunks=[json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith('data: ') and line!='data: [DONE]']
        text=''.join(c['choices'][0].get('delta',{}).get('content','') for c in chunks if c.get('choices'))
        assert text==response.json()['choices'][0]['message']['content']
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows=list(pool.map(lambda _:client.post('/v1/chat/completions',headers=headers,json=body),range(4)))
        assert all(row.status_code==200 for row in rows)
        assert all(row.json()['choices']==response.json()['choices'] for row in rows)
        assert r.context_pool.snapshot()['reserved_tokens']==0
    finally:r.close()
