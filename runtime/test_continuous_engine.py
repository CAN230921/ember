"""CPU tensor/ownership oracles. These do not validate CUDA kernels or speed."""
from types import SimpleNamespace
import unittest
import torch

from engine.continuous import GreedyBatchRunner
from engine.decode_graph import DecodeGraphBlock
from engine.decoder import DecoderState
from engine.model import EngineState
from engine.ops.gdn import GDNState
from engine.ops.greedy_select import GreedySelector
from engine.greedy_prefix_cache import GreedyPrefixCache
from engine.prefix_cache import clone_state


class ToyEngine:
    """History-sensitive CPU model; input/output states have actual tensors."""
    def __init__(self):
        self.embedding=torch.zeros(1);self.calls=[]

    def __call__(self,ids,state=None,prefill_chunk=False,owned_graph_cache=False):
        self.calls.append(dict(prefill=prefill_chunk,owned=owned_graph_cache,rows=len(ids)))
        tokens=(state.tokens if state else [])+list(ids)
        value=sum((i+1)*token for i,token in enumerate(tokens))%997
        layer=DecoderState(GDNState(torch.tensor([value]),torch.tensor([value+1])),
                           ple=torch.tensor(tokens[-3:]))
        state=EngineState([layer],tokens)
        logits=torch.full((1,64),-100.)
        logits[0,1+value%31]=100.
        return logits,state


class ToyBatch:
    def __init__(self,engine):self.engine=engine;self.calls=[]
    def decode(self,ids,states):
        self.calls.append([list(s.tokens) for s in states])
        result=[self.engine([token],state,owned_graph_cache=True) for token,state in zip(ids,states)]
        return torch.cat([r[0] for r in result]),[r[1] for r in result]


def make_runner(cache=None):
    runner=GreedyBatchRunner.__new__(GreedyBatchRunner)
    runner.engine=ToyEngine();runner.cache=cache
    runner.batch=ToyBatch(runner.engine);runner.batch_selector=GreedySelector()
    return runner


def finish_serial(runner,ids,budget=7,eos=(63,)):
    session=runner.start(ids,budget,eos,64)
    while not session.done:
        if session.phase=='prefill':runner.prefill(session)
        else:runner.decode([session])
    return session


class ContinuousEngineTests(unittest.TestCase):
    def test_m1_graph_a_b_a_snapshots_survive_static_bank_overwrite(self):
        block=DecodeGraphBlock([SimpleNamespace(layer_type='linear_attention',ple=None)])
        block.pending=(torch.zeros(1),None,None)
        block.states=[DecoderState(GDNState(torch.zeros(1),torch.zeros(1)))]
        block.output=block.pending
        def replay():
            block.states[0].attention.conv.add_(block.pending[0])
            block.states[0].attention.recurrent.add_(block.pending[0]*2)
        block.graph=SimpleNamespace(replay=replay)
        def state(value):return [DecoderState(GDNState(torch.tensor([value]),torch.tensor([value])))]
        _,a=block((torch.ones(1),None,None),state(100.),owned=True)
        _,b=block((torch.tensor([2.]),None,None),state(200.),owned=True)
        self.assertEqual(a[0].attention.conv.item(),101.)
        self.assertNotEqual(a[0].attention.conv.data_ptr(),block.states[0].attention.conv.data_ptr())
        _,next_a=block((torch.ones(1),None,None),a,owned=True)
        self.assertEqual(next_a[0].attention.conv.item(),102.)
        self.assertEqual(b[0].attention.conv.item(),202.)

    def test_independent_batch_then_single_then_swapped_batch_matches_serial(self):
        runner=make_runner();a=runner.start([1,2,3],7,{63},64);b=runner.start([9,8],3,{63},64)
        runner.prefill(a);runner.prefill(b)
        runner.decode([a,b]);runner.decode([a,b])
        self.assertTrue(b.done)
        b_state=clone_state(b.state);runner.close(b)
        runner.decode([a])
        c=runner.start([4,5,6,7],4,{63},64);runner.prefill(c)
        while not a.done and not c.done:runner.decode([c,a])
        for session in (a,c):
            while not session.done:runner.decode([session])
        gold_a=finish_serial(make_runner(),[1,2,3],7)
        gold_b=finish_serial(make_runner(),[9,8],3)
        gold_c=finish_serial(make_runner(),[4,5,6,7],4)
        for actual,gold in ((a,gold_a),(c,gold_c)):
            self.assertEqual(actual.generated,gold.generated)
            self.assertEqual(actual.state.tokens,gold.state.tokens)
            self.assertTrue(torch.equal(actual.state.layers[0].attention.conv,gold.state.layers[0].attention.conv))
            self.assertTrue(torch.equal(actual.state.layers[0].ple,gold.state.layers[0].ple))
        self.assertEqual(b_state.tokens,gold_b.state.tokens)
        self.assertTrue(all(call['owned'] for call in runner.engine.calls if not call['prefill']))
        self.assertGreater(a.batch_steps,0);self.assertGreater(a.single_steps,0)

    def test_progress_failure_occurs_after_both_batch_rows_commit(self):
        runner=make_runner();a=runner.start([1],4,{63},64);b=runner.start([2],4,{63},64)
        runner.prefill(a);runner.prefill(b)
        def bad(*args):raise ValueError('progress failed')
        a.on_progress=bad
        runner.decode([a,b])
        self.assertEqual(len(a.generated),2);self.assertEqual(len(b.generated),2)
        with self.assertRaisesRegex(ValueError,'progress'):a.progress()
        while not b.done:runner.decode([b])
        self.assertEqual(b.generated,finish_serial(make_runner(),[2],4).generated)

    def test_prefix_checkout_private_and_usage_belongs_to_each_session(self):
        cache=GreedyPrefixCache(4096);runner=make_runner(cache)
        warm=finish_serial(runner,[3]*130);runner.close(warm)
        a=runner.start([3]*130,3,{63},64);b=runner.start([4]*130,3,{63},64)
        self.assertEqual((a.reused,b.reused),(128,0))
        other,end=cache.checkout([3]*130,64)
        original=other.layers[0].attention.conv.clone()
        a.state.layers[0].attention.conv.add_(100)
        self.assertTrue(torch.equal(other.layers[0].attention.conv,original))
        self.assertTrue(torch.equal(cache.entries[-1].target.layers[0].attention.conv,original))
        self.assertEqual(a.position,128);self.assertEqual(end,128)
        self.assertIsNone(cache.checkout([3]*130,128)[0])
        runner.close(a);runner.close(b)

    def test_single_cache_writer_avoids_dual_prefill_snapshot_thrash(self):
        cache=GreedyPrefixCache(4096);runner=make_runner(cache)
        a=runner.start([3]*193,3,{63},64);b=runner.start([4]*193,3,{63},64)
        runner.prefill(a);first=cache.entries[0].target
        runner.prefill(b)
        self.assertIs(cache.entries[0].target,first)
        runner.prefill(a);runner.prefill(b)
        self.assertEqual(cache.entries[-1].key,tuple([3]*128))
        self.assertEqual(b.checkpoint_tokens,0)
        runner.close(a);runner.prefill(b)
        self.assertEqual(cache.entries[-1].key,tuple([4]*192))
        self.assertLessEqual(cache.bytes,4096)
        runner.close(b);self.assertIsNone(cache.owner)

    def test_eos_and_budget_leave_last_output_unconsumed(self):
        probe=finish_serial(make_runner(),[2,3],1)
        session=finish_serial(make_runner(),[2,3],8,{probe.generated[0]})
        self.assertEqual(session.generated,probe.generated)
        self.assertEqual(session.state.tokens,[2,3])
        self.assertEqual(session.result(0)['finish_reason'],'stop')
        budget=finish_serial(make_runner(),[2,3],3)
        self.assertEqual(budget.state.tokens,[2,3]+budget.generated[:-1])
        self.assertEqual(budget.result(0)['finish_reason'],'length')


if __name__=='__main__':unittest.main()
