"""Allocator-policy boundaries through injected CPU doubles; no CUDA proof."""
from contextlib import contextmanager
import unittest

from engine.continuous_mtp import ContinuousMTPRunner
from engine.prefill_memory import PrefillMemoryTrimmer, TRIM_THRESHOLD_BYTES
from test_continuous_mtp import ToyTarget, ToyDraft, greedy_gold


GIB=1024**3


class FakeCuda:
    def __init__(self,reserved,allocated,*,capturing=False,retained_free=0,events=None):
        self.reserved,self.allocated=reserved,allocated
        self.capturing,self.retained_free=capturing,retained_free
        self.events=[] if events is None else events
        # Live tensor and Graph ownership must survive allocator reclamation.
        self.live=object();self.graph=object()

    @contextmanager
    def device(self,device):
        self.events.append(('device',str(device)))
        yield

    def is_current_stream_capturing(self):
        self.events.append(('capture_check',))
        return self.capturing

    def memory_reserved(self,device):
        self.events.append(('reserved',str(device)))
        return self.reserved

    def memory_allocated(self,device):
        self.events.append(('allocated',str(device)))
        return self.allocated

    def synchronize(self,device):self.events.append(('synchronize',str(device)))

    def empty_cache(self):
        self.events.append(('empty_cache',))
        self.reserved=self.allocated+self.retained_free


class PrefillMemoryTests(unittest.TestCase):
    def test_exact_threshold_does_not_trim_but_one_byte_above_does(self):
        for extra,expected in ((0,False),(1,True)):
            with self.subTest(extra=extra):
                cuda=FakeCuda(5*GIB+TRIM_THRESHOLD_BYTES+extra,5*GIB)
                trimmer=PrefillMemoryTrimmer('cuda:0',cuda=cuda)
                event=trimmer.before_prefill()
                self.assertEqual(event['trimmed'],expected)
                self.assertEqual(trimmer.snapshot()['checks'],1)
                self.assertEqual(trimmer.snapshot()['trims'],int(expected))
                self.assertEqual(('empty_cache',) in cuda.events,expected)

    def test_trim_retains_live_allocations_graph_owner_and_device_identity(self):
        cuda=FakeCuda(10*GIB,5*GIB,retained_free=GIB)
        live,graph=cuda.live,cuda.graph
        trimmer=PrefillMemoryTrimmer('cuda:1',cuda=cuda)
        event=trimmer.before_prefill()
        self.assertIs(cuda.live,live);self.assertIs(cuda.graph,graph)
        self.assertEqual(event['allocated_before_bytes'],5*GIB)
        self.assertEqual(event['allocated_after_bytes'],5*GIB)
        self.assertEqual(event['reserved_before_bytes'],10*GIB)
        self.assertEqual(event['reserved_after_bytes'],6*GIB)
        self.assertEqual(event['released_reserved_bytes'],4*GIB)
        self.assertEqual(cuda.events[0],('device','cuda:1'))
        self.assertLess(cuda.events.index(('synchronize','cuda:1')),
                        cuda.events.index(('empty_cache',)))

    def test_non_cuda_never_calls_cuda_or_reclaims_memory(self):
        cuda=FakeCuda(10*GIB,5*GIB)
        trimmer=PrefillMemoryTrimmer('cpu',cuda=cuda)
        event=trimmer.before_prefill()
        self.assertFalse(event['enabled']);self.assertFalse(event['checked'])
        self.assertEqual(event['skipped'],'non_cuda')
        self.assertEqual(cuda.events,[])
        self.assertEqual(trimmer.snapshot()['checks'],0)

    def test_graph_capture_skips_counter_reads_synchronization_and_trim(self):
        cuda=FakeCuda(10*GIB,5*GIB,capturing=True)
        trimmer=PrefillMemoryTrimmer('cuda:0',cuda=cuda)
        event=trimmer.before_prefill()
        self.assertFalse(event['checked']);self.assertFalse(event['trimmed'])
        self.assertEqual(event['skipped'],'cuda_graph_capture')
        self.assertEqual(cuda.events,[('device','cuda:0'),('capture_check',)])
        stats=trimmer.snapshot()
        self.assertEqual((stats['checks'],stats['trims'],stats['capture_skips']),(0,0,1))

    def test_counter_summary_is_bounded_and_snapshot_has_no_mutable_alias(self):
        cuda=FakeCuda(10*GIB,5*GIB)
        trimmer=PrefillMemoryTrimmer('cuda:0',cuda=cuda)
        trimmer.before_prefill();trimmer.before_prefill()
        stats=trimmer.snapshot()
        self.assertEqual((stats['checks'],stats['trims']),(2,1))
        self.assertEqual(stats['released_reserved_bytes'],5*GIB)
        self.assertEqual(stats['peak_unused_reserved_bytes'],5*GIB)
        stats['last_check']['allocated_before_bytes']=-1
        self.assertEqual(trimmer.snapshot()['last_check']['allocated_before_bytes'],5*GIB)
        cuda.reserved=4*GIB
        self.assertEqual(trimmer.before_prefill()['unused_reserved_bytes'],0)

    def test_runner_trims_before_target_and_reports_request_own_statistics(self):
        order=[]
        class Target:
            def __init__(self):
                self.inner=ToyTarget();self.embedding=self.inner.embedding
            def __call__(self,*args,**kwargs):
                order.append(('target',));return self.inner(*args,**kwargs)
        runner=ContinuousMTPRunner(Target(),ToyDraft(),draft_tokens=1)
        runner.prefill_allocator=PrefillMemoryTrimmer('cuda:0',cuda=FakeCuda(10*GIB,5*GIB,events=order))
        first=runner.start([1,2,3],2,{255},64)
        second=runner.start([9,8],2,{255},64)
        runner.prefill(first);runner.prefill(second)
        self.assertLess(order.index(('empty_cache',)),order.index(('target',)))
        runner.decode([first]);runner.decode([second])
        for session,prompt,trims in ((first,[1,2,3],1),(second,[9,8],0)):
            self.assertEqual(session.generated,greedy_gold(prompt,2))
            summary=session.result(0)['prefill_memory']
            self.assertEqual((summary['checks'],summary['trims']),(1,trims))
            runner.close(session)
        self.assertEqual((runner.prefill_allocator.snapshot()['checks'],
                          runner.prefill_allocator.snapshot()['trims']),(2,1))


if __name__=='__main__':unittest.main()
