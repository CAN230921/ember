"""Release unused CUDA allocator reservations between prefill turns.

This does not release live tensor storage, alter KV representation, or clear
CUDA Graph pools. Calls belong outside capture, after the previous turn's
temporary Python references have left scope. Memory counters are evidence of
allocator behavior, not model capacity or a bandwidth measurement.
"""
from copy import deepcopy

import torch


TRIM_THRESHOLD_BYTES=2*1024**3


def memory_summary(enabled,threshold_bytes=TRIM_THRESHOLD_BYTES):
    return dict(enabled=enabled,threshold_bytes=threshold_bytes,checks=0,trims=0,
                capture_skips=0,released_reserved_bytes=0,
                peak_unused_reserved_bytes=0,last_check=None)


def record_memory_check(summary,event):
    summary['checks']+=int(event['checked'])
    summary['trims']+=int(event['trimmed'])
    summary['capture_skips']+=int(event.get('skipped')=='cuda_graph_capture')
    summary['released_reserved_bytes']+=event.get('released_reserved_bytes',0)
    summary['peak_unused_reserved_bytes']=max(summary['peak_unused_reserved_bytes'],
        event.get('unused_reserved_bytes',0))
    if event['checked']:summary['last_check']=dict(event)


class PrefillMemoryTrimmer:
    def __init__(self,device,*,threshold_bytes=TRIM_THRESHOLD_BYTES,cuda=None):
        if type(threshold_bytes) is not int or threshold_bytes<0:
            raise ValueError('Expected a nonnegative byte threshold')
        self.device=torch.device(device)
        self.enabled=self.device.type=='cuda'
        self.threshold_bytes=threshold_bytes
        self.cuda=torch.cuda if cuda is None else cuda
        self.stats=memory_summary(self.enabled,threshold_bytes)

    def before_prefill(self):
        event=dict(enabled=self.enabled,checked=False,trimmed=False)
        if not self.enabled:
            event['skipped']='non_cuda'
            return event
        # empty_cache acts on the current device. Select the request engine's
        # device explicitly, including when a caller's current device differs.
        with self.cuda.device(self.device):
            if self.cuda.is_current_stream_capturing():
                event['skipped']='cuda_graph_capture'
                record_memory_check(self.stats,event)
                return event
            reserved=self.cuda.memory_reserved(self.device)
            allocated=self.cuda.memory_allocated(self.device)
            unused=max(0,reserved-allocated)
            event.update(checked=True,reserved_before_bytes=reserved,
                allocated_before_bytes=allocated,unused_reserved_bytes=unused,
                reserved_after_bytes=reserved,allocated_after_bytes=allocated,
                released_reserved_bytes=0)
            if unused>self.threshold_bytes:
                # Finish deferred allocator frees from earlier work before
                # returning only unoccupied reservations to the driver.
                self.cuda.synchronize(self.device)
                self.cuda.empty_cache()
                after_reserved=self.cuda.memory_reserved(self.device)
                after_allocated=self.cuda.memory_allocated(self.device)
                event.update(trimmed=True,reserved_after_bytes=after_reserved,
                    allocated_after_bytes=after_allocated,
                    released_reserved_bytes=max(0,reserved-after_reserved))
        record_memory_check(self.stats,event)
        return event

    def snapshot(self):
        return deepcopy(self.stats)
