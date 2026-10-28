"""CUDA Graph for contiguous GDN decoder layers, excluding disk-backed PLE.

Static caches belong to this engine instance. Warmup/capture use cloned
state; a fresh request restores its own prefill state. Cache writebacks are
inside the graph, so replay advances each layer exactly once.
"""
import torch

from engine.limits import VERIFY_WIDTHS
from engine.graph_capture import graph as capture_graph

from engine.decoder import DecoderState
from engine.ops.gdn import GDNState


class DecodeGraphBlock:
    def __init__(self,layers,pool=None,return_prefixes=False):
        self.layers=list(layers)
        if not self.layers or any(l.layer_type!='linear_attention' or l.ple is not None for l in self.layers):
            raise ValueError('Graph block requires GDN layers without PLE')
        self.pool=pool
        self.return_prefixes=return_prefixes
        self.graph=None
        self.intermediates=[]
        self.replays=0

    def _copy_inputs(self,pending,states):
        if len(pending)!=3 or len(states)!=len(self.layers):raise ValueError('Graph input structure changed')
        for source,target in zip(pending,self.pending):
            if (source is None)!=(target is None):raise ValueError('Graph pending structure changed')
            if source is not None:
                if source.shape!=target.shape or source.dtype!=target.dtype or source.device!=target.device:
                    raise ValueError('Graph input shape/dtype changed')
                target.copy_(source)
        for source,target in zip(states,self.states):
            if source.ple is not None:raise ValueError('Graph block cannot hold PLE')
            for name in ('conv','recurrent'):
                a,b=getattr(source.attention,name),getattr(target.attention,name)
                if a.shape!=b.shape or a.dtype!=b.dtype or a.device!=b.device:raise ValueError('Graph cache shape/dtype changed')
                # Replay returns these same persistent buffers. Only a new
                # request's prefill state needs a cache copy outside the graph.
                if a.data_ptr()!=b.data_ptr():b.copy_(a)

    def _run(self):
        value=self.pending
        intermediate=[]
        for layer,state in zip(self.layers,self.states):
            value,updated=layer(value[0],None,value[1],value[2],state=state,prefill=False,return_prefixes=self.return_prefixes)
            intermediate.append(value)
            state.attention.conv.copy_(updated.attention.conv)
            state.attention.recurrent.copy_(updated.attention.recurrent)
            if self.return_prefixes:state.prefixes=updated.prefixes
        self.intermediates=intermediate
        return value

    def _capture(self,pending,states):
        widths=VERIFY_WIDTHS if self.return_prefixes else (1,)
        if len(states)!=len(self.layers) or pending[0].shape[0] not in widths:
            raise ValueError('Invalid graph decode width')
        self.pending=tuple(x.detach().clone() if x is not None else None for x in pending)
        self.states=[DecoderState(GDNState(s.attention.conv.detach().clone(),
                                          s.attention.recurrent.detach().clone())) for s in states]
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._copy_inputs(pending,states)
                self._run()
        torch.cuda.current_stream().wait_stream(stream)
        self._copy_inputs(pending,states)
        torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with capture_graph(self.graph,pool=self.pool,stream=stream):
            self.output=self._run()
        # Capturing executed once. Restore the true caller's unadvanced
        # caches before the first replay; caller-owned state was never touched.
        self._copy_inputs(pending,states)

    def forward(self,pending,states,borrowed=False,owned=False,lazy=False):
        if borrowed and not self.return_prefixes:raise ValueError('Borrowed caches require verification prefixes')
        if borrowed and owned:raise ValueError('Graph caches cannot be both borrowed and owned')
        if self.graph is None:self._capture(pending,states)
        else:self._copy_inputs(pending,states)
        self.graph.replay()
        self.replays+=1
        if self.return_prefixes:
            if borrowed:
                # Private MTP ownership: prefix buffers remain valid until
                # this graph is replayed again. Accepted rows use the graph's
                # persistent final cache; rejection restores prefix zero
                # before replay, in the same CUDA stream. Public callers
                # retain the default immutable snapshot behavior below.
                return self.output,self.states
            if lazy:
                # Replay-buffer views; the caller clones only the prefix it commits (materialize_views)
                # before any later replay can overwrite them (saves cloning every rejected row's bank).
                viewed=[]
                for state in self.states:
                    prefixes=[]
                    for s in state.prefixes:
                        view=GDNState(s.attention.conv,s.attention.recurrent); view.graph_view=True
                        prefixes.append(DecoderState(view))
                    viewed.append(DecoderState(prefixes[-1].attention,prefixes=prefixes))
                return self.output,viewed
            # Replay overwrites graph-owned prefix buffers. Return independent
            # cache snapshots so another replay cannot mutate the caller's
            # accepted/rejected state while it is still a valid commit point.
            copied=[]
            for state in self.states:
                prefixes=[DecoderState(GDNState(s.attention.conv.clone(),s.attention.recurrent.clone()))
                          for s in state.prefixes]
                copied.append(DecoderState(prefixes[-1].attention,prefixes=prefixes))
            return self.output,copied
        if owned:
            # M1 also returns graph-owned mutable banks by default. A scheduler
            # must snapshot these before another request replays this graph.
            return self.output,[DecoderState(GDNState(s.attention.conv.clone(),
                s.attention.recurrent.clone())) for s in self.states]
        return self.output,self.states

    def __call__(self,pending,states,borrowed=False,owned=False,lazy=False):return self.forward(pending,states,borrowed,owned,lazy)
