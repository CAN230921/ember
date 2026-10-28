"""Batch x time MTP projections with private recurrent/KV commit points.

One to four lanes, each one to three contiguous positions. Large target
projections and expert calls share token rows; attention histories never do.
New MoE row counts use the existing non-fused vendor heuristic, not invented
entries in the screened M1/M2/M3 tactic cache. This opt-in path requires GPU
numerical/quality/performance acceptance before deployment.
"""
import os

import numpy as np
import torch
from torch.nn import functional as F

from .decoder import DecoderState
from .model import EngineState, ForwardHidden
from .mtp import MTPOutput
from .ops.fp8 import linear
from .ops.gdn import GDNState
from .ops.kv_cache import append_kv
from .ops.qsa_attention import pack_kv, sparse_attention
from .limits import VERIFY_ROWS
from .graph_capture import graph as capture_graph, shared_pool


def spans(lengths):
    # Up to three tokens per lane, or VERIFY_ROWS for deep (DFlash) lanes.
    if not 1 <= len(lengths) <= 4 or any(type(n) is not int or not 1 <= n <= max(3, VERIFY_ROWS) for n in lengths):
        raise ValueError('Expected one to four lanes of one to VERIFY_ROWS tokens')
    start = 0
    result = []
    for n in lengths:
        result.append(slice(start, start+n)); start += n
    return result


def hc_rows(module, method, tensors, sections):
    if getattr(module, 'fused_rows', None) and module.fused_rows(sections[-1].stop):
        # Every row keeps its own 16-row tile arithmetic; one weight traversal serves all lanes.
        return getattr(module, method)(*tensors, preserve_rows=True)
    # Preserve the screened per-sequence HC projection reduction layout.
    rows = [getattr(module, method)(*[t[s] for t in tensors], preserve_rows=True) for s in sections]
    return tuple(torch.cat([r[i] for r in rows]) if rows[0][i] is not None else None
                 for i in range(len(rows[0])))


DRAFT_BATCH = os.environ.get('SPARK_DRAFT_BATCH', '0') == '1'
# Graph layouts with a deep (> 3 row, DFlash) lane hold per-row GDN snapshots for every lane: 1-2.4 GB each
# (2026-10-21). Off by default: those layouts run eagerly (GPU-bound steps hide most launch time).
DEEP_GRAPH = os.environ.get('SPARK_BATCH_DEEP_GRAPH', '0') == '1'
LAZY_CLONE = os.environ.get('SPARK_LAZY_CLONE', '0') == '1'


def materialize_views(state):
    # Detach a committed EngineState from batch-graph replay buffers (exact copies).
    if not LAZY_CLONE:
        return state
    for layer in getattr(state, 'layers', ()):
        attention = getattr(layer, 'attention', None)
        if getattr(attention, 'graph_view', False):
            layer.attention = GDNState(attention.conv.clone(), attention.recurrent.clone())
    return state


def gdn_qkvz(attention, x):
    # Opt-in NVFP4 routing lives on GDNRef; plain modules keep the original FP8 call.
    if hasattr(attention, 'qkvz_linear'):
        return attention.qkvz_linear(x)
    return linear(x, attention.qkvz, attention.qkvz_scale, attention.activation_backend)


def gdn_out(attention, x):
    if hasattr(attention, 'out_linear'):
        return attention.out_linear(x)
    return linear(x, attention.out, attention.out_scale, attention.activation_backend)


ROW_GEMV = os.environ.get('SPARK_ROW_GEMV', '0') == '1'


def row_linear(x, weight):
    # Tiny per-row projections (GDN ba, shared-expert gate) keep one M=1 call per row.
    # A strided-batched cuBLAS call matched these bits eagerly but not under graph
    # capture (GPU A/B 2026-10-19), so it is deliberately not used. The opt-in Triton
    # kernel reproduces the M=1 cuBLAS GEMV order for every row in one weight traversal.
    if ROW_GEMV and fused_row_shape(x, weight):
        from .ops.row_gemv import rows_linear
        return rows_linear(x, weight)
    return torch.cat([F.linear(row, weight) for row in x.split(1)])


def fused_row_shape(x, weight):
    from .ops.row_gemv import VERIFIED, MIN_ROWS
    return (tuple(weight.shape) in VERIFIED and MIN_ROWS[weight.shape[0]] <= x.shape[0] <= 16
            and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16 and weight.is_contiguous())


def route_rows(moe, x, sections):
    # Per-lane routes; with ROW_GEMV the router logits of all lanes come from one bit-equal traversal.
    if ROW_GEMV and moe.router_backend == 'native' and fused_row_shape(x, moe.router):
        from .ops.native_kernels import topk_softmax
        from .ops.row_gemv import rows_linear
        logits = rows_linear(x, moe.router)
        return [topk_softmax(logits[s], moe.top_k, moe.renormalize)[:2] for s in sections]
    return [moe.route(x[s], preserve_rows=True)[:2] for s in sections]


def head_rows(head, hidden):
    # Packed INT8 heads take up to ROW_LIMIT rows per traversal (3, or 12 when opted in).
    from .ops.int8_head import ROW_LIMIT
    return torch.cat([head(rows.contiguous()) for rows in hidden.split(ROW_LIMIT)])


def target_moe_rows(moe, x, sections):
    if getattr(moe, 'expert_backend', None) != 'flashinfer':
        # CPU/reference models keep their ordinary expert implementation.
        return moe(x, preserve_rows=True)
    routes = route_rows(moe, x, sections)
    weights, ids = [torch.cat([r[i] for r in routes]) for i in (0, 1)]
    routed = moe.packed_forward(x, weights, ids)
    gate_up = linear(x, moe.shared_up, moe.shared_up_scale, moe.activation_backend)
    native = moe.resident_experts and moe.activation_backend == 'sm121-native'
    if native:
        from .ops.shared_epilogue import activation, finalize
        activated = torch.cat([activation(rows.contiguous()) for rows in gate_up.split(3)])
    else:
        gate, up = gate_up.chunk(2, -1)
        activated = F.silu(gate.float()).to(x.dtype)*up
    shared = linear(activated, moe.shared_down, moe.shared_down_scale, moe.activation_backend)
    from .ops.moe import SHARED_GATE_BATCH
    gates = F.linear(x, moe.shared_gate) if SHARED_GATE_BATCH else row_linear(x, moe.shared_gate)
    if native:
        return torch.cat([finalize(a.contiguous(), b.contiguous(), c.contiguous())
                          for a, b, c in zip(routed.split(3), shared.split(3), gates.split(3))])
    return routed+shared*torch.sigmoid(gates)


def qsa_rows(attention, x, positions, states, sections, capture):
    q, k, v, gate = attention.project(x, positions)
    packed = pack_kv(k, v)
    cores, updated, prefixes = [], [], []
    for previous, section in zip(states, sections):
        if previous is None:
            raise ValueError('Batch QSA requires prefilled private state')
        indexed = attention.index(x[section], positions[section], previous,
                                  position_start=previous.length, return_prefixes=capture)
        indices, state = indexed[:2]
        state.packed = append_kv(previous.packed, packed[section], attention.kv_backend)
        if attention.attention_backend == 'triton':
            from .ops.qsa_triton import sparse_attention as kernel
            # verify=capture: a verified lane keeps the ordinary decode split boundaries up to VERIFY_ROWS
            # rows, as single-lane verify_tokens does (lanes > 3 rows otherwise took the wide-tail profile).
            core = kernel(q[section].contiguous(), state.packed, indices,
                          prefill_mode=getattr(attention, 'prefill_mode', 'packed'), verify=capture)
        else:
            core = sparse_attention(q[section], state.packed, indices)
        cores.append(core); updated.append(state)
        if capture:
            lane = indexed[2]
            for i, prefix in enumerate(lane, 1):
                prefix.packed = state.packed[:previous.length+i]
            prefixes.append(lane)
    gated = torch.cat(cores).flatten(1)*torch.sigmoid(gate)
    output = (F.linear(gated, attention.out) if attention.out_scale is None else
              linear(gated, attention.out, attention.out_scale, attention.activation_backend))
    return output, updated, prefixes


class BatchMTPModel:
    def __init__(self, engine, *, graph=True, max_graph_layouts=9):
        if type(max_graph_layouts) is not int or not 1 <= max_graph_layouts <= 9:
            raise ValueError('Expected one to nine retained batch/time graph layouts')
        self.engine = engine
        self.graph = bool(graph and engine.graph_decode)
        self.graphs = {}; self.pools = {}
        self.layouts = {}; self.max_graph_layouts = max_graph_layouts
        self.eager_layout_fallbacks = 0
        self.calls = self.token_rows = 0

    def snapshot(self):
        return dict(calls=self.calls, token_rows=self.token_rows, graph_enabled=self.graph,
                    graph_blocks=len(self.graphs), graph_replays=sum(g.replays for g in self.graphs.values()),
                    max_batch=4, max_time=3, head_row_limit=3,
                    retained_graph_layouts=len(self.layouts), max_graph_layouts=self.max_graph_layouts,
                    eager_layout_fallbacks=self.eager_layout_fallbacks,
                    new_moe_rows='vendor_nonfused_heuristic_requires_gpu_acceptance')

    def layer(self, index, pending, states, sections, positions=None, ids=None, contexts=None):
        layer = self.engine.layers[index]
        hidden, block, injection = pending
        ple = [s.ple for s in states]; ple_prefixes = [None]*len(states)
        if layer.ple is not None:
            if block is not None:
                hidden = layer.attn_hc.combine(hidden, block, injection); block = injection = None
            enhancements = []
            for lane, section in enumerate(sections):
                value, ple[lane], ple_prefixes[lane] = layer.ple(
                    hidden[section], np.asarray(ids[lane], dtype=np.int64), [0, section.stop-section.start],
                    contexts[lane], ple[lane], True)
                enhancements.append(value)
            hidden = hidden+torch.cat(enhancements)
        if block is None:
            residual, mixed, injection = hc_rows(layer.attn_hc, 'mix', [hidden], sections)
        else:
            residual, mixed, injection = hc_rows(layer.attn_hc, 'combine_and_mix', [hidden, block, injection], sections)
        attention = layer.attention
        if layer.layer_type == 'linear_attention':
            # Tiny BA gate arithmetic stays per-sequence; qkvz/out are shared.
            projected = gdn_qkvz(attention, mixed)
            ba = row_linear(mixed, attention.ba)
            cores = [attention.run_core(projected[s], ba[s], state.attention,
                                       prefill=False, return_prefixes=True) for s, state in zip(sections, states)]
            attended = gdn_out(attention, torch.cat([c[0] for c in cores]))
            histories, prefixes = [c[1] for c in cores], [c[2] for c in cores]
        else:
            attended, histories, prefixes = qsa_rows(attention, mixed, positions,
                                                    [s.attention for s in states], sections, True)
        residual, mixed, injection = hc_rows(layer.mlp_hc, 'combine_and_mix',
                                           [residual, attended, injection], sections)
        output = target_moe_rows(layer.moe, mixed, sections)
        updated = []
        for lane, state in enumerate(states):
            final = DecoderState(histories[lane], ple[lane])
            final.prefixes = [DecoderState(att, ple_prefixes[lane][i] if ple_prefixes[lane] is not None else state.ple)
                              for i, att in enumerate(prefixes[lane])]
            updated.append(final)
        return (residual, output, injection), updated

    @torch.no_grad()
    def verify(self, input_ids, states, observer=None):
        """observer(layer_index, (hidden, block, injection)) sees every layer's output rows (all lanes,
        concatenated in lane order), as the single-lane engine's observer does."""
        lengths = [len(ids) for ids in input_ids]; sections = spans(lengths)
        if len(states) != len(sections) or len({id(s) for s in states}) != len(states):
            raise ValueError('Batch requires distinct private states')
        cfg = self.engine.config
        starts = [len(s.tokens) for s in states]
        if any(not s.tokens or len(s.layers) != len(self.engine.layers) for s in states):
            raise ValueError('Batch requires prefilled layer states')
        if any(start+n > cfg['max_position_embeddings'] for start, n in zip(starts, lengths)):
            raise ValueError('Context exceeds actual checkpoint override')
        if any(type(t) is not int or not 0 <= t < cfg['vocab_size'] for ids in input_ids for t in ids):
            raise ValueError('Token is outside checkpoint vocabulary')
        device = self.engine.embedding.device
        positions = torch.tensor([p for start, n in zip(starts, lengths) for p in range(start, start+n)], device=device)
        tokens = torch.tensor([t for ids in input_ids for t in ids], device=device)
        hidden = self.engine.embedding[tokens].repeat(1, cfg['hc_count']); pending = (hidden, None, None)
        preceding = cfg['ngram_size']-1
        contexts = [np.asarray([([cfg['eos_token_id']]*preceding+s.tokens)[-preceding:] if preceding else []], dtype=np.int64)
                    for s in states]
        updated = [EngineState(list(s.layers), list(s.tokens)+list(ids)) for s, ids in zip(states, input_ids)]
        layout = tuple(lengths)
        use_graph = (self.graph and (layout in self.layouts or len(self.layouts) < self.max_graph_layouts)
                     and (DEEP_GRAPH or max(lengths) <= 3))
        if use_graph:
            # Vendor operators can retain captured workspace pointers outside
            # Torch tensors. Never evict a live graph pool on shape changes.
            # The nine uniform B2/B3/B4 x T1/T2/T3 layouts fit the fixed bound;
            # additional ragged layouts share operators in eager mode.
            self.layouts[layout] = None
        elif self.graph:
            self.eager_layout_fallbacks += 1
        index = 0
        while index < len(self.engine.layers):
            if use_graph and index in self.engine._graph_ranges:
                end = self.engine._graph_ranges[index]; key = (index, tuple(lengths))
                if key not in self.graphs:
                    # Different batch/time layouts never borrow each other's banks.
                    if layout not in self.pools:
                        self.pools[layout] = shared_pool() or torch.cuda.graph_pool_handle()
                    pool = self.pools[layout]
                    self.graphs[key] = BatchMTPGraph(self, list(range(index, end)), sections, pool)
                graph = self.graphs[key]
                pending, next_states = graph(pending, [s.layers[index:end] for s in states])
                for lane, result in enumerate(next_states):
                    updated[lane].layers[index:end] = result
                if observer is not None:
                    for offset, value in enumerate(graph.intermediates): observer(index+offset, value)
                index = end
            else:
                pending, next_states = self.layer(index, pending, [s.layers[index] for s in states],
                                                  sections, positions, input_ids, contexts)
                for lane, state in enumerate(next_states):
                    updated[lane].layers[index] = state
                if observer is not None: observer(index, pending)
                index += 1
        multi, sampled, _ = hc_rows(self.engine.final_mixer, 'combine_and_mix', list(pending), sections)
        logits = head_rows(self.engine.logits, sampled)
        outputs = []
        for lane, section in enumerate(sections):
            state = updated[lane]
            state.prefixes = [EngineState([layer.prefixes[i] for layer in state.layers],
                                         state.tokens[:starts[lane]+i+1]) for i in range(lengths[lane])]
            outputs.append((logits[section], state, ForwardHidden(multi[section].clone(), sampled[section].clone())))
        self.calls += 1; self.token_rows += sum(lengths)
        return outputs


class BatchMTPDraft:
    def __init__(self, draft):
        self.draft = draft; self.calls = 0

    @torch.no_grad()
    def forward(self, input_ids, multi, states):
        sections = spans([len(ids) for ids in input_ids]); draft = self.draft
        if len(states) != len(sections) or len(multi) != len(sections):
            raise ValueError('Draft lane counts differ')
        if len({id(s) for s in states}) != len(states):
            raise ValueError('Draft lanes require distinct private states')
        if any(type(t) is not int or not 0 <= t < draft.config['vocab_size'] for ids in input_ids for t in ids):
            raise ValueError('Draft token is outside checkpoint vocabulary')
        if any(s is None or s.length+len(ids) > draft.config['max_position_embeddings'] for s, ids in zip(states, input_ids)):
            raise ValueError('Draft requires prefilled states within checkpoint context')
        if DRAFT_BATCH:
            # Draft arithmetic only steers proposals; the target verifies every token,
            # so one shared traversal of the BF16 draft weights serves every lane.
            hidden = draft.fuse_inputs([t for ids in input_ids for t in ids], torch.cat(list(multi)))
        else:
            # Keep the BF16 fusion's screened per-sequence reduction arithmetic.
            hidden = torch.cat([draft.fuse_inputs(ids, value) for ids, value in zip(input_ids, multi)])
        positions = torch.tensor([p for s, ids in zip(states, input_ids) for p in range(s.length, s.length+len(ids))],
                                 device=hidden.device)
        residual, mixed, injection = hc_rows(draft.attn_hc, 'mix', [hidden], sections)
        attention, histories, _ = qsa_rows(draft.attention, mixed, positions, states, sections, False)
        residual, mixed, injection = hc_rows(draft.mlp_hc, 'combine_and_mix', [residual, attention, injection], sections)
        if DRAFT_BATCH:
            weights, ids, _ = draft.moe.route(mixed)
            block = draft.moe.routed(mixed, weights, ids)+draft.moe.shared(mixed)
        else:
            routes = [draft.moe.route(mixed[s]) for s in sections]
            weights, ids = [torch.cat([r[i] for r in routes]) for i in (0, 1)]
            # Share the BF16 expert bank traversal; preserve the small shared path.
            block = draft.moe.routed(mixed, weights, ids)+torch.cat([draft.moe.shared(mixed[s]) for s in sections])
        multi, sampled, _ = hc_rows(draft.final_mixer, 'combine_and_mix', [residual, block, injection], sections)
        last = torch.cat([sampled[s.stop-1:s.stop] for s in sections])
        logits = head_rows(draft.head, last)
        self.calls += 1
        return [MTPOutput(logits[i:i+1], histories[i], multi[s].clone(), sampled[s].clone(),
                          getattr(draft.head, 'vocab_ids', None)) for i, s in enumerate(sections)]


class BatchMTPGraph:
    """Bounded GDN runs with owned per-lane, per-time speculative snapshots."""
    def __init__(self, batch, indices, sections, pool):
        if any(batch.engine.layers[i].layer_type != 'linear_attention' or batch.engine.layers[i].ple is not None for i in indices):
            raise ValueError('Batch graphs contain GDN-only layers without PLE')
        self.batch, self.indices, self.sections, self.pool = batch, indices, sections, pool
        self.graph = None; self.replays = 0; self.intermediates = []

    def copy_inputs(self, pending, states):
        for value, destination in zip(pending, self.pending):
            if (value is None) != (destination is None):
                raise ValueError('Graph pending structure changed')
            if value is not None:
                if value.shape != destination.shape or value.dtype != destination.dtype:
                    raise ValueError('Graph pending tensor layout changed')
                destination.copy_(value)
        for lane, layers in enumerate(states):
            for i, state in enumerate(layers):
                if state.ple is not None:
                    raise ValueError('PLE cannot enter a GDN graph')
                if any(a.shape!=b.shape or a.dtype!=b.dtype for a,b in
                       ((state.attention.conv,self.conv[i,lane]),(state.attention.recurrent,self.recurrent[i,lane]))):
                    raise ValueError('Graph private state tensor layout changed')
                self.conv[i, lane].copy_(state.attention.conv)
                self.recurrent[i, lane].copy_(state.attention.recurrent)

    def run(self):
        pending = self.pending; updated = []; intermediates = []
        for i, index in enumerate(self.indices):
            pending, next_states = self.batch.layer(index, pending, [s[i] for s in self.states], self.sections)
            updated.append(next_states); intermediates.append(pending)
        # Replay-owned layer outputs (read before the next replay of this graph, like the single-lane block).
        self.intermediates = intermediates
        return pending, updated

    def capture(self, pending, states):
        self.pending = tuple(x.clone() if x is not None else None for x in pending)
        self.conv = torch.stack([torch.stack([s[i].attention.conv for s in states]) for i in range(len(self.indices))])
        self.recurrent = torch.stack([torch.stack([s[i].attention.recurrent for s in states]) for i in range(len(self.indices))])
        self.states = [[DecoderState(GDNState(self.conv[i, lane], self.recurrent[i, lane]))
                        for i in range(len(self.indices))] for lane in range(len(states))]
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.copy_inputs(pending, states); self.run()
        torch.cuda.current_stream().wait_stream(stream)
        self.copy_inputs(pending, states); torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with capture_graph(self.graph, pool=self.pool, stream=stream):
            self.output, self.updated = self.run()
        self.copy_inputs(pending, states)

    def __call__(self, pending, states):
        if self.graph is None:
            self.capture(pending, states)
        else:
            self.copy_inputs(pending, states)
        self.graph.replay(); self.replays += 1
        snapshots = []
        for lane in range(len(states)):
            layers = []
            for i in range(len(self.indices)):
                if LAZY_CLONE:
                    # Replay-buffer views; the runner clones only the prefix it commits
                    # (materialize_views) before any later replay can overwrite them.
                    prefixes = []
                    for p in self.updated[i][lane].prefixes:
                        view = GDNState(p.attention.conv, p.attention.recurrent); view.graph_view = True
                        prefixes.append(DecoderState(view))
                else:
                    # Clones detach from replay buffers and from every peer lane.
                    prefixes = [DecoderState(GDNState(p.attention.conv.clone(), p.attention.recurrent.clone()))
                                for p in self.updated[i][lane].prefixes]
                final = DecoderState(prefixes[-1].attention, prefixes=prefixes)
                layers.append(final)
            snapshots.append(layers)
        return self.output, snapshots
