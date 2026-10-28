"""Per-request draft depth from observed first-proposal acceptance on GB10."""
from collections import deque
import math
import os


class AdaptiveDraftDepth:
    def __init__(self):
        self.recent=deque(maxlen=8)

    def depth(self):
        # Start with the measured faster depth and retain it for short,
        # predictable outputs. After four observations, avoid the recursive
        # draft when the first proposal itself succeeds less than 75%.
        return 2 if len(self.recent)<4 or 4*sum(self.recent)>=3*len(self.recent) else 1

    def observe(self,first_accepted):
        self.recent.append(bool(first_accepted))


class CostAwareDraftDepth:
    """Opt-in per-request choice using committed tokens / measured round time.

    Inspired by Strata's measured draft-window policy, without its CPU/GPU
    cost priors or lookup drafter. Only our verified one/two-draft shapes are
    eligible. A round includes draft, target verification and state commit,
    but excludes time waiting for another request. Timings must represent
    completed work, not asynchronous kernel enqueue time.
    """
    def __init__(self, *, window=8, probes=2, revisit=16, margin=.03):
        if window < 1 or not 1 <= probes <= window or revisit < 1:
            raise ValueError('Invalid draft policy observation bounds')
        if not math.isfinite(margin) or margin < 0:
            raise ValueError('Expected a finite nonnegative switch margin')
        self.samples = {depth: deque(maxlen=window) for depth in (1, 2)}
        self.counts = {1: 0, 2: 0}
        self.last_seen = {1: 0, 2: 0}
        self.probes, self.revisit, self.margin = probes, revisit, margin
        self.rounds, self.current = 0, 2

    def rate(self, depth):
        samples = self.samples[depth]
        return (sum(tokens for tokens, _ in samples) /
                sum(seconds for _, seconds in samples)) if samples else None

    def depth(self):
        # Preserve the established initial depth, then collect bounded samples
        # of both shapes. Calling depth() alone never advances exploration.
        for depth in (2, 1):
            if self.counts[depth] < self.probes:
                return depth
        other = 3 - self.current
        if self.rounds - self.last_seen[other] >= self.revisit:
            return other
        if self.rate(other) > self.rate(self.current) * (1 + self.margin):
            self.current = other
        return self.current

    def observe_round(self, first_accepted, *, depth, accepted_tokens, elapsed_seconds):
        # first_accepted keeps the observer protocol shared with acceptance-
        # only policies. Reward includes the correction/bonus actually emitted.
        if depth not in (1, 2) or not 1 <= accepted_tokens <= depth + 1:
            raise ValueError('Invalid completed draft round')
        if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
            raise ValueError('Draft costs require positive completed-round time')
        self.rounds += 1
        self.samples[depth].append((accepted_tokens, elapsed_seconds))
        self.counts[depth] += 1
        self.last_seen[depth] = self.rounds

    def snapshot(self):
        return dict(strategy='committed_tokens_per_second', rounds=self.rounds,
                    observations=dict(self.counts),
                    tokens_per_second={d: self.rate(d) for d in (1, 2)})


class DeepDraftDepth:
    """Opt-in single-lane depth 1..max_depth (SPARK_VERIFY_ROWS > 3).

    Tracks the conditional acceptance of each draft position (position k is attempted only when the
    k-1 earlier drafts were accepted) with exponential decay and a prior centred on the previous position's estimate, and picks the depth
    maximizing expected committed tokens 1 + p1 + p1*p2 + ... per relative round cost
    1 + row_cost*(d-1) (row_cost 0.18-0.2 measured on GB10 top10). Starts at depth 2. Only the
    verification shape changes; every token is still checked by the target.
    """
    def __init__(self, max_depth, *, row_cost=0.2, decay=0.95, warmup=4):
        if not 1 <= max_depth <= 7 or not row_cost >= 0 or not 0 < decay <= 1:
            raise ValueError('Invalid deep draft policy bounds')
        self.max_depth, self.row_cost, self.decay, self.warmup = max_depth, row_cost, decay, warmup
        self.tried = [0.0]*max_depth; self.hits = [0.0]*max_depth; self.rounds = 0

    def acceptance(self):
        # Beta prior of weight 2 centred on the previous position (0.5 for the first), so a position
        # that has not been tried yet inherits its predecessor's estimate and gets explored.
        out, prior = [], 0.5
        for h, t in zip(self.hits, self.tried):
            prior = (h+2.0*prior)/(t+2.0); out.append(prior)
        return out

    def depth(self):
        if self.rounds < self.warmup:
            return min(2, self.max_depth)
        best = (-1.0, 1); expected = 1.0; chain = 1.0
        for d, p in enumerate(self.acceptance(), 1):
            chain *= p; expected += chain
            value = expected/(1+self.row_cost*(d-1))
            if value > best[0]+1e-12:
                best = (value, d)
        return best[1]

    def observe_counts(self, accepted, rejected, proposed=None):
        # accepted: matched drafts; rejected: whether a draft was refused; proposed: drafts sent.
        proposed = accepted + (1 if rejected else 0) if proposed is None else proposed
        self.rounds += 1
        for k in range(self.max_depth):
            self.tried[k] *= self.decay; self.hits[k] *= self.decay
        for k in range(min(proposed, accepted+1, self.max_depth)):
            self.tried[k] += 1.0
            if k < accepted: self.hits[k] += 1.0

    def snapshot(self):
        return dict(strategy='deep_positional', max_depth=self.max_depth, row_cost=self.row_cost,
                    acceptance=self.acceptance(), depth=self.depth())


class DraftSourceChooser:
    """Per-request choice between the MTP chain and the DFlash block drafter (draft-only).

    Deterministic cost model instead of wall clocks (graph capture / JIT made timing-based choices flip):
      step cost = verify(rows) + draft cost, verify(rows) = 1 + VERIFY_ROW_COST*(rows-1)   (one-row step = 1)
      MTP draft cost = MTP_DRAFT_COST per chained draft beyond the first; DFlash = DFLASH_COST per proposal.
    Measured on GB10 2026-10-21: verify 1/2/5/8 rows 39.6/46.5/64.6/79.1 ms, MTP draft ~2.3 ms, DFlash B8
    propose ~10.4 ms BF16 / ~3 ms NVFP4 fused. Rates = emitted tokens / modelled cost over each source's
    last `window` steps; MTP catch-ups are charged to DFlash. Each source gets `warmup` steps; afterwards
    DFlash is used only while it leads MTP by SWITCH_MARGIN, and the other source is re-probed every
    `explore` steps (every slow*explore while the better one leads by more than `margin`).
    """
    VERIFY_ROW_COST = float(os.environ.get('SPARK_CHOOSER_ROW_COST', '0.14'))
    MTP_DRAFT_COST = float(os.environ.get('SPARK_CHOOSER_MTP_DRAFT_COST', '0.058'))
    # NVFP4 + fused drafter: ~3 ms per B8 proposal on a ~40 ms one-row step.
    DFLASH_COST = float(os.environ.get('SPARK_CHOOSER_DFLASH_COST', '0.07'))
    # MTP catch-up after DFlash steps (one prefill-mode draft over the backlog rows): fixed + per-row cost,
    # charged to DFlash, whose steps left MTP behind.
    CATCH_UP_COST = float(os.environ.get('SPARK_CHOOSER_CATCH_UP_COST', '0.06'))
    CATCH_UP_ROW_COST = float(os.environ.get('SPARK_CHOOSER_CATCH_UP_ROW_COST', '0.015'))
    # DFlash must lead MTP by this much to be preferred (hysteresis against near-ties, which pay catch-ups).
    SWITCH_MARGIN = float(os.environ.get('SPARK_CHOOSER_SWITCH_MARGIN', '0.10'))

    def __init__(self, warmup=2, explore=12, window=16, margin=0.15, slow=5, give_up=0.30):
        from collections import deque
        self.warmup, self.explore, self.margin, self.slow, self.give_up = warmup, explore, margin, slow, give_up
        self.obs = dict(mtp=deque(maxlen=window), dflash=deque(maxlen=window))
        self.n = dict(mtp=0, dflash=0); self.last = dict(mtp=0, dflash=0); self.step = 0

    def cost(self, source, rows, drafts, row_cost=None, dflash_cost=None, base=1.0):
        """base: this lane's share of the step's fixed cost (1 alone; 1/lanes in a shared multi-lane step,
        so a deep lane must pay for the time it adds to every peer)."""
        verify = base+(self.VERIFY_ROW_COST if row_cost is None else row_cost)*(rows-1)
        draft = ((self.DFLASH_COST if dflash_cost is None else dflash_cost) if source == 'dflash'
                 else self.MTP_DRAFT_COST*max(drafts-1, 0))
        return verify+draft

    def rate(self, source):
        q = self.obs[source]
        cost = sum(c for _, c in q)
        return sum(t for t, _ in q)/cost if cost > 0 else 0.0

    def preferred(self):
        if min(self.n.values()) < self.warmup: return None
        return 'dflash' if self.rate('dflash') > (1+self.SWITCH_MARGIN)*self.rate('mtp') else 'mtp'

    def hopeless(self):
        """DFlash has had its warm-up plus one probe and MTP still leads by more than give_up."""
        return (self.n['dflash'] >= self.warmup+1 and self.n['mtp'] >= self.warmup
                and self.rate('mtp') > (1+self.give_up)*self.rate('dflash'))

    def choose(self):
        self.step += 1
        forced = os.environ.get('SPARK_DFLASH_FORCE')      # diagnostics: 'mtp' or 'dflash'
        if forced in ('mtp', 'dflash'):
            self.last[forced] = self.step; return forced
        for source in ('dflash', 'mtp'):
            if self.n[source] < self.warmup:
                choice = source; break
        else:
            best = self.preferred(); other = 'mtp' if best == 'dflash' else 'dflash'
            lead = self.rate(best)/max(self.rate(other), 1e-9)-1
            interval = self.explore*(self.slow if lead > self.margin else 1)
            choice = other if self.step-self.last[other] >= interval else best
        self.last[choice] = self.step
        return choice

    def charge(self, source, seconds):
        """Kept for API compatibility (wall-clock charges are ignored: the model is deterministic)."""

    def charge_catch_up(self, rows):
        """Add the modelled cost of an MTP catch-up over `rows` backlog rows to the latest DFlash observation."""
        q = self.obs['dflash']
        if q:
            tokens, cost = q[-1]; q[-1] = (tokens, cost+self.CATCH_UP_COST+self.CATCH_UP_ROW_COST*rows)

    def observe(self, source, tokens, seconds=None, *, rows=None, drafts=None, row_cost=None, dflash_cost=None, base=1.0):
        """row_cost / dflash_cost / base: this step's relative costs (multi-lane steps pass their own)."""
        rows = rows if rows is not None else max(tokens, 1)
        drafts = drafts if drafts is not None else rows-1
        self.obs[source].append((tokens, self.cost(source, rows, drafts, row_cost, dflash_cost, base))); self.n[source] += 1
