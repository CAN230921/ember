"""Per-request token allowance for an already-open checkpoint think section."""
from repetition_control import ReasoningRepetitionDetector


class ReasoningControl:
    def __init__(self, enabled, effort, budget, closing_ids, *, opening_ids=()):
        if type(enabled) is not bool or (budget is not None and (type(budget) is not int or budget < 1)):
            raise ValueError('Invalid reasoning control settings')
        if (enabled and not closing_ids) or any(type(t) is not int or t < 0 for t in closing_ids):
            raise ValueError('Expected tokenized closing think marker')
        if any(type(t) is not int or t < 0 for t in opening_ids):
            raise ValueError('Expected tokenized opening think marker')
        self.enabled, self.effort, self.budget = enabled, effort, budget
        self.closing_ids = tuple(closing_ids)
        self.opening_ids = tuple(opening_ids)
        if self.opening_ids and self.opening_ids == self.closing_ids:
            raise ValueError('Opening and closing think markers must differ')
        self.active, self.seen, self.used = enabled, 0, 0
        self.pending, self.tail = [], []
        self.triggered, self.forced = False, 0
        self.repetition = ReasoningRepetitionDetector() if enabled else None
        self.loop_triggered,self.loop_reason,self.trigger_at_tokens = False,None,None
        self.loop_period_tokens = None
        self.reasoning_reopened,self.termination_reason = False,None
        self.intercepted_reasoning_tokens = 0
        self.marker_tail = []

    def plan(self):
        if self.pending:
            return self.pending[0], None
        if self.active:
            remaining = self.budget-self.used if self.budget is not None else None
            if (remaining is not None and remaining <= 0) or self.loop_triggered:
                if remaining is not None and remaining <= 0:
                    self.triggered = True
                # A multi-token marker may already have started naturally at
                # the boundary. Commit only its missing suffix.
                overlap = 0
                for size in range(1, len(self.closing_ids)):
                    if tuple(self.tail[-size:]) == self.closing_ids[:size]:
                        overlap = size
                self.pending = list(self.closing_ids[overlap:])
                return self.pending[0], None
            if remaining is not None:
                return None, remaining
        return None, None

    def observe(self, generated):
        if len(generated) < self.seen:
            raise ValueError('Reasoning token history must be monotonic')
        for token in generated[self.seen:]:
            self.marker_tail.append(token)
            self.marker_tail = self.marker_tail[-max(len(self.closing_ids),len(self.opening_ids),1):]
            if self.pending:
                if token != self.pending.pop(0):
                    raise RuntimeError('Committed control token differs from its plan')
                self.forced += 1
                if not self.pending:
                    self.active = False
                    self.repetition.reset()
            elif self.active:
                self.used += 1
                self.tail.append(token)
                self.tail = self.tail[-len(self.closing_ids):]
                if tuple(self.tail) == self.closing_ids:
                    self.active = False
                    self.repetition.reset()
                elif self.repetition.observe(token) and not self.loop_triggered:
                    self.loop_triggered=True
                    self.loop_reason=self.repetition.reason
                    self.trigger_at_tokens=self.used
                    self.loop_period_tokens=self.repetition.period_tokens
            elif self.enabled:
                if self.reasoning_reopened:
                    # These tokens were already committed by the same MTP
                    # round. The caller must stop at its after_step boundary;
                    # do not recount them as allowance-compliant reasoning.
                    self.intercepted_reasoning_tokens += 1
                elif (self.opening_ids and
                        tuple(self.marker_tail[-len(self.opening_ids):])==self.opening_ids):
                    self.reasoning_reopened=True
                    self.termination_reason='reasoning_reopened'
        self.seen = len(generated)
        if self.budget is not None and self.used > self.budget:
            raise RuntimeError('Reasoning token allowance was exceeded')

    def snapshot(self):
        return dict(enabled=self.enabled, effort=self.effort, budget_tokens=self.budget,
            used_tokens=self.used, budget_triggered=self.triggered,
            forced_control_tokens=self.forced,loop_triggered=self.loop_triggered,
            reason=self.loop_reason,trigger_at_tokens=self.trigger_at_tokens,
            loop_period_tokens=self.loop_period_tokens,
            reasoning_reopened=self.reasoning_reopened,termination_reason=self.termination_reason,
            intercepted_reasoning_tokens=self.intercepted_reasoning_tokens)
