"""Admission reservations; KV pages themselves remain allocated on demand.

Reserve the whole input + output contract before model state is created. A
reservation survives cancellation until the model-owner thread drops that
request's state. Prefix-cache hits do not discount the contract: shared pages
can diverge on append. This pool accounts for tokens, not device memory.
"""
from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class ContextReservation:
    input_tokens: int
    output_tokens: int

    @property
    def tokens(self):
        return self.input_tokens + self.output_tokens


class ContextPool:
    def __init__(self, total_tokens, per_request_tokens):
        for value in (total_tokens, per_request_tokens):
            if type(value) is not int or value <= 0:
                raise ValueError('Context capacities must be positive integers')
        self.total_tokens = total_tokens
        self.per_request_tokens = per_request_tokens
        self._lock = threading.Lock()
        self._reservations = {}
        self._reserved = 0
        self._peak = 0

    def validate(self, input_tokens, output_tokens):
        if (type(input_tokens) is not int or input_tokens < 0 or
                type(output_tokens) is not int or output_tokens < 1):
            raise ValueError('Expected nonnegative input and positive output tokens')
        demand = input_tokens + output_tokens
        if demand > self.per_request_tokens:
            raise ValueError('Prompt plus output budget exceeds the per-request context')
        if demand > self.total_tokens:
            raise ValueError('Prompt plus output budget exceeds the configured shared context pool')
        return ContextReservation(input_tokens, output_tokens)

    def try_reserve(self, request_id, input_tokens, output_tokens):
        demand = self.validate(input_tokens, output_tokens)
        with self._lock:
            existing = self._reservations.get(request_id)
            if existing is not None:
                if existing != demand:
                    raise ValueError('A live context reservation cannot be resized')
                return True
            if self._reserved + demand.tokens > self.total_tokens:
                return False
            self._reservations[request_id] = demand
            self._reserved += demand.tokens
            self._peak = max(self._peak, self._reserved)
            return True

    def release(self, request_id):
        with self._lock:
            demand = self._reservations.pop(request_id, None)
            if demand is None:
                return False
            self._reserved -= demand.tokens
            return True

    def snapshot(self):
        with self._lock:
            return dict(total_tokens=self.total_tokens, reserved_tokens=self._reserved,
                available_tokens=self.total_tokens-self._reserved,
                peak_reserved_tokens=self._peak, reservation_count=len(self._reservations),
                allocation_policy='input_plus_output_reservation',
                kv_allocation='on_demand', accounting_unit='tokens',
                reservations={key:dict(input_tokens=value.input_tokens,
                    output_tokens=value.output_tokens, reserved_tokens=value.tokens)
                    for key,value in self._reservations.items()})
