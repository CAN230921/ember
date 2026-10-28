"""Bounded exact-token evidence of a repeated reasoning cycle.

Four nonoverlapping equal 32-grams alone also occur in legitimate repeated
templates. Require equally spaced occurrences and exact periodicity across
their intervening tokens. Four nonoverlapping copies of an identical longer
96-token block also qualify, even when the surrounding prose varies. This
detects exact repetition, not semantic loops.
The caller supplies reasoning tokens only and resets at section boundaries.
"""
from collections import deque


class ReasoningRepetitionDetector:
    def __init__(self, *, gram_tokens=32, repeats=4, min_tokens=512, window_tokens=2048):
        settings=(gram_tokens,repeats,min_tokens,window_tokens)
        if any(type(value) is not int for value in settings):
            raise ValueError('Expected integer repetition thresholds')
        if (gram_tokens < 32 or repeats < 4 or min_tokens < 512 or
                window_tokens not in (2048,4096) or gram_tokens*repeats > window_tokens):
            raise ValueError('Expected conservative bounded reasoning repetition thresholds')
        self.gram_tokens,self.repeats = gram_tokens,repeats
        self.min_tokens,self.window_tokens = min_tokens,window_tokens
        self.reset()

    def reset(self):
        self.seen=0;self.triggered=False;self.period_tokens=None;self.reason=None
        self._tokens=deque(maxlen=self.window_tokens)
        self._tail=deque(maxlen=self.gram_tokens)
        self._grams=deque()
        self._positions={}
        self._long_tail=deque(maxlen=96)
        self._long_grams=deque()
        self._long_positions={}

    def observe(self, token):
        if type(token) is not int or token < 0:
            raise ValueError('Expected a nonnegative token ID')
        if self.triggered:
            return True
        self._tokens.append(token);self._tail.append(token);self._long_tail.append(token);self.seen+=1
        start=self.seen-self.gram_tokens
        oldest=max(0,self.seen-self.window_tokens)
        while self._grams and self._grams[0][0] < oldest:
            expired,key=self._grams.popleft()
            positions=self._positions[key]
            if positions.popleft()!=expired:
                raise RuntimeError('Repetition window metadata diverged')
            if not positions:del self._positions[key]
        if start < 0:
            return False
        # Fixed-width suffix extraction and exact tuple keys avoid probabilistic
        # hash equality or unbounded copies of the generated token history.
        key=tuple(self._tail)
        positions=self._positions.setdefault(key,deque())
        positions.append(start);self._grams.append((start,key))
        while self._long_grams and self._long_grams[0][0] < oldest:
            expired,long_key=self._long_grams.popleft()
            long_positions=self._long_positions[long_key]
            if long_positions.popleft()!=expired:
                raise RuntimeError('Long repetition window metadata diverged')
            if not long_positions:del self._long_positions[long_key]
        if self.seen >= 96:
            long_key=tuple(self._long_tail)
            long_positions=self._long_positions.setdefault(long_key,deque())
            long_start=self.seen-96
            long_positions.append(long_start);self._long_grams.append((long_start,long_key))
            if self.seen >= self.min_tokens:
                count=0;last=None
                for position in long_positions:
                    if last is None or position-last >= 96:
                        count+=1;last=position
                        if count >= self.repeats:
                            self.triggered=True;self.reason='repeated_reasoning_block'
                            return True
        if self.seen < self.min_tokens or len(positions) < self.repeats:
            return False
        available=set(positions)
        history=None
        for previous in reversed(positions):
            period=start-previous
            if period < self.gram_tokens:
                continue
            first=start-(self.repeats-1)*period
            if first < oldest:
                break
            if not all(start-index*period in available for index in range(1,self.repeats)):
                continue
            if history is None:history=list(self._tokens)
            segment=history[first-oldest:start+self.gram_tokens-oldest]
            if segment[period:] == segment[:-period]:
                self.triggered=True;self.period_tokens=period;self.reason='repeated_reasoning_cycle'
                return True
        return False

    def snapshot(self):
        return dict(gram_tokens=self.gram_tokens,repeats=self.repeats,
            min_tokens=self.min_tokens,window_tokens=self.window_tokens,
            buffered_tokens=len(self._tokens),indexed_grams=len(self._grams),
            distinct_grams=len(self._positions),triggered=self.triggered,
            period_tokens=self.period_tokens,reason=self.reason,
            long_block_tokens=96,indexed_long_blocks=len(self._long_grams),
            distinct_long_blocks=len(self._long_positions))
