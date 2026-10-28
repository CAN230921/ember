"""Cancellation at completed GPU work boundaries."""


class GenerationCancelled(Exception):
    def __init__(self, phase='request', completed_prefill_tokens=0, checkpoint_tokens=0):
        super().__init__('Generation cancelled')
        self.phase = phase
        self.completed_prefill_tokens = completed_prefill_tokens
        self.checkpoint_tokens = checkpoint_tokens
