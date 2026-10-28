"""Per-request output text; model token IDs and termination stay unchanged.

The validated Qwen ByteLevel backend uses its UTF-8-aware DecodeStream. Other
tokenizers retain the existing full-prefix behavior. Finishing decodes the
complete visible sequence once to preserve unfinished-byte replacement text.
"""


def _validated_stream(tokenizer):
    tokenizer_type = type(tokenizer)
    # The same frozen tokenizer loads as Qwen2Tokenizer with transformers
    # 5.15.1 and Qwen3_5Tokenizer with 5.18.0. Both inherit the raw backend
    # _decode path when clean_up_tokenization_spaces=False.
    supported = {
        ('Qwen2Tokenizer', 'transformers.models.qwen2.tokenization_qwen2'),
        ('Qwen3_5Tokenizer', 'transformers.models.qwen3_5.tokenization_qwen3_5'),
    }
    if ((tokenizer_type.__name__, tokenizer_type.__module__) not in supported or
            not getattr(tokenizer, 'is_fast', False)):
        return None, None
    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel, DecodeStream
    except ImportError:
        return None, None
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    if not isinstance(backend, Tokenizer) or not isinstance(backend.decoder, ByteLevel):
        return None, None
    try:
        stream = DecodeStream(skip_special_tokens=False)
    except (AttributeError, TypeError):
        return None, None
    if not callable(getattr(stream, 'step', None)):
        return None, None
    return stream, backend


class IncrementalOutputDecoder:
    """Accept newly committed IDs and return cumulative protocol text.

    Incremental failures cannot safely replay text through another decoder, so
    they fail the request explicitly. ``finish`` accepts unseen terminal IDs,
    but the already observed token sequence must remain their exact prefix.
    """

    def __init__(self, tokenizer, eos):
        self.tokenizer = tokenizer
        self.eos = frozenset(eos)
        self.raw = ''
        self._ids = []
        self._stream, self._backend = _validated_stream(tokenizer)
        self.incremental = self._stream is not None
        self._finished = self._failed = False

    def _check_live(self):
        if self._failed:
            raise RuntimeError('Output decoder previously failed')
        if self._finished:
            raise RuntimeError('Output decoder is already finished')

    def _decode(self, ids):
        raw = self.tokenizer.decode([token for token in ids if token not in self.eos],
            skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if not isinstance(raw, str):
            raise RuntimeError('Output tokenizer did not return text')
        return raw

    def feed(self, new_ids):
        self._check_live()
        new_ids = list(new_ids)
        if not new_ids:
            return self.raw
        self._ids.extend(new_ids)
        try:
            if self.incremental:
                chunks = []
                for token in new_ids:
                    if token in self.eos:
                        continue
                    # A list ending in unfinished UTF-8 can hide an earlier
                    # complete control marker. Advance one ID at a time.
                    chunk = self._stream.step(self._backend, token)
                    if chunk is not None:
                        if not isinstance(chunk, str):
                            raise RuntimeError('Incremental tokenizer did not return text')
                        chunks.append(chunk)
                self.raw += ''.join(chunks)
            else:
                raw = self._decode(self._ids).rstrip('\ufffd')
                if not raw.startswith(self.raw):
                    raise RuntimeError('Non-monotonic output detokenization')
                self.raw = raw
        except Exception as error:
            self._failed = True
            raise RuntimeError('Output decoding failed; text cannot be safely replayed') from error
        return self.raw

    def finish(self, all_ids):
        if self._failed:
            raise RuntimeError('Output decoder previously failed')
        all_ids = list(all_ids)
        if self._finished:
            if all_ids != self._ids:
                raise ValueError('Finished output token history changed')
            return self.raw
        if len(all_ids) < len(self._ids) or all_ids[:len(self._ids)] != self._ids:
            self._failed = True
            raise ValueError('Output token history must keep its committed prefix')
        try:
            raw = self._decode(all_ids)
            if not raw.startswith(self.raw):
                raise RuntimeError('Final output detokenization changed streamed text')
        except Exception as error:
            self._failed = True
            raise RuntimeError('Final output decoding failed; text cannot be safely replayed') from error
        self.raw = raw
        self._ids = all_ids
        self._stream = self._backend = None
        self._finished = True
        return self.raw
