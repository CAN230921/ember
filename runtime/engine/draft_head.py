"""Hash-bound compact vocabulary for the learned draft, never the target.

The selected GPTQ columns and group scales are copied without requantization.
The same packed INT8 head and BF16-to-FP32 conversion as the target are used.
Loading is explicit; callers that retain the existing full head are unaffected.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import operator
from pathlib import Path

import torch
from torch import nn

from .ops.int8_head import packed_head


PRODUCTION_VOCAB_SIZES = (32768, 65536)
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
PROTOCOL_MARKERS = ('<|endoftext|>', '<|im_start|>', '<|im_end|>', '<think>', '</think>',
                    '<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>',
                    '<|fim_prefix|>', '<|fim_middle|>', '<|fim_suffix|>', '<|fim_pad|>',
                    '<|repo_name|>', '<|file_sep|>')


def tokenizer_vocabulary_requirements(tokenizer, eos):
    """Defined IDs and required protocol IDs from the actual loaded tokenizer."""
    vocabulary = tokenizer.get_vocab()
    defined = set(vocabulary.values())
    if any(type(token) is not int or token < 0 for token in defined):
        raise ValueError('Tokenizer contains invalid token IDs')
    required = set(tokenizer.get_added_vocab().values()) | set(tokenizer.all_special_ids) | set(eos)
    for marker in PROTOCOL_MARKERS:
        if marker not in vocabulary:
            raise ValueError('Tokenizer omits a required protocol marker')
        required.update(tokenizer.encode(marker, add_special_tokens=False))
    if any(type(token) is not int for token in required) or not required <= defined:
        raise ValueError('Tokenizer protocol or EOS IDs are undefined')
    return defined, tuple(sorted(required))


def vocabulary_ids_sha256(ids):
    """Canonical JSON hash shared with the offline frequency-list builder."""
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':'),
                                    ensure_ascii=True).encode('utf-8')).hexdigest()


def _sha256(value, name):
    if (not isinstance(value, str) or len(value) != 64 or
            any(char not in '0123456789abcdefABCDEF' for char in value)):
        raise ValueError(f'{name} must be a SHA256 hexadecimal digest')
    return value.lower()


def _ordered_ids(values, full_vocab_size, name, *, nonempty=False):
    if not isinstance(values, (list, tuple)) or (nonempty and not values):
        raise ValueError(f'{name} must be an ordered token ID sequence')
    ids = tuple(values)
    if any(type(token) is not int or not 0 <= token < full_vocab_size for token in ids):
        raise ValueError(f'{name} contains invalid or out-of-vocabulary token IDs')
    if any(left >= right for left, right in zip(ids, ids[1:])):
        raise ValueError(f'{name} must contain unique ascending token IDs')
    return ids


def _no_duplicate_keys(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f'Duplicate draft vocabulary JSON key: {key}')
        output[key] = value
    return output


@dataclass(frozen=True)
class _VerifiedVocabulary:
    ids: tuple
    metadata: dict


class _ValidatedVocabularyIds(tuple):
    """Immutable tuple marker created only after the head's full validation.

    Keeping this distinction avoids rescanning 32K/64K Python integers for
    every draft proposal. There are no mutable per-output validation flags.
    """
    __slots__ = ()


def _parse_manifest(blob, *, expected_manifest_sha256, expected_tokenizer_sha256,
                    full_vocab_size, mandatory_ids=(), defined_ids=None,
                    _allowed_sizes=PRODUCTION_VOCAB_SIZES):
    """Validate bytes before allocating head columns.

    The private size override supports small CPU contract fixtures. Public
    loading always uses PRODUCTION_VOCAB_SIZES, and has no size override.
    """
    if not isinstance(blob, bytes) or len(blob) > _MAX_MANIFEST_BYTES:
        raise ValueError('Draft vocabulary manifest is invalid or too large')
    expected_manifest_sha256 = _sha256(expected_manifest_sha256,
                                       'expected_manifest_sha256')
    expected_tokenizer_sha256 = _sha256(expected_tokenizer_sha256,
                                        'expected_tokenizer_sha256')
    actual_hash = hashlib.sha256(blob).hexdigest()
    if actual_hash != expected_manifest_sha256:
        raise ValueError('Draft vocabulary manifest SHA256 mismatch')
    try:
        manifest = json.loads(blob.decode('utf-8'), object_pairs_hook=_no_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('Invalid draft vocabulary JSON') from error
    if not isinstance(manifest, dict):
        raise ValueError('Draft vocabulary manifest must be a JSON object')
    if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1:
        raise ValueError('Unsupported draft vocabulary manifest schema')
    if type(full_vocab_size) is not int or full_vocab_size <= 0:
        raise ValueError('Expected a positive full vocabulary size')
    if (type(manifest.get('full_vocab_size')) is not int or
            manifest['full_vocab_size'] != full_vocab_size):
        raise ValueError('Draft vocabulary full_vocab_size does not match the packed head')
    size = manifest.get('vocab_size')
    if type(size) is not int or size not in _allowed_sizes or size % 32:
        raise ValueError('Draft vocabulary size must be 32768 or 65536 and 32-aligned')
    ids = _ordered_ids(manifest.get('vocab_ids'), full_vocab_size, 'vocab_ids', nonempty=True)
    if len(ids) != size:
        raise ValueError('Draft vocabulary width does not match vocab_size')
    if defined_ids is not None and not set(ids).issubset(defined_ids):
        raise ValueError('Draft vocabulary contains undefined tokenizer IDs')
    ids_hash = vocabulary_ids_sha256(ids)
    if ids_hash != _sha256(manifest.get('vocab_ids_sha256'), 'vocab_ids_sha256'):
        raise ValueError('Draft vocabulary IDs SHA256 mismatch')
    tokenizer_hash = _sha256(manifest.get('tokenizer_sha256'), 'tokenizer_sha256')
    if tokenizer_hash != expected_tokenizer_sha256:
        raise ValueError('Draft vocabulary tokenizer SHA256 mismatch')
    declared_mandatory = _ordered_ids(manifest.get('mandatory_ids'), full_vocab_size,
                                      'mandatory_ids')
    # The caller may naturally pass a set/unsorted EOS + protocol ID union.
    # Validate its elements before normalization so bool never becomes 0/1.
    try:
        required = tuple(mandatory_ids)
    except TypeError as error:
        raise ValueError('mandatory_ids must contain legal token IDs') from error
    if any(type(token) is not int or not 0 <= token < full_vocab_size for token in required):
        raise ValueError('Caller mandatory_ids contains invalid token IDs')
    required = tuple(sorted(set(required)))
    selected = set(ids)
    if not selected.issuperset(declared_mandatory) or not selected.issuperset(required):
        raise ValueError('Draft vocabulary omits mandatory token IDs')
    frequency = manifest.get('frequency_source')
    if (not isinstance(frequency, dict) or
            not isinstance(frequency.get('description'), str) or
            not frequency['description'].strip()):
        raise ValueError('Draft vocabulary must describe its independent frequency source')
    frequency_hash = _sha256(frequency.get('sha256'), 'frequency_source.sha256')
    # Copy only provenance, not corpus content or arbitrary manifest extras.
    metadata = dict(schema_version=1, manifest_sha256=actual_hash,
                    full_vocab_size=full_vocab_size, vocab_size=size,
                    vocab_ids_sha256=ids_hash, tokenizer_sha256=tokenizer_hash,
                    frequency_source=dict(description=frequency['description'],
                                          sha256=frequency_hash),
                    manifest_mandatory_ids=declared_mandatory,
                    required_mandatory_ids=required,
                    weight_selection='original_packed_columns_no_requantization')
    return _VerifiedVocabulary(ids, metadata)


def _head_shape(qweight, scales):
    if (not isinstance(qweight, torch.Tensor) or qweight.ndim != 2 or
            qweight.dtype != torch.int32 or qweight.shape[0] != 640 or
            not qweight.shape[1] or qweight.shape[1] % 32):
        raise ValueError('Expected original GPTQ INT32 qweight [640,N], N aligned to 32')
    if (not isinstance(scales, torch.Tensor) or scales.dtype != torch.float16 or
            scales.shape != (20, qweight.shape[1])):
        raise ValueError('Expected original FP16 group128 scales [20,N]')
    if qweight.device != scales.device:
        raise ValueError('Draft head buffers must share a device')
    if not qweight.is_contiguous() or not scales.is_contiguous():
        raise ValueError('Expected contiguous original head buffers')
    return qweight.shape[1]


class DraftVocabularyHead(nn.Module):
    """Draft-only compact head initialized from a verified frequency manifest.

    Use ``from_manifest`` in serving code. The constructor's private verified
    object is also used by small CPU tests; it is not a deployment option.
    """
    def __init__(self, qweight, scales, vocabulary):
        super().__init__()
        full_size = _head_shape(qweight, scales)
        if not isinstance(vocabulary, _VerifiedVocabulary):
            raise ValueError('Draft head requires a hash-verified vocabulary manifest')
        ids = _ordered_ids(vocabulary.ids, full_size, 'vocab_ids', nonempty=True)
        if len(ids) % 32 or vocabulary.metadata.get('full_vocab_size') != full_size:
            raise ValueError('Verified draft vocabulary does not match the original head')
        self._vocab_ids = _ValidatedVocabularyIds(ids)
        self._metadata = deepcopy(vocabulary.metadata)
        columns = torch.tensor(ids, dtype=torch.long, device=qweight.device)
        with torch.no_grad():
            self.register_buffer('qweight', qweight.index_select(1, columns).contiguous(),
                                 persistent=False)
            self.register_buffer('scales', scales.index_select(1, columns).contiguous(),
                                 persistent=False)

    @classmethod
    def from_manifest(cls, qweight, scales, manifest_path, *,
                      expected_manifest_sha256, expected_tokenizer_sha256,
                      mandatory_ids=(), defined_ids=None):
        full_size = _head_shape(qweight, scales)
        with Path(manifest_path).open('rb') as source:
            blob = source.read(_MAX_MANIFEST_BYTES + 1)
        vocabulary = _parse_manifest(
            blob, expected_manifest_sha256=expected_manifest_sha256,
            expected_tokenizer_sha256=expected_tokenizer_sha256,
            full_vocab_size=full_size, mandatory_ids=mandatory_ids, defined_ids=defined_ids)
        return cls(qweight, scales, vocabulary)

    @property
    def vocab_ids(self):
        return self._vocab_ids

    @property
    def metadata(self):
        # Audit callers may add fields to their copy without changing the head.
        return deepcopy(self._metadata)

    def forward(self, hidden):
        return packed_head(hidden, self.qweight, self.scales).float()


def select_draft(output, selector):
    """Preserve the selector gate/tie rule, then map compact columns to IDs.

    The full-head path delegates directly and returns the original result.
    Ascending subset IDs preserve the original lowest-token tie preference.
    """
    ids = getattr(output, 'vocab_ids', None)
    if ids is None:
        return selector(output.logits)
    logits = output.logits
    if not isinstance(ids, _ValidatedVocabularyIds):
        # Legacy CPU fixtures may provide plain tuples. Production outputs
        # always retain the head's validated immutable tuple, an O(1) check.
        if (not isinstance(ids, tuple) or not ids or
                any(type(token) is not int or token < 0 for token in ids) or
                any(left >= right for left, right in zip(ids, ids[1:]))):
            raise ValueError('Compact draft output requires unique ascending immutable vocab_ids')
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or
            logits.shape[1] != len(ids)):
        raise ValueError('Compact draft logits width does not match vocab_ids')
    indexes = selector(logits)
    if not isinstance(indexes, (list, tuple)) or len(indexes) != logits.shape[0]:
        raise ValueError('Draft selector must return one column index per row')
    mapped = []
    for value in indexes:
        if isinstance(value, bool):
            raise ValueError('Draft selector returned an invalid compact column index')
        try:
            index = operator.index(value)
        except TypeError as error:
            raise ValueError('Draft selector returned a non-integer column index') from error
        if not 0 <= index < len(ids):
            raise ValueError('Draft selector compact column index is out of range')
        mapped.append(ids[index])
    return mapped
