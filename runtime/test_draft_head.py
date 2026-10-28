"""CPU contracts for draft column copies and ID mapping; no CUDA claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from engine.draft_head import (DraftVocabularyHead, _ValidatedVocabularyIds, _parse_manifest,
                               select_draft, vocabulary_ids_sha256, tokenizer_vocabulary_requirements, PROTOCOL_MARKERS)
from engine.ops.greedy_select import GreedySelector


TOKENIZER_SHA = '12' * 32
FREQUENCY_SHA = '34' * 32


def manifest(ids=None, full_size=96):
    ids = list(range(1, 65, 2)) if ids is None else list(ids)
    return dict(schema_version=1, full_vocab_size=full_size, vocab_size=len(ids),
                vocab_ids=ids, vocab_ids_sha256=vocabulary_ids_sha256(ids),
                tokenizer_sha256=TOKENIZER_SHA, mandatory_ids=[ids[0], ids[-1]],
                frequency_source=dict(description='Independent synthetic CPU fixture',
                                      sha256=FREQUENCY_SHA))


def blob_and_hash(value):
    blob = json.dumps(value, separators=(',', ':')).encode('utf-8')
    return blob, hashlib.sha256(blob).hexdigest()


def parsed(value=None, *, required=(), **options):
    value = manifest() if value is None else value
    blob, checksum = blob_and_hash(value)
    kwargs = dict(expected_manifest_sha256=checksum,
                  expected_tokenizer_sha256=TOKENIZER_SHA,
                  full_vocab_size=96, mandatory_ids=required,
                  _allowed_sizes=(32,))
    kwargs.update(options)
    return _parse_manifest(blob, **kwargs)


def small_head():
    # Every source row/group/column differs: a misplaced scale is detectable.
    weights = torch.arange(640 * 96, dtype=torch.int32).reshape(640, 96)
    scales = (torch.arange(20 * 96).reshape(20, 96) * 0.001).to(torch.float16)
    return weights, scales, DraftVocabularyHead(weights, scales, parsed())


class DraftHeadTests(unittest.TestCase):
    def test_original_packed_columns_and_group_scales_are_copied_exactly(self):
        weights, scales, head = small_head()
        columns = torch.tensor(head.vocab_ids)
        self.assertEqual(head.qweight.dtype, torch.int32)
        self.assertEqual(head.scales.dtype, torch.float16)
        self.assertEqual(head.qweight.shape, (640, 32))
        self.assertEqual(head.scales.shape, (20, 32))
        self.assertTrue(head.qweight.is_contiguous())
        self.assertTrue(head.scales.is_contiguous())
        self.assertEqual(head.qweight.device, weights.device)
        self.assertEqual(head.scales.device, scales.device)
        torch.testing.assert_close(head.qweight, weights.index_select(1, columns), rtol=0, atol=0)
        torch.testing.assert_close(head.scales, scales.index_select(1, columns), rtol=0, atol=0)
        self.assertNotEqual(head.qweight.data_ptr(), weights.data_ptr())
        self.assertNotEqual(head.scales.data_ptr(), scales.data_ptr())
        original_weight, original_scale = int(weights[0, 1]), float(scales[0, 1])
        head.qweight[0, 0] = -99
        head.scales[0, 0] = -99
        self.assertEqual(int(weights[0, 1]), original_weight)
        self.assertEqual(float(scales[0, 1]), original_scale)

    def test_forward_uses_original_packed_operator_and_float_conversion(self):
        _, _, head = small_head()
        hidden = torch.zeros(1, 2560, dtype=torch.bfloat16)
        output = torch.arange(32, dtype=torch.bfloat16).reshape(1, 32)
        with patch('engine.draft_head.packed_head', return_value=output) as native:
            result = head(hidden)
        native.assert_called_once()
        self.assertIs(native.call_args.args[0], hidden)
        self.assertIs(native.call_args.args[1], head.qweight)
        self.assertIs(native.call_args.args[2], head.scales)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, output.float(), rtol=0, atol=0)

    def test_vocab_ids_are_immutable_and_audit_metadata_is_private(self):
        _, _, head = small_head()
        self.assertIsInstance(head.vocab_ids, tuple)
        with self.assertRaises(AttributeError):
            head.vocab_ids = (1,)
        metadata = head.metadata
        self.assertEqual(metadata['tokenizer_sha256'], TOKENIZER_SHA)
        self.assertEqual(metadata['vocab_ids_sha256'], vocabulary_ids_sha256(head.vocab_ids))
        metadata['frequency_source']['description'] = 'changed only in copy'
        metadata['vocab_size'] = 1
        self.assertEqual(head.metadata['vocab_size'], 32)
        self.assertEqual(head.metadata['frequency_source']['description'],
                         'Independent synthetic CPU fixture')
        self.assertEqual(head.state_dict(), {})

    def test_unverified_constructor_is_rejected(self):
        weights, scales, _ = small_head()
        with self.assertRaisesRegex(ValueError, 'hash-verified'):
            DraftVocabularyHead(weights, scales, list(range(32)))

    def test_original_buffer_dtype_shape_and_layout_are_validated(self):
        weights, scales, _ = small_head()
        bad = [(weights.to(torch.int64), scales), (weights[:639], scales),
               (weights[:, :-1].contiguous(), scales[:, :-1].contiguous()),
               (weights, scales.to(torch.float32)), (weights, scales[:19]),
               (weights[:, ::2], scales[:, ::2])]
        for weight, scale in bad:
            with self.subTest(shape=weight.shape, dtype=weight.dtype, scale=scale.shape):
                with self.assertRaises(ValueError):
                    DraftVocabularyHead(weight, scale, parsed())

    def test_full_head_fallback_calls_selector_without_new_checks_or_copy(self):
        # Even invalid dummy logits must go unchanged to the original selector.
        logits = object()
        expected = object()
        for output in (SimpleNamespace(logits=logits),
                       SimpleNamespace(logits=logits, vocab_ids=None)):
            with self.subTest(output=output):
                def original(value):
                    self.assertIs(value, logits)
                    return expected
                self.assertIs(select_draft(output, original), expected)

    def test_compact_selection_maps_each_row_and_preserves_lowest_id_tie(self):
        logits = torch.tensor([[2., 9., 9., -1.], [3., 2., 1., 8.], [5., 5., 5., 5.]])
        output = SimpleNamespace(logits=logits, vocab_ids=(2, 7, 41, 88))
        self.assertEqual(select_draft(output, GreedySelector()), [7, 88, 2])
        full = torch.full((3, 96), -100.)
        full[:, torch.tensor(output.vocab_ids)] = logits
        self.assertEqual(select_draft(output, GreedySelector()), GreedySelector()(full))

    def test_compact_selection_retains_selector_nonfinite_error(self):
        for bad in (float('nan'), float('inf'), -float('inf')):
            logits = torch.tensor([[0., bad]])
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(RuntimeError, 'Nonfinite'):
                    select_draft(SimpleNamespace(logits=logits, vocab_ids=(1, 2)),
                                 GreedySelector())

    def test_validated_hot_path_never_iterates_or_revalidates_full_mapper(self):
        class NoIterationIds(_ValidatedVocabularyIds):
            def __iter__(self):
                raise AssertionError('Hot path iterated the full draft vocabulary')
        ids = NoIterationIds(range(32768))
        output = SimpleNamespace(logits=torch.zeros(1, 32768), vocab_ids=ids)
        for _ in range(3):
            with patch('engine.draft_head._ordered_ids', side_effect=AssertionError('revalidated')):
                self.assertEqual(select_draft(output, lambda _: [32767]), [32767])

    def test_compact_mapper_and_width_are_validated_before_selector(self):
        for ids, logits in [([1, 2], torch.zeros(1, 2)), ((), torch.zeros(1, 0)),
                            ((True, 2), torch.zeros(1, 2)), ((-1, 2), torch.zeros(1, 2)),
                            ((2, 1), torch.zeros(1, 2)), ((2, 2), torch.zeros(1, 2)),
                            ((1, 2), torch.zeros(1, 3)), ((1, 2), torch.zeros(2))]:
            with self.subTest(ids=ids, shape=logits.shape):
                def never_called(_):
                    self.fail('Malformed mapper reached selector')
                with self.assertRaises(ValueError):
                    select_draft(SimpleNamespace(logits=logits, vocab_ids=ids), never_called)

    def test_selector_indexes_cannot_be_bool_float_negative_or_out_of_bounds(self):
        output = SimpleNamespace(logits=torch.zeros(1, 2), vocab_ids=(10, 20))
        for result in ([True], [1.5], [-1], [2], [], [0, 1], 1):
            with self.subTest(result=result):
                with self.assertRaises(ValueError):
                    select_draft(output, lambda _: result)

    def test_manifest_and_ids_hashes_bind_actual_bytes(self):
        value = manifest()
        with self.assertRaisesRegex(ValueError, 'manifest SHA256 mismatch'):
            parsed(value, expected_manifest_sha256='00' * 32)
        value['vocab_ids_sha256'] = '00' * 32
        with self.assertRaisesRegex(ValueError, 'IDs SHA256 mismatch'):
            parsed(value)
        for field in ('vocab_ids_sha256', 'tokenizer_sha256'):
            value = manifest()
            value[field] = 'not-a-hash'
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    parsed(value)
        with self.assertRaises(ValueError):
            parsed(expected_manifest_sha256=None)

    def test_manifest_tokenizer_size_schema_and_width_are_bound(self):
        mutations = [('tokenizer_sha256', '00' * 32), ('full_vocab_size', 64),
                     ('full_vocab_size', True), ('schema_version', True),
                     ('schema_version', 2), ('vocab_size', True), ('vocab_size', 31),
                     ('vocab_size', 64), ('vocab_ids', list(range(31)))]
        for key, bad in mutations:
            value = manifest()
            value[key] = bad
            with self.subTest(key=key, bad=bad):
                with self.assertRaises(ValueError):
                    parsed(value)

    def test_manifest_rejects_unsorted_duplicate_bool_or_out_of_range_ids(self):
        for bad in ([3, 1] + list(range(5, 65, 2)),
                    [1, 1] + list(range(5, 65, 2)),
                    [True] + list(range(3, 65, 2)),
                    [-1] + list(range(3, 65, 2)),
                    list(range(1, 63, 2)) + [96]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parsed(manifest(bad))

    def test_both_declared_and_runtime_protocol_ids_must_be_covered(self):
        verified = parsed(required={63, 1, 3})
        self.assertEqual(verified.metadata['required_mandatory_ids'], (1, 3, 63))
        for ids in ([0], [True], [96], None):
            with self.subTest(required=ids):
                with self.assertRaises(ValueError):
                    parsed(required=ids)
        for ids in ([0], [True], [63, 1], [1, 1], [96], None):
            value = manifest()
            value['mandatory_ids'] = ids
            with self.subTest(declared=ids):
                with self.assertRaises(ValueError):
                    parsed(value)
        value = manifest()
        del value['mandatory_ids']
        with self.assertRaises(ValueError):
            parsed(value)

    def test_frequency_source_is_required_and_audit_excludes_unneeded_content(self):
        for frequency in (None, {}, {'description': '', 'sha256': FREQUENCY_SHA},
                          {'description': 'source', 'sha256': 'bad'}):
            value = manifest()
            value['frequency_source'] = frequency
            with self.subTest(frequency=frequency):
                with self.assertRaises(ValueError):
                    parsed(value)
        value = manifest()
        value['frequency_source']['corpus_content'] = 'should not appear in head metadata'
        self.assertNotIn('corpus_content', parsed(value).metadata['frequency_source'])

    def test_manifest_rejects_duplicate_keys_invalid_json_and_oversize(self):
        for blob in (b'{"schema_version":1,"schema_version":1}', b'null',
                     b'[]', b'not json', b'\xff', b' ' * (4 * 1024 * 1024 + 1)):
            with self.subTest(prefix=blob[:40]):
                with self.assertRaises(ValueError):
                    _parse_manifest(blob, expected_manifest_sha256=hashlib.sha256(blob).hexdigest(),
                                    expected_tokenizer_sha256=TOKENIZER_SHA, full_vocab_size=96)

    def test_production_sizes_are_accepted_but_public_loader_rejects_small_fixture(self):
        for size in (32768, 65536):
            value = manifest(range(size), full_size=248320)
            blob, checksum = blob_and_hash(value)
            verified = _parse_manifest(blob, expected_manifest_sha256=checksum,
                                       expected_tokenizer_sha256=TOKENIZER_SHA,
                                       full_vocab_size=248320)
            self.assertEqual(len(verified.ids), size)
        weights, scales, _ = small_head()
        blob, checksum = blob_and_hash(manifest())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'draft-vocab.json'
            path.write_bytes(blob)
            with self.assertRaisesRegex(ValueError, '32768 or 65536'):
                DraftVocabularyHead.from_manifest(
                    weights, scales, path, expected_manifest_sha256=checksum,
                    expected_tokenizer_sha256=TOKENIZER_SHA)

    def test_public_loader_reads_bound_manifest_before_copying_columns(self):
        # Real production K, CPU tensors only. No model weights or native head.
        full_size, size = 32800, 32768
        value = manifest(range(32, full_size), full_size=full_size)
        blob, checksum = blob_and_hash(value)
        weights = torch.zeros(640, full_size, dtype=torch.int32)
        weights[0] = torch.arange(full_size, dtype=torch.int32)
        scales = torch.zeros(20, full_size, dtype=torch.float16)
        scales[3] = (torch.arange(full_size) / 32).to(torch.float16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'draft-vocab.json'
            path.write_bytes(blob)
            head = DraftVocabularyHead.from_manifest(
                weights, scales, path, expected_manifest_sha256=checksum,
                expected_tokenizer_sha256=TOKENIZER_SHA, mandatory_ids={32, full_size - 1})
        self.assertEqual(head.vocab_ids[0], 32)
        self.assertEqual(head.vocab_ids[-1], full_size - 1)
        self.assertEqual(head.qweight.shape, (640, size))
        torch.testing.assert_close(head.qweight, weights[:, 32:], rtol=0, atol=0)
        torch.testing.assert_close(head.scales, scales[:, 32:], rtol=0, atol=0)


    def test_undefined_tokenizer_rows_are_rejected_before_gpu_columns(self):
        value=manifest()
        with self.assertRaisesRegex(ValueError, 'undefined tokenizer'):
            parsed(value, defined_ids=set(value['vocab_ids'][:-1]))

    def test_actual_tokenizer_marker_added_and_eos_requirements(self):
        vocab={marker:i for i,marker in enumerate(PROTOCOL_MARKERS)}
        vocab.update(other=40, piece=41, eos=42)
        tokenizer=SimpleNamespace(get_vocab=lambda:vocab,get_added_vocab=lambda:{'other':40},
            all_special_ids=[0],encode=lambda marker,**kwargs:[vocab[marker],41])
        defined,required=tokenizer_vocabulary_requirements(tokenizer,{42})
        self.assertEqual(defined,set(vocab.values()))
        self.assertEqual(set(required),defined)
        with self.assertRaisesRegex(ValueError,'undefined'):
            tokenizer_vocabulary_requirements(tokenizer,{99})
        del vocab[PROTOCOL_MARKERS[0]]
        with self.assertRaisesRegex(ValueError,'marker'):
            tokenizer_vocabulary_requirements(tokenizer,{42})


if __name__ == '__main__':
    unittest.main()
