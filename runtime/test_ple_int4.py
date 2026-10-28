"""INT4 g32 PLE rows: packing round trip equals the evaluated fake quantisation bit for bit, and DiskPleTable
serves U8 shards unchanged. CPU only."""
import json, struct, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from engine.ops.ple_lookup import (DiskPleTable, dequant_int4g32, quantize_int4g32, quantize_tern, dequant_tern,
                                   dequant_rows, tern_global_scale, tern_row_bytes)


def reference(x, gs):   # diagnostics/ple_quant_eval_20261021.py: rtn + q_int4 + group_scale
    g = x.unflatten(-1, (-1, 32))
    s = (g.abs().amax(-1, keepdim=True)/7.0/gs).to(torch.float8_e4m3fn).float()*gs
    s = s.clamp(min=1e-30)
    return ((g/s).round().clamp(-8, 7)*s).flatten(-2)


def table_rows(n, seed):
    gen = torch.Generator().manual_seed(seed)
    raw = (torch.randn(n, 160, generator=gen)*40).clamp(-448, 448).to(torch.float8_e4m3fn)
    raw[3] = 0; raw[5, :32] = 0; raw[7, 40] = 448   # empty row, empty group, saturated value
    return raw


class Int4PleTest(unittest.TestCase):
    def test_round_trip_matches_fake_quant(self):
        for seed, escale in ((1, 0.0123), (2, 3.5e-4), (3, 1.0)):
            raw = table_rows(4096, seed)
            x = (raw.to(torch.bfloat16)*torch.tensor(escale).to(torch.bfloat16)).float()
            gs = float(torch.tensor(escale).float())/7.0
            rows = quantize_int4g32(x, gs)
            self.assertEqual(rows.dtype, torch.uint8); self.assertEqual(tuple(rows.shape), (4096, 85))
            got = dequant_int4g32(rows.reshape(256, 16*85), gs).reshape(4096, 160)
            self.assertTrue(torch.equal(got, reference(x, gs)))

    def test_disk_table_serves_u8_rows(self):
        raw = table_rows(300, 4); x = (raw.to(torch.bfloat16)*torch.tensor(0.01).to(torch.bfloat16)).float()
        rows = quantize_int4g32(x, 0.01/7).numpy()
        with tempfile.TemporaryDirectory() as d:
            prefix = 'model.language_model.layers.1.ple.ple_embedding.ngram_embedding'
            names = [f'{prefix}.shard_{i}.weight' for i in range(2)]
            parts = [rows[:150], rows[150:]]; header = {}; off = 0
            for name, part in zip(names, parts):
                header[name] = dict(dtype='U8', shape=list(part.shape), data_offsets=[off, off+part.nbytes]); off += part.nbytes
            blob = json.dumps(header).encode(); blob += b' '*(-len(blob) % 8)
            (Path(d)/'ple.safetensors').write_bytes(struct.pack('<Q', len(blob))+blob+b''.join(p.tobytes() for p in parts))
            (Path(d)/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={n: 'ple.safetensors' for n in names})))
            table = DiskPleTable(d, prefix, cache_rows=8)
            self.assertEqual(table.format, 'int4g32'); self.assertEqual(table.width, 85)
            ids = np.array([[0, 149, 150, 299, 7, 7, 3, 151, 10, 20, 30, 40, 50, 60, 70, 80]])
            out = table.lookup(ids)
            self.assertEqual(out.dtype, torch.uint8); self.assertEqual(tuple(out.shape), (1, 16*85))
            self.assertTrue(np.array_equal(out.reshape(16, 85).numpy(), rows[ids[0]]))
            self.assertTrue(np.array_equal(table.lookup(ids).numpy(), out.numpy()))   # cached path


def write_table(d, rows, fmt=None):
    prefix = 'model.language_model.layers.1.ple.ple_embedding.ngram_embedding'
    names = [f'{prefix}.shard_{i}.weight' for i in range(2)]
    half = len(rows)//2; parts = [rows[:half], rows[half:]]; header = {}; off = 0
    for name, part in zip(names, parts):
        header[name] = dict(dtype='U8', shape=list(part.shape), data_offsets=[off, off+part.nbytes]); off += part.nbytes
    if fmt: header['__metadata__'] = dict(ple_format=fmt)
    blob = json.dumps(header).encode(); blob += b' '*(-len(blob) % 8)
    (Path(d)/'ple.safetensors').write_bytes(struct.pack('<Q', len(blob))+blob+b''.join(p.tobytes() for p in parts))
    (Path(d)/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={n: 'ple.safetensors' for n in names})))
    return prefix


class TernaryPleTest(unittest.TestCase):
    def test_pack_unpack_and_optimal_alpha(self):
        raw = table_rows(2048, 9); es = 0.0123
        x = (raw.to(torch.bfloat16)*torch.tensor(es).to(torch.bfloat16)).float()
        for g in (32, 16):
            for rot in (False, True):
                gs = tern_global_scale(es, g, rot); rows = quantize_tern(x, gs, g, rot)
                self.assertEqual(tuple(rows.shape), (2048, tern_row_bytes(g)))
                self.assertTrue(int(rows[:, :32].max()) <= 242)
                y = dequant_tern(rows.reshape(128, -1), gs, g, rot).reshape(2048, 160)
                self.assertFalse(bool(y.isnan().any()))
                self.assertTrue(torch.equal(dequant_rows(rows.reshape(128, -1), f'tern_g{g}_{"rot" if rot else "none"}', es).reshape(2048, 160), y))
                if not rot:   # every value is 0 or +-alpha of its group
                    vals = y.unflatten(-1, (-1, g)).abs()
                    self.assertTrue(bool(((vals == 0) | (vals == vals.amax(-1, keepdim=True))).all()))
        self.assertTrue(float((y-x).norm()/x.norm()) < 0.5)

    def test_disk_table_reads_ternary_metadata(self):
        raw = table_rows(300, 11); es = 0.01
        x = (raw.to(torch.bfloat16)*torch.tensor(es).to(torch.bfloat16)).float()
        rows = quantize_tern(x, tern_global_scale(es, 32, True), 32, True).numpy()
        with tempfile.TemporaryDirectory() as d:
            prefix = write_table(d, rows, 'tern_g32_rot')
            table = DiskPleTable(d, prefix, cache_rows=0)
            self.assertEqual(table.format, 'tern_g32_rot'); self.assertEqual(table.width, 37)
            ids = np.arange(16)[None]*17
            self.assertTrue(np.array_equal(table.lookup(ids).reshape(16, 37).numpy(), rows[ids[0]]))
        with tempfile.TemporaryDirectory() as d:   # wrong width for the declared format is refused
            prefix = write_table(d, rows[:, :36].copy(), 'tern_g32_rot')
            with self.assertRaises(ValueError): DiskPleTable(d, prefix)


class ResidentPleTest(unittest.TestCase):
    def test_resident_rows_equal_disk_rows(self):
        import os
        raw = table_rows(300, 13); es = 0.01
        x = (raw.to(torch.bfloat16)*torch.tensor(es).to(torch.bfloat16)).float()
        rows = quantize_tern(x, tern_global_scale(es, 16, False), 16, False).numpy()
        with tempfile.TemporaryDirectory() as d:
            prefix = write_table(d, rows, 'tern_g16_none')
            disk = DiskPleTable(d, prefix, cache_rows=4)
            os.environ['ENGINE_PLE_RESIDENT'] = 'cpu'
            try: res = DiskPleTable(d, prefix)
            finally: os.environ.pop('ENGINE_PLE_RESIDENT')
            self.assertIsNotNone(res.resident); self.assertFalse(res.io_enabled)
            ids = np.random.default_rng(0).integers(0, 300, size=(5, 16))
            self.assertTrue(torch.equal(res.lookup(ids), disk.lookup(ids)))
            with self.assertRaises(IndexError): res.lookup(np.array([[300]]))


if __name__ == '__main__': unittest.main()
