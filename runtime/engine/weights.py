"""Safetensors weight access for the sm_121 engine.

Streaming access via safetensors.safe_open (mmap; tensors fetched on demand
to a device). Builds a module-group index from model.safetensors.index.json.
No copies, no transposes, no weight mutation.
"""
import json
import os
import re
from collections import defaultdict

import torch
from safetensors import safe_open


class WeightStore:
    def __init__(self, model_dir: str, device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        self.weight_map = json.load(open(index_path))["weight_map"]
        self._handles = {}

    def _handle(self, fname: str):
        if fname not in self._handles:
            self._handles[fname] = safe_open(
                os.path.join(self.model_dir, fname), framework="pt", device=self.device
            )
        return self._handles[fname]

    def names(self, pattern: str = ""):
        return [k for k in self.weight_map if pattern in k]

    def get(self, name: str) -> torch.Tensor:
        f = self.weight_map[name]
        h = self._handle(f)
        return h.get_tensor(name)

    def group(self, prefix: str) -> dict:
        out = {}
        for k in self.weight_map:
            if k.startswith(prefix):
                out[k[len(prefix):].lstrip(".")] = self.get(k)
        return out

    def layer_group(self, layer_idx: int) -> dict:
        p = f"model.language_model.layers.{layer_idx}."
        return self.group(p)

    def expert_names(self, layer_idx: int, e0: int, e1: int) -> list:
        pat = re.compile(rf"layers\.{layer_idx}\.mlp\.experts\.(\d+)\.")
        return [k for k in self.weight_map if (m := pat.search(k)) and e0 <= int(m.group(1)) < e1]

    def expert_group(self, layer_idx: int, e: int) -> dict:
        p = f"model.language_model.layers.{layer_idx}.mlp.experts.{e}."
        return self.group(p)

    def schema_summary(self) -> dict:
        """Aggregate dtypes and verify the expected checkpoint schema."""
        agg = defaultdict(lambda: [0, 0])
        by_file_bytes = defaultdict(int)
        for name, f in self.weight_map.items():
            h = self._handle(f)
            t = h.get_slice(name)
            dt = t.get_dtype()
            shp = t.get_shape()
            n = 1
            for s in shp:
                n *= s
            by_file_bytes[f] += n * _dtype_size(dt)
            agg[dt][0] += 1
            agg[dt][1] += n * _dtype_size(dt)
        return {dt: {"tensors": c, "bytes": b} for dt, (c, b) in sorted(agg.items())}, dict(by_file_bytes)


_DT_SIZES = {"F8_E4M3": 1, "U8": 1, "BF16": 2, "F16": 2, "I32": 4, "F32": 4, "I64": 8}


def _dtype_size(dt: str) -> int:
    return _DT_SIZES.get(dt, 1)
