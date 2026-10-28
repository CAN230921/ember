"""Model config parsing for the sm_121 engine (real checkpoint + runtime override)."""
import json
from dataclasses import dataclass, field


@dataclass
class EngineConfig:
    hidden_size: int
    num_layers: int
    layer_types: list
    num_experts: int
    experts_per_tok: int
    moe_intermediate: int
    shared_expert_intermediate: int
    vocab_size: int
    hc_count: int
    hc_lowrank: int
    head_dim: int
    num_attn_heads: int
    num_kv_heads: int
    linear_key_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_value_head_dim: int
    linear_conv_kernel: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    full_attention_interval: int
    ple_layer_ids: list
    ngram_size: int
    indexer_n_heads: int
    indexer_head_dim: int
    indexer_kv_heads: int
    indexer_budget: int
    indexer_compress_ratio: int
    mtp_num_layers: int
    rope_factor: float = 1.0
    rope_original_max_pos: int = 262144
    raw: dict = field(default_factory=dict, repr=False)


def load_engine_config(config_path: str, override_path: str | None = None) -> EngineConfig:
    cfg = json.load(open(config_path))
    if override_path:
        ov = json.load(open(override_path))
        # merge shallow: override wins (runtime mounts it as /model/config.json)
        for k, v in ov.items():
            if k == "text_config" and isinstance(v, dict):
                cfg.setdefault("text_config", {}).update(v)
            else:
                cfg[k] = v
    t = cfg["text_config"]
    rope = t.get("rope_parameters", {})
    return EngineConfig(
        hidden_size=t["hidden_size"],
        num_layers=t["num_hidden_layers"],
        layer_types=t["layer_types"],
        num_experts=t["num_experts"],
        experts_per_tok=t["num_experts_per_tok"],
        moe_intermediate=t["moe_intermediate_size"],
        shared_expert_intermediate=t["shared_expert_intermediate_size"],
        vocab_size=t["vocab_size"],
        hc_count=t["hc_count"],
        hc_lowrank=t["hc_lowrank"],
        head_dim=t["head_dim"],
        num_attn_heads=t["num_attention_heads"],
        num_kv_heads=t["num_key_value_heads"],
        linear_key_head_dim=t["linear_key_head_dim"],
        linear_num_key_heads=t["linear_num_key_heads"],
        linear_num_value_heads=t["linear_num_value_heads"],
        linear_value_head_dim=t["linear_value_head_dim"],
        linear_conv_kernel=t["linear_conv_kernel_dim"],
        rms_norm_eps=t["rms_norm_eps"],
        rope_theta=float(rope.get("rope_theta", t.get("rope_theta", 10000000))),
        partial_rotary_factor=float(rope.get("partial_rotary_factor", 0.25)),
        full_attention_interval=t["full_attention_interval"],
        ple_layer_ids=t["ple_layer_ids"],
        ngram_size=t["ngram_size"],
        indexer_n_heads=t["indexer_n_heads"],
        indexer_head_dim=t["indexer_head_dim"],
        indexer_kv_heads=t["indexer_kv_heads"],
        indexer_budget=t["indexer_budget"],
        indexer_compress_ratio=t["indexer_compress_ratio"],
        mtp_num_layers=t.get("mtp_num_hidden_layers", 0),
        rope_factor=float(rope.get("factor", 1.0)),
        rope_original_max_pos=int(rope.get("original_max_position_embeddings", 262144)),
        raw=cfg,
    )
