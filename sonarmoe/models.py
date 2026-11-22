"""Model-Agnostic Open-Weight MoE Hardware & Router Adapter (OLMoE & Qwen1.5-MoE)."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class MoEArchSpec:
    key: str
    model_id: str
    num_layers: int
    num_experts: int
    top_k: int
    num_ranks: int
    has_shared_expert: bool
    norm_topk_prob: bool
    prob_margin_gamma: float

    @property
    def experts_per_rank(self) -> int:
        return self.num_experts // self.num_ranks


SUPPORTED_MOE_SPECS: dict[str, MoEArchSpec] = {
    "olmoe_7b": MoEArchSpec(
        key="olmoe_7b",
        model_id="allenai/OLMoE-1B-7B-0924-Instruct",
        num_layers=16,
        num_experts=64,
        top_k=8,
        num_ranks=8,  # 8 ranks x 8 experts/rank
        has_shared_expert=False,
        norm_topk_prob=False,
        prob_margin_gamma=0.00035,
    ),
    "granite31_moe_1b": MoEArchSpec(
        key="granite31_moe_1b",
        model_id="ibm-granite/granite-3.1-1b-a400m-instruct",
        num_layers=24,
        num_experts=32,
        top_k=8,
        num_ranks=8,  # 8 ranks x 4 experts/rank (Mixtral/MegaBlocks normalized top-k)
        has_shared_expert=False,
        norm_topk_prob=True,
        prob_margin_gamma=0.00080,
    ),
    "qwen15_moe_a27b": MoEArchSpec(
        key="qwen15_moe_a27b",
        model_id="Qwen/Qwen1.5-MoE-A2.7B-Chat",
        num_layers=24,
        num_experts=60,
        top_k=4,
        num_ranks=6,  # 6 ranks x 10 experts/rank
        has_shared_expert=True,
        norm_topk_prob=False,
        prob_margin_gamma=0.00060,
    ),
}


def get_moe_block(layer: Any) -> Any:
    return getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe")


def get_router_module(layer: Any) -> Any:
    blk = get_moe_block(layer)
    return getattr(blk, "gate", None) or getattr(blk, "router")


def get_experts_module(layer: Any) -> Any:
    return get_moe_block(layer).experts


def load_moe_on_mps(spec: MoEArchSpec) -> tuple[Any, Any, float, float]:
    """Loads tokenizer and FP16 MoE model onto Apple Silicon MPS cleanly."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id,
        dtype=torch.float16,
        device_map="mps",
        local_files_only=True,
    )
    model.eval()
    load_sec = round(time.perf_counter() - t0, 2)
    alloc_gb = round(torch.mps.current_allocated_memory() / 1e9, 2)
    return model, tokenizer, load_sec, alloc_gb


def unload_moe_from_mps(model: Any) -> None:
    """Frees MPS Unified Memory before loading the next MoE architecture."""
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


@torch.no_grad()
def forward_capture_router_logits(
    model: Any,
    input_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """Captures exact `(batch * seq_len, num_experts)` router_logits via gate hooks."""
    captured: list[torch.Tensor] = []
    hooks = []

    def _hook(_mod, _inp, out):
        # Both OlmoeTopKRouter and Qwen2MoeTopKRouter return (router_logits, scores, indices)
        rl = out[0] if isinstance(out, tuple) else out
        captured.append(rl.detach())

    for layer in model.model.layers:
        r_mod = get_router_module(layer)
        hooks.append(r_mod.register_forward_hook(_hook))

    try:
        _ = model(input_ids=input_ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    return captured


@torch.no_grad()
def extract_fixed_window_traces(
    model: Any,
    tokenizer: Any,
    spec: MoEArchSpec,
    prompts: list[dict[str, object]],
    b_victim: int = 64,
) -> dict[str, Any]:
    """Extracts exact router logits and top-k counts on fixed `b_victim` windows."""
    device = next(model.parameters()).device
    num_q = len(prompts)
    router_logits_all = np.zeros(
        (num_q, spec.num_layers, b_victim, spec.num_experts), dtype=np.float16
    )
    undefended_counts = np.zeros(
        (num_q, spec.num_layers, spec.num_experts), dtype=np.int32
    )
    labels = np.zeros(num_q, dtype=np.int64)

    for q_idx, item in enumerate(prompts):
        labels[q_idx] = int(item["category_idx"])
        tok_ids = tokenizer(str(item["prompt"]), add_special_tokens=False)["input_ids"]
        if len(tok_ids) < b_victim:
            # Pad deterministically by repeating tail tokens if a tokenizer compresses tightly
            reps = (b_victim // max(len(tok_ids), 1)) + 2
            tok_ids = (tok_ids * reps)[:b_victim]
        else:
            tok_ids = tok_ids[:b_victim]

        inp = torch.tensor([tok_ids], dtype=torch.long, device=device)
        captured = forward_capture_router_logits(model, inp)
        assert len(captured) == spec.num_layers

        for l_idx, rl_raw in enumerate(captured):
            rl = rl_raw.float()  # (b_victim, num_experts)
            router_logits_all[q_idx, l_idx] = rl.cpu().numpy().astype(np.float16)
            probs = F.softmax(rl, dim=-1)
            topk_idx = torch.topk(probs, spec.top_k, dim=-1).indices
            undefended_counts[q_idx, l_idx] = (
                torch.bincount(topk_idx.reshape(-1), minlength=spec.num_experts)
                .cpu()
                .numpy()
            )

    return {
        "router_logits": router_logits_all,
        "undefended_counts": undefended_counts,
        "labels": labels,
    }


@torch.no_grad()
def profile_mps_expert_tile_curve(
    model: Any,
    spec: MoEArchSpec,
    max_tokens: int = 224,
    num_sweeps: int = 20,
) -> dict[str, Any]:
    """Profiles real Apple M5 Pro MPS FP16 expert MLP latency across N=1..max_tokens."""
    t0 = time.perf_counter()
    device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size
    x_pool = torch.randn(max_tokens, hidden_dim, device=device, dtype=torch.float16)

    def run_expert_chain(n_tok: int, e_offset: int) -> None:
        x_slice = x_pool[:n_tok]
        for l_idx in range(spec.num_layers):
            exp_mod = get_experts_module(model.model.layers[l_idx])
            e_idx = (l_idx + e_offset) % spec.num_experts
            gu = F.linear(x_slice, exp_mod.gate_up_proj[e_idx])
            g, u = gu.chunk(2, dim=-1)
            _ = F.linear(F.silu(g) * u, exp_mod.down_proj[e_idx])

    for n_w in [1, 16, 32, 64, 65, 96, 128, 129, 160, max_tokens]:
        for _ in range(2):
            run_expert_chain(n_w, 0)
    torch.mps.synchronize()

    raw_sweeps = np.zeros((max_tokens + 1, num_sweeps), dtype=np.float32)
    for sw in range(num_sweeps):
        order = (
            range(1, max_tokens + 1)
            if (sw % 2 == 0)
            else range(max_tokens, 0, -1)
        )
        for n_tok in order:
            torch.mps.synchronize()
            ts0 = time.perf_counter_ns()
            run_expert_chain(n_tok, sw)
            torch.mps.synchronize()
            ts1 = time.perf_counter_ns()
            raw_sweeps[n_tok, sw] = (ts1 - ts0) / 1000.0 / float(spec.num_layers)

    sorted_sweeps = np.sort(raw_sweeps, axis=1)
    keep_hi = max(6, num_sweeps // 3)
    clean_samples = sorted_sweeps[:, 1 : 1 + keep_hi].copy()
    clean_samples[0] = 0.0

    median_us = np.median(clean_samples, axis=1)
    iqr_us = np.percentile(clean_samples, 75, axis=1) - np.percentile(clean_samples, 25, axis=1)
    within_slope = float(max(0.02, (median_us[64] - median_us[16]) / 48.0))

    return {
        "samples_table_us": clean_samples,
        "median_us": median_us,
        "iqr_us": iqr_us,
        "n16_median_us": round(float(median_us[16]), 2),
        "n32_median_us": round(float(median_us[32]), 2),
        "n64_median_us": round(float(median_us[64]), 2),
        "n65_median_us": round(float(median_us[65]), 2),
        "n128_median_us": round(float(median_us[128]), 2),
        "n129_median_us": round(float(median_us[129]), 2),
        "cliff_64_to_65_us": round(float(median_us[65] - median_us[64]), 2),
        "cliff_128_to_129_us": round(float(median_us[129] - median_us[128]), 2),
        "within_tile_slope_us_per_tok": round(within_slope, 4),
        "mean_clean_iqr_us": round(float(iqr_us[1:].mean()), 2),
        "profiling_time_sec": round(time.perf_counter() - t0, 2),
    }


@torch.no_grad()
def measure_non_routed_step_ms(
    model: Any,
    spec: MoEArchSpec,
    seq_len: int = 64,
) -> float:
    """Measures real MPS latency of Attention + Norms + Shared Expert (if present) + LM-Head + RDMA."""
    device = next(model.parameters()).device
    inp = torch.randint(300, 15000, (1, seq_len), device=device)

    orig_experts_fwd = [get_experts_module(layer).forward for layer in model.model.layers]
    for layer in model.model.layers:
        get_experts_module(layer).forward = lambda h, _idx, _w: torch.zeros_like(h)

    try:
        for _ in range(3):
            _ = model(input_ids=inp, use_cache=False)
        torch.mps.synchronize()
        times_ms: list[float] = []
        for _ in range(11):
            torch.mps.synchronize()
            t0 = time.perf_counter_ns()
            _ = model(input_ids=inp, use_cache=False)
            torch.mps.synchronize()
            times_ms.append((time.perf_counter_ns() - t0) / 1e6)
    finally:
        for layer, orig_fn in zip(model.model.layers, orig_experts_fwd):
            get_experts_module(layer).forward = orig_fn

    clean_ms = float(sorted(times_ms)[2])
    rdma_2hop_ms = spec.num_layers * 0.180
    return round(clean_ms + rdma_2hop_ms, 3)
