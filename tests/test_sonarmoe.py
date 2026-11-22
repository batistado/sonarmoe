"""Unit and Integration Tests for SonarMoE, OblivDispatch, Theory & MoELeak-Bench."""

from __future__ import annotations

import http.client
import json
import threading
import numpy as np
import pytest
import torch
import torch.nn as nn

from moeleak_bench import (
    CATEGORIES_5,
    MCP_TOOL_SCHEMAS_20,
    MCP_TOOLS_20_PROMPTS,
    MOELEAK_DOMAIN_100,
    PPL_EVAL_CORPUS,
)
from sonarmoe.deconvolver import evaluate_deconvolver_kfold
from sonarmoe.http_loopback_afd import run_live_http_loopback_benchmark
from sonarmoe.models import MoEArchSpec, SUPPORTED_MOE_SPECS
from sonarmoe.oblivdispatch import (
    apply_oblivdispatch_defense,
    simulate_afd_itl_observations,
)
from sonarmoe.theory import (
    awgn_binary_step_mi_bits,
    theorem1_tile_capacity_comparison,
    theorem2_oblivdispatch_dp_mi_bound,
)


def test_moeleak_bench_integrity() -> None:
    assert len(CATEGORIES_5) == 5
    assert len(MOELEAK_DOMAIN_100) == 100
    assert len(MCP_TOOL_SCHEMAS_20) == 20
    assert len(MCP_TOOLS_20_PROMPTS) == 160
    assert len(PPL_EVAL_CORPUS) >= 8

    # Verify balanced class distribution
    dom_counts = np.bincount([int(x["category_idx"]) for x in MOELEAK_DOMAIN_100])
    assert np.all(dom_counts == 20)
    mcp_counts = np.bincount([int(x["category_idx"]) for x in MCP_TOOLS_20_PROMPTS])
    assert np.all(mcp_counts == 8)

    # Verify non-trivial prompt lengths (all >= 240 chars)
    assert min(len(str(x["prompt"])) for x in MOELEAK_DOMAIN_100) >= 200
    assert min(len(str(x["prompt"])) for x in MCP_TOOLS_20_PROMPTS) >= 240


def test_theorem1_tile_capacity_amplification() -> None:
    # Zero step -> zero mutual information
    assert awgn_binary_step_mi_bits(0.0, 20.0) == 0.0
    # High SNR -> approaches 1.0 bit
    assert awgn_binary_step_mi_bits(500.0, 10.0) > 0.999

    # Tile-sized probe (+35 us/layer across 6 layers) strictly dominates naive within-tile slope
    res = theorem1_tile_capacity_comparison(
        delta_tile_single_layer_us=35.0,
        within_tile_slope_us_per_tok=0.18,
        active_layers_hit=6,
        victim_token_delta=15.0,
        sigma_net_us=50.0,
        stream_repeats=6,
    )
    assert res["capacity_tile_bits_per_probe"] > 0.95
    assert res["capacity_naive_bits_per_probe"] < 0.25
    assert res["capacity_gain_ratio"] > 4.0


def test_theorem2_dp_mi_monotonicity() -> None:
    b_hi = theorem2_oblivdispatch_dp_mi_bound(epsilon=8.0, delta=1e-5, num_classes=5)
    b_mid = theorem2_oblivdispatch_dp_mi_bound(epsilon=2.0, delta=1e-5, num_classes=5)
    b_lo = theorem2_oblivdispatch_dp_mi_bound(epsilon=0.5, delta=1e-5, num_classes=5)

    assert b_lo["theoretical_mi_upper_bits"] < b_mid["theoretical_mi_upper_bits"]
    assert b_mid["theoretical_mi_upper_bits"] <= b_hi["theoretical_mi_upper_bits"]
    assert b_lo["fano_bayes_accuracy_ceiling_pct"] < b_hi["fano_bayes_accuracy_ceiling_pct"]
    assert b_mid["theoretical_delta_ppl_bound"] < 0.40


def test_oblivdispatch_token_conservation_and_defense() -> None:
    spec = SUPPORTED_MOE_SPECS["olmoe_7b"]
    rng = np.random.default_rng(42)

    # Synthetic structured logits across 25 queries (5 per class)
    num_q = 25
    logits = rng.normal(0.0, 0.03, size=(num_q, spec.num_layers, 64, spec.num_experts)).astype(
        np.float16
    )
    for q in range(num_q):
        cls = q // 5
        hot_e = cls * 10
        logits[q, :, :, hot_e : hot_e + 4] += 0.25

    def_counts_1, stats_1 = apply_oblivdispatch_defense(
        logits, spec, epsilon=1.0, delta=1e-5, rng=rng
    )
    assert def_counts_1.shape == (num_q, spec.num_layers, spec.num_experts)
    # At epsilon=1.0, idle_cover_prob == 1.0 -> every expert executes >= 1 token (zero kernel-skip leak!)
    assert int((def_counts_1 == 0).sum()) == 0
    # Total dispatched tokens >= real routed tokens (zero dropped real tokens!)
    min_per_layer = 64 * spec.top_k
    assert np.all(def_counts_1.sum(axis=-1) >= min_per_layer)
    assert 0.0 < stats_1["shadow_redirect_fraction_pct"] < 100.0


def test_deconvolver_random_floor_vs_separable_signal() -> None:
    rng = np.random.default_rng(123)
    labels = np.repeat(np.arange(5), 20)

    # Pure noise -> near 20% random-guess floor, near-zero MI
    pure_noise = rng.normal(0.0, 1.0, size=(100, 32)).astype(np.float32)
    m_noise = evaluate_deconvolver_kfold(pure_noise, labels, CATEGORIES_5)
    assert m_noise["accuracy_pct"] <= 34.0
    assert m_noise["empirical_mi_bits"] <= 0.20

    # Separable class prototypes -> >95% accuracy
    prototypes = rng.normal(0.0, 3.0, size=(5, 32)).astype(np.float32)
    separable = prototypes[labels] + rng.normal(0.0, 0.4, size=(100, 32)).astype(np.float32)
    m_sep = evaluate_deconvolver_kfold(separable, labels, CATEGORIES_5)
    assert m_sep["accuracy_pct"] >= 95.0
    assert m_sep["empirical_mi_bits"] >= 1.90


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires Apple MPS")
def test_live_http_loopback_sse_roundtrip() -> None:
    # Miniature 2-layer, 8-expert MPS mock MoE to test real TCP/HTTP chunked SSE framing
    mini_spec = MoEArchSpec(
        key="mini_mps",
        model_id="mock",
        num_layers=2,
        num_experts=8,
        top_k=2,
        num_ranks=2,
        has_shared_expert=False,
        norm_topk_prob=False,
        prob_margin_gamma=0.0005,
    )

    class _MiniExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(
                torch.randn(8, 256, 128, device="mps", dtype=torch.float16) * 0.02
            )
            self.down_proj = nn.Parameter(
                torch.randn(8, 128, 128, device="mps", dtype=torch.float16) * 0.02
            )

    class _MiniLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.experts = _MiniExperts()

    class _MiniModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Cfg", (), {"hidden_size": 128})()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([_MiniLayer(), _MiniLayer()])

    mini_model = _MiniModel()
    vic = np.ones((3, 2, 8), dtype=np.int32) * 4
    can = np.ones((4, 2, 8), dtype=np.int32) * 16
    res = run_live_http_loopback_benchmark(
        mini_model, mini_spec, vic, can, stream_tokens_per_probe=2
    )
    assert res["delta_itl_us"].shape == (3, 4)
    assert res["total_http_sse_frames_streamed"] == (3 + 1) * 4 * 3
