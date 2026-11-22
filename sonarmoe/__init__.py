"""SonarMoE & OblivDispatch Core Library."""

from __future__ import annotations

from sonarmoe.canary import synthesize_canary_codebook
from sonarmoe.deconvolver import evaluate_deconvolver_kfold
from sonarmoe.http_loopback_afd import run_live_http_loopback_benchmark
from sonarmoe.models import (
    MoEArchSpec,
    SUPPORTED_MOE_SPECS,
    extract_fixed_window_traces,
    load_moe_on_mps,
    measure_non_routed_step_ms,
    profile_mps_expert_tile_curve,
    unload_moe_from_mps,
)
from sonarmoe.oblivdispatch import (
    apply_oblivdispatch_defense,
    evaluate_corpus_perplexity,
    simulate_afd_itl_observations,
)
from sonarmoe.theory import (
    awgn_binary_step_mi_bits,
    theorem1_tile_capacity_comparison,
    theorem2_oblivdispatch_dp_mi_bound,
)

__all__ = [
    "MoEArchSpec",
    "SUPPORTED_MOE_SPECS",
    "load_moe_on_mps",
    "unload_moe_from_mps",
    "extract_fixed_window_traces",
    "profile_mps_expert_tile_curve",
    "measure_non_routed_step_ms",
    "synthesize_canary_codebook",
    "apply_oblivdispatch_defense",
    "evaluate_corpus_perplexity",
    "simulate_afd_itl_observations",
    "evaluate_deconvolver_kfold",
    "run_live_http_loopback_benchmark",
    "awgn_binary_step_mi_bits",
    "theorem1_tile_capacity_comparison",
    "theorem2_oblivdispatch_dp_mi_bound",
]
