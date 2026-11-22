"""Formal Channel-Capacity & Differential Privacy Bounds (Theorems 1 & 2).

Implements exact numerical evaluators for the two theoretical pillars in the paper:
  - Theorem 1 (Tile-Boundary Side-Channel Capacity Amplification):
    Computes exact binary/multi-step AWGN mutual information I(B; ΔITL) in bits/probe
    for Tile-Boundary Primed Canaries (step Δ_tile across M_tile) vs. Naive Unprimed
    Canaries (within-tile slope α_in).
  - Theorem 2 (OblivDispatch Sound Shifted Two-Sided Gaussian (ε, δ)-DP & MI Bound):
    Calibrates exact ρ-zCDP with split failure probability δ = δ_zCDP + δ_tail
    (where δ_tail = L * E * Φ(-μ_shift / σ_DP) bounds the union-bound probability
    of any slot underflowing c_{l,e} under two-sided Gaussian noise and data-independent
    clipping [N_min, N_max]), derives the exact MI upper bound I(Y; ΔITL_adv) <=
    min(log2 K, ρ / ln 2 + δ * log2 K), Fano Bayes accuracy ceiling, and Lipschitz
    perplexity perturbation bound.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def binary_entropy_bits(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def inverse_normal_cdf(p: float) -> float:
    """High-precision binary search for Φ^{-1}(p) with p in (0, 1)."""
    assert 0.0 < p < 1.0
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def zcdp_rho_for_eps_delta(epsilon: float, delta_zcdp: float) -> float:
    """Solves ε = ρ + 2 * sqrt(ρ * ln(1 / δ_zCDP)) for ρ > 0."""
    ln_inv = math.log(1.0 / delta_zcdp)
    u = -math.sqrt(ln_inv) + math.sqrt(ln_inv + epsilon)
    return u * u


def awgn_binary_step_mi_bits(
    step_amplitude_us: float,
    sigma_eff_us: float,
    prior_p: float = 0.5,
    num_grid: int = 2001,
) -> float:
    """Exact mutual information I(S; Y) in bits for S in {0, Δ} + N(0, σ_eff^2)."""
    if sigma_eff_us <= 1e-9:
        return binary_entropy_bits(prior_p)
    if abs(step_amplitude_us) <= 1e-12:
        return 0.0

    snr_half = abs(step_amplitude_us) / (2.0 * sigma_eff_us)
    if snr_half > 8.0:
        return binary_entropy_bits(prior_p)

    lo = -7.0 * sigma_eff_us
    hi = abs(step_amplitude_us) + 7.0 * sigma_eff_us
    y = np.linspace(lo, hi, num_grid, dtype=np.float64)
    dy = y[1] - y[0]

    inv_norm = 1.0 / (math.sqrt(2.0 * math.pi) * sigma_eff_us)
    p0 = inv_norm * np.exp(-0.5 * ((y - 0.0) / sigma_eff_us) ** 2)
    p1 = inv_norm * np.exp(-0.5 * ((y - abs(step_amplitude_us)) / sigma_eff_us) ** 2)
    pmix = (1.0 - prior_p) * p0 + prior_p * p1

    mask0 = (p0 > 1e-300) & (pmix > 1e-300)
    mask1 = (p1 > 1e-300) & (pmix > 1e-300)

    kl0 = float(np.sum(p0[mask0] * np.log2(p0[mask0] / pmix[mask0])) * dy)
    kl1 = float(np.sum(p1[mask1] * np.log2(p1[mask1] / pmix[mask1])) * dy)
    mi = (1.0 - prior_p) * kl0 + prior_p * kl1
    return float(max(0.0, min(binary_entropy_bits(prior_p), mi)))


def theorem1_tile_capacity_comparison(
    delta_tile_single_layer_us: float,
    within_tile_slope_us_per_tok: float,
    active_layers_hit: int,
    victim_token_delta: float,
    sigma_net_us: float,
    stream_repeats: int = 6,
) -> dict[str, float]:
    """Evaluates Theorem 1 exact per-probe channel capacity (Tile-Primed vs. Naive)."""
    sigma_eff = sigma_net_us / math.sqrt(max(stream_repeats, 1))
    tile_amp_us = active_layers_hit * delta_tile_single_layer_us
    naive_amp_us = active_layers_hit * within_tile_slope_us_per_tok * victim_token_delta

    c_tile = awgn_binary_step_mi_bits(tile_amp_us, sigma_eff, prior_p=0.5)
    c_naive = awgn_binary_step_mi_bits(naive_amp_us, sigma_eff, prior_p=0.5)
    ratio = c_tile / max(c_naive, 1e-6)
    return {
        "sigma_net_us": sigma_net_us,
        "sigma_eff_us": round(sigma_eff, 3),
        "tile_echo_amplitude_us": round(tile_amp_us, 2),
        "naive_echo_amplitude_us": round(naive_amp_us, 2),
        "capacity_tile_bits_per_probe": round(c_tile, 4),
        "capacity_naive_bits_per_probe": round(c_naive, 4),
        "capacity_gain_ratio": round(ratio, 2),
    }


def theorem2_oblivdispatch_dp_mi_bound(
    epsilon: float,
    delta: float = 1e-5,
    num_classes: int = 5,
    num_layers: int = 16,
    num_experts: int = 64,
    prob_margin_gamma: float = 0.00035,
    base_ppl: float = 28.5865,
    layer_residual_lip: float = 1.85,
) -> dict[str, Any]:
    """Evaluates Theorem 2 with split δ = δ_zCDP + δ_tail (δ/2 each).

    Under the sound shifted two-sided Gaussian mechanism:
      T_{l,e} = Clip_{[N_min, N_max]}(Round((1 - α_ε) c_{l,e} + α_ε τ_cap + μ_shift + Z_{l,e})),
    where Z_{l,e} ~ N(0, σ_DP^2), σ_DP = Δ_2(ε) / sqrt(2 * ρ), and
    μ_shift = σ_DP * Φ^{-1}(1 - δ_tail / (L * E)):
      - With probability >= 1 - δ_tail, no slot underflows c_{l,e}, so zero
        secret-dependent lower clipping occurs and the output is pure post-processing
        of a ρ-zCDP Gaussian mechanism (with δ_zCDP = δ/2).
      - By Bun & Steinke (2016) and Cuff & Yu (2016),
          I(Y; ΔITL_adv) <= min(log2 K, ρ / ln(2) + δ * log2(K)) bits.
    """
    h_max_bits = math.log2(num_classes)
    delta_zcdp = 0.5 * delta
    delta_tail = 0.5 * delta
    rho_zcdp = zcdp_rho_for_eps_delta(epsilon, delta_zcdp)

    # Required z-score for union-bound underflow probability <= δ_tail across L*E slots
    per_slot_tail = delta_tail / float(num_layers * num_experts)
    z_tail = inverse_normal_cdf(1.0 - per_slot_tail)

    # Exact chain-rule MI bound under split δ = δ_zCDP + δ_tail:
    #   I(Y; O) <= H_2(δ_tail) + (1 - δ_tail) * (ρ / ln 2) + δ_tail * log2(K)
    mi_upper_bits = min(
        h_max_bits,
        (1.0 - delta_tail) * (rho_zcdp / math.log(2.0))
        + delta_tail * h_max_bits
        + binary_entropy_bits(delta_tail),
    )

    # Fano's inequality ceiling on optimal Bayes accuracy
    max_bayes_acc = 1.0 / num_classes
    for acc_cand in np.linspace(1.0 / num_classes, 1.0, 2001):
        pe = 1.0 - float(acc_cand)
        fano_rhs = binary_entropy_bits(pe) + pe * math.log2(max(num_classes - 1, 1))
        required_mi = max(0.0, h_max_bits - fano_rhs)
        if required_mi <= mi_upper_bits + 1e-9:
            max_bayes_acc = float(acc_cand)
        else:
            break

    delta_nll_bound = 2.0 * num_layers * prob_margin_gamma * layer_residual_lip
    delta_ppl_upper = base_ppl * (math.exp(delta_nll_bound) - 1.0)

    return {
        "epsilon": epsilon,
        "delta": delta,
        "delta_zcdp": delta_zcdp,
        "delta_tail": delta_tail,
        "z_tail_sigma": round(z_tail, 3),
        "rho_zcdp_nats": round(rho_zcdp, 5),
        "theoretical_mi_upper_bits": round(mi_upper_bits, 4),
        "fano_bayes_accuracy_ceiling_pct": round(100.0 * max_bayes_acc, 2),
        "theoretical_delta_ppl_bound": round(delta_ppl_upper, 4),
    }
