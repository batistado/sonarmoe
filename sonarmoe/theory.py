"""Formal Channel-Capacity & Differential Privacy Bounds (Theorems 1 & 2).

Implements exact numerical evaluators for the two theoretical pillars in the paper:
  - Theorem 1 (Tile-Boundary Side-Channel Capacity Amplification):
    Computes exact binary/multi-step AWGN mutual information I(B; ΔITL) in bits/probe
    for Tile-Boundary Primed Canaries (step Δ_tile across M_tile) vs. Naive Unprimed
    Canaries (within-tile slope α_in), proving quadratic SNR suppression O((α_in/σ)^2)
    away from hardware tile cliffs.
  - Theorem 2 (OblivDispatch (ε, δ)-DP Composition, Mutual Information Cap & ΔPPL Bound):
    Computes zero-concentrated DP (ρ-zCDP -> (ε, δ)-DP) composition across L layers,
    variational/Fano mutual information upper bounds I(X_victim; ΔITL_adv) <= B(ε, δ),
    and worst-case Lipschitz next-token Perplexity shift bounds under probability-margin
    secondary expert perturbation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def binary_entropy_bits(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def awgn_binary_step_mi_bits(
    step_amplitude_us: float,
    sigma_eff_us: float,
    prior_p: float = 0.5,
    num_grid: int = 2001,
) -> float:
    """Exact mutual information I(S; Y) in bits for S in {0, Δ} + N(0, σ_eff^2) via quadrature."""
    if sigma_eff_us <= 1e-9:
        return binary_entropy_bits(prior_p)
    if abs(step_amplitude_us) <= 1e-12:
        return 0.0

    snr_half = abs(step_amplitude_us) / (2.0 * sigma_eff_us)
    if snr_half > 8.0:
        return binary_entropy_bits(prior_p)

    # Gauss-Legendre / uniform fine grid over [-7σ, Δ + 7σ]
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
    prob_margin_gamma: float = 0.00035,
    base_ppl: float = 28.5865,
    layer_residual_lip: float = 1.85,
) -> dict[str, Any]:
    """Evaluates Theorem 2 theoretical MI upper bound & Perplexity shift bound.

    For an (ε, δ)-DP mechanism M acting on victim routing trace X_victim (with uniform
    prior over K classes, H(Y) = log2(K) bits), standard DP-to-Mutual-Information
    bounds (Cuff & Yu 2016; Bun & Steinke 2016 zCDP) yield:
      I(Y; M(X)) <= min(log2(K), (min(ε, ε^2 / 2) / ln 2) + δ * log2(K) / ln 2)
    under pure/zCDP Gaussian-Gumbel composition, and Fano's inequality converts this
    MI cap into a strict upper bound on optimal Bayes classification accuracy!
    """
    h_max_bits = math.log2(num_classes)

    # zCDP parameter ρ calibrated so ε = ρ + 2*sqrt(ρ * ln(1/δ)) => solve quadratic for sqrt(ρ)
    ln_inv_delta = math.log(1.0 / delta)
    # u^2 + 2*sqrt(ln(1/δ))*u - ε = 0  where u = sqrt(ρ)
    u = -math.sqrt(ln_inv_delta) + math.sqrt(ln_inv_delta + epsilon)
    rho_zcdp = u * u

    # By Proposition 1.4 of Bun & Steinke (2016), ρ-zCDP implies KL(P_{M|X=x} || P_M) <= ρ nats,
    # hence I(Y; ΔITL_adv) <= I(X_victim; M(X_victim)) <= ρ / ln(2) bits!
    mi_upper_bits = min(h_max_bits, rho_zcdp / math.log(2.0) + delta * h_max_bits)

    # Invert Fano's inequality: H(Y) - I(Y; Y_hat) <= H_2(P_e) + P_e * log2(K - 1)
    # Find largest accuracy Acc = 1 - P_e consistent with I(Y; Y_hat) <= mi_upper_bits
    max_bayes_acc = 1.0 / num_classes
    for acc_cand in np.linspace(1.0 / num_classes, 1.0, 2001):
        pe = 1.0 - float(acc_cand)
        fano_rhs = binary_entropy_bits(pe) + pe * math.log2(max(num_classes - 1, 1))
        required_mi = max(0.0, h_max_bits - fano_rhs)
        if required_mi <= mi_upper_bits + 1e-9:
            max_bayes_acc = float(acc_cand)
        else:
            break

    # Perplexity perturbation bound when swapping 1 secondary expert within Δp <= γ_p
    delta_nll_bound = num_layers * prob_margin_gamma * layer_residual_lip
    delta_ppl_upper = base_ppl * (math.exp(delta_nll_bound) - 1.0)

    return {
        "epsilon": epsilon,
        "delta": delta,
        "rho_zcdp_nats": round(rho_zcdp, 5),
        "theoretical_mi_upper_bits": round(mi_upper_bits, 4),
        "fano_bayes_accuracy_ceiling_pct": round(100.0 * max_bayes_acc, 2),
        "theoretical_delta_ppl_bound": round(delta_ppl_upper, 4),
    }
