"""OblivDispatch (ε, δ)-DP Defense Engine & Disaggregated AFD Simulator."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sonarmoe.models import MoEArchSpec, get_router_module
from sonarmoe.theory import inverse_normal_cdf, zcdp_rho_for_eps_delta


def compute_oblivdispatch_schedule(
    spec: MoEArchSpec,
    epsilon: float,
    delta: float = 1e-5,
    b_vic: int = 64,
) -> dict[str, float | int]:
    """Computes the exact, reproducible OblivDispatch parameter schedule for (spec, ε, δ).

    Uses the sound shifted two-sided Gaussian mechanism with split failure probability
    δ = δ_zCDP + δ_tail (δ/2 each) and data-independent clipping [N_min, N_max]:
      1. Stage 2 caps every home expert at τ_cap > k * B_vic / E (guaranteeing 100%
         real-token conservation inside the E slots with 0 <= c_{l,e} <= τ_cap).
      2. Raw L-layer l2 sensitivity across neighboring B_vic-token queries is bounded by
         Δ_{2,raw} = sqrt(2 * L * τ_cap * min(k * B_vic, E * τ_cap - k * B_vic)).
      3. Stage 3 applies linear leveling (1 - α_ε) c_{l,e} + α_ε τ_cap + μ_shift + Z_{l,e},
         where Z_{l,e} ~ N(0, σ_DP^2) is two-sided Gaussian noise with exact calibration
         σ_DP = (1 - α_ε) * Δ_{2,raw} / sqrt(2 * ρ(ε, δ/2)), and
         μ_shift = σ_DP * Φ^{-1}(1 - (δ/2) / (L * E)) absorbs the union-bound underflow
         event {∃ (l,e): T_{l,e} < c_{l,e}} into δ_tail = δ/2.
    """
    prob_margin_gamma = spec.prob_margin_gamma
    gumbel_scale = float(np.clip(0.0020 / math.sqrt(epsilon), 0.0006, 0.0032))

    # Average routed tokens per expert per layer (OLMoE: 8.0, Granite: 16.0)
    total_layer_assign = spec.top_k * b_vic
    mean_per_expert = total_layer_assign / float(spec.num_experts)
    base_cap = math.ceil(mean_per_expert)

    if epsilon >= 8.0:
        cap_tau = base_cap + 8
        sigma_target = 2.0
    elif epsilon >= 4.0:
        cap_tau = base_cap + 5
        sigma_target = 2.0
    elif epsilon >= 2.0:
        cap_tau = base_cap + 3
        sigma_target = 1.8
    elif epsilon >= 1.0:
        cap_tau = base_cap + 2
        sigma_target = 1.6
    else:  # epsilon = 0.5
        cap_tau = base_cap + 1
        sigma_target = 1.5

    delta_zcdp = 0.5 * delta
    delta_tail = 0.5 * delta
    rho_zcdp = zcdp_rho_for_eps_delta(epsilon, delta_zcdp)

    # Exact L-layer query-level l2 sensitivity of Stage-2 counts c in [0, τ_cap]^{L x E}
    l1_half_cap = min(total_layer_assign, spec.num_experts * cap_tau - total_layer_assign)
    delta_2_raw = math.sqrt(2.0 * spec.num_layers * cap_tau * max(l1_half_cap, 0))

    # Calibrate leveling factor α_ε so that Δ_2(ε) = (1 - α_ε) Δ_{2,raw} = σ_DP * sqrt(2ρ)
    if delta_2_raw > 0.0:
        one_minus_alpha = min(1.0, (sigma_target * math.sqrt(2.0 * rho_zcdp)) / delta_2_raw)
        level_alpha = 1.0 - one_minus_alpha
        delta_2_eff = one_minus_alpha * delta_2_raw
        dp_sigma = delta_2_eff / math.sqrt(2.0 * rho_zcdp)
    else:
        level_alpha = 1.0
        delta_2_eff = 0.0
        dp_sigma = 0.0

    # Union-bound tail shift μ_shift so Pr[∃(l,e): Round(U_{l,e}) < c_{l,e}] <= δ_tail
    per_slot_tail = delta_tail / float(spec.num_layers * spec.num_experts)
    z_tail = inverse_normal_cdf(1.0 - per_slot_tail)
    mu_shift = dp_sigma * z_tail

    # Data-independent clipping bounds [N_min, N_max] (N_min=1 closes kernel-skip channel)
    n_min = 1
    n_max = int(math.ceil(cap_tau + 2.0 * mu_shift + 2.0))

    return {
        "epsilon": float(epsilon),
        "delta": float(delta),
        "delta_zcdp": float(delta_zcdp),
        "delta_tail": float(delta_tail),
        "rho_zcdp_nats": round(rho_zcdp, 5),
        "gumbel_scale": round(gumbel_scale, 5),
        "prob_margin_gamma": prob_margin_gamma,
        "shadow_cap_tau": int(cap_tau),
        "level_alpha": round(level_alpha, 5),
        "delta_2_raw": round(delta_2_raw, 3),
        "delta_2_eff": round(delta_2_eff, 4),
        "dp_sigma": round(dp_sigma, 4),
        "z_tail_sigma": round(z_tail, 3),
        "mu_shift": round(mu_shift, 3),
        "n_min": int(n_min),
        "n_max": int(n_max),
        "idle_cover_prob": 1.0,
    }


def apply_oblivdispatch_defense(
    router_logits_all: np.ndarray,
    spec: MoEArchSpec,
    epsilon: float,
    delta: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Applies sound per-tenant OblivDispatch (ε, δ)-DP defense to victim dispatch.

    Implements:
      1. Probability-Margin-Bounded Secondary Expert Gumbel Jitter (Δp <= γ_p)
      2. Per-Tenant Shadow-Expert Overflow Routing into Victim A's own lightest local
         slots (capped at τ_cap, preserving 100% of real tokens with zero cross-tenant
         inspection of Attacker B's workload)
      3. Sound Shifted Two-Sided Gaussian Count Blinding with Data-Independent Clipping
         [N_min, N_max] and Split Failure Budget δ = δ_zCDP + δ_tail
    """
    num_q, num_l, b_tok, num_e = router_logits_all.shape
    defended_counts = np.zeros((num_q, num_l, num_e), dtype=np.int32)

    sched = compute_oblivdispatch_schedule(spec, epsilon=epsilon, delta=delta, b_vic=b_tok)
    prob_margin_gamma = float(sched["prob_margin_gamma"])
    gumbel_scale = float(sched["gumbel_scale"])
    cap_tau = int(sched["shadow_cap_tau"])
    level_alpha = float(sched["level_alpha"])
    dp_sigma = float(sched["dp_sigma"])
    mu_shift = float(sched["mu_shift"])
    n_min = int(sched["n_min"])
    n_max = int(sched["n_max"])

    core_k = spec.top_k - 1
    total_real_tokens = float(num_q * num_l * b_tok * spec.top_k)
    total_shadow_redirected = 0.0
    total_dummy_padded = 0.0
    total_slots = num_q * num_l * num_e
    underflow_slots = 0

    for q_idx in range(num_q):
        for l_idx in range(num_l):
            rl = router_logits_all[q_idx, l_idx].astype(np.float32)
            rl_shift = rl - rl.max(axis=-1, keepdims=True)
            probs = np.exp(rl_shift) / np.exp(rl_shift).sum(axis=-1, keepdims=True)
            sorted_idx = np.argsort(-probs, axis=-1)
            core_idx = sorted_idx[:, :core_k]

            p_ref = np.take_along_axis(probs, sorted_idx[:, core_k : core_k + 1], axis=-1)
            near_tie_mask = probs >= (p_ref - prob_margin_gamma)
            row_idx = np.arange(b_tok)[:, None]
            near_tie_mask[row_idx, core_idx] = False

            u_rand = np.clip(rng.random(rl.shape, dtype=np.float32), 1e-6, 1.0 - 1e-6)
            gumbel = -np.log(-np.log(u_rand)) * gumbel_scale
            cand_scores = np.where(near_tie_mask, probs + gumbel, -1e9)
            sec1 = np.argmax(cand_scores, axis=-1, keepdims=True)

            chosen = np.concatenate([core_idx, sec1], axis=-1)
            raw_counts = np.bincount(chosen.reshape(-1), minlength=num_e).astype(np.int32)

            # Stage 2: Per-Tenant Shadow-Expert Overflow into Victim A's own lightest slots
            # Every slot is strictly capped at cap_tau, conserving 100% of real tokens.
            overflow = np.maximum(raw_counts - cap_tau, 0)
            work_counts = np.minimum(raw_counts, cap_tau)
            n_over = int(overflow.sum())
            total_shadow_redirected += n_over

            if n_over > 0:
                # Pseudorandomly fill available headroom (cap_tau - work_counts) on lightest slots
                headroom = cap_tau - work_counts
                avail_idx = np.where(headroom > 0)[0]
                # Shuffle ties via per-tick CSPRNG/PRNG while favoring lightest slots
                tie_break = rng.random(len(avail_idx))
                order = np.lexsort((tie_break, work_counts[avail_idx]))
                for idx in avail_idx[order]:
                    if n_over <= 0:
                        break
                    take = min(int(headroom[idx]), n_over)
                    work_counts[idx] += take
                    n_over -= take

            # Stage 3: Sound Shifted Two-Sided Gaussian + Data-Independent Clipping [N_min, N_max]
            gauss_noise = rng.normal(loc=0.0, scale=dp_sigma, size=num_e)
            noisy_real = (
                (1.0 - level_alpha) * work_counts.astype(np.float64)
                + level_alpha * float(cap_tau)
                + mu_shift
                + gauss_noise
            )
            padded = np.clip(np.round(noisy_real).astype(np.int32), n_min, n_max)
            underflow_slots += int(np.sum(padded < work_counts))
            total_dummy_padded += float(np.maximum(padded - work_counts, 0).sum())
            defended_counts[q_idx, l_idx] = padded

    stats: dict[str, float | int] = dict(sched)
    stats.update(
        {
            "shadow_redirect_fraction_pct": round(
                100.0 * total_shadow_redirected / total_real_tokens, 2
            ),
            "dummy_token_overhead_ratio": round(
                total_dummy_padded / total_real_tokens, 3
            ),
            "empirical_underflow_slots": int(underflow_slots),
            "empirical_underflow_rate": round(underflow_slots / float(max(total_slots, 1)), 8),
        }
    )
    return defended_counts, stats


@torch.no_grad()
def evaluate_corpus_perplexity(
    model: Any,
    tokenizer: Any,
    spec: MoEArchSpec,
    corpus: list[str],
    gumbel_scales: dict[float, float],
    seed: int = 20261118,
) -> dict[str, float]:
    """Evaluates exact full-model Perplexity on `corpus` with live router hooks."""
    device = next(model.parameters()).device
    encoded_batches = [
        tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        for text in corpus
    ]
    core_k = spec.top_k - 1
    prob_margin_gamma = spec.prob_margin_gamma

    def compute_ppl() -> float:
        total_nll = 0.0
        total_toks = 0
        for inp in encoded_batches:
            out = model(input_ids=inp, use_cache=False)
            shift_logits = out.logits[:, :-1, :].contiguous().float()
            shift_labels = inp[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_nll += float(loss.item())
            total_toks += int(shift_labels.numel())
        return math.exp(total_nll / max(total_toks, 1))

    base_ppl = compute_ppl()
    ppl_by_eps: dict[str, float] = {"undefended": round(base_ppl, 4)}

    for eps, g_scale in gumbel_scales.items():
        gen = torch.Generator(device="cpu").manual_seed(seed + int(eps * 100))
        hooks = []

        def make_hook(scale_val: float):
            def hook_fn(_module, _inputs, outputs):
                router_logits, _, _ = outputs
                rl_f = router_logits.float()
                probs = F.softmax(rl_f, dim=-1)

                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                core_idx = sorted_idx[:, :core_k]
                core_vals = sorted_probs[:, :core_k]

                p_ref = sorted_probs[:, core_k : core_k + 1]
                near_tie = probs >= (p_ref - prob_margin_gamma)
                near_tie.scatter_(1, core_idx, False)

                u = torch.rand(probs.shape, generator=gen, dtype=torch.float32).to(
                    probs.device
                )
                u = torch.clamp(u, 1e-6, 1.0 - 1e-6)
                gumbel = -torch.log(-torch.log(u)) * scale_val
                cand_scores = torch.where(
                    near_tie, probs + gumbel, torch.full_like(probs, -1e9)
                )
                sec1_idx = torch.argmax(cand_scores, dim=-1, keepdim=True)
                sec1_vals = torch.gather(probs, 1, sec1_idx)

                new_idx = torch.cat([core_idx, sec1_idx], dim=-1)
                new_scores = torch.cat([core_vals, sec1_vals], dim=-1)
                if spec.norm_topk_prob:
                    new_scores = new_scores / new_scores.sum(dim=-1, keepdim=True)
                return router_logits, new_scores.to(router_logits.dtype), new_idx

            return hook_fn

        for layer in model.model.layers:
            hooks.append(get_router_module(layer).register_forward_hook(make_hook(g_scale)))

        try:
            def_ppl = compute_ppl()
        finally:
            for h in hooks:
                h.remove()

        ppl_by_eps[f"eps_{eps}"] = round(def_ppl, 4)

    return ppl_by_eps


def simulate_afd_itl_observations(
    victim_counts: np.ndarray,
    canary_counts: np.ndarray,
    spec: MoEArchSpec,
    samples_table_us: np.ndarray,
    sigma_net_us: float,
    stream_repeats: int,
    rng: np.random.Generator,
    combine_mode: str = "global_barrier",
    non_routed_step_ms: float = 22.5,
) -> tuple[np.ndarray, float]:
    """Computes remote HTTP ΔITL observations from real M5 Pro MPS kernel samples."""
    num_q = victim_counts.shape[0]
    num_m = canary_counts.shape[0]
    max_n, num_trials = samples_table_us.shape
    max_idx = max_n - 1
    num_l = spec.num_layers
    num_r = spec.num_ranks
    e_per_r = spec.experts_per_rank

    tot_counts = np.minimum(
        victim_counts[:, None, :, :] + canary_counts[None, :, :, :], max_idx
    )

    base_counts = np.minimum(canary_counts, max_idx)
    base_med = np.median(samples_table_us, axis=1)
    base_med[0] = 0.0
    base_rank_us = (
        base_med[base_counts].reshape(num_m, num_l, num_r, e_per_r).sum(axis=-1)
    )

    canary_rank_heavy = (
        (canary_counts >= (canary_counts.max() * 0.8))
        .reshape(num_m, num_l, num_r, e_per_r)
        .sum(axis=-1)
        >= 2
    )
    canary_rank_any = (
        canary_counts.reshape(num_m, num_l, num_r, e_per_r).sum(axis=-1) > 0
    )
    sel_mask = np.where(
        canary_rank_heavy.any(axis=-1, keepdims=True),
        canary_rank_heavy,
        canary_rank_any,
    )

    if combine_mode == "global_barrier":
        baseline_itl_us = base_rank_us.max(axis=-1).sum(axis=-1)
    else:
        baseline_itl_us = np.where(sel_mask, base_rank_us, 0.0).max(axis=-1).sum(axis=-1)

    delta_itl_accum = np.zeros((num_q, num_m), dtype=np.float64)

    for _ in range(stream_repeats):
        trial_choices = rng.integers(0, num_trials, size=tot_counts.shape)
        lat_e = samples_table_us[tot_counts, trial_choices]
        lat_e = np.where(tot_counts > 0, lat_e, 0.0)
        rank_us = lat_e.reshape(num_q, num_m, num_l, num_r, e_per_r).sum(axis=-1)

        if combine_mode == "global_barrier":
            step_itl_us = rank_us.max(axis=-1).sum(axis=-1)
        else:
            step_itl_us = (
                np.where(sel_mask[None, :, :, :], rank_us, 0.0).max(axis=-1).sum(axis=-1)
            )

        net_jitter = rng.normal(loc=0.0, scale=sigma_net_us, size=(num_q, num_m))
        delta_itl_accum += (step_itl_us + net_jitter) - baseline_itl_us[None, :]

    mean_delta_itl = (delta_itl_accum / float(stream_repeats)).astype(np.float32)

    vic_clamped = np.minimum(victim_counts, max_idx)
    vic_lat_e = np.where(vic_clamped > 0, base_med[vic_clamped], 0.0)
    vic_rank_us = vic_lat_e.reshape(num_q, num_l, num_r, e_per_r).sum(axis=-1)
    mean_tpot_ms = (
        non_routed_step_ms + float(vic_rank_us.max(axis=-1).sum(axis=-1).mean()) / 1000.0
    )

    return mean_delta_itl, mean_tpot_ms
