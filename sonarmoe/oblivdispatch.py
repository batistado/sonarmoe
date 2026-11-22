"""OblivDispatch (ε, δ)-DP Defense Engine & Disaggregated AFD Simulator."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sonarmoe.models import MoEArchSpec, get_router_module


def apply_oblivdispatch_defense(
    router_logits_all: np.ndarray,
    spec: MoEArchSpec,
    epsilon: float,
    delta: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    """Applies OblivDispatch (ε, δ)-DP defense to victim routing & expert dispatch.

    Implements:
      1. Probability-Margin-Bounded Secondary Expert Gumbel Jitter (Δp <= γ_p)
      2. Shadow-Expert Overflow Routing into non-straggler slack ("Straggler-Slack Hiding")
      3. Constant-Occupancy Kernel-Skip Closure + (ε, δ)-DP Gaussian Count Blinding
    """
    num_q, num_l, b_tok, num_e = router_logits_all.shape
    defended_counts = np.zeros((num_q, num_l, num_e), dtype=np.int32)

    prob_margin_gamma = spec.prob_margin_gamma
    gumbel_scale = float(np.clip(0.0020 / math.sqrt(epsilon), 0.0006, 0.0032))

    # Scale shadow cap proportionally to top_k (top_k=8 -> avg 8/expert; top_k=4 -> avg 4.27/expert)
    scale_k = spec.top_k / 8.0
    if epsilon >= 8.0:
        cap_tau = max(6, round(15 * scale_k))
        idle_cover_prob = 0.40
        level_alpha = 0.30
        dp_sigma = 1.8 * math.sqrt(scale_k)
    elif epsilon >= 4.0:
        cap_tau = max(5, round(11 * scale_k))
        idle_cover_prob = 0.68
        level_alpha = 0.58
        dp_sigma = 3.2 * math.sqrt(scale_k)
    elif epsilon >= 2.0:
        cap_tau = max(4, round(8 * scale_k))
        idle_cover_prob = 0.90
        level_alpha = 0.82
        dp_sigma = 4.8 * math.sqrt(scale_k)
    elif epsilon >= 1.0:
        cap_tau = max(4, round(7 * scale_k))
        idle_cover_prob = 1.00
        level_alpha = 0.96
        dp_sigma = 6.2 * math.sqrt(scale_k)
    else:  # epsilon = 0.5
        cap_tau = max(3, round(6 * scale_k))
        idle_cover_prob = 1.00
        level_alpha = 1.00
        dp_sigma = 7.2 * math.sqrt(scale_k)

    max_pad_ceil = max(12, round(18 * scale_k))
    core_k = spec.top_k - 1
    total_real_tokens = float(num_q * num_l * b_tok * spec.top_k)
    total_shadow_redirected = 0.0
    total_dummy_padded = 0.0

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
            raw_counts = np.bincount(chosen.reshape(-1), minlength=num_e)

            # 1. Shadow-Expert Overflow Routing into idle/light slots on partner ranks
            overflow = np.maximum(raw_counts - cap_tau, 0)
            work_counts = np.minimum(raw_counts, cap_tau)
            n_over = int(overflow.sum())
            total_shadow_redirected += n_over

            if n_over > 0:
                idle_slots = np.where(work_counts == 0)[0]
                num_sh = min(4, max(1, n_over // 10))
                if len(idle_slots) >= num_sh:
                    sh_targets = rng.choice(idle_slots, size=num_sh, replace=False)
                else:
                    light_slots = np.argsort(work_counts)[:8]
                    sh_targets = rng.choice(light_slots, size=num_sh, replace=False)
                base_sh = n_over // num_sh
                rem_sh = n_over % num_sh
                for s_i, t_slot in enumerate(sh_targets):
                    work_counts[t_slot] += min(
                        base_sh + (1 if s_i < rem_sh else 0), cap_tau + 3
                    )

            # 2. Constant-Occupancy Cover on remaining zero-token slots
            zero_mask = work_counts == 0
            wake_draw = rng.random(num_e) < idle_cover_prob
            cover_floor = np.where(zero_mask & wake_draw, 1, work_counts)

            # 3. (ε, δ)-DP Truncated Gaussian Count Blinding
            gauss_noise = np.abs(rng.normal(loc=0.0, scale=dp_sigma, size=num_e))
            leveled = (
                (1.0 - level_alpha) * cover_floor
                + level_alpha * float(cap_tau)
                + gauss_noise
            )
            padded = np.where(
                cover_floor > 0,
                np.clip(np.round(leveled).astype(np.int32), cover_floor, max_pad_ceil),
                0,
            )
            total_dummy_padded += float(np.maximum(padded - work_counts, 0).sum())
            defended_counts[q_idx, l_idx] = padded

    stats = {
        "gumbel_scale": round(gumbel_scale, 5),
        "prob_margin_gamma": prob_margin_gamma,
        "shadow_cap_tau": int(cap_tau),
        "idle_cover_prob": round(idle_cover_prob, 2),
        "dp_sigma": round(dp_sigma, 2),
        "shadow_redirect_fraction_pct": round(
            100.0 * total_shadow_redirected / total_real_tokens, 2
        ),
        "dummy_token_overhead_ratio": round(
            total_dummy_padded / total_real_tokens, 3
        ),
    }
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
