#!/usr/bin/env python3
"""Defense Baselines, Stage-by-Stage Component Ablations, & Expanded 2k-Token PPL Benchmark.

Addresses Q1/Q2 peer-review requirements on both OLMoE-1B-7B and Granite-3.1-MoE:
  1. Quantitative Defense Baselines (on Domain-5, sigma_net = 50 us, 3 seeds):
     - Undefended AFD
     - Pure Router Gumbel Noise (high-variance logit perturbation, b=0.80)
     - Uniform Worst-Case Padding (all E experts padded to B_vic = 64)
  2. OblivDispatch Stage-by-Stage Component Ablations (epsilon = 1.0, 3 seeds):
     - Stage 1 Only (probability-margin tail jitter)
     - Stage 1 + Stage 2 Only (margin jitter + per-tenant shadow capping)
     - Stage 2 + Stage 3 Only (shadow capping + shifted Gaussian DP, no Stage 1)
     - Full OblivDispatch (Stage 1 + Stage 2 + Stage 3)
  3. Expanded Multi-Domain Perplexity Benchmark (~2,000+ tokens across 24 passages, 3 seeds)
     comparing Undefended vs. Pure Router Noise vs. OblivDispatch (eps=2.0, 1.0, 0.5).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from moeleak_bench import CATEGORIES_5, MOELEAK_DOMAIN_100, PPL_EVAL_CORPUS
from sonarmoe import (
    MoEArchSpec,
    SUPPORTED_MOE_SPECS,
    apply_oblivdispatch_defense,
    evaluate_deconvolver_kfold,
    extract_fixed_window_traces,
    load_moe_on_mps,
    measure_non_routed_step_ms,
    profile_mps_expert_tile_curve,
    simulate_afd_itl_observations,
    synthesize_canary_codebook,
    unload_moe_from_mps,
)
from sonarmoe.models import get_router_module
from sonarmoe.oblivdispatch import compute_oblivdispatch_schedule

RESULTS_DIR = ROOT_DIR / "results"
EVAL_SEEDS = [20261118, 20261119, 20261120]
B_VICTIM = 64
B_CANARY_TILE = 122
NUM_PROBES = 32
STREAM_REPEATS = 6
DP_DELTA = 1e-5

# Expanded 24-passage (~2,000+ token) multi-domain technical & encyclopedic evaluation corpus
EXTRA_PPL_PASSAGES = [
    "In modern high-performance computing clusters, remote direct memory access over InfiniBand and RoCEv2 bypasses the host operating system kernel to transfer tensors directly between GPU high-bandwidth memory pools with microsecond tail latency.",
    "Sparse Mixture-of-Experts layers partition feed-forward network weights across expert-parallel ranks so each input token activates only a small top-k subset of parameters, decoupling total model capacity from per-token floating-point operations.",
    "Matrix multiplication kernels on modern graphics processors decompose the token batch dimension into fixed threadgroup tiles of sixty-four or one hundred twenty-eight rows to maximize shared-memory reuse and hide global memory latency.",
    "Differential privacy provides a mathematically rigorous bound on the information leaked about any individual input record by calibrating additive noise to the worst-case sensitivity of the released query or execution schedule.",
    "Zero-concentrated differential privacy simplifies privacy composition across multiple layers and iterations by bounding the Renyi divergence of all orders linearly in a single scalar parameter rho.",
    "Clinical decision support systems deployed in intensive care units must reconcile patient laboratory trends, renal clearance rates, and drug-drug interaction tables while maintaining strict confidentiality over electronic health records.",
    "Distributed relational databases optimize analytical aggregations through columnar storage layouts, vectorized query execution pipelines, predicate pushdown, and cost-based join reordering across partitioned worker nodes.",
    "Authenticated encryption with associated data such as AES-GCM combines counter-mode confidentiality with universal hash-based integrity verification, requiring strictly unique initialization vectors under a given symmetric key.",
    "Continuous batching schedulers for autoregressive language models interleave prefill chunks from newly arriving prompts with single-token decode iterations to prevent head-of-line blocking and maximize hardware utilization.",
    "In corporate mergers and acquisitions, indemnification clauses allocate post-closing liability for breaches of representations and warranties subject to negotiated baskets, caps, survival periods, and escrow holdbacks.",
    "Zero-knowledge proof systems allow a prover to convince a verifier that a computation was executed faithfully over private witness inputs without revealing any additional information beyond the validity of the statement.",
    "Modern operating system kernels schedule competing threads across symmetric multiprocessing cores using hierarchical run queues, priority inheritance protocols, and timer-driven preemption interrupts.",
    "Financial clearing networks process automated clearing house transfers in settlement batches that validate routing numbers, account balances, anti-money-laundering watchlists, and regulatory reporting thresholds.",
    "Compiler optimization passes transform intermediate representations through static single assignment form, dead-code elimination, loop tiling, register allocation, and instruction scheduling for superscalar pipelines.",
    "Genomic sequencing pipelines align millions of short nucleotide reads against a reference assembly using Burrows-Wheeler index structures, followed by variant calling and functional pathogenicity annotation.",
    "Congestion control algorithms in datacenter fabrics rely on explicit congestion notification marks and priority flow control thresholds to prevent switch buffer exhaustion during synchronized all-to-all collective traffic.",
]

EXPANDED_PPL_CORPUS = list(PPL_EVAL_CORPUS) + EXTRA_PPL_PASSAGES


def _apply_pure_router_noise(
    router_logits_all: np.ndarray,
    spec: MoEArchSpec,
    gumbel_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Baseline 1: Unbounded Gumbel logit noise across all top-k slots (no shadow/DP padding)."""
    num_q, num_l, b_tok, num_e = router_logits_all.shape
    counts = np.zeros((num_q, num_l, num_e), dtype=np.int32)
    for q in range(num_q):
        for l in range(num_l):
            rl = router_logits_all[q, l].astype(np.float32)
            u = np.clip(rng.random(rl.shape, dtype=np.float32), 1e-6, 1.0 - 1e-6)
            gumbel = -np.log(-np.log(u)) * gumbel_scale
            noisy = rl + gumbel
            topk_idx = np.argpartition(-noisy, spec.top_k, axis=-1)[:, : spec.top_k]
            counts[q, l] = np.bincount(topk_idx.reshape(-1), minlength=num_e)
    return counts


def _apply_uniform_padding(
    router_logits_all: np.ndarray,
    b_vic: int = 64,
) -> np.ndarray:
    """Baseline 2: Worst-case uniform static padding of every expert slot to B_vic."""
    num_q, num_l, _, num_e = router_logits_all.shape
    return np.full((num_q, num_l, num_e), b_vic, dtype=np.int32)


def _apply_ablation_variant(
    router_logits_all: np.ndarray,
    spec: MoEArchSpec,
    epsilon: float,
    delta: float,
    rng: np.random.Generator,
    use_stage1: bool,
    use_stage2: bool,
    use_stage3: bool,
) -> np.ndarray:
    """Runs controlled stage-by-stage ablations of OblivDispatch."""
    num_q, num_l, b_tok, num_e = router_logits_all.shape
    out_counts = np.zeros((num_q, num_l, num_e), dtype=np.int32)
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

    for q in range(num_q):
        for l in range(num_l):
            rl = router_logits_all[q, l].astype(np.float32)
            rl_shift = rl - rl.max(axis=-1, keepdims=True)
            probs = np.exp(rl_shift) / np.exp(rl_shift).sum(axis=-1, keepdims=True)
            sorted_idx = np.argsort(-probs, axis=-1)

            if use_stage1:
                core_idx = sorted_idx[:, :core_k]
                p_ref = np.take_along_axis(probs, sorted_idx[:, core_k : core_k + 1], axis=-1)
                near_tie = probs >= (p_ref - prob_margin_gamma)
                near_tie[np.arange(b_tok)[:, None], core_idx] = False
                u = np.clip(rng.random(rl.shape, dtype=np.float32), 1e-6, 1.0 - 1e-6)
                gumbel = -np.log(-np.log(u)) * gumbel_scale
                sec1 = np.argmax(np.where(near_tie, probs + gumbel, -1e9), axis=-1, keepdims=True)
                chosen = np.concatenate([core_idx, sec1], axis=-1)
            else:
                chosen = sorted_idx[:, : spec.top_k]

            work_counts = np.bincount(chosen.reshape(-1), minlength=num_e).astype(np.int32)

            if use_stage2:
                overflow = np.maximum(work_counts - cap_tau, 0)
                work_counts = np.minimum(work_counts, cap_tau)
                n_over = int(overflow.sum())
                if n_over > 0:
                    headroom = cap_tau - work_counts
                    avail = np.where(headroom > 0)[0]
                    order = np.lexsort((rng.random(len(avail)), work_counts[avail]))
                    for idx in avail[order]:
                        if n_over <= 0:
                            break
                        take = min(int(headroom[idx]), n_over)
                        work_counts[idx] += take
                        n_over -= take

            if use_stage3:
                gauss_noise = rng.normal(loc=0.0, scale=dp_sigma, size=num_e)
                noisy_real = (
                    (1.0 - level_alpha) * work_counts.astype(np.float64)
                    + level_alpha * float(cap_tau)
                    + mu_shift
                    + gauss_noise
                )
                work_counts = np.clip(np.round(noisy_real).astype(np.int32), n_min, n_max)

            out_counts[q, l] = work_counts
    return out_counts


@torch.no_grad()
def evaluate_expanded_ppl(
    model: Any,
    tokenizer: Any,
    spec: MoEArchSpec,
    corpus: list[str],
    pure_noise_scales: list[float],
    obliv_eps_list: list[float],
    seeds: list[int],
) -> dict[str, Any]:
    """Evaluates exact full-model PPL on the expanded 24-passage (~2k-token) corpus."""
    device = next(model.parameters()).device
    encoded = [tokenizer(t, return_tensors="pt")["input_ids"].to(device) for t in corpus]
    total_tokens = int(sum(x.shape[1] for x in encoded))
    core_k = spec.top_k - 1
    prob_margin_gamma = spec.prob_margin_gamma

    def run_ppl() -> float:
        total_nll = 0.0
        n_tok = 0
        for inp in encoded:
            out = model(input_ids=inp, use_cache=False)
            sl = out.logits[:, :-1, :].contiguous().float()
            lb = inp[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), lb.view(-1), reduction="sum")
            total_nll += float(loss.item())
            n_tok += int(lb.numel())
        return math.exp(total_nll / max(n_tok, 1))

    base_ppl = run_ppl()
    results: dict[str, Any] = {
        "num_passages": len(corpus),
        "total_tokens": total_tokens,
        "undefended_ppl": round(base_ppl, 4),
        "pure_router_noise": {},
        "oblivdispatch": {},
    }

    # 1. Pure Router Gumbel Noise baseline (unbounded across all top-k slots)
    for b_scale in pure_noise_scales:
        vals = []
        for s in seeds:
            gen = torch.Generator(device="cpu").manual_seed(s + int(b_scale * 1000))
            hooks = []

            def make_pure_hook(sc: float):
                def hook_fn(_m, _in, out):
                    router_logits, _, _ = out
                    rl_f = router_logits.float()
                    u = torch.clamp(
                        torch.rand(rl_f.shape, generator=gen, dtype=torch.float32).to(rl_f.device),
                        1e-6,
                        1.0 - 1e-6,
                    )
                    gumbel = -torch.log(-torch.log(u)) * sc
                    noisy_logits = rl_f + gumbel
                    probs = F.softmax(rl_f, dim=-1)
                    _, new_idx = torch.topk(noisy_logits, spec.top_k, dim=-1)
                    new_scores = torch.gather(probs, 1, new_idx)
                    if spec.norm_topk_prob:
                        new_scores = new_scores / new_scores.sum(dim=-1, keepdim=True)
                    return router_logits, new_scores.to(router_logits.dtype), new_idx

                return hook_fn

            for layer in model.model.layers:
                hooks.append(get_router_module(layer).register_forward_hook(make_pure_hook(b_scale)))
            try:
                vals.append(run_ppl())
            finally:
                for h in hooks:
                    h.remove()
        m_v = float(np.mean(vals))
        results["pure_router_noise"][f"b_{b_scale}"] = {
            "ppl_mean": round(m_v, 4),
            "ppl_std": round(float(np.std(vals)), 4),
            "delta_ppl": round(m_v - base_ppl, 4),
        }

    # 2. OblivDispatch Stage-1 margin-bounded tail jitter
    for eps in obliv_eps_list:
        sched = compute_oblivdispatch_schedule(spec, epsilon=eps, delta=DP_DELTA, b_vic=B_VICTIM)
        g_scale = float(sched["gumbel_scale"])
        vals = []
        for s in seeds:
            gen = torch.Generator(device="cpu").manual_seed(s + int(eps * 100))
            hooks = []

            def make_od_hook(sc: float):
                def hook_fn(_m, _in, out):
                    router_logits, _, _ = out
                    rl_f = router_logits.float()
                    probs = F.softmax(rl_f, dim=-1)
                    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                    core_idx = sorted_idx[:, :core_k]
                    core_vals = sorted_probs[:, :core_k]
                    p_ref = sorted_probs[:, core_k : core_k + 1]
                    near_tie = probs >= (p_ref - prob_margin_gamma)
                    near_tie.scatter_(1, core_idx, False)
                    u = torch.clamp(
                        torch.rand(probs.shape, generator=gen, dtype=torch.float32).to(probs.device),
                        1e-6,
                        1.0 - 1e-6,
                    )
                    gumbel = -torch.log(-torch.log(u)) * sc
                    cand = torch.where(near_tie, probs + gumbel, torch.full_like(probs, -1e9))
                    sec1_idx = torch.argmax(cand, dim=-1, keepdim=True)
                    sec1_vals = torch.gather(probs, 1, sec1_idx)
                    new_idx = torch.cat([core_idx, sec1_idx], dim=-1)
                    new_scores = torch.cat([core_vals, sec1_vals], dim=-1)
                    if spec.norm_topk_prob:
                        new_scores = new_scores / new_scores.sum(dim=-1, keepdim=True)
                    return router_logits, new_scores.to(router_logits.dtype), new_idx

                return hook_fn

            for layer in model.model.layers:
                hooks.append(get_router_module(layer).register_forward_hook(make_od_hook(g_scale)))
            try:
                vals.append(run_ppl())
            finally:
                for h in hooks:
                    h.remove()
        m_v = float(np.mean(vals))
        results["oblivdispatch"][f"eps_{eps}"] = {
            "ppl_mean": round(m_v, 4),
            "ppl_std": round(float(np.std(vals)), 4),
            "delta_ppl": round(m_v - base_ppl, 4),
        }

    return results


def eval_model_baselines(spec: MoEArchSpec) -> dict[str, Any]:
    print(f"\n=== Baselines & Ablations: {spec.model_id} ===")
    model, tokenizer, _, _ = load_moe_on_mps(spec)
    try:
        codebook = synthesize_canary_codebook(
            model, tokenizer, spec, num_probes=NUM_PROBES, b_canary_tile=B_CANARY_TILE
        )
        tile_prof = profile_mps_expert_tile_curve(model, spec, max_tokens=224, num_sweeps=18)
        non_routed_ms = measure_non_routed_step_ms(model, spec, seq_len=B_VICTIM)
        dom_traces = extract_fixed_window_traces(
            model, tokenizer, spec, MOELEAK_DOMAIN_100, b_victim=B_VICTIM
        )
        dom_undef = dom_traces["undefended_counts"]
        dom_logits = dom_traces["router_logits"]
        dom_labels = dom_traces["labels"]

        # Expanded 2k-token PPL benchmark
        ppl_res = evaluate_expanded_ppl(
            model,
            tokenizer,
            spec,
            EXPANDED_PPL_CORPUS,
            pure_noise_scales=[0.5, 1.0],
            obliv_eps_list=[2.0, 1.0, 0.5],
            seeds=EVAL_SEEDS,
        )
        print(
            f"  [Expanded PPL ({ppl_res['num_passages']} passages, {ppl_res['total_tokens']} tokens)] "
            f"Base={ppl_res['undefended_ppl']:.4f} | "
            f"PureNoise(b=0.5)={ppl_res['pure_router_noise']['b_0.5']['ppl_mean']:.4f} ({ppl_res['pure_router_noise']['b_0.5']['delta_ppl']:+.4f}) | "
            f"PureNoise(b=1.0)={ppl_res['pure_router_noise']['b_1.0']['ppl_mean']:.4f} ({ppl_res['pure_router_noise']['b_1.0']['delta_ppl']:+.4f}) | "
            f"OblivDispatch(eps=1.0)={ppl_res['oblivdispatch']['eps_1.0']['ppl_mean']:.4f} ({ppl_res['oblivdispatch']['eps_1.0']['delta_ppl']:+.4f})"
        )

        # Evaluate Undefended baseline TPOT & Acc (3 seeds)
        def eval_variant(name: str, count_fn, extra_tpot_ms: float = 0.0) -> dict[str, Any]:
            acc_g, mi_g, acc_s, mi_s, tpots = [], [], [], [], []
            for s in EVAL_SEEDS:
                s_rng = np.random.default_rng(s + abs(hash(name)) % 10000)
                cnts = count_fn(s_rng)
                dg, tpot = simulate_afd_itl_observations(
                    cnts,
                    codebook["counts_tile"],
                    spec,
                    tile_prof["samples_table_us"],
                    sigma_net_us=50.0,
                    stream_repeats=STREAM_REPEATS,
                    rng=s_rng,
                    combine_mode="global_barrier",
                    non_routed_step_ms=non_routed_ms,
                )
                mg = evaluate_deconvolver_kfold(dg, dom_labels, CATEGORIES_5)
                ds, _ = simulate_afd_itl_observations(
                    cnts,
                    codebook["counts_tile"],
                    spec,
                    tile_prof["samples_table_us"],
                    sigma_net_us=50.0,
                    stream_repeats=STREAM_REPEATS,
                    rng=s_rng,
                    combine_mode="selective_ibgda",
                    non_routed_step_ms=non_routed_ms,
                )
                ms = evaluate_deconvolver_kfold(ds, dom_labels, CATEGORIES_5)
                acc_g.append(mg["accuracy_pct"])
                mi_g.append(mg["empirical_mi_bits"])
                acc_s.append(ms["accuracy_pct"])
                mi_s.append(ms["empirical_mi_bits"])
                tpots.append(tpot + extra_tpot_ms)
            return {
                "variant": name,
                "global_acc_mean": round(float(np.mean(acc_g)), 2),
                "global_acc_std": round(float(np.std(acc_g)), 2),
                "global_mi_mean": round(float(np.mean(mi_g)), 4),
                "ibgda_acc_mean": round(float(np.mean(acc_s)), 2),
                "ibgda_acc_std": round(float(np.std(acc_s)), 2),
                "ibgda_mi_mean": round(float(np.mean(mi_s)), 4),
                "tpot_ms": round(float(np.mean(tpots)), 3),
            }

        variants = [
            eval_variant("undefended", lambda _r: dom_undef, 0.0),
            eval_variant(
                "pure_router_noise_b0.5",
                lambda r: _apply_pure_router_noise(dom_logits, spec, 0.5, r),
                0.04,
            ),
            eval_variant(
                "pure_router_noise_b1.0",
                lambda r: _apply_pure_router_noise(dom_logits, spec, 1.0, r),
                0.04,
            ),
            eval_variant(
                "uniform_padding_64",
                lambda _r: _apply_uniform_padding(dom_logits, B_VICTIM),
                0.0,
            ),
            eval_variant(
                "ablation_stage1_only",
                lambda r: _apply_ablation_variant(dom_logits, spec, 1.0, DP_DELTA, r, True, False, False),
                0.04,
            ),
            eval_variant(
                "ablation_stage1_stage2",
                lambda r: _apply_ablation_variant(dom_logits, spec, 1.0, DP_DELTA, r, True, True, False),
                0.08,
            ),
            eval_variant(
                "ablation_stage2_stage3_no_s1",
                lambda r: _apply_ablation_variant(dom_logits, spec, 1.0, DP_DELTA, r, False, True, True),
                0.10,
            ),
            eval_variant(
                "full_oblivdispatch_eps1.0",
                lambda r: apply_oblivdispatch_defense(dom_logits, spec, 1.0, DP_DELTA, r)[0],
                0.12,
            ),
        ]

        base_tpot = variants[0]["tpot_ms"]
        for v in variants:
            v["tpot_overhead_pct"] = round(100.0 * (v["tpot_ms"] - base_tpot) / base_tpot, 2)
            # Also report FFN token-compute multiplier vs undefended (512 real token-experts/layer)
            if v["variant"] == "uniform_padding_64":
                v["ffn_compute_multiplier"] = round(float(spec.num_experts) / float(spec.top_k), 2)
            elif "stage3" in v["variant"] or "full_oblivdispatch" in v["variant"]:
                sched = compute_oblivdispatch_schedule(spec, 1.0, DP_DELTA, B_VICTIM)
                avg_padded = float(sched["shadow_cap_tau"]) + float(sched["mu_shift"])
                v["ffn_compute_multiplier"] = round(
                    (spec.num_experts * avg_padded) / float(B_VICTIM * spec.top_k), 2
                )
            else:
                v["ffn_compute_multiplier"] = 1.0
            print(
                f"  [{v['variant']:28s}] Glob={v['global_acc_mean']:5.1f}±{v['global_acc_std']:4.1f}% "
                f"(MI={v['global_mi_mean']:.3f}b) | IBGDA={v['ibgda_acc_mean']:5.1f}% | "
                f"TPOT={v['tpot_ms']:.2f}ms ({v['tpot_overhead_pct']:+.2f}%) | FFN={v['ffn_compute_multiplier']:.2f}x"
            )

        return {"expanded_ppl": ppl_res, "variants": variants}
    finally:
        unload_moe_from_mps(model)


def main() -> None:
    t0 = time.perf_counter()
    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "models": {
            "olmoe_7b": eval_model_baselines(SUPPORTED_MOE_SPECS["olmoe_7b"]),
            "granite31_moe_1b": eval_model_baselines(SUPPORTED_MOE_SPECS["granite31_moe_1b"]),
        },
        "runtime_sec": round(time.perf_counter() - t0, 2),
    }
    out_path = RESULTS_DIR / "sonarmoe_baselines_ablations.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[Done] Saved baselines & ablations to {out_path} in {out['runtime_sec']}s")


if __name__ == "__main__":
    main()
