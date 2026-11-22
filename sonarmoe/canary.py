"""Blind Offline Orthogonal Canary Codebook Synthesis (Model-Agnostic)."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sonarmoe.models import MoEArchSpec, forward_capture_router_logits, get_router_module


@torch.no_grad()
def synthesize_canary_codebook(
    model: Any,
    tokenizer: Any,
    spec: MoEArchSpec,
    num_probes: int = 32,
    b_canary_tile: int = 122,
    b_canary_naive32: int = 32,
    b_canary_naive16: int = 16,
) -> dict[str, Any]:
    """Builds a Rank-Concentrated Orthogonal Canary Codebook strictly from public weights."""
    t0 = time.perf_counter()
    device = next(model.parameters()).device

    emb = model.model.embed_tokens.weight
    layer0 = model.model.layers[0]
    normed0 = layer0.post_attention_layernorm(layer0.input_layernorm(emb))
    probs0 = F.softmax(F.linear(normed0, get_router_module(layer0).weight).float(), dim=-1)
    topk0 = torch.topk(probs0, spec.top_k, dim=-1).indices
    ranks0 = topk0 // spec.experts_per_rank

    vocab_size = probs0.shape[0]
    valid_mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
    valid_mask[:300] = False
    if vocab_size > 50250:
        # Exclude trailing special control tokens
        valid_mask[vocab_size - 512 :] = False

    cands_per_rank = max(24, (216 // spec.num_ranks))
    cand_set: list[int] = []
    seen: set[int] = set()
    for r in range(spec.num_ranks):
        rank_hits = (ranks0 == r).sum(dim=-1).float()
        rank_mass = probs0[
            :, r * spec.experts_per_rank : (r + 1) * spec.experts_per_rank
        ].sum(dim=-1)
        score_r = rank_hits + 2.0 * rank_mass
        score_r[~valid_mask] = -1.0
        top_r = torch.topk(score_r, k=96).indices.tolist()
        added_r = 0
        for tid in top_r:
            if tid in seen:
                continue
            try:
                tok_str = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            except Exception:
                continue
            if len(tok_str.strip()) >= 1 and tok_str.isprintable():
                seen.add(tid)
                cand_set.append(tid)
                added_r += 1
                if added_r >= cands_per_rank:
                    break

    cand_ids = torch.tensor(cand_set, dtype=torch.long, device=device)
    num_cands = len(cand_set)

    expert_fractions = np.zeros(
        (num_cands, spec.num_layers, spec.num_experts), dtype=np.float32
    )
    batch_sz = 12
    seqs = cand_ids[:, None].repeat(1, b_canary_tile)
    for start in range(0, num_cands, batch_sz):
        sub = seqs[start : start + batch_sz]
        bs = sub.shape[0]
        captured = forward_capture_router_logits(model, sub)
        for l_idx, rl_raw in enumerate(captured):
            rl = rl_raw.view(bs, b_canary_tile, spec.num_experts).float()
            topk_idx = torch.topk(F.softmax(rl, dim=-1), spec.top_k, dim=-1).indices
            for i in range(bs):
                counts = torch.bincount(
                    topk_idx[i].reshape(-1), minlength=spec.num_experts
                ).cpu().numpy()
                expert_fractions[start + i, l_idx] = counts / float(b_canary_tile)

    cliff_primed = (expert_fractions >= 0.85).astype(np.float32)
    rank_active = cliff_primed.reshape(
        num_cands, spec.num_layers, spec.num_ranks, spec.experts_per_rank
    ).sum(axis=-1)

    feat_vecs = cliff_primed.reshape(num_cands, -1)
    unit_feats = feat_vecs / (np.linalg.norm(feat_vecs, axis=1, keepdims=True) + 1e-8)

    selected_indices: list[int] = []
    gram_inv = np.eye(unit_feats.shape[1], dtype=np.float64) / 5e-2

    for probe_slot in range(num_probes):
        target_rank = probe_slot % spec.num_ranks
        best_idx = -1
        best_score = -1e18
        for c_idx in range(num_cands):
            if c_idx in selected_indices:
                continue
            r_curve = rank_active[c_idx]
            other_max = np.delete(r_curve, target_rank, axis=-1).max(axis=-1)
            straggler_margin = float((r_curve[:, target_rank] - other_max).mean())

            v = unit_feats[c_idx].astype(np.float64)
            quad = float(v @ (gram_inv @ v))
            d_opt_gain = math.log1p(max(quad, 0.0))

            max_cos_pen = 0.0
            if selected_indices:
                sims = unit_feats[selected_indices] @ unit_feats[c_idx]
                max_cos_pen = float(sims.max())

            score = 1.4 * d_opt_gain + 2.2 * straggler_margin - 2.0 * max_cos_pen
            if score > best_score:
                best_score = score
                best_idx = c_idx

        selected_indices.append(best_idx)
        v = unit_feats[best_idx].astype(np.float64)
        gv = gram_inv @ v
        gram_inv = gram_inv - np.outer(gv, gv) / (1.0 + float(v @ gv))

    sel_token_ids = [cand_set[i] for i in selected_indices]
    sel_tokens = [
        tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        for tid in sel_token_ids
    ]

    def get_exact_counts(seq_len: int) -> np.ndarray:
        arr = np.zeros(
            (num_probes, spec.num_layers, spec.num_experts), dtype=np.int32
        )
        t_tensor = torch.tensor(sel_token_ids, dtype=torch.long, device=device)[
            :, None
        ].repeat(1, seq_len)
        for start in range(0, num_probes, 12):
            sub = t_tensor[start : start + 12]
            bs = sub.shape[0]
            captured = forward_capture_router_logits(model, sub)
            for l_idx, rl_raw in enumerate(captured):
                rl = rl_raw.view(bs, seq_len, spec.num_experts).float()
                topk_idx = torch.topk(F.softmax(rl, dim=-1), spec.top_k, dim=-1).indices
                for m in range(bs):
                    arr[start + m, l_idx] = (
                        torch.bincount(
                            topk_idx[m].reshape(-1), minlength=spec.num_experts
                        )
                        .cpu()
                        .numpy()
                    )
        return arr

    counts_tile = get_exact_counts(b_canary_tile)
    counts_naive32 = get_exact_counts(b_canary_naive32)
    counts_naive16 = get_exact_counts(b_canary_naive16)

    total_slots = spec.num_layers * spec.num_experts
    distinct_covered = int(((counts_tile >= int(0.85 * b_canary_tile)).sum(axis=0) > 0).sum())

    return {
        "num_candidates_screened": num_cands,
        "num_probes": num_probes,
        "canary_batch_tokens": b_canary_tile,
        "probe_token_ids": sel_token_ids,
        "probe_tokens": sel_tokens,
        "target_ranks": [m % spec.num_ranks for m in range(num_probes)],
        "counts_tile": counts_tile,
        "counts_naive32": counts_naive32,
        "counts_naive16": counts_naive16,
        "distinct_layer_experts_covered": distinct_covered,
        "total_layer_expert_slots": total_slots,
        "coverage_pct": round(100.0 * distinct_covered / total_slots, 2),
        "max_single_expert_hits": int(counts_tile.max()),
        "synthesis_time_sec": round(time.perf_counter() - t0, 2),
    }
