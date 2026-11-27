#!/usr/bin/env python3
"""Generates Fig. 6: Defense Baselines & Stage-by-Stage Component Ablation Figure
strictly from `results/sonarmoe_baselines_ablations.json`."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
IN_JSON = ROOT_DIR / "results" / "sonarmoe_baselines_ablations.json"
PAPER_FIG_DIR = ROOT_DIR / "paper" / "figures"
RESULTS_DIR = ROOT_DIR / "results"


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.8,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 220,
        }
    )
    data = json.loads(IN_JSON.read_text(encoding="utf-8"))
    ol = data["models"]["olmoe_7b"]
    gr = data["models"]["granite31_moe_1b"]

    ol_v = {v["variant"]: v for v in ol["variants"]}
    gr_v = {v["variant"]: v for v in gr["variants"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.95))

    # Panel (a): Stage-by-stage component ablation & baselines accuracy
    order = [
        ("undefended", "Undef."),
        ("ablation_stage1_only", "S1 Only"),
        ("pure_router_noise_b0.5", "Noise\n(b=0.5)"),
        ("ablation_stage1_stage2", "S1+S2"),
        ("uniform_padding_64", "Uniform\nPad (8x)"),
        ("ablation_stage2_stage3_no_s1", "S2+S3\n(0 PPL)"),
        ("full_oblivdispatch_eps1.0", "OblivD.\n(Full)"),
    ]
    x = np.arange(len(order))
    w = 0.36
    ol_acc = [ol_v[k]["global_acc_mean"] for k, _ in order]
    ol_std = [ol_v[k]["global_acc_std"] for k, _ in order]
    gr_acc = [gr_v[k]["global_acc_mean"] for k, _ in order]
    gr_std = [gr_v[k]["global_acc_std"] for k, _ in order]

    ax1.bar(x - w / 2, ol_acc, width=w, yerr=ol_std, capsize=2.2, color="#c00000", label="OLMoE-1B-7B")
    ax1.bar(x + w / 2, gr_acc, width=w, yerr=gr_std, capsize=2.2, color="#1f4e79", label="Granite-3.1-MoE")
    ax1.axhline(20.0, color="black", ls=":", lw=1.1, label="Random Floor (20%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in order])
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("5-Domain Global Acc (%)")
    ax1.set_title("(a) Baselines & Stage-by-Stage Ablation")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="upper right", fontsize=7.0)

    # Panel (b): Privacy vs FFN Compute & Perplexity Frontier (OLMoE-1B-7B)
    # Show log-scale Delta PPL vs Global Accuracy, annotated with FFN compute multiplier
    pts = [
        ("Undefended\n(1.0x FFN)", ol_v["undefended"]["global_acc_mean"], 0.01, "#666666", "o"),
        ("Pure Noise b=0.5\n(+13.9 PPL)", ol_v["pure_router_noise_b0.5"]["global_acc_mean"], ol["expanded_ppl"]["pure_router_noise"]["b_0.5"]["delta_ppl"], "#d95f02", "^"),
        ("Pure Noise b=1.0\n(+630 PPL)", ol_v["pure_router_noise_b1.0"]["global_acc_mean"], ol["expanded_ppl"]["pure_router_noise"]["b_1.0"]["delta_ppl"], "#a63603", "v"),
        ("Uniform Pad\n(8.0x FFN, 0 PPL)", ol_v["uniform_padding_64"]["global_acc_mean"], 0.012, "#5b9bd5", "s"),
        ("OblivDispatch\n(2.4x FFN, +0.15 PPL)", ol_v["full_oblivdispatch_eps1.0"]["global_acc_mean"], ol["expanded_ppl"]["oblivdispatch"]["eps_1.0"]["delta_ppl"], "#1b5e20", "*"),
    ]
    for lbl, acc, dppl, col, mk in pts:
        ms = 12 if mk == "*" else 7.5
        ax2.scatter([acc], [max(dppl, 0.01)], color=col, marker=mk, s=ms**2, zorder=5)

    ax2.annotate("Undefended\n(1.0x FFN)", (ol_v["undefended"]["global_acc_mean"], 0.012), xytext=(-46, 18), textcoords="offset points", fontsize=6.8, color="#444444", arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))
    ax2.annotate("Pure Noise b=0.5\n(+13.9 PPL)", (ol_v["pure_router_noise_b0.5"]["global_acc_mean"], 13.89), xytext=(-58, -22), textcoords="offset points", fontsize=6.8, color="#d95f02", arrowprops=dict(arrowstyle="->", color="#d95f02", lw=0.8))
    ax2.annotate("Pure Noise b=1.0\n(+630.3 PPL)", (ol_v["pure_router_noise_b1.0"]["global_acc_mean"], 630.3), xytext=(8, -14), textcoords="offset points", fontsize=6.8, color="#a63603", arrowprops=dict(arrowstyle="->", color="#a63603", lw=0.8))
    ax2.annotate("Uniform Pad\n(8.0x FFN)", (ol_v["uniform_padding_64"]["global_acc_mean"], 0.014), xytext=(12, 6), textcoords="offset points", fontsize=6.8, color="#1f4e79", arrowprops=dict(arrowstyle="->", color="#5b9bd5", lw=0.8))
    ax2.annotate("OblivDispatch (Ours)\n2.4x FFN, +0.15 PPL", (ol_v["full_oblivdispatch_eps1.0"]["global_acc_mean"], 0.154), xytext=(12, 16), textcoords="offset points", fontsize=7.0, fontweight="bold", color="#1b5e20", arrowprops=dict(arrowstyle="->", color="#1b5e20", lw=1.0))

    ax2.axvline(20.0, color="black", ls=":", lw=1.0)
    ax2.set_yscale("log")
    ax2.set_xlim(12, 92)
    ax2.set_ylim(0.005, 1800)
    ax2.set_xlabel("Attacker 5-Domain Accuracy (%) [Lower is Better]")
    ax2.set_ylabel("Perplexity Shift $\\Delta$PPL (log scale)")
    ax2.set_title("(b) Privacy vs. Quality Frontier (OLMoE-1B-7B)")
    ax2.grid(True, which="both", alpha=0.22)

    fig.tight_layout()
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_DIR / "fig6_baselines_ablations.pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "fig6_baselines_ablations.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("[Fig 6] Generated fig6_baselines_ablations.pdf/.png cleanly!")


if __name__ == "__main__":
    main()
