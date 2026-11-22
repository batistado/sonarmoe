#!/usr/bin/env python3
"""Generates all 5 vector-PDF and PNG figures for the IEEE TDSC/TIFS manuscript
strictly from `results/sonarmoe_phase1_full.json` hardware telemetry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_JSON = ROOT_DIR / "results" / "sonarmoe_phase1_full.json"
PAPER_FIG_DIR = ROOT_DIR / "paper" / "figures"
RESULTS_DIR = ROOT_DIR / "results"


def _setup_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.0,
            "axes.titlesize": 9.8,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 7.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 200,
        }
    )


def _save_both(fig: plt.Figure, stem: str) -> None:
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def make_fig1_tile_cliff_and_capacity(data: dict[str, Any]) -> None:
    models = data["models"]
    m_keys = list(models.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.15, 2.75))

    colors = ["#1f4e79", "#843c0c"]
    labels_m = {
        "olmoe_7b": "OLMoE-1B-7B (64E, 16L)",
        "granite31_moe_1b": "Granite-3.1-MoE (32E, 24L)",
        "qwen15_moe_a27b": "Qwen1.5-MoE-A2.7B (60+Shared)",
    }

    for idx, mk in enumerate(m_keys):
        tp = models[mk]["mps_tile_profile"]
        med = np.asarray(tp["median_us"], dtype=np.float64)
        iqr = np.asarray(tp["iqr_us"], dtype=np.float64)
        n_max = min(192, len(med) - 1)
        xs = np.arange(1, n_max + 1)
        lbl = labels_m.get(mk, mk) + f" (+{tp['cliff_128_to_129_us']:.1f} µs @129)"
        ax1.plot(xs, med[1 : n_max + 1], lw=1.7, color=colors[idx % 2], label=lbl)
        ax1.fill_between(
            xs,
            med[1 : n_max + 1] - 0.5 * iqr[1 : n_max + 1],
            med[1 : n_max + 1] + 0.5 * iqr[1 : n_max + 1],
            color=colors[idx % 2],
            alpha=0.15,
        )

    ax1.axvline(128, color="#c00000", ls="--", lw=1.3, label="Hardware Tile Cliff (N=128)")
    ax1.axvspan(116, 128, color="#ffd966", alpha=0.35, label="Canary Priming (N=122)")
    ax1.set_xlabel("Co-Batched Expert Tokens ($N_{\\mathrm{vic}} + N_{\\mathrm{can}}$)")
    ax1.set_ylabel("Per-Expert MLP Latency (µs)")
    ax1.set_ylim(25, 178)
    ax1.set_title("(a) Measured M5 Pro MPS Tile-Stepping Cliffs")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", fontsize=6.8)

    thm1 = data["theory_verification"]["theorem1_tile_capacity"]
    s_vals = [row["sigma_net_us"] for row in thm1]
    c_tile = [row["capacity_tile_bits_per_probe"] for row in thm1]
    c_naive = [row["capacity_naive_bits_per_probe"] for row in thm1]

    ax2.semilogx(
        s_vals,
        c_tile,
        marker="o",
        lw=1.9,
        color="#c00000",
        label="Tile-Primed Canary ($N_{\\mathrm{can}}=M_{\\mathrm{tile}}-6$)",
    )
    ax2.semilogx(
        s_vals,
        c_naive,
        marker="s",
        lw=1.7,
        ls="--",
        color="#5b9bd5",
        label="Uncalibrated Within-Tile Canary",
    )
    ax2.set_xlabel("Network Jitter $\\sigma_{\\mathrm{net}}$ (µs, log scale)")
    ax2.set_ylabel("Side-Channel Capacity (bits / probe)")
    ax2.set_ylim(-0.03, 1.08)
    ax2.set_title("(b) Theorem 1 Capacity: Tile Cliff vs. Slope")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="lower left", fontsize=7.0)

    fig.tight_layout()
    _save_both(fig, "fig1_tile_cliff_capacity")


def make_fig2_multimodel_jitter_asr(data: dict[str, Any]) -> None:
    models = data["models"]
    m_keys = list(models.keys())
    n_panels = len(m_keys)
    fig, axes = plt.subplots(1, n_panels, figsize=(7.15, 2.75), sharey=True)
    if n_panels == 1:
        axes = [axes]

    titles = {
        "olmoe_7b": "(a) OLMoE-1B-7B (Unnorm. Top-8, 64 Exp)",
        "granite31_moe_1b": "(b) Granite-3.1-MoE (Norm. Top-8, 32 Exp)",
        "qwen15_moe_a27b": "(b) Qwen1.5-MoE-A2.7B (Shared + Top-4)",
    }
    sigmas = [10, 50, 200]
    x = np.arange(len(sigmas))
    w = 0.20

    for ax, mk in zip(axes, m_keys):
        sw = models[mk]["domain5_benchmark"]["attack_sweep"]
        v_sel = [sw[f"sigma_{s}us"]["tile_selective_ibgda"]["accuracy_pct"] for s in sigmas]
        v_glob = [sw[f"sigma_{s}us"]["tile_global_barrier"]["accuracy_pct"] for s in sigmas]
        v_n32 = [sw[f"sigma_{s}us"]["naive_32_global_barrier"]["accuracy_pct"] for s in sigmas]
        v_n16 = [sw[f"sigma_{s}us"]["naive_16_global_barrier"]["accuracy_pct"] for s in sigmas]

        ax.bar(x - 1.5 * w, v_sel, width=w, color="#843c0c", label="SonarMoE (IBGDA, N=122)")
        ax.bar(x - 0.5 * w, v_glob, width=w, color="#c00000", label="SonarMoE (Global, N=122)")
        ax.bar(x + 0.5 * w, v_n32, width=w, color="#5b9bd5", label="Naive Canary (N=32)")
        ax.bar(x + 1.5 * w, v_n16, width=w, color="#a5a5a5", label="Naive Canary (N=16)")
        ax.axhline(20.0, color="black", ls=":", lw=1.1, label="Random Floor (20%)")

        ax.set_xticks(x)
        ax.set_xticklabels([f"$\\sigma={s}$ µs" for s in sigmas])
        ax.set_ylim(0, 105)
        ax.set_title(titles.get(mk, mk))
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("5-Domain Blind Accuracy (%)")
    axes[-1].legend(loc="upper right", fontsize=6.9)
    fig.tight_layout()
    _save_both(fig, "fig2_multimodel_jitter_asr")


def make_fig3_mcp20_reconstruction(data: dict[str, Any]) -> None:
    models = data["models"]
    m_keys = list(models.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.15, 2.85))

    sigmas = [10, 50, 200]
    x = np.arange(len(sigmas))
    w = 0.18
    palette = ["#c00000", "#f4b183", "#1f4e79", "#9dc3e6"]

    offset = -1.5 * w
    for idx, mk in enumerate(m_keys):
        mcp_att = models[mk]["mcp20_benchmark"]["attack"]
        short_lbl = "OLMoE" if "olmoe" in mk else ("GraniteMoE" if "granite" in mk else "QwenMoE")
        t1 = [mcp_att[f"sigma_{s}us"]["tile_selective_ibgda"]["accuracy_pct"] for s in sigmas]
        t3 = [mcp_att[f"sigma_{s}us"]["tile_selective_ibgda"]["top3_accuracy_pct"] for s in sigmas]
        ax1.bar(x + offset, t1, width=w, color=palette[2 * idx], label=f"{short_lbl} Top-1")
        offset += w
        ax1.bar(x + offset, t3, width=w, color=palette[2 * idx + 1], label=f"{short_lbl} Top-3")
        offset += w

    ax1.axhline(5.0, color="#444444", ls=":", lw=1.1, label="Random Top-1 (5%)")
    ax1.axhline(15.0, color="#777777", ls="--", lw=1.0, label="Random Top-3 (15%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"$\\sigma={s}$ µs" for s in sigmas])
    ax1.set_ylim(0, 128)
    ax1.set_ylabel("20-Schema MCP Accuracy (%)")
    ax1.set_title("(a) 20-Way MCP Tool-Call Attack vs. Jitter")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="upper right", fontsize=6.6, ncol=2)

    # Panel (b): Defense collapse on 20-way MCP benchmark at sigma = 50 us
    eps_labels = ["Undef.", "$\\epsilon=8$", "$\\epsilon=2$", "$\\epsilon=1$", "$\\epsilon=0.5$"]
    xp = np.arange(len(eps_labels))
    for idx, mk in enumerate(m_keys):
        short_lbl = "OLMoE-1B-7B" if "olmoe" in mk else ("Granite-3.1-MoE" if "granite" in mk else "Qwen1.5-MoE")
        mcp_b = models[mk]["mcp20_benchmark"]
        u_t1 = mcp_b["attack"]["sigma_50us"]["tile_selective_ibgda"]["accuracy_pct"]
        u_t3 = mcp_b["attack"]["sigma_50us"]["tile_selective_ibgda"]["top3_accuracy_pct"]
        d_t1 = [u_t1] + [d["top1_acc_pct_ibgda"] for d in mcp_b["defense"]]
        d_t3 = [u_t3] + [d["top3_acc_pct_ibgda"] for d in mcp_b["defense"]]
        col = "#c00000" if idx == 0 else "#1f4e79"
        ax2.plot(xp, d_t1, marker="o", lw=1.8, color=col, label=f"{short_lbl} Top-1")
        ax2.plot(xp, d_t3, marker="^", lw=1.4, ls="--", color=col, alpha=0.75, label=f"{short_lbl} Top-3")

    ax2.axhline(5.0, color="black", ls=":", lw=1.0)
    ax2.set_xticks(xp)
    ax2.set_xticklabels(eps_labels)
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("20-Schema MCP Accuracy (%)")
    ax2.set_title("(b) OblivDispatch on 20 MCP Tool Schemas")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", fontsize=6.8)

    fig.tight_layout()
    _save_both(fig, "fig3_mcp20_reconstruction")


def make_fig4_oblivdispatch_pareto_and_theory(data: dict[str, Any]) -> None:
    models = data["models"]
    m_keys = list(models.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.15, 2.80))

    eps_x = ["Undef.", "8.0", "4.0", "2.0", "1.0", "0.5"]
    xp = np.arange(len(eps_x))

    for idx, mk in enumerate(m_keys):
        short_lbl = "OLMoE" if "olmoe" in mk else ("GraniteMoE" if "granite" in mk else "QwenMoE")
        dom_b = models[mk]["domain5_benchmark"]
        u_acc = dom_b["attack_sweep"]["sigma_50us"]["tile_global_barrier"]["accuracy_pct"]
        accs = [u_acc] + [d["accuracy_pct_sigma50_global"] for d in dom_b["oblivdispatch_sweep"]]
        tpots = [0.0] + [d["tpot_overhead_pct"] for d in dom_b["oblivdispatch_sweep"]]
        col = "#c00000" if idx == 0 else "#1f4e79"
        ax1.plot(xp, accs, marker="o", lw=1.9, color=col, label=f"{short_lbl} Acc (%)")

    ax1.axhline(20.0, color="black", ls=":", lw=1.1, label="Random Floor (20%)")
    ax1.set_xticks(xp)
    ax1.set_xticklabels(eps_x)
    ax1.set_xlabel("Privacy Budget $\\epsilon$ ($\\delta = 10^{-5}$)")
    ax1.set_ylabel("5-Domain Global-Barrier Acc (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.25)

    ax1r = ax1.twinx()
    for idx, mk in enumerate(m_keys):
        short_lbl = "OLMoE" if "olmoe" in mk else ("GraniteMoE" if "granite" in mk else "QwenMoE")
        dom_b = models[mk]["domain5_benchmark"]
        tpots = [0.0] + [d["tpot_overhead_pct"] for d in dom_b["oblivdispatch_sweep"]]
        col = "#d95f02" if idx == 0 else "#2e75b6"
        ax1r.plot(xp, tpots, marker="s", ls="--", lw=1.5, color=col, label=f"{short_lbl} TPOT Δ%")
    ax1r.axhline(5.5, color="#2e75b6", ls="-.", lw=1.0, alpha=0.6)
    ax1r.set_ylabel("Serving TPOT Overhead (%)")
    ax1r.set_ylim(-0.5, 10.0)

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1r.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=6.4)
    ax1.set_title("(a) Accuracy Collapse vs. TPOT (<5.5%)")

    # Panel (b): Empirical Mutual Information vs Theorem 2 zCDP Upper Bound
    thm2 = data["theory_verification"]["theorem2_dp_bounds"]
    eps_num = [d["epsilon"] for d in thm2]
    xp2 = np.arange(len(eps_num))
    mi_bound = [d["theoretical_mi_upper_bits"] for d in thm2]
    ax2.plot(
        xp2,
        mi_bound,
        marker="D",
        lw=1.9,
        ls="--",
        color="#333333",
        label="Theorem 2 zCDP Bound",
    )

    for idx, mk in enumerate(m_keys):
        short_lbl = "OLMoE" if "olmoe" in mk else ("GraniteMoE" if "granite" in mk else "QwenMoE")
        dom_b = models[mk]["domain5_benchmark"]
        emp_mi = [d["empirical_mi_bits_sigma50_global"] for d in dom_b["oblivdispatch_sweep"]]
        col = "#c00000" if idx == 0 else "#1f4e79"
        ax2.plot(xp2, emp_mi, marker="o", lw=1.9, color=col, label=f"{short_lbl} Empirical MI")

    ax2.set_xticks(xp2)
    ax2.set_xticklabels([str(e) for e in eps_num])
    ax2.set_xlabel("Privacy Budget $\\epsilon$ ($\\delta = 10^{-5}$)")
    ax2.set_ylabel("Mutual Information $I(Y; \\hat{Y})$ (bits)")
    ax2.set_ylim(-0.05, 2.45)
    ax2.set_title("(b) Empirical Leakage vs. Theorem 2 Bound")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", fontsize=7.0)

    fig.tight_layout()
    _save_both(fig, "fig4_oblivdispatch_pareto_theory")


def make_fig5_live_http_loopback(data: dict[str, Any]) -> None:
    primary_key = next(iter(data["models"]))
    http_res = data["models"][primary_key]["live_http_loopback"]
    if not http_res:
        return

    cm_u = np.asarray(http_res["undefended_http_sse"]["confusion_matrix"], dtype=np.float64)
    cm_d = np.asarray(
        http_res["oblivdispatch_eps1_http_sse"]["confusion_matrix"], dtype=np.float64
    )
    short_cats = ["Medical", "SQL", "Stripe", "Crypto", "Legal"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.15, 2.85))

    for ax, cm, title_str in [
        (
            ax1,
            cm_u,
            f"(a) Undefended HTTP/SSE (Acc={http_res['undefended_http_sse']['accuracy_pct']:.0f}%)",
        ),
        (
            ax2,
            cm_d,
            f"(b) OblivDispatch $\\epsilon=1.0$ (Acc={http_res['oblivdispatch_eps1_http_sse']['accuracy_pct']:.0f}%)",
        ),
    ]:
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=20)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(short_cats, rotation=28, ha="right")
        ax.set_yticklabels(short_cats)
        ax.set_xlabel("Predicted Victim Class")
        ax.set_title(title_str)
        for i in range(5):
            for j in range(5):
                val = int(cm[i, j])
                txt_col = "white" if val >= 11 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=txt_col, fontsize=8.0)

    ax1.set_ylabel("True Victim Class (20 / class)")
    fig.colorbar(im, ax=[ax1, ax2], fraction=0.032, pad=0.03, label="Query Count")
    _save_both(fig, "fig5_live_http_loopback_cm")


def main() -> None:
    _setup_ieee_style()
    data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    make_fig1_tile_cliff_and_capacity(data)
    make_fig2_multimodel_jitter_asr(data)
    make_fig3_mcp20_reconstruction(data)
    make_fig4_oblivdispatch_pareto_and_theory(data)
    make_fig5_live_http_loopback(data)
    print(f"[Figures] Generated 5 PDF + PNG figures in {PAPER_FIG_DIR}")


if __name__ == "__main__":
    main()
