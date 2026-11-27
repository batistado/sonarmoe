#!/usr/bin/env python3
"""Generates Fig. 1: Publication-grade vector System & Pipeline Architecture Diagram
for SonarMoE (Disaggregated MoE Tile-Cliff Attack) and OblivDispatch (3-Stage DP Defense)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT_DIR = Path(__file__).resolve().parent.parent
PAPER_FIG_DIR = ROOT_DIR / "paper" / "figures"
RESULTS_DIR = ROOT_DIR / "results"


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    fc: str,
    ec: str,
    title_color: str = "#111111",
    fontsize: float = 8.4,
    sub_fontsize: float = 7.2,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.02",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.35,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2.0,
        y + h * 0.72,
        title,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        x + w / 2.0,
        y + h * 0.30,
        subtitle,
        ha="center",
        va="center",
        fontsize=sub_fontsize,
        color="#222222",
        linespacing=1.28,
    )


def _arrow(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str = "#333333",
    lw: float = 1.5,
) -> None:
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>,head_length=5.5,head_width=3.2",
        color=color,
        linewidth=lw,
    )
    ax.add_patch(arr)


def main() -> None:
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.6, 4.9), gridspec_kw={"height_ratios": [1.12, 0.88]}
    )

    # =========================================================================
    # Panel (a): Disaggregated AFD & SonarMoE Tile-Cliff Attack
    # =========================================================================
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)
    ax1.axis("off")
    ax1.text(
        0.01,
        0.96,
        "(a) Disaggregated Attention-FFN Serving and SonarMoE Remote Tile-Cliff Probing",
        fontsize=9.8,
        fontweight="bold",
        color="#1f4e79",
        va="top",
    )

    _box(
        ax1,
        0.015,
        0.52,
        0.255,
        0.34,
        "Victim A (Attn Server 1)",
        "Isolated KV Cache (0-bit share)\nFixed $B_{\\mathrm{vic}}=64$ token prefill",
        fc="#e8f1fa",
        ec="#1f4e79",
        title_color="#1f4e79",
    )
    _box(
        ax1,
        0.015,
        0.08,
        0.255,
        0.34,
        "Attacker B (Attn Server 2)",
        "External HTTP API Key Only\nCanary $N_{\\mathrm{can}} = M_{\\mathrm{tile}} - 6 = 122$",
        fc="#fce8e6",
        ec="#c00000",
        title_color="#990000",
    )

    # Shared Expert Pool container
    pool = FancyBboxPatch(
        (0.325, 0.06),
        0.355,
        0.82,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="#fff9e6",
        edgecolor="#b7791f",
        linewidth=1.4,
        linestyle="--",
    )
    ax1.add_patch(pool)
    ax1.text(
        0.5025,
        0.835,
        "Shared Expert-Parallel GPU Pool",
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color="#843c0c",
    )
    _box(
        ax1,
        0.345,
        0.47,
        0.315,
        0.30,
        "Target Rank $p^*$ (Primed Experts)",
        "$N = 122 + N_{\\mathrm{vic}} > 128$ (Tile Cliff)\nDiscrete $+27.7\\,\\mu$s/layer step",
        fc="#ffe0b2",
        ec="#c00000",
        title_color="#990000",
        fontsize=8.1,
        sub_fontsize=7.1,
    )
    _box(
        ax1,
        0.345,
        0.10,
        0.315,
        0.29,
        "Partner Ranks (Idle Barrier Slack)",
        "Combine Barrier $\\max_p T_{\\mathrm{rank}}$\n$180$–$320\\,\\mu$s idle slack / layer",
        fc="#f3f4f6",
        ec="#666666",
        title_color="#333333",
        fontsize=8.1,
        sub_fontsize=7.1,
    )

    _box(
        ax1,
        0.735,
        0.50,
        0.25,
        0.34,
        "HTTP/1.1 SSE Stream",
        "Client perf_counter_ns()\n$\\Delta$ITL Echo ($M=32$ probes)",
        fc="#fce8e6",
        ec="#c00000",
        title_color="#990000",
    )
    _box(
        ax1,
        0.735,
        0.08,
        0.25,
        0.34,
        "SonarMoE Deconvolver",
        "CMR Matched Filter + MLP\nDomain-5 / MCP-Tools-20",
        fc="#f3e5f5",
        ec="#6a1b9a",
        title_color="#4a148c",
    )

    _arrow(ax1, 0.278, 0.69, 0.338, 0.64, color="#1f4e79")
    _arrow(ax1, 0.278, 0.25, 0.338, 0.56, color="#c00000")
    _arrow(ax1, 0.668, 0.62, 0.728, 0.65, color="#c00000")
    _arrow(ax1, 0.86, 0.49, 0.86, 0.43, color="#6a1b9a")

    # =========================================================================
    # Panel (b): Per-Tenant 3-Stage OblivDispatch (Straggler-Slack Hiding)
    # =========================================================================
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax2.axis("off")
    ax2.text(
        0.01,
        0.95,
        "(b) Per-Tenant OblivDispatch Pipeline (Executed on Victim A Before Cross-Tenant Coalescing)",
        fontsize=9.8,
        fontweight="bold",
        color="#1b5e20",
        va="top",
    )

    _box(
        ax2,
        0.015,
        0.22,
        0.215,
        0.60,
        "Stage 1: Tail Jitter",
        "Keep top-$(k{-}1)$ core\nPerturb $k$-th slot within\nmargin $\\Delta p \\leq \\gamma_p$",
        fc="#e8f5e9",
        ec="#2e7d32",
        title_color="#1b5e20",
        fontsize=8.2,
        sub_fontsize=7.2,
    )
    _box(
        ax2,
        0.265,
        0.22,
        0.235,
        0.60,
        "Stage 2: Shadow Cap",
        "Cap counts $c_{\\ell,e} \\leq \\tau_{\\mathrm{cap}}$\nSpill overflow into Victim's\nown lightest slack slots",
        fc="#e8f5e9",
        ec="#2e7d32",
        title_color="#1b5e20",
        fontsize=8.2,
        sub_fontsize=7.2,
    )
    _box(
        ax2,
        0.535,
        0.22,
        0.245,
        0.60,
        "Stage 3: Shifted DP",
        "2-Sided $Z \\sim \\mathcal{N}(0, \\sigma_{\\mathrm{DP}}^2)$\n$+\\mu_{\\mathrm{shift}}$, Clip $[1, N_{\\max}]$\nSplit $\\delta = \\delta/2 + \\delta/2$",
        fc="#e3f2fd",
        ec="#1565c0",
        title_color="#0d47a1",
        fontsize=8.2,
        sub_fontsize=7.2,
    )
    _box(
        ax2,
        0.815,
        0.22,
        0.17,
        0.60,
        "Guarantee",
        "$(\\epsilon, \\delta)$-DP\n0 Underflow Slots\n0.000 PPL (S2-3)\n+2.3-8.9% TPOT",
        fc="#f1f8e9",
        ec="#33691e",
        title_color="#1b5e20",
        fontsize=8.2,
        sub_fontsize=7.1,
    )

    _arrow(ax2, 0.236, 0.52, 0.260, 0.52, color="#2e7d32")
    _arrow(ax2, 0.506, 0.52, 0.530, 0.52, color="#1565c0")
    _arrow(ax2, 0.786, 0.52, 0.810, 0.52, color="#33691e")

    ax2.text(
        0.50,
        0.07,
        "Zero cross-tenant inspection: OblivDispatch never reads Attacker B's tokens; dummy tokens carry gate weight 0 and stay inside Tile 1 ($N_{\\max} \\leq 49 \\ll 128$).",
        ha="center",
        va="center",
        fontsize=7.3,
        style="italic",
        color="#444444",
    )

    fig.tight_layout(pad=0.5)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_DIR / "fig0_system_architecture.pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "fig0_system_architecture.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("[Fig 1] Regenerated clean fig0_system_architecture.pdf/.png!")


if __name__ == "__main__":
    main()
