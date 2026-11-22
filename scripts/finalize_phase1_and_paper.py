#!/usr/bin/env python3
"""Merges OLMoE-1B-7B and Granite-3.1-MoE Phase 1 JSONs, generates all 5 paper
figures, populates Granite telemetry placeholders in `paper/main.tex`, runs
`validate_latex.py`, and builds `paper/sonarmoe_tdsc_overleaf.zip`."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
PAPER_DIR = ROOT_DIR / "paper"


def main() -> int:
    olmoe_json = RESULTS_DIR / "sonarmoe_olmoe_phase1.json"
    granite_json = RESULTS_DIR / "sonarmoe_granite_phase1.json"
    full_json = RESULTS_DIR / "sonarmoe_phase1_full.json"

    d_o = json.loads(olmoe_json.read_text(encoding="utf-8"))
    d_g = json.loads(granite_json.read_text(encoding="utf-8"))

    merged = {
        "project": "SonarMoE",
        "suite": "Phase 1 Local Multi-Model (OLMoE-1B-7B + Granite-3.1-MoE), 20-MCP-Tool & Live HTTP/SSE Suite",
        "timestamp": d_g["timestamp"],
        "total_runtime_sec": round(
            d_o["total_runtime_sec"] + d_g["total_runtime_sec"], 2
        ),
        "hardware": d_o["hardware"],
        "models": {
            "olmoe_7b": d_o["models"]["olmoe_7b"],
            "granite31_moe_1b": d_g["models"]["granite31_moe_1b"],
        },
        "theory_verification": d_o["theory_verification"],
    }
    full_json.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"[1/4] Wrote merged multi-model telemetry -> {full_json}")

    # 2. Generate all 5 vector-PDF + PNG figures
    subprocess.check_call(
        [sys.executable, str(ROOT_DIR / "scripts" / "generate_paper_figures.py")]
    )

    # 3. Populate Granite-3.1-MoE empirical values in paper/main.tex
    g_mod = merged["models"]["granite31_moe_1b"]
    g_tp = g_mod["mps_tile_profile"]
    g_can = g_mod["canary_codebook"]
    g_base = g_mod["baseline_serving"]
    g_dom = g_mod["domain5_benchmark"]
    g_att = g_dom["attack_sweep"]
    g_def = {row["epsilon"]: row for row in g_dom["oblivdispatch_sweep"]}
    g_mcp = g_mod["mcp20_benchmark"]
    g_mcp_def = {row["epsilon"]: row for row in g_mcp["defense"]}

    def _gain_str(s_key: str) -> str:
        ibg = g_att[s_key]["tile_selective_ibgda"]["accuracy_pct"]
        n16 = g_att[s_key]["naive_16_global_barrier"]["accuracy_pct"]
        diff = ibg - n16
        return f"$\\mathbf{{{diff:+.1f}\\%}}$ (IBGDA)" if diff >= 5.0 else f"${diff:+.1f}\\%$"

    tpot_ohs = [row["tpot_overhead_pct"] for row in g_dom["oblivdispatch_sweep"]]
    replacements = {
        "__GRAN_ORACLE__": f"${g_dom['whitebox_rank_oracle']['accuracy_pct']:.1f}\\%$ (${g_dom['whitebox_rank_oracle']['empirical_mi_bits']:.3f}~\\mathrm{{b}}$)",
        "__GRAN_ORACLE_SHORT__": f"${g_dom['whitebox_rank_oracle']['accuracy_pct']:.1f}\\%$ accuracy (${g_dom['whitebox_rank_oracle']['empirical_mi_bits']:.3f}~\\mathrm{{bits}}$)",
        "__GRAN_S10_IBGDA__": f"${g_att['sigma_10us']['tile_selective_ibgda']['accuracy_pct']:.1f}\\%$ / ${g_att['sigma_10us']['tile_selective_ibgda']['macro_f1_pct']:.1f}\\%$",
        "__GRAN_S10_GLOB__": f"${g_att['sigma_10us']['tile_global_barrier']['accuracy_pct']:.1f}\\%$ / ${g_att['sigma_10us']['tile_global_barrier']['macro_f1_pct']:.1f}\\%$",
        "__GRAN_S10_MI__": f"${g_att['sigma_10us']['tile_global_barrier']['empirical_mi_bits']:.3f}~\\mathrm{{bits}}$",
        "__GRAN_S10_N32__": f"${g_att['sigma_10us']['naive_32_global_barrier']['accuracy_pct']:.1f}\\%$",
        "__GRAN_S10_N16__": f"${g_att['sigma_10us']['naive_16_global_barrier']['accuracy_pct']:.1f}\\%$",
        "__GRAN_S10_GAIN__": _gain_str("sigma_10us"),
        "__GRAN_S50_IBGDA__": f"$\\mathbf{{{g_att['sigma_50us']['tile_selective_ibgda']['accuracy_pct']:.1f}\\%}}$ / $\\mathbf{{{g_att['sigma_50us']['tile_selective_ibgda']['macro_f1_pct']:.1f}\\%}}$",
        "__GRAN_S50_GLOB__": f"$\\mathbf{{{g_att['sigma_50us']['tile_global_barrier']['accuracy_pct']:.1f}\\%}}$ / $\\mathbf{{{g_att['sigma_50us']['tile_global_barrier']['macro_f1_pct']:.1f}\\%}}$",
        "__GRAN_S50_MI__": f"$\\mathbf{{{g_att['sigma_50us']['tile_global_barrier']['empirical_mi_bits']:.3f}~\\mathrm{{bits}}}}$",
        "__GRAN_S50_N32__": f"${g_att['sigma_50us']['naive_32_global_barrier']['accuracy_pct']:.1f}\\%$",
        "__GRAN_S50_N16__": f"${g_att['sigma_50us']['naive_16_global_barrier']['accuracy_pct']:.1f}\\%$",
        "__GRAN_S50_GAIN__": _gain_str("sigma_50us"),
        "__GRAN_S200_IBGDA__": f"$\\mathbf{{{g_att['sigma_200us']['tile_selective_ibgda']['accuracy_pct']:.1f}\\%}}$ / $\\mathbf{{{g_att['sigma_200us']['tile_selective_ibgda']['macro_f1_pct']:.1f}\\%}}$",
        "__GRAN_S200_GLOB__": f"${g_att['sigma_200us']['tile_global_barrier']['accuracy_pct']:.1f}\\%$ / ${g_att['sigma_200us']['tile_global_barrier']['macro_f1_pct']:.1f}\\%$",
        "__GRAN_S200_MI__": f"${g_att['sigma_200us']['tile_global_barrier']['empirical_mi_bits']:.3f}~\\mathrm{{bits}}$",
        "__GRAN_S200_N32__": f"${g_att['sigma_200us']['naive_32_global_barrier']['accuracy_pct']:.1f}\\%$",
        "__GRAN_S200_N16__": f"${g_att['sigma_200us']['naive_16_global_barrier']['accuracy_pct']:.1f}\\%$",
        "__GRAN_S200_GAIN__": _gain_str("sigma_200us"),
        "__GRAN_CLIFF_STR__": f"${g_tp['n128_median_us']:.2f} \\to {g_tp['n129_median_us']:.2f}~\\mu\\mathrm{{s}}$, ${g_tp['cliff_128_to_129_us']:+.2f}~\\mu\\mathrm{{s}}$/expert",
        "__GRAN_CAN_SEC__": f"${g_can['synthesis_time_sec']:.1f}~\\mathrm{{s}}$",
        "__GRAN_COV_STR__": f"${g_can['distinct_layer_experts_covered']}/{g_can['total_layer_expert_slots']}$ (${g_can['coverage_pct']:.2f}\\%$)",
        "__GRAN_DOM_SUMMARY__": f"${g_att['sigma_50us']['tile_selective_ibgda']['accuracy_pct']:.1f}\\%$ (IBGDA) and ${g_att['sigma_50us']['tile_global_barrier']['accuracy_pct']:.1f}\\%$ (Global Barrier) at $\\sigma_{{\\mathrm{{net}}}} = 50~\\mu\\mathrm{{s}}$",
        "__GMCP_10_T1__": f"$\\mathbf{{{g_mcp['attack']['sigma_10us']['tile_selective_ibgda']['accuracy_pct']:.1f}}}$",
        "__GMCP_10_T3__": f"$\\mathbf{{{g_mcp['attack']['sigma_10us']['tile_selective_ibgda']['top3_accuracy_pct']:.1f}}}$",
        "__GMCP_10_F1__": f"${g_mcp['attack']['sigma_10us']['tile_selective_ibgda']['macro_f1_pct']:.1f}\\%$",
        "__GMCP_10_MI__": f"${g_mcp['attack']['sigma_10us']['tile_selective_ibgda']['empirical_mi_bits']:.3f}$",
        "__GMCP_50_T1__": f"$\\mathbf{{{g_mcp['attack']['sigma_50us']['tile_selective_ibgda']['accuracy_pct']:.1f}}}$",
        "__GMCP_50_T3__": f"$\\mathbf{{{g_mcp['attack']['sigma_50us']['tile_selective_ibgda']['top3_accuracy_pct']:.1f}}}$",
        "__GMCP_50_F1__": f"${g_mcp['attack']['sigma_50us']['tile_selective_ibgda']['macro_f1_pct']:.1f}\\%$",
        "__GMCP_50_MI__": f"${g_mcp['attack']['sigma_50us']['tile_selective_ibgda']['empirical_mi_bits']:.3f}$",
        "__GMCP_N16_T1__": f"${g_mcp['attack']['sigma_50us']['naive_16_global_barrier']['accuracy_pct']:.1f}$",
        "__GMCP_N16_T3__": f"${g_mcp['attack']['sigma_50us']['naive_16_global_barrier']['top3_accuracy_pct']:.1f}$",
        "__GMCP_N16_F1__": f"${g_mcp['attack']['sigma_50us']['naive_16_global_barrier']['macro_f1_pct']:.1f}\\%$",
        "__GMCP_N16_MI__": f"${g_mcp['attack']['sigma_50us']['naive_16_global_barrier']['empirical_mi_bits']:.3f}$",
        "__GMCP_D2_T1__": f"$\\mathbf{{{g_mcp_def[2.0]['top1_acc_pct_ibgda']:.1f}}}$",
        "__GMCP_D2_T3__": f"$\\mathbf{{{g_mcp_def[2.0]['top3_acc_pct_ibgda']:.1f}}}$",
        "__GMCP_D2_F1__": f"${g_mcp_def[2.0]['macro_f1_pct_ibgda']:.1f}\\%$",
        "__GMCP_D2_MI__": f"${g_mcp_def[2.0]['empirical_mi_bits_ibgda']:.3f}$",
        "__GMCP_D1_T1__": f"$\\mathbf{{{g_mcp_def[1.0]['top1_acc_pct_ibgda']:.1f}}}$",
        "__GMCP_D1_T3__": f"$\\mathbf{{{g_mcp_def[1.0]['top3_acc_pct_ibgda']:.1f}}}$",
        "__GMCP_D1_F1__": f"${g_mcp_def[1.0]['macro_f1_pct_ibgda']:.1f}\\%$",
        "__GMCP_D1_MI__": f"${g_mcp_def[1.0]['empirical_mi_bits_ibgda']:.3f}$",
        "__GD_U_GLOB__": f"${g_att['sigma_50us']['tile_global_barrier']['accuracy_pct']:.1f}\\%$",
        "__GD_U_IBGDA__": f"${g_att['sigma_50us']['tile_selective_ibgda']['accuracy_pct']:.1f}\\%$",
        "__GD_U_MI__": f"${g_att['sigma_50us']['tile_global_barrier']['empirical_mi_bits']:.3f}$",
        "__GD_U_PPL__": f"${g_base['undefended_ppl']:.4f}$",
        "__GD_U_TPOT__": f"${g_base['undefended_tpot_ms']:.2f}$",
        "__GRAN_TPOT_RANGE__": f"${min(tpot_ohs):+.2f}\\%$ to ${max(tpot_ohs):+.2f}\\%$",
    }

    for eps_val, tag in [(8.0, "8"), (4.0, "4"), (2.0, "2"), (1.0, "1"), (0.5, "05")]:
        row = g_def[eps_val]
        bold = eps_val <= 1.0
        g_s = (
            f"$\\mathbf{{{row['accuracy_pct_sigma50_global']:.1f}\\%}}$"
            if bold
            else f"${row['accuracy_pct_sigma50_global']:.1f}\\%$"
        )
        i_s = (
            f"$\\mathbf{{{row['accuracy_pct_sigma50_ibgda']:.1f}\\%}}$"
            if bold
            else f"${row['accuracy_pct_sigma50_ibgda']:.1f}\\%$"
        )
        m_s = (
            f"$\\mathbf{{{row['empirical_mi_bits_sigma50_global']:.3f}}}$"
            if bold
            else f"${row['empirical_mi_bits_sigma50_global']:.3f}$"
        )
        replacements[f"__GD_{tag}_GLOB__"] = g_s
        replacements[f"__GD_{tag}_IBGDA__"] = i_s
        replacements[f"__GD_{tag}_MI__"] = m_s
        replacements[f"__GD_{tag}_PPL__"] = f"${row['ppl']:.4f}$"
        replacements[f"__GD_{tag}_DPPL__"] = f"${row['delta_ppl']:+.4f}$"
        replacements[f"__GD_{tag}_TPOT__"] = f"${row['tpot_ms']:.2f}$"
        replacements[f"__GD_{tag}_OH__"] = f"${row['tpot_overhead_pct']:+.2f}\\%$"

    tex_path = PAPER_DIR / "main.tex"
    tex_content = tex_path.read_text(encoding="utf-8")
    for k, v in replacements.items():
        tex_content = tex_content.replace(k, v)
    tex_path.write_text(tex_content, encoding="utf-8")
    print(f"[2/4] Populated {len(replacements)} Granite-3.1-MoE telemetry cells in {tex_path}")

    # 4. Run static IEEE TDSC LaTeX validator
    rc = subprocess.call([sys.executable, str(PAPER_DIR / "validate_latex.py")])
    if rc != 0:
        return rc

    # 5. Build 1-click Overleaf submission zip
    zip_path = PAPER_DIR / "sonarmoe_tdsc_overleaf.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in ["main.tex", "references.bib", "IEEEtran.cls", "IEEEtran.bst"]:
            zf.write(PAPER_DIR / fname, arcname=fname)
        for fig_pdf in sorted((PAPER_DIR / "figures").glob("*.pdf")):
            zf.write(fig_pdf, arcname=f"figures/{fig_pdf.name}")

    print(
        f"[4/4] Packaged 1-click Overleaf bundle -> {zip_path} "
        f"({zip_path.stat().st_size / 1024:.1f} KB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
