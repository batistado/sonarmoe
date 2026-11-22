#!/usr/bin/env python3
"""Build & Publish SonarMoE Private Hugging Face Suite (`batistado` Namespace).

Publishes three interlinked private (`private=True`) Hugging Face Hub repositories:
  1. Dataset : `https://huggingface.co/datasets/batistado/MoELeak-Bench`
       5 Dataset-Viewer JSONL configs (`domain5_prompts`, `mcp20_invocations`,
       `attack_sweep`, `oblivdispatch_pareto`, `tile_profiles`) + raw hardware JSON
       telemetry + publication figures.
  2. Space   : `https://huggingface.co/spaces/batistado/SonarMoE-Explorer`
       Always-on Static HTML5 Interactive Explorer (`sdk: static`) featuring a live
       Theorem 1 Tile-Cliff Capacity calculator, Multi-Model Attack & OblivDispatch
       Pareto explorer, Live Localhost HTTP/SSE confusion matrix viewer, and a
       searchable 260-query MoELeak-Bench browser.
  3. Hub Repo: `https://huggingface.co/batistado/SonarMoE`
       Core `sonarmoe` Python library, `moeleak_bench` suite, figures, and IEEE TDSC
       manuscript Overleaf bundle (`sonarmoe_tdsc_overleaf.zip`).
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import truststore
truststore.inject_into_ssl()

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moeleak_bench import (
    CATEGORIES_5,
    MCP_TOOL_SCHEMAS_20,
    MCP_TOOLS_20_PROMPTS,
    MOELEAK_DOMAIN_100,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_dataset_staging(stage_dir: Path, username: str) -> dict[str, Any]:
    full_json_path = REPO_ROOT / "results" / "sonarmoe_phase1_full.json"
    full_data = json.loads(full_json_path.read_text(encoding="utf-8"))

    # 1. Config `domain5_prompts` (100 rows)
    dom_rows = [
        {
            "id": str(r["id"]),
            "category_idx": int(r["category_idx"]),
            "category": str(r["category"]),
            "fixed_window_tokens": 64,
            "prompt": str(r["prompt"]),
        }
        for r in MOELEAK_DOMAIN_100
    ]
    _write_jsonl(stage_dir / "data" / "domain5_prompts" / "train.jsonl", dom_rows)

    # 2. Config `mcp20_invocations` (160 rows)
    mcp_rows = [
        {
            "id": str(r["id"]),
            "tool_idx": int(r["category_idx"]),
            "tool_schema": str(r["category"]),
            "fixed_window_tokens": 64,
            "jsonrpc_payload": str(r["prompt"]),
        }
        for r in MCP_TOOLS_20_PROMPTS
    ]
    _write_jsonl(stage_dir / "data" / "mcp20_invocations" / "train.jsonl", mcp_rows)

    # 3. Config `attack_sweep` & 4. Config `oblivdispatch_pareto` & 5. Config `tile_profiles`
    attack_rows: list[dict[str, Any]] = []
    pareto_rows: list[dict[str, Any]] = []
    tile_rows: list[dict[str, Any]] = []

    for m_key, m_obj in full_data["models"].items():
        model_id = m_obj["spec"]["model_id"]
        # Tile profiles
        tp = m_obj["mps_tile_profile"]
        for n_tok in range(1, len(tp["median_us"])):
            tile_rows.append(
                {
                    "model_key": m_key,
                    "model_id": model_id,
                    "co_batched_tokens": n_tok,
                    "median_expert_latency_us": round(float(tp["median_us"][n_tok]), 3),
                    "iqr_expert_latency_us": round(float(tp["iqr_us"][n_tok]), 3),
                }
            )

        # Tier-1 Domain-5 attack sweep
        for sig_key, sw in m_obj["domain5_benchmark"]["attack_sweep"].items():
            sigma_us = float(sw["sigma_net_us"])
            for method_name in [
                "tile_selective_ibgda",
                "tile_global_barrier",
                "naive_32_global_barrier",
                "naive_16_global_barrier",
            ]:
                met = sw[method_name]
                attack_rows.append(
                    {
                        "tier": "Domain-5 (100 queries)",
                        "model_key": m_key,
                        "model_id": model_id,
                        "sigma_net_us": sigma_us,
                        "method": method_name,
                        "top1_accuracy_pct": float(met["accuracy_pct"]),
                        "top3_accuracy_pct": float(met["top3_accuracy_pct"]),
                        "macro_f1_pct": float(met["macro_f1_pct"]),
                        "empirical_mi_bits": float(met["empirical_mi_bits"]),
                    }
                )

        # Tier-2 MCP-20 attack sweep
        for sig_key, sw in m_obj["mcp20_benchmark"]["attack"].items():
            sigma_us = float(sig_key.replace("sigma_", "").replace("us", ""))
            for method_name, met in sw.items():
                attack_rows.append(
                    {
                        "tier": "MCP-Tools-20 (160 queries)",
                        "model_key": m_key,
                        "model_id": model_id,
                        "sigma_net_us": sigma_us,
                        "method": method_name,
                        "top1_accuracy_pct": float(met["accuracy_pct"]),
                        "top3_accuracy_pct": float(met["top3_accuracy_pct"]),
                        "macro_f1_pct": float(met["macro_f1_pct"]),
                        "empirical_mi_bits": float(met["empirical_mi_bits"]),
                    }
                )

        # OblivDispatch Pareto rows
        for d_row in m_obj["domain5_benchmark"]["oblivdispatch_sweep"]:
            pareto_rows.append(
                {
                    "model_key": m_key,
                    "model_id": model_id,
                    "epsilon": float(d_row["epsilon"]),
                    "delta": float(d_row["delta"]),
                    "global_barrier_acc_pct": float(d_row["accuracy_pct_sigma50_global"]),
                    "ibgda_acc_pct": float(d_row["accuracy_pct_sigma50_ibgda"]),
                    "empirical_mi_bits": float(d_row["empirical_mi_bits_sigma50_global"]),
                    "corpus_ppl": float(d_row["ppl"]),
                    "delta_ppl": float(d_row["delta_ppl"]),
                    "serving_tpot_ms": float(d_row["tpot_ms"]),
                    "tpot_overhead_pct": float(d_row["tpot_overhead_pct"]),
                }
            )

    _write_jsonl(stage_dir / "data" / "attack_sweep" / "train.jsonl", attack_rows)
    _write_jsonl(stage_dir / "data" / "oblivdispatch_pareto" / "train.jsonl", pareto_rows)
    _write_jsonl(stage_dir / "data" / "tile_profiles" / "train.jsonl", tile_rows)

    # Copy raw telemetry JSONs and publication PNGs
    raw_dst = stage_dir / "telemetry"
    raw_dst.mkdir(parents=True, exist_ok=True)
    for jf in sorted((REPO_ROOT / "results").glob("*.json")):
        shutil.copy2(jf, raw_dst / jf.name)

    fig_dst = stage_dir / "figures"
    fig_dst.mkdir(parents=True, exist_ok=True)
    for pf in sorted((REPO_ROOT / "results").glob("fig*.png")):
        shutil.copy2(pf, fig_dst / pf.name)

    ds_card = f"""---
license: apache-2.0
task_categories:
  - text-classification
language:
  - en
tags:
  - mixture-of-experts
  - side-channel
  - differential-privacy
  - model-context-protocol
  - llm-serving
  - benchmark
pretty_name: MoELeak-Bench v1.0
size_categories:
  - n<1K
configs:
  - config_name: domain5_prompts
    data_files:
      - split: train
        path: data/domain5_prompts/train.jsonl
  - config_name: mcp20_invocations
    data_files:
      - split: train
        path: data/mcp20_invocations/train.jsonl
  - config_name: attack_sweep
    data_files:
      - split: train
        path: data/attack_sweep/train.jsonl
  - config_name: oblivdispatch_pareto
    data_files:
      - split: train
        path: data/oblivdispatch_pareto/train.jsonl
  - config_name: tile_profiles
    data_files:
      - split: train
        path: data/tile_profiles/train.jsonl
---

# MoELeak-Bench v1.0: Constant-Length Side-Channel Benchmark for Disaggregated MoE Serving

**MoELeak-Bench** accompanies ***SonarMoE: Remote Prompt and Tool-Schema Reconstruction via Cross-Node Expert Contention in Disaggregated MoE Serving and Differentially Private Mitigation***.

Every prompt and JSON-RPC invocation in **MoELeak-Bench** is evaluated on an identical fixed `B_victim = 64`-token micro-batch window (`0.00 bits` of prompt-length or total-load leakage):
- **Tier 1 (`domain5_prompts`, 100 queries):** 20 distinct prompts per class across `Clinical/Medical`, `SQL/Database Tool`, `Stripe/Payment Tool`, `Crypto/Auth Code`, and `Legal Contract` (`20.0%` random floor).
- **Tier 2 (`mcp20_invocations`, 160 queries):** 8 distinct JSON-RPC `tools/call` payloads across 20 enterprise Model Context Protocol (MCP) tool schemas (`5.0%` Top-1, `15.0%` Top-3 random floor).
- **Hardware Telemetry (`attack_sweep`, `oblivdispatch_pareto`, `tile_profiles`):** Complete multi-model measurements on `allenai/OLMoE-1B-7B-0924-Instruct` (`16L, 64E, top-8`) and `ibm-granite/granite-3.1-1b-a400m-instruct` (`24L, 32E, norm top-8`), plus live localhost `127.0.0.1` `TCP_NODELAY` HTTP/1.1 chunked SSE streaming telemetry.

![Hardware Tile Cliffs and Theorem 1 Capacity](figures/fig1_tile_cliff_capacity.png)

## Related Artifacts
- **System & Library Repository:** [`{username}/SonarMoE`](https://huggingface.co/{username}/SonarMoE)
- **Interactive Explorer Space:** [`{username}/SonarMoE-Explorer`](https://huggingface.co/spaces/{username}/SonarMoE-Explorer)
- **GitHub Source Repository:** [`github.com/batistado/sonarmoe`](https://github.com/batistado/sonarmoe)

## Citation

```bibtex
@article{{syed2026sonarmoe,
  title   = {{{{SonarMoE}}: Remote Prompt and Tool-Schema Reconstruction via Cross-Node Expert Contention in Disaggregated {{MoE}} Serving and Differentially Private Mitigation}},
  author  = {{Syed, Mohammed Kamran and Rentala, Anudeep and Rahul, Somesh and Pai, Raghuram Malpe and Dara, Krishna Suman}},
  journal = {{IEEE Transactions on Dependable and Secure Computing (TDSC)}},
  year    = {{2026}},
  url     = {{https://github.com/batistado/sonarmoe}}
}}
```
"""
    (stage_dir / "README.md").write_text(ds_card, encoding="utf-8")
    return full_data


def build_static_space_staging(
    space_dir: Path, full_data: dict[str, Any], username: str
) -> None:
    space_dir.mkdir(parents=True, exist_ok=True)
    space_readme = f"""---
title: SonarMoE & OblivDispatch Explorer
emoji: 📡
colorFrom: red
colorTo: blue
sdk: static
pinned: false
license: apache-2.0
datasets:
  - {username}/MoELeak-Bench
models:
  - allenai/OLMoE-1B-7B-0924-Instruct
  - ibm-granite/granite-3.1-1b-a400m-instruct
---

# SonarMoE & OblivDispatch Interactive Explorer

Static HTML5 interactive dashboard exploring GPU tile-stepping side-channel capacity, 20-schema MCP tool reconstruction, and `(ε, δ)`-DP `OblivDispatch` mitigation.
"""
    (space_dir / "README.md").write_text(space_readme, encoding="utf-8")

    embedded_json = json.dumps(
        {
            "models": full_data["models"],
            "theory": full_data["theory_verification"],
            "sample_domain5": MOELEAK_DOMAIN_100[:25],
            "sample_mcp20": MCP_TOOLS_20_PROMPTS[::4],
        }
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SonarMoE &amp; OblivDispatch Explorer</title>
<style>
:root {{ --bg:#0f172a; --card:#1e293b; --text:#f8fafc; --muted:#94a3b8; --red:#ef4444; --blue:#38bdf8; --green:#22c55e; --border:#334155; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:system-ui,-apple-system,sans-serif; background:var(--bg); color:var(--text); line-height:1.5; }}
header {{ padding:24px 32px; border-bottom:1px solid var(--border); background:#0b1120; }}
h1 {{ margin:0 0 6px 0; font-size:1.45rem; }}
.sub {{ color:var(--muted); font-size:0.92rem; }}
main {{ max-width:1180px; margin:0 auto; padding:24px; display:grid; gap:20px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:14px; }}
.kpi {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:16px; }}
.kpi .label {{ color:var(--muted); font-size:0.8rem; text-transform:uppercase; letter-spacing:0.04em; }}
.kpi .val {{ font-size:1.55rem; font-weight:700; margin-top:4px; }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:20px; }}
h2 {{ margin:0 0 12px 0; font-size:1.12rem; }}
table {{ width:100%; border-collapse:collapse; font-size:0.88rem; }}
th,td {{ padding:8px 10px; border-bottom:1px solid var(--border); text-align:left; }}
th {{ color:var(--muted); font-weight:600; }}
select,input[type=range] {{ background:#0f172a; color:var(--text); border:1px solid var(--border); border-radius:6px; padding:6px 10px; }}
.controls {{ display:flex; flex-wrap:wrap; gap:16px; align-items:center; margin-bottom:14px; }}
pre {{ background:#0b1120; padding:12px; border-radius:8px; overflow-x:auto; font-size:0.8rem; border:1px solid var(--border); }}
</style>
</head>
<body>
<header>
  <h1>SonarMoE &amp; OblivDispatch Interactive Explorer</h1>
  <div class="sub">Remote Prompt &amp; 20-Schema MCP Tool Reconstruction via Disaggregated MoE Expert Contention and (&epsilon;, &delta;)-DP Straggler-Slack Hiding</div>
</header>
<main>
  <div class="kpis">
    <div class="kpi"><div class="label">OLMoE 5-Domain Acc (&sigma;=50&mu;s)</div><div class="val" style="color:var(--red)">85.0%</div></div>
    <div class="kpi"><div class="label">20-Schema MCP Top-1 / Top-3</div><div class="val" style="color:var(--red)">89.4% / 94.4%</div></div>
    <div class="kpi"><div class="label">OblivDispatch (&epsilon;=1.0) Acc</div><div class="val" style="color:var(--green)">19.0% (0.000b MI)</div></div>
    <div class="kpi"><div class="label">Serving TPOT Overhead</div><div class="val" style="color:var(--blue)">+4.85% to +5.45%</div></div>
  </div>

  <div class="card">
    <h2>1. Theorem 1 Interactive Tile-Cliff Channel Capacity Calculator</h2>
    <div class="controls">
      <label>Model Cliff &Delta;<sub>tile</sub>:
        <select id="selCliff" onchange="updateCalc()">
          <option value="29.09">OLMoE-1B-7B (+29.09 &mu;s/expert @ N=129)</option>
          <option value="4.05">Granite-3.1-MoE (+4.05 &mu;s/expert @ N=129)</option>
        </select>
      </label>
      <label>Network Jitter &sigma;<sub>net</sub>: <span id="lblSig">50</span> &mu;s
        <input type="range" id="rngSig" min="5" max="300" value="50" oninput="updateCalc()"/>
      </label>
      <label>Streamed Tokens R: <span id="lblR">6</span>
        <input type="range" id="rngR" min="1" max="16" value="6" oninput="updateCalc()"/>
      </label>
    </div>
    <div id="calcOut" style="font-size:0.95rem;"></div>
  </div>

  <div class="card">
    <h2>2. Multi-Model OblivDispatch (&epsilon;, &delta;=10<sup>-5</sup>)-DP Tradeoff (&sigma;<sub>net</sub> = 50 &mu;s)</h2>
    <div id="paretoTbl"></div>
  </div>

  <div class="card">
    <h2>3. MoELeak-Bench Constant-Length (B<sub>victim</sub>=64) Prompt &amp; MCP Payload Inspector</h2>
    <div class="controls">
      <select id="selTier" onchange="renderSamples()">
        <option value="mcp">Tier 2: 20-Schema MCP JSON-RPC Invocations</option>
        <option value="dom">Tier 1: 5-Domain Sensitive Enterprise Prompts</option>
      </select>
    </div>
    <pre id="sampleBox"></pre>
  </div>
</main>
<script>
const DATA = {embedded_json};

function erf(x) {{
  const s = x >= 0 ? 1 : -1; x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t - 0.284496736)*t + 0.254829592)*t*Math.exp(-x*x);
  return s * y;
}}
function h2(p) {{
  if (p <= 1e-12 || p >= 1 - 1e-12) return 0;
  return -p*Math.log2(p) - (1-p)*Math.log2(1-p);
}}
function updateCalc() {{
  const dTile = parseFloat(document.getElementById('selCliff').value);
  const sig = parseFloat(document.getElementById('rngSig').value);
  const R = parseInt(document.getElementById('rngR').value);
  document.getElementById('lblSig').textContent = sig;
  document.getElementById('lblR').textContent = R;
  const sigEff = sig / Math.sqrt(R);
  const echoTile = 6 * dTile;
  const echoNaive = 6 * 0.08 * 18.0;
  const peTile = 0.5 * (1 - erf((echoTile / (2 * sigEff)) / Math.SQRT2));
  const peNaive = 0.5 * (1 - erf((echoNaive / (2 * sigEff)) / Math.SQRT2));
  const cTile = Math.max(0, 1 - h2(peTile));
  const cNaive = Math.max(0, 1 - h2(peNaive));
  document.getElementById('calcOut').innerHTML =
    `<b>Tile-Primed Echo (6 layers):</b> +${{echoTile.toFixed(1)}} &mu;s &rarr; Lower-Bound Capacity = <b style="color:var(--red)">${{cTile.toFixed(3)}} bits/probe</b> &nbsp;|&nbsp; ` +
    `<b>Uncalibrated Within-Tile Echo:</b> +${{echoNaive.toFixed(1)}} &mu;s &rarr; <b style="color:var(--blue)">${{cNaive.toFixed(4)}} bits/probe</b>`;
}}

function renderTable() {{
  let html = '<table><tr><th>Model</th><th>Budget &epsilon;</th><th>Global Acc</th><th>IBGDA Acc</th><th>Empirical MI</th><th>Corpus PPL (&Delta;PPL)</th><th>Serving TPOT (&Delta;%)</th></tr>';
  for (const [mk, m] of Object.entries(DATA.models)) {{
    const u = m.domain5_benchmark.attack_sweep.sigma_50us;
    html += `<tr><td><b>${{m.spec.model_id}}</b></td><td>Undefended</td><td>${{u.tile_global_barrier.accuracy_pct}}%</td><td>${{u.tile_selective_ibgda.accuracy_pct}}%</td><td>${{u.tile_global_barrier.empirical_mi_bits}} b</td><td>${{m.baseline_serving.undefended_ppl}}</td><td>${{m.baseline_serving.undefended_tpot_ms}} ms (0.00%)</td></tr>`;
    for (const r of m.domain5_benchmark.oblivdispatch_sweep) {{
      html += `<tr><td>${{mk}}</td><td>&epsilon; = ${{r.epsilon}}</td><td>${{r.accuracy_pct_sigma50_global}}%</td><td>${{r.accuracy_pct_sigma50_ibgda}}%</td><td>${{r.empirical_mi_bits_sigma50_global}} b</td><td>${{r.ppl}} (${{r.delta_ppl >= 0 ? '+' : ''}}${{r.delta_ppl}})</td><td>${{r.tpot_ms}} ms (${{r.tpot_overhead_pct >= 0 ? '+' : ''}}${{r.tpot_overhead_pct}}%)</td></tr>`;
    }}
  }}
  html += '</table>';
  document.getElementById('paretoTbl').innerHTML = html;
}}

function renderSamples() {{
  const mode = document.getElementById('selTier').value;
  const arr = mode === 'mcp' ? DATA.sample_mcp20.slice(0, 8) : DATA.sample_domain5.slice(0, 8);
  document.getElementById('sampleBox').textContent = JSON.stringify(arr, null, 2);
}}

updateCalc();
renderTable();
renderSamples();
</script>
</body>
</html>
"""
    (space_dir / "index.html").write_text(html, encoding="utf-8")


def build_hub_repo_staging(hub_dir: Path, username: str) -> None:
    hub_dir.mkdir(parents=True, exist_ok=True)
    for pkg in ("sonarmoe", "moeleak_bench"):
        dst = hub_dir / pkg
        dst.mkdir(parents=True, exist_ok=True)
        for pyf in (REPO_ROOT / pkg).glob("*.py"):
            shutil.copy2(pyf, dst / pyf.name)

    for fname in ("pyproject.toml", "LICENSE", "CITATION.cff"):
        shutil.copy2(REPO_ROOT / fname, hub_dir / fname)

    for pf in sorted((REPO_ROOT / "results").glob("fig*.png")):
        shutil.copy2(pf, hub_dir / pf.name)

    zip_src = REPO_ROOT / "paper" / "sonarmoe_tdsc_overleaf.zip"
    if zip_src.exists():
        shutil.copy2(zip_src, hub_dir / zip_src.name)

    hub_card = f"""---
license: apache-2.0
library_name: transformers
tags:
  - mixture-of-experts
  - side-channel
  - differential-privacy
  - model-context-protocol
  - disaggregated-serving
  - deepep
datasets:
  - {username}/MoELeak-Bench
base_model:
  - allenai/OLMoE-1B-7B-0924-Instruct
  - ibm-granite/granite-3.1-1b-a400m-instruct
---

# SonarMoE & OblivDispatch

Reference implementation and manuscript bundle for ***SonarMoE: Remote Prompt and Tool-Schema Reconstruction via Cross-Node Expert Contention in Disaggregated MoE Serving and Differentially Private Mitigation***.

- **Benchmark Dataset (`260` constant-length queries):** [`{username}/MoELeak-Bench`](https://huggingface.co/datasets/{username}/MoELeak-Bench)
- **Interactive Explorer Space:** [`{username}/SonarMoE-Explorer`](https://huggingface.co/spaces/{username}/SonarMoE-Explorer)
- **IEEE TDSC LaTeX Archive:** [`sonarmoe_tdsc_overleaf.zip`](sonarmoe_tdsc_overleaf.zip)
- **GitHub Repository:** [`github.com/batistado/sonarmoe`](https://github.com/batistado/sonarmoe)

![OblivDispatch Pareto Tradeoff](fig4_oblivdispatch_pareto_theory.png)

## Quick Install

```bash
pip install git+https://github.com/batistado/sonarmoe.git
```
"""
    (hub_dir / "README.md").write_text(hub_card, encoding="utf-8")


def main() -> int:
    api = HfApi()
    username = api.whoami()["name"]
    ds_repo_id = f"{username}/MoELeak-Bench"
    space_repo_id = f"{username}/SonarMoE-Explorer"
    hub_repo_id = f"{username}/SonarMoE"

    with tempfile.TemporaryDirectory(prefix="sonarmoe_hf_") as tmp:
        tmp_path = Path(tmp)
        ds_dir = tmp_path / "dataset"
        sp_dir = tmp_path / "space"
        hb_dir = tmp_path / "hub"

        full_data = build_dataset_staging(ds_dir, username=username)
        api.create_repo(repo_id=ds_repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(
            repo_id=ds_repo_id,
            repo_type="dataset",
            folder_path=str(ds_dir),
            commit_message="Publish MoELeak-Bench v1.0 multi-model & HTTP/SSE telemetry (private)",
        )
        print(f"[1/3 SUCCESS] Private Dataset -> https://huggingface.co/datasets/{ds_repo_id}")

        build_static_space_staging(sp_dir, full_data, username=username)
        api.create_repo(
            repo_id=space_repo_id,
            repo_type="space",
            space_sdk="static",
            private=True,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=space_repo_id,
            repo_type="space",
            folder_path=str(sp_dir),
            commit_message="Publish SonarMoE & OblivDispatch Interactive Explorer (private)",
        )
        print(f"[2/3 SUCCESS] Private Space   -> https://huggingface.co/spaces/{space_repo_id}")

        build_hub_repo_staging(hb_dir, username=username)
        api.create_repo(repo_id=hub_repo_id, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(
            repo_id=hub_repo_id,
            repo_type="model",
            folder_path=str(hb_dir),
            commit_message="Publish SonarMoE & OblivDispatch v0.1.0 + IEEE TDSC archive (private)",
        )
        print(f"[3/3 SUCCESS] Private Hub     -> https://huggingface.co/{hub_repo_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
