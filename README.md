# SonarMoE & OblivDispatch

Reference implementation, benchmark suite (`MoELeak-Bench`), localhost HTTP/1.1 Server-Sent Events (SSE) disaggregated serving harness, and manuscript sources for:

> **SonarMoE: Remote Prompt and Tool-Schema Reconstruction via Cross-Node Expert Contention in Disaggregated MoE Serving, and Differentially Private Mitigation**

## Overview

Modern Mixture-of-Experts (MoE) serving clusters disaggregate Attention frontends from shared Expert-Parallel (EP) feed-forward (FFN) pools (`MegaScale-Infer`, `DeepEP`). Tenants execute prefill and attention on isolated frontend containers with zero shared KV-cache memory, which blocks prefix-cache timing channels (`PROMPTPEEK`, `InputSnatch`), and zero host co-residency, which blocks local hardware probes (`MoEcho`).

`SonarMoE` shows that an external API client holding only a standard streaming HTTP key can still infer co-scheduled victim prompts and Model Context Protocol (MCP) tool invocations across disaggregated nodes:

1. **Offline Rank-Concentrated Canary Synthesis (`sonarmoe/canary.py`):** Using public router weights and zero victim data, the attacker solves a D-optimal selection objective over the model vocabulary to find single-token canary sequences that saturate target experts across layers while steering the cross-rank combine straggler bottleneck.
2. **Hardware Tile-Boundary Amplification (`sonarmoe/models.py`, `sonarmoe/theory.py`):** Sizing canary token counts to $N_{\text{canary}} = M_{\text{tile}} - \delta$ right below a GPU GEMM tile cliff (`N = 128 -> 129` on Apple Silicon MPS FP16) converts a few co-scheduled victim tokens into a discrete `+35 us/expert` hardware step jump observable over remote HTTP Inter-Token Latency (ITL).
3. **`OblivDispatch` Mitigation (`sonarmoe/oblivdispatch.py`):** Combines probability-margin-bounded secondary router Gumbel perturbation (`ΔPPL < 0.05`), Shadow-Expert overflow routing into non-straggler barrier slack (*Straggler-Slack Hiding*), constant-occupancy kernel-skip closure, and $(\epsilon, \delta)$-differentially private truncated Gaussian dummy padding (`< 5.5%` TPOT overhead).

## Repository Layout

- `sonarmoe/`: Core Python package (`models.py`, `canary.py`, `oblivdispatch.py`, `deconvolver.py`, `http_loopback_afd.py`, `theory.py`).
- `moeleak_bench/`: Two-tier fixed-window (`B_victim = 64` tokens, `0.00` bits length leakage) benchmark:
  - Tier 1 (`domain_5.py`): 100 enterprise prompts across 5 sensitive categories (`Clinical/Medical`, `SQL/Database Tool`, `Stripe/Payment Tool`, `Crypto/Auth Code`, `Legal Contract`).
  - Tier 2 (`mcp_tools_20.py`): 160 JSON-RPC tool calls across 20 distinct MCP tool schemas (`5.0%` random-guess floor).
- `pilot/`: Self-contained single-model feasibility pilot (`sonarmoe_pilot.py`).
- `scripts/`: Full multi-model, 20-MCP-schema, and live localhost HTTP/SSE evaluation runner (`run_phase1_suite.py`), vector figure generator (`generate_paper_figures.py`), and Hugging Face publisher (`publish_huggingface.py`).
- `tests/`: Pytest suite verifying benchmark balance, Theorem 1 and Theorem 2 bounds, token conservation, and live localhost TCP/HTTP-1.1 SSE round-trips.
- `paper/`: Two-column `IEEEtran` (`10pt,journal,compsoc`) manuscript (`main.tex`, `references.bib`), static validator (`validate_latex.py`), and generated vector PDF figures.

## Reproducing Locally on Apple Silicon (48 GB Unified Memory)

```bash
# 1. Run unit and live localhost HTTP/SSE integration tests
PYTHONPATH=. pytest tests/ -v

# 2. Run the full multi-model + 20-MCP-Schema + Live HTTP/SSE suite
python3 -u scripts/run_phase1_suite.py

# 3. Regenerate all 5 vector PDF and PNG figures from raw JSON telemetry
python3 scripts/generate_paper_figures.py

# 4. Validate the IEEE TDSC / TIFS LaTeX manuscript
python3 paper/validate_latex.py
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
