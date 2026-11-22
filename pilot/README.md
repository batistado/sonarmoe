# SonarMoE & OblivDispatch — Phase 0 Local Pilot (Apple M5 Pro)

End-to-end hardware-grounded pilot for:
> ***SonarMoE: Remote Prompt and Tool-Schema Reconstruction via Cross-Node Expert Contention in Disaggregated MoE Serving—and Differentially Private Mitigation***

## Contents
- `moeleak_bench.py`: **100-query `MoELeak-Bench` dataset** (20 distinct prompts per class across `Clinical/Medical`, `SQL/Database Tool`, `Stripe/Payment Tool`, `Crypto/Auth Code`, and `Legal Contract`, plus an independent general-domain Perplexity evaluation corpus).
- `sonarmoe_pilot.py`: End-to-end Phase 0 execution pipeline on **`allenai/OLMoE-1B-7B-0924-Instruct`** (`16` MoE layers, `64` routed experts/layer, `top-8` routing, `FP16` on Apple M5 Pro MPS):
  1. **Step A (`build_orthogonal_canary_codebook`):** Blind offline Canary synthesis purely from public vocabulary embeddings & router gates (zero victim prompt access), selecting $M=16$ orthogonal probes via D-optimal experimental design.
  2. **Step B (`profile_real_mps_expert_tile_stepping` + `verify_live_cobatched_rank_timing`):** Real Apple M5 Pro MPS `FP16` kernel latency profiling across $N \in [1, 256]$ tokens on `OLMoE` expert MLPs + live co-batched 16-layer Expert-Rank timing verification.
  3. **Step C (`simulate_afd_itl_observations`, `evaluate_deconvolver_5fold`, `apply_oblivdispatch_counts`, `evaluate_real_olmoe_perplexity`):** Evaluates blind 5-fold stratified attack Accuracy / Macro-F1 / Mutual Information across $\sigma_{\text{net}} \in \{10, 50, 200\}\,\mu\text{s}$ (with & without Tile-Boundary Canary Sizing $M_{\text{tile}} - \delta$) and evaluates `OblivDispatch` across $\epsilon \in \{8.0, 4.0, 2.0, 1.0, 0.5\}, \delta = 10^{-5}$.

## Reproducing Locally
```bash
python3 -u pilot/sonarmoe_pilot.py
```

Outputs are written to `results/sonarmoe_olmoe_pilot.json` and `fig1..fig3_*.png`.
