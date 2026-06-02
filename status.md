# status.md — Autoresearch journal: BERT-style EMR encoder

**Run owner:** autoresearcher agent · **Started:** 2026-06-01 (overnight) · **GPU:** NVIDIA RTX A4500 (20 GB), CUDA 12.4, torch 2.4.1+cu124
**Branch:** `autoresearcher-updates` · **Remote:** github.com/shaharoded/Transform-EMR-Encoder

## Goal for the night

Finalise the architecture so the headline beats the in-house STRATS / GRU-D
benchmarks (both ~0.90 weighted AUROC, ~0.65 weighted AUPRC) while keeping the
per-outcome time prediction and length-of-stay MAE reasonable. AUPRC weighted
by support is the headline; AUROC is a secondary sanity check; time MAE matters
on its own. KEEP/DISCARD follow the weighted-by-support, AUPRC-led rule in
program.md.

## Plan

1. **baseline** — Train + eval the *current, unmodified* architecture at
   `sample=10000` (Phase-1 budget). This is the reference every Phase-1
   direction is measured against. Precede it with a `sample=50` smoke to lock
   the Gate-A–D references.
2. Work the listed Phase-1 directions in order, one falsifiable hypothesis per
   loop, KEEP/DISCARD per gate:
   1. MLM mask granularity (`phase2_mlm_mask_mode` × `phase2_mlm_ratio`)
   2. MLM head loss (CE → MaskedFocalBCE) — only on rare-token-collapse evidence
   3. Phase-2 aux-loss caps (`t_pos` × `t_local`)
   4. Phase-3 time-loss weight (`phase3_time_lambda`)
   5. Phase-3 backbone LR factor
   6. Task-head capacity (`phase3_head_hidden`)
   7. Per-outcome pool head count
   8. Agent-proposed directions from diagnose.py / loss-trace evidence
3. Propose new directions when diagnostics point off-list (cite paper/signal).
4. Stop when ≥2 consecutive iterations produce no KEEP across the remaining
   open levers and diagnose.py shows no obvious lever left.

Tooling note: diagnostics run via `diagnose_run.py` (mirrors api.load_data's
val pipeline; `api.py` trains at import-time so it cannot be imported for this).

**Infra note (2026-06-02):** `git push` has no credentials in this environment
(anonymous read works; push prompts for a username). Per supervisor decision,
I commit every iteration locally — history and `git revert` rollbacks are fully
preserved — and the branch will be pushed in one go once a token is supplied.
Also fixed `utils.py` tee log path to `/tmp/training_<uid>.log` (a stale
root-owned `/tmp/training.log` from the prep run blocked writes with
PermissionError).

---

### baseline  (commit 90506c1 · 2026-06-02)

**Hypothesis.** None — reference run of the *current, unmodified* architecture
at `sample=10000`. Establishes the numbers every Phase-1 direction is measured
against, and locks the Gate-A–D references.

**Change.** No architecture change. Harness only: `sample=10000`, `utils.py`
tee-log path fix. Config = defaults (embed_dim 128, n_layer 4, n_head 2;
positional MLM @ 0.15; phase2 caps t_pos 0.40 / t_local 0.30; phase3_time_lambda
0.5; head_hidden 256; backbone_lr_factor 0.01).

**Smoke (sample=50, 1 epoch/phase).** Gate-A (no NaN/inf): PASS. Gate-D (summary
+ all headline keys): PASS. Gate-B/C not exercisable — aux unlock epochs (P1=3,
P2=4) exceed the 1-epoch smoke, so no aux activated/calibrated; their references
come from this 10k run. Ran end-to-end on CUDA, 72 MB peak, 40 s.

**Headline (10k, held-out test, 1500 patients / 1307 with LoS).**
```
patient_auprc_weighted:    0.686507   <- PRIMARY; already > STRATS/GRU-D ~0.65
patient_auroc_weighted:    0.846871   <- target ~0.90; this is the gap
patient_auprc_simple:      0.566331
patient_auroc_simple:      0.827540
patient_max_f1_weighted:   0.706229
patient_f1_at_0_5_weighted:0.683120
length_of_stay_mae_hours:  121.4905   (median 108.5, p90 230.6)
time_mae weighted (6 risk outcomes by n_pos): 41.71 h
total_seconds: 6123.8  (P1 ~7min early-stop; P2 full 101 ep ~54min; P3 61 ep early-stop)
num_params: 1,849,354   peak_vram: 1089 MB
```
Per-outcome AUROC / AUPRC / time-MAE (reference):
| outcome | AUROC | AUPRC | time-MAE h | n_pos | prev |
|---|---|---|---|---|---|
| DISGLYCEMIA_Hyperglycemia | 0.922 | 0.923 | 16.9 | 557 | 0.371 |
| HYPEROSMOLALITY | 0.836 | 0.700 | 22.5 | 617 | 0.411 |
| KIDNEY_COMPLICATION | 0.844 | 0.696 | 32.2 | 485 | 0.323 |
| DISGLYCEMIA_Hypoglycemia | 0.817 | 0.447 | 48.2 | 146 | 0.097 |
| CARDIO-VASCULAR | 0.855 | 0.375 | 12.5 | 86 | 0.057 |
| DEATH | 0.691 | 0.258 | 206.7 | 193 | 0.129 |
| RELEASE (LoS) | — | — | 121.5 | 1307 | — |
Weak heads: DEATH (AUPRC 0.26, only sub-0.80 AUROC) + rare CVD (5.7%) /
Hypoglycemia (9.7%). Hyperglycemia is saturated-good.

**Per-aux training trace.**
| phase | aux | unlock ep | λ_max | anchor raw | final raw | Δ% | learning? |
|---|---|---|---|---|---|---|---|
| P1 | dt | 3 | 0.0660 | 1.3395 | 0.7011 | −47.7% | yes |
| P2 | MLM (main) | — | — | 5.8521 | 3.5669 | −39.1% | yes |
| P2 | t_pos | 4 | 10.0 (clamp) | 0.0675 | ~0.00024 | −99.6% | yes (near-perfect) |
| P2 | t_local | 4 | 10.0 (clamp) | 0.0010 | ~0.00062 | −38% | yes (tiny task) |
| P3 | Risk (main) | — | — | 0.977 | 0.456 | −53% | yes |
| P3 | Time (smooth-L1) | — | λ=0.5 fixed | 89.31 | 7.86 | −91% | yes |

λ-clamp note: both P2 time auxes hit λ_max=10 because their raw MSE is already
tiny at calibration (0.0675 / 0.0010), so the 40%/30% caps are unreachable —
t_pos/t_local contribute ≪ their cap. Not a failure (residuals alive), but the
time auxes are under-weighted vs design.

**Diagnose.py (4 val batches, positional p=0.15).**
- MLM top1=0.094, top5=0.318, majority=0.028 → beats majority 3.4× (T3 PASS);
  modest sharpness, legality top20=0.70.
- t_pos residual std=0.0278 (~9 h), mean −0.033; t_local std=0.0210 (~0.5 h),
  mean −0.0005 — both ALIVE, not collapsed.
- Risk logits well-spread (std 2.8–5.0), no saturation/collapse.
- Pool entropy 4.31–5.49 across outcomes (DEATH/RELEASE most focused); healthy,
  non-degenerate → pool-head-count search (dir #7) not indicated yet.

**Verdict.** REFERENCE (no KEEP/DISCARD). AUPRC 0.687 already beats ~0.65; AUROC
0.847 is the gap to ~0.90. Architecture is healthy — the gap is representation
sharpness + Phase-3 risk/time balance, not a broken pathway.

**What I'd try next.**
1. **Direction #1 (next, per playbook): MLM mask granularity.** Hierarchical
   (HEART-style) masking narrows the MLM target space per raw-concept family;
   hypothesis: lifts MLM top-1/top-5 → sharper representation → AUROC up. Test
   hierarchical @ 0.15 first (isolate the mode), then ratio 0.25.
2. **Strong data-driven candidate — Direction #4 (Phase-3 time λ).** Time term
   dominates Risk ~8.6:1 in the P3 objective (raw Time 7.86 × 0.5 = 3.93 vs Risk
   0.46). Lowering `phase3_time_lambda` (0.25 / 0.1) should rebalance gradient
   toward risk — the most direct lever on the AUROC gap. Run after #1 unless #1
   closes the gap.
3. DEATH head is worst (AUPRC 0.26); watch whether either lever lifts it.

### i1-hier  (direction #1: MLM mask granularity — hierarchical @ 0.15)

**Hypothesis.** Switching `phase2_mlm_mask_mode` positional→hierarchical
(HEART-style; non-interval masks become per-raw-family `[MASK_RAW_<family>]`)
narrows the MLM target space, so the head sees *which raw concept* was hidden
and only chooses among that family's values. Expectation: MLM top-1/top-5 rise
(baseline 0.094/0.318), yielding a sharper backbone representation that lifts
`patient_auroc_weighted` (target +0.010 past 0.847) without regressing AUPRC
0.687. Ratio held at 0.15 to isolate the mode (ratio 0.25 is a separate iter).
Cite: HEART hierarchical-masking / family-aware MLM curriculum (program.md dir #1).

**Change.** `transform_emr/config/model_config.py`: `phase2_mlm_mask_mode`
`"positional"` → `"hierarchical"`. No tokenizer rebuild needed — the family
`[MASK_RAW_*]` specials are always emitted at tokenizer build (dataset.py), so
the cached tokenizer + Phase-1 embedder (config unchanged) are reused; only
Phase-2/Phase-3 retrain.
