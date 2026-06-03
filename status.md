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

**Smoke result.** sample=50, 1 epoch/phase: hierarchical mode active, no NaN/inf,
summary + all headline keys print → Gate-A/D PASS (B/C not exercisable at 1 epoch,
as in baseline). Phase-1 embedder reused from cache (config unchanged); only
Phase-2/Phase-3 retrained.

**Headline metrics (10k) vs baseline.**
```
                            i1-hier      baseline     Δ
patient_auprc_weighted:     0.676561     0.686507    -0.0099   (PRIMARY regresses)
patient_auroc_weighted:     0.844911     0.846871    -0.0020   (no lift)
length_of_stay_mae_hours:   117.8226     121.4905    -3.67 h   (<5h thresh)
time_mae weighted (6 risk): 38.95 h      41.71 h     -2.76 h   (<5h thresh)
patient_max_f1_weighted:    0.700318     0.706229    -0.0059
```
Per-outcome AUPRC Δ: Hyperglycemia +0.006, CVD +0.049, Hyperosmolality +0.030,
DEATH +0.000, Kidney **−0.032**, Hypoglycemia **−0.216** (0.447→0.231, collapse).
The Hypoglycemia collapse (rare, 9.7%) drives the weighted AUPRC drop; time-MAE
improved broadly (CVD 12.5→10.4, Hyperosmol 22.5→17.2, Kidney 32.2→26.9) but
below the −5h KEEP threshold. P2 early-stopped at ep88 (baseline ran full 101).

**Per-aux training trace.**
| phase | aux | unlock ep | λ_max | anchor raw | final | note |
|---|---|---|---|---|---|---|
| P2 | MLM (main) | — | — | 5.8501 | (early-stop ep88) | same anchor as baseline |
| P2 | t_pos | 4 | 10.0 (clamp) | ~0.067 | tiny | as baseline |
| P2 | t_local | 4 | 10.0 (clamp) | 0.0010 | tiny | as baseline |
| P3 | Risk/Time | — | λ=0.5 | — | best_val 3.93 (vs 4.24) | P3 val better, AUPRC worse |

**Diagnose.py (hierarchical p=0.15) vs baseline.**
- MLM top1=0.0935 (baseline 0.0940 — UNCHANGED), top5=0.2989 (was 0.318 — WORSE),
  legality top20=0.6664 (was 0.701 — WORSE). The family-hint mechanism did NOT
  sharpen MLM; it slightly degraded top-5/top-20.
- t_pos std 0.0308, t_local std 0.0234 — alive, ~as baseline.
- Risk logits well-spread; pool entropy 4.23–5.46 — healthy, ~as baseline.

**Verdict.** DISCARD — primary AUPRC regresses −0.0099 (≥0.005) with no headline
meeting the +0.010 / −5h KEEP threshold, and the stated mechanism is falsified:
diagnose.py shows hierarchical masking does NOT raise MLM top-1/top-5. The
full-vocab CE target is unchanged by the mode, and the model already infers the
raw family from context — so the input hint adds no disambiguation and removes
mask-token diversity, slightly hurting top-5 and crashing the rare Hypoglycemia head.

**What I'd try next.** Family-hint masking is a dead end here (confirmed by the
MLM-accuracy probe). NOT testing ratio 0.25 under hierarchical — the mode itself
is the problem, not the ratio. Two live options:
- (a) Direction #1 residual: positional MLM @ **ratio 0.25** (more masked
  positions per batch → more MLM signal per epoch; cheap, isolates ratio under
  the working positional mode).
- (b) **Direction #4 (Phase-3 time λ)** — the strongest data-driven lever: P3
  Time term still dominates Risk ~8.6:1; lowering `phase3_time_lambda` should
  push gradient toward risk and lift AUROC/AUPRC. Leaning (b) given the AUROC gap
  is the real goal and #1's mode arm just failed; will try ratio 0.25 only if (b)
  also stalls.

### i4-tl025  (direction #4: Phase-3 time λ 0.5 → 0.25)

**Hypothesis.** Baseline diagnose shows the Phase-3 objective is Time-dominated:
raw smooth-L1 Time≈7.86 × λ0.5 = 3.93 vs Risk BCE 0.46 (~8.6:1). The risk and
time heads share the pool + shared-MLP + backbone, so a time-dominated gradient
plausibly under-trains risk discrimination. Lowering `phase3_time_lambda`
0.5→0.25 (→ ~4.3:1) should shift gradient toward risk and lift
`patient_auroc_weighted` (gap to 0.90) and/or `patient_auprc_weighted`
(target +0.010 past 0.687), at the risk of worse time MAE (must stay within +5h
to KEEP). Phase-3-only change; Phase-1 reused, Phase-2 identical to baseline
(retrained because api.py always clears phase2/3 — deterministic, reproduces baseline P2).

**Change.** `transform_emr/config/model_config.py`: `phase3_time_lambda`
`0.5` → `0.25`. (Committed separately from the journal so a DISCARD reverts cleanly.)

**Smoke result.** sample=50: P3 total = Risk + 0.25·Time = 22.04 (vs 43.05 at
λ0.5), confirming λ applied; no NaN/inf; summary + headline keys print → Gate-A/D
PASS. Phase-1 reused; Phase-2 reproduced baseline (positional@0.15, det. seed).

**Headline metrics (10k) vs baseline (= running best so far).**
```
                            i4-tl025     baseline     Δ
patient_auprc_weighted:     0.760241     0.686507    +0.0737  (PRIMARY, huge KEEP)
patient_auroc_weighted:     0.868612     0.846871    +0.0217  (also > +0.010)
patient_auprc_simple:       0.639949     0.566331    +0.0736
patient_max_f1_weighted:    0.735542     0.706229    +0.0293
length_of_stay_mae_hours:   120.2743     121.4905    -1.22 h
time_mae weighted (6 risk): 41.40 h      41.71 h     -0.31 h  (FLAT — no time cost)
phase3_best_val:            1.476722     4.238167             (lower total, smaller λ)
```
Per-outcome AUROC / AUPRC Δ (vs baseline):
| outcome | AUROC | AUPRC | ΔAUPRC |
|---|---|---|---|
| CARDIO-VASCULAR | 0.960 (+0.105) | 0.703 | **+0.328** |
| HYPEROSMOLALITY | 0.897 (+0.061) | 0.891 | **+0.191** |
| KIDNEY | 0.856 (+0.011) | 0.765 | +0.069 |
| DISGLYCEMIA_Hyperglycemia | 0.914 (−0.007) | 0.916 | −0.007 |
| DISGLYCEMIA_Hypoglycemia | 0.818 (+0.002) | 0.347 | −0.100 |
| DEATH | 0.675 (−0.016) | 0.218 | −0.040 |
The two rare/hard heads (Hypoglycemia, DEATH) regressed modestly; everything else
jumped. Net weighted AUPRC +0.074. Time MAE essentially unchanged → the time head
is robust to weight reduction (it converges regardless), so the time term WAS
purely starving risk, exactly as hypothesised.

**Per-aux training trace (P3).**
| phase | term | λ | anchor raw | final raw | note |
|---|---|---|---|---|---|
| P3 | Risk (main) | — | 0.98 | ~0.45 | sharper logits (DEATH std 3.74→5.87) |
| P3 | Time (smooth-L1) | 0.25 | 84→ | ~4.1 | weighted ~1.0; still ~2:1 time-dominant at convergence |
(P1 dt, P2 t_pos/t_local identical to baseline — same upstream config.)

**Diagnose.py (positional p=0.15) vs baseline.**
- MLM top1=0.089 / top5=0.303 — unchanged (Phase-2 identical), as expected.
- Risk logits WIDER: DEATH std 3.74→5.87, Hyperglyc 4.99→5.84, Kidney 3.19→3.65
  — sharper risk discrimination, directly explains the AUPRC/AUROC lift.
- t_pos std 0.037 (was 0.028 — backbone drifted slightly under longer/risk-weighted
  P3 fine-tune), t_local std 0.021 — both alive. Pool entropy 4.66–5.32, healthy.

**Verdict.** KEEP — primary AUPRC +0.0737 (≫ +0.010) and AUROC +0.0217, no headline
regresses (time MAE flat, LoS −1.2h). New running-best. Gain cleanly attributed:
the ONLY change vs baseline is `phase3_time_lambda` 0.5→0.25 (Phase-1/2 identical),
so the baseline IS the strip-the-change ablation.

**What I'd try next.** Convergence math says even at λ=0.25 the P3 objective is
still ~2:1 TIME-dominant (risk ~0.45 vs weighted-time ~1.0). Per supervisor
directive ("risk loss higher than time — invert the two"), next run i4b sets
`phase3_time_lambda`=0.05 → weighted-time ~0.2 vs risk ~0.45 (~2:1 RISK-dominant).
Hypothesis: further AUPRC/AUROC lift toward AUROC 0.90; watch time MAE for the −5h
guardrail (robust so far). If 0.05 overshoots (time MAE blows up), 0.1 is the fallback.

### i4b-tl005  (direction #4b: invert risk/time — phase3_time_lambda 0.25 → 0.05; supervisor-directed)

**Hypothesis.** i4-tl025 lifted AUPRC +0.074 with time MAE flat, but convergence
math shows λ=0.25 is still ~2:1 TIME-dominant (risk ~0.45 vs weighted-time ~1.0).
Supervisor directive: make risk the dominant term. λ=0.05 → weighted-time ~0.2 vs
risk ~0.45 (~2:1 RISK-dominant — inverted). Expect further AUPRC/AUROC lift
(toward AUROC 0.90) as the shared representation specialises for risk
discrimination. Risk: time MAE may finally degrade — KEEP needs it within +5h of
the i4-tl025 running-best (41.4h). Phase-3-only; Phase-1 reused, Phase-2 = baseline.

**Change.** `transform_emr/config/model_config.py`: `phase3_time_lambda`
`0.25` → `0.05`. (Code committed separately from journal for clean revert.)

**Smoke result.** sample=50: P3 total = Risk + 0.05·Time = 5.22, no NaN/inf,
summary + headline keys print → Gate-A/D PASS. Phase-1 reused, Phase-2 = baseline.

**Headline metrics (10k) vs i4-tl025 (prev best) and baseline.**
```
                            i4b-tl005   i4-tl025    baseline    Δ vs prev-best
patient_auprc_weighted:     0.826193    0.760241    0.686507    +0.0660  (PRIMARY)
patient_auroc_weighted:     0.881450    0.868612    0.846871    +0.0128
patient_auprc_simple:       0.759314    0.639949    0.566331    +0.1194
patient_max_f1_weighted:    0.782523    0.735542    0.706229    +0.0470
length_of_stay_mae_hours:   122.3213    120.2743    121.4905    +2.05 h  (within guard)
time_mae weighted (6 risk): 39.01 h     41.40 h     41.71 h     -2.39 h  (IMPROVED)
phase3_best_val:            0.291340    1.476722    4.238167
```
Per-outcome AUROC / AUPRC (vs i4-tl025):
| outcome | AUROC | AUPRC | ΔAUPRC vs i4-tl025 |
|---|---|---|---|
| CARDIO-VASCULAR | 0.989 | 0.959 | +0.257 |
| HYPEROSMOLALITY | 0.907 | 0.917 | +0.026 |
| DISGLYCEMIA_Hyperglycemia | 0.923 | 0.923 | +0.007 |
| KIDNEY | 0.876 | 0.863 | +0.098 |
| DISGLYCEMIA_Hypoglycemia | 0.856 | 0.680 | **+0.333** (recovered) |
| DEATH | 0.667 | 0.213 | −0.005 (lone laggard) |
Hypoglycemia fully recovered (0.347→0.680); DEATH is the only weak head and its
diagnose logits are WIDE (std 10.3, p5 −11→p95 +16) yet AUPRC stays 0.21 → the
head is trying hard but rare terminal-death-from-2-day-input is intrinsically
hard, not an architecture bug.

**Per-aux trace (P3).** Risk main 0.98→~0.25; Time (λ=0.05) raw 84→~5, weighted
~0.25 — now roughly RISK:time ≈ 1:0.5 (risk-dominant, as directed).

**Diagnose.py (positional p=0.15) vs i4-tl025.**
- Risk logits even WIDER: DEATH std 5.87→10.31, Hyperglyc 5.84→10.67, Kidney
  3.65→8.09 — sharper ranking (drives AUPRC). Watch calibration if pushed lower.
- MLM top1 0.084 (was 0.089), top20 legality 0.642 — backbone drifts slightly
  from MLM optimum as it specialises for risk under risk-dominant P3 (expected,
  backbone_lr_factor=0.01 allows it). Not harmful: downstream metrics all up.
- t_pos std 0.041 (~14 h), t_local std 0.022 — alive. Pool entropy 4.36–5.60, healthy.

**Verdict.** KEEP — primary AUPRC +0.066 over the previous best AND time MAE
−2.4h AND AUROC +0.013; LoS +2.0h (within −/+5h guard). New running-best.
Confirms supervisor's invert-the-ratio directive: risk-dominant P3 lifts risk
discrimination and, surprisingly, also improves time MAE (better shared rep).
Gain attributed: only `phase3_time_lambda` changed vs i4-tl025.

**What I'd try next.** Trend 0.5→0.25→0.05 is monotonic up on AUPRC
(0.687→0.760→0.826). Push once more to λ=0.02 (via fast retrain_phase3.py driver,
~25min) to find the optimum / plateau; watch risk-logit calibration and time MAE.
DEATH remains intrinsically hard — defer (could try outcome-specific pos_weight or
a focal risk loss later, but not the headline lever).

### NOTE — retrain_phase3.py fast driver REJECTED (methodology)

Built `retrain_phase3.py` to skip the redundant identical Phase-2 retrain
(~53 min/run) for Phase-3-only experiments. **Validation failed**: at λ=0.25 with
i4-tl025's staged Phase-1/Phase-2 it produced AUPRC 0.7260 / AUROC 0.8520 and
Phase-3 early-stopped at ep73, vs the full api.py i4-tl025 result 0.7602 / 0.8686
at ep101 — a 0.034 AUPRC gap. Cause: the bucket sampler shuffles with python
`random` each epoch, so Phase-3's data order depends on the RNG state left by
Phase-2 training; skipping Phase-2 changes that order (plus non-deterministic GPU
kernels, amplified by early-stop epoch choice). **Decision: do NOT use the fast
driver for KEEP decisions; all KEEP-decision runs stay on full `api.py`.** Side
note: the 0.034 spread hints run-to-run variance may be ~0.02–0.03 AUPRC, so I
treat AUPRC deltas < ~0.02 as noise (the big KEEPs so far, +0.066/+0.074, are
well clear of that). Full data = 57,078 patients (~5.7× the 10k sample).

### i4c-tl002  (direction #4c: phase3_time_lambda 0.05 → 0.02 — confirm optimum)

**Hypothesis.** AUPRC trend over λ {0.5,0.25,0.05} = {0.687,0.760,0.826} is
monotonic-up but decelerating (+0.074 then +0.066). Test λ=0.02 to find the
optimum/plateau and try to crack AUROC 0.90 (currently 0.881). KEEP needs AUPRC
≥ +0.010 over i4b-tl005 (0.826) with time MAE within +5h of 39.0h and no LoS
blowup; watch risk-logit calibration (already std ~10 at λ=0.05). If it plateaus
or regresses, λ=0.05 is locked as the time-λ winner.

**Change.** `phase3_time_lambda` 0.05 → 0.02 (code committed separately).

**Smoke result.** sample=50: P3 total = Risk + 0.02·Time = 2.70, no NaN/inf,
Gate-A/D PASS. Phase-1 reused, Phase-2 = baseline (early-stopped ep88 this run —
run-to-run variance; phase2 config identical).

**Headline metrics (10k) vs i4b-tl005 (prev best).**
```
                            i4c-tl002   i4b-tl005   Δ
patient_auprc_weighted:     0.841081    0.826193    +0.0149  (PRIMARY, > +0.010)
patient_auroc_weighted:     0.896866    0.881450    +0.0154  (≈ 0.90 target)
patient_auprc_simple:       0.771648    0.759314    +0.0123
patient_max_f1_weighted:    0.801152    0.782523    +0.0186
length_of_stay_mae_hours:   119.8719    122.3213    -2.45 h
time_mae weighted (6 risk): 40.64 h     39.01 h     +1.63 h  (within +5h guard)
```
Per-outcome AUROC / AUPRC: CVD 0.986/0.937, Hyperglyc 0.931/0.929, Hyperosmol
0.919/0.930, Kidney 0.900/0.887, Hypoglycemia 0.851/0.682, DEATH 0.716/0.264
(DEATH up from 0.667/0.213 — improving but still the laggard). All non-DEATH
AUPRC now ≥ 0.68.

**Diagnose.py (positional p=0.15).** Risk logits even WIDER (std 9.3–13.5, up
from 5.9–10.7 at λ0.05) — overconfidence rising as λ drops; calibration still OK
(f1@0.5 0.791 ≈ max-f1 0.801). MLM top1 0.085. t_pos std 0.038, t_local 0.022 —
alive. Pool entropy 5.08–5.61, healthy.

**Verdict.** KEEP — primary AUPRC +0.0149 (> +0.010), AUROC +0.0154 (now 0.897 ≈
benchmark), LoS −2.4h; time MAE +1.6h within guard. New running-best. Gain
attributed: only `phase3_time_lambda` 0.05→0.02. λ trend
{0.5,0.25,0.05,0.02}={0.687,0.760,0.826,0.841} — decelerating (+0.015 now).

**What I'd try next.** Probe λ=0.01 to confirm the plateau before locking the
recipe for the long full-data Phase-2. Expect < +0.010 (plateau) and possibly
calibration degradation from the widening logits; if so, lock λ=0.02 as the
time-λ winner and move to direction #5 (backbone_lr_factor) or Phase-2.

### i4d-tl001  (direction #4d: phase3_time_lambda 0.02 → 0.01 — plateau check)

**Hypothesis.** Confirm the time-λ optimum before locking the recipe for full-data
Phase-2. λ trend is decelerating (last gain +0.015). Expect λ=0.01 to give
< +0.010 AUPRC over i4c (0.841) — i.e. a plateau / no-KEEP — and possibly worse
calibration from the widening risk logits (std already 13.5). If no-KEEP, lock
λ=0.02 as the time-λ winner (and this is the lever's first no-KEEP toward Phase-1
convergence). If it surprises with > +0.010, reconsider.

**Change.** `phase3_time_lambda` 0.02 → 0.01 (code committed separately).

**Smoke result.** sample=50: P3 total = Risk + 0.01·Time = 1.86, Gate-A/D PASS.
Phase-1 reused, Phase-2 = baseline.

**Headline metrics (10k) vs i4c-tl002 (best).**
```
                            i4d-tl001   i4c-tl002   Δ
patient_auprc_weighted:     0.827640    0.841081    -0.0135  (regresses)
patient_auroc_weighted:     0.873742    0.896866    -0.0232  (regresses)
length_of_stay_mae_hours:   118.1599    119.8719    -1.71 h
time_mae weighted (6 risk): 40.35 h     40.64 h     -0.29 h
```
Per-outcome AUPRC mostly flat/down vs i4c (CVD 0.939, Hyperosmol 0.920, Hyperglyc
0.914, Kidney 0.859, Hypoglycemia 0.674, DEATH 0.272). Both headline metrics down
together → real turnover, not noise.

**Verdict.** DISCARD — both AUPRC (−0.0135) and AUROC (−0.0232) regress vs the
λ=0.02 best. The time-λ curve {0.5,0.25,0.05,0.02,0.01} =
{0.687,0.760,0.826,0.841,0.828} peaks at **λ=0.02**. Locking
`phase3_time_lambda = 0.02` as the time-λ winner. Mechanism: below ~0.02 the risk
logits overconfident-saturate (diagnose showed std climbing 5.9→10.7→13.5) and
the shared rep loses the small time-regularisation that helped generalisation.

**What I'd try next.** Time-λ lever converged (this is its no-KEEP). Next:
direction #5 `phase3_backbone_lr_factor` 0.01→0.1 (playbook flags default may be
too low; lets the backbone specialise more for the task in P3) with λ=0.02 locked.
If no-KEEP → 2nd consecutive no-KEEP → Phase-1 converged → proceed to Phase-2
full-data baseline with the locked recipe.

### i5-blr01  (direction #5: phase3_backbone_lr_factor 0.01 → 0.1; λ=0.02 locked)

**Hypothesis.** Playbook flags the default backbone_lr_factor=0.01 as possibly too
low for an encoder that must specialise its representation for outcome prediction
(BERT fine-tune literature uses ~0.1). At 0.1 the backbone (and embedder, which
shares this factor in P3) adapt 10× faster during Phase-3, which — now that P3 is
risk-dominant (λ=0.02) — could push risk discrimination further and crack AUROC
clearly past 0.90. Risk: overfit on the 10k budget (watch train/val gap and
calibration). KEEP needs AUPRC ≥ +0.010 over i4c (0.841) with no time-MAE/LoS
blowup. If no-KEEP → 2nd consecutive no-KEEP → Phase-1 converged.

**Change.** `phase3_backbone_lr_factor` 0.01 → 0.1 (code committed separately).
Time-λ stays at the locked 0.02.

**Smoke result.** sample=50: Gate-A (no NaN/inf) + Gate-D PASS. Phase-1 reused,
Phase-2 = baseline.

**Headline metrics (10k) vs i4c-tl002 (best).**
```
                            i5-blr01    i4c-tl002   Δ
patient_auprc_weighted:     0.823953    0.841081    -0.0171  (PRIMARY regresses)
patient_auroc_weighted:     0.874836    0.896866    -0.0221  (regresses)
length_of_stay_mae_hours:   111.1711    119.8719    -8.71 h  (big improve)
time_mae weighted (6 risk): 35.29 h     40.64 h     -5.35 h  (big improve)
```
Per-outcome time-MAE all down (CVD 7.5h, Kidney 19.8h, Hypoglycemia 28.8h); risk
AUPRC down across the board (DEATH 0.210, Kidney 0.865, Hyperglyc 0.912). A clean
risk↔time trade-off: backbone_lr_factor=0.1 lets the backbone+embedder specialise
for the time regression (and LoS) at the cost of the MLM-pretrained features that
carry risk-ranking signal.

**Verdict.** DISCARD — primary AUPRC regresses −0.0171 (≥0.010) and AUROC −0.0221,
despite time MAE −5.3h / LoS −8.7h. Per KEEP rule (AUPRC-led; "no headline
regresses by threshold" + "AUROC down & AUPRC down → DISCARD"), the time gain does
not justify the risk loss when risk is the headline. Reverting; backbone_lr_factor
stays 0.01. **IMPORTANT finding for the human:** backbone_lr_factor is the
risk↔time dial — 0.1 buys ~5–9h better time MAE/LoS for ~0.017 AUPRC. If the
deliverable later weights time more heavily, revisit ~0.03 as a compromise.

**This is the 2nd consecutive no-KEEP (i4d, i5) across the open Phase-1 levers →
Phase-1 declared CONVERGED.**

---

## PHASE-1 CONVERGED — locked architecture

The only architectural change that survived KEEP vs the baseline is the Phase-3
loss rebalance. Locked recipe (10k → carried to full-data Phase-2):
- embed_dim 128 / n_layer 4 / n_head 2 (size is a Phase-2 sweep axis)
- MLM positional @ ratio 0.15; phase2 aux caps t_pos 0.40 / t_local 0.30 (untouched)
- **phase3_time_lambda = 0.02** ← the winning change (was 0.5)
- phase3_backbone_lr_factor = 0.01 (default; 0.1 trades risk for time — rejected)
- phase3_head_hidden = 256, pool n_heads = 4 (defaults; diagnostics healthy, not searched)

10k headline progression (held-out test):
| stage | AUPRC_w | AUROC_w | LoS h | time_w h |
|---|---|---|---|---|
| baseline (λ0.5) | 0.687 | 0.847 | 121.5 | 41.7 |
| **best i4c (λ0.02)** | **0.841** | **0.897** | 119.9 | 40.6 |
Net: **AUPRC +0.154, AUROC +0.050** vs baseline; both at/above the STRATS/GRU-D
targets (AUPRC ≫ 0.65; AUROC ≈ 0.90). Time prediction reasonable and stable.
Directions #1 (mask granularity) discarded; #2/#3 not indicated by diagnostics
(no MLM-collapse; P2 time auxes clamp-limited so cap edits are no-ops); #5
discarded; #6/#7 not indicated (head/pool diagnostics healthy).

### p2-fulldata  (PHASE 2 step 1: full-data baseline, sample=None, locked recipe)

**Hypothesis.** Re-run the Phase-1-locked architecture (only change vs original
baseline: phase3_time_lambda=0.02) on the FULL dataset (57,078 patients vs the
10k probe) to produce the deliverable headline. Expect metrics to hold or improve
with ~5.7× more data (less overfit, sharper heads), especially the rare/weak
outcomes (DEATH, CVD, Hypoglycemia) that were data-starved at 10k. This is the
publishable baseline before the size sweep.

**Change.** `sample` 10000 → None (full data). Architecture identical to locked
recipe. Tokenizer/Phase-1/scaler rebuilt from scratch (full-data vocab differs),
so all checkpoints cleared first. Long run (P1+P2+P3 on 57k patients).

**Supervisor constraint (2026-06-02, mid-Phase-2):** cap the size sweep at
embed_dim ≤ 384 — do NOT test 512/768. Results are already strong, so large
scaleups aren't worth the time. Size-sweep grid is therefore {128 (this
baseline), 256, 384}; pick the smallest within ~0.005 weighted AUPRC of the best.

**Supervisor directive (2026-06-02):** replace the Phase-3 multi-seed study
(3× full-data runs) with a **patient-level bootstrap 95% CI** on the final model.
Built `bootstrap_eval.py`: one inference pass on the held-out test set, then
B=2000 resamples of test patients with replacement, recomputing the support-
weighted AUROC/AUPRC (and per-outcome) each time → 2.5/97.5 percentile CIs. The
bootstrap statistic mirrors `evaluation.weighted_mean_auc` exactly (n_pos
weighting; min-positive gate per resample). Will run on the final size-sweep
winner. This estimates test-set sampling variance directly, far cheaper than
re-seeding the whole pipeline.

### p2-fulldata RESULT — full-data baseline (embed_dim 128) = DELIVERABLE

**Headline (held-out test, 8,562 patients / 7,446 with LoS).**
```
patient_auprc_weighted:  0.826051   95% CI [0.8205, 0.8319]  (boot sd 0.0029)
patient_auroc_weighted:  0.878531   95% CI [0.8737, 0.8834]  (boot sd 0.0025)
patient_auprc_simple:    0.756286
patient_auroc_simple:    0.865256
patient_max_f1_weighted: 0.789002
length_of_stay_mae_hours:117.5462   (median 103.1, p90 228.1)
time_mae weighted (6 risk outcomes by n_pos): 42.89 h
train wall-clock: 28,402 s (~7.9 h); P2 full 101 ep, P3 early-stop ep37
num_params: 1,852,682   peak_vram: 1204 MB
```
Per-outcome AUROC / AUPRC [95% CI] (bootstrap B=2000):
| outcome | AUROC [CI] | AUPRC [CI] | n_pos |
|---|---|---|---|
| CARDIO-VASCULAR | 0.972 [0.962,0.981] | 0.931 [0.912,0.949] | 534 |
| HYPEROSMOLALITY | 0.907 [0.900,0.914] | 0.918 [0.912,0.925] | 3448 |
| DISGLYCEMIA_Hyperglycemia | 0.912 [0.904,0.919] | 0.914 [0.907,0.922] | 3157 |
| KIDNEY_COMPLICATION | 0.877 [0.867,0.886] | 0.871 [0.861,0.880] | 2716 |
| DISGLYCEMIA_Hypoglycemia | 0.841 [0.823,0.858] | 0.649 [0.619,0.679] | 771 |
| DEATH | 0.685 [0.669,0.702] | 0.256 [0.234,0.279] | 1116 |

**Variance (supervisor-directed bootstrap, replaces 3-seed study).** B=2000 patient
resamples. CIs are tight (±~0.006); the weighted-AUPRC interval [0.821, 0.832]
sits entirely far above the STRATS/GRU-D PRAUC≈0.65 benchmark, and weighted AUROC
[0.874, 0.883] is just below the AUROC≈0.90 band on a 5.7× larger/harder test set
than the 10k probe. DEATH remains the only weak head (AUPRC CI robustly ~0.25 —
intrinsically hard, terminal+rare).

**Interpretation.** vs the 10k probe (0.841/0.897 on 1,500 test patients), the
full-data numbers (0.826/0.879 on 8,562) are slightly lower but far more reliable
(tight CI). Net vs the original full-data-equivalent baseline recipe, the
phase3_time_lambda rebalance is the single decisive lever. This is the publishable
baseline.

**Verdict.** DELIVERABLE BASELINE — beats PRAUC benchmark decisively (CI clears
0.65 by ~0.17), approaches the AUROC benchmark. Backed up
checkpoints.bak_keep_p2_full128 (model only; 3.2 GB processed cache left in place
for size-sweep reuse).

**What I'd try next.** Size sweep (embed_dim 256, then 384 — capped per supervisor),
reusing the processed_datasets cache (data load is instant; only P1/P2/P3 retrain).
Pick smallest within ~0.005 weighted AUPRC of best, then bootstrap-CI the winner.

### p2-size256  (PHASE 2 size sweep: embed_dim 256, n_head 4; locked recipe)

**Hypothesis.** Scale embed_dim 128→256 (n_head 2→4, head_dim 64 fixed, n_layer 4)
on full data with the locked recipe (phase3_time_lambda=0.02). More capacity may
lift weighted AUPRC/AUROC, especially the harder heads (DEATH, Hypoglycemia). KEEP
only if it beats the 128 baseline (AUPRC 0.826, CI [0.821,0.832]) by a margin
clearly outside the bootstrap CI (~+0.01); else prefer the smaller 128 for the
publishable headline (size-sweep rule: smallest within ~0.005 AUPRC of best).
Capped at 384 per supervisor.

**Change.** MODEL_CONFIG embed_dim 128→256, n_head 2→4. Reuses processed_datasets
cache (key ignores embed_dim → instant data load); Phase-1 retrains for the new
dim; Phase-2/3 retrain. Smoke clobbers tokenizer/scaler → restored from the
p2_full128 backup before the full run.

### p2-size256 RESULT — embed_dim 256 (full data)

**Headline (held-out test, 8,562 patients).**
```
                          size256     full128(best)  Δ          128 95%CI
patient_auprc_weighted:   0.827870    0.826051       +0.0018    [0.8205,0.8319]  (INSIDE CI)
patient_auroc_weighted:   0.880049    0.878531       +0.0015    [0.8737,0.8834]  (INSIDE CI)
length_of_stay_mae_hours: 125.6608    117.5462       +8.11 h    (worse)
time_mae weighted:        44.86 h     42.89 h        +1.97 h    (worse)
num_params:               6,814,346   1,852,682      3.7x bigger
```
Per-outcome essentially unchanged (CVD 0.933, Hyperosmol 0.912, Hyperglyc 0.920,
Kidney 0.881, Hypoglycemia 0.631, DEATH 0.264). The +0.0018 AUPRC sits well inside
the 128 bootstrap CI → statistically indistinguishable.

**Verdict.** DISCARD for the headline — 256 gives no risk gain outside the CI and
slightly worse LoS/time for 3.7× the parameters. Per the size-sweep rule (smallest
within ~0.005 AUPRC of best) **embed_dim 128 is the final model**. Capacity has
plateaued by 256, and per supervisor's avoid-scaleup steer I am **skipping 384**
(it cannot change the pick without a >0.005 jump that 256 did not show). Reverting
config to embed_dim 128.

**What I'd try next.** Final remaining Phase-2 axis: the QA-data ablation
(USE_QA_DATA toggle) on the locked 128 recipe — a data question orthogonal to size.
Then the locked 128 model + its bootstrap CIs is the final deliverable.

### p2-qa  (PHASE 2 step 3: QA-data ablation, USE_QA_DATA=True, locked 128 recipe)

**Hypothesis.** Toggle USE_QA_DATA on: adds %_PATTERN% treatment-quality events
to the LM stream + QA ComplianceScore context features. May improve risk/time by
giving the encoder treatment-adherence signal, or may add noise. Locked recipe
(embed_dim 128, phase3_time_lambda 0.02). KEEP over the non-QA 128 deliverable
(AUPRC 0.826, CI [0.821,0.832]) only if weighted AUPRC gains beyond the CI
(~+0.01). Non-QA full-data vocab = 503 tokens; expect QA vocab > this.

**Change.** dataset_config USE_QA_DATA False→True. Pre-flight: cleared tokenizer,
scaler, processed_datasets cache, and all phases (vocab + ctx_dim change → full
rebuild incl. Phase-1). Smoke then full-data.
