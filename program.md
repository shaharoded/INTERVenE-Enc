# autoresearch — BERT-style EMR Encoder

## What this model is

A bidirectional BERT-style transformer for **outcome prediction from temporal
interval EMR data**. Three phases:

1. **Phase 1 — `EMREmbedding`** (`intervene_enc/embedder.py`). Hierarchical
   token embeddings (raw → concept → concept+value → position), Time2Vec for
   absolute timestamps, static patient context added as AdaLN-Zero bias. Loss:
   per-window outcome BCE + Δt MSE auxiliary.
2. **Phase 2 — `InterveneEncoder`** (`intervene_enc/transformer.py`). 4-layer
   bidirectional transformer with AdaLN-Zero patient conditioning + temporal
   RoPE. MLM pre-training: full-vocab CE on masked positions (atomic-interval
   mask) + `t_pos` (time-since-admission) and `t_local` (time-to-neighbour)
   auxiliaries, all scheduled by `LambdaScheduleController` with per-aux
   fraction caps.
3. **Phase 3 — `TaskHeads`** (per-outcome attention pool + shared MLP →
   risk_head + time_head). Backbone runs at `phase3_backbone_lr_factor × lr`;
   heads at full LR. Risk = `BCEWithLogitsLoss` with per-outcome `pos_weight`;
   time = `smooth_l1_loss` over positive patients only.

Inference is a **single bidirectional pass** per patient via
`inference.predict`. Output: one row per patient with `P_<outcome>` (sigmoid
of risk logit) and `T_<outcome>` (softplus of time logit, hours). No
autoregressive trajectory.

## Eval framework (locked)

For each (patient, outcome) one `(P_<outcome>, label)` pair from the
held-out 15 % test split, then per outcome:

- **AUPRC** (primary; weighted by support across outcomes)
- **Time MAE** to the nearest GT occurrence (positives only; weighted by support)
- **Length-of-stay MAE** = `|T_RELEASE_EVENT − GT_RELEASE_time|` (always reported)
- **AUROC** (secondary; weighted by support)
- **max-F1 / F1@0.5** (calibration sanity)

**Metric priorities:**

- **AUPRC weighted by support is the headline** (`patient_auprc_weighted`).
  EMR outcomes are class-imbalanced; AUPRC is what moves the needle.
- **Time MAE matters** — both per-outcome time-head MAE and LoS MAE. Use the
  support-weighted means, not the macro means.
- AUROC is still tracked but is **less needle-moving**. Discrimination sanity
  check, not a target.
- Prefer **weighted-by-support** over macro for both AUROC and AUPRC.

`RELEASE_EVENT` is excluded from AUC/AUPRC/F1 (≈ ¬DEATH in this cohort) and
reported via length-of-stay MAE.

**Do not touch `api.py` or `evaluation.py`.** The fixed-driver / fixed-eval
contract is what makes ledger rows comparable across iterations. Edits live in
`intervene_enc/**` and `intervene_enc/config/*.py`.

---

## Research directions / experiments

### Stage A / B / C / D / E — closing the STraTS gap, then sizing and abstraction

**Goal.** Beat STraTS on every headline metric on the `user_mimic_iv` cohort.
The encoder already wins on weighted AUPRC (0.749 vs 0.612) but loses on
weighted AUROC (0.845 vs 0.890), almost entirely because of two concentrated
failure modes:

1. **RELEASE length-of-stay MAE** at 112 h (STraTS 48 h) — caused by a
   smooth-L1 time-head loss that's L1 in practice, exerts no gradient pressure
   to predict per-patient variance, and collapses to the conditional median.
2. **DEATH AUPRC / AUROC** (0.234 / 0.638 vs STraTS 0.518 / 0.893) —
   rare-positive underdiscrimination on the risk side.

**Headline metric for all decisions in this sweep is weighted AUPRC**
(`patient_auprc_weighted`). AUROC is tracked but used only as a sanity check.
Per-outcome AUPRC on **DEATH** and **SEVERE_HYPOGLYCEMIA** is the second
decision metric — we want to see improvement there, so any regression from the current result needs to be inspected.

The sweep is five experiment directions. A, B run unconditionally. C is **optional**.
D and E run after the architecture + recipe are chosen.

**Sample size convention.** Exp A, B, C run at `sample=10000` (Phase 1/2/3)
— they are decision sweeps, not deliverables, and the 10 k subset is enough
for a clear ranking under the loop's KEEP rule. Exp D (size) and Exp E
(abstraction) **must run on the full dataset** (`sample=None` for Phase
2/3, `sample=10000` for Phase 1 per program convention) because their
outputs are headline deliverables — sizes and the abstraction
transferability finding both go into the paper. **Parameter counts must be
reported for every D and E variant** (one row per `M-<embed_dim>` arm in
the size sweep, one row per Exp-E arm) alongside the metric table.

#### Stop-at-B gate (skip Exp C)

Skip Exp C if the Exp-B winner clears both targets on the sample=10000 eval:

- RELEASE LoS MAE ≤ 70 h, **AND**
- DEATH AUPRC ≥ 0.40 (almost double the locked 0.234 - within the same areas as the SS-STraTS DEATH. 
We don't need to win STraTS on every outcome, but I want to close this gap).

If both targets hold, the Exp-B winner is the chosen recipe — run Exp D and
Exp E on it. If either misses, run Exp C and re-test the same targets on
the Exp-C winner.

#### Exp A — Phase-3 loss + regulariser bundle (Phase-3-only re-finetune)

Resume from the existing full-data Phase-1 and Phase-2 checkpoints; only
Phase-3 retrains. Each variant lands in its own
`checkpoints.bak_keep_<tag>/` dir.

Config knobs added in commit `8954e9e`:
- `phase3_time_lambda` (per-outcome z-MSE time loss weight). Default 0.5.
- `phase3_focal_gamma` (focal-BCE γ on the risk head). Default 2.0.
- `phase3_cbm_p` (CBM input-token replacement during Phase-3 only). Default 0.25.
- `phase3_pool_dropout` (independent of backbone dropout). Default 0.20.
- `phase3_pos_weight_mode` ∈ {`"inv_prev"`, `"uniform"`}. Default `"inv_prev"`.

**Exp A is a knock-out sweep that must explicitly pick a winner per knob.**
The agent runs the four variants below, then journals one chosen Phase-3
recipe whose every knob has been individually justified by the data.

| Tag | Config delta vs `A0` | What it decides |
|---|---|---|
| `A0` (baseline bundle) | (all defaults above) | Anchor for the three knock-outs. |
| `A_no_focal` | `phase3_focal_gamma = 0.0` | Keep focal-BCE iff `A0` > `A_no_focal` on weighted AUPRC (margin ≥ 0.005) **or** on DEATH AUPRC (margin ≥ 0.01). Otherwise strip and fall back to γ=0. |
| `A_no_cbm` | `phase3_cbm_p = 0.0` | Same rule, CBM dropout vs none. |
| `A_posweight_uniform` | `phase3_pos_weight_mode = "uniform"` | Same rule, inverse-prevalence vs uniform. The agent reports both arms and **picks the one with higher weighted AUPRC, tie-broken by DEATH AUPRC**, then journals which mode was chosen. |

Only do the γ ∈ {1, 3} sweep if `A0` beats `A_no_focal` cleanly (so we know γ
is doing something real) — otherwise it's tuning on noise.

The output of Exp A is a single "Phase-3 recipe" config dict, journalled
verbatim into `results/status.md`. Exp B uses it as a frozen starting
point.

#### Exp B — Attention-head budget (Phase-1/2/3 retrain)

Start from the Exp-A winner's Phase-3 recipe. Vary attention heads while
keeping `embed_dim = 128` constant — total parameter count moves only via
head allocation, so the comparison isolates "more attention pathways at
smaller per-head dim" (the STraTS inductive bias).

| Tag | `n_head` | `head_dim` | Note |
|---|---:|---:|---|
| `B_4heads`  | 4  | 32 | Moderate step toward STraTS |
| `B_8heads`  | 8  | 16 | Aggressive |
| `B_16heads` | 16 | 8  | STraTS-match (their MIMIC-III config) |

**Early-stop the sweep** if `B_8heads` is within 0.005 weighted AUPRC of
`B_16heads` and DEATH AUPRC is within 0.01 — the smaller variant is
preferred per program convention.

#### Exp C — Pretraining objective (optional; Phase-1/2/3 retrain)

**Only runs if the Exp-B winner misses the stop-at-B gate above.** Start from
the Exp-B winner. Add a forecasting auxiliary to Phase 2: predict the next
position's `(concept_id, value_id)` at a 2 h horizon alongside MLM,
scheduled by `LambdaScheduleController` with its own `aux_fraction_cap`.

| Tag | Config delta |
|---|---|
| `C_forecast_aux` | `forecast_next` aux added to `phase2_scheduler.aux_fraction_caps`, cap = 0.2 |
| `C_forecast_aux_strong` | (only if `C_forecast_aux` moves the needle but undershoots) cap = 0.4 |

#### Exp D — Total-size ablation (after A+B+C land)

Once the Phase-3 recipe (Exp A) and head ratio (Exp B, optionally C) are
fixed, sweep total model capacity by scaling `embed_dim` and `n_head`
**proportionally** so the per-head dim stays at the Exp-B winner's value.
`n_layer` stays at 4 (program-wide convention) unless an arm explicitly
varies it.

Notation: `M-<embed_dim>` (same as the AR P6 sweep).

| Tag | `embed_dim` | `n_head` | Approx params |
|---|---:|---:|---:|
| `D_M128` | 128 | (Exp-B winner) | ≈ 1.85 M (current size) |
| `D_M192` | 192 | embed_dim / head_dim | ≈ 4 M |
| `D_M256` | 256 | embed_dim / head_dim | ≈ 7 M |
| `D_M384` | 384 | embed_dim / head_dim | ≈ 16 M |
| `D_M512` | 512 | embed_dim / head_dim | (only if D_M384 wins by ≥ CI margin) |

Each row's journal block **must report the exact param count**
(`count_parameters`) alongside its headline metrics.

**Stop rule.** As soon as a step improves weighted AUPRC by **less than the
bootstrap 95 % CI half-width of the previous size's eval** (≈ 0.005 typical),
the previous size is the chosen total capacity. The program prefers smaller
under tie. If `D_M256` already saturates (within CI of `D_M192`), `D_M192`
wins and we don't run `D_M384`+.

OOM handling: Size is the ceiling. No need to start fighting it with batching sizes and grad accumulations, unless it happened before M384.

#### Exp E — Temporal-abstraction ablation (transferability check; single retrain)

Tests whether the encoder's gains depend on the TAK-abstracted
`temporal_data.csv` upstream or generalise to a simpler discretisation
scheme. The Exp-D winner is already the full-data headline run on the
original TAK input, so Exp E is a **single retrain** of the same
architecture + recipe on the std-binned input and a direct comparison
against the Exp-D winner — no separate baseline arm needed.

**Local preprocessing script** — will be created at
`ablation/preprocess_std_bins.py` **before Exp E runs** (the human will hand
off the script and the resulting `temporal_data.csv` together). The script
runs on the developer's local machine and the std-binned CSV is scp'd to
the pod. Until then, the agent should treat Exp E as a parked deliverable
and not attempt to write the script or run the ablation. Specification (for
record-keeping; the script will conform):

1. **Inputs**: `data/source/mimic-iv-input-data.csv` (raw measurements,
   STraTS-style; this file lives **only locally** — it is not part of the
   GPU pod's setup — so the script runs on the developer's machine and the
   resulting `temporal_data.csv` is shipped to the pod via scp), and
   `data/source/context_data.csv` (unchanged).
2. **Numeric measurement bucketing**:
   - For each `ConceptName` whose `Value` parses to a numeric (measurement
     concepts only), compute the **train-split** mean μ and std σ.
   - Bin every observation into one of 5 categories:
     - `VERY_LOW`  if `v < μ − 2σ`
     - `LOW`       if `μ − 2σ ≤ v < μ − 1σ`
     - `NORMAL`    if `μ − 1σ ≤ v ≤ μ + 1σ`
     - `HIGH`      if `μ + 1σ < v ≤ μ + 2σ`
     - `VERY_HIGH` if `v > μ + 2σ`
   - The output row's `ConceptName` becomes `<orig_concept>_STD_<bin>` and
     its `Value` becomes the bin string. Numeric value is dropped; the
     encoder will indicator-encode it on input (value = 1.0).
3. **Interval collapsing** (the "concat consecutive same bin within 24 h"
   rule): per patient, sort observations by `StartDateTime`. For each
   measurement concept, walk consecutive observations; if two adjacent
   observations of the same concept share the same bin **and** their
   `StartDateTime` are within 24 h, collapse them into one event with
   `StartDateTime` = first observation's time and `EndDateTime` =
   last observation's time. This gives an interval representation.
4. **Outcomes pass through unchanged**: rows whose `ConceptName` matches any
   outcome regex in `EVENT_OUTCOME_REGEX` are copied verbatim — no binning,
   no collapse. Same for the terminal tokens (`DEATH`, `RELEASE`) and the
   `ADMISSION` event.
5. **Categorical / boolean measurements** (Value not numeric): pass through
   unchanged.
6. **Validation block** (script must print before exiting):
   - Total row count before / after collapse.
   - Per-outcome support (n_pos / n_total) on the train split — must match
     the support extracted by the existing autoresearch preprocess to within
     ±0.5 % per outcome.
   - Per-concept bin distribution (sanity: NORMAL ≈ 68 %, ±2σ outliers ≈ 2.3 %
     each side).
7. **Output**: write `ablation/data/source_std_bins/temporal_data.csv` and
   copy `context_data.csv`. The autoresearch loader is unchanged; only the
   `TEMPORAL_DATA_FILE` / `CONTEXT_DATA_FILE` constants point at the new
   directory for this experiment.

**Single arm**:

| Tag | Config |
|---|---|
| `E_std_bins` | Exp-D winner architecture + Exp-A recipe, retrained Phase-1/2/3 on the std-binned `temporal_data.csv`. |

Report parameter count alongside the headline metrics — the std-bins vocab
will differ from TAK's after the `<concept>_STD_<bin>` expansion, so the
total param count will differ too and should be documented.

**What it answers.** If `E_std_bins` lands within 0.010 weighted AUPRC of
the Exp-D winner and DEATH AUPRC isn't worse by > 0.01, the encoder's gains
are **architecture-driven, not TAK-abstraction-driven** — that's the
transferability finding. If it loses materially, TAK abstraction is part of
the recipe and the paper has to say so.

#### Additional gates for this sweep (overlay on Loop discipline)

The standard smoke gates A–D, post-train T1–T3, and KEEP rule in the Loop
discipline section apply unchanged. The checks below are extra requirements
that apply to every variant of every experiment in this sweep:

- **Per-loss magnitude monitor.** Risk loss, time loss, MLM CE, dt aux,
  t_pos, t_local, and (when present) forecast — emit raw and weighted
  values each epoch. Flag any term that drops below `1e-4 × main` for ≥ 2
  epochs as **starving**; recommend the λ bump and do not declare KEEP. In
  Phase 3 specifically, if `lambda_time × time_loss < 1e-2 × risk_loss`
  for an extended span, the time term is effectively absent and the LoS /
  time-MAE numbers cannot be trusted.
- **Focal sanity.** Once per epoch-1, print one batch's loss with γ=0
  alongside the focal loss. If `focal_loss / bce_loss < 0.05`, γ is too
  aggressive — journal the observation and recommend a lower γ.
- **Per-outcome AUPRC blocker.** After Phase-3, include the per-outcome
  AUROC / AUPRC / time-head-MAE table in the journal with explicit
  attention to `DEATH_EVENT`, `SEVERE_HYPOGLYCEMIA_EVENT`, and LoS. A
  regression of ≥ 0.01 absolute AUPRC (or ≥ 5 h on LoS) vs the locked
  baseline is a blocker even if weighted averages lift.
- **Diagnose-time probes** required in the journal block before KEEP:
  - `probe_time_head_predictions`: every outcome must show
    `pred_std > 0.5 × gt_std`. If an outcome is collapsed, do not KEEP —
    investigate, try different constants, or drop the experiment.
  - `probe_outcome_logit_distribution`: no head saturated.
  - `probe_pool_attention`: per-outcome entropy in
    `[0.3·log(seq_len), 0.9·log(seq_len)]`.

#### Exit condition

The sweep terminates when:
- Exp A is complete and the Phase-3 recipe winner is journalled, **AND**
- Exp B is complete (with early-stop applied), **AND**
- Either the Exp-B winner clears the stop-at-B gate **OR** Exp C is complete, **AND**
- Exp D terminates at the within-CI size, **AND**
- Exp E (`E_std_bins`) is complete and compared against the Exp-D winner.

---

## Loop discipline

```
1. Read program.md. Check git log + last rows of results/results.tsv.
2. Propose ONE change with a falsifiable hypothesis. Document the hypothesis
   in results/status.md BEFORE running.
3. SMOKE (sample=50, phase{1,2,3}_n_epochs=1):
     python api.py --smoke > results/logs/smoke.log 2>&1
   Gate-A: no NaN/inf in any tr_* loss term.
   Gate-B: every aux's raw magnitude within ~1–2 OOM of its main loss.
   Gate-C: calibrated λ in [1e-3, 10].
   Gate-D: summary block prints; all headline keys present.
4. git add <files> && git commit -m "<tag>: change / why / expected" && git push.
5. EXPERIMENT (sample=10000 in Phase 1; sample=None in Phase 2/3):
     python api.py > results/logs/run.log 2>&1
   POST-TRAIN:
   T1: every aux's raw loss decreases across its active phase.
   T2: early stop did not fire before auxes finished ramping.
   T3: diagnose.run_diagnostics shows real signal — MLM top-1 above
       majority-class baseline, pool attention entropy non-saturated,
       time-aux residual percentiles sane.
6. Append row to results/results.tsv with the headline keys.
7. Write `### <tag>` block in results/status.md → `Verdict: KEEP|DISCARD — …`.
   Mandatory per-aux training trace table (unlock epoch, λ_max, anchor
   raw_aux, final raw_aux, Δ%). Flag |Δ| < 5 % as "not learning."
8. Journal commit + push.
9. DISCARD → `git revert --no-edit <CODE_SHA> && git push`.
   **Never** `git reset --hard` or force-push.
10. KEEP → cp -r checkpoints checkpoints.bak_keep_<tag>.
    Run an ablation that strips the new change → confirms gain attribution.
11. After each KEEP, re-eval the running best to refresh baseline (if not
    already produced by the eval that produced the KEEP).
12. FULL-DATA CONFIRM (sample=None) is reserved for late-stage benchmarking.
```

### KEEP rule (weighted-by-support, AUPRC-led)

- All smoke gates A–D + post-train T1–T3 passed.
- ≥ 1 headline lifts past noise:
  - `patient_auprc_weighted` ≥ +0.010 (**primary lever**), OR
  - support-weighted mean `time_head_mae_hrs` ≤ −5 h, OR
  - `length_of_stay_mae_hours` ≤ −5 h, OR
  - `patient_auroc_weighted` ≥ +0.010 (treat as secondary).
- No headline regresses by the same threshold.
- If AUROC improves but AUPRC drops by ≥ 0.005, **DISCARD** — AUPRC is the
  headline.

**Sanity check for time predictions** — every run emits `lift_hours =
baseline_mae − model_mae` per outcome (and for LoS), where the baseline is
the MAE-optimal constant predictor (predict GT median time-to-event). Use
this as a sanity signal, not a KEEP gate:
- `lift_hours > 0` ⇒ the time head has learned timing beyond a constant.
- `lift_hours ≤ 0` ⇒ the time head is no better than predicting the GT
  median; AUPRC may still be fine but **flag in `status.md`** so the failure
  is on the record before the next iteration.

## Cheap optimisation: reuse Phase-1 cached checkpoint

`api.py` automatically reuses the Phase-1 embedder when
`(embed_dim, time2vec_dim, ctx_dim)` matches the cached checkpoint, so
Phase-2/3-only changes never re-train the embedder.

If the **tokenizer** changes (vocab additions — `USE_QA_DATA` toggle or
hierarchical MLM mode adding `[MASK_RAW_<family>]` tokens), pre-clear:

```
rm -f checkpoints/tokenizer.pt checkpoints/scaler.pkl checkpoints/processed_datasets.pt
rm -rf checkpoints/phase1
```

## Useful diagnostics

`intervene_enc.diagnose.run_diagnostics(model, val_dl)` runs the full sweep:
MLM top-1/top-5 accuracy, time-aux residual percentiles, pool attention
entropy per outcome, risk-logit distributions, legality starvation. **Call
this after every architecture change** to catch silent collapse before
staking a full eval on it. CLI: `python api.py --diagnose`.

Bootstrap CIs (B=2000, patient resample) on a trained checkpoint:
`python api.py --bootstrap [B]` — reads the same data pipeline as the
training run, loads the final checkpoint, and emits per-outcome AUROC/AUPRC
CIs plus the weighted headline CIs. Implementation: `evaluation.bootstrap_evaluate`.

## Communication discipline — `results/status.md` is the contract

`results/status.md` is **the only place the human sees what the agent is
doing**. Treat every entry as a research-journal message to the supervisor:
complete sentences, no shorthand from earlier in the session, no assumption
the reader was watching.

For every loop iteration, append a `### <tag>` block to `results/status.md`
containing — in this order — and commit it **before the next iteration
starts**:

1. **Hypothesis.** One sentence. What change, why it should help, what
   metric you expect to move and by how much. Cite the paper / failure-mode
   if you borrowed an idea or chased a diagnose.py signal.
2. **Change.** The exact files / config keys touched. Diff-equivalent
   summary, not the diff itself.
3. **Smoke result.** Pass/fail per gate A–D.
4. **Headline metrics.** All headline keys verbatim from the summary block,
   plus per-outcome AUPRC/time-MAE deltas vs running best.
5. **Per-aux training trace table** (mandatory): unlock epoch, λ_max,
   anchor raw_aux, final raw_aux, Δ %. Flag |Δ| < 5 % as "not learning."
6. **Diagnose.py observations.** MLM top-1/top-5, pool attention entropy,
   time-aux residual percentiles, risk-logit distributions.
7. **Verdict.** `KEEP | DISCARD — <one-sentence reason>`.
8. **What I'd try next.** Even on KEEP. Be specific.

If something is **surprising or anomalous** — record it even if it doesn't
fit the KEEP/DISCARD verdict. A failed experiment with a clear diagnosis
is more valuable to the next iteration than a quiet rollback.

## Git discipline — never lose history

- Working branch is **`autoresearcher-updates`**. No commits to `main`.
- **Forbidden** (no exceptions): `git reset --hard <ref>` on the working
  branch, `git push --force`, `git checkout .` / `git restore .` /
  `git clean -fd` that would discard uncommitted journal entries,
  force-deleting commits/branches with `-D`, rewriting history after a push.
- **Rollback path is `git revert`**, always.
- Before any destructive-looking command, stop: does this throw away a
  `status.md` entry, a `results.tsv` row, or a checkpoint backup? If yes,
  find the non-destructive alternative.
- If the working tree is dirty in a confusing way, `git stash` and inspect,
  never `git reset --hard`.

## Reproducibility

- Repo: `https://github.com/shaharoded/INTERVenE-Enc.git`.
- Branch: `autoresearcher-updates`. No force-push, ever.
- Ledger: `results/results.tsv`. Append one row per experiment with the
  headline keys + per-outcome breakdown reference. Never overwrite rows.
- Running-best backups: `checkpoints.bak_keep_<tag>/` (gitignored;
  archived separately for paper / inference).
- Per-experiment logs: `results/logs/run.log` (overwritten per run; commit
  the version that matters with the KEEP).
- Journal: `results/status.md` (agent appends `### <tag>` blocks here —
  append only, never edit older blocks except to add a follow-up note
  dated and signed).
- Final results summary: `results/README.md` (regenerated after each
  publishable milestone; the journal is the source of truth, the README is
  the polished read).
