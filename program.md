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

<!-- INSERT RESEARCH DIRECTIONS / EXPERIMENTS HERE -->

When kicking off a new sweep, append a section below with:

- Goal (one paragraph).
- Listed directions (numbered, one falsifiable hypothesis each).
- Performance expectations + the literature baseline you want to beat.
- Exit condition (when does this sweep terminate?).

Past sweeps and their verdicts live in `results/status.md`; the headline
ledger is `results/results.tsv`.

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

- Repo: `https://github.com/shaharoded/Transform-EMR-Encoder.git`.
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
