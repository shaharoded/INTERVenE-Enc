# autoresearch — BERT-style EMR Encoder

## What this model is

A bidirectional BERT-style transformer for **outcome prediction from temporal
interval EMR data**. Three phases:

1. **Phase 1 — `EMREmbedding`** (`transform_emr/embedder.py`). Hierarchical
   token embeddings (raw → concept → concept+value → position), Time2Vec for
   absolute timestamps, static patient context added as AdaLN-Zero bias. Loss:
   per-window outcome BCE + Δt MSE auxiliary.
2. **Phase 2 — `EMREncoder`** (`transform_emr/transformer.py`). 4-layer
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

**Metric priorities (read carefully):**

- **AUPRC weighted by support is the headline** (`patient_auprc_weighted`).
  EMR outcomes are class-imbalanced; AUPRC is what moves the needle and is
  what STRATS / HEART report.
- **Time MAE matters** — both per-outcome time-head MAE and LoS MAE. Use the
  support-weighted means, not the macro means.
- AUROC is still tracked (`patient_auroc_weighted`) but is **less
  needle-moving**. Treat it as a discrimination sanity check, not a target.
- Prefer **weighted-by-support** over macro for both AUROC and AUPRC. Macro
  means are reported for sanity but never used in KEEP decisions.

`RELEASE_EVENT` is excluded from AUC/AUPRC/F1 (≈ ¬DEATH in this cohort) and
reported via length-of-stay MAE.

Headline keys emitted by `api.py` after the `---` separator:

- `patient_auprc_weighted`  ← **primary**
- `patient_auroc_weighted`
- `patient_max_f1_weighted`, `patient_f1_at_0_5_weighted`
- `length_of_stay_mae_hours`, `length_of_stay_median_hrs`, `length_of_stay_p90_hours`
- `patient_per_outcome\t…` TSV rows
- `time_head_mae_hrs\t…` TSV rows

**Do not touch `api.py` or `evaluation.py`.** Agent edits
`transform_emr/**` and `transform_emr/config/*.py` only.

---

## Experiment plan

### Phase 1 — Algorithmic architecture search (sample = 10 000)

Goal: decide on the final algorithmic configuration. Every Phase-1 run uses
`TRAINING_SETTINGS["sample"] = 10000` so iterations stay cheap. Each run is
a smoke + 10k full run + KEEP/DISCARD per the loop discipline below.

**Scope discipline — architectural changes only.** Phase 1 is for
algorithmic / structural decisions: mask granularity, loss formulation,
aux-loss curriculum, head topology, pool design, scheduler stage layout.

Do **not** burn Phase-1 iterations on tuning numeric hyperparameters that
are not part of the architecture — `dropout`, `learning_rate` per phase,
`weight_decay`, `batch_size`, `grad_accumulation_steps`, `lr_warmup_epochs`,
`early-stop-patience`. Those are size-coupled and belong to Phase 2's size
sweep, where a bigger model may genuinely want different HPs. Touching them
during Phase 1 confounds the architectural signal and adds search variance
the 10k budget cannot afford.

If a loss trace or `diagnose.py` output suggests an HP is *misconfigured*
for the current architecture (e.g. dropout=0 leaves pool entropy saturated;
LR is so high that auxes diverge), file the observation as a note in
`status.md` and revisit it in Phase 2 against the locked architecture.
Don't fork the Phase-1 sweep to chase it. only if it's a very extreme "missrepresentaion"
of the current architecture you are trying, then you may try a HP modification experiment. But never to "explore". 

Run the listed directions in order. The agent must also propose **new
directions** of its own based on observation:

- Per-aux loss traces (which auxes are descending, which are flat or
  diverging)
- `diagnose.run_diagnostics(model, val_dl)` output: MLM top-1/top-5,
  pool-attention entropy per outcome, time-aux residual percentiles,
  risk-logit distributions, legality starvation
- Per-outcome AUPRC and time-MAE per ledger row — does a specific outcome
  family lag and suggest a targeted change (e.g. CBM ratio for noisy
  outcomes, pool head count for saturated attention)?

If an observation points to an architectural change not in the list below,
**propose and test it** — one falsifiable hypothesis per loop iteration, same
KEEP gate.

**Performance expectations + creative latitude.** STRATS reports
`patient_auroc_weighted ≈ 0.89–0.92` on MIMIC mortality / decompensation
with a much simpler triplet encoder; HEART reports similar with hierarchical
masking; GRU-D matches that band with imputation-aware recurrence. Our
backbone is strictly stronger than any of these (4-layer bidirectional
transformer + temporal RoPE + AdaLN-Zero patient conditioning), and our task
is a strict superset (risk *plus* per-outcome time estimation, not just max
risk over a 12-day horizon). The risk side of the headline should therefore
**match or beat STRATS** — if it isn't, something architectural is wrong,
not just under-tuned.

Stuck at sub-STRATS AUPRC after 2–3 loop iterations, or hitting a wall
where every direction in the list fails to lift the headline? **Be
creative**:

- Read the relevant module(s) end-to-end, instrument the suspect path with
  per-batch logging, and isolate WHICH term / pathway is the problem
  (risk-head gradient norm collapsing? pool attention degenerate? MLM CE
  starving on rare tokens? time-aux dominating risk-aux at calibration?).
- Borrow loss-term ideas from the EMR-modelling literature when they
  address the diagnosed failure. Examples worth considering when symptoms
  match — not a sequential list to grind through:
  * **STRATS** — observation triplet encoding, contrastive pre-training on
    observation co-occurrence. (Note: STRATS's value-aware embedding has no
    direct analogue here — value is already absorbed into our hierarchical
    token structure, e.g. `GLUCOSE → GLUCOSE_TREND → GLUCOSE_TREND_Inc →
    GLUCOSE_TREND_Inc_START`, so the embedder sees the discretised value
    bin as part of the token identity rather than as a separate scalar.)
  * **HEART** — hierarchical masking + reconstruction at the concept level,
    family-aware MLM curriculum.
  * **GRU-D** — masking + decay variables in the embedding so the encoder
    knows *how stale* each measurement is, not just *what* it was.
  * **mTAN / SeFT / Raindrop** — continuous-time attention, irregularly-
    sampled time-series encoders.
  Cite the paper in `status.md` next to the hypothesis when adapting an
  idea — the journal entry must say "borrowing GRU-D's decay term because
  diagnose.py shows the encoder is treating 12h-old labs identically to
  fresh ones."
- Architecture-side moves (head topology, query design, intermediate
  conditioning, additional auxes) are fair game when the loss-side
  diagnosis points there.

Time estimation is the part of the task with no direct STRATS/HEART
precedent — under-performance on `time_head_mae_hrs:<outcome>` is **not**
automatically a failure, but it must be debugged the same way: read the
time-head trace, check `phase3_time_lambda` calibration, verify the
positive-only mask, and if needed propose a new time-loss formulation
(e.g. survival-style log-likelihood instead of smooth-L1).

**Listed directions (start here):**

1. **MLM mask granularity.**
   `TRAINING_SETTINGS["phase2_mlm_mask_mode"] ∈ {"positional", "hierarchical"}`
   × `phase2_mlm_ratio ∈ {0.15, 0.25}`.
   Hierarchical replaces non-interval masks with `[MASK_RAW_<family>]` per
   raw-concept family — narrows the MLM target space, can speed convergence
   on family-confounded concepts. Tokenizer auto-emits the family specials
   when the flag flips (requires clearing tokenizer cache; see below).

2. **MLM head loss.**
   Currently full-vocab CE (hard-coded in `pretrain_transformer.run_epoch`).
   Swap to `MaskedFocalBCE.from_counts(counts=tokenizer.token_counts,
   gamma ∈ {0.5, 1.0, 2.0})` if `probe_mlm_accuracy` shows the head
   collapsing onto the modal token. Only worth it on evidence of rare-token
   collapse — BCE is heavier than CE and can hurt calibration when GT is
   genuinely a single token.

3. **Phase-2 aux-loss caps.**
   Edit `TRAINING_SETTINGS["phase2_scheduler"]["aux_fraction_caps"]`:
   `t_pos ∈ {0.20, 0.40, 0.60}` × `t_local ∈ {0.15, 0.30, 0.45}`.
   Time signal is load-bearing for both LoS and per-outcome time MAE — too
   low under-trains time prediction, too high pulls the encoder away from
   MLM quality.

4. **Phase-3 time-loss weight.**
   `phase3_time_lambda ∈ {0.25, 0.5, 1.0, 2.0}`.
   Watch the `tr_risk` / `tr_time` ratio in the Phase-3 epoch log to pick
   the right anchor — `time_head_mae_hrs` per outcome should descend, but
   `patient_auprc_weighted` must not regress.

5. **Phase-3 backbone LR factor.**
   `phase3_backbone_lr_factor ∈ {0.0, 0.01, 0.1}`.
   `0.0` is the fully-frozen sanity baseline; `0.1` follows BERT fine-tune
   literature. Default `0.01` may be too low for an encoder that needs to
   specialise its representation for outcome prediction.

6. **Task-head capacity.**
   `phase3_head_hidden ∈ {128, 256, 512}`.
   Pool already mixes per-outcome queries across the hidden state; the
   shared MLP exists mostly to non-linearly re-express the pooled vector.

7. **Per-outcome pool head count.**
   `EMREncoder.attach_task_heads(n_heads=…) ∈ {2, 4, 8}`.
   Persisted in checkpoints under `task_heads_n_heads`. Only worth searching
   if `diagnose.probe_pool_attention` reports saturated or near-zero entropy
   in the headline runs.

8. **Agent-proposed directions.**
   Anything the loss traces or diagnose.py output suggests — document the
   hypothesis in `status.md` before running it.

Phase-1 exits when the agent judges the architecture is stable (≥ 2 loop
iterations in a row produce no KEEP across all open directions, and
diagnostics show no obvious lever left).

---

### Phase 2 — Benchmarking the finalised architecture

The architecture from Phase 1 is now locked. Phase 2 measures how well it
scales, not how to redesign it.

1. **Full-data baseline.** Re-run the Phase-1 winner with `sample = None` to
   produce the headline numbers on the full dataset. This is the deliverable
   baseline.

2. **Size sweep (subject to GPU).**
   Grid: `embed_dim ∈ {128, 256, 384, 512, 768}` (head_dim=64 fixed; n_head =
   embed_dim/64; n_layer=4 fixed). Each variant a full-data run.
   - OOM → halve `batch_size` and double `grad_accumulation_steps`. If still
     OOM, that's the size ceiling on the rented GPU.
   - **HP tuning is allowed here**, scoped to the *current size*. A bigger
     encoder may want lower LR / higher dropout / different
     `phase3_backbone_lr_factor` to avoid overfit on the same data budget.
     If a size variant under-performs and a loss trace shows a clear HP
     pathology (e.g. train/val gap blowing up early), run one targeted HP
     adjustment for that variant and re-confirm. Document each adjustment
     in `status.md` against the size it belongs to — different sizes may
     legitimately settle on different HPs.
   - Pick the smallest variant within ~0.005 weighted AUPRC of the best
     (prefer smaller for the publishable headline).

3. **QA-data ablation.** Toggle `dataset_config.USE_QA_DATA`. This is a
   tokenizer-affecting change — **pre-flight**:
   ```
   rm -f checkpoints/tokenizer.pt checkpoints/scaler.pkl checkpoints/processed_datasets.pt
   rm -rf checkpoints/phase1
   ```
   Phase 1 (embedder) retrains from scratch because the vocab + ctx_dim
   change. Verify `len(tokenizer.token2id) > non-QA value` before scaling.
   Run smoke first, then full-data. Compare against the Phase-2 baseline —
   keep whichever wins on weighted AUPRC.

The output of Phase 2 is **one final model configuration** (architecture +
size + QA toggle) that we'll benchmark in Phase 3.

---

### Phase 3 — Final benchmarking, multi-seed

At this point the model should be hitting STRATS/HEART territory
(`patient_auroc_weighted` ≳ 0.90, with the dataset-dependent AUPRC the
real story). If we are not there, return to Phase 1 — something
architectural is still off.

**Multi-seed runs.** Re-run the final configuration with three seeds:
`SEED ∈ {2023, 2024, 2025}` (set in `transform_emr/config/model_config.py`).
Each seed is a full-data run with the locked architecture.

**Report.** For each headline key (`patient_auprc_weighted` first,
`length_of_stay_mae_hours`, `time_head_mae_hrs:<outcome>`,
`patient_auroc_weighted`) report **mean ± std across the three seeds**, KDD
style. Per-outcome rows in the same format.

That's the publishable result.

---

## Loop discipline

```
1. Read program.md. Check git log + last rows of results/results.tsv.
2. Propose ONE change with a falsifiable hypothesis. Document the hypothesis
   in status.md BEFORE running.
3. SMOKE (sample=50, phase{1,2,3}_n_epochs=1):
     python api.py > smoke.log 2>&1
   Gate-A: no NaN/inf in any tr_* loss term.
   Gate-B: every aux's raw magnitude within ~1–2 OOM of its main loss.
   Gate-C: calibrated λ in [1e-3, 10].
   Gate-D: summary block prints; all headline keys present.
4. git add <files> && git commit -m "<tag>: change / why / expected" && git push.
5. EXPERIMENT (sample=10000 in Phase 1; sample=None in Phase 2/3):
     python api.py > run.log 2>&1
   POST-TRAIN:
   T1: every aux's raw loss decreases across its active phase.
   T2: early stop did not fire before auxes finished ramping.
   T3: diagnose.run_diagnostics shows real signal — MLM top-1 above
       majority-class baseline, pool attention entropy non-saturated,
       time-aux residual percentiles sane.
6. Append row to results/results.tsv with the headline keys.
7. Write `### <tag>` block in status.md → `Verdict: KEEP|DISCARD — …`.
   Mandatory per-aux training trace table (unlock epoch, λ_max, anchor
   raw_aux, final raw_aux, Δ%). Flag |Δ| < 5 % as "not learning."
8. Journal commit + push.
9. DISCARD → `git revert --no-edit <CODE_SHA> && git push`.
   **Never** `git reset --hard` or force-push — the failed commit + its
   `status.md` block must stay in history so the next iteration can read
   why the direction was abandoned.
10. KEEP → cp -r checkpoints checkpoints.bak_keep_<tag>.
    Run an ablation that strips the new change → confirms gain attribution.
11. After each KEEP, re-eval the running best at 10k to refresh baseline (if not already produced by the eval that produced the KEEP).
12. FULL-DATA CONFIRM (sample=None) is reserved for Phase 2 / Phase 3.
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

`transform_emr.diagnose.run_diagnostics(model, val_dl)` runs the full sweep:
MLM top-1/top-5 accuracy, time-aux residual percentiles, pool attention
entropy per outcome, risk-logit distributions, legality starvation. **Call
this after every architecture change** to catch silent collapse before
staking a full eval on it. The agent should treat this output as the primary
signal for proposing new directions in Phase 1.

## Communication discipline — `status.md` is the contract

`status.md` is **the only place the human sees what the agent is doing**.
Treat every entry as a research-journal message to the supervisor: complete
sentences, no shorthand from earlier in the session, no assumption the
reader was watching.

For every loop iteration, append a `### <tag>` block to `status.md`
containing — in this order — and commit it **before the next iteration
starts**:

1. **Hypothesis.** One sentence. What change, why it should help, what
   metric you expect to move and by how much. Cite the paper / failure-mode
   if you borrowed an idea or chased a diagnose.py signal.
2. **Change.** The exact files / config keys touched. Diff-equivalent
   summary, not the diff itself.
3. **Smoke result.** Pass/fail per gate A–D. If fail, what broke and how
   you fixed it before running full.
4. **Headline metrics.** All headline keys verbatim from the summary block,
   plus per-outcome AUPRC/time-MAE deltas vs running best.
5. **Per-aux training trace table** (mandatory): unlock epoch, λ_max,
   anchor raw_aux, final raw_aux, Δ %. Flag |Δ| < 5 % as "not learning."
6. **Diagnose.py observations.** MLM top-1/top-5, pool attention entropy,
   time-aux residual percentiles, risk-logit distributions. Even when
   nothing surprising — record the boring numbers so trends are visible
   across iterations.
7. **Verdict.** `KEEP | DISCARD — <one-sentence reason>`.
8. **What I'd try next.** Even on KEEP. The next agent iteration (or the
   human) reads this to decide direction. Be specific: "diagnose.py shows
   pool-head 3 attending uniformly → try n_heads=2 to see if specialisation
   sharpens."

If something is **surprising or anomalous** — record it even if it doesn't
fit the KEEP/DISCARD verdict. A failed experiment with a clear diagnosis
is more valuable to the next iteration than a quiet rollback. Never let an
observation die in a `git revert` without first being written down.

## Git discipline — never lose history

- The remote is **`https://github.com/shaharoded/Transform-EMR-Encoder.git`**.
- The working branch is **`autoresearch-updates`**. Every commit and push
  goes there. Do not commit to `main`, do not branch off `autoresearch-
  updates` without naming the new branch.
- **Forbidden operations** (no exceptions, even on "broken" state):
  * `git reset --hard <anything>` on `autoresearch-updates`.
  * `git push --force` / `git push -f` to `autoresearch-updates` or `main`.
  * `git checkout .` / `git restore .` / `git clean -fd` that would discard
    uncommitted `status.md` / `results/results.tsv` notes.
  * Force-deleting commits, branches with `-D`, or rewriting history with
    rebase / amend after a push.
- **Rollback path is `git revert`**, always. A failed experiment is
  reverted with `git revert --no-edit <CODE_SHA> && git push` — the failed
  commit (and its `status.md` block) stays in history as a record of what
  was tried and why it didn't work.
- Before any destructive-looking command, stop and check: does this throw
  away a `status.md` entry, a `results/results.tsv` row, or a checkpoint
  backup? If yes, find the non-destructive alternative.
- If the working tree is dirty in a confusing way, `git stash` (which is
  reversible) and inspect, never `git reset --hard`.

## Reproducibility

- Repo: `https://github.com/shaharoded/Transform-EMR-Encoder.git`.
- Branch: `autoresearch-updates`. No force-push, ever.
- Ledger: `results/results.tsv`. Append one row per experiment with the
  headline keys + per-outcome breakdown reference. Never overwrite rows.
- Running-best backups: `checkpoints.bak_keep_<tag>/`.
- Per-experiment full log: `run.log` (kept in the experiment commit's tree).
- Journal: `status.md` (agent appends `### <tag>` blocks here — append
  only, never edit older blocks except to add a follow-up note dated and
  signed).
