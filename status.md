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
