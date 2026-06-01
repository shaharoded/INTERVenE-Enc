# autoresearch — EMR Event Prediction (BERT-pivot)

Autonomous architecture and hyperparameter search for the BERT-style
bidirectional transformer adaptation of my thesis's EMR complication-prediction
model. Adapted from Karpathy's autoresearcher framework.

An AI agent drives the loop: edit `transform_emr/`, train all three phases on
the held-out 70/15/15 patient split, evaluate on the held-out test set via
single-pass inference, KEEP or DISCARD per the rules in `program.md`, log to
`results/results.tsv`, repeat.

See `program.md` for the model overview, the three-phase experiment plan,
and the loop discipline the agent follows.

---

## Repository layout

```
api.py                       fixed: data load, training orchestration, eval call
evaluation.py                fixed: single-pass eval (patient AUPRC/AUROC/F1, LoS MAE)
build_cache.py               helper to pre-build a minimal processed_datasets.pt
program.md                   model overview + 3-phase experiment plan + loop discipline
status.md                    sweep narrative (agent appends ### <tag> blocks)
results/
  results.tsv                full per-experiment ledger
transform_emr/
  config/
    model_config.py          MODEL_CONFIG + TRAINING_SETTINGS (agent edits this)
    dataset_config.py        paths, tokens, USE_QA_DATA flag
    tak-repo-portable.json   knowledge-base hierarchy (Mediator output)
  embedder.py                Phase-1 EMREmbedding + train_embedder
  transformer.py             Phase-2/3 EMREncoder + TaskHeads + pretrain/finetune
  inference.py               single-pass predict()
  dataset.py                 DataProcessor, EMRTokenizer, dataloaders
  loss.py / schedulers.py / utils.py / diagnose.py
data/source/                 temporal_data.csv + context_data.csv  (gitignored)
checkpoints/                 phase{1,2,3}/ckpt_best.pt + tokenizer + scaler
```

`api.py` and `evaluation.py` are the fixed contract. The agent only edits
`transform_emr/config/model_config.py` (primary) and architecture files under
`transform_emr/`.

---

## Three-phase model

- **Phase 1 — `EMREmbedding`** — hierarchical token embeddings (raw → concept →
  concept+value → position), Time2Vec for absolute timestamps, static patient
  context. Loss: per-window outcome BCE + Δt MSE.
- **Phase 2 — `EMREncoder`** — 4-layer bidirectional transformer with AdaLN-Zero
  patient conditioning + temporal RoPE. MLM pre-training: full-vocab CE on
  masked positions (atomic-interval mask) + `t_pos` and `t_local` time
  auxiliaries.
- **Phase 3 — `TaskHeads`** — per-outcome attention pool + shared MLP →
  (risk_head, time_head). Backbone at `phase3_backbone_lr_factor` LR; heads at
  full LR. Risk BCE + λ_time · smooth-L1.

Inference is a single bidirectional pass via `inference.predict`. No
autoregressive generation. Output: one row per patient with `P_<outcome>` and
`T_<outcome>` columns.

Evaluation: 3-way patient split by `PatientId` with seed=42 (70 % train / 15 %
val / 15 % test). The 15 % test split is held out, processed once with the
training-fitted scaler, and consumed only by `evaluate_on_test_set`.

---

## Running locally

```bash
pip install -e .

# Place source CSVs at:
#   data/source/temporal_data.csv
#   data/source/context_data.csv

# Smoke test (50 patients, 1 epoch per phase, fast). In
# transform_emr/config/model_config.py set:
#   TRAINING_SETTINGS["sample"] = 50
#   TRAINING_SETTINGS["phase1_n_epochs"] = 1
#   TRAINING_SETTINGS["phase2_n_epochs"] = 1
#   TRAINING_SETTINGS["phase3_n_epochs"] = 1
python api.py > smoke.log 2>&1
grep "^patient_auroc_weighted:\|^---" smoke.log

# Full run (restore sample=None and original epoch counts)
python api.py > run.log 2>&1
grep "^patient_auroc_weighted:\|^patient_auprc_weighted:\|^length_of_stay_mae_hours:\|^peak_vram_mb:" run.log
```

All output goes to a final summary block after the `---` separator.
Per-outcome AUROC/AUPRC/F1 + time-MAE are emitted as grep-friendly TSV rows
(`patient_per_outcome\t…`, `time_head_mae_hrs\t…`).

---

## Running on a RunPod GPU pod

The agent runs autonomously inside `tmux` so it survives SSH drops.

**One-time setup on a fresh pod:**

```bash
# SSH in (RunPod gives you the host/port on the Connect page)
ssh root@<HOST> -p <PORT> -i ~/.ssh/id_ed25519

# Install tmux + Node 20 + Claude Code
apt-get update && apt-get install -y tmux
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
npm install -g @anthropic-ai/claude-code

# Clone + install Python deps
cd /workspace
git clone https://github.com/shaharoded/Transform-EMR-Encoder.git autoresearch
cd autoresearch
git checkout autoresearch-updates       # working branch — all experiments live here
pip install -e .

# Create a non-root user (Claude refuses --dangerously-skip-permissions as root)
useradd -m -s /bin/bash agent
cp -r /root/.ssh /home/agent/ && chown -R agent:agent /home/agent/.ssh
chmod -R a+rwX /workspace/autoresearch
```

**SCP the data files (from local PowerShell):**

```powershell
scp -P <PORT> -i ~/.ssh/id_ed25519 data\source\temporal_data.csv root@<HOST>:/workspace/autoresearch/data/source/
scp -P <PORT> -i ~/.ssh/id_ed25519 data\source\context_data.csv  root@<HOST>:/workspace/autoresearch/data/source/
```

**Start the agent (each session):**

```bash
ssh root@<HOST> -p <PORT> -i ~/.ssh/id_ed25519
su - agent
cd /workspace/autoresearch
tmux new -s claude
claude --dangerously-skip-permissions
```

In the Claude prompt, kick off the loop with something like:

> Read `program.md`. We are on branch `autoresearch-updates` of
> `Transform-EMR-Encoder`. Run the experiment loop autonomously — smoke test,
> full run, KEEP/DISCARD, update `status.md` + `results/results.tsv` after
> every meaningful step, and commit & push to `autoresearch-updates`. Never
> `git reset --hard` or force-push — DISCARDs go through `git revert`.

Detach with `Ctrl-b d`. Reattach later with `tmux attach -t claude`.

**Monitoring from your laptop:**

```bash
# Read the live journal without disturbing the agent
ssh root@<HOST> -p <PORT> -i ~/.ssh/id_ed25519 "cat /workspace/autoresearch/status.md"

# Or pull whenever the agent has pushed
git pull --ff-only
```

**Before stopping the pod:** push the branch from the pod so nothing is lost
on container disk. SCP off `checkpoints/` if you want to keep the trained
weights (gitignored, too large for git).

---

## Metrics

All on the held-out 15 % test split via a single bidirectional encoder pass,
per outcome, then averaged across outcomes with ≥ 1 % patient prevalence.
`RELEASE_EVENT` is excluded from AUC/AUPRC/F1 (≈ ¬DEATH in this cohort) and
reported separately via length-of-stay MAE.

- **`patient_auprc_weighted`** — **primary**, ↑. Support-weighted mean per-outcome AUPRC.
- **`time_head_mae_hrs:<outcome>`** — per-outcome time-head MAE on positives, ↓.
- **`length_of_stay_mae_hours`** — RELEASE time-head regression, ↓.
- **`patient_auroc_weighted`** — secondary, ↑. Discrimination sanity check.
- **`patient_max_f1_weighted`** / **`patient_f1_at_0_5_weighted`** — calibration sanity, ↑.

All means are weighted by support (`n_pos`); macro means are emitted but never
drive KEEP decisions. See `program.md` for the full priority rules.
