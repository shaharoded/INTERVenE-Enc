# autoresearch — EMR Event Prediction

Bidirectional BERT-style transformer for multi-outcome prediction from
temporal interval EMR data. Inputs are knowledge-base temporal abstractions
(TAK / KB intervals); outputs are per-outcome patient-level risk + per-outcome
time-to-event. The repo also hosts an autonomous-research loop where an AI
agent edits the encoder, retrains, evaluates, and KEEPs / DISCARDs each
direction. Adapted from Karpathy's autoresearcher framework.

- **Methodology, eval contract, KEEP rule, loop discipline** → [`program.md`](program.md).
- **Polished headline results (CIs, per-outcome breakdowns, ablations)** → [`results/README.md`](results/README.md).
- **Full agent journal (every iteration)** → [`results/status.md`](results/status.md).
- **Headline ledger (one row per experiment)** → [`results/results.tsv`](results/results.tsv).

> The `results/` folder is gitignored; it lives locally / on Drive, not on
> GitHub. The agent writes to it at training time and the human reads it
> when writing the paper.

---

## Repository layout

```
api.py                       fixed: data load, training orchestration, eval call
                             CLI: --smoke | --build-cache | --diagnose | --bootstrap [B]
evaluation.py                fixed: single-pass eval + bootstrap CIs (patient AUPRC/AUROC/F1, LoS MAE)
program.md                   model overview + eval contract + loop discipline (methodology)
README.md                    this file — usage + ops

intervene_enc/
  config/
    model_config.py          MODEL_CONFIG + TRAINING_SETTINGS (the things you edit)
    dataset_config.py        paths, tokens, USE_QA_DATA flag
    tak-repo-portable.json   knowledge-base hierarchy (TAK / KB intervals)
  embedder.py                Phase-1 EMREmbedding + train_embedder
  transformer.py             Phase-2/3 InterveneEncoder + TaskHeads + pretrain/finetune
  inference.py               single-pass predict()
  dataset.py                 DataProcessor, EMRTokenizer, dataloaders
  diagnose.py                run_diagnostics — MLM/pool/time-aux probes
  schedulers.py              LR + aux-lambda schedule controllers
  utils.py                   tensor helpers (masking, LUTs, targets)

results/                     ← gitignored, Drive-backed
  README.md                  polished paper-facing results
  status.md                  full agent journal
  results.tsv                ledger
  logs/                      per-run logs (run.log, smoke.log, boot_*, diag_*)

data/source/                 temporal_data.csv + context_data.csv  (gitignored)
checkpoints/                 phase{1,2,3}/ckpt_best.pt + tokenizer + scaler  (gitignored)
checkpoints.bak_keep_<tag>/  per-KEEP weight backups  (gitignored)
checkpoints_export/          local mirror of deliverable weights  (gitignored)
```

`api.py` and `evaluation.py` are the fixed contract — they define the
training pipeline and the metrics that every ledger row is comparable against.
Edits live under `intervene_enc/` (primarily `config/model_config.py`).

---

## Running locally

```bash
pip install -e .

# Place source CSVs at:
#   data/source/temporal_data.csv
#   data/source/context_data.csv

# (Optional) Pre-warm the processed-datasets cache so the first real
# training run starts with zero data-pipeline overhead. The default
# `python api.py` does this automatically on first run.
python api.py --build-cache

# Smoke test (sample=50, 1 epoch per phase, ~60s on GPU)
python api.py --smoke > results/logs/smoke.log 2>&1
grep "^patient_auroc_weighted:\|^---" results/logs/smoke.log

# Full run (sample=None, default epoch counts)
python api.py > results/logs/run.log 2>&1
grep "^patient_auroc_weighted:\|^patient_auprc_weighted:\|^length_of_stay_mae_hours:\|^peak_vram_mb:" results/logs/run.log

# Bootstrap CIs on the trained Phase-3 checkpoint (B=2000 patient resample)
python api.py --bootstrap 2000 > results/logs/boot.log 2>&1

# Post-training diagnostics (MLM top-1/5, pool entropy, time-aux residuals)
python api.py --diagnose --sample 10000 > results/logs/diag.log 2>&1
```

`api.py` emits a summary block after the `---` separator. Per-outcome
metrics are emitted as grep-friendly TSV rows (`patient_per_outcome\t…`,
`time_head_mae_hrs\t…`).

**Tuning the recipe.** Edit `intervene_enc/config/model_config.py` and re-run
`python api.py`. The Phase-1 embedder is reused automatically when
`(embed_dim, time2vec_dim, ctx_dim)` is unchanged — set a different value to
force a Phase-1 retrain. Tokenizer-affecting changes (`USE_QA_DATA` toggle)
require clearing the cache before the next run:

```bash
rm -f checkpoints/tokenizer.pt checkpoints/scaler.pkl checkpoints/processed_datasets.pt
rm -rf checkpoints/phase1
```

---

## Running on a RunPod GPU pod

Training is GPU-bound; one full run is ~8 h on an RTX A4500. The agent
runs autonomously inside `tmux` so it survives SSH drops.

### One-time setup on a fresh pod

```bash
# SSH in (RunPod gives you HOST/PORT on the Connect page)
ssh root@<HOST> -p <PORT> -i ~/.ssh/id_ed25519

# Install tmux + Node 20 + Claude Code
apt-get update && apt-get install -y tmux
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
npm install -g @anthropic-ai/claude-code

# Clone + install Python deps (working branch holds all experiment history)
cd /workspace
git clone https://github.com/shaharoded/Transform-EMR-Encoder.git autoresearch
cd autoresearch
git checkout autoresearcher-updates
pip install -e .

# Non-root user (Claude refuses --dangerously-skip-permissions as root)
useradd -m -s /bin/bash agent
cp -r /root/.ssh /home/agent/ && chown -R agent:agent /home/agent/.ssh
chmod -R a+rwX /workspace/autoresearch
```

### SCP the data files (from local PowerShell)

```powershell
scp -P <PORT> -i ~/.ssh/id_ed25519 data\source\temporal_data.csv root@<HOST>:/workspace/autoresearch/data/source/
scp -P <PORT> -i ~/.ssh/id_ed25519 data\source\context_data.csv  root@<HOST>:/workspace/autoresearch/data/source/
```

### Start the agent (each session)

```bash
ssh root@<HOST> -p <PORT> -i ~/.ssh/id_ed25519
su - agent
cd /workspace/autoresearch
tmux new -A -s claude            # -A reattaches if the session already exists
claude --dangerously-skip-permissions
```

Initial kickoff prompt (paste once at the Claude prompt):

> Read `program.md`. We are on branch `autoresearcher-updates` of
> `Transform-EMR-Encoder`. Run the experiment loop autonomously — smoke
> test, full run, KEEP / DISCARD per the rules, update `results/status.md`
> and `results/results.tsv` after every meaningful step, and commit + push
> to `autoresearcher-updates`. Never `git reset --hard` or force-push —
> DISCARDs go through `git revert`.

Detach with `Ctrl-b d`. Reattach with the same `tmux new -A -s claude`
or `tmux attach -t claude`. Make tmux scrollable with `Ctrl-b [` (then
`q` to exit copy mode), or enable mouse wheel via `set -g mouse on` in
`~/.tmux.conf`.

### Monitoring without disturbing the agent

```bash
# Read the live journal
ssh root@<HOST> -p <PORT> -i ~/.ssh/id_ed25519 \
  "cat /workspace/autoresearch/results/status.md"

# Or pull on your laptop whenever the agent has pushed
git pull --ff-only
```

### Before stopping the pod

1. Push the working branch from inside the pod — anything uncommitted on
   container disk is lost when the pod stops.
2. SCP off `checkpoints.bak_keep_*/` if you want the trained weights (each
   ~95 MB; gitignored).
3. SCP off `results/` if you want logs / journal on Drive (gitignored).

If the pod can't push directly (no GitHub creds), bundle the branch and
pull from the bundle on your laptop:

```bash
# on the pod
git bundle create /tmp/agent.bundle autoresearcher-updates

# on your laptop
scp -P <PORT> -i ~/.ssh/id_ed25519 root@<HOST>:/tmp/agent.bundle ./
git fetch ./agent.bundle autoresearcher-updates:refs/remotes/pod/autoresearcher-updates
git merge --ff-only pod/autoresearcher-updates
git push
```

---

## Git workflow

- Working branch is **`autoresearcher-updates`**. All commits land there.
- `main` is untouched.
- DISCARDs go through `git revert` — never `git reset --hard` and never
  force-push. The failed-experiment commit + its `status.md` block stays in
  history as a record of what was tried and why.
- Tokenizer changes (vocab additions) bump the model checkpoint format; the
  Phase-1 reuse guard in `api.py` handles the safe cases.
- `results/` is gitignored — it travels via Drive, not GitHub.

The full git discipline (forbidden ops, rollback path, rationale) lives in
`program.md` under **Git discipline**.
