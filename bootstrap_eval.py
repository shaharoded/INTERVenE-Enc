"""
bootstrap_eval.py — variance via patient-level bootstrap CIs (replaces 3-seed study).

Loads the FINAL trained model from checkpoints/, runs the locked
evaluate_on_test_set ONCE to get per-patient predictions + ground-truth, then
bootstraps over the held-out TEST PATIENTS (resample with replacement, B reps) to
produce 95% percentile CIs for the support-weighted AUROC / AUPRC headline and
per-outcome AUROC/AUPRC. Single model, single inference pass — far cheaper than
re-seeding the full pipeline 3×, and a more direct estimate of sampling variance
on the fixed test set.

The bootstrap statistic mirrors evaluation.weighted_mean_auc exactly: per-outcome
AUROC/AUPRC weighted by n_pos within each resample; outcomes that fall below the
min-positive gate in a resample are dropped from that resample's weighted mean
(same as the headline).

Usage:
    python bootstrap_eval.py [B]            # B bootstrap reps (default 2000)
Reads TRAINING_SETTINGS["sample"] like api.py; uses the processed_datasets cache
(full-data) when present, else rebuilds the split.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import torch

os.environ.setdefault("TQDM_DISABLE", "1")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transform_emr.dataset import (
    DataProcessor, EMRTokenizer, EMRDataset, collate_emr, get_dataloader,
)
from transform_emr.config.dataset_config import TAK_REPO_PATH
from transform_emr.config.model_config import MODEL_CONFIG, TRAINING_SETTINGS
from transform_emr.embedder import EMREmbedding
from transform_emr.transformer import EMREncoder
from evaluation import evaluate_on_test_set, AUC_EXCLUDE, _min_positives

CKPT_DIR    = os.environ.get("BOOT_CKPT_DIR", os.path.join(PROJECT_ROOT, "checkpoints"))
PHASE1_CKPT = os.path.join(CKPT_DIR, "phase1", "ckpt_best.pt")
PHASE3_CKPT = os.path.join(CKPT_DIR, "phase3", "ckpt_best.pt")
PHASE3_LAST = os.path.join(CKPT_DIR, "phase3", "ckpt_last.pt")
TOKENIZER   = os.path.join(CKPT_DIR, "tokenizer.pt")
PROCESSED   = os.path.join(CKPT_DIR, "processed_datasets.pt")
DATA_DIR    = os.path.join(PROJECT_ROOT, "data", "source")
TEST_SPLIT, VAL_SPLIT, SEED = 0.15, 0.15, 42

B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample = TRAINING_SETTINGS.get("sample")
print(f"[boot] B={B} sample={sample} ckpt_dir={CKPT_DIR} device={device}")


def get_test_raw_and_tokenizer():
    """Prefer the processed_datasets cache (full-data) for test_raw + tokenizer."""
    cache = Path(PROCESSED)
    if sample is None and cache.exists():
        cached = torch.load(str(cache), map_location="cpu", weights_only=False)
        print(f"[boot] using cached test_raw + tokenizer from {cache.name}")
        return cached["test_raw"], cached["tokenizer"]
    # Rebuild the split (sampled case).
    temporal_raw = pd.read_csv(os.path.join(DATA_DIR, "temporal_data.csv"), low_memory=False)
    ctx_raw      = pd.read_csv(os.path.join(DATA_DIR, "context_data.csv"))
    if sample is not None:
        pids = temporal_raw["PatientId"].unique()
        rng = np.random.RandomState(SEED)
        chosen = rng.choice(pids, size=min(sample, len(pids)), replace=False)
        temporal_raw = temporal_raw[temporal_raw["PatientId"].isin(chosen)]
        ctx_raw = ctx_raw[ctx_raw["PatientId"].isin(chosen)]
    all_pids = temporal_raw["PatientId"].unique()
    trainval_ids, test_ids = train_test_split(all_pids, test_size=TEST_SPLIT, random_state=SEED)
    test_temporal = temporal_raw[temporal_raw["PatientId"].isin(test_ids)].copy()
    test_ctx      = ctx_raw[ctx_raw["PatientId"].isin(test_ids)].copy()
    tokenizer = EMRTokenizer.load(TOKENIZER)
    return (test_temporal, test_ctx), tokenizer


test_raw, tokenizer = get_test_raw_and_tokenizer()
embedder, *_ = EMREmbedding.load(PHASE1_CKPT, tokenizer=tokenizer)
embedder.to(device)
ckpt = PHASE3_CKPT if os.path.exists(PHASE3_CKPT) else PHASE3_LAST
print(f"[boot] loading model {ckpt}")
model, *_ = EMREncoder.load(ckpt, embedder=embedder, map_location=device, attach_task_heads=True)
model.to(device)

scaler = joblib_load(os.path.join(CKPT_DIR, "scaler.pkl"))
test_temporal_raw, test_ctx_raw = test_raw
res = evaluate_on_test_set(
    model=model, tokenizer=tokenizer,
    test_temporal_raw=test_temporal_raw, test_ctx_raw=test_ctx_raw,
    scaler=scaler, checkpoint_dir=CKPT_DIR, batch_size=TRAINING_SETTINGS["batch_size"],
)
predictions = res["predictions"]
gt_episodes = res["gt_episodes"]
outcome_names = [n for n in model.outcome_names if n not in AUC_EXCLUDE]
N = predictions.shape[0]
print(f"[boot] point estimate: AUPRC_w={res['patient_auprc_weighted']:.4f} "
      f"AUROC_w={res['patient_auroc_weighted']:.4f} | N_test={N}")

# Precompute per-outcome (scores, labels) aligned to a fixed patient order.
pids = list(predictions.index)
cols = {}
for name in outcome_names:
    pcol = f"P_{name}"
    if pcol not in predictions.columns:
        continue
    scores = predictions[pcol].to_numpy()
    labels = np.array([int(len(gt_episodes.get(p, {}).get(name, [])) > 0) for p in pids], dtype=np.int64)
    cols[name] = (scores, labels)

min_pos = _min_positives(N)  # same gate as headline (uses fixed N)


def weighted_stat(idx):
    """Support-weighted AUROC/AUPRC over a resampled patient index array."""
    aurocs, auprcs, weights = [], [], []
    for name, (scores, labels) in cols.items():
        s, l = scores[idx], labels[idx]
        n_pos = int(l.sum()); n_neg = len(l) - n_pos
        if n_pos < min_pos or n_neg < min_pos:
            continue  # dropped from weighted mean (matches headline)
        aurocs.append(roc_auc_score(l, s))
        auprcs.append(average_precision_score(l, s))
        weights.append(n_pos)
    if not weights:
        return np.nan, np.nan
    w = np.array(weights, float); w /= w.sum()
    return float((np.array(aurocs) * w).sum()), float((np.array(auprcs) * w).sum())


# Per-outcome bootstrap accumulators too.
per_out = {name: {"auroc": [], "auprc": []} for name in cols}
boot_auroc, boot_auprc = [], []
rng = np.random.RandomState(SEED)
t0 = time.time()
for b in range(B):
    idx = rng.randint(0, N, size=N)
    a, p = weighted_stat(idx)
    if not (np.isnan(a) or np.isnan(p)):
        boot_auroc.append(a); boot_auprc.append(p)
    for name, (scores, labels) in cols.items():
        s, l = scores[idx], labels[idx]
        n_pos = int(l.sum()); n_neg = len(l) - n_pos
        if n_pos < min_pos or n_neg < min_pos:
            continue
        per_out[name]["auroc"].append(roc_auc_score(l, s))
        per_out[name]["auprc"].append(average_precision_score(l, s))
print(f"[boot] {B} resamples in {time.time()-t0:.1f}s")


def ci(arr):
    a = np.asarray(arr)
    return np.percentile(a, 2.5), np.percentile(a, 97.5), a.mean(), a.std()


print("\n=== BOOTSTRAP 95% CI (patient resample, B=%d) ===" % B)
for label, point, arr in [
    ("patient_auprc_weighted", res["patient_auprc_weighted"], boot_auprc),
    ("patient_auroc_weighted", res["patient_auroc_weighted"], boot_auroc),
]:
    lo, hi, mean, sd = ci(arr)
    print(f"{label}: point={point:.4f}  boot_mean={mean:.4f}  95%CI=[{lo:.4f}, {hi:.4f}]  sd={sd:.4f}")

print("\n--- per-outcome 95% CI ---")
print(f"{'outcome':<34}{'AUROC [95% CI]':<30}{'AUPRC [95% CI]'}")
for name in cols:
    ar, pr = per_out[name]["auroc"], per_out[name]["auprc"]
    if not ar:
        print(f"{name:<34}(insufficient positives in resamples)"); continue
    alo, ahi, am, _ = ci(ar); plo, phi, pm, _ = ci(pr)
    print(f"{name:<34}{am:.3f} [{alo:.3f},{ahi:.3f}]      {pm:.3f} [{plo:.3f},{phi:.3f}]")
print("\n[boot] done.")
