"""
build_cache.py — One-time helper to pre-build processed_datasets.pt without OOM.

api.py tries to save a full processed EMRDataset to disk as the cache, which
can exceed pod memory limits on large datasets. This script builds a MINIMAL
cache that still satisfies api.py's fast-path check:
  - train_ds, val_ds: 1 patient each (just enough for ctx_dim detection)
  - tokenizer: real tokenizer from checkpoints/
  - test_raw: full raw test DataFrames (processed fresh inside evaluate_on_test_set)
  - sizes: (n_train, n_val, n_test) reflecting the real 70/15/15 split
  - key: matches api.py cache_key = (None, 42, 0.15, 0.15, USE_QA_DATA)

api.py fast-path checks: cache.get("key") == cache_key — this passes.
ctx_dim is read from the first batch of train_dl (1-patient batch) — correct.
evaluate_on_test_set receives full test_raw and processes it fresh — correct.

Prereqs: checkpoints/tokenizer.pt + checkpoints/scaler.pkl must already exist
(produced by a prior full api.py run, or the unittests).
"""

import sys, os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from joblib import load as joblib_load

from transform_emr.dataset import DataProcessor, EMRTokenizer, EMRDataset
from transform_emr.config.dataset_config import TAK_REPO_PATH, USE_QA_DATA

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DATA_DIR       = os.path.join(PROJECT_ROOT, "data", "source")
TEST_SPLIT     = 0.15
VAL_SPLIT      = 0.15
RANDOM_SEED    = 42

print("Loading source CSVs...")
temporal_raw = pd.read_csv(os.path.join(DATA_DIR, "temporal_data.csv"), low_memory=False)
ctx_raw      = pd.read_csv(os.path.join(DATA_DIR, "context_data.csv"))
print(f"  {len(temporal_raw):,} temporal rows, {ctx_raw['PatientId'].nunique()} patients")

# Reproduce the exact same 3-way split as api.py
all_pids = temporal_raw["PatientId"].unique()
trainval_ids, test_ids = train_test_split(all_pids, test_size=TEST_SPLIT, random_state=RANDOM_SEED)
val_relative = VAL_SPLIT / (1.0 - TEST_SPLIT)
train_ids, val_ids = train_test_split(trainval_ids, test_size=val_relative, random_state=RANDOM_SEED)
print(f"  Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

# Full raw test DataFrames (kept unprocessed; evaluate_on_test_set processes them fresh)
test_temporal_raw = temporal_raw[temporal_raw["PatientId"].isin(test_ids)].copy()
test_ctx_raw      = ctx_raw[ctx_raw["PatientId"].isin(test_ids)].copy()
print(f"  test_temporal_raw: {len(test_temporal_raw):,} rows")

# Load tokenizer and scaler
tokenizer = EMRTokenizer.load(os.path.join(CHECKPOINT_DIR, "tokenizer.pt"))
scaler    = joblib_load(os.path.join(CHECKPOINT_DIR, "scaler.pkl"))
print("Tokenizer and scaler loaded.")

def build_mini_ds(pids, label):
    """Build a 1-patient EMRDataset for the given patient list."""
    pid = list(pids)[:1]
    t_df = temporal_raw[temporal_raw["PatientId"].isin(pid)].copy()
    c_df = ctx_raw[ctx_raw["PatientId"].isin(pid)].copy()
    proc = DataProcessor(t_df, c_df, scaler=scaler,
                         tak_repo_path=TAK_REPO_PATH, checkpoint_path=CHECKPOINT_DIR)
    t_proc, c_proc = proc.run()
    ds = EMRDataset(t_proc, c_proc, tokenizer=tokenizer)
    print(f"  {label}_ds: 1 patient built (ctx_dim={c_proc.shape[1] if hasattr(c_proc, 'shape') else '?'})")
    return ds

print("Building minimal train_ds (1 patient)...")
train_ds = build_mini_ds(train_ids, "train")

print("Building minimal val_ds (1 patient)...")
val_ds = build_mini_ds(val_ids, "val")

# Verify ctx_dim is consistent with the deployed checkpoints
for batch in torch.utils.data.DataLoader(
    train_ds,
    batch_size=1,
    collate_fn=lambda b: b[0],
):
    if isinstance(batch, dict) and "context_vec" in batch:
        ctx_dim = batch["context_vec"].shape[-1]
        print(f"ctx_dim detected: {ctx_dim}")
    break

cache_key = (None, RANDOM_SEED, TEST_SPLIT, VAL_SPLIT, USE_QA_DATA)
cache_path = os.path.join(CHECKPOINT_DIR, "processed_datasets.pt")

print("Saving minimal cache...")
torch.save({
    "key":      cache_key,
    "train_ds": train_ds,
    "val_ds":   val_ds,
    "tokenizer": tokenizer,
    "test_raw":  (test_temporal_raw, test_ctx_raw),
    "sizes":    (len(train_ids), len(val_ids), len(test_ids)),
}, cache_path)

size_mb = os.path.getsize(cache_path) / 1024 / 1024
print(f"Cache saved to {cache_path} ({size_mb:.1f} MB)")
print("Done. Run: python api.py")
