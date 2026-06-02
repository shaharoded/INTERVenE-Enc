"""
diagnose_run.py — standalone post-training diagnostics.

Rebuilds the validation DataLoader exactly the way api.load_data builds it
(same SEED / TEST_SPLIT / VAL_SPLIT / sample), loads the best Phase-3
checkpoint, attaches task heads, and runs transform_emr.diagnose.run_diagnostics.

We cannot `import api` (it trains at import-time), so the val pipeline is
mirrored here. Kept read-only w.r.t. checkpoints.

Usage:
    DIAG_SAMPLE=10000 python diagnose_run.py > diag.log 2>&1
    DIAG_SAMPLE=50    python diagnose_run.py            # smoke
"""
import os
import sys

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from sklearn.model_selection import train_test_split
import torch

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
from transform_emr.diagnose import run_diagnostics

DATA_DIR    = os.path.join(PROJECT_ROOT, "data", "source")
TEMPORAL    = os.path.join(DATA_DIR, "temporal_data.csv")
CONTEXT     = os.path.join(DATA_DIR, "context_data.csv")
CKPT_DIR    = os.path.join(PROJECT_ROOT, "checkpoints")
PHASE1_CKPT = os.path.join(CKPT_DIR, "phase1", "ckpt_best.pt")
PHASE3_CKPT = os.path.join(CKPT_DIR, "phase3", "ckpt_best.pt")
PHASE3_LAST = os.path.join(CKPT_DIR, "phase3", "ckpt_last.pt")
PHASE2_CKPT = os.path.join(CKPT_DIR, "phase2", "ckpt_best.pt")
TOKENIZER   = os.path.join(CKPT_DIR, "tokenizer.pt")

TEST_SPLIT = 0.15
VAL_SPLIT  = 0.15
SEED       = 42

sample = int(os.environ.get("DIAG_SAMPLE", "10000"))
mode   = TRAINING_SETTINGS.get("phase2_mlm_mask_mode", "positional")
p      = TRAINING_SETTINGS.get("phase2_mlm_ratio", 0.15)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[diag] sample={sample} mask_mode={mode} p={p} device={device}")

temporal_raw = pd.read_csv(TEMPORAL, low_memory=False)
ctx_raw      = pd.read_csv(CONTEXT)
if sample is not None:
    pids   = temporal_raw["PatientId"].unique()
    rng    = np.random.RandomState(SEED)
    chosen = rng.choice(pids, size=min(sample, len(pids)), replace=False)
    temporal_raw = temporal_raw[temporal_raw["PatientId"].isin(chosen)]
    ctx_raw      = ctx_raw[ctx_raw["PatientId"].isin(chosen)]

all_pids = temporal_raw["PatientId"].unique()
trainval_ids, test_ids = train_test_split(all_pids, test_size=TEST_SPLIT, random_state=SEED)
val_relative = VAL_SPLIT / (1.0 - TEST_SPLIT)
train_ids, val_ids = train_test_split(trainval_ids, test_size=val_relative, random_state=SEED)

val_temporal = temporal_raw[temporal_raw["PatientId"].isin(val_ids)].copy()
val_ctx      = ctx_raw[ctx_raw["PatientId"].isin(val_ids)].copy()

scaler = joblib_load(os.path.join(CKPT_DIR, "scaler.pkl"))
val_temporal_df, val_ctx_df = DataProcessor(
    val_temporal, val_ctx, scaler=scaler, tak_repo_path=TAK_REPO_PATH,
    checkpoint_path=CKPT_DIR,
).run()

tokenizer = EMRTokenizer.load(TOKENIZER)
val_ds = EMRDataset(val_temporal_df, val_ctx_df, tokenizer=tokenizer)
val_dl = get_dataloader(val_ds, batch_size=TRAINING_SETTINGS["batch_size"],
                        collate_fn=collate_emr, oversample=False, bucket_batching=True)

embedder, *_ = EMREmbedding.load(PHASE1_CKPT, tokenizer=tokenizer)
embedder.to(device)

ckpt = PHASE3_CKPT if os.path.exists(PHASE3_CKPT) else (
    PHASE3_LAST if os.path.exists(PHASE3_LAST) else PHASE2_CKPT)
print(f"[diag] loading model from {ckpt}")
model, *_ = EMREncoder.load(ckpt, embedder=embedder, map_location=device, attach_task_heads=True)
model.to(device)

run_diagnostics(model, val_dl, n_batches=4, p=p, mode=mode)
print("[diag] done.")
