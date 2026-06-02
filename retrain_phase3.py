"""
retrain_phase3.py — Phase-3-only re-train + eval, reusing cached Phase-1/Phase-2.

Motivation: api.py unconditionally clears + retrains Phase-2 every run, so a
Phase-3-only experiment (e.g. phase3_time_lambda / head_hidden / backbone_lr_factor)
wastes ~53 min retraining an identical Phase-2. This driver mirrors api.py's
data-loading, Phase-3 fine-tune, and evaluation EXACTLY, but loads Phase-1 and
Phase-2 from checkpoints/ (staged from a backup) instead of retraining them.

Correctness contract: this must reproduce api.py's headline numbers bit-for-bit
when given the same staged Phase-1/Phase-2 and the same config. Validated against
i4-tl025 (full api.py run) before being trusted for KEEP decisions.

Prerequisite: checkpoints/phase1/ckpt_best.pt and checkpoints/phase2/ckpt_best.pt
must already be in place (copy from a checkpoints.bak_keep_* backup). This script
does NOT touch Phase-1/Phase-2; it trains Phase-3 into checkpoints/phase3/ and evals.

Usage:
    python retrain_phase3.py > run.log 2>&1
Reads phase3_* settings (incl. phase3_time_lambda) from model_config like api.py.
"""
import gc
import json
import os
import sys
import time
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from sklearn.model_selection import train_test_split
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["TQDM_DISABLE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transform_emr.dataset import (
    DataProcessor, EMRTokenizer, EMRDataset, collate_emr, get_dataloader,
)
from transform_emr.config.dataset_config import TAK_REPO_PATH, USE_QA_DATA
from transform_emr.config.model_config import MODEL_CONFIG, TRAINING_SETTINGS
from transform_emr.embedder import EMREmbedding
from transform_emr.transformer import EMREncoder, finetune_transformer
from evaluation import evaluate_on_test_set

# Fixed paths — mirror api.py
DATA_DIR               = os.path.join(PROJECT_ROOT, "data", "source")
TEMPORAL_DATA_FILE     = os.path.join(DATA_DIR, "temporal_data.csv")
CONTEXT_DATA_FILE      = os.path.join(DATA_DIR, "context_data.csv")
CHECKPOINT_DIR         = os.path.join(PROJECT_ROOT, "checkpoints")
EMBEDDER_CHECKPOINT    = os.path.join(CHECKPOINT_DIR, "phase1", "ckpt_best.pt")
TRANSFORMER_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "phase2", "ckpt_best.pt")
PHASE3_CHECKPOINT      = os.path.join(CHECKPOINT_DIR, "phase3", "ckpt_best.pt")
TOKENIZER_PATH         = os.path.join(CHECKPOINT_DIR, "tokenizer.pt")

TEST_SPLIT  = 0.15
VAL_SPLIT   = 0.15
RANDOM_SEED = 42


def load_data(sample=None, batch_size=64):
    """Mirror of api.load_data (non-cached path). Returns train_dl, val_dl,
    tokenizer, (test_temporal_raw, test_ctx_raw)."""
    print("[Data]: Loading source temporal events and context data...")
    temporal_raw = pd.read_csv(TEMPORAL_DATA_FILE, low_memory=False)
    ctx_raw      = pd.read_csv(CONTEXT_DATA_FILE)

    if sample is not None:
        pids   = temporal_raw["PatientId"].unique()
        rng    = np.random.RandomState(RANDOM_SEED)
        chosen = rng.choice(pids, size=min(sample, len(pids)), replace=False)
        temporal_raw = temporal_raw[temporal_raw["PatientId"].isin(chosen)]
        ctx_raw      = ctx_raw[ctx_raw["PatientId"].isin(chosen)]

    all_pids = temporal_raw["PatientId"].unique()
    trainval_ids, test_ids = train_test_split(all_pids, test_size=TEST_SPLIT, random_state=RANDOM_SEED)
    val_relative = VAL_SPLIT / (1.0 - TEST_SPLIT)
    train_ids, val_ids = train_test_split(trainval_ids, test_size=val_relative, random_state=RANDOM_SEED)

    train_temporal_raw = temporal_raw[temporal_raw["PatientId"].isin(train_ids)].copy()
    train_ctx_raw      = ctx_raw[ctx_raw["PatientId"].isin(train_ids)].copy()
    val_temporal_raw   = temporal_raw[temporal_raw["PatientId"].isin(val_ids)].copy()
    val_ctx_raw        = ctx_raw[ctx_raw["PatientId"].isin(val_ids)].copy()
    test_temporal_raw  = temporal_raw[temporal_raw["PatientId"].isin(test_ids)].copy()
    test_ctx_raw       = ctx_raw[ctx_raw["PatientId"].isin(test_ids)].copy()

    print("[Data]: Processing train split (fitting scaler)...")
    train_processor = DataProcessor(train_temporal_raw.copy(), train_ctx_raw.copy(),
                                    scaler=None, tak_repo_path=TAK_REPO_PATH,
                                    checkpoint_path=CHECKPOINT_DIR)
    train_temporal_df, train_ctx_df = train_processor.run()

    scaler = joblib_load(os.path.join(CHECKPOINT_DIR, "scaler.pkl"))
    print("[Data]: Processing val split (applying fitted scaler)...")
    val_processor = DataProcessor(val_temporal_raw.copy(), val_ctx_raw.copy(),
                                  scaler=scaler, tak_repo_path=TAK_REPO_PATH,
                                  checkpoint_path=CHECKPOINT_DIR)
    val_temporal_df, val_ctx_df = val_processor.run()

    # Tokenizer must already exist (staged with phase1/phase2). Mirror api: load if present.
    tokenizer_path = Path(TOKENIZER_PATH)
    if tokenizer_path.exists():
        print("[Data]: Loading tokenizer from cache...")
        tokenizer = EMRTokenizer.load(str(tokenizer_path))
    else:
        print("[Data]: Building tokenizer...")
        tokenizer = EMRTokenizer.from_processed_df(train_temporal_df)
        tokenizer.save(str(tokenizer_path))

    train_ds = EMRDataset(train_temporal_df, train_ctx_df, tokenizer=tokenizer)
    val_ds   = EMRDataset(val_temporal_df,   val_ctx_df,   tokenizer=tokenizer)
    print(f"[Data]: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test patients")

    train_dl = get_dataloader(train_ds, batch_size=batch_size, collate_fn=collate_emr,
                              oversample=False, bucket_batching=True)
    val_dl   = get_dataloader(val_ds, batch_size=batch_size, collate_fn=collate_emr,
                              oversample=False, bucket_batching=True)
    return train_dl, val_dl, tokenizer, (test_temporal_raw, test_ctx_raw)


t_start = time.time()
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Sanity: phase1/phase2 must be staged.
for req in (EMBEDDER_CHECKPOINT, TRANSFORMER_CHECKPOINT):
    if not os.path.exists(req):
        raise FileNotFoundError(f"[retrain_phase3] required checkpoint missing: {req}. "
                                f"Stage Phase-1/Phase-2 from a checkpoints.bak_keep_* backup first.")

# Clear ONLY phase3 (keep phase1/phase2 — that is the whole point).
_p3dir = Path(CHECKPOINT_DIR) / "phase3"
if _p3dir.exists():
    shutil.rmtree(_p3dir)
_p3dir.mkdir(parents=True, exist_ok=True)

train_dl, val_dl, tokenizer, test_raw = load_data(
    sample=TRAINING_SETTINGS.get("sample"),
    batch_size=TRAINING_SETTINGS["batch_size"],
)

for _batch in train_dl:
    MODEL_CONFIG["ctx_dim"] = _batch["context_vec"].shape[-1]
    break
print(f"Model config: {MODEL_CONFIG}")

# Phase-1 embedder (load cached — never retrain here).
embedder, *_ = EMREmbedding.load(str(EMBEDDER_CHECKPOINT), tokenizer=tokenizer)
embedder.to(device)

# Phase-3 — load Phase-2 best, attach task heads, fine-tune (mirror api.py).
model_p3, *_ = EMREncoder.load(str(TRANSFORMER_CHECKPOINT), embedder=embedder,
                               map_location=device, attach_task_heads=True)
model_p3, _, p3_val_losses = finetune_transformer(
    model=model_p3, train_dl=train_dl, val_dl=val_dl, resume=False,
    checkpoint_path=PHASE3_CHECKPOINT, training_settings=TRAINING_SETTINGS,
)

del train_dl
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Eval — mirror api.py exactly.
_p3_path = Path(PHASE3_CHECKPOINT)
_p3_last = _p3_path.parent / "ckpt_last.pt"
if _p3_path.exists():
    best_model, *_ = EMREncoder.load(str(_p3_path), embedder=embedder, attach_task_heads=True)
elif _p3_last.exists():
    best_model, *_ = EMREncoder.load(str(_p3_last), embedder=embedder, attach_task_heads=True)
else:
    best_model = model_p3

test_temporal_raw, test_ctx_raw = test_raw
scaler = joblib_load(os.path.join(CHECKPOINT_DIR, "scaler.pkl"))
eval_results = evaluate_on_test_set(
    model=best_model, tokenizer=tokenizer,
    test_temporal_raw=test_temporal_raw, test_ctx_raw=test_ctx_raw,
    scaler=scaler, checkpoint_dir=CHECKPOINT_DIR,
    batch_size=TRAINING_SETTINGS["batch_size"],
)

t_end        = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
num_params   = best_model.get_num_params() if hasattr(best_model, "get_num_params") else sum(p.numel() for p in best_model.parameters())

print("---")
print(f"patient_auroc_weighted:    {eval_results['patient_auroc_weighted']:.6f}")
print(f"patient_auprc_weighted:    {eval_results['patient_auprc_weighted']:.6f}")
print(f"patient_auroc_simple:      {eval_results['patient_auroc_simple']:.6f}")
print(f"patient_auprc_simple:      {eval_results['patient_auprc_simple']:.6f}")
print(f"patient_max_f1_weighted:   {eval_results['patient_max_f1_weighted']:.6f}")
print(f"patient_max_f1_simple:     {eval_results['patient_max_f1_simple']:.6f}")
print(f"patient_f1_at_0_5_weighted:{eval_results['patient_f1_at_0_5_weighted']:.6f}")
print(f"patient_f1_at_0_5_simple:  {eval_results['patient_f1_at_0_5_simple']:.6f}")
print(f"n_outcomes_used:           {eval_results['n_outcomes_used']}")
print(f"length_of_stay_mae_hours:  {eval_results['length_of_stay_mae_hours']:.4f}")
print(f"length_of_stay_median_hrs: {eval_results['length_of_stay_median_hours']:.4f}")
print(f"length_of_stay_p90_hours:  {eval_results['length_of_stay_p90_hours']:.4f}")
print(f"length_of_stay_n_patients: {eval_results['length_of_stay_n_patients']}")

print("patient_per_outcome\toutcome\tauroc\tauprc\tmax_f1\tmax_f1_threshold\tf1_at_0_5\tn_pos\tn_neg\tprevalence")
_pat_tbl = eval_results.get("patient_auc_table")
if _pat_tbl is not None:
    for _outcome, _row in _pat_tbl.iterrows():
        _auroc = f"{_row['auroc']:.6f}" if not pd.isna(_row['auroc']) else "nan"
        _auprc = f"{_row['auprc']:.6f}" if not pd.isna(_row['auprc']) else "nan"
        _maxf1 = f"{_row['max_f1']:.6f}" if not pd.isna(_row.get('max_f1', np.nan)) else "nan"
        _mthr  = f"{_row['max_f1_threshold']:.6f}" if not pd.isna(_row.get('max_f1_threshold', np.nan)) else "nan"
        _f105  = f"{_row['f1_at_0_5']:.6f}" if not pd.isna(_row.get('f1_at_0_5', np.nan)) else "nan"
        print(f"patient_per_outcome\t{_outcome}\t{_auroc}\t{_auprc}\t{_maxf1}\t{_mthr}\t{_f105}\t"
              f"{int(_row['n_pos'])}\t{int(_row['n_neg'])}\t{_row['prevalence']:.6f}")

print("time_head_mae_hrs\toutcome\tmae_hours\tn_patients")
_mae_tbl = eval_results.get("time_mae_table")
if _mae_tbl is not None and len(_mae_tbl) > 0:
    for _outcome, _row in _mae_tbl.iterrows():
        _mae = f"{_row['mae_hours']:.4f}" if not pd.isna(_row['mae_hours']) else "nan"
        print(f"time_head_mae_hrs\t{_outcome}\t{_mae}\t{int(_row['n_patients'])}")

print(f"phase3_best_val:  {float(min(p3_val_losses)) if p3_val_losses else float('nan'):.6f}")
print(f"phase3_epochs:    {len(p3_val_losses)}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"embed_dim:        {MODEL_CONFIG['embed_dim']}")
print(f"phase3_time_lambda: {TRAINING_SETTINGS.get('phase3_time_lambda')}")
print(f"num_params:       {num_params:,}")
