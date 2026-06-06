"""
api.py — EMR Autoresearch immutable training/eval contract (BERT-pivot edition).

The default invocation (`python api.py`) is the immutable training+eval
pipeline. The DO NOT MODIFY rule applies to that path: the per-run summary
block printed after the "---" separator is the ground-truth result for every
ledger row. To experiment with model architecture or hyperparameters, edit
files under intervene_enc/ (and its config/ sub-package).

Four additional CLI modes wrap utility flows on top of the same data
pipeline (none of them affect a default training run):
    --smoke              quick gate: sample=50, 1 epoch per phase
    --build-cache        run load_data() and exit — pre-warms processed_datasets.pt
                          so subsequent training starts with zero data overhead
    --diagnose           skip training; run diagnose.run_diagnostics
    --bootstrap [B]      skip training; B-resample patient bootstrap CIs

Note: the default training run already builds + persists the cache the first
time it runs; --build-cache is purely a "pre-warm without spending epochs"
convenience for fresh pods or data-pipeline sanity checks.

Usage:
    python api.py
    python api.py > results/logs/run.log 2>&1
    python api.py --smoke > results/logs/smoke.log 2>&1
    python api.py --build-cache
    python api.py --diagnose > results/logs/diag.log 2>&1
    python api.py --bootstrap 2000 > results/logs/boot.log 2>&1

The agent reads program.md for context, edits intervene_enc/ files, then runs
this script to train and evaluate. The summary block (after the "---" separator)
is the ground-truth result for each run.

Optimization target (all from the held-out test set, not the val split):
    patient_auroc_weighted  — PRIMARY headline. Support-weighted mean of per-
                              outcome patient-level AUROC. Each (patient,
                              outcome) contributes one (P_<outcome>, label)
                              pair from a single bidirectional encoder pass.
    patient_auprc_weighted  — same, AUPRC.
    patient_max_f1_weighted — same, max-F1 swept over the PR curve.
    patient_f1_at_0_5_weighted — F1 at the fixed 0.5 threshold (calibration sanity).
    length_of_stay_mae_hours   — RELEASE time-head regression: |T_RELEASE_EVENT − GT|.
    time_head_mae_hrs:<outcome> — per-outcome time-head MAE to the nearest GT.
    patient_auroc_simple / patient_auprc_simple — unweighted means (sanity).

Phase 3 must run with task heads attached. If a checkpoint is loaded from
Phase 2 (no heads), api.py attaches fresh task heads before fine-tuning.
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

# Force UTF-8 stdout/stderr so Windows cp1252 doesn't choke on Δ / unicode in logs.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Suppress tqdm progress bars — keeps run.log clean (one line per epoch only).
os.environ["TQDM_DISABLE"] = "1"
# Reduce CUDA memory fragmentation (helps on larger models during backward).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from intervene_enc.dataset import (
    DataProcessor, EMRTokenizer, EMRDataset, collate_emr, get_dataloader,
)
from intervene_enc.config.dataset_config import TAK_REPO_PATH, USE_QA_DATA
from intervene_enc.config.model_config import MODEL_CONFIG, TRAINING_SETTINGS
from intervene_enc.embedder import EMREmbedding, train_embedder
from intervene_enc.transformer import InterveneEncoder, pretrain_transformer, finetune_transformer

from evaluation import evaluate_on_test_set, bootstrap_evaluate

# ===========================================================================
# CLI dispatcher  (only active when api.py is executed directly; importable
# as a module — `import api` from the smoke/diagnose paths used to do this
# explicitly, now handled by --smoke below — does not parse argv).
# ===========================================================================
_CLI = None
if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    _p.add_argument("--smoke", action="store_true",
                    help="quick gate: sample=50, 1 epoch per phase")
    _p.add_argument("--build-cache", dest="build_cache", action="store_true",
                    help="run load_data() and exit — pre-warms processed_datasets.pt")
    _p.add_argument("--diagnose", action="store_true",
                    help="skip training; run diagnose.run_diagnostics on the cached Phase-3 checkpoint")
    _p.add_argument("--bootstrap", type=int, nargs="?", const=2000, default=None, metavar="B",
                    help="skip training; B patient-resample bootstrap CIs on the cached checkpoint (default 2000)")
    _p.add_argument("--sample", type=int, default=None,
                    help="override TRAINING_SETTINGS['sample'] (for --diagnose / --bootstrap on sampled splits)")
    _CLI = _p.parse_args()
    if _CLI.smoke:
        TRAINING_SETTINGS.update({
            "sample": 50, "phase1_n_epochs": 1,
            "phase2_n_epochs": 1, "phase3_n_epochs": 1,
        })
        print(f"[smoke] settings patched: sample=50, all phases 1 epoch")

# ===========================================================================
# Fixed paths — do not modify
# ===========================================================================

DATA_DIR               = os.path.join(PROJECT_ROOT, "data", "source")
TEMPORAL_DATA_FILE     = os.path.join(DATA_DIR, "temporal_data.csv")
CONTEXT_DATA_FILE      = os.path.join(DATA_DIR, "context_data.csv")

CHECKPOINT_DIR         = os.path.join(PROJECT_ROOT, "checkpoints")
EMBEDDER_CHECKPOINT    = os.path.join(CHECKPOINT_DIR, "phase1", "ckpt_best.pt")
TRANSFORMER_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "phase2", "ckpt_best.pt")
PHASE3_CHECKPOINT      = os.path.join(CHECKPOINT_DIR, "phase3", "ckpt_best.pt")
TOKENIZER_PATH         = os.path.join(CHECKPOINT_DIR, "tokenizer.pt")
# Cache of fully-processed train/val EMRDatasets + raw test split + tokenizer.
# Built once after the first full data load; reused across every architecture
# experiment in the sweep. Invalidated by `rm -rf checkpoints/`.
PROCESSED_CACHE        = os.path.join(CHECKPOINT_DIR, "processed_datasets.pt")

TEST_SPLIT  = 0.15  # held-out, never seen until final evaluation
VAL_SPLIT   = 0.15  # used for early-stop monitoring during P2/P3
RANDOM_SEED = 42

# ===========================================================================
# Fixed API: data loading — do not modify
# ===========================================================================

def load_data(sample=None, batch_size=64):
    """
    Purpose: Load and prepare EMR data from source CSVs into DataLoaders for all three phases.
    Method: Reads source CSVs, fits scaler on the train portion via DataProcessor
            (saved to checkpoints/scaler.pkl), builds/caches tokenizer, splits patients
            into train/val/test, and keeps the raw test data for post-training evaluation.

    Args:
        sample (int or None): If set, restrict to this many randomly-sampled patients
            (useful for quick smoke-tests; use None for full training).
        batch_size (int): Batch size for all DataLoaders.

    Returns:
        train_dl (DataLoader): Natural-distribution loader for Phase-1 / Phase-2 / Phase-3.
        val_dl (DataLoader): Natural-distribution validation loader (early-stop monitor).
        tokenizer (EMRTokenizer): Fitted vocabulary.
        test_raw (tuple): (test_temporal_df_raw, test_ctx_df_raw) — unprocessed test data,
            held out until final evaluation. Passed to evaluate_on_test_set().
    """
    # ── Fast path: reload the cached processed datasets if present ──────────
    cache_path = Path(PROCESSED_CACHE)
    cache_key  = (sample, RANDOM_SEED, TEST_SPLIT, VAL_SPLIT, USE_QA_DATA)
    if sample is None and cache_path.exists():
        try:
            cached = torch.load(str(cache_path), map_location="cpu", weights_only=False)
            if cached.get("key") == cache_key:
                print(f"[Data]: Loading cached processed datasets from {cache_path.name}...")
                train_ds   = cached["train_ds"]
                val_ds     = cached["val_ds"]
                tokenizer  = cached["tokenizer"]
                test_raw   = cached["test_raw"]
                n_train, n_val, n_test = cached["sizes"]
                print(f"[Data]: cached — {n_train} train / {n_val} val / {n_test} test patients")

                train_dl = get_dataloader(train_ds, batch_size=batch_size,
                                          collate_fn=collate_emr, oversample=False, bucket_batching=True)
                val_dl   = get_dataloader(val_ds, batch_size=batch_size,
                                          collate_fn=collate_emr, oversample=False, bucket_batching=True)
                return train_dl, val_dl, tokenizer, test_raw
        except Exception as e:
            print(f"[Data]: cache load failed ({e}); rebuilding from source CSVs.")

    print("[Data]: Loading source temporal events and context data...")
    temporal_raw = pd.read_csv(TEMPORAL_DATA_FILE, low_memory=False)
    ctx_raw      = pd.read_csv(CONTEXT_DATA_FILE)

    if sample is not None:
        pids   = temporal_raw["PatientId"].unique()
        rng    = np.random.RandomState(RANDOM_SEED)
        chosen = rng.choice(pids, size=min(sample, len(pids)), replace=False)
        temporal_raw = temporal_raw[temporal_raw["PatientId"].isin(chosen)]
        ctx_raw      = ctx_raw[ctx_raw["PatientId"].isin(chosen)]

    # Three-way split by PatientId: train / val (early stop) / test (held-out for final eval).
    all_pids = temporal_raw["PatientId"].unique()
    trainval_ids, test_ids = train_test_split(
        all_pids, test_size=TEST_SPLIT, random_state=RANDOM_SEED
    )
    val_relative = VAL_SPLIT / (1.0 - TEST_SPLIT)
    train_ids, val_ids = train_test_split(
        trainval_ids, test_size=val_relative, random_state=RANDOM_SEED
    )

    train_temporal_raw = temporal_raw[temporal_raw["PatientId"].isin(train_ids)].copy()
    train_ctx_raw      = ctx_raw[ctx_raw["PatientId"].isin(train_ids)].copy()
    val_temporal_raw   = temporal_raw[temporal_raw["PatientId"].isin(val_ids)].copy()
    val_ctx_raw        = ctx_raw[ctx_raw["PatientId"].isin(val_ids)].copy()
    test_temporal_raw  = temporal_raw[temporal_raw["PatientId"].isin(test_ids)].copy()
    test_ctx_raw       = ctx_raw[ctx_raw["PatientId"].isin(test_ids)].copy()

    # Fit scaler on train patients and save to CHECKPOINT_DIR/scaler.pkl
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

    tokenizer_path = Path(TOKENIZER_PATH)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    if tokenizer_path.exists():
        print("[Data]: Loading tokenizer from cache...")
        tokenizer = EMRTokenizer.load(str(tokenizer_path))
    else:
        print("[Data]: Building tokenizer (one-time, may take a few minutes)...")
        tokenizer = EMRTokenizer.from_processed_df(train_temporal_df)
        tokenizer.save(str(tokenizer_path))
        print(f"[Data]: Tokenizer saved to {tokenizer_path}")

    train_ds = EMRDataset(train_temporal_df, train_ctx_df, tokenizer=tokenizer)
    val_ds   = EMRDataset(val_temporal_df,   val_ctx_df,   tokenizer=tokenizer)

    print(f"[Data]: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test patients  "
          f"({len(train_ds.tokens_df):,} train records, {len(val_ds.tokens_df):,} val records; "
          f"test held out, processed at eval time)")

    # Persist processed datasets so the next architecture experiment skips the
    # CSV reading + DataProcessor transforms. Sampled smoke tests stay un-cached.
    if sample is None:
        try:
            torch.save({
                "key": cache_key,
                "train_ds": train_ds,
                "val_ds": val_ds,
                "tokenizer": tokenizer,
                "test_raw": (test_temporal_raw, test_ctx_raw),
                "sizes": (len(train_ids), len(val_ids), len(test_ids)),
            }, str(cache_path))
            print(f"[Data]: cached processed datasets to {cache_path.name}")
        except Exception as e:
            print(f"[Data]: cache save failed ({e}); continuing without cache.")

    train_dl = get_dataloader(train_ds, batch_size=batch_size,
                              collate_fn=collate_emr, oversample=False, bucket_batching=True)
    val_dl   = get_dataloader(val_ds, batch_size=batch_size,
                              collate_fn=collate_emr, oversample=False, bucket_batching=True)
    return train_dl, val_dl, tokenizer, (test_temporal_raw, test_ctx_raw)


TRAIN_SUMMARY_PATH = os.path.join(CHECKPOINT_DIR, "train_summary.json")

# ===========================================================================
# --build-cache early dispatch (run load_data() once and exit; the natural
# `processed_datasets.pt` is created as a side-effect for subsequent runs).
# ===========================================================================
if _CLI is not None and _CLI.build_cache:
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _sample = _CLI.sample if _CLI.sample is not None else TRAINING_SETTINGS.get("sample")
    print(f"[build-cache] sample={_sample}")
    _train_dl, _val_dl, _tokenizer, _test_raw = load_data(
        sample=_sample, batch_size=TRAINING_SETTINGS["batch_size"],
    )
    _ntrain = len(_train_dl.dataset) if hasattr(_train_dl, "dataset") else "?"
    _nval   = len(_val_dl.dataset)   if hasattr(_val_dl, "dataset")   else "?"
    print(f"[build-cache] done — {_ntrain} train / {_nval} val patients "
          f"(cache at {PROCESSED_CACHE}).")
    sys.exit(0)

# ===========================================================================
# --diagnose / --bootstrap early dispatch (skip training entirely; load the
# cached Phase-3 checkpoint, dispatch to the requested utility, then exit).
# ===========================================================================
if _CLI is not None and (_CLI.diagnose or _CLI.bootstrap):
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _sample = _CLI.sample if _CLI.sample is not None else TRAINING_SETTINGS.get("sample")
    print(f"[cli] mode={'diagnose' if _CLI.diagnose else 'bootstrap'} "
          f"sample={_sample} device={_device}")

    _train_dl, _val_dl, _tokenizer, _test_raw = load_data(
        sample=_sample, batch_size=TRAINING_SETTINGS["batch_size"],
    )
    for _batch in _val_dl:
        MODEL_CONFIG["ctx_dim"] = _batch["context_vec"].shape[-1]
        break

    _p3_path = Path(PHASE3_CHECKPOINT)
    _p3_last = _p3_path.parent / "ckpt_last.pt"
    _ckpt = _p3_path if _p3_path.exists() else (_p3_last if _p3_last.exists() else None)
    if _ckpt is None:
        print(f"[cli][error] no Phase-3 checkpoint at {PHASE3_CHECKPOINT}; train first.")
        sys.exit(1)

    _embedder, *_ = EMREmbedding.load(EMBEDDER_CHECKPOINT, tokenizer=_tokenizer)
    _embedder.to(_device)
    _model, *_ = InterveneEncoder.load(str(_ckpt), embedder=_embedder, attach_task_heads=True)
    _model.to(_device)
    print(f"[cli] loaded model from {_ckpt}")

    if _CLI.diagnose:
        from intervene_enc.diagnose import run_diagnostics
        _p = TRAINING_SETTINGS.get("phase2_mlm_ratio", 0.15)
        run_diagnostics(_model, _val_dl, n_batches=4, p=_p)
        print("[diag] done.")
    else:  # --bootstrap
        _scaler = joblib_load(os.path.join(CHECKPOINT_DIR, "scaler.pkl"))
        _test_temporal_raw, _test_ctx_raw = _test_raw
        bootstrap_evaluate(
            model=_model, tokenizer=_tokenizer,
            test_temporal_raw=_test_temporal_raw, test_ctx_raw=_test_ctx_raw,
            scaler=_scaler, checkpoint_dir=CHECKPOINT_DIR,
            batch_size=TRAINING_SETTINGS["batch_size"],
            B=int(_CLI.bootstrap), seed=RANDOM_SEED,
        )
    sys.exit(0)

t_start = time.time()
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ===========================================================================
# Training orchestration
# ===========================================================================

# Clear Phase-2 and Phase-3 checkpoints for a fresh run.
# Phase-1 (embedder) is preserved and reused when the (embed_dim, time2vec_dim,
# ctx_dim) tuple matches the cached checkpoint — no need to re-train the
# embedder unless those change.
for _phase in ["phase2", "phase3"]:
    _phase_path = Path(CHECKPOINT_DIR) / _phase
    if _phase_path.exists():
        shutil.rmtree(_phase_path)
    _phase_path.mkdir(parents=True, exist_ok=True)
(Path(CHECKPOINT_DIR) / "phase1").mkdir(parents=True, exist_ok=True)

train_dl, val_dl, tokenizer, test_raw = load_data(
    sample=TRAINING_SETTINGS.get("sample"),
    batch_size=TRAINING_SETTINGS["batch_size"],
)

# Auto-detect context vector dimension from the first batch
for _batch in train_dl:
    MODEL_CONFIG["ctx_dim"] = _batch["context_vec"].shape[-1]
    break

print(f"Model config: {MODEL_CONFIG}")

# ---------------------------------------------------------------------------
# Phase 1 — Train embedder (token + time + context representations)
# ---------------------------------------------------------------------------

_embedder_key = (
    MODEL_CONFIG["embed_dim"],
    MODEL_CONFIG["time2vec_dim"],
    MODEL_CONFIG["ctx_dim"],
)

_cached_ckpt     = Path(EMBEDDER_CHECKPOINT)
_embedder_reused = False

if _cached_ckpt.exists():
    try:
        _ckpt_cfg   = torch.load(str(_cached_ckpt), map_location="cpu", weights_only=True)["config"]
        _cached_key = (
            _ckpt_cfg["embed_dim"],
            _ckpt_cfg["time2vec_dim"],
            _ckpt_cfg["ctx_dim"],
        )
        if _cached_key == _embedder_key:
            print("[Phase 1]: Config unchanged — loading cached embedder, skipping training.")
            embedder, *_ = EMREmbedding.load(str(_cached_ckpt), tokenizer=tokenizer)
            _embedder_reused = True
    except Exception as e:
        print(f"[Phase 1]: Could not load cached embedder ({e}), retraining.")

if not _embedder_reused:
    embedder = EMREmbedding(
        tokenizer    = tokenizer,
        ctx_dim      = MODEL_CONFIG["ctx_dim"],
        time2vec_dim = MODEL_CONFIG["time2vec_dim"],
        embed_dim    = MODEL_CONFIG["embed_dim"],
        dropout      = MODEL_CONFIG["dropout"],
    )
    embedder, _, _ = train_embedder(
        embedder          = embedder,
        train_loader      = train_dl,
        val_loader        = val_dl,
        resume            = False,
        checkpoint_path   = EMBEDDER_CHECKPOINT,
        training_settings = TRAINING_SETTINGS,
    )

# ---------------------------------------------------------------------------
# Phase 2 — Bidirectional MLM pre-training over the learned embeddings
# ---------------------------------------------------------------------------
encoder = InterveneEncoder(cfg=MODEL_CONFIG, embedder=embedder)
encoder, _, val_losses = pretrain_transformer(
    model             = encoder,
    train_dl          = train_dl,
    val_dl            = val_dl,
    resume            = False,
    checkpoint_path   = TRANSFORMER_CHECKPOINT,
    training_settings = TRAINING_SETTINGS,
)

# ---------------------------------------------------------------------------
# Phase 3 — Outcome + time fine-tune (task heads attached on top of P2 best)
# ---------------------------------------------------------------------------
_p2_best = Path(TRANSFORMER_CHECKPOINT)
_p2_last = _p2_best.parent / "ckpt_last.pt"
_p2_ckpt = _p2_best if _p2_best.exists() else (_p2_last if _p2_last.exists() else None)

if _p2_ckpt is not None:
    # attach_task_heads=True rebuilds the TaskHeads module on top of the Phase-2
    # encoder even though P2 didn't train heads.
    model_p3, *_ = InterveneEncoder.load(str(_p2_ckpt), embedder=embedder, attach_task_heads=True)
else:
    model_p3 = encoder
    if model_p3.task_heads is None:
        model_p3.attach_task_heads(
            hidden  = TRAINING_SETTINGS.get("phase3_head_hidden", 256),
        )

model_p3, _, p3_val_losses = finetune_transformer(
    model             = model_p3,
    train_dl          = train_dl,
    val_dl            = val_dl,
    resume            = False,
    checkpoint_path   = PHASE3_CHECKPOINT,
    training_settings = TRAINING_SETTINGS,
)

train_summary = {
    "phase2_best_val":  float(min(val_losses)) if val_losses else float("nan"),
    "phase2_epochs":    int(len(val_losses)),
    "phase3_best_val":  float(min(p3_val_losses)) if p3_val_losses else float("nan"),
    "phase3_epochs":    int(len(p3_val_losses)),
    "training_seconds": float(time.time() - t_start),
    "embed_dim":        MODEL_CONFIG["embed_dim"],
    "n_layer":          MODEL_CONFIG["n_layer"],
    "n_head":           MODEL_CONFIG["n_head"],
}
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
with open(TRAIN_SUMMARY_PATH, "w") as f:
    json.dump(train_summary, f, indent=2)
print(f"[Train] Done. Summary persisted to {TRAIN_SUMMARY_PATH}")

# Release training DataLoaders before evaluation so memory is clean.
del train_dl
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Final evaluation on held-out test set
# ---------------------------------------------------------------------------

_p3_path = Path(PHASE3_CHECKPOINT)
_p3_last = _p3_path.parent / "ckpt_last.pt"
_p2_path = _p2_best
_p2_last_path = _p2_last

if _p3_path.exists():
    best_model, *_ = InterveneEncoder.load(str(_p3_path), embedder=embedder, attach_task_heads=True)
elif _p3_last.exists():
    best_model, *_ = InterveneEncoder.load(str(_p3_last), embedder=embedder, attach_task_heads=True)
elif _p2_path.exists():
    best_model, *_ = InterveneEncoder.load(str(_p2_path), embedder=embedder, attach_task_heads=True)
elif _p2_last_path.exists():
    best_model, *_ = InterveneEncoder.load(str(_p2_last_path), embedder=embedder, attach_task_heads=True)
else:
    best_model = model_p3

test_temporal_raw, test_ctx_raw = test_raw
scaler = joblib_load(os.path.join(CHECKPOINT_DIR, "scaler.pkl"))
eval_results = evaluate_on_test_set(
    model=best_model,
    tokenizer=tokenizer,
    test_temporal_raw=test_temporal_raw,
    test_ctx_raw=test_ctx_raw,
    scaler=scaler,
    checkpoint_dir=CHECKPOINT_DIR,
    batch_size=TRAINING_SETTINGS["batch_size"],
)

# ===========================================================================
# Summary  (grep-friendly format — one key per line)
# ===========================================================================

t_end         = time.time()
peak_vram_mb  = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
num_params    = best_model.get_num_params() if hasattr(best_model, "get_num_params") else sum(p.numel() for p in best_model.parameters())

phase2_best   = train_summary.get("phase2_best_val", float("nan"))
phase2_epochs = train_summary.get("phase2_epochs",   0)
phase3_best   = train_summary.get("phase3_best_val", float("nan"))
phase3_epochs = train_summary.get("phase3_epochs",   0)

print("---")
# === HEADLINE — patient-level AUC / F1 ==================================
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
# Constant-predictor baseline (MAE of "always predict GT median LoS") + lift.
# A useful model must do better than `length_of_stay_baseline_mae_hours`; lift
# ≤ 0 means the time head has not learned LoS signal beyond a constant.
print(f"length_of_stay_baseline_mae_hours: {eval_results.get('length_of_stay_baseline_mae_hours', float('nan')):.4f}")
print(f"length_of_stay_gt_median_hours:    {eval_results.get('length_of_stay_gt_median_hours',    float('nan')):.4f}")
print(f"length_of_stay_lift_hours:         {eval_results.get('length_of_stay_lift_hours',         float('nan')):.4f}")

# Per-outcome patient AUC + F1 — grep-friendly TSV.
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

# Per-outcome time-head MAE (positives only, distance to nearest GT occurrence).
# Model MAE: all positives. Baseline: MAE-optimal constant predictor (GT
# median) over the FORECASTING cohort only — events that occur AFTER the
# encoder's input window. lift = baseline − model (positive ⇒ model beats
# the forecast-only constant predictor; ≤ 0 ⇒ no useful time signal).
print("time_head_mae_hrs\toutcome\tmae_hours\tbaseline_mae_hours\tlift_hours\tgt_median_hours\tn_patients\tn_forecasting")
_mae_tbl = eval_results.get("time_mae_table")
if _mae_tbl is not None and len(_mae_tbl) > 0:
    for _outcome, _row in _mae_tbl.iterrows():
        _mae   = f"{_row['mae_hours']:.4f}"          if not pd.isna(_row['mae_hours'])          else "nan"
        _base  = f"{_row['baseline_mae_hours']:.4f}" if not pd.isna(_row['baseline_mae_hours']) else "nan"
        _lift  = f"{_row['lift_hours']:.4f}"         if not pd.isna(_row['lift_hours'])         else "nan"
        _gtmed = f"{_row['gt_median_hours']:.4f}"    if not pd.isna(_row['gt_median_hours'])    else "nan"
        _nfc   = int(_row['n_forecasting']) if 'n_forecasting' in _row.index and not pd.isna(_row.get('n_forecasting', np.nan)) else 0
        print(f"time_head_mae_hrs\t{_outcome}\t{_mae}\t{_base}\t{_lift}\t{_gtmed}\t{int(_row['n_patients'])}\t{_nfc}")

print(f"phase2_best_val:  {phase2_best:.6f}")
print(f"phase2_epochs:    {phase2_epochs}")
print(f"phase3_best_val:  {phase3_best:.6f}")
print(f"phase3_epochs:    {phase3_epochs}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"embed_dim:        {MODEL_CONFIG['embed_dim']}")
print(f"n_layer:          {MODEL_CONFIG['n_layer']}")
print(f"n_head:           {MODEL_CONFIG['n_head']}")
print(f"num_params:       {num_params:,}")
