"""
ablation_deinterval.py — De-interval ablation preprocessor.

Collapses every temporal event to a 1-second duration by setting
    EndDateTime = StartDateTime + 1 second

_expand_tokens() treats events with duration <= 1s as instantaneous (single
point token), so after this transform NO event in the dataset will produce
START/END token pairs. The model is then trained and evaluated without any
interval information, only event presence + discretized timing.

Usage (run once before api.py, from the project root):
    python ablation_deinterval.py
    python api.py > results/logs/run.log 2>&1

The script overwrites data/source/temporal_data.csv in-place.
The original file is backed up to data/source/temporal_data.csv.original.
"""

import os
import shutil
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "source")
TEMPORAL_FILE = os.path.join(DATA_DIR, "temporal_data.csv")
BACKUP_FILE   = os.path.join(DATA_DIR, "temporal_data.csv.original")


def main():
    if not os.path.exists(TEMPORAL_FILE):
        raise FileNotFoundError(f"Temporal data not found at: {TEMPORAL_FILE}")

    # Backup original once
    if not os.path.exists(BACKUP_FILE):
        shutil.copy2(TEMPORAL_FILE, BACKUP_FILE)
        print(f"[ablation] Original backed up to: {BACKUP_FILE}")
    else:
        print(f"[ablation] Backup already exists, skipping: {BACKUP_FILE}")

    print(f"[ablation] Loading {TEMPORAL_FILE} ...")
    df = pd.read_csv(TEMPORAL_FILE, low_memory=False)

    required = {"StartDateTime", "EndDateTime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    n_rows = len(df)
    print(f"[ablation] Loaded {n_rows:,} rows.")

    # Parse datetimes — keep UTC so arithmetic is unambiguous
    df["StartDateTime"] = pd.to_datetime(df["StartDateTime"], utc=True, errors="raise")
    df["EndDateTime"]   = pd.to_datetime(df["EndDateTime"],   utc=True, errors="raise")

    # --- Ablation: collapse every interval to exactly 1 second ---
    # _expand_tokens uses: is_interval = (duration_sec > min_interval_duration_sec)
    # where min_interval_duration_sec defaults to 1.  Setting duration = 1.0 s
    # means the strict-greater-than check fails → single point token, no START/END.
    df["EndDateTime"] = df["StartDateTime"] + pd.Timedelta(seconds=1)

    # Sanity check
    durations = (df["EndDateTime"] - df["StartDateTime"]).dt.total_seconds()
    assert (durations == 1.0).all(), "Not all durations collapsed to 1s — check datetime parsing."
    print(f"[ablation] All {n_rows:,} events collapsed to 1-second duration.")

    # Restore ISO8601 string format that DataProcessor expects
    df["StartDateTime"] = df["StartDateTime"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    df["EndDateTime"]   = df["EndDateTime"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    df.to_csv(TEMPORAL_FILE, index=False)
    print(f"[ablation] Written back to {TEMPORAL_FILE}")
    print("[ablation] Done. Run: python api.py > results/logs/run.log 2>&1")


if __name__ == "__main__":
    main()
