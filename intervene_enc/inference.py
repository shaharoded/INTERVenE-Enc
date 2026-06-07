"""
inference.py
============

Inference utilities for the BERT-style EMR encoder.

The autoregressive trajectory generation that the legacy GPT used is gone —
the encoder reads the full seed in a single bidirectional pass and emits

* ``risk_pred[k]``  per-outcome probability (sigmoid of the risk-head logit),
* ``time_pred[k]``  per-outcome predicted hours from seed end (softplus),

which the evaluation notebook scores with the existing
``per_patient_max_auc`` / ``length_of_stay_mae`` framework.
"""

from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from intervene_enc.dataset import EMRDataset, collate_emr
from intervene_enc.transformer import InterveneEncoder
from intervene_enc.config.dataset_config import RELEASE_TOKEN, DEATH_TOKEN


@torch.no_grad()
def predict(model: InterveneEncoder, dataset: EMRDataset, batch_size: int = 16,
            num_workers: int = 0, device: Optional[torch.device] = None) -> pd.DataFrame:
    """
    Purpose: Run the encoder + task heads on every patient and return a tidy
             one-row-per-patient dataframe.
    Method:  Standard batched eval — encode → pool → (risk, time).  Outputs
             both the risk-head probabilities (``P_<outcome>``) and the
             time-head predictions in hours (``T_<outcome>``).  RELEASE is
             absent from ``P_*`` columns (dropped from the risk head) but
             present in ``T_*`` as the length-of-stay regression.

    Args:
        model       (InterveneEncoder): Phase-3 (or later) trained model with
                                  task heads attached.
        dataset     (EMRDataset): pre-truncated input dataset.
        batch_size  (int):        loader batch size.
        num_workers (int):        loader workers (0 = main thread).
        device      (torch.device|None): defaults to the model's device.

    Returns:
        DataFrame indexed by PatientId with columns:
          P_<risk_outcome_name> ...  — sigmoid(risk_logit) per risk outcome.
          T_<time_outcome_name> ...  — softplus(time_logit) hours per outcome.
    """
    if model.task_heads is None:
        raise RuntimeError(
            "[inference.predict] InterveneEncoder has no task_heads attached. "
            "Load a Phase-3 checkpoint or call model.attach_task_heads()."
        )

    if device is None:
        device = next(model.parameters()).device
    model.eval()

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_emr, num_workers=num_workers,
    )

    risk_idx = model.task_heads.risk_idx.cpu().tolist()
    time_idx = model.task_heads.time_idx.cpu().tolist()
    risk_names = [model.outcome_names[i] for i in risk_idx]
    time_names = [model.outcome_names[i] for i in time_idx]

    risk_chunks = []
    time_chunks = []

    # patient_ids preserves the dataset's iteration order — DataLoader with
    # shuffle=False walks the same order so a flat concatenate is safe.
    all_pids = list(dataset.patient_ids)

    for batch in tqdm(loader, desc="Inference", leave=False, dynamic_ncols=True):
        batch = {k: v.to(device) for k, v in batch.items()}
        risk_logits, time_pred, _, _ = model.predict(
            parent_raw_ids=batch["parent_raw_ids"],
            concept_ids=batch["concept_ids"],
            value_ids=batch["value_ids"],
            position_ids=batch["position_ids"],
            abs_ts=batch["abs_ts"],
            context_vec=batch["context_vec"],
        )
        risk_chunks.append(torch.sigmoid(risk_logits).cpu())
        time_chunks.append(time_pred.cpu())

    risk_mat = torch.cat(risk_chunks, dim=0).numpy()
    time_mat = torch.cat(time_chunks, dim=0).numpy()

    out = pd.DataFrame(
        {f"P_{n}": risk_mat[:, j] for j, n in enumerate(risk_names)},
        index=pd.Index(all_pids[: risk_mat.shape[0]], name="PatientId"),
    )
    for j, n in enumerate(time_names):
        out[f"T_{n}"] = time_mat[:, j]

    # Convenience: if DEATH is in the risk head, expose P_RELEASE as 1 - P(DEATH).
    if DEATH_TOKEN in risk_names and f"P_{RELEASE_TOKEN}" not in out.columns:
        out[f"P_{RELEASE_TOKEN}"] = 1.0 - out[f"P_{DEATH_TOKEN}"]

    return out


if __name__ == "__main__":
    import random
    import joblib
    from pathlib import Path
    from intervene_enc.embedder import EMREmbedding
    from intervene_enc.dataset import DataProcessor
    from intervene_enc.config.model_config import (
        CHECKPOINT_PATH, PHASE1_CHECKPOINT, PHASE2_CHECKPOINT, PHASE3_CHECKPOINT,
    )
    from intervene_enc.config.dataset_config import (
        TEST_TEMPORAL_DATA_FILE, TEST_CTX_DATA_FILE, TAK_REPO_PATH,
    )

    print("Loading dataset...")
    df = pd.read_csv(TEST_TEMPORAL_DATA_FILE, low_memory=False)
    ctx_df = pd.read_csv(TEST_CTX_DATA_FILE)

    # Subset: pick N random patients for this inference batch
    print("Getting subset...")
    patient_ids = df["PatientID"].unique()
    N = 10
    selected_ids = sorted(random.sample(list(patient_ids), N))
    df_subset  = df[df["PatientID"].isin(selected_ids)].copy()
    ctx_subset = ctx_df.loc[selected_ids].copy()

    print("Loading resources...")
    tokenizer = EMRTokenizer.load(Path(CHECKPOINT_PATH) / "tokenizer.pt")
    scaler    = joblib.load(Path(CHECKPOINT_PATH) / "scaler.pkl")

    # Truncate to the same input window used during Phase-3 alignment so the
    # encoder sees a seed comparable to training time, not the full GT trajectory.
    print("Building input dataset...")
    processor = DataProcessor(
        df_subset.copy(), ctx_subset.copy(),
        scaler=scaler, tak_repo_path=TAK_REPO_PATH, max_input_days=5,
    )
    df_input, ctx_input = processor.run()
    dataset = EMRDataset(df_input, ctx_input, tokenizer=tokenizer)

    # Prefer Phase-3 (has task heads); fall back to Phase-2 with heads attached fresh.
    print("Loading model and running inference...")
    embedder, *_ = EMREmbedding.load(PHASE1_CHECKPOINT, tokenizer=tokenizer)
    p3_ckpt = Path(PHASE3_CHECKPOINT)
    p2_ckpt = Path(PHASE2_CHECKPOINT)
    ckpt_path = p3_ckpt if p3_ckpt.exists() else p2_ckpt
    model, *_ = InterveneEncoder.load(
        str(ckpt_path), embedder=embedder, attach_task_heads=True,
    )
    model.eval()

    predictions = predict(model, dataset)

    output_path = Path(CHECKPOINT_PATH) / "inference_results.xlsx"
    predictions.to_excel(output_path, sheet_name="Predictions")
    print(f"Inference results saved to: {output_path}")
