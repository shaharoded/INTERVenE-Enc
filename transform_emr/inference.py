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

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transform_emr.dataset import EMRDataset, collate_emr
from transform_emr.transformer import EMREncoder
from transform_emr.config.dataset_config import RELEASE_TOKEN, DEATH_TOKEN


@torch.no_grad()
def predict(model: EMREncoder, dataset: EMRDataset, batch_size: int = 16,
            num_workers: int = 0, device: Optional[torch.device] = None,
            include_pad_for_release: bool = True) -> pd.DataFrame:
    """
    Purpose: Run the encoder + task heads on every patient and return a tidy
             one-row-per-patient dataframe.
    Method:  Standard batched eval — encode → pool → (risk, time).  Outputs
             both the risk-head probabilities (``P_<outcome>``) and the
             time-head predictions in hours (``T_<outcome>``).  RELEASE is
             absent from ``P_*`` columns (dropped from the risk head) but
             present in ``T_*`` as the length-of-stay regression.

    Args:
        model           (EMREncoder): Phase-3 (or later) trained model with
                                      task heads attached.
        dataset         (EMRDataset): pre-truncated input dataset.
        batch_size      (int):        loader batch size.
        num_workers     (int):        loader workers (0 = main thread).
        device          (torch.device|None): defaults to the model's device.
        include_pad_for_release (bool): kept for forward compatibility — has
                                        no effect in the current bidirectional
                                        flow (no AR padding semantics).

    Returns:
        DataFrame indexed by PatientId with columns:
          P_<risk_outcome_name> ...  — sigmoid(risk_logit) per risk outcome.
          T_<time_outcome_name> ...  — softplus(time_logit) hours per outcome.
    """
    if model.task_heads is None:
        raise RuntimeError(
            "[inference.predict] EMREncoder has no task_heads attached. "
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

    pid_rows = []
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
        pid_rows.append(len(risk_logits))

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


def get_token_embedding(embedder, token: str) -> torch.Tensor:
    """
    Purpose: Return the row of the position embedding table for a single token.
    Method:  Maps the token to its position id and indexes ``position_embed``.

    Args:
        embedder (EMREmbedding): trained embedder.
        token (str): token string (must be present in tokenizer.token2id).

    Returns:
        Tensor of shape [embed_dim] on the embedder's device.
    """
    tid = embedder.tokenizer.token2id.get(token)
    if tid is None:
        raise KeyError(f"[get_token_embedding] Token '{token}' missing from vocab.")
    with torch.no_grad():
        return embedder.position_embed.weight[tid].detach().clone()


if __name__ == "__main__":
    # Smoke-run helper: load best checkpoints and dump predictions for the
    # test split.  Mirrors the AR module's old __main__ section.
    import joblib
    from transform_emr.embedder import EMREmbedding
    from transform_emr.dataset import DataProcessor, EMRTokenizer
    from transform_emr.config.model_config import (
        PHASE1_CHECKPOINT, PHASE2_CHECKPOINT, PHASE3_CHECKPOINT, CHECKPOINT_PATH,
    )
    from transform_emr.config.dataset_config import (
        TEST_TEMPORAL_DATA_FILE, TEST_CTX_DATA_FILE, TAK_REPO_PATH,
    )

    tokenizer = EMRTokenizer.load(str(Path(CHECKPOINT_PATH) / "tokenizer.pt"))
    scaler = joblib.load(str(Path(CHECKPOINT_PATH) / "scaler.pkl"))

    df = pd.read_csv(TEST_TEMPORAL_DATA_FILE, low_memory=False)
    ctx_df = pd.read_csv(TEST_CTX_DATA_FILE)
    processor = DataProcessor(df, ctx_df, scaler=scaler, tak_repo_path=TAK_REPO_PATH,
                              max_input_days=5)
    df, ctx_df = processor.run()
    ds = EMRDataset(df, ctx_df, tokenizer=tokenizer)

    embedder, *_ = EMREmbedding.load(PHASE1_CHECKPOINT, tokenizer=tokenizer)
    p3_ckpt = Path(PHASE3_CHECKPOINT)
    p2_ckpt = Path(PHASE2_CHECKPOINT)
    ckpt = p3_ckpt if p3_ckpt.exists() else p2_ckpt
    model, *_ = EMREncoder.load(str(ckpt), embedder=embedder, attach_task_heads=True)
    model.eval()

    preds = predict(model, ds)
    print(preds.head())
