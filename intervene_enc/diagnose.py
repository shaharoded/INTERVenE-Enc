"""
diagnose.py
===========

Post-training health checks for the BERT-pivot ``InterveneEncoder``.

All probes consume a trained model + a validation DataLoader and print a short
report; nothing here mutates the model.  Helpful when investigating why a
sweep underperformed, or to spot-check that a freshly trained model actually
learnt non-trivial structure before kicking off long evaluation runs.

The standard set is:

* :func:`probe_mlm_accuracy`             — top-1 / top-5 accuracy on masked
                                           positions; sanity-checks that the
                                           MLM head learns something beyond
                                           majority-class prediction.
* :func:`probe_time_aux_residuals`       — distribution of ``t_pos`` and
                                           ``t_local`` residuals on validation
                                           batches.  Spots aux-loss collapse
                                           (predicting the mean).
* :func:`probe_pool_attention`           — average attention weight per
                                           outcome query, plus entropy
                                           statistics.  Confirms the pool is
                                           not collapsing onto a single
                                           position.
* :func:`probe_outcome_logit_distribution` — risk-head logit and probability
                                             histograms per outcome.  Flags
                                             saturated / collapsed heads.
* :func:`probe_legality_starvation`      — fraction of masked positions where
                                           the GT token sits in the model's
                                           top-K.  Diagnoses whether the
                                           encoder even considers the right
                                           class.
* :func:`probe_time_head_predictions`    — Phase-3 sanity: per-outcome
                                           ``time_head`` prediction
                                           distribution + MAE vs the
                                           constant-GT-median baseline.
                                           Flags collapsed / trivial
                                           time predictions.
* :func:`run_diagnostics`                — convenience wrapper that runs all
                                           of the above and prints a summary.

All probes are CPU-safe, fall back gracefully when the model lacks task heads,
and honour ``torch.no_grad``.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from intervene_enc.transformer import InterveneEncoder
from intervene_enc.utils import apply_mlm_mask, build_luts


# ───────── helpers ────────────────────────────────────────────────────── #
def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _take_n(loader: Iterable[dict], n_batches: int) -> list[dict]:
    out = []
    for i, b in enumerate(loader):
        if i >= n_batches:
            break
        out.append(b)
    return out


def _percentiles(values: np.ndarray, qs=(5, 25, 50, 75, 95)) -> str:
    if values.size == 0:
        return "(empty)"
    return ", ".join(f"p{q}={np.percentile(values, q):.4f}" for q in qs)


# ───────── MLM accuracy ──────────────────────────────────────────────── #
@torch.no_grad()
def probe_mlm_accuracy(model: InterveneEncoder, loader, n_batches: int = 2,
                       p: float = 0.15,
                       top_k: int = 5) -> dict:
    """
    Purpose: Top-1 and top-K accuracy of the MLM head on freshly-masked
             validation batches.
    Method:  Re-apply :func:`apply_mlm_mask` to each batch, forward, then
             compare ``argmax`` (and ``topk``) of the lm_logits at masked
             positions against the original token ids.

    Returns:
        dict with ``top1``, ``top{top_k}``, ``n_masked``, ``majority_top1``
        (frequency of the most common GT token among masked positions — a
        trivial-predictor lower bound).
    """
    model.eval()
    device = next(model.parameters()).device
    luts = build_luts(model.embedder.tokenizer)
    luts = {k: v.to(device) if torch.is_tensor(v) else v for k, v in luts.items()}

    correct_top1 = correct_topk = total = 0
    token_counter = {}

    for batch in _take_n(loader, n_batches):
        batch = _to_device(batch, device)
        batch, target_ids, mlm_mask = apply_mlm_mask(
            batch=batch, tokenizer=model.embedder.tokenizer,
            forbid_ids=luts["forbid_mask_ids"], luts=luts, p=p,
        )
        if not mlm_mask.any():
            continue
        lm_logits, *_ = model(
            parent_raw_ids=batch["parent_raw_ids"], concept_ids=batch["concept_ids"],
            value_ids=batch["value_ids"], position_ids=batch["position_ids"],
            abs_ts=batch["abs_ts"], context_vec=batch["context_vec"],
        )
        lm_logits = lm_logits.float()
        flat_logits = lm_logits[mlm_mask]            # [N, V]
        flat_target = target_ids[mlm_mask]            # [N]

        top1 = flat_logits.argmax(dim=-1)
        correct_top1 += (top1 == flat_target).sum().item()

        k = min(top_k, flat_logits.size(-1))
        topk = flat_logits.topk(k=k, dim=-1).indices
        correct_topk += (topk == flat_target.unsqueeze(-1)).any(dim=-1).sum().item()

        total += flat_target.numel()
        for t in flat_target.cpu().tolist():
            token_counter[t] = token_counter.get(t, 0) + 1

    if total == 0:
        print("[probe_mlm_accuracy] No masked positions in the sampled batches.")
        return {"top1": float("nan"), f"top{top_k}": float("nan"),
                "n_masked": 0, "majority_top1": float("nan")}

    majority = max(token_counter.values()) / total
    out = {
        "top1":           correct_top1 / total,
        f"top{top_k}":    correct_topk / total,
        "n_masked":       total,
        "majority_top1":  majority,
    }
    print(f"[probe_mlm_accuracy] n={total}  top1={out['top1']:.4f}  "
          f"top{top_k}={out[f'top{top_k}']:.4f}  majority_top1={majority:.4f}")
    if out["top1"] <= majority + 1e-3:
        print("  WARN: MLM head not beating majority-class baseline.")
    return out


# ───────── time aux residuals ─────────────────────────────────────────── #
@torch.no_grad()
def probe_time_aux_residuals(model: InterveneEncoder, loader, n_batches: int = 2,
                             p: float = 0.15) -> dict:
    """
    Purpose: Distribution of t_pos and t_local residuals on val batches.
    Method:  Re-mask each batch (same as the train loop), forward, and
             collect (pred - target) for both auxiliaries.  A near-zero
             standard deviation on either is a strong signal of collapse to
             the mean — print the percentile spread + std for both.

    Returns:
        dict with per-aux ``mean``, ``std`` and percentile string.
    """
    from intervene_enc.utils import time_to_neighbour_targets

    model.eval()
    device = next(model.parameters()).device
    luts = build_luts(model.embedder.tokenizer)
    luts = {k: v.to(device) if torch.is_tensor(v) else v for k, v in luts.items()}

    tpos_res, tloc_res = [], []
    for batch in _take_n(loader, n_batches):
        batch = _to_device(batch, device)
        batch, _, mlm_mask = apply_mlm_mask(
            batch=batch, tokenizer=model.embedder.tokenizer,
            forbid_ids=luts["forbid_mask_ids"], luts=luts, p=p,
        )
        _, t_pos_pred, t_local_pred, pad_mask = model(
            parent_raw_ids=batch["parent_raw_ids"], concept_ids=batch["concept_ids"],
            value_ids=batch["value_ids"], position_ids=batch["position_ids"],
            abs_ts=batch["abs_ts"], context_vec=batch["context_vec"],
        )
        t_pos_pred = t_pos_pred.float()
        t_local_pred = t_local_pred.float()

        # t_pos at every non-pad position.
        valid_pos = pad_mask
        if valid_pos.any():
            tpos_res.append(
                (t_pos_pred[valid_pos] - batch["abs_ts"][valid_pos]).cpu().numpy()
            )

        # t_local only at masked positions with valid neighbours.
        t_target, t_valid = time_to_neighbour_targets(
            batch["abs_ts"], pad_mask, mlm_mask, max_hours=24.0,
        )
        if t_valid.any():
            tloc_res.append((t_local_pred[t_valid] - t_target[t_valid]).cpu().numpy())

    tpos_arr = np.concatenate(tpos_res) if tpos_res else np.array([])
    tloc_arr = np.concatenate(tloc_res) if tloc_res else np.array([])

    out = {
        "t_pos_mean":   float(tpos_arr.mean()) if tpos_arr.size else float("nan"),
        "t_pos_std":    float(tpos_arr.std())  if tpos_arr.size else float("nan"),
        "t_local_mean": float(tloc_arr.mean()) if tloc_arr.size else float("nan"),
        "t_local_std":  float(tloc_arr.std())  if tloc_arr.size else float("nan"),
    }
    print(f"[probe_time_aux_residuals] t_pos: n={tpos_arr.size}  "
          f"mean={out['t_pos_mean']:.4f}  std={out['t_pos_std']:.4f}  "
          f"{_percentiles(tpos_arr)}")
    print(f"[probe_time_aux_residuals] t_local: n={tloc_arr.size}  "
          f"mean={out['t_local_mean']:.4f}  std={out['t_local_std']:.4f}  "
          f"{_percentiles(tloc_arr)}")
    if tpos_arr.size and out["t_pos_std"] < 1e-3:
        print("  WARN: t_pos predictions are nearly constant — likely collapsed.")
    if tloc_arr.size and out["t_local_std"] < 1e-3:
        print("  WARN: t_local predictions are nearly constant — likely collapsed.")
    return out


# ───────── pool attention diagnostics ─────────────────────────────────── #
@torch.no_grad()
def probe_pool_attention(model: InterveneEncoder, loader, n_batches: int = 2) -> dict:
    """
    Purpose: Inspect the per-outcome attention pool.
    Method:  Re-run the pool with ``need_weights=True`` on a few batches and
             report per-outcome attention entropy (mean across batches).
             Low entropy = pool collapsed onto a single position; high
             entropy = pool ignoring position information.

    Returns:
        dict with ``per_outcome_entropy`` (list aligned to outcome_names).
    """
    if model.task_heads is None:
        print("[probe_pool_attention] No task_heads attached — skipping.")
        return {}

    model.eval()
    device = next(model.parameters()).device
    pool = model.task_heads.pool

    entropies = []
    for batch in _take_n(loader, n_batches):
        batch = _to_device(batch, device)
        hidden, pad_mask = model.encode(
            parent_raw_ids=batch["parent_raw_ids"], concept_ids=batch["concept_ids"],
            value_ids=batch["value_ids"], position_ids=batch["position_ids"],
            abs_ts=batch["abs_ts"], context_vec=batch["context_vec"],
        )
        B = hidden.size(0)
        K = pool.queries.size(0)
        Q = pool.queries.unsqueeze(0).expand(B, K, -1)
        kpm = ~pad_mask                                            # True at PAD
        _, attn_w = pool.attn(Q, hidden, hidden, key_padding_mask=kpm,
                              need_weights=True, average_attn_weights=True)
        # attn_w shape: [B, K, T] — distribution over key positions per query.
        attn_w = attn_w.float().clamp_min(1e-12)
        ent = -(attn_w * attn_w.log()).sum(dim=-1)                 # [B, K]
        entropies.append(ent.cpu().numpy())

    if not entropies:
        print("[probe_pool_attention] No batches collected.")
        return {}
    ent_arr = np.concatenate(entropies, axis=0)                    # [N, K]
    per_out = ent_arr.mean(axis=0)
    out = {"per_outcome_entropy": per_out.tolist(),
           "outcome_names": model.outcome_names}
    print("[probe_pool_attention] mean attention entropy per outcome:")
    for n, e in zip(model.outcome_names, per_out):
        print(f"  {n:<40s} entropy = {e:.3f}")
    return out


# ───────── outcome logit distribution ─────────────────────────────────── #
@torch.no_grad()
def probe_outcome_logit_distribution(model: InterveneEncoder, loader,
                                     n_batches: int = 2) -> dict:
    """
    Purpose: Risk-head logit distribution per outcome.
    Method:  Forward the task heads, collect ``risk_logits`` per outcome,
             report mean / std / percentile spread.  Saturated heads (all
             logits ≈ same value) or wildly bimodal heads stand out here.
    """
    if model.task_heads is None:
        print("[probe_outcome_logit_distribution] No task_heads attached — skipping.")
        return {}
    model.eval()
    device = next(model.parameters()).device

    logits_per_out: list[np.ndarray] = []
    for batch in _take_n(loader, n_batches):
        batch = _to_device(batch, device)
        risk_logits, _, _, _ = model.predict(
            parent_raw_ids=batch["parent_raw_ids"], concept_ids=batch["concept_ids"],
            value_ids=batch["value_ids"], position_ids=batch["position_ids"],
            abs_ts=batch["abs_ts"], context_vec=batch["context_vec"],
        )
        logits_per_out.append(risk_logits.float().cpu().numpy())

    if not logits_per_out:
        print("[probe_outcome_logit_distribution] No batches collected.")
        return {}
    stacked = np.concatenate(logits_per_out, axis=0)               # [N, K_risk]
    risk_idx = model.task_heads.risk_idx.cpu().tolist()
    names = [model.outcome_names[i] for i in risk_idx]
    out = {"means": stacked.mean(axis=0).tolist(),
           "stds":  stacked.std(axis=0).tolist(),
           "outcome_names": names}
    print("[probe_outcome_logit_distribution] per outcome risk logit stats:")
    for n, m, s, col in zip(names, out["means"], out["stds"], stacked.T):
        print(f"  {n:<40s} mean={m:+.3f}  std={s:.3f}  {_percentiles(col)}")
    return out


# ───────── legality starvation ────────────────────────────────────────── #
@torch.no_grad()
def probe_legality_starvation(model: InterveneEncoder, loader, n_batches: int = 2,
                              p: float = 0.15,
                              ks=(1, 5, 20)) -> dict:
    """
    Purpose: Does the MLM head ever rank the GT token in the top-K?
    Method:  Re-mask validation batches, take the top-K predictions per
             masked position and report the rate at which the GT id is
             present.  A failing model will have ~0% at K=1 but a healthy
             distribution at K=20+; a confused model fails at all K.
    """
    model.eval()
    device = next(model.parameters()).device
    luts = build_luts(model.embedder.tokenizer)
    luts = {k: v.to(device) if torch.is_tensor(v) else v for k, v in luts.items()}

    hits = {k: 0 for k in ks}
    total = 0
    for batch in _take_n(loader, n_batches):
        batch = _to_device(batch, device)
        batch, target_ids, mlm_mask = apply_mlm_mask(
            batch=batch, tokenizer=model.embedder.tokenizer,
            forbid_ids=luts["forbid_mask_ids"], luts=luts, p=p,
        )
        if not mlm_mask.any():
            continue
        lm_logits, *_ = model(
            parent_raw_ids=batch["parent_raw_ids"], concept_ids=batch["concept_ids"],
            value_ids=batch["value_ids"], position_ids=batch["position_ids"],
            abs_ts=batch["abs_ts"], context_vec=batch["context_vec"],
        )
        flat_logits = lm_logits[mlm_mask].float()
        flat_target = target_ids[mlm_mask]
        for k in ks:
            kk = min(k, flat_logits.size(-1))
            topk = flat_logits.topk(k=kk, dim=-1).indices
            hits[k] += (topk == flat_target.unsqueeze(-1)).any(dim=-1).sum().item()
        total += flat_target.numel()

    if total == 0:
        return {f"top{k}": float("nan") for k in ks}
    out = {f"top{k}": hits[k] / total for k in ks}
    pretty = "  ".join(f"top{k}={out[f'top{k}']:.4f}" for k in ks)
    print(f"[probe_legality_starvation] n={total}  {pretty}")
    return out


# ───────── Phase-3 time-head check ────────────────────────────────────── #
@torch.no_grad()
def probe_time_head_predictions(model: InterveneEncoder, loader,
                                n_batches: int = 4,
                                std_warn_hours: float = 0.5) -> dict:
    """
    Purpose: Verify that the Phase-3 ``time_head`` produces non-trivial
             per-outcome time predictions — i.e., not a constant, and at least
             as good as the constant-GT-median baseline used by ``evaluation.py``.

    Method:  Run ``model.predict`` in eval mode on ``n_batches`` validation
             batches.  For each outcome that has a time head
             (``model.task_heads.time_idx``), collect:
               * the distribution of predicted hours (mean / std / percentiles)
                 over all patients in the batch — catches collapse to a single
                 number;
               * the model MAE on positive (label==1) entries and the matching
                 constant-baseline MAE (predict the median GT time among the
                 sampled positives).  ``lift = baseline − model``, with the
                 same sign convention as the evaluator: positive ⇒ beats the
                 constant predictor.

    Warns when:
        * the per-outcome prediction std (over all patients) is below
          ``std_warn_hours`` ⇒ the head is essentially a constant;
        * the lift ≤ 0 ⇒ the head matches or loses to the constant predictor.

    Returns:
        ``{outcome_name: {n_pos, pred_mean, pred_std, pred_percentiles,
                          model_mae, baseline_mae, lift}}``.
    """
    if model.task_heads is None:
        print("[probe_time_head_predictions] No task_heads attached — skipping.")
        return {}

    # Late imports to avoid a circular dependency with utils.build_patient_labels
    # (utils → transformer → diagnose → utils).
    from intervene_enc.utils import build_patient_labels
    from intervene_enc.config.model_config import TRAINING_SETTINGS

    model.eval()
    device = next(model.parameters()).device

    time_idx = model.task_heads.time_idx.detach().cpu().tolist()
    outcome_names = list(model.outcome_names)

    # Per-outcome accumulators.
    all_preds  = {k: [] for k in time_idx}
    pos_preds  = {k: [] for k in time_idx}
    pos_gts    = {k: [] for k in time_idx}

    for batch in _take_n(loader, n_batches):
        batch = _to_device(batch, device)
        labels, gt_time, _present = build_patient_labels(
            model, batch, TRAINING_SETTINGS, device,
        )
        _, time_pred, _, _ = model.predict(
            parent_raw_ids=batch["parent_raw_ids"],
            concept_ids=batch["concept_ids"],
            value_ids=batch["value_ids"],
            position_ids=batch["position_ids"],
            abs_ts=batch["abs_ts"],
            context_vec=batch["context_vec"],
        )
        time_pred = time_pred.float().cpu().numpy()      # [B, K_time]
        labels_np = labels.cpu().numpy()                  # [B, K_all]
        gt_np     = gt_time.float().cpu().numpy()         # [B, K_all]

        for j, k in enumerate(time_idx):
            all_preds[k].append(time_pred[:, j])
            pos_mask = (labels_np[:, k] == 1) & np.isfinite(gt_np[:, k])
            if pos_mask.any():
                pos_preds[k].append(time_pred[pos_mask, j])
                pos_gts[k].append(gt_np[pos_mask, k])

    report = {}
    flagged_constant = []
    flagged_no_lift  = []
    print("[probe_time_head_predictions] per-outcome time head check")
    print(f"  {'outcome':<32s}  {'n_pos':>5s}  {'pred_mean':>9s}  "
          f"{'pred_std':>8s}  {'p5':>6s}  {'p50':>6s}  {'p95':>6s}  "
          f"{'model_mae':>9s}  {'baseline':>8s}  {'lift':>7s}")
    for k in time_idx:
        name = outcome_names[k]
        preds_all = np.concatenate(all_preds[k]) if all_preds[k] else np.array([])
        preds_pos = np.concatenate(pos_preds[k]) if pos_preds[k] else np.array([])
        gts_pos   = np.concatenate(pos_gts[k])   if pos_gts[k]   else np.array([])

        pred_mean = float(preds_all.mean()) if preds_all.size else float("nan")
        pred_std  = float(preds_all.std())  if preds_all.size else float("nan")
        pcts = np.percentile(preds_all, [5, 50, 95]).tolist() if preds_all.size else [float("nan")] * 3

        if preds_pos.size:
            model_mae    = float(np.mean(np.abs(preds_pos - gts_pos)))
            baseline_mae = float(np.mean(np.abs(gts_pos - np.median(gts_pos))))
            lift         = baseline_mae - model_mae
        else:
            model_mae = baseline_mae = lift = float("nan")

        out = {
            "n_pos":            int(preds_pos.size),
            "pred_mean":        pred_mean,
            "pred_std":         pred_std,
            "pred_percentiles": pcts,
            "model_mae":        model_mae,
            "baseline_mae":     baseline_mae,
            "lift":             lift,
        }
        report[name] = out
        print(f"  {name:<32s}  {out['n_pos']:>5d}  "
              f"{pred_mean:>9.2f}  {pred_std:>8.2f}  "
              f"{pcts[0]:>6.1f}  {pcts[1]:>6.1f}  {pcts[2]:>6.1f}  "
              f"{model_mae:>9.2f}  {baseline_mae:>8.2f}  {lift:>+7.2f}")
        if preds_all.size and pred_std < std_warn_hours:
            flagged_constant.append(name)
        if not math.isnan(lift) and lift <= 0:
            flagged_no_lift.append(name)

    if flagged_constant:
        print(f"  WARN: near-constant time predictions for {flagged_constant} "
              f"(pred_std < {std_warn_hours} h).")
    if flagged_no_lift:
        print(f"  WARN: time head does not beat constant predictor for "
              f"{flagged_no_lift} (lift ≤ 0).")
    return report


# ───────── one-shot wrapper ───────────────────────────────────────────── #
def run_diagnostics(model: InterveneEncoder, loader, n_batches: int = 2,
                    p: float = 0.15) -> dict:
    """
    Purpose: Run the standard suite of BERT-encoder health checks and return
             a dict aggregating their outputs.
    Method:  Calls each probe with the provided loader and prints results to
             stdout.  Safe to call after Phase 2 (no task heads) — the
             outcome-head / pool / time-head probes self-skip in that case.

    Args:
        model     : trained InterveneEncoder.
        loader    : validation DataLoader.
        n_batches : how many batches to sample per probe.
        p         : MLM ratio to re-apply for the MLM / time-aux / legality probes.
        mode      : MLM mode (positional / hierarchical) — should match training.
    """
    print(f"\n=== run_diagnostics  n_batches={n_batches}  p={p} ===")
    report = {
        "mlm_accuracy":            probe_mlm_accuracy(model, loader, n_batches, p),
        "time_aux_residuals":      probe_time_aux_residuals(model, loader, n_batches, p),
        "legality_starvation":     probe_legality_starvation(model, loader, n_batches, p),
        "outcome_logit_distribution": probe_outcome_logit_distribution(model, loader, n_batches),
        "pool_attention":          probe_pool_attention(model, loader, n_batches),
        "time_head_predictions":   probe_time_head_predictions(
                                       model, loader, n_batches=max(n_batches, 4)),
    }
    print("=== end run_diagnostics ===\n")
    return report
