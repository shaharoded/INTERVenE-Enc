"""
evaluation.py — Fixed post-training evaluation for EMR autoresearch (BERT-pivot).

DO NOT MODIFY — these metrics define the optimization target for each research round.
The agent should NOT edit this file. Improving these metrics is the goal.

Metrics (computed on the held-out test set, not the training validation split):

  Primary    — patient_auroc_weighted  : support-weighted mean per-outcome AUROC.
                                         Each (patient, outcome) contributes one
                                         (P_<outcome>, label) pair from a single
                                         bidirectional encoder pass. Higher is better.
  Secondary  — patient_auprc_weighted  : same, AUPRC.
  Tertiary   — patient_max_f1_weighted : same, max-F1 swept over the PR curve.
  Quaternary — length_of_stay_mae_hours: |T_RELEASE_EVENT - GT release time| MAE
                                         over released patients. Lower is better.
  Per-outcome — time_head_mae_hrs:<o>  : |T_<outcome> - nearest GT occurrence|.

Evaluation protocol (mirrors evaluation.ipynb exactly):
  1. Load held-out test data (3-way split from data/source/, never seen during training).
  2. Re-process with the scaler fitted on the training pool.
  3. Build two datasets:
       full     — untruncated (for ground-truth extraction).
       seed     — EVAL_INPUT_DAYS-day truncation (matches Phase-3 input window).
  4. Run inference.predict(model, seed_ds) → one row per patient with
       P_<outcome> = sigmoid(risk_logit)
       T_<outcome> = softplus(time_logit) hours.
  5. Score patient-level AUROC/AUPRC/max-F1/F1@0.5 per outcome.
  6. RELEASE_EVENT is excluded from AUC/AUPRC/F1 (it is the negation of DEATH in
     this cohort — including both double-counts the same terminal-ranking task)
     and reported via length_of_stay_mae instead.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transform_emr.dataset import DataProcessor, EMRDataset
from transform_emr.config.dataset_config import TAK_REPO_PATH, OUTCOME_RARE_THRESHOLD_PCT, RELEASE_TOKEN
from transform_emr.inference import predict

# ---------------------------------------------------------------------------
# Fixed evaluation constants (do not change)
# ---------------------------------------------------------------------------

EVAL_INPUT_DAYS = 2  # seed window matches Phase-3 input alignment
# Time-head forecasting cutoff: GT events that occur within the encoder's
# input window are *given to the model as input*, not forecast targets — so
# time-MAE scoring on positives only counts events occurring after this many
# hours from admission (= EVAL_INPUT_DAYS * 24).
FORECAST_CUTOFF_HOURS = EVAL_INPUT_DAYS * 24.0

# Outcome-support threshold = same 1 % used at data-load time
# (OUTCOME_RARE_THRESHOLD_PCT in dataset_config). Outcomes that passed train-set
# filtering can still be rarer in the held-out test set, so we re-check at eval
# time. Below this share an outcome's AUROC/AUPRC is reported as NaN (still
# printed per-outcome) and excluded from headline means.
EVAL_PREVALENCE_THRESHOLD = OUTCOME_RARE_THRESHOLD_PCT / 100.0

# Outcomes excluded from the AUROC/AUPRC/F1 evaluation entirely.
# RELEASE_EVENT is the negation of DEATH_EVENT in this cohort (essentially no
# censoring — every patient has either DEATH or RELEASE). Including both in
# the discrimination headline double-counts the same terminal-event ranking
# task. RELEASE stays in the LM vocab + risk head training and is reported via
# length_of_stay_mae instead.
AUC_EXCLUDE = (RELEASE_TOKEN,)


def _min_positives(n_patients, threshold=EVAL_PREVALENCE_THRESHOLD):
    """Minimum positive count for an outcome's AUC to be emitted (≥1)."""
    return max(1, int(round(threshold * n_patients)))


# ---------------------------------------------------------------------------
# Ground-truth extraction
# ---------------------------------------------------------------------------

def extract_ground_truth(eval_ds, outcome_names):
    """
    Purpose: Per-patient first-occurrence times for each outcome.
    Method:  Scan each patient's full (untruncated) token sequence; record the
             earliest TimePoint where its token appears as an outcome.

    Args:
        eval_ds       (EMRDataset): untruncated test dataset.
        outcome_names (list[str]):  outcome tokens to collect.

    Returns:
        dict: {patient_id: {outcome_name: first_time_hours or np.inf}}.
    """
    outcome_set = set(outcome_names)
    tok_col = "PositionToken" if "PositionToken" in next(iter(eval_ds.patient_groups.values())).columns else "Token"
    gt = {}
    for pid in eval_ds.patient_ids:
        df = eval_ds.patient_groups[pid]
        patient_gt = {n: np.inf for n in outcome_names}
        for _, row in df.iterrows():
            tok = row[tok_col]
            if tok in outcome_set:
                t = row["TimePoint"]
                if t < patient_gt[tok]:
                    patient_gt[tok] = t
        gt[pid] = patient_gt
    return gt


def extract_ground_truth_episodes(eval_ds, outcome_names):
    """
    Purpose: Per-patient all-occurrence ground truth (list of times) per outcome.
    Method:  Scan each patient's full (untruncated) token sequence.

    Returns:
        dict: {patient_id: {outcome_name: [t1, t2, ...]}}.
    """
    outcome_set = set(outcome_names)
    tok_col = "PositionToken" if "PositionToken" in next(iter(eval_ds.patient_groups.values())).columns else "Token"
    gt = {}
    for pid in eval_ds.patient_ids:
        df = eval_ds.patient_groups[pid]
        patient_gt = {n: [] for n in outcome_names}
        for _, row in df.iterrows():
            tok = row[tok_col]
            if tok in outcome_set:
                patient_gt[tok].append(row["TimePoint"])
        gt[pid] = patient_gt
    return gt


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_patient_auc(predictions, gt_episodes, outcome_names, min_positives=None):
    """
    Purpose: Patient-level AUROC / AUPRC / max-F1 / F1@0.5 per outcome.
    Method:  Each patient contributes one (P_<outcome>, label) pair where
             label = 1 iff the outcome occurred in the un-truncated GT.

    Args:
        predictions   (pd.DataFrame): one row per patient (from inference.predict).
        gt_episodes   (dict): {pid: {outcome: [t1, ...]}}.
        outcome_names (list[str]): outcomes to score.
        min_positives (int, optional): minimum positive patients to emit an AUC.

    Returns:
        pd.DataFrame indexed by outcome:
            auroc, auprc, max_f1, max_f1_threshold, f1_at_0_5,
            n_pos, n_neg, prevalence.
    """
    n_patients = predictions.shape[0]
    if min_positives is None:
        min_positives = _min_positives(n_patients)

    rows = []
    for name in outcome_names:
        pcol = f"P_{name}"
        if pcol not in predictions.columns:
            continue
        scores = predictions[pcol].to_numpy()
        labels = np.array([
            int(len(gt_episodes.get(pid, {}).get(name, [])) > 0)
            for pid in predictions.index
        ], dtype=np.int64)
        n_pos = int(labels.sum())
        n_neg = int(len(labels) - n_pos)
        prevalence = n_pos / max(1, n_pos + n_neg)

        if n_pos < min_positives or n_neg < min_positives:
            rows.append({"outcome": name, "auroc": np.nan, "auprc": np.nan,
                         "max_f1": np.nan, "max_f1_threshold": np.nan,
                         "f1_at_0_5": np.nan,
                         "n_pos": n_pos, "n_neg": n_neg, "prevalence": prevalence})
            continue

        precisions, recalls, thresholds = precision_recall_curve(labels, scores)
        f1s = np.where(
            (precisions + recalls) > 0,
            2 * precisions * recalls / np.maximum(precisions + recalls, 1e-12),
            0.0,
        )
        best_idx = int(np.argmax(f1s))
        max_f1 = float(f1s[best_idx])
        if best_idx < len(thresholds):
            max_f1_thr = float(thresholds[best_idx])
        else:
            max_f1_thr = float(thresholds[-1]) if len(thresholds) else 0.5

        preds_05 = (scores >= 0.5).astype(int)
        tp = int(((preds_05 == 1) & (labels == 1)).sum())
        fp = int(((preds_05 == 1) & (labels == 0)).sum())
        fn = int(((preds_05 == 0) & (labels == 1)).sum())
        prec_05 = tp / max(tp + fp, 1)
        rec_05  = tp / max(tp + fn, 1)
        f1_at_0_5 = 2 * prec_05 * rec_05 / (prec_05 + rec_05) if (prec_05 + rec_05) > 0 else 0.0

        rows.append({
            "outcome":          name,
            "auroc":            float(roc_auc_score(labels, scores)),
            "auprc":            float(average_precision_score(labels, scores)),
            "max_f1":           max_f1,
            "max_f1_threshold": max_f1_thr,
            "f1_at_0_5":        float(f1_at_0_5),
            "n_pos":            n_pos,
            "n_neg":            n_neg,
            "prevalence":       prevalence,
        })

    return pd.DataFrame(rows).set_index("outcome").sort_values("auroc", ascending=False)


def weighted_mean_auc(auc_table, by="n_pos"):
    """
    Purpose: Support-weighted mean AUROC / AUPRC / F1 across outcomes.
    Method:  Σ(w_o · M_o) / Σ(w_o) over outcomes with non-NaN AUC.
             Default weight is n_pos so rare outcomes contribute less.

    Returns:
        dict with weighted and simple means for AUROC, AUPRC, max_f1, f1_at_0_5,
        plus n_outcomes_used.
    """
    tbl = auc_table.dropna(subset=["auroc"])
    if len(tbl) == 0:
        nan = float("nan")
        return {"auroc_weighted": nan, "auprc_weighted": nan,
                "max_f1_weighted": nan, "f1_at_0_5_weighted": nan,
                "auroc_simple": nan, "auprc_simple": nan,
                "max_f1_simple": nan, "f1_at_0_5_simple": nan, "n_outcomes_used": 0}
    w = tbl[by].astype(float).values
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    return {
        "auroc_weighted":     float((tbl["auroc"].values     * w).sum()),
        "auprc_weighted":     float((tbl["auprc"].values     * w).sum()),
        "max_f1_weighted":    float((tbl["max_f1"].values    * w).sum()),
        "f1_at_0_5_weighted": float((tbl["f1_at_0_5"].values * w).sum()),
        "auroc_simple":       float(tbl["auroc"].mean()),
        "auprc_simple":       float(tbl["auprc"].mean()),
        "max_f1_simple":      float(tbl["max_f1"].mean()),
        "f1_at_0_5_simple":   float(tbl["f1_at_0_5"].mean()),
        "n_outcomes_used":    int(len(tbl)),
    }


def length_of_stay_mae(predictions, gt_episodes, release_token=RELEASE_TOKEN,
                      forecast_cutoff_hours=FORECAST_CUTOFF_HOURS):
    """
    Purpose: Length-of-stay regression MAE from the time head's RELEASE slot,
             reported alongside the constant-predictor baseline so the lift is
             explicit (a constant-median predictor minimizes MAE; the resulting
             MAE equals the mean absolute deviation around the GT median).
    Method:  Forecasting-cohort only — patients whose GT release occurs AFTER
             ``forecast_cutoff_hours`` (events inside the input window are
             context, not a forecasting task). For each such patient:
               GT_LoS   = earliest GT RELEASE TimePoint.
               Pred_LoS = predictions.loc[pid, "T_<RELEASE_TOKEN>"] (hours).
             - Model MAE = mean |Pred_LoS − GT_LoS|.
             - Baseline (constant-median) MAE = mean |GT_LoS − median(GT_LoS)|.
               This is the MAE floor any constant predictor cannot beat; a
               useful model must do better.

    Returns:
        dict with keys
          mae_hours, median_hours, p90_hours, n_patients,
          gt_mean_hours, pred_mean_hours,
          baseline_mae_hours, gt_median_hours, lift_hours.
        `lift_hours` = baseline_mae − model_mae (positive ⇒ model beats the
        constant predictor; ≤ 0 ⇒ model has not learned LoS signal).
    """
    tcol = f"T_{release_token}"
    _empty = {"mae_hours": float("nan"), "median_hours": float("nan"),
              "p90_hours": float("nan"), "n_patients": 0,
              "gt_mean_hours": float("nan"), "pred_mean_hours": float("nan"),
              "baseline_mae_hours": float("nan"), "gt_median_hours": float("nan"),
              "lift_hours": float("nan")}
    if tcol not in predictions.columns:
        return dict(_empty)
    errs, gts, preds = [], [], []
    for pid in predictions.index:
        gt_releases = gt_episodes.get(pid, {}).get(release_token, [])
        if not gt_releases:
            continue
        gt_los = float(min(gt_releases))
        if gt_los <= forecast_cutoff_hours:
            continue  # release within input window — context, not a forecast
        pred_los = float(predictions.loc[pid, tcol])
        errs.append(abs(pred_los - gt_los))
        gts.append(gt_los)
        preds.append(pred_los)
    if not errs:
        return dict(_empty)
    arr      = np.asarray(errs)
    gts_arr  = np.asarray(gts)
    mae      = float(arr.mean())
    gt_median = float(np.median(gts_arr))
    baseline = float(np.mean(np.abs(gts_arr - gt_median)))   # MAE-optimal constant predictor
    return {
        "mae_hours":          mae,
        "median_hours":       float(np.median(arr)),
        "p90_hours":          float(np.percentile(arr, 90)),
        "n_patients":         int(len(arr)),
        "gt_mean_hours":      float(np.mean(gts)),
        "pred_mean_hours":    float(np.mean(preds)),
        "baseline_mae_hours": baseline,
        "gt_median_hours":    gt_median,
        "lift_hours":         baseline - mae,
    }


def time_head_mae(predictions, gt_episodes, outcome_names,
                  forecast_cutoff_hours=FORECAST_CUTOFF_HOURS):
    """
    Purpose: Per-outcome time-head MAE on the FORECASTING cohort (events that
             occur AFTER the encoder's input window — events inside the window
             are in the encoder's input and are not a forecasting task), plus
             the constant-predictor baseline so the lift is explicit (and the
             agent can tell whether the time head is genuinely learning timing
             beyond a constant).
    Method:  For each (patient, outcome) where the outcome occurred in GT
             beyond ``forecast_cutoff_hours``:
               - take the GT episode(s) of that outcome occurring after the
                 cutoff, pick the one nearest the model's prediction, score
                 ``|pred − nearest_gt|`` (matches the patient-level rule).
             Baseline = MAE of the MAE-optimal constant predictor (the GT
             median over the same forecasting cohort) = mean |GT − median|.

    Returns:
        pd.DataFrame indexed by outcome, columns:
            mae_hours          — model MAE on the forecasting cohort.
            n_patients         — # patients with at least one >cutoff event.
            gt_median_hours    — median GT time-to-event (forecasting cohort).
            baseline_mae_hours — MAE of the constant=median predictor (floor).
            lift_hours         — baseline_mae − mae (positive ⇒ model beats
                                 the constant predictor; ≤ 0 ⇒ no temporal
                                 signal learned beyond a constant).
    """
    rows = []
    for name in outcome_names:
        tcol = f"T_{name}"
        if tcol not in predictions.columns:
            continue
        errs, gts = [], []
        for pid in predictions.index:
            episodes = gt_episodes.get(pid, {}).get(name, [])
            # Forecasting cohort: only GT episodes beyond the input window.
            episodes_f = [t for t in episodes if t > forecast_cutoff_hours]
            if not episodes_f:
                continue
            pred_t = float(predictions.loc[pid, tcol])
            nearest_gt = min(episodes_f, key=lambda t: abs(pred_t - t))
            errs.append(abs(pred_t - nearest_gt))
            gts.append(float(nearest_gt))
        if not errs:
            rows.append({"outcome": name, "mae_hours": float("nan"),
                         "n_patients": 0,
                         "gt_median_hours": float("nan"),
                         "baseline_mae_hours": float("nan"),
                         "lift_hours": float("nan")})
            continue
        gts_arr   = np.asarray(gts)
        gt_median = float(np.median(gts_arr))
        baseline  = float(np.mean(np.abs(gts_arr - gt_median)))
        mae       = float(np.mean(errs))
        rows.append({
            "outcome":            name,
            "mae_hours":          mae,
            "n_patients":         len(errs),
            "gt_median_hours":    gt_median,
            "baseline_mae_hours": baseline,
            "lift_hours":         baseline - mae,
        })
    return pd.DataFrame(rows).set_index("outcome").sort_values("mae_hours")


# ---------------------------------------------------------------------------
# Main evaluation entry point (called by api.py)
# ---------------------------------------------------------------------------

def evaluate_on_test_set(model, tokenizer, test_temporal_raw, test_ctx_raw,
                          scaler, checkpoint_dir, batch_size=16):
    """
    Purpose: Full post-training evaluation on the held-out test set.
    Method:  Re-process the raw test data twice — untruncated (for GT) and
             EVAL_INPUT_DAYS-truncated (as the input window). Run a single
             bidirectional encoder pass via inference.predict and score
             patient-level AUROC/AUPRC/F1, length-of-stay MAE, time-head MAE.

    Args:
        model:             Trained EMREncoder with task heads attached.
        tokenizer:         EMRTokenizer fitted on training data.
        test_temporal_raw: Raw test temporal events.
        test_ctx_raw:      Raw test context features.
        scaler:            StandardScaler fitted on train (from checkpoints/scaler.pkl).
        checkpoint_dir:    Path to the checkpoints directory.
        batch_size:        DataLoader batch size for prediction.

    Returns:
        dict (see api.py summary block for the full key list).
    """
    print("[Eval] Processing full test sequences (ground truth)...")
    full_proc = DataProcessor(
        test_temporal_raw.copy(), test_ctx_raw.copy(),
        scaler=scaler,
        tak_repo_path=TAK_REPO_PATH,
        checkpoint_path=checkpoint_dir,
    )
    full_temporal_df, full_ctx_df = full_proc.run()
    eval_ds_full = EMRDataset(full_temporal_df, full_ctx_df, tokenizer=tokenizer)

    print(f"[Eval] Processing truncated test sequences ({EVAL_INPUT_DAYS}-day input)...")
    trunc_proc = DataProcessor(
        test_temporal_raw.copy(), test_ctx_raw.copy(),
        scaler=scaler,
        tak_repo_path=TAK_REPO_PATH,
        checkpoint_path=checkpoint_dir,
        max_input_days=EVAL_INPUT_DAYS,
    )
    trunc_temporal_df, trunc_ctx_df = trunc_proc.run()
    eval_ds_input = EMRDataset(trunc_temporal_df, trunc_ctx_df, tokenizer=tokenizer)

    print("[Eval] Running single-pass bidirectional inference...")
    model.eval()
    predictions = predict(model, eval_ds_input, batch_size=batch_size)
    print(f"[Eval] Predictions: {predictions.shape[0]} patients, "
          f"{sum(c.startswith('P_') for c in predictions.columns)} risk outputs, "
          f"{sum(c.startswith('T_') for c in predictions.columns)} time outputs.")

    outcome_names = model.outcome_names

    gt_first    = extract_ground_truth(eval_ds_full, outcome_names)
    gt_episodes = extract_ground_truth_episodes(eval_ds_full, outcome_names)

    auc_outcome_names = [n for n in outcome_names if n not in AUC_EXCLUDE]
    print(f"[Eval] AUC/F1 computed over {len(auc_outcome_names)} outcomes "
          f"(excluded from AUROC headline: {list(AUC_EXCLUDE)}).")

    patient_auc_table = per_patient_auc(predictions, gt_episodes, auc_outcome_names)
    patient_mean      = weighted_mean_auc(patient_auc_table, by="n_pos")

    time_mae_table = time_head_mae(predictions, gt_episodes, outcome_names)
    los_stats      = length_of_stay_mae(predictions, gt_episodes)
    print(f"[Eval] Length-of-stay MAE: {los_stats['mae_hours']:.2f}h "
          f"(median {los_stats['median_hours']:.1f}h, p90 {los_stats['p90_hours']:.1f}h, "
          f"n={los_stats['n_patients']}, GT mean {los_stats['gt_mean_hours']:.1f}h, "
          f"pred mean {los_stats['pred_mean_hours']:.1f}h)")
    print(f"[Eval] LoS baseline (predict GT median {los_stats['gt_median_hours']:.1f}h): "
          f"baseline_MAE={los_stats['baseline_mae_hours']:.2f}h  "
          f"model_MAE={los_stats['mae_hours']:.2f}h  "
          f"lift={los_stats['lift_hours']:+.2f}h "
          f"({'beats' if los_stats['lift_hours'] > 0 else 'matches/loses to'} constant predictor)")

    print("[Eval] Patient-level AUC / F1 per outcome:")
    for outcome, row in patient_auc_table.iterrows():
        if not np.isnan(row["auroc"]):
            print(f"  {outcome:<45} AUROC={row['auroc']:.3f}  AUPRC={row['auprc']:.3f}  "
                  f"maxF1={row['max_f1']:.3f}(τ={row['max_f1_threshold']:.3f})  "
                  f"F1@0.5={row['f1_at_0_5']:.3f}  "
                  f"n_pos={int(row['n_pos'])}  prev={row['prevalence']:.3f}")
    print(f"[Eval] Patient-level mean (support-weighted): "
          f"AUROC={patient_mean['auroc_weighted']:.3f}  "
          f"AUPRC={patient_mean['auprc_weighted']:.3f}  "
          f"maxF1={patient_mean['max_f1_weighted']:.3f}  "
          f"F1@0.5={patient_mean['f1_at_0_5_weighted']:.3f}  "
          f"(simple AUROC={patient_mean['auroc_simple']:.3f} / "
          f"AUPRC={patient_mean['auprc_simple']:.3f}, "
          f"n_outcomes={patient_mean['n_outcomes_used']})")
    if len(time_mae_table):
        print("[Eval] Time-head MAE per outcome (positives only, nearest GT; "
              "baseline = MAE of predict-GT-median; lift = baseline − model, "
              "positive ⇒ beats constant predictor):")
        for outcome, row in time_mae_table.iterrows():
            if not np.isnan(row["mae_hours"]):
                print(f"  {outcome:<45} MAE={row['mae_hours']:.2f}h  "
                      f"baseline={row['baseline_mae_hours']:.2f}h  "
                      f"lift={row['lift_hours']:+.2f}h  "
                      f"gt_median={row['gt_median_hours']:.1f}h  "
                      f"n_pos={int(row['n_patients'])}")

    return dict(
        patient_auc_table=patient_auc_table,
        patient_auroc_weighted=patient_mean["auroc_weighted"],
        patient_auprc_weighted=patient_mean["auprc_weighted"],
        patient_auroc_simple=patient_mean["auroc_simple"],
        patient_auprc_simple=patient_mean["auprc_simple"],
        patient_max_f1_weighted=patient_mean["max_f1_weighted"],
        patient_max_f1_simple=patient_mean["max_f1_simple"],
        patient_f1_at_0_5_weighted=patient_mean["f1_at_0_5_weighted"],
        patient_f1_at_0_5_simple=patient_mean["f1_at_0_5_simple"],
        n_outcomes_used=patient_mean["n_outcomes_used"],
        time_mae_table=time_mae_table,
        length_of_stay_mae_hours=los_stats["mae_hours"],
        length_of_stay_median_hours=los_stats["median_hours"],
        length_of_stay_p90_hours=los_stats["p90_hours"],
        length_of_stay_n_patients=los_stats["n_patients"],
        length_of_stay_baseline_mae_hours=los_stats["baseline_mae_hours"],
        length_of_stay_gt_median_hours=los_stats["gt_median_hours"],
        length_of_stay_lift_hours=los_stats["lift_hours"],
        predictions=predictions,
        gt_first=gt_first,
        gt_episodes=gt_episodes,
    )


# ===========================================================================
# Bootstrap variance for a trained checkpoint
# ===========================================================================

def bootstrap_evaluate(model, tokenizer, test_temporal_raw, test_ctx_raw,
                      scaler, checkpoint_dir, batch_size, B=2000, seed=42):
    """
    Purpose: Patient-level bootstrap CIs for the locked test-set headline.
    Method:  Run `evaluate_on_test_set` ONCE to get per-patient predictions +
             ground-truth, then bootstrap over the held-out TEST PATIENTS
             (resample with replacement, B reps) to produce 95% percentile CIs
             for the support-weighted AUROC / AUPRC headline, per-outcome
             AUROC/AUPRC, and length-of-stay MAE (RELEASE-only + all-patients
             terminal-time supplementary). Single model, single inference pass —
             far cheaper than re-seeding the full pipeline.

    Args:
        model               : trained EMREncoder with task heads attached.
        tokenizer           : EMRTokenizer matching the training vocab.
        test_temporal_raw   : held-out test split temporal DataFrame.
        test_ctx_raw        : held-out test split context DataFrame.
        scaler              : fitted scaler (joblib-loaded).
        checkpoint_dir (str): for evaluate_on_test_set's DataProcessor path.
        batch_size (int)    : inference batch size.
        B (int)             : number of bootstrap resamples (default 2000).
        seed (int)          : RNG seed for reproducibility.

    Returns:
        dict with the same headline keys plus *_ci_lo / *_ci_hi / *_boot_mean
        / *_boot_sd entries and per_outcome_ci DataFrame. Also prints a
        grep-friendly summary block.
    """
    import time
    from transform_emr.config.dataset_config import DEATH_TOKEN

    res = evaluate_on_test_set(
        model=model, tokenizer=tokenizer,
        test_temporal_raw=test_temporal_raw, test_ctx_raw=test_ctx_raw,
        scaler=scaler, checkpoint_dir=checkpoint_dir, batch_size=batch_size,
    )
    predictions   = res["predictions"]
    gt_episodes   = res["gt_episodes"]
    outcome_names = [n for n in model.outcome_names if n not in AUC_EXCLUDE]
    pids = list(predictions.index)
    N = len(pids)
    print(f"[boot] point estimate: AUPRC_w={res['patient_auprc_weighted']:.4f} "
          f"AUROC_w={res['patient_auroc_weighted']:.4f} | N_test={N}")

    # Precompute aligned (scores, labels) per outcome.
    cols = {}
    for name in outcome_names:
        pcol = f"P_{name}"
        if pcol not in predictions.columns:
            continue
        scores = predictions[pcol].to_numpy()
        labels = np.array([int(len(gt_episodes.get(p, {}).get(name, [])) > 0)
                           for p in pids], dtype=np.int64)
        cols[name] = (scores, labels)

    min_pos = _min_positives(N)

    def _weighted_stat(idx):
        aurocs, auprcs, weights = [], [], []
        for nm, (sc, lb) in cols.items():
            s, l = sc[idx], lb[idx]
            n_pos = int(l.sum()); n_neg = len(l) - n_pos
            if n_pos < min_pos or n_neg < min_pos:
                continue
            aurocs.append(roc_auc_score(l, s))
            auprcs.append(average_precision_score(l, s))
            weights.append(n_pos)
        if not weights:
            return np.nan, np.nan
        w = np.array(weights, float); w /= w.sum()
        return float((np.array(aurocs) * w).sum()), float((np.array(auprcs) * w).sum())

    per_out = {name: {"auroc": [], "auprc": []} for name in cols}
    boot_auroc, boot_auprc = [], []
    rng = np.random.RandomState(seed)
    t0 = time.time()
    for b in range(B):
        idx = rng.randint(0, N, size=N)
        a, p = _weighted_stat(idx)
        if not (np.isnan(a) or np.isnan(p)):
            boot_auroc.append(a); boot_auprc.append(p)
        for nm, (sc, lb) in cols.items():
            s, l = sc[idx], lb[idx]
            n_pos = int(l.sum()); n_neg = len(l) - n_pos
            if n_pos < min_pos or n_neg < min_pos:
                continue
            per_out[nm]["auroc"].append(roc_auc_score(l, s))
            per_out[nm]["auprc"].append(average_precision_score(l, s))
    print(f"[boot] {B} resamples in {time.time()-t0:.1f}s")

    # Length-of-stay bootstrap: RELEASE-only + supplementary all-patients-terminal.
    tcol = f"T_{RELEASE_TOKEN}"
    los_rel_pairs, los_term_pairs = [], []
    if tcol in predictions.columns:
        for p in pids:
            pred_los = float(predictions.loc[p, tcol])
            rel = gt_episodes.get(p, {}).get(RELEASE_TOKEN, [])
            dth = gt_episodes.get(p, {}).get(DEATH_TOKEN, [])
            if rel:
                los_rel_pairs.append(abs(pred_los - float(min(rel))))
            term_times = list(rel) + list(dth)
            if term_times:
                los_term_pairs.append(abs(pred_los - float(min(term_times))))
    los_rel_arr = np.asarray(los_rel_pairs)
    los_term_arr = np.asarray(los_term_pairs)

    boot_los_rel, boot_los_term = [], []
    rng2 = np.random.RandomState(seed + 1)
    if los_rel_arr.size:
        nR = los_rel_arr.size
        for _ in range(B):
            boot_los_rel.append(los_rel_arr[rng2.randint(0, nR, size=nR)].mean())
    if los_term_arr.size:
        nT = los_term_arr.size
        for _ in range(B):
            boot_los_term.append(los_term_arr[rng2.randint(0, nT, size=nT)].mean())

    def _ci(arr):
        a = np.asarray(arr)
        return np.percentile(a, 2.5), np.percentile(a, 97.5), a.mean(), a.std()

    print(f"\n=== BOOTSTRAP 95pct CI (patient resample, B={B}) ===")
    out = dict(res)
    for label, point, arr in [
        ("patient_auprc_weighted", res["patient_auprc_weighted"], boot_auprc),
        ("patient_auroc_weighted", res["patient_auroc_weighted"], boot_auroc),
    ]:
        if not arr:
            print(f"{label}: (insufficient successful resamples)")
            continue
        lo, hi, mean, sd = _ci(arr)
        out[f"{label}_ci_lo"] = float(lo)
        out[f"{label}_ci_hi"] = float(hi)
        out[f"{label}_boot_mean"] = float(mean)
        out[f"{label}_boot_sd"] = float(sd)
        print(f"{label}: point={point:.4f}  boot_mean={mean:.4f}  "
              f"95%CI=[{lo:.4f}, {hi:.4f}]  sd={sd:.4f}")

    if boot_los_rel:
        lo, hi, mean, sd = _ci(boot_los_rel)
        out["length_of_stay_mae_hours_ci_lo"] = float(lo)
        out["length_of_stay_mae_hours_ci_hi"] = float(hi)
        out["length_of_stay_mae_hours_boot_mean"] = float(mean)
        out["length_of_stay_mae_hours_boot_sd"] = float(sd)
        print(f"length_of_stay_mae_hours (RELEASE-only, n={los_rel_arr.size}): "
              f"point={res['length_of_stay_mae_hours']:.4f}  "
              f"boot_mean={mean:.4f}  95%CI=[{lo:.4f}, {hi:.4f}]  sd={sd:.4f}")
    if boot_los_term:
        lo, hi, mean, sd = _ci(boot_los_term)
        out["length_of_stay_mae_hours_terminal_ci_lo"] = float(lo)
        out["length_of_stay_mae_hours_terminal_ci_hi"] = float(hi)
        print(f"length_of_stay_mae_hours (ALL-patients terminal time, n={los_term_arr.size}): "
              f"boot_mean={mean:.4f}  95%CI=[{lo:.4f}, {hi:.4f}]  sd={sd:.4f}")

    # Per-outcome CI table.
    print("\n--- per-outcome 95% CI ---")
    print(f"{'outcome':<34}{'AUROC [95% CI]':<30}{'AUPRC [95% CI]'}")
    per_out_rows = []
    for name in cols:
        ar, pr = per_out[name]["auroc"], per_out[name]["auprc"]
        if not ar:
            print(f"{name:<34}(insufficient positives in resamples)")
            per_out_rows.append({"outcome": name, "auroc_mean": np.nan,
                                "auroc_lo": np.nan, "auroc_hi": np.nan,
                                "auprc_mean": np.nan,
                                "auprc_lo": np.nan, "auprc_hi": np.nan})
            continue
        alo, ahi, am, _ = _ci(ar); plo, phi, pm, _ = _ci(pr)
        per_out_rows.append({"outcome": name,
                            "auroc_mean": float(am), "auroc_lo": float(alo), "auroc_hi": float(ahi),
                            "auprc_mean": float(pm), "auprc_lo": float(plo), "auprc_hi": float(phi)})
        print(f"{name:<34}{am:.3f} [{alo:.3f},{ahi:.3f}]      "
              f"{pm:.3f} [{plo:.3f},{phi:.3f}]")
    out["per_outcome_ci"] = pd.DataFrame(per_out_rows).set_index("outcome")
    print("\n[boot] done.")
    return out
