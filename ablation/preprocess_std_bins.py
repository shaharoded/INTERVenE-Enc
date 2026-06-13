"""
ablation/preprocess_std_bins.py
==============================

Build a std-bin variant of ``temporal_data.csv`` for the Exp-E temporal-
abstraction ablation (see ``program.md`` → Stage E).

Input is ``data/source/mimic-iv-input-data.csv`` — the same raw-measurement
file STraTS / GRU-D consume. The output mirrors the Mediator TAK pipeline's
``temporal_data.csv`` schema (PatientId, ConceptName, StartDateTime,
EndDateTime, Value) so the encoder's data loader can read it unchanged once
the data-source path is repointed at the output directory. **Outcome
ConceptNames are kept identical to the Mediator's** (``HYPERGLYCEMIA_EVENT``,
``DEATH_EVENT``, ``ADMISSION_EVENT`` etc.) so no dataset-config change is
required downstream.

What it does
------------

1. **Outcome events** — synthesised + pass-through, both kept *in* the
   temporal stream (the encoder's loader treats them as both input tokens
   *and* label sources, mirroring the TAK output):

   * ``HYPERGLYCEMIA_EVENT`` / ``HYPOGLYCEMIA_EVENT`` /
     ``SEVERE_HYPERGLYCEMIA_EVENT`` / ``SEVERE_HYPOGLYCEMIA_EVENT`` /
     ``KIDNEY_COMPLICATION_EVENT`` are *synthesised* from glucose /
     creatinine measurements using the **exact rules in
     med-transformers-baseline/scripts/preprocess_mimic_iv.py** (single
     source of truth — we import those helpers).
   * Pass-through outcomes (``DEATH``, ``CARDIOVASCULAR_DISORDER``,
     ``ACIDOSIS``, ``KETOACIDOSIS``, ``HYPEROSMOLALITY``, ``INFECTION``,
     ``DIABETIC_COMA``, ``ACUTE_RESPIRATORY_DISORDER``,
     ``OTHER_COMPLICATION``, ``DIABETES_DIAGNOSIS``) and terminals
     (``RELEASE``) are renamed to the canonical ``<NAME>_EVENT`` form.
   * ``ADMISSION`` → ``ADMISSION_EVENT``.

2. **Numeric measurements** — std-binned globally (no train/test split
   awareness — the Mediator computes its abstractions globally too).
   Per-concept (mean, std) is computed over all rows; each observation is
   z-scored and binned into one of seven categories at ±0.5σ, ±1σ, ±2σ.
   The output row's ``ConceptName`` becomes ``<orig>_STD_<bin>`` and its
   ``Value`` becomes the bin string.

3. **Interval collapsing** — per (patient, ConceptName), consecutive
   observations within 24 h merge into one event whose
   ``StartDateTime = first.StartDateTime`` and
   ``EndDateTime  = last.StartDateTime``. Chains of arbitrary length merge
   because the 24 h rule is checked pairwise against the previous
   observation (cumsum on the "new group" boolean).

4. **Non-numeric values** (boolean ``True``, MEAL categories, etc.)
   pass through unchanged; they're indicator-style tokens already.

5. **Validation block** prints row counts before / after collapse,
   per-outcome positive-patient prevalence, and per-concept bin
   distributions (NORMAL should be ~38 % with ±0.5σ edges; ±2σ tails
   ≈ 2.3 % each).

Outputs
-------

::

    ablation/data/source_std_bins/temporal_data.csv
    ablation/data/source_std_bins/context_data.csv   (copied verbatim)
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import med-transformers' synthesis helpers + value-bound constants.
# Single source of truth for the Mediator HYPER/HYPO/SEVERE/KIDNEY rules and
# the outcome-name regex map.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
MED_TRANSFORMERS_ROOT = REPO_ROOT.parent.parent / "med-transformers-baseline"
if not MED_TRANSFORMERS_ROOT.exists():
    raise SystemExit(
        f"Expected med-transformers-baseline at {MED_TRANSFORMERS_ROOT}. "
        "Edit MED_TRANSFORMERS_ROOT or sit the two repos side-by-side under "
        "Personal/."
    )
sys.path.insert(0, str(MED_TRANSFORMERS_ROOT / "scripts"))
sys.path.insert(0, str(MED_TRANSFORMERS_ROOT / "src"))

from preprocess_mimic_iv import (  # noqa: E402
    _hyperglycemia_events,
    _hypoglycemia_events,
    _severe_hyperglycemia_events,
    _severe_hypoglycemia_events,
    _kidney_complication_events,
    _matching_concepts,
    _to_numeric_value,
)
from config import (  # noqa: E402
    GLUCOSE_CONCEPT_REGEX,
    GLUCOSE_VALUE_MIN,
    GLUCOSE_VALUE_MAX,
    CREATININE_CONCEPT_REGEX,
    CREATININE_VALUE_MIN,
    CREATININE_VALUE_MAX,
    EVENT_OUTCOME_REGEX,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Temporal input lives only locally (not on the GPU pod). Look in the
# autoresearch source dir first; fall back to the med-transformers data dir
# so the script runs without first having to copy the 1.7 GB file across.
_TEMPORAL_CANDIDATES = [
    REPO_ROOT / "data" / "source" / "mimic-iv-input-data.csv",
    MED_TRANSFORMERS_ROOT / "data" / "mimic-iv-input-data.csv",
]
_CONTEXT_CANDIDATES = [
    REPO_ROOT / "data" / "source" / "context_data.csv",
    MED_TRANSFORMERS_ROOT / "data" / "context_data.csv",
]
INPUT_TEMPORAL_CSV  = next((p for p in _TEMPORAL_CANDIDATES if p.exists()),
                           _TEMPORAL_CANDIDATES[0])
INPUT_CONTEXT_CSV   = next((p for p in _CONTEXT_CANDIDATES if p.exists()),
                           _CONTEXT_CANDIDATES[0])
OUTPUT_DIR          = REPO_ROOT / "ablation" / "data" / "source_std_bins"
OUTPUT_TEMPORAL_CSV = OUTPUT_DIR / "temporal_data.csv"
OUTPUT_CONTEXT_CSV  = OUTPUT_DIR / "context_data.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COLLAPSE_HOURS = 24.0
# 7-bin scheme. Edges in standardised units (z = (v − μ) / σ).
# Expected fractions under a true N(0,1):
#   VERY_LOW  : 2.28 %     SLIGHTLY_HIGH : 14.99 %
#   LOW       : 13.59 %    HIGH          : 13.59 %
#   SLIGHTLY_LOW : 14.99 % VERY_HIGH     : 2.28 %
#   NORMAL    : 38.29 %
BIN_EDGES  = [-np.inf, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, np.inf]
BIN_LABELS = [
    "VERY_LOW", "LOW", "SLIGHTLY_LOW",
    "NORMAL",
    "SLIGHTLY_HIGH", "HIGH", "VERY_HIGH",
]

# ---------------------------------------------------------------------------
# Concept-name conventions
# ---------------------------------------------------------------------------
ADMISSION_RAW   = "ADMISSION"
ADMISSION_EVENT = "ADMISSION_EVENT"
RELEASE_RAW     = "RELEASE"
RELEASE_EVENT   = "RELEASE_EVENT"

# Synthesised outcome names emitted by the imported med-transformers helpers.
# Pass-through detection skips these so we don't double-count.
SYNTHESIZED_OUTCOMES = (
    "HYPERGLYCEMIA_EVENT",
    "HYPOGLYCEMIA_EVENT",
    "SEVERE_HYPERGLYCEMIA_EVENT",
    "SEVERE_HYPOGLYCEMIA_EVENT",
    "KIDNEY_COMPLICATION_EVENT",
)
# Names of every outcome/terminal token the validation audit walks. Keys of
# EVENT_OUTCOME_REGEX (pass-through outcomes) + the synthesised ones +
# RELEASE_EVENT (terminal, not in EVENT_OUTCOME_REGEX).
ALL_OUTCOME_TOKENS = (
    *EVENT_OUTCOME_REGEX.keys(),
    *SYNTHESIZED_OUTCOMES,
    RELEASE_EVENT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def apply_value_bounds(temporal: pd.DataFrame) -> pd.DataFrame:
    """Drop out-of-range glucose / creatinine rows. Mirrors med-transformers."""
    all_concepts = sorted(temporal["ConceptName"].astype(str).unique())
    for label, concept_regex, vmin, vmax in (
        ("glucose",    GLUCOSE_CONCEPT_REGEX,    GLUCOSE_VALUE_MIN,    GLUCOSE_VALUE_MAX),
        ("creatinine", CREATININE_CONCEPT_REGEX, CREATININE_VALUE_MIN, CREATININE_VALUE_MAX),
    ):
        matched = set(_matching_concepts(all_concepts, concept_regex))
        if not matched:
            continue
        idx     = temporal.index[temporal["ConceptName"].isin(matched)]
        values  = pd.to_numeric(temporal.loc[idx, "Value"], errors="coerce")
        in_range = values.between(vmin, vmax, inclusive="both").fillna(False)
        dropped = idx[~in_range]
        if len(dropped):
            temporal = temporal.drop(dropped)
        print(f"[bounds] {label}: dropped {len(dropped):,} out-of-range "
              f"rows (of {len(idx):,} matched).")
    return temporal


def compute_minute_offsets(temporal: pd.DataFrame) -> pd.DataFrame:
    """Compute minute-from-admission per row; drop pre-admission rows.

    Required by the imported synthesis helpers — they sort by ``minute``.
    """
    admissions = (
        temporal.loc[temporal["ConceptName"] == ADMISSION_RAW,
                     ["PatientId", "StartDateTime"]]
        .groupby("PatientId", as_index=False)["StartDateTime"].min()
        .rename(columns={"StartDateTime": "admission_time"})
    )
    temporal = temporal.merge(admissions, on="PatientId", how="inner")
    temporal["minute"] = (
        (temporal["StartDateTime"] - temporal["admission_time"])
        .dt.total_seconds() / 60.0
    )
    temporal = temporal.loc[temporal["minute"] >= 0].copy()
    return temporal


def synthesize_outcome_events(temporal: pd.DataFrame) -> pd.DataFrame:
    """Apply the five Mediator TAK rules to produce synthesised outcome rows."""
    temporal = temporal.copy()
    temporal["numeric_value"] = _to_numeric_value(temporal["Value"])
    all_concepts = sorted(temporal["ConceptName"].astype(str).unique())
    glucose_concepts    = _matching_concepts(all_concepts, GLUCOSE_CONCEPT_REGEX)
    creatinine_concepts = _matching_concepts(all_concepts, CREATININE_CONCEPT_REGEX)

    glucose_rows = temporal.loc[
        temporal["ConceptName"].isin(glucose_concepts)
        & temporal["numeric_value"].notna()
    ].copy()
    creatinine_rows = temporal.loc[
        temporal["ConceptName"].isin(creatinine_concepts)
        & temporal["numeric_value"].notna()
    ].copy()

    frames = []
    if len(glucose_rows):
        frames.append(_hyperglycemia_events(glucose_rows))
        frames.append(_hypoglycemia_events(glucose_rows))
        frames.append(_severe_hyperglycemia_events(glucose_rows))
        frames.append(_severe_hypoglycemia_events(glucose_rows))
    if len(creatinine_rows):
        frames.append(_kidney_complication_events(creatinine_rows))

    frames = [f for f in frames if len(f) > 0]
    if not frames:
        return temporal.iloc[0:0]
    return pd.concat(frames, ignore_index=True)


def rename_passthrough_outcomes_inplace(temporal: pd.DataFrame) -> pd.DataFrame:
    """Rename raw outcome ConceptNames to their canonical ``_EVENT`` form.

    Uses ``EVENT_OUTCOME_REGEX`` (imported) — its keys are the canonical
    names (e.g. ``DEATH_EVENT``) and its values are regexes matching raw
    aliases (e.g. ``^DEATH(?:_EVENT)?$``). Synthesised outcomes are
    skipped — their rows are produced separately and aren't present here.
    """
    temporal = temporal.copy()
    cn = temporal["ConceptName"].astype(str)
    for canonical, pattern in EVENT_OUTCOME_REGEX.items():
        if canonical in SYNTHESIZED_OUTCOMES:
            continue
        mask = cn.str.match(pattern)
        if mask.any():
            n = int(mask.sum())
            temporal.loc[mask, "ConceptName"] = canonical
            print(f"[rename] {canonical:<36s} <- {n:>5d} raw rows")
    # ADMISSION → ADMISSION_EVENT and RELEASE → RELEASE_EVENT — neither is
    # in EVENT_OUTCOME_REGEX (admission isn't an outcome; release is a
    # terminal, not a target), but the encoder's tokenizer expects the
    # canonical names. Handle them explicitly here.
    for raw, canonical in ((ADMISSION_RAW, ADMISSION_EVENT),
                           (RELEASE_RAW,   RELEASE_EVENT)):
        mask = temporal["ConceptName"] == raw
        if mask.any():
            n = int(mask.sum())
            temporal.loc[mask, "ConceptName"] = canonical
            print(f"[rename] {canonical:<36s} <- {n:>5d} raw rows")
    return temporal


def std_bin_numeric_values(temporal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Std-bin rows whose ``Value`` parses as numeric; pass others through.

    Per-concept (mean, std) is computed globally (all patients) — the
    Mediator pipeline likewise abstracts over the whole dataset.  After
    binning, the row's ``ConceptName`` becomes ``<orig>_STD_<bin>`` and its
    ``Value`` becomes the bin label. Rows with non-numeric Value (booleans,
    MEAL categories, freshly-renamed outcome events with Value="True") are
    untouched — categorical pass-through.

    Returns
    -------
    (binned_or_passthrough_df, stats_df)
        ``stats_df`` is keyed by ConceptName with columns ``[count, mean, std]``
        — used for the per-concept bin-distribution sanity check.
    """
    temporal = temporal.copy()
    temporal["numeric_value"] = pd.to_numeric(temporal["Value"], errors="coerce")
    is_numeric = temporal["numeric_value"].notna()
    numeric = temporal.loc[is_numeric].copy()
    other   = temporal.loc[~is_numeric].copy()

    stats = numeric.groupby("ConceptName")["numeric_value"].agg(
        ["count", "mean", "std"]
    ).reset_index()
    stats["std"] = stats["std"].fillna(1.0)
    stats.loc[stats["std"] <= 0, "std"] = 1.0
    # Concepts whose train support is zero are unreachable here (groupby
    # already drops them), but keep the guard explicit.
    stats = stats.loc[stats["count"] > 0]

    numeric = numeric.merge(stats[["ConceptName", "mean", "std"]],
                            on="ConceptName", how="inner")
    z = (numeric["numeric_value"] - numeric["mean"]) / numeric["std"]
    # `right=False` puts the upper edge of each bin in the next bin — z = 0.5
    # exactly lands in SLIGHTLY_HIGH, not NORMAL.  The cohort-level fractions
    # still hit the expected ~38 % NORMAL because the boundary mass is small.
    numeric["bin"] = pd.cut(z, bins=BIN_EDGES, labels=BIN_LABELS, right=False)

    # `pd.cut` returns a Categorical; cast to str so concat below stays clean.
    numeric["ConceptName"] = (
        numeric["ConceptName"].astype(str) + "_STD_" + numeric["bin"].astype(str)
    )
    numeric["Value"] = numeric["bin"].astype(str)

    out_cols = ["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"]
    return (
        pd.concat([numeric[out_cols], other[out_cols]], ignore_index=True),
        stats,
    )


def collapse_intervals(rows: pd.DataFrame, max_gap_hours: float = COLLAPSE_HOURS) -> pd.DataFrame:
    """Per (patient, ConceptName), merge consecutive rows within 24 h.

    Multi-row chains collapse correctly because the 24 h rule is checked
    pairwise against the previous observation: e.g. observations at
    t = 0, 20 h, 40 h with the same ConceptName produce one merged event
    (gaps 20 h, 20 h both ≤ 24 h), whereas t = 0, 20 h, 50 h produce two
    (the 30 h gap breaks the chain).

    Output schema: ``StartDateTime`` is the first observation's start,
    ``EndDateTime`` is the last observation's start (interval semantics —
    the source EndDateTime column is not relied on because it may be NaT
    for point-in-time events in the raw file).
    """
    rows = rows.copy()
    rows["StartDateTime"] = pd.to_datetime(rows["StartDateTime"])
    rows = rows.sort_values(["PatientId", "ConceptName", "StartDateTime"]).reset_index(drop=True)

    prev_start = rows.groupby(["PatientId", "ConceptName"])["StartDateTime"].shift()
    gap_hours  = (rows["StartDateTime"] - prev_start).dt.total_seconds() / 3600.0
    new_group  = gap_hours.isna() | (gap_hours > max_gap_hours)
    rows["group_id"] = new_group.cumsum()

    agg = rows.groupby(
        ["PatientId", "ConceptName", "group_id"], sort=False,
    ).agg(
        StartDateTime=("StartDateTime", "first"),
        EndDateTime  =("StartDateTime", "last"),
        Value        =("Value",         "first"),
    ).reset_index().drop(columns=["group_id"])
    return agg[["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"]]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(final: pd.DataFrame,
             pre_collapse_count: int,
             stats: pd.DataFrame) -> None:
    print("\n[validate] ============================================================")
    print(f"[validate] rows pre-collapse:  {pre_collapse_count:,}")
    print(f"[validate] rows post-collapse: {len(final):,}  "
          f"(reduction: {1.0 - len(final) / max(pre_collapse_count, 1):.1%})")

    print("\n[validate] per-outcome positive-patient prevalence "
          "(over all patients in the output):")
    n_patients = final["PatientId"].nunique()
    for name in sorted(set(ALL_OUTCOME_TOKENS)):
        pos = final.loc[final["ConceptName"] == name, "PatientId"].nunique()
        print(f"[validate]   {name:<36s}  n_pos_patients={pos:>5d}  "
              f"prevalence={pos / max(n_patients, 1):.3%}")

    print("\n[validate] per-concept bin distribution (post-binning; expected "
          "under N(0,1) -- NORMAL ~= 38 %, +-0.5sigma to +-1sigma tails "
          "~= 15 % each, +-1sigma to +-2sigma tails ~= 14 % each, "
          "+-2sigma outliers ~= 2.3 % each):")
    binned = final.loc[final["ConceptName"].str.contains(
        r"_STD_(?:" + "|".join(BIN_LABELS) + r")$", regex=True
    )]
    if len(binned):
        # Recover original concept name by stripping the trailing _STD_<bin>.
        m = binned["ConceptName"].str.extract(
            r"^(.*?)_STD_(" + "|".join(BIN_LABELS) + r")$"
        )
        binned = binned.assign(_base=m[0], _bin=m[1])
        dist = binned.groupby(["_base", "_bin"]).size().unstack(fill_value=0)
        dist_frac = dist.div(dist.sum(axis=1), axis=0)
        order = dist.sum(axis=1).sort_values(ascending=False).index
        cols = [c for c in BIN_LABELS if c in dist_frac.columns]
        # Top-15 most-populated concepts so the log stays readable.
        for concept in list(order)[:15]:
            row = dist_frac.loc[concept, cols]
            pretty = "  ".join(f"{c[:5]}={row.get(c, 0.0):.1%}" for c in cols)
            print(f"[validate]   {concept:<32s}  {pretty}")
    print(f"\n[validate] {len(stats):,} concepts bucketed; "
          f"{final['ConceptName'].nunique():,} unique ConceptName tokens in output.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not INPUT_TEMPORAL_CSV.exists():
        raise SystemExit(
            f"Missing {INPUT_TEMPORAL_CSV}. Place mimic-iv-input-data.csv "
            "alongside the autoresearch temporal_data.csv (e.g. by copying "
            "from med-transformers-baseline/data/)."
        )

    print(f"[load] reading {INPUT_TEMPORAL_CSV} ...")
    temporal = pd.read_csv(INPUT_TEMPORAL_CSV)
    context  = pd.read_csv(INPUT_CONTEXT_CSV)
    temporal["StartDateTime"] = pd.to_datetime(temporal["StartDateTime"])
    if "EndDateTime" in temporal.columns:
        temporal["EndDateTime"] = pd.to_datetime(temporal["EndDateTime"], errors="coerce")
    else:
        temporal["EndDateTime"] = pd.NaT
    print(f"[load] {len(temporal):,} temporal rows, "
          f"{temporal['PatientId'].nunique():,} patients.")

    print("[bounds] applying Mediator glucose / creatinine bounds...")
    temporal = apply_value_bounds(temporal)

    print("[admission] computing minute offsets, dropping pre-admission rows...")
    temporal = compute_minute_offsets(temporal)

    print("[outcomes] synthesising Mediator HYPER/HYPO/SEVERE/KIDNEY events...")
    synthesised_outcomes = synthesize_outcome_events(temporal)
    print(f"[outcomes] {len(synthesised_outcomes):,} synthesised rows.")

    print("[rename] renaming raw outcome / admission rows to canonical names...")
    temporal = rename_passthrough_outcomes_inplace(temporal)

    # Synthesised outcome rows are not in the raw stream — append them now,
    # before binning.  Their Value="True" is non-numeric so the binner will
    # pass them through unchanged.
    out_cols = ["PatientId", "ConceptName", "StartDateTime", "EndDateTime", "Value"]
    temporal = pd.concat(
        [temporal[out_cols], synthesised_outcomes.reindex(columns=out_cols)],
        ignore_index=True,
    )

    print("[binning] std-binning numeric-Value rows; categorical rows "
          "pass through...")
    binned, stats = std_bin_numeric_values(temporal)

    print(f"[collapse] merging consecutive intervals within "
          f"{COLLAPSE_HOURS:.0f} h ...")
    pre_collapse = len(binned)
    final = collapse_intervals(binned, max_gap_hours=COLLAPSE_HOURS)
    final = final.sort_values(
        ["PatientId", "StartDateTime", "ConceptName"]
    ).reset_index(drop=True)

    validate(final, pre_collapse, stats)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[write] {OUTPUT_TEMPORAL_CSV}  ({len(final):,} rows)")
    final.to_csv(OUTPUT_TEMPORAL_CSV, index=False)
    shutil.copy2(INPUT_CONTEXT_CSV, OUTPUT_CONTEXT_CSV)
    print(f"[write] {OUTPUT_CONTEXT_CSV}  (copied verbatim)")
    print("[done]")


if __name__ == "__main__":
    main()
