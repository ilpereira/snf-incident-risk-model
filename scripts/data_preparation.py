"""
data_preparation.py
──────────────────────
Builds the feature matrix from raw parquet files.

Run:  python data_preparation.py

Outputs
───────
  outputs/features.parquet  —  one row per (resident_id, obs_date), ~250 feature columns
  outputs/labels.parquet    —  parallel file with target columns + metadata for CV grouping

Design notes
────────────
• Observation spine:  one row per resident per 7-day step, covering the period where
  we have enough clinical history (≥14 days lookback) and enough future horizon
  (≥30 days before data cutoff) to form valid labels.

• Temporal integrity:  every feature is computed strictly from data BEFORE obs_date.
  Labels are computed from data in (obs_date, obs_date + 30 days].
  Residents discharged or deceased within the label window are handled by labelling
  on events that occurred before discharge (they are not excluded).

• Rolling window strategy:
    1. Resample each time-series table to daily granularity per resident.
    2. Use pandas groupby().rolling() — vectorised, handles date gaps correctly.
    3. Merge daily-level rolling stats to the weekly observation spine via exact date match.
  This avoids a quadratic cross-join while staying fully vectorised.
"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── project imports ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, OUTPUT_DIR,
    DATA_START, DATA_END, DATA_AVAILABILITY_END,
    LABEL_HORIZONS, MIN_HISTORY_DAYS, OBS_STEP_DAYS,
    WINDOWS, TARGETS,
    COMORBIDITY_MAP, ADL_DOMAIN_MAP, GG_MOBILITY_TASKS, GG_SELFCARE_TASKS,
    VITAL_TYPES, VITAL_COL_NAMES, VITAL_FLAGS,
    PSYCHOTROPIC_PATTERNS, OPIOID_PATTERNS, DIURETIC_PATTERNS, ANTICOAGULANT_PATTERNS,
    HIGH_RISK_DOC_TAGS, charlson_age_points,
)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all parquet files into a dict keyed by table name."""
    tables = [
        "residents", "diagnoses", "adl_responses", "gg_responses", "vitals",
        "medications", "document_tags", "lab_reports", "incidents", "injuries",
        "hospital_transfers", "hospital_admissions", "factors",
        "needs", "care_plans", "physician_orders", "therapy_tracks",
    ]
    data = {}
    for t in tables:
        path = data_dir / f"{t}.parquet"
        if path.exists():
            data[t] = pd.read_parquet(path)
            print(f"  ✓ {t:30s} {data[t].shape}")
        else:
            print(f"  ✗ {t} not found — skipping")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# 2. OBSERVATION SPINE
# ══════════════════════════════════════════════════════════════════════════════

def build_observation_spine(residents: pd.DataFrame) -> pd.DataFrame:
    """
    Generate one row per (resident_id, obs_date) for every resident
    during their active stay within the valid modelling window.

    Valid window per resident:
        start = max(admission_date + MIN_HISTORY_DAYS, DATA_START)
        end   = min(discharge_date, deceased_date, DATA_END)

    DATA_END is set conservatively (30 days before data availability end) so that
    the shortest-horizon target (fall_30d, 30 days) always has valid labels.
    Longer-horizon targets (rth_60d, wound_60d, 60 days) mask rows near the end
    where the label window would extend past DATA_AVAILABILITY_END.
    """
    data_start  = pd.Timestamp(DATA_START)
    data_end    = pd.Timestamp(DATA_END)
    # Use the shortest horizon so the spine is as large as possible;
    # longer-horizon labels are masked per-target in label_observations()
    min_horizon = pd.Timedelta(days=min(LABEL_HORIZONS.values()))
    hist_delta  = pd.Timedelta(days=MIN_HISTORY_DAYS)

    res = residents.copy()
    res["admission_date"]  = pd.to_datetime(res["admission_date"], utc=False).dt.tz_localize(None)
    res["discharge_date"]  = pd.to_datetime(res["discharge_date"], utc=False).dt.tz_localize(None)
    res["deceased_date"]   = pd.to_datetime(res["deceased_date"],  utc=False).dt.tz_localize(None)
    res["date_of_birth"]   = pd.to_datetime(res["date_of_birth"],  utc=False).dt.tz_localize(None)

    rows = []
    for _, r in res.iterrows():
        # Earliest valid obs date for this resident
        latest_start = max(r["admission_date"] + hist_delta, data_start)

        # Latest valid obs date: we need a full prediction window after it
        end_bounds = [data_end - min_horizon]
        if pd.notna(r["discharge_date"]):
            end_bounds.append(r["discharge_date"])
        if pd.notna(r["deceased_date"]):
            end_bounds.append(r["deceased_date"])
        earliest_end = min(end_bounds)

        if latest_start > earliest_end:
            continue

        for obs_date in pd.date_range(latest_start, earliest_end, freq=f"{OBS_STEP_DAYS}D"):
            rows.append({
                "resident_id":   r["resident_id"],
                "facility_id":   r["facility_id"],
                "obs_date":      obs_date.normalize(),
                "admission_date": r["admission_date"],
                "date_of_birth": r["date_of_birth"],
                "discharge_date": r.get("discharge_date"),
                "deceased_date":  r.get("deceased_date"),
                "outpatient":     r.get("outpatient", False),
            })

    spine = pd.DataFrame(rows)
    print(f"  Spine: {len(spine):,} rows | {spine['resident_id'].nunique():,} residents | "
          f"{spine['facility_id'].nunique()} facilities")
    return spine


# ══════════════════════════════════════════════════════════════════════════════
# 3. LABEL GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def label_observations(spine: pd.DataFrame, data: dict) -> pd.DataFrame:
    """
    Create binary outcome columns on the spine using per-target prediction horizons.

    fall_30d  : any Fall incident in (obs_date, obs_date + 30d]
    rth_60d   : any unplanned hospital transfer in (obs_date, obs_date + 60d]
    wound_60d : any Wound incident in (obs_date, obs_date + 60d]

    Rows where obs_date + horizon > DATA_AVAILABILITY_END cannot have complete
    labels and are marked NaN for that target.  The training script drops NaN
    labels per target before fitting.
    """
    data_avail_end = pd.Timestamp(DATA_AVAILABILITY_END)

    # ── Falls & Wounds ────────────────────────────────────────────────────
    incidents = data["incidents"].copy()
    incidents = incidents[incidents["strikeout"] == False]
    incidents["occurred_at"] = pd.to_datetime(incidents["occurred_at"], utc=False).dt.tz_localize(None)

    # ── RTH: unplanned transfers only ─────────────────────────────────────
    transfers = data["hospital_transfers"].copy()
    transfers["effective_date"] = pd.to_datetime(transfers["effective_date"], utc=False).dt.tz_localize(None)
    unplanned = transfers[
        (transfers["planned_flag"].fillna(False) == False) &
        (transfers["transfer_outcome"].fillna("").str.contains("Admitted", na=False))
    ]

    def _event_in_window(
        spine_df: pd.DataFrame,
        events_df: pd.DataFrame,
        event_date_col: str,
        horizon_days: int,
        filter_mask=None,
    ) -> pd.Series:
        """
        Vectorised: for each spine row, flag whether any matching event falls in
        (obs_date, obs_date + horizon_days].  Rows where obs_date + horizon exceeds
        DATA_AVAILABILITY_END are returned as NaN (label cannot be verified).
        """
        ev = events_df[filter_mask] if filter_mask is not None else events_df
        merged = spine_df[["resident_id", "obs_date"]].merge(
            ev[["resident_id", event_date_col]], on="resident_id", how="left"
        )
        days_ahead = (merged[event_date_col] - merged["obs_date"]).dt.days
        in_window  = (days_ahead > 0) & (days_ahead <= horizon_days)
        flag = (merged[in_window]
                .groupby(["resident_id", "obs_date"])
                .size().gt(0).astype(float)
                .reset_index(name="flag"))
        result = spine_df.merge(flag, on=["resident_id", "obs_date"], how="left")["flag"].fillna(0)

        # Censor rows whose label window extends past data availability
        censored = spine_df["obs_date"] + pd.Timedelta(days=horizon_days) > data_avail_end
        result[censored.values] = float("nan")
        return result

    spine = spine.copy()
    spine["fall_30d"]  = _event_in_window(
        spine, incidents, "occurred_at", horizon_days=30,
        filter_mask=(incidents["incident_type"] == "Fall")
    )
    spine["wound_60d"] = _event_in_window(
        spine, incidents, "occurred_at", horizon_days=60,
        filter_mask=(incidents["incident_type"] == "Wound")
    )
    spine["rth_60d"]   = _event_in_window(
        spine, unplanned, "effective_date", horizon_days=60
    )

    for t in ["fall_30d", "rth_60d", "wound_60d"]:
        valid = spine[t].notna()
        print(f"  {t}: prevalence={spine.loc[valid, t].mean():.2%}  "
              f"valid rows={valid.sum():,}  censored={(~valid).sum():,}")
    return spine


# ══════════════════════════════════════════════════════════════════════════════
# 4. HELPER: ROLLING WINDOW JOIN
# ══════════════════════════════════════════════════════════════════════════════

def _rolling_on_daily(
    events: pd.DataFrame,
    resident_col: str,
    date_col: str,
    value_cols: list[str],
    windows: list[int],
    prefix: str,
    aggs: list[str] = ("mean", "std"),
) -> pd.DataFrame:
    """
    Vectorised rolling feature computation:
      1. Aggregate events to daily granularity (mean per resident per day).
      2. Sort by (resident, date) and use groupby().rolling() for each window.
      3. Return a tidy dataframe indexed by (resident_id, date) ready to merge.

    NaN propagation is intentional — missing data for a resident in a window
    is represented as NaN, which XGBoost handles natively.
    """
    df = events.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.normalize()
    # Coerce value columns to float (some sources send strings, e.g. ADL responses)
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Step 1: daily mean per resident
    daily = (
        df.groupby([resident_col, date_col])[value_cols]
        .mean()
        .reset_index()
        .sort_values([resident_col, date_col])
    )

    # Step 2: compute rolling stats using pandas groupby+rolling (vectorised)
    daily = daily.set_index(date_col)
    result_parts = [daily[[resident_col]].reset_index()]

    for w in windows:
        rolled = (
            daily.groupby(resident_col)[value_cols]
            .rolling(f"{w}D", min_periods=1)
        )
        for agg in aggs:
            agg_df = getattr(rolled, agg)().reset_index()
            agg_df.columns = [resident_col, date_col] + [
                f"{prefix}_{c}_{agg}_{w}d" for c in value_cols
            ]
            result_parts.append(agg_df.drop(columns=[resident_col, date_col]))

    out = pd.concat(result_parts, axis=1)
    # Keep (resident, date) as the merge key
    out[date_col] = daily.reset_index()[date_col].values
    return out.rename(columns={resident_col: "resident_id", date_col: "date"})


def _join_rolling_to_spine(spine: pd.DataFrame, rolling_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join rolling features to spine on (resident_id, obs_date == date)."""
    return spine.merge(
        rolling_df.rename(columns={"date": "obs_date"}),
        on=["resident_id", "obs_date"],
        how="left",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. FEATURE BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

# ── 5a. Static / demographic ──────────────────────────────────────────────

def feature_static(spine: pd.DataFrame) -> pd.DataFrame:
    """Age at observation, length of stay, demographic flags."""
    s = spine.copy()
    s["age_at_obs"]  = (s["obs_date"] - s["date_of_birth"]).dt.days / 365.25
    s["los_days"]    = (s["obs_date"] - s["admission_date"]).dt.days
    s["age_bucket"]  = pd.cut(
        s["age_at_obs"],
        bins=[0, 65, 75, 85, 95, 130],
        labels=["<65", "65-74", "75-84", "85-94", "95+"],
    ).astype("category")
    s["is_outpatient"] = s["outpatient"].astype(int)
    return s


# ── 5b. Comorbidities & Charlson Score ───────────────────────────────────

def feature_comorbidities(spine: pd.DataFrame, diagnoses: pd.DataFrame) -> pd.DataFrame:
    """
    For each observation, flag which comorbidities are active (onset ≤ obs_date,
    resolved_at is null or > obs_date) and compute the Charlson Comorbidity Index.
    """
    dx = diagnoses.copy()
    dx = dx[dx["strikeout"] == False]
    dx["onset_at"]      = pd.to_datetime(dx["onset_at"],    utc=False).dt.tz_localize(None).dt.normalize()
    dx["resolved_at"]   = pd.to_datetime(dx["resolved_at"], utc=False).dt.tz_localize(None).dt.normalize()
    dx["icd_10_code"]   = dx["icd_10_code"].fillna("").str.upper().str.replace(".", "", regex=False)

    # Merge spine with all diagnoses for matching residents
    merged = spine[["resident_id", "obs_date"]].merge(
        dx[["resident_id", "icd_10_code", "onset_at", "resolved_at"]],
        on="resident_id", how="left"
    )
    # Active at obs_date: onset ≤ obs_date AND (resolved_at is null OR resolved_at > obs_date)
    active = merged[
        (merged["onset_at"].isna() | (merged["onset_at"] <= merged["obs_date"])) &
        (merged["resolved_at"].isna() | (merged["resolved_at"] > merged["obs_date"]))
    ]

    flag_cols = {}
    cci_weights = {}

    for feat_name, (prefixes, cci_w) in COMORBIDITY_MAP.items():
        pattern = "|".join(f"^{p}" for p in prefixes)
        matched_residents_dates = (
            active[active["icd_10_code"].str.match(pattern, na=False)]
            .groupby(["resident_id", "obs_date"])
            .size()
            .gt(0)
            .astype(int)
            .reset_index(name=feat_name)
        )
        flag_cols[feat_name] = matched_residents_dates
        cci_weights[feat_name] = cci_w

    # Merge all flag columns to spine
    result = spine.copy()
    for feat_name, flag_df in flag_cols.items():
        result = result.merge(flag_df, on=["resident_id", "obs_date"], how="left")
        result[feat_name] = result[feat_name].fillna(0).astype(int)

    # Charlson Comorbidity Index
    dx_cols = list(COMORBIDITY_MAP.keys())
    charlson = sum(result[c] * w for c, (_, w) in COMORBIDITY_MAP.items())
    result["charlson_score"] = charlson + result["age_at_obs"].apply(charlson_age_points)
    result["n_comorbidities"] = result[dx_cols].sum(axis=1)

    return result


# ── 5c. ADL functional status ─────────────────────────────────────────────

def feature_adl(spine: pd.DataFrame, adl_responses: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling mean/std of ADL self-performance scores per domain.
    Higher score = more dependent. Trend features capture trajectory.
    """
    adl = adl_responses[adl_responses["category"] == "Self-Performance"].copy()
    adl["assessment_date"] = pd.to_datetime(adl["assessment_date"], utc=False).dt.tz_localize(None).dt.normalize()

    result = spine.copy()

    for domain_key, activity_name in ADL_DOMAIN_MAP.items():
        domain_df = adl[adl["activity"] == activity_name][
            ["resident_id", "assessment_date", "response"]
        ].rename(columns={"assessment_date": "date", "response": "value"})

        if domain_df.empty:
            continue

        rolling = _rolling_on_daily(
            domain_df, "resident_id", "date", ["value"],
            windows=[7, 30, 90], prefix=f"adl_{domain_key}", aggs=["mean", "std"]
        )
        result = _join_rolling_to_spine(result, rolling)

    # Summary across all domains (total ADL burden)
    all_adl = adl[adl["activity"].isin(ADL_DOMAIN_MAP.values())][
        ["resident_id", "assessment_date", "response"]
    ].rename(columns={"assessment_date": "date", "response": "value"})

    if not all_adl.empty:
        rolling_total = _rolling_on_daily(
            all_adl, "resident_id", "date", ["value"],
            windows=[7, 30, 90], prefix="adl_total", aggs=["mean", "std"]
        )
        result = _join_rolling_to_spine(result, rolling_total)

    # Trend: 7-day mean vs 30-day mean (positive = recent decline)
    if "adl_total_value_mean_7d" in result.columns and "adl_total_value_mean_30d" in result.columns:
        result["adl_trend_7v30"] = result["adl_total_value_mean_7d"] - result["adl_total_value_mean_30d"]

    return result


# ── 5d. GG mobility / self-care ───────────────────────────────────────────

def feature_gg(spine: pd.DataFrame, gg_responses: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling features from CMS Section GG assessments.
    Invert response_code so higher = more dependent (matches ADL direction).
    """
    gg = gg_responses.copy()
    gg = gg[gg["response_code"].notna() & ~gg["response_code"].isin([88, 9])]
    gg["created_at"] = pd.to_datetime(gg["created_at"], utc=False).dt.tz_localize(None).dt.normalize()
    gg["dependency"] = 7.0 - gg["response_code"].clip(1, 6)  # invert scale

    result = spine.copy()

    # Mobility aggregate
    mob = gg[gg["task_group"] == "Mobility"][["resident_id", "created_at", "dependency"]].rename(
        columns={"created_at": "date", "dependency": "value"}
    )
    if not mob.empty:
        rolling = _rolling_on_daily(mob, "resident_id", "date", ["value"],
                                    windows=[7, 30], prefix="gg_mobility", aggs=["mean", "std"])
        result = _join_rolling_to_spine(result, rolling)

    # Self-care aggregate
    sc = gg[gg["task_group"] == "Self Care"][["resident_id", "created_at", "dependency"]].rename(
        columns={"created_at": "date", "dependency": "value"}
    )
    if not sc.empty:
        rolling = _rolling_on_daily(sc, "resident_id", "date", ["value"],
                                    windows=[7, 30], prefix="gg_selfcare", aggs=["mean", "std"])
        result = _join_rolling_to_spine(result, rolling)

    # Specific high-value mobility tasks
    for task in GG_MOBILITY_TASKS:
        task_df = gg[gg["task_name"] == task][["resident_id", "created_at", "dependency"]].rename(
            columns={"created_at": "date", "dependency": "value"}
        )
        if task_df.empty:
            continue
        slug = re.sub(r"[^a-z0-9]+", "_", task.lower()).strip("_")
        rolling = _rolling_on_daily(task_df, "resident_id", "date", ["value"],
                                    windows=[7, 30], prefix=f"gg_{slug}", aggs=["mean"])
        result = _join_rolling_to_spine(result, rolling)

    return result


# ── 5e. Vital signs ───────────────────────────────────────────────────────

def feature_vitals(spine: pd.DataFrame, vitals: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling mean/std per vital type (7 and 30 day windows).
    Also computes clinical threshold flags: hypoxia, hypertensive urgency, etc.
    Weight % change captures malnutrition / fluid shifts.
    """
    v = vitals[vitals["strikeout"] == False].copy()
    v["measured_at"] = pd.to_datetime(v["measured_at"], utc=False).dt.tz_localize(None).dt.normalize()

    result = spine.copy()

    for vtype in VITAL_TYPES:
        col_name = VITAL_COL_NAMES[vtype]
        vdf = v[v["vital_type"] == vtype][["resident_id", "measured_at", "value"]].rename(
            columns={"measured_at": "date", "value": col_name}
        )
        if vdf.empty:
            continue
        rolling = _rolling_on_daily(
            vdf, "resident_id", "date", [col_name],
            windows=[7, 30], prefix="vital", aggs=["mean", "std", "min", "max"]
        )
        result = _join_rolling_to_spine(result, rolling)

    # Coefficient of variation for BP (instability predicts both falls and RTH)
    bp_mean = result.get("vital_bp_sys_mean_30d")
    bp_std  = result.get("vital_bp_sys_std_30d")
    if bp_mean is not None and bp_std is not None:
        result["vital_bp_sys_cv_30d"] = (bp_std / bp_mean.replace(0, np.nan)).fillna(0)

    # Binary clinical flag features (any reading crossing threshold in last 7 days)
    for flag_name, (vtype, op, thresh) in VITAL_FLAGS.items():
        col = f"vital_{VITAL_COL_NAMES[vtype]}"
        # Using the 7-day min/max captured above
        ref_col = f"{col}_min_7d" if op == "<" else f"{col}_max_7d"
        if ref_col in result.columns:
            if op == "<":
                result[f"flag_{flag_name}"] = (result[ref_col] < thresh).astype(int)
            else:
                result[f"flag_{flag_name}"] = (result[ref_col] >= thresh).astype(int)

    # Weight percentage change (30-day window vs single recent reading)
    w_mean_30 = result.get("vital_weight_mean_30d")
    w_max_7   = result.get("vital_weight_max_7d")
    if w_mean_30 is not None and w_max_7 is not None:
        result["vital_weight_pct_change_30d"] = (
            (w_max_7 - w_mean_30) / w_mean_30.replace(0, np.nan) * 100
        ).fillna(0)

    return result


# ── 5f. Medication profile ────────────────────────────────────────────────

def feature_medications(spine: pd.DataFrame, medications: pd.DataFrame) -> pd.DataFrame:
    """
    Polypharmacy count, fall-risk drug classes, and medication adherence metrics.
    All computed over a 30-day lookback (medications don't have structured end dates).
    """
    meds = medications.copy()
    meds["scheduled_at"] = pd.to_datetime(meds["scheduled_at"], utc=False).dt.tz_localize(None).dt.normalize()

    desc_lower = meds["description"].fillna("").str.lower()

    def _flag_class(patterns: list[str]) -> pd.Series:
        combined = "|".join(patterns)
        return desc_lower.str.contains(combined, regex=True, na=False)

    meds["is_psychotropic"]    = _flag_class(PSYCHOTROPIC_PATTERNS).astype(int)
    meds["is_opioid"]          = _flag_class(OPIOID_PATTERNS).astype(int)
    meds["is_diuretic"]        = _flag_class(DIURETIC_PATTERNS).astype(int)
    meds["is_anticoagulant"]   = _flag_class(ANTICOAGULANT_PATTERNS).astype(int)
    meds["is_missed"]          = (meds["status"] == "Missed").astype(int)
    meds["is_late"]            = (meds["status"] == "Late").astype(int)
    meds["is_refused"]         = (meds["status"] == "Refused").astype(int)
    meds["n_doses"]            = 1  # count column

    # ── Build daily aggregates ────────────────────────────────────────────
    med_daily_value_cols = [
        "is_psychotropic", "is_opioid", "is_diuretic", "is_anticoagulant",
        "is_missed", "is_late", "is_refused", "n_doses",
    ]
    rolling = _rolling_on_daily(
        meds.rename(columns={"scheduled_at": "date"}),
        "resident_id", "date", med_daily_value_cols,
        windows=[7, 30], prefix="med", aggs=["sum", "mean"]
    )
    result = _join_rolling_to_spine(spine, rolling)

    # Unique drug count (proxy for polypharmacy) — computed separately
    window_days = 30
    label_delta = pd.Timedelta(days=window_days)

    unique_meds = []
    for res_id, res_meds in meds.groupby("resident_id"):
        res_spine = spine[spine["resident_id"] == res_id][["resident_id", "obs_date"]]
        if res_spine.empty:
            continue
        res_meds_sub = res_meds[["scheduled_at", "description"]].copy()
        cross = res_spine.assign(_k=1).merge(res_meds_sub.assign(_k=1), on="_k").drop("_k", axis=1)
        days_ago = (cross["obs_date"] - cross["scheduled_at"]).dt.days
        in_w = (days_ago >= 0) & (days_ago < window_days)
        n_unique = (cross[in_w]
                    .groupby(["resident_id", "obs_date"])["description"]
                    .nunique()
                    .reset_index(name="med_n_unique_drugs_30d"))
        unique_meds.append(n_unique)

    if unique_meds:
        unique_df = pd.concat(unique_meds, ignore_index=True)
        result = result.merge(unique_df, on=["resident_id", "obs_date"], how="left")

    # Miss rate
    n_miss  = result.get("med_is_missed_sum_30d",  pd.Series(0, index=result.index))
    n_total = result.get("med_n_doses_sum_30d",    pd.Series(1, index=result.index)).replace(0, np.nan)
    result["med_miss_rate_30d"] = (n_miss / n_total).fillna(0)

    # Polypharmacy flag (≥ 9 drugs is a common clinical threshold)
    if "med_n_unique_drugs_30d" in result.columns:
        result["med_polypharmacy_flag"] = (result["med_n_unique_drugs_30d"] >= 9).astype(int)

    return result


# ── 5g. Document tags ─────────────────────────────────────────────────────

def feature_document_tags(spine: pd.DataFrame, document_tags: pd.DataFrame) -> pd.DataFrame:
    """
    Count occurrences of high-signal NLP-derived tags in the last 30 days.
    These tags encode clinical information from unstructured progress notes.
    """
    dt = document_tags[document_tags["deleted_at"].isna()].copy()
    dt["created_at"] = pd.to_datetime(dt["created_at"], utc=False).dt.tz_localize(None).dt.normalize()

    result = spine.copy()
    window_days = 30

    for tag in HIGH_RISK_DOC_TAGS:
        tag_df = dt[dt["tag_id"].str.contains(tag, case=False, na=False)][
            ["resident_id", "created_at"]
        ].assign(n=1).rename(columns={"created_at": "date"})

        if tag_df.empty:
            continue

        rolling = _rolling_on_daily(
            tag_df, "resident_id", "date", ["n"],
            windows=[30], prefix=f"tag_{tag}", aggs=["sum"]
        )
        result = _join_rolling_to_spine(result, rolling)

    # Total high-risk tag count
    tag_sum_cols = [c for c in result.columns if c.startswith("tag_") and c.endswith("_30d")]
    if tag_sum_cols:
        result["tag_total_highrisk_30d"] = result[tag_sum_cols].fillna(0).sum(axis=1)

    return result


# ── 5h. Lab reports ───────────────────────────────────────────────────────

def feature_labs(spine: pd.DataFrame, lab_reports: pd.DataFrame) -> pd.DataFrame:
    """Count abnormal and critical lab results in the last 30 days."""
    labs = lab_reports.copy()
    labs["reported_at"] = pd.to_datetime(labs["reported_at"], utc=False).dt.tz_localize(None).dt.normalize()
    labs["is_abnormal"] = (labs["severity_status"].isin(["Abnormal", "Critical"])).astype(float)
    labs["is_critical"] = (labs["severity_status"] == "Critical").astype(float)
    labs["n_lab"]       = 1.0

    rolling = _rolling_on_daily(
        labs.rename(columns={"reported_at": "date"}),
        "resident_id", "date", ["is_abnormal", "is_critical", "n_lab"],
        windows=[30], prefix="lab", aggs=["sum"]
    )
    return _join_rolling_to_spine(spine, rolling)


# ── 5i. Prior incident history ────────────────────────────────────────────

def feature_prior_incidents(spine: pd.DataFrame, data: dict) -> pd.DataFrame:
    """
    Counts of past incidents by type, days-since-last-fall, and injury severity.
    Prior incidents are among the strongest predictors of future incidents.
    """
    incidents = data["incidents"][data["incidents"]["strikeout"] == False].copy()
    incidents["occurred_at"] = pd.to_datetime(incidents["occurred_at"], utc=False).dt.tz_localize(None).dt.normalize()

    injuries = data["injuries"].copy()

    result = spine.copy()
    window_days = 90
    windows = [30, 90]

    for inc_type, col_prefix in [("Fall", "fall"), ("Wound", "wound"), ("Altercation", "altercation")]:
        type_df = incidents[incidents["incident_type"] == inc_type][
            ["resident_id", "occurred_at"]
        ].assign(n=1.0).rename(columns={"occurred_at": "date"})

        if type_df.empty:
            continue

        rolling = _rolling_on_daily(
            type_df, "resident_id", "date", ["n"],
            windows=windows, prefix=f"prior_{col_prefix}", aggs=["sum"]
        )
        result = _join_rolling_to_spine(result, rolling)

    # Days since last fall (bounded at 999 if no prior fall)
    falls = incidents[incidents["incident_type"] == "Fall"][["resident_id", "occurred_at"]]
    merged_falls = result[["resident_id", "obs_date"]].merge(falls, on="resident_id", how="left")
    merged_falls["days_ago"] = (merged_falls["obs_date"] - merged_falls["occurred_at"]).dt.days
    merged_falls = merged_falls[(merged_falls["days_ago"] > 0)]
    days_since = (merged_falls
                  .groupby(["resident_id", "obs_date"])["days_ago"]
                  .min()
                  .reset_index(name="prior_days_since_last_fall"))
    result = result.merge(days_since, on=["resident_id", "obs_date"], how="left")
    result["prior_days_since_last_fall"] = result["prior_days_since_last_fall"].fillna(999)

    # Did last fall result in an injury?
    falls_with_injury = incidents[incidents["incident_type"] == "Fall"].merge(
        injuries[["incident_id"]].assign(had_injury=1), on="incident_id", how="left"
    )
    falls_with_injury["had_injury"] = falls_with_injury["had_injury"].fillna(0)
    most_recent_fall_injury = (
        result[["resident_id", "obs_date"]]
        .merge(falls_with_injury[["resident_id", "occurred_at", "had_injury"]], on="resident_id", how="left")
    )
    most_recent_fall_injury = most_recent_fall_injury[
        (most_recent_fall_injury["occurred_at"] < most_recent_fall_injury["obs_date"])
    ]
    last_fall_injury = (
        most_recent_fall_injury
        .sort_values("occurred_at")
        .groupby(["resident_id", "obs_date"])
        .last()[["had_injury"]]
        .reset_index()
        .rename(columns={"had_injury": "prior_last_fall_had_injury"})
    )
    result = result.merge(last_fall_injury, on=["resident_id", "obs_date"], how="left")
    result["prior_last_fall_had_injury"] = result["prior_last_fall_had_injury"].fillna(0)

    return result


# ── 5j. Prior RTH ─────────────────────────────────────────────────────────

def feature_prior_rth(spine: pd.DataFrame, hospital_transfers: pd.DataFrame) -> pd.DataFrame:
    """Counts of prior unplanned transfers; strong predictor of future RTH."""
    transfers = hospital_transfers.copy()
    transfers["effective_date"] = pd.to_datetime(transfers["effective_date"], utc=False).dt.tz_localize(None).dt.normalize()
    unplanned = transfers[transfers["planned_flag"].fillna(False) == False]
    unplanned = unplanned.assign(n=1.0).rename(columns={"effective_date": "date"})

    if unplanned.empty:
        return spine

    rolling = _rolling_on_daily(
        unplanned, "resident_id", "date", ["n"],
        windows=[30, 90], prefix="prior_rth", aggs=["sum"]
    )
    result = _join_rolling_to_spine(spine, rolling)

    # Days since last RTH
    merged = result[["resident_id", "obs_date"]].merge(
        unplanned[["resident_id", "date"]], on="resident_id", how="left"
    )
    merged["days_ago"] = (merged["obs_date"] - merged["date"]).dt.days
    merged = merged[merged["days_ago"] > 0]
    days_since = (merged
                  .groupby(["resident_id", "obs_date"])["days_ago"]
                  .min()
                  .reset_index(name="prior_rth_days_since_last"))
    result = result.merge(days_since, on=["resident_id", "obs_date"], how="left")
    result["prior_rth_days_since_last"] = result["prior_rth_days_since_last"].fillna(999)
    result["prior_rth_ever"] = (result["prior_rth_n_sum_90d"].fillna(0) > 0).astype(int)

    return result


# ── 5k. Active care needs ─────────────────────────────────────────────────

def feature_care_needs(spine: pd.DataFrame, needs: pd.DataFrame) -> pd.DataFrame:
    """
    Active care plan needs at obs_date (initiated ≤ obs_date, not yet resolved).
    When a clinician flags a Fall or Wound need, that is a validated risk signal.
    """
    n = needs[needs["strikeout"] == False].copy()
    n["initiated_at"] = pd.to_datetime(n["initiated_at"], utc=False).dt.tz_localize(None).dt.normalize()
    n["resolved_at"]  = pd.to_datetime(n["resolved_at"],  utc=False).dt.tz_localize(None).dt.normalize()

    merged = spine[["resident_id", "obs_date"]].merge(
        n[["resident_id", "need_category", "initiated_at", "resolved_at"]],
        on="resident_id", how="left"
    )
    active_needs = merged[
        (merged["initiated_at"] <= merged["obs_date"]) &
        (merged["resolved_at"].isna() | (merged["resolved_at"] > merged["obs_date"]))
    ]

    result = spine.copy()
    for category, col_name in [("Fall", "need_fall_active"), ("Wound", "need_wound_active"),
                                ("Nutrition", "need_nutrition_active")]:
        cat_df = (active_needs[active_needs["need_category"] == category]
                  .groupby(["resident_id", "obs_date"])
                  .size().gt(0).astype(int).reset_index(name=col_name))
        result = result.merge(cat_df, on=["resident_id", "obs_date"], how="left")
        result[col_name] = result[col_name].fillna(0).astype(int)

    return result


# ── 5l. Therapy enrollment ────────────────────────────────────────────────

def feature_therapy(spine: pd.DataFrame, therapy_tracks: pd.DataFrame) -> pd.DataFrame:
    """Active PT / OT enrollment at obs_date."""
    th = therapy_tracks.copy()
    th["start_at"] = pd.to_datetime(th["start_at"], utc=False).dt.tz_localize(None).dt.normalize()
    th["end_at"]   = pd.to_datetime(th["end_at"],   utc=False).dt.tz_localize(None).dt.normalize()

    merged = spine[["resident_id", "obs_date"]].merge(
        th[["resident_id", "discipline", "start_at", "end_at"]],
        on="resident_id", how="left"
    )
    active = merged[
        (merged["start_at"] <= merged["obs_date"]) &
        (merged["end_at"].isna() | (merged["end_at"] >= merged["obs_date"]))
    ]

    result = spine.copy()
    for discipline, col_name in [("PT", "therapy_pt_active"), ("OT", "therapy_ot_active")]:
        disc_df = (active[active["discipline"] == discipline]
                   .groupby(["resident_id", "obs_date"])
                   .size().gt(0).astype(int).reset_index(name=col_name))
        result = result.merge(disc_df, on=["resident_id", "obs_date"], how="left")
        result[col_name] = result[col_name].fillna(0).astype(int)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)

    print("\n[1] Loading raw data …")
    data = load_raw_data(data_dir)

    required = ["residents"]
    missing = [t for t in required if t not in data]
    if missing:
        raise FileNotFoundError(
            f"Required table(s) not found in {data_dir}: {missing}. "
            "Check that DATA_DIR in config.py points to the directory containing the parquet files."
        )

    print("\n[2] Building observation spine …")
    spine = build_observation_spine(data["residents"])

    print("\n[3] Generating labels …")
    spine = label_observations(spine, data)

    print("\n[4] Computing features …")

    print("  [4a] Static / demographic")
    spine = feature_static(spine)

    print("  [4b] Comorbidities & Charlson CCI")
    spine = feature_comorbidities(spine, data["diagnoses"])

    print("  [4c] ADL functional status")
    spine = feature_adl(spine, data["adl_responses"])

    print("  [4d] GG mobility / self-care")
    spine = feature_gg(spine, data["gg_responses"])

    print("  [4e] Vital signs (rolling)")
    spine = feature_vitals(spine, data["vitals"])

    print("  [4f] Medication profile")
    spine = feature_medications(spine, data["medications"])

    print("  [4g] Document tags (NLP signals)")
    spine = feature_document_tags(spine, data["document_tags"])

    print("  [4h] Lab abnormality counts")
    spine = feature_labs(spine, data["lab_reports"])

    print("  [4i] Prior incident history")
    spine = feature_prior_incidents(spine, data)

    print("  [4j] Prior RTH history")
    spine = feature_prior_rth(spine, data["hospital_transfers"])

    print("  [4k] Active care needs")
    spine = feature_care_needs(spine, data["needs"])

    print("  [4l] Therapy enrollment")
    spine = feature_therapy(spine, data["therapy_tracks"])

    # ── Separate labels / metadata from features ──────────────────────────
    meta_cols   = ["resident_id", "facility_id", "obs_date",
                   "admission_date", "date_of_birth", "discharge_date",
                   "deceased_date", "outpatient"]
    label_cols  = TARGETS
    drop_cols   = meta_cols + label_cols + ["age_bucket"]  # age_bucket saved separately

    feature_cols = [c for c in spine.columns if c not in drop_cols]

    feature_df = spine[["resident_id", "facility_id", "obs_date"] + feature_cols].copy()
    label_df   = spine[["resident_id", "facility_id", "obs_date"] + label_cols].copy()

    # ── Encode facility as integer category ───────────────────────────────
    fac_map = {f: i for i, f in enumerate(feature_df["facility_id"].unique())}
    feature_df["facility_id_enc"] = feature_df["facility_id"].map(fac_map)
    feature_df = feature_df.drop(columns=["facility_id", "obs_date"])

    # Persist facility map so scoring script can handle unseen facilities
    import json
    fac_map_path = OUTPUT_DIR / "models"
    fac_map_path.mkdir(parents=True, exist_ok=True)
    with open(fac_map_path / "facility_map.json", "w") as f:
        json.dump(fac_map, f)

    print(f"\n[5] Feature matrix: {feature_df.shape[0]:,} rows × {feature_df.shape[1]} cols")
    print(f"    Missing values: {feature_df.isnull().sum().sum():,} "
          f"({feature_df.isnull().mean().mean():.1%} of cells)")
    print(f"    Label NaN counts (censored rows per target):")
    for t in TARGETS:
        if t in label_df.columns:
            print(f"      {t}: {label_df[t].isna().sum():,} censored / {len(label_df):,} total")

    feat_path  = output_dir / "features.parquet"
    label_path = output_dir / "labels.parquet"
    feature_df.to_parquet(feat_path,  index=False)
    label_df.to_parquet(label_path,   index=False)
    print(f"\n  ✓ Features → {feat_path}")
    print(f"  ✓ Labels   → {label_path}")

    return feature_df, label_df


if __name__ == "__main__":
    import time
    t0 = time.time()
    build_feature_matrix(DATA_DIR, OUTPUT_DIR)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
