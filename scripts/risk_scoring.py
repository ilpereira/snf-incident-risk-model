"""
risk_scoring.py
──────────────────────────────────────
Scores currently-active residents and produces a ranked risk dashboard.

For each active resident:
  • 0-1 risk score per outcome (fall, RTH, wound)
  • Overall risk tier (High / Medium / Low)
  • Top 3 SHAP-driven reasons per outcome
  • Expected 30-day claim cost

Output
──────
  outputs/metrics/risk_dashboard_{date}.csv
  outputs/metrics/risk_dashboard_{date}.html

Run:  python risk_scoring.py [YYYY-MM-DD]
      (defaults to today if no date given)
"""

import json
import pickle
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, OUTPUT_DIR, TARGETS, LABEL_HORIZONS,
    DASHBOARD_ACTIVE_TARGETS,
)
from data_preparation import (
    load_raw_data,
    feature_static, feature_comorbidities, feature_adl, feature_gg,
    feature_vitals, feature_medications, feature_document_tags,
    feature_labs, feature_prior_incidents, feature_prior_rth,
    feature_care_needs, feature_therapy,
)

warnings.filterwarnings("ignore")

CLAIM_COSTS = {"fall_30d": 3_500, "rth_60d": 20_000, "wound_60d": 4_000}

# Percentile thresholds for tier assignment.
# Top 25% of scored residents = High, next 25% = Medium, bottom 50% = Low.
# Percentile-based tiers prevent alert fatigue regardless of the absolute
# score level — a poorly calibrated model that inflates all scores to 0.45
# would tier everyone "High" with a fixed threshold, whereas percentile tiers
# always produce a sensible split.
RISK_TIER_PERCENTILES = {"high": 75, "medium": 50}

TARGET_LABELS = {
    "fall_30d":  "Fall (30d)",
    "rth_60d":   "Return-to-hospital (60d)",
    "wound_60d": "Wound (60d)",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. SCORING SPINE
# ══════════════════════════════════════════════════════════════════════════════

def build_scoring_spine(residents: pd.DataFrame, score_date: pd.Timestamp) -> pd.DataFrame:
    """One row per currently-active resident (not discharged, not deceased)."""
    res = residents.copy()
    for col in ["admission_date", "discharge_date", "deceased_date", "date_of_birth"]:
        res[col] = pd.to_datetime(res[col], utc=False).dt.tz_localize(None)

    active = res[
        res["discharge_date"].isna() &
        res["deceased_date"].isna() &
        (res["admission_date"] <= score_date)
    ].copy()

    active["obs_date"]      = score_date
    active["is_outpatient"] = active["outpatient"].astype(int)

    print(f"  Active residents to score: {len(active):,}")
    return active.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING (reuses functions from data_preparation.py)
# ══════════════════════════════════════════════════════════════════════════════

def compute_scoring_features(
    spine: pd.DataFrame,
    data: dict,
    feat_cols: list[str],
) -> pd.DataFrame:
    """
    Apply all feature functions then align to the training feature column list.
    Any column present at training time but absent at scoring time is filled
    with NaN — XGBoost handles missing values natively.
    """
    s = feature_static(spine)
    s = feature_comorbidities(s, data["diagnoses"])
    s = feature_adl(s, data["adl_responses"])
    s = feature_gg(s, data["gg_responses"])
    s = feature_vitals(s, data["vitals"])
    s = feature_medications(s, data["medications"])
    s = feature_document_tags(s, data["document_tags"])
    s = feature_labs(s, data["lab_reports"])
    s = feature_prior_incidents(s, data)
    s = feature_prior_rth(s, data["hospital_transfers"])
    s = feature_care_needs(s, data["needs"])
    s = feature_therapy(s, data["therapy_tracks"])

    # Encode facility_id (use -1 for facilities not seen during training)
    fac_map_path = OUTPUT_DIR / "models" / "facility_map.json"
    if fac_map_path.exists():
        with open(fac_map_path) as f:
            fac_map = json.load(f)
        s["facility_id_enc"] = s["facility_id"].map(fac_map).fillna(-1).astype(int)
    else:
        s["facility_id_enc"] = -1

    # Align to training feature set
    for col in set(feat_cols) - set(s.columns):
        s[col] = np.nan

    return s[feat_cols]


# ══════════════════════════════════════════════════════════════════════════════
# 3. SCORING AND SHAP EXPLANATIONS
# ══════════════════════════════════════════════════════════════════════════════

def score_residents(
    models: dict,
    calibrators: dict,
    target_feat_cols: dict,
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Produce calibrated risk scores + SHAP values for all active residents.
    Scores pass through the isotonic calibrator so they reflect true event
    probabilities rather than raw XGBoost outputs.
    """
    scores    = {}
    shap_vals = {}

    for target in TARGETS:
        model      = models[target]
        calibrator = calibrators.get(target)
        tcols      = target_feat_cols[target]
        X_target   = X[tcols]

        raw   = model.predict_proba(X_target)[:, 1]
        probs = calibrator.predict(raw) if calibrator is not None else raw
        scores[f"{target}_score"] = probs

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_target)
        shap_vals[target] = (sv, tcols)   # store (values, col_names) together

    return pd.DataFrame(scores, index=X.index), shap_vals


def extract_top_reasons(
    shap_row: np.ndarray,
    feat_cols: list[str],
    n: int = 3,
) -> str:
    """
    Return the top N feature names by |SHAP| for a single resident.

    Each entry is prefixed with ↑ (positive SHAP — feature is increasing this
    resident's risk) or ↓ (negative SHAP — feature is reducing it).  This lets
    a care coordinator immediately distinguish risk factors from protective ones.
    """
    top_idx = np.argsort(np.abs(shap_row))[::-1][:n]
    parts = []
    for i in top_idx:
        arrow = "↑" if shap_row[i] > 0 else "↓"
        parts.append(f"{arrow} {feat_cols[i].replace('_', ' ')}")
    return " | ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# 4. DASHBOARD ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def assemble_dashboard(
    spine: pd.DataFrame,
    scores_df: pd.DataFrame,
    shap_vals: dict,
    feat_cols: list[str],
) -> pd.DataFrame:
    """Combine metadata, scores, tiers, expected cost, and SHAP reasons."""
    dash = pd.DataFrame({
        "resident_id": spine["resident_id"].values,
        "facility_id": spine["facility_id"].values,
        "obs_date":    spine["obs_date"].values,
    })

    # Optionally add age / LOS if available
    if "date_of_birth" in spine.columns:
        dob = pd.to_datetime(spine["date_of_birth"])
        obs = pd.to_datetime(spine["obs_date"])
        dash["age"] = ((obs - dob).dt.days / 365.25).round(1)
    if "admission_date" in spine.columns:
        adm = pd.to_datetime(spine["admission_date"])
        dash["los_days"] = (pd.to_datetime(spine["obs_date"]) - adm).dt.days

    # ── Scores (all targets — wound included for CSV auditability) ────────
    for t in TARGETS:
        dash[f"{t}_score"] = scores_df[f"{t}_score"].values

    # ── Expected cost — active targets only ───────────────────────────────
    # wound_60d is excluded: BSS = -0.019 (no-skill predictor), scores
    # compressed to within 0.3 pp of base rate for 90% of residents.
    # Its scores are saved to the CSV but do not drive cost or tier.
    dash["expected_cost_30d"] = (
        sum(dash[f"{t}_score"] * CLAIM_COSTS[t] for t in DASHBOARD_ACTIVE_TARGETS)
    ).round(0)

    # ── Overall tier — percentile of expected cost (not per-target union) ─
    # Tiering per-target then taking the worst produced ~58% "High" due to
    # the union of three 25%-High groups.  Tiering on the combined cost
    # column guarantees exactly 25% High / 25% Medium / 50% Low.
    high_thr = np.percentile(dash["expected_cost_30d"], RISK_TIER_PERCENTILES["high"])
    med_thr  = np.percentile(dash["expected_cost_30d"], RISK_TIER_PERCENTILES["medium"])
    dash["overall_tier"] = pd.cut(
        dash["expected_cost_30d"],
        bins=[-0.001, med_thr, high_thr, float("inf")],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    # ── SHAP top reasons — dominant active-target model per resident ──────
    # Dominant model is selected among DASHBOARD_ACTIVE_TARGETS only.
    # Wound SHAP drivers are still stored in the CSV for future monitoring.
    target_costs   = {t: dash[f"{t}_score"] * CLAIM_COSTS[t] for t in DASHBOARD_ACTIVE_TARGETS}
    cost_df        = pd.DataFrame(target_costs)
    dominant_target = cost_df.idxmax(axis=1).values

    TARGET_SHORT = {"fall_30d": "Fall", "rth_60d": "RTH", "wound_60d": "Wound"}

    dominant_reasons = []
    for i in range(len(dash)):
        t = dominant_target[i]
        sv, tcols = shap_vals[t]
        reasons = extract_top_reasons(sv[i], tcols)
        dominant_reasons.append(f"[{TARGET_SHORT[t]}] {reasons}")

    dash["dominant_model"] = dominant_target
    dash["top_drivers"]    = dominant_reasons

    # All models' drivers stored in CSV for full auditability
    for t in TARGETS:
        sv, tcols = shap_vals[t]
        dash[f"{t}_top_reasons"] = [
            extract_top_reasons(sv[i], tcols) for i in range(len(dash))
        ]

    return (dash
            .sort_values("expected_cost_30d", ascending=False)
            .reset_index(drop=True)
            .assign(rank=lambda d: range(1, len(d) + 1))
            .pipe(lambda d: d[["rank"] + [c for c in d.columns if c != "rank"]]))


# ══════════════════════════════════════════════════════════════════════════════
# 5. HTML REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_html_report(dashboard: pd.DataFrame, score_date: date) -> str:
    tier_colors = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}

    def badge(tier):
        c = tier_colors.get(str(tier), "#95a5a6")
        return (f'<span style="background:{c};color:white;padding:2px 8px;'
                f'border-radius:4px;font-size:12px">{tier}</span>')

    rows_html = ""
    for _, row in dashboard.head(100).iterrows():
        top_drivers = str(row.get("top_drivers", "")).replace("|", "·")
        wound_score = f"{row['wound_60d_score']:.2%}" if "wound_60d_score" in row else "—"
        rows_html += f"""
        <tr>
          <td style="text-align:center">{row['rank']}</td>
          <td><code style="font-size:11px">{str(row['resident_id'])[-8:]}</code></td>
          <td><code style="font-size:11px">{str(row['facility_id'])[-8:]}</code></td>
          <td style="text-align:center">{row.get('age', '—')}</td>
          <td style="text-align:center">{badge(row['overall_tier'])}</td>
          <td style="text-align:center">{row['fall_30d_score']:.2%}</td>
          <td style="text-align:center">{row['rth_60d_score']:.2%}</td>
          <td style="text-align:center;color:#aaa;font-style:italic">{wound_score}</td>
          <td style="text-align:right;font-weight:bold">${row['expected_cost_30d']:,.0f}</td>
          <td style="font-size:11px;color:#555">{top_drivers}</td>
        </tr>"""

    n_high   = (dashboard["overall_tier"] == "High").sum()
    n_medium = (dashboard["overall_tier"] == "Medium").sum()
    n_low    = (dashboard["overall_tier"] == "Low").sum()
    total    = dashboard["expected_cost_30d"].sum()
    horizons = " · ".join(
        f"{TARGET_LABELS[t]} = {LABEL_HORIZONS[t]}d" for t in DASHBOARD_ACTIVE_TARGETS
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SNF Risk Dashboard — {score_date}</title>
<style>
  body  {{ font-family:-apple-system,Arial,sans-serif;margin:0;padding:24px;background:#f5f5f5 }}
  h1   {{ color:#2c3e50;font-size:22px;margin-bottom:4px }}
  .sub {{ color:#7f8c8d;font-size:13px;margin-bottom:8px }}
  .note {{ background:#fff8e1;border-left:3px solid #f39c12;padding:8px 12px;
           font-size:12px;color:#7f6d3a;margin-bottom:20px;border-radius:0 4px 4px 0 }}
  .cards {{ display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap }}
  .card  {{ background:white;border-radius:8px;padding:16px 20px;min-width:140px;border:1px solid #e8e8e8 }}
  .card .label {{ font-size:12px;color:#7f8c8d;text-transform:uppercase;letter-spacing:.5px }}
  .card .value {{ font-size:28px;font-weight:600;margin-top:4px }}
  .card.red   .value {{ color:#e74c3c }}
  .card.amber .value {{ color:#f39c12 }}
  .card.green .value {{ color:#27ae60 }}
  .card.blue  .value {{ color:#2980b9;font-size:20px }}
  table {{ width:100%;border-collapse:collapse;background:white;border-radius:8px;
           overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.1) }}
  thead {{ background:#2c3e50;color:white }}
  th    {{ padding:10px 12px;text-align:left;font-size:12px;font-weight:500 }}
  th.muted {{ color:#94a3b8 }}
  td    {{ padding:9px 12px;font-size:13px;border-bottom:1px solid #f0f0f0 }}
  tr:last-child td {{ border-bottom:none }}
  tr:hover td {{ background:#fafafa }}
</style>
</head>
<body>
<h1>&#127973; Resident Risk Dashboard</h1>
<p class="sub">Score date: {score_date} &nbsp;|&nbsp; Model: XGBoost
  &nbsp;|&nbsp; {horizons}
  &nbsp;|&nbsp; Top 100 of {len(dashboard):,} residents</p>
<p class="note">
  Tier and expected cost are based on <strong>fall</strong> and <strong>RTH</strong> only.
  Wound (60d) is shown for monitoring but is suppressed from triage
  (Brier Skill Score &minus;0.019 &mdash; no-skill predictor).
  Tiers are percentile-ranked on combined expected cost: top&nbsp;25%&nbsp;=&nbsp;High,
  next&nbsp;25%&nbsp;=&nbsp;Medium, bottom&nbsp;50%&nbsp;=&nbsp;Low.
</p>
<div class="cards">
  <div class="card red">  <div class="label">High risk</div>  <div class="value">{n_high}</div></div>
  <div class="card amber"><div class="label">Medium risk</div><div class="value">{n_medium}</div></div>
  <div class="card green"><div class="label">Low risk</div>   <div class="value">{n_low}</div></div>
  <div class="card blue"> <div class="label">Expected cost (fall + RTH)</div>
                          <div class="value">${total:,.0f}</div></div>
</div>
<table>
<thead><tr>
  <th>#</th><th>Resident</th><th>Facility</th><th>Age</th>
  <th>Tier</th><th>Fall (30d)</th><th>RTH (60d)</th>
  <th class="muted">Wound (60d) &#9888;</th>
  <th>Exp. cost</th><th>Top drivers (dominant model)</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(score_date_str: str | None = None):
    models_dir  = OUTPUT_DIR / "models"
    metrics_dir = OUTPUT_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    score_date = (pd.Timestamp(score_date_str) if score_date_str
                  else pd.Timestamp.today().normalize())
    print(f"\n[0] Score date: {score_date.date()}")

    print("\n[1] Loading XGBoost models and calibrators …")
    models, calibrators, target_feat_cols = {}, {}, {}
    for t in TARGETS:
        with open(models_dir / f"xgb_{t}.pkl", "rb") as f:
            models[t] = pickle.load(f)
        cal_path = models_dir / f"calibrator_{t}.pkl"
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                calibrators[t] = pickle.load(f)
        else:
            print(f"  WARNING: no calibrator for {t} — using raw probabilities")
            calibrators[t] = None
        tfc_path = models_dir / f"feature_cols_{t}.json"
        if tfc_path.exists():
            with open(tfc_path) as f:
                target_feat_cols[t] = json.load(f)
        else:
            with open(models_dir / "feature_cols.json") as f:
                target_feat_cols[t] = json.load(f)

    with open(models_dir / "feature_cols.json") as f:
        feat_cols = json.load(f)
    print(f"  Models: {list(models.keys())}  |  Calibrators loaded: {sum(v is not None for v in calibrators.values())}")

    print("\n[2] Loading raw data …")
    data = load_raw_data(DATA_DIR)

    print("\n[3] Building scoring spine (active residents) …")
    spine = build_scoring_spine(data["residents"], score_date)
    if spine.empty:
        print("  No active residents — exiting.")
        return

    print("\n[4] Computing features …")
    X = compute_scoring_features(spine, data, feat_cols)

    print("\n[5] Scoring and computing SHAP explanations …")
    scores_df, shap_vals = score_residents(
        models, calibrators, target_feat_cols, X
    )

    print("\n[6] Assembling dashboard …")
    dashboard = assemble_dashboard(spine, scores_df, shap_vals, feat_cols)

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  TOP 10 HIGHEST-RISK RESIDENTS")
    print(f"{'─'*70}")
    show = ["rank", "overall_tier", "fall_30d_score", "rth_60d_score",
            "wound_60d_score", "expected_cost_30d"]
    print(dashboard[show].head(10).to_string(index=False))
    print(f"\n  High:    {(dashboard['overall_tier'] == 'High').sum()}")
    print(f"  Medium:  {(dashboard['overall_tier'] == 'Medium').sum()}")
    print(f"  Low:     {(dashboard['overall_tier'] == 'Low').sum()}")
    print(f"  Total expected cost: ${dashboard['expected_cost_30d'].sum():,.0f}")

    # ── Save outputs ──────────────────────────────────────────────────────
    date_str  = score_date.strftime("%Y%m%d")
    csv_path  = metrics_dir / f"risk_dashboard_{date_str}.csv"
    html_path = metrics_dir / f"risk_dashboard_{date_str}.html"

    dashboard.to_csv(csv_path, index=False)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(generate_html_report(dashboard, score_date.date()))

    print(f"\n  ✓ CSV  → {csv_path}")
    print(f"  ✓ HTML → {html_path}")
    return dashboard


if __name__ == "__main__":
    import time
    t0 = time.time()
    main(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"\nDone in {time.time()-t0:.1f}s")
