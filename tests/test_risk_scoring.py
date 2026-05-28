from datetime import date

import numpy as np
import pandas as pd
import pytest

import risk_scoring as rs
from config import TARGETS, DASHBOARD_ACTIVE_TARGETS
from helpers import OBS1, OBS2, ADM, DOB1, DOB2


# ══════════════════════════════════════════════════════════════════════════════
# build_scoring_spine
# ══════════════════════════════════════════════════════════════════════════════

def test_scoring_spine_excludes_discharged(residents):
    spine = rs.build_scoring_spine(residents, OBS1)
    assert "R003" not in spine["resident_id"].values


def test_scoring_spine_excludes_deceased():
    res = pd.DataFrame({
        "resident_id":    ["R001"],
        "facility_id":    ["F001"],
        "admission_date": [ADM],
        "discharge_date": [pd.NaT],
        "deceased_date":  [pd.Timestamp("2024-10-15")],
        "date_of_birth":  [DOB1],
        "outpatient":     [False],
    })
    spine = rs.build_scoring_spine(res, OBS1)
    assert len(spine) == 0


def test_scoring_spine_excludes_not_yet_admitted(residents):
    future_adm = pd.Timestamp("2024-12-01")
    res = residents.copy()
    res.loc[res["resident_id"] == "R001", "admission_date"] = future_adm
    spine = rs.build_scoring_spine(res, OBS1)
    assert "R001" not in spine["resident_id"].values


def test_scoring_spine_obs_date_equals_score_date(residents):
    spine = rs.build_scoring_spine(residents, OBS1)
    assert (spine["obs_date"] == OBS1).all()


def test_scoring_spine_includes_active_residents(residents):
    spine = rs.build_scoring_spine(residents, OBS1)
    assert set(spine["resident_id"].values) == {"R001", "R002"}


def test_scoring_spine_is_outpatient_encoded(residents):
    spine = rs.build_scoring_spine(residents, OBS1)
    assert spine["is_outpatient"].isin([0, 1]).all()


# ══════════════════════════════════════════════════════════════════════════════
# extract_top_reasons
# ══════════════════════════════════════════════════════════════════════════════

def test_extract_top_reasons_returns_n_parts():
    shap_row = np.array([0.5, -0.3, 0.8, 0.1, -0.6])
    cols     = [f"feat_{i}" for i in range(5)]
    result   = rs.extract_top_reasons(shap_row, cols, n=3)
    parts    = result.split(" | ")
    assert len(parts) == 3


def test_extract_top_reasons_sorted_by_abs_shap():
    # |0.8| > |0.6| > |0.5| → feat_2, feat_4, feat_0
    shap_row = np.array([0.5, -0.3, 0.8, 0.1, -0.6])
    cols     = ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"]
    result   = rs.extract_top_reasons(shap_row, cols, n=3)
    assert "feat 2" in result
    assert result.index("feat 2") < result.index("feat 4")
    assert result.index("feat 4") < result.index("feat 0")


def test_extract_top_reasons_up_arrow_for_positive_shap():
    shap_row = np.array([1.0, 0.0, 0.0])
    cols     = ["a", "b", "c"]
    result   = rs.extract_top_reasons(shap_row, cols, n=1)
    assert result.startswith("↑")


def test_extract_top_reasons_down_arrow_for_negative_shap():
    shap_row = np.array([-1.0, 0.0, 0.0])
    cols     = ["a", "b", "c"]
    result   = rs.extract_top_reasons(shap_row, cols, n=1)
    assert result.startswith("↓")


def test_extract_top_reasons_replaces_underscores():
    shap_row = np.array([1.0])
    result   = rs.extract_top_reasons(shap_row, ["my_feat"], n=1)
    assert "my feat" in result
    assert "_" not in result


def test_extract_top_reasons_n1_returns_single_part():
    shap_row = np.array([0.5, -0.3, 0.8])
    cols     = ["a", "b", "c"]
    result   = rs.extract_top_reasons(shap_row, cols, n=1)
    assert " | " not in result


# ══════════════════════════════════════════════════════════════════════════════
# assemble_dashboard  (needs models_dict + spine from scoring)
# ══════════════════════════════════════════════════════════════════════════════

def _make_scoring_inputs(models_dict, feat_cols):
    """Build a scoring spine + run score_residents to get scores + SHAP values."""
    residents = pd.DataFrame({
        "resident_id":    ["R001", "R002"],
        "facility_id":    ["F001", "F002"],
        "admission_date": ADM,
        "discharge_date": pd.NaT,
        "deceased_date":  pd.NaT,
        "date_of_birth":  [DOB1, DOB2],
        "outpatient":     False,
    })
    spine = rs.build_scoring_spine(residents, OBS1)

    rng = np.random.default_rng(5)
    X   = pd.DataFrame(rng.standard_normal((len(spine), len(feat_cols))), columns=feat_cols)

    scores_df, shap_vals = rs.score_residents(
        models_dict,
        {t: None for t in TARGETS},
        {t: feat_cols for t in TARGETS},
        X,
    )
    return spine, scores_df, shap_vals


def test_assemble_dashboard_row_count(models_dict, feat_cols):
    spine, scores_df, shap_vals = _make_scoring_inputs(models_dict, feat_cols)
    dash = rs.assemble_dashboard(spine, scores_df, shap_vals, feat_cols)
    assert len(dash) == len(spine)


def test_assemble_dashboard_sorted_by_expected_cost(models_dict, feat_cols):
    spine, scores_df, shap_vals = _make_scoring_inputs(models_dict, feat_cols)
    dash = rs.assemble_dashboard(spine, scores_df, shap_vals, feat_cols)
    costs = dash["expected_cost_30d"].values
    assert (np.diff(costs) <= 0).all()


def test_assemble_dashboard_rank_starts_at_one(models_dict, feat_cols):
    spine, scores_df, shap_vals = _make_scoring_inputs(models_dict, feat_cols)
    dash = rs.assemble_dashboard(spine, scores_df, shap_vals, feat_cols)
    assert dash["rank"].iloc[0] == 1
    assert list(dash["rank"]) == list(range(1, len(dash) + 1))


def test_assemble_dashboard_overall_tier_valid_values(models_dict, feat_cols):
    spine, scores_df, shap_vals = _make_scoring_inputs(models_dict, feat_cols)
    dash = rs.assemble_dashboard(spine, scores_df, shap_vals, feat_cols)
    assert dash["overall_tier"].isin({"High", "Medium", "Low"}).all()


def test_assemble_dashboard_no_per_target_tier_columns(models_dict, feat_cols):
    # overall_tier replaces per-target tiers; individual {t}_tier columns
    # are not generated in the updated assemble_dashboard.
    spine, scores_df, shap_vals = _make_scoring_inputs(models_dict, feat_cols)
    dash = rs.assemble_dashboard(spine, scores_df, shap_vals, feat_cols)
    for t in TARGETS:
        assert f"{t}_tier" not in dash.columns


def test_assemble_dashboard_scores_in_zero_one(models_dict, feat_cols):
    spine, scores_df, shap_vals = _make_scoring_inputs(models_dict, feat_cols)
    dash = rs.assemble_dashboard(spine, scores_df, shap_vals, feat_cols)
    for t in TARGETS:
        col = dash[f"{t}_score"]
        assert (col >= 0).all() and (col <= 1).all()


def test_assemble_dashboard_top_drivers_column_present(models_dict, feat_cols):
    spine, scores_df, shap_vals = _make_scoring_inputs(models_dict, feat_cols)
    dash = rs.assemble_dashboard(spine, scores_df, shap_vals, feat_cols)
    assert "top_drivers" in dash.columns
    assert dash["top_drivers"].notna().all()


def test_assemble_dashboard_top_drivers_contains_target_prefix(models_dict, feat_cols):
    # top_drivers format is "[Fall] ↑ ..." or "[RTH] ↑ ..."
    spine, scores_df, shap_vals = _make_scoring_inputs(models_dict, feat_cols)
    dash = rs.assemble_dashboard(spine, scores_df, shap_vals, feat_cols)
    assert dash["top_drivers"].str.startswith("[").all()


def test_assemble_dashboard_dominant_model_is_active_target(models_dict, feat_cols):
    # dominant_model is chosen from DASHBOARD_ACTIVE_TARGETS, not all TARGETS
    spine, scores_df, shap_vals = _make_scoring_inputs(models_dict, feat_cols)
    dash = rs.assemble_dashboard(spine, scores_df, shap_vals, feat_cols)
    assert dash["dominant_model"].isin(DASHBOARD_ACTIVE_TARGETS).all()


def test_assemble_dashboard_wound_not_dominant_model(models_dict, feat_cols):
    # wound_60d is excluded from DASHBOARD_ACTIVE_TARGETS and must never
    # be selected as the dominant model driving cost or triage.
    spine, scores_df, shap_vals = _make_scoring_inputs(models_dict, feat_cols)
    dash = rs.assemble_dashboard(spine, scores_df, shap_vals, feat_cols)
    assert "wound_60d" not in dash["dominant_model"].values


# ══════════════════════════════════════════════════════════════════════════════
# score_residents
# ══════════════════════════════════════════════════════════════════════════════

def test_score_residents_probabilities_in_zero_one(models_dict, eval_X, feat_cols):
    scores_df, _ = rs.score_residents(
        models_dict,
        {t: None for t in TARGETS},
        {t: feat_cols for t in TARGETS},
        eval_X,
    )
    for t in TARGETS:
        col = scores_df[f"{t}_score"]
        assert (col >= 0).all() and (col <= 1).all()


def test_score_residents_shap_shape(models_dict, eval_X, feat_cols):
    _, shap_vals = rs.score_residents(
        models_dict,
        {t: None for t in TARGETS},
        {t: feat_cols for t in TARGETS},
        eval_X,
    )
    for t in TARGETS:
        sv, tcols = shap_vals[t]
        assert sv.shape == (len(eval_X), len(feat_cols))


# ══════════════════════════════════════════════════════════════════════════════
# generate_html_report
# ══════════════════════════════════════════════════════════════════════════════

def _make_minimal_dashboard():
    return pd.DataFrame({
        "rank":              [1, 2],
        "resident_id":       ["R001", "R002"],
        "facility_id":       ["F001", "F002"],
        "obs_date":          [OBS1,   OBS1],
        "age":               [84.0,   72.0],
        "fall_30d_score":    [0.30,   0.10],
        "rth_60d_score":     [0.05,   0.02],
        "wound_60d_score":   [0.10,   0.05],
        "overall_tier":      ["High", "Low"],
        "expected_cost_30d": [2000.0, 500.0],
        "top_drivers":       ["[Fall] ↑ prior days since last fall", "[RTH] ↑ n comorbidities"],
        "dominant_model":    ["fall_30d", "rth_60d"],
        "fall_30d_top_reasons":  ["↑ feat a", "↑ feat b"],
        "rth_60d_top_reasons":   ["↑ feat c", "↑ feat d"],
        "wound_60d_top_reasons": ["↑ feat e", "↑ feat f"],
    })


def test_generate_html_report_returns_string():
    html = rs.generate_html_report(_make_minimal_dashboard(), date(2024, 11, 1))
    assert isinstance(html, str)


def test_generate_html_report_contains_score_date():
    html = rs.generate_html_report(_make_minimal_dashboard(), date(2024, 11, 1))
    assert "2024-11-01" in html


def test_generate_html_report_is_valid_html():
    html = rs.generate_html_report(_make_minimal_dashboard(), date(2024, 11, 1))
    assert html.strip().startswith("<!DOCTYPE html>")
    assert "</html>" in html


def test_generate_html_report_contains_tier_counts():
    dash = _make_minimal_dashboard()
    html = rs.generate_html_report(dash, date(2024, 11, 1))
    # 1 High, 1 Low resident → both counts appear in the summary cards
    assert ">1<" in html
