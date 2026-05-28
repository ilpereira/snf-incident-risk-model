import numpy as np
import pandas as pd
import pytest

import data_preparation as dp
from helpers import OBS1, OBS2, ADM, DOB1, DOB2


# ══════════════════════════════════════════════════════════════════════════════
# build_observation_spine
# ══════════════════════════════════════════════════════════════════════════════

def test_build_spine_produces_rows_within_valid_window(residents):
    spine = dp.build_observation_spine(residents)
    # All obs_dates must be after admission + MIN_HISTORY_DAYS
    from config import MIN_HISTORY_DAYS
    for _, row in spine.iterrows():
        res = residents[residents["resident_id"] == row["resident_id"]].iloc[0]
        adm = pd.to_datetime(res["admission_date"])
        assert row["obs_date"] >= adm + pd.Timedelta(days=MIN_HISTORY_DAYS)


def test_build_spine_obs_dates_are_weekly(residents):
    spine = dp.build_observation_spine(residents)
    for res_id, grp in spine.groupby("resident_id"):
        if len(grp) < 2:
            continue
        diffs = grp["obs_date"].sort_values().diff().dropna().dt.days
        assert (diffs == 7).all(), f"Non-weekly gaps for {res_id}"


def test_build_spine_discharge_bounds_window(residents):
    # R003 has discharge_date=2024-11-20; no obs_date should exceed that
    spine = dp.build_observation_spine(residents)
    r3 = spine[spine["resident_id"] == "R003"]
    discharge = pd.Timestamp("2024-11-20")
    assert (r3["obs_date"] <= discharge).all()


def test_build_spine_resident_columns_present(residents):
    spine = dp.build_observation_spine(residents)
    assert {"resident_id", "facility_id", "obs_date"}.issubset(spine.columns)


# ══════════════════════════════════════════════════════════════════════════════
# label_observations
# ══════════════════════════════════════════════════════════════════════════════

def test_label_fall_within_horizon(spine, data_dict):
    # INC002: Fall for R001 at OBS1+15d → fall_30d=1 for R001 at OBS1
    labeled = dp.label_observations(spine, data_dict)
    r001_obs1 = labeled[(labeled["resident_id"] == "R001") & (labeled["obs_date"] == OBS1)]
    assert r001_obs1["fall_30d"].iloc[0] == 1


def test_label_fall_outside_horizon_is_zero(spine, data_dict):
    # R002 has no future falls → fall_30d=0
    labeled = dp.label_observations(spine, data_dict)
    r002 = labeled[labeled["resident_id"] == "R002"]
    assert (r002["fall_30d"] == 0).all()


def test_label_rth_within_horizon(spine, data_dict):
    # Unplanned transfer for R001 at OBS1+10d → rth_60d=1 at OBS1 (within 60d horizon)
    labeled = dp.label_observations(spine, data_dict)
    r001_obs1 = labeled[(labeled["resident_id"] == "R001") & (labeled["obs_date"] == OBS1)]
    assert r001_obs1["rth_60d"].iloc[0] == 1


def test_label_wound_within_horizon(spine, data_dict):
    # Wound for R002 at OBS1, but label window is (obs_date, obs_date+60d]
    # The wound is AT OBS1, not strictly after → should be 0 (days_ahead == 0 not > 0)
    labeled = dp.label_observations(spine, data_dict)
    r002_obs1 = labeled[(labeled["resident_id"] == "R002") & (labeled["obs_date"] == OBS1)]
    assert r002_obs1["wound_60d"].iloc[0] == 0


def test_label_columns_are_binary(spine, data_dict):
    labeled = dp.label_observations(spine, data_dict)
    for col in ["fall_30d", "rth_60d", "wound_60d"]:
        assert labeled[col].isin([0, 1]).all(), f"{col} has non-binary values"


# ══════════════════════════════════════════════════════════════════════════════
# _rolling_on_daily
# ══════════════════════════════════════════════════════════════════════════════

def test_rolling_returns_expected_columns():
    events = pd.DataFrame({
        "resident_id": ["R001", "R001"],
        "date":        [OBS1, OBS2],
        "value":       [3.0, 4.0],
    })
    result = dp._rolling_on_daily(events, "resident_id", "date", ["value"],
                                   windows=[7], prefix="test", aggs=["mean"])
    assert "test_value_mean_7d" in result.columns
    assert "resident_id" in result.columns
    assert "date" in result.columns


def test_rolling_converts_string_values():
    # Strings must be coerced to float (as in real adl_responses)
    events = pd.DataFrame({
        "resident_id": ["R001"],
        "date":        [OBS1],
        "value":       ["3"],   # string
    })
    result = dp._rolling_on_daily(events, "resident_id", "date", ["value"],
                                   windows=[7], prefix="test", aggs=["mean"])
    assert result["test_value_mean_7d"].dtype in (float, np.float64, np.float32)
    assert result["test_value_mean_7d"].iloc[0] == pytest.approx(3.0)


def test_rolling_mean_is_correct():
    events = pd.DataFrame({
        "resident_id": ["R001", "R001"],
        "date":        [OBS1, OBS1],   # same date → daily mean = (2+4)/2 = 3
        "value":       [2.0, 4.0],
    })
    result = dp._rolling_on_daily(events, "resident_id", "date", ["value"],
                                   windows=[7], prefix="x", aggs=["mean"])
    assert result["x_value_mean_7d"].iloc[0] == pytest.approx(3.0)


def test_rolling_sum_accumulates_across_days():
    # Use a 3-day gap so d1 is clearly inside the 7D window at d2.
    # Pandas offset-based rolling uses (t-7D, t] — exactly 7 days back is excluded.
    d1 = pd.Timestamp("2024-11-01")
    d2 = pd.Timestamp("2024-11-04")
    events = pd.DataFrame({
        "resident_id": ["R001", "R001"],
        "date":        [d1, d2],
        "value":       [1.0, 1.0],
    })
    result = dp._rolling_on_daily(events, "resident_id", "date", ["value"],
                                   windows=[7], prefix="x", aggs=["sum"])
    obs2_row = result[result["date"] == d2]
    assert obs2_row["x_value_sum_7d"].iloc[0] == pytest.approx(2.0)


# ══════════════════════════════════════════════════════════════════════════════
# _join_rolling_to_spine
# ══════════════════════════════════════════════════════════════════════════════

def test_join_rolling_preserves_all_spine_rows(spine):
    rolling = pd.DataFrame({
        "resident_id": ["R001"],
        "date":        [OBS1],
        "feat_val":    [5.0],
    })
    result = dp._join_rolling_to_spine(spine, rolling)
    assert len(result) == len(spine)


def test_join_rolling_fills_nan_for_missing_dates(spine):
    rolling = pd.DataFrame({
        "resident_id": ["R001"],
        "date":        [OBS1],
        "feat_val":    [5.0],
    })
    result = dp._join_rolling_to_spine(spine, rolling)
    # R001/OBS2 and all R002 rows have no rolling record → NaN
    r001_obs2 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS2)]
    assert r001_obs2["feat_val"].isna().all()


# ══════════════════════════════════════════════════════════════════════════════
# feature_static
# ══════════════════════════════════════════════════════════════════════════════

def test_feature_static_age_at_obs(spine):
    result = dp.feature_static(spine)
    r001_obs1 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS1)]
    expected_age = (OBS1 - DOB1).days / 365.25
    assert r001_obs1["age_at_obs"].iloc[0] == pytest.approx(expected_age, abs=0.1)


def test_feature_static_los_days(spine):
    result = dp.feature_static(spine)
    r001_obs1 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS1)]
    assert r001_obs1["los_days"].iloc[0] == (OBS1 - ADM).days


def test_feature_static_is_outpatient(spine):
    result = dp.feature_static(spine)
    assert (result["is_outpatient"] == 0).all()


def test_feature_static_age_bucket_created(spine):
    result = dp.feature_static(spine)
    assert "age_bucket" in result.columns


# ══════════════════════════════════════════════════════════════════════════════
# feature_comorbidities
# ══════════════════════════════════════════════════════════════════════════════

def test_comorbidities_flags_active_diagnosis(spine, diagnoses):
    result = dp.feature_comorbidities(dp.feature_static(spine), diagnoses)
    r001_obs1 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS1)]
    assert r001_obs1["dx_dementia"].iloc[0] == 1


def test_comorbidities_ignores_resolved_diagnosis(spine, diagnoses):
    # I50 (CHF) resolved at 2024-10-15, before OBS1 → should be 0
    result = dp.feature_comorbidities(dp.feature_static(spine), diagnoses)
    r001_obs1 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS1)]
    assert r001_obs1["dx_chf"].iloc[0] == 0


def test_comorbidities_charlson_score_includes_age(spine, diagnoses):
    result = dp.feature_comorbidities(dp.feature_static(spine), diagnoses)
    r001_obs1 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS1)]
    # dx_dementia weight=2 + age ~84 → age points = 4 → charlson >= 6
    assert r001_obs1["charlson_score"].iloc[0] >= 6


def test_comorbidities_n_comorbidities_count(spine, diagnoses):
    result = dp.feature_comorbidities(dp.feature_static(spine), diagnoses)
    r002_obs1 = result[(result["resident_id"] == "R002") & (result["obs_date"] == OBS1)]
    assert r002_obs1["n_comorbidities"].iloc[0] == 1   # only dx_hypertension


# ══════════════════════════════════════════════════════════════════════════════
# feature_adl
# ══════════════════════════════════════════════════════════════════════════════

def test_feature_adl_creates_rolling_columns(spine, adl_responses):
    result = dp.feature_adl(dp.feature_static(spine), adl_responses)
    assert "adl_eating_value_mean_7d" in result.columns
    assert "adl_eating_value_std_7d" in result.columns


def test_feature_adl_value_is_numeric(spine, adl_responses):
    result = dp.feature_adl(dp.feature_static(spine), adl_responses)
    col = "adl_eating_value_mean_7d"
    non_null = result[col].dropna()
    assert non_null.dtype in (float, np.float64)
    assert np.allclose(non_null.values, 2.0)


def test_feature_adl_total_trend_column(spine, adl_responses):
    result = dp.feature_adl(dp.feature_static(spine), adl_responses)
    assert "adl_trend_7v30" in result.columns


# ══════════════════════════════════════════════════════════════════════════════
# feature_gg
# ══════════════════════════════════════════════════════════════════════════════

def test_feature_gg_creates_mobility_columns(spine, gg_responses):
    result = dp.feature_gg(dp.feature_static(spine), gg_responses)
    assert "gg_mobility_value_mean_7d" in result.columns
    assert "gg_mobility_value_mean_30d" in result.columns


def test_feature_gg_dependency_is_inverted():
    # response_code=3 → dependency = 7.0 - 3 = 4.0
    single_spine = pd.DataFrame({
        "resident_id":    ["R001"],
        "facility_id":    ["F001"],
        "obs_date":       [OBS1],
        "admission_date": ADM,
        "date_of_birth":  DOB1,
        "discharge_date": pd.NaT,
        "deceased_date":  pd.NaT,
        "outpatient":     False,
    })
    gg = pd.DataFrame({
        "resident_id":   ["R001"],
        "created_at":    [OBS1],
        "task_group":    ["Mobility"],
        "task_name":     ["Sit to Stand"],
        "response_code": [3],
    })
    result = dp.feature_gg(dp.feature_static(single_spine), gg)
    val = result["gg_mobility_value_mean_7d"].iloc[0]
    assert val == pytest.approx(4.0)   # 7 - 3 = 4


# ══════════════════════════════════════════════════════════════════════════════
# feature_vitals
# ══════════════════════════════════════════════════════════════════════════════

def test_feature_vitals_creates_rolling_columns(spine, vitals):
    result = dp.feature_vitals(dp.feature_static(spine), vitals)
    assert "vital_bp_sys_mean_7d" in result.columns
    assert "vital_bp_sys_mean_30d" in result.columns


def test_feature_vitals_flag_bp_high(spine, vitals):
    # R002 BP=165 ≥ 160 → flag_bp_high should be 1
    result = dp.feature_vitals(dp.feature_static(spine), vitals)
    r002_obs1 = result[(result["resident_id"] == "R002") & (result["obs_date"] == OBS1)]
    assert r002_obs1["flag_bp_high"].iloc[0] == 1


def test_feature_vitals_flag_bp_high_not_set_for_normal(spine, vitals):
    result = dp.feature_vitals(dp.feature_static(spine), vitals)
    r001_obs1 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS1)]
    assert r001_obs1["flag_bp_high"].iloc[0] == 0


def test_feature_vitals_flag_o2_low(spine, vitals):
    # R001 O2=89 < 92 → flag_o2_low should be 1 at OBS1
    result = dp.feature_vitals(dp.feature_static(spine), vitals)
    r001_obs1 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS1)]
    assert r001_obs1["flag_o2_low"].iloc[0] == 1


# ══════════════════════════════════════════════════════════════════════════════
# feature_medications
# ══════════════════════════════════════════════════════════════════════════════

def test_feature_medications_creates_rolling_columns(spine, medications):
    result = dp.feature_medications(spine, medications)
    assert "med_is_opioid_sum_7d" in result.columns
    assert "med_n_doses_sum_30d" in result.columns


def test_feature_medications_opioid_flag(spine, medications):
    # _rolling_on_daily first averages within a day, then rolls.
    # 6 doses for R001 on OBS1 (1 opioid) → daily mean of is_opioid = 1/6.
    # Rolling sum of that daily fraction > 0 confirms the opioid was detected.
    result = dp.feature_medications(spine, medications)
    r001_obs1 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS1)]
    assert r001_obs1["med_is_opioid_sum_7d"].iloc[0] > 0


def test_feature_medications_miss_rate(spine, medications):
    # 1 missed out of 6 total → miss_rate > 0
    result = dp.feature_medications(spine, medications)
    r001_obs1 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS1)]
    assert r001_obs1["med_miss_rate_30d"].iloc[0] > 0


# ══════════════════════════════════════════════════════════════════════════════
# feature_document_tags
# ══════════════════════════════════════════════════════════════════════════════

def test_feature_document_tags_creates_tag_columns(spine, document_tags):
    result = dp.feature_document_tags(spine, document_tags)
    assert "tag_fall_risk_n_sum_30d" in result.columns


def test_feature_document_tags_total_highrisk(spine, document_tags):
    result = dp.feature_document_tags(spine, document_tags)
    r001_obs1 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS1)]
    # R001 has fall_risk and pain_intervention tags at OBS1 → total ≥ 2
    assert r001_obs1["tag_total_highrisk_30d"].iloc[0] >= 2


def test_feature_document_tags_excludes_deleted(spine):
    dt = pd.DataFrame({
        "resident_id": ["R001"],
        "tag_id":      ["fall_risk"],
        "created_at":  [OBS1],
        "deleted_at":  [pd.Timestamp("2024-11-02")],   # deleted → excluded
    })
    result = dp.feature_document_tags(spine, dt)
    assert "tag_fall_risk_n_sum_30d" not in result.columns


# ══════════════════════════════════════════════════════════════════════════════
# feature_labs
# ══════════════════════════════════════════════════════════════════════════════

def test_feature_labs_creates_columns(spine, lab_reports):
    result = dp.feature_labs(spine, lab_reports)
    assert "lab_is_abnormal_sum_30d" in result.columns
    assert "lab_is_critical_sum_30d" in result.columns
    assert "lab_n_lab_sum_30d" in result.columns


def test_feature_labs_abnormal_count(spine, lab_reports):
    result = dp.feature_labs(spine, lab_reports)
    r001_obs1 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS1)]
    # R001 has 2 labs on OBS1: both Abnormal/Critical.
    # _rolling_on_daily first takes the daily mean: mean([1,1]) = 1.0, then rolling sum = 1.0.
    assert r001_obs1["lab_is_abnormal_sum_30d"].iloc[0] == pytest.approx(1.0)


def test_feature_labs_critical_count(spine, lab_reports):
    result = dp.feature_labs(spine, lab_reports)
    r001_obs1 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS1)]
    # 1 Critical out of 2 records on OBS1 → daily mean of is_critical = 0.5, rolling sum = 0.5.
    assert r001_obs1["lab_is_critical_sum_30d"].iloc[0] == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════════════════════
# feature_prior_incidents
# ══════════════════════════════════════════════════════════════════════════════

def test_prior_incidents_fall_count_at_obs1(spine, data_dict):
    s = dp.feature_static(spine)
    result = dp.feature_prior_incidents(s, data_dict)
    r001_obs1 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS1)]
    # INC001: Fall for R001 at OBS1 → rolling sum at OBS1 includes it
    assert r001_obs1["prior_fall_n_sum_30d"].iloc[0] >= 1


def test_prior_incidents_days_since_last_fall(spine, data_dict):
    s = dp.feature_static(spine)
    result = dp.feature_prior_incidents(s, data_dict)
    # R001 has a fall at OBS1; at OBS2 (7 days later), days_since = 7
    r001_obs2 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS2)]
    assert r001_obs2["prior_days_since_last_fall"].iloc[0] == 7


def test_prior_incidents_days_since_defaults_to_999_when_no_fall(spine, data_dict):
    s = dp.feature_static(spine)
    result = dp.feature_prior_incidents(s, data_dict)
    # No falls in the fixture for R002 in the past
    r002_obs1 = result[(result["resident_id"] == "R002") & (result["obs_date"] == OBS1)]
    assert r002_obs1["prior_days_since_last_fall"].iloc[0] == 999


def test_prior_incidents_last_fall_had_injury(spine, data_dict):
    s = dp.feature_static(spine)
    result = dp.feature_prior_incidents(s, data_dict)
    # INC001 (fall for R001 at OBS1) is linked to INJ001 → had_injury=1 at OBS2
    r001_obs2 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS2)]
    assert r001_obs2["prior_last_fall_had_injury"].iloc[0] == 1


# ══════════════════════════════════════════════════════════════════════════════
# feature_prior_rth
# ══════════════════════════════════════════════════════════════════════════════

def test_prior_rth_count(spine, hospital_transfers):
    s = dp.feature_static(spine)
    result = dp.feature_prior_rth(s, hospital_transfers)
    r001_obs1 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS1)]
    assert r001_obs1["prior_rth_n_sum_30d"].iloc[0] >= 1


def test_prior_rth_days_since(spine, hospital_transfers):
    s = dp.feature_static(spine)
    result = dp.feature_prior_rth(s, hospital_transfers)
    # Transfer at OBS1; at OBS2 → days_since = 7
    r001_obs2 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS2)]
    assert r001_obs2["prior_rth_days_since_last"].iloc[0] == 7


def test_prior_rth_ever_flag(spine, hospital_transfers):
    s = dp.feature_static(spine)
    result = dp.feature_prior_rth(s, hospital_transfers)
    # Transfer is at OBS1; rolling join finds a record at OBS1 → prior_rth_ever = 1 there.
    r001_obs1 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS1)]
    assert r001_obs1["prior_rth_ever"].iloc[0] == 1


def test_prior_rth_ever_zero_when_no_transfers(spine, hospital_transfers):
    r002 = spine[spine["resident_id"] == "R002"].copy()
    s = dp.feature_static(r002)
    result = dp.feature_prior_rth(s, hospital_transfers)
    assert result["prior_rth_ever"].iloc[0] == 0


# ══════════════════════════════════════════════════════════════════════════════
# feature_care_needs
# ══════════════════════════════════════════════════════════════════════════════

def test_feature_care_needs_active_fall_need(spine, needs):
    result = dp.feature_care_needs(spine, needs)
    r001_obs1 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS1)]
    assert r001_obs1["need_fall_active"].iloc[0] == 1


def test_feature_care_needs_wound_for_r002(spine, needs):
    result = dp.feature_care_needs(spine, needs)
    r002_obs1 = result[(result["resident_id"] == "R002") & (result["obs_date"] == OBS1)]
    assert r002_obs1["need_wound_active"].iloc[0] == 1


def test_feature_care_needs_resolved_need_is_zero():
    single_spine = pd.DataFrame({
        "resident_id": ["R001"], "facility_id": ["F001"],
        "obs_date":    [OBS1],   "admission_date": ADM,
        "date_of_birth": DOB1,  "discharge_date": pd.NaT,
        "deceased_date": pd.NaT, "outpatient": False,
    })
    needs_resolved = pd.DataFrame({
        "resident_id":   ["R001"],
        "need_category": ["Fall"],
        "initiated_at":  [pd.Timestamp("2024-10-01")],
        "resolved_at":   [pd.Timestamp("2024-10-20")],  # resolved before OBS1
        "strikeout":     [False],
    })
    result = dp.feature_care_needs(single_spine, needs_resolved)
    assert result["need_fall_active"].iloc[0] == 0


# ══════════════════════════════════════════════════════════════════════════════
# feature_therapy
# ══════════════════════════════════════════════════════════════════════════════

def test_feature_therapy_active_pt(spine, therapy_tracks):
    result = dp.feature_therapy(spine, therapy_tracks)
    r001_obs1 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS1)]
    assert r001_obs1["therapy_pt_active"].iloc[0] == 1


def test_feature_therapy_no_ot_for_r001(spine, therapy_tracks):
    result = dp.feature_therapy(spine, therapy_tracks)
    r001_obs1 = result[(result["resident_id"] == "R001") & (result["obs_date"] == OBS1)]
    assert r001_obs1["therapy_ot_active"].iloc[0] == 0


def test_feature_therapy_no_pt_for_r002(spine, therapy_tracks):
    result = dp.feature_therapy(spine, therapy_tracks)
    r002_obs1 = result[(result["resident_id"] == "R002") & (result["obs_date"] == OBS1)]
    assert r002_obs1["therapy_pt_active"].iloc[0] == 0
