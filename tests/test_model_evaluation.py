import json

import numpy as np
import pandas as pd
import pytest

import model_evaluation as me
from config import TARGETS


# ══════════════════════════════════════════════════════════════════════════════
# compute_all_metrics
# ══════════════════════════════════════════════════════════════════════════════

def test_compute_all_metrics_covers_all_targets(artifacts):
    metrics = me.compute_all_metrics(artifacts)
    assert set(metrics.keys()) == set(TARGETS)


def test_compute_all_metrics_auc_in_valid_range(artifacts):
    metrics = me.compute_all_metrics(artifacts)
    for t in TARGETS:
        assert 0 <= metrics[t]["pr_auc"]  <= 1
        assert 0 <= metrics[t]["roc_auc"] <= 1


def test_compute_all_metrics_brier_in_valid_range(artifacts):
    metrics = me.compute_all_metrics(artifacts)
    for t in TARGETS:
        assert 0 <= metrics[t]["brier"] <= 1


def test_compute_all_metrics_threshold_in_zero_one(artifacts):
    metrics = me.compute_all_metrics(artifacts)
    for t in TARGETS:
        assert 0 <= metrics[t]["best_threshold"] <= 1


def test_compute_all_metrics_counts_match_input(artifacts):
    metrics = me.compute_all_metrics(artifacts)
    eval_X = artifacts["X_test"]
    eval_y = artifacts["y_test"]
    for t in TARGETS:
        assert metrics[t]["n_total"]    == len(eval_X)
        assert metrics[t]["n_positive"] == int(eval_y[t].sum())


def test_compute_all_metrics_stores_internal_arrays(artifacts):
    metrics = me.compute_all_metrics(artifacts)
    eval_X = artifacts["X_test"]
    for t in TARGETS:
        assert "_y_prob" in metrics[t]
        assert len(metrics[t]["_y_prob"]) == len(eval_X)


# ══════════════════════════════════════════════════════════════════════════════
# compute_shap_values
# ══════════════════════════════════════════════════════════════════════════════

def test_compute_shap_values_shape(tmp_path, artifacts):
    shap_results = me.compute_shap_values(
        artifacts, tmp_path, tmp_path, n_shap_samples=20
    )
    feat_cols = artifacts["feat_cols"]
    for t in TARGETS:
        sv = shap_results[t]["shap_values"]
        assert sv.ndim == 2
        assert sv.shape[1] == len(artifacts["target_feat_cols"][t])


def test_compute_shap_values_writes_parquet(tmp_path, artifacts):
    me.compute_shap_values(artifacts, tmp_path, tmp_path, n_shap_samples=20)
    for t in TARGETS:
        assert (tmp_path / f"shap_values_{t}.parquet").exists()


def test_compute_shap_values_writes_importance_csv(tmp_path, artifacts):
    me.compute_shap_values(artifacts, tmp_path, tmp_path, n_shap_samples=20)
    for t in TARGETS:
        path = tmp_path / f"feature_importance_{t}.csv"
        assert path.exists()
        df = pd.read_csv(path)
        assert list(df.columns) == ["feature", "mean_abs_shap"]
        assert len(df) == len(artifacts["target_feat_cols"][t])


def test_compute_shap_values_importance_sorted_descending(tmp_path, artifacts):
    me.compute_shap_values(artifacts, tmp_path, tmp_path, n_shap_samples=20)
    for t in TARGETS:
        df = pd.read_csv(tmp_path / f"feature_importance_{t}.csv")
        assert (df["mean_abs_shap"].diff().dropna() <= 0).all()


# ══════════════════════════════════════════════════════════════════════════════
# facility_level_analysis
# ══════════════════════════════════════════════════════════════════════════════

def test_facility_level_analysis_one_row_per_facility(tmp_path, artifacts):
    y_test = artifacts["y_test"]
    fac_df = me.facility_level_analysis(artifacts, y_test, tmp_path, tmp_path)
    assert len(fac_df) == y_test["facility_id"].nunique()


def test_facility_level_analysis_obs_rate_in_zero_one(tmp_path, artifacts):
    y_test = artifacts["y_test"]
    fac_df = me.facility_level_analysis(artifacts, y_test, tmp_path, tmp_path)
    for t in TARGETS:
        col   = f"{t}_obs_rate"
        valid = fac_df[col].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()


def test_facility_level_analysis_writes_csv(tmp_path, artifacts):
    y_test = artifacts["y_test"]
    me.facility_level_analysis(artifacts, y_test, tmp_path, tmp_path)
    assert (tmp_path / "facility_scores.csv").exists()


# ══════════════════════════════════════════════════════════════════════════════
# save_evaluation_report
# ══════════════════════════════════════════════════════════════════════════════

def _make_metrics(artifacts):
    return me.compute_all_metrics(artifacts)


def test_save_evaluation_report_creates_json(tmp_path, artifacts):
    metrics = _make_metrics(artifacts)
    fac_df  = pd.DataFrame({"facility_id": ["F001", "F002"]})
    me.save_evaluation_report(metrics, fac_df, tmp_path)
    assert (tmp_path / "evaluation_report.json").exists()


def test_save_evaluation_report_valid_json(tmp_path, artifacts):
    metrics = _make_metrics(artifacts)
    fac_df  = pd.DataFrame({"facility_id": ["F001"]})
    me.save_evaluation_report(metrics, fac_df, tmp_path)
    with open(tmp_path / "evaluation_report.json") as f:
        report = json.load(f)
    assert set(TARGETS).issubset(report.keys())


def test_save_evaluation_report_contains_required_fields(tmp_path, artifacts):
    metrics = _make_metrics(artifacts)
    fac_df  = pd.DataFrame({"facility_id": ["F001"]})
    me.save_evaluation_report(metrics, fac_df, tmp_path)
    with open(tmp_path / "evaluation_report.json") as f:
        report = json.load(f)
    for t in TARGETS:
        for field in ("pr_auc", "roc_auc", "brier_score", "sensitivity", "specificity"):
            assert field in report[t], f"{t} missing {field}"


def test_save_evaluation_report_serializes_numpy_floats(tmp_path, artifacts):
    # np.float32/float64 values should not raise a TypeError
    metrics = _make_metrics(artifacts)
    fac_df  = pd.DataFrame({"facility_id": ["F001"]})
    me.save_evaluation_report(metrics, fac_df, tmp_path)   # must not raise
