import json
import pickle

import numpy as np
import pandas as pd
import pytest

import calibration
from config import TARGETS


# ══════════════════════════════════════════════════════════════════════════════
# Fixture: calibration_dir
# Sets up a tmp directory with xgb_*.pkl and oof_*.parquet for each target.
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def calibration_dir(tmp_path, models_dict, training_data, feat_cols):
    models_dir  = tmp_path / "models"
    metrics_dir = tmp_path / "metrics"
    models_dir.mkdir()
    metrics_dir.mkdir()

    features, labels = training_data
    X = features[feat_cols]

    for target in TARGETS:
        # Save model
        model = models_dict[target]
        with open(models_dir / f"xgb_{target}.pkl", "wb") as f:
            pickle.dump(model, f)

        # Create OOF parquet
        rng = np.random.default_rng(99)
        n = len(X)
        oof_pred = model.predict_proba(X)[:, 1]
        oof_df = pd.DataFrame({
            "resident_id": [f"R{i:04d}" for i in range(n)],
            "facility_id": [f"FAC{i % 4}" for i in range(n)],
            "obs_date":    pd.Timestamp("2024-11-01"),
            target:        labels[target].values,
            "oof_pred":    oof_pred,
        })
        oof_df.to_parquet(metrics_dir / f"oof_{target}.parquet", index=False)

    return tmp_path


# ══════════════════════════════════════════════════════════════════════════════
# discover_targets
# ══════════════════════════════════════════════════════════════════════════════

def test_discover_targets_finds_available(calibration_dir):
    found = calibration.discover_targets(calibration_dir)
    assert set(found) == set(TARGETS)


def test_discover_targets_excludes_missing_oof(calibration_dir):
    # Remove the OOF file for the first target
    first_target = TARGETS[0]
    oof_path = calibration_dir / "metrics" / f"oof_{first_target}.parquet"
    oof_path.unlink()

    found = calibration.discover_targets(calibration_dir)
    assert first_target not in found
    # All other targets should still be present
    for t in TARGETS[1:]:
        assert t in found


# ══════════════════════════════════════════════════════════════════════════════
# fit_and_save_calibrators
# ══════════════════════════════════════════════════════════════════════════════

def test_fit_and_save_calibrators_creates_pkl_files(calibration_dir):
    calibration.fit_and_save_calibrators(TARGETS, calibration_dir)
    for target in TARGETS:
        assert (calibration_dir / "models" / f"calibrator_{target}.pkl").exists()


def test_fit_and_save_calibrators_report_has_required_fields(calibration_dir):
    report = calibration.fit_and_save_calibrators(TARGETS, calibration_dir)
    for target in TARGETS:
        assert target in report
        for field in ("pre_cal_brier", "post_cal_brier", "positive_rate", "n_oof"):
            assert field in report[target], f"{target} missing field '{field}'"


def test_fit_and_save_calibrators_brier_nonnegative(calibration_dir):
    report = calibration.fit_and_save_calibrators(TARGETS, calibration_dir)
    for target in TARGETS:
        assert report[target]["pre_cal_brier"]  >= 0
        assert report[target]["post_cal_brier"] >= 0


# ══════════════════════════════════════════════════════════════════════════════
# save_report
# ══════════════════════════════════════════════════════════════════════════════

def _minimal_report():
    return {
        t: {
            "positive_rate":       0.05,
            "baseline_brier":      0.0475,
            "pre_cal_brier":       0.08,
            "pre_cal_bss":         -0.68,
            "pre_cal_mean_pred":   0.45,
            "post_cal_brier":      0.04,
            "post_cal_bss":        0.16,
            "post_cal_mean_pred":  0.05,
            "brier_improvement_pct": 50.0,
            "pr_auc":              0.20,
            "roc_auc":             0.70,
            "n_oof":               240,
        }
        for t in TARGETS
    }


def test_save_report_creates_json(tmp_path):
    calibration.save_report(_minimal_report(), tmp_path)
    assert (tmp_path / "metrics" / "calibration_report.json").exists()


def test_save_report_valid_json(tmp_path):
    calibration.save_report(_minimal_report(), tmp_path)
    with open(tmp_path / "metrics" / "calibration_report.json") as f:
        data = json.load(f)
    assert set(TARGETS).issubset(data.keys())
