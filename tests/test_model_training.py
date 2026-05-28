import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

import model_training as mt
from config import XGB_BASE_PARAMS, SCALE_POS_WEIGHT_MAX


# ══════════════════════════════════════════════════════════════════════════════
# facility_train_test_split
# ══════════════════════════════════════════════════════════════════════════════

def test_split_no_facility_overlap(training_data):
    features, labels = training_data
    X_tr, X_te, y_tr, y_te, groups, _ = mt.facility_train_test_split(features, labels)
    train_facs = set(y_tr["facility_id"].unique())
    test_facs  = set(y_te["facility_id"].unique())
    assert train_facs.isdisjoint(test_facs)


def test_split_covers_all_rows(training_data):
    features, labels = training_data
    X_tr, X_te, y_tr, y_te, groups, _ = mt.facility_train_test_split(features, labels)
    assert len(X_tr) + len(X_te) == len(features)


def test_split_feat_cols_exclude_metadata(training_data):
    features, labels = training_data
    _, _, _, _, _, feat_cols = mt.facility_train_test_split(features, labels)
    assert "resident_id" not in feat_cols
    assert "facility_id" not in feat_cols
    assert "obs_date"    not in feat_cols


def test_split_groups_match_train_facilities(training_data):
    features, labels = training_data
    _, _, y_tr, _, groups, _ = mt.facility_train_test_split(features, labels)
    assert list(groups) == list(y_tr["facility_id"].values)


def test_split_respects_test_fraction(training_data):
    features, labels = training_data
    total_facs = labels["facility_id"].nunique()
    _, _, y_tr, y_te, _, _ = mt.facility_train_test_split(
        features, labels, test_facility_frac=0.25
    )
    n_test_facs = y_te["facility_id"].nunique()
    expected = max(1, int(total_facs * 0.25))
    assert n_test_facs == expected


# ══════════════════════════════════════════════════════════════════════════════
# _build_model
# ══════════════════════════════════════════════════════════════════════════════

def test_build_model_returns_xgbclassifier():
    model = mt._build_model(XGB_BASE_PARAMS, spw=5.0)
    assert isinstance(model, xgb.XGBClassifier)


def test_build_model_sets_scale_pos_weight():
    model = mt._build_model(XGB_BASE_PARAMS, spw=3.0)
    assert model.get_params()["scale_pos_weight"] == 3.0


def test_build_model_excludes_none_params():
    params = {**XGB_BASE_PARAMS, "scale_pos_weight": None}
    model = mt._build_model(params, spw=2.0)
    # scale_pos_weight=2.0 is the arg, not None from params
    assert model.get_params()["scale_pos_weight"] == 2.0


# ══════════════════════════════════════════════════════════════════════════════
# _train_single_model
# ══════════════════════════════════════════════════════════════════════════════

def _make_train_val(n_train=80, n_val=20, n_feat=5, seed=0):
    rng = np.random.default_rng(seed)
    X_tr  = pd.DataFrame(rng.standard_normal((n_train, n_feat)),
                         columns=[f"f{i}" for i in range(n_feat)])
    y_tr  = pd.Series((rng.random(n_train) > 0.8).astype(int))
    X_val = pd.DataFrame(rng.standard_normal((n_val, n_feat)),
                         columns=[f"f{i}" for i in range(n_feat)])
    y_val = pd.Series((rng.random(n_val) > 0.8).astype(int))
    return X_tr, y_tr, X_val, y_val


def test_train_single_model_returns_fitted_model():
    X_tr, y_tr, X_val, y_val = _make_train_val()
    params = {**XGB_BASE_PARAMS, "n_estimators": 10, "early_stopping_rounds": 5}
    model = mt._train_single_model(X_tr, y_tr, X_val, y_val, params)
    assert isinstance(model, xgb.XGBClassifier)


def test_train_single_model_can_predict_proba():
    X_tr, y_tr, X_val, y_val = _make_train_val()
    params = {**XGB_BASE_PARAMS, "n_estimators": 10, "early_stopping_rounds": 5}
    model = mt._train_single_model(X_tr, y_tr, X_val, y_val, params)
    probs = model.predict_proba(X_val)
    assert probs.shape == (len(X_val), 2)
    assert (probs >= 0).all() and (probs <= 1).all()


# ══════════════════════════════════════════════════════════════════════════════
# train_final_model
# ══════════════════════════════════════════════════════════════════════════════

def test_train_final_model_sets_n_estimators():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.standard_normal((100, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((rng.random(100) > 0.8).astype(int))
    best_iter = 20
    model = mt.train_final_model(X, y, XGB_BASE_PARAMS, best_iter)
    assert model.n_estimators == int(best_iter * 1.1)


def test_train_final_model_can_predict(training_data, feat_cols):
    features, labels = training_data
    X = features[feat_cols]
    y = labels["fall_30d"]
    model = mt.train_final_model(X, y, XGB_BASE_PARAMS, best_iteration=10)
    probs = model.predict_proba(X)[:, 1]
    assert probs.shape == (len(X),)
    assert ((probs >= 0) & (probs <= 1)).all()


# ══════════════════════════════════════════════════════════════════════════════
# _capped_spw
# ══════════════════════════════════════════════════════════════════════════════

def test_capped_spw_balanced_returns_one():
    # 50/50 split → spw should be ~1.0
    y = pd.Series([0, 1] * 50)
    spw = mt._capped_spw(y)
    assert spw == pytest.approx(1.0, abs=0.01)


def test_capped_spw_very_low_positive_rate_is_capped():
    # 1% positive rate → raw spw = 99, capped at SCALE_POS_WEIGHT_MAX
    y = pd.Series([0] * 99 + [1])
    spw = mt._capped_spw(y)
    assert spw == pytest.approx(SCALE_POS_WEIGHT_MAX)


def test_capped_spw_all_same_label_does_not_crash():
    # All zeros → uses max(..., 1e-6) to avoid divide-by-zero
    y = pd.Series([0] * 100)
    spw = mt._capped_spw(y)
    assert spw == pytest.approx(SCALE_POS_WEIGHT_MAX)


# ══════════════════════════════════════════════════════════════════════════════
# fit_calibrator
# ══════════════════════════════════════════════════════════════════════════════

def test_fit_calibrator_returns_isotonic_regression():
    rng = np.random.default_rng(10)
    oof_preds = rng.random(100)
    y_true    = (rng.random(100) > 0.8).astype(int)
    cal = mt.fit_calibrator(oof_preds, y_true)
    assert isinstance(cal, IsotonicRegression)


def test_fit_calibrator_predictions_in_zero_one():
    rng = np.random.default_rng(11)
    oof_preds = rng.random(100)
    y_true    = (rng.random(100) > 0.8).astype(int)
    cal = mt.fit_calibrator(oof_preds, y_true)
    preds = cal.predict(oof_preds)
    assert (preds >= 0).all() and (preds <= 1).all()


# ══════════════════════════════════════════════════════════════════════════════
# calibrated_predict
# ══════════════════════════════════════════════════════════════════════════════

def test_calibrated_predict_output_shape(training_data, feat_cols):
    features, labels = training_data
    X = features[feat_cols]
    y = labels["fall_30d"]
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=42, verbosity=0)
    model.fit(X, y)
    raw = model.predict_proba(X)[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(raw, y.values)
    out = mt.calibrated_predict(model, cal, X)
    assert out.shape == (len(X),)


def test_calibrated_predict_values_in_zero_one(training_data, feat_cols):
    features, labels = training_data
    X = features[feat_cols]
    y = labels["fall_30d"]
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=42, verbosity=0)
    model.fit(X, y)
    raw = model.predict_proba(X)[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(raw, y.values)
    out = mt.calibrated_predict(model, cal, X)
    assert (out >= 0).all() and (out <= 1).all()


# ══════════════════════════════════════════════════════════════════════════════
# evaluate_on_test
# ══════════════════════════════════════════════════════════════════════════════

def _make_tiny_calibrator(model, X, y):
    raw = model.predict_proba(X)[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(raw, y.values)
    return cal


def test_evaluate_on_test_returns_required_keys(models_dict, eval_X, eval_y):
    model = models_dict["fall_30d"]
    cal   = _make_tiny_calibrator(model, eval_X, eval_y["fall_30d"])
    result = mt.evaluate_on_test(model, cal, eval_X, eval_y["fall_30d"], "fall_30d")
    for key in ("test_pr_auc", "test_roc_auc", "test_brier_raw",
                "test_brier_calibrated", "test_brier_skill_score",
                "test_positive_rate", "test_n_rows"):
        assert key in result, f"Missing key: {key}"


def test_evaluate_on_test_metrics_in_valid_range(models_dict, eval_X, eval_y):
    model = models_dict["fall_30d"]
    cal   = _make_tiny_calibrator(model, eval_X, eval_y["fall_30d"])
    result = mt.evaluate_on_test(model, cal, eval_X, eval_y["fall_30d"], "fall_30d")
    assert 0 <= result["test_pr_auc"]           <= 1
    assert 0 <= result["test_roc_auc"]          <= 1
    assert 0 <= result["test_brier_raw"]        <= 1
    assert 0 <= result["test_brier_calibrated"] <= 1
    assert result["test_n_rows"] == len(eval_X)
