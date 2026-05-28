"""
model_training.py
────────────────────────────────────────
Trains XGBoost binary classifiers for each outcome and applies post-training
isotonic calibration to correct systematic probability over-prediction.

Additional fixes
──────────────────────────
1. scale_pos_weight cap (SCALE_POS_WEIGHT_MAX = 10)
   The raw ratio n_negative/n_positive reached 49× for rth_60d (2% positive
   rate), pushing all predicted probabilities far above their true values
   (Brier Skill Score was -9.4 in previous version).  Capping at 10× dramatically reduces
   this distortion while still up-weighting the minority class.

2. Per-target feature exclusion (EXCLUDE_FEATURES_BY_TARGET)
   facility_id_enc was the 2nd-ranked SHAP feature in the wound model —
   it encoded memorized facility effects that do not generalize to unseen
   facilities.  It is dropped for wound_60d before training.

3. Post-training isotonic calibration using OOF predictions
   After the full CV run, an IsotonicRegression is fitted on the assembled
   out-of-fold predicted probabilities vs true labels.  This non-parametric
   calibrator is saved alongside the XGBoost model and applied transparently
   at evaluation and scoring time.  Isotonic regression is chosen over Platt
   scaling (sigmoid) because the miscalibration shape is non-linear.

Run:  python model_training.py
"""

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OUTPUT_DIR, TARGETS,
    XGB_BASE_PARAMS, OPTUNA_SEARCH_SPACE,
    OPTUNA_N_TRIALS, N_CV_FOLDS, EARLY_STOPPING_ROUNDS,
    SCALE_POS_WEIGHT_MAX, EXCLUDE_FEATURES_BY_TARGET,
    CALIBRATION_METHOD,
)

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Device detection ───────────────────────────────────────────────────────────
# Probe XGBoost with device="cuda" on a tiny dataset.  If no CUDA GPU is
# available XGBoost raises immediately (no timeout), so the fallback is instant.

def _detect_device() -> str:
    try:
        probe = xgb.XGBClassifier(device="cuda", n_estimators=1, verbosity=0)
        probe.fit([[0], [1]], [0, 1])
        return "cuda"
    except Exception:
        return "cpu"


DEVICE: str = _detect_device()


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING AND SPLITTING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(output_dir / "features.parquet")
    labels   = pd.read_parquet(output_dir / "labels.parquet")
    print(f"  Features: {features.shape}  |  Labels: {labels.shape}")
    return features, labels


def facility_train_test_split(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    test_facility_frac: float = 0.20,
    random_state: int = 42,
) -> tuple:
    facilities = labels["facility_id"].unique()
    rng = np.random.default_rng(random_state)
    n_test = max(1, int(len(facilities) * test_facility_frac))
    test_facilities  = set(rng.choice(facilities, size=n_test, replace=False))
    train_facilities = set(facilities) - test_facilities

    train_mask = labels["facility_id"].isin(train_facilities)
    test_mask  = labels["facility_id"].isin(test_facilities)

    meta      = ["resident_id", "facility_id", "obs_date"]
    feat_cols = [c for c in features.columns if c not in meta and c != "resident_id"]

    X_train = features.loc[train_mask.values, feat_cols].reset_index(drop=True)
    X_test  = features.loc[test_mask.values,  feat_cols].reset_index(drop=True)
    y_train = labels.loc[train_mask].reset_index(drop=True)
    y_test  = labels.loc[test_mask].reset_index(drop=True)
    groups  = y_train["facility_id"].values

    print(f"  Train: {len(X_train):,} rows | {len(train_facilities)} facilities")
    print(f"  Test:  {len(X_test):,} rows  | {len(test_facilities)} facilities")
    return X_train, X_test, y_train, y_test, groups, feat_cols


# ══════════════════════════════════════════════════════════════════════════════
# 2. HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _capped_spw(y: pd.Series) -> float:
    """
    Compute scale_pos_weight = n_neg / n_pos, capped at SCALE_POS_WEIGHT_MAX.

    Without the cap, a 2% positive-rate target gets spw = 49×, which inflates
    all predicted probabilities far above their true values — confirmed by a
    Brier Skill Score of -9.4 in previous version.  The cap limits this distortion; residual
    miscalibration is corrected by isotonic calibration post-training.
    """
    pos_rate = y.mean()
    raw_spw  = (1.0 - pos_rate) / max(pos_rate, 1e-6)
    return min(raw_spw, SCALE_POS_WEIGHT_MAX)


def _build_model(params: dict, spw: float) -> xgb.XGBClassifier:
    p = {k: v for k, v in {**params, "scale_pos_weight": spw, "device": DEVICE}.items()
         if v is not None}
    return xgb.XGBClassifier(**p)


def _train_single_model(
    X_tr: pd.DataFrame, y_tr: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    params: dict,
) -> xgb.XGBClassifier:
    model = _build_model(params, _capped_spw(y_tr))
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 3. ISOTONIC CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def fit_calibrator(oof_predictions: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    """
    Fit an isotonic regression calibrator on the out-of-fold predictions.

    IsotonicRegression is monotone and non-parametric — it can correct any
    shape of miscalibration, including the severe over-prediction (raw scores
    clustered around 0.45 against true rates of 0.02-0.09) seen in previous version.
    out_of_bounds='clip' ensures scores outside the training range are handled
    gracefully at inference time.
    """
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(oof_predictions, y_true)
    return cal


def calibrated_predict(
    model: xgb.XGBClassifier,
    calibrator: IsotonicRegression,
    X: pd.DataFrame,
) -> np.ndarray:
    """Apply model then isotonic calibrator — single call used throughout."""
    raw = model.predict_proba(X)[:, 1]
    return calibrator.predict(raw)


# ══════════════════════════════════════════════════════════════════════════════
# 4. OPTUNA HYPERPARAMETER SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def _make_objective(X, y, groups, base_params):
    def objective(trial: optuna.Trial) -> float:
        params = {
            **base_params,
            "learning_rate":    trial.suggest_float("learning_rate",    *OPTUNA_SEARCH_SPACE["learning_rate"],    log=True),
            "max_depth":        trial.suggest_int(  "max_depth",        *OPTUNA_SEARCH_SPACE["max_depth"]),
            "min_child_weight": trial.suggest_int(  "min_child_weight", *OPTUNA_SEARCH_SPACE["min_child_weight"]),
            "subsample":        trial.suggest_float("subsample",        *OPTUNA_SEARCH_SPACE["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *OPTUNA_SEARCH_SPACE["colsample_bytree"]),
            "reg_alpha":        trial.suggest_float("reg_alpha",        *OPTUNA_SEARCH_SPACE["reg_alpha"]),
            "reg_lambda":       trial.suggest_float("reg_lambda",       *OPTUNA_SEARCH_SPACE["reg_lambda"]),
        }
        gkf = GroupKFold(n_splits=N_CV_FOLDS)
        pr_aucs = []
        for tr_idx, val_idx in gkf.split(X, y, groups):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            model = _train_single_model(X_tr, y_tr, X_val, y_val, params)
            preds = model.predict_proba(X_val)[:, 1]
            pr_aucs.append(average_precision_score(y_val, preds))
        return np.mean(pr_aucs)
    return objective


def tune_hyperparameters(X_train, y_train, groups, target_name) -> dict:
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(_make_objective(X_train, y_train, groups, XGB_BASE_PARAMS),
                   n_trials=OPTUNA_N_TRIALS, show_progress_bar=True)
    best = {**XGB_BASE_PARAMS, **study.best_params}
    print(f"    Best PR-AUC ({target_name}): {study.best_value:.4f} | "
          f"lr={best['learning_rate']:.3f}  depth={best['max_depth']}  "
          f"min_child_w={best['min_child_weight']}")
    return best


# ══════════════════════════════════════════════════════════════════════════════
# 5. CROSS-VALIDATED TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate_model(X_train, y_train, groups, params, target_name) -> dict:
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    oof_preds    = np.zeros(len(y_train))
    fold_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model = _train_single_model(X_tr, y_tr, X_val, y_val, params)
        preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds

        pr_auc = average_precision_score(y_val, preds)
        roc    = roc_auc_score(y_val, preds)
        brier  = brier_score_loss(y_val, preds)

        fold_metrics.append({
            "fold": fold + 1, "target": target_name,
            "n_val_rows": len(y_val),
            "n_val_facilities": len(np.unique(groups[val_idx])),
            "pr_auc": pr_auc, "roc_auc": roc, "brier_score": brier,
            "positive_rate": float(y_val.mean()),
            "best_iteration": model.best_iteration + 1,
        })
        print(f"    Fold {fold+1}/{N_CV_FOLDS}  PR-AUC={pr_auc:.4f}  "
              f"ROC-AUC={roc:.4f}  Brier={brier:.4f}  "
              f"best_iter={model.best_iteration+1}")

    oof_pr_auc = average_precision_score(y_train, oof_preds)
    oof_roc    = roc_auc_score(y_train, oof_preds)
    print(f"  ▸ OOF  PR-AUC={oof_pr_auc:.4f}  ROC-AUC={oof_roc:.4f}  [{target_name}]")
    return {"fold_metrics": fold_metrics, "oof_predictions": oof_preds,
            "oof_pr_auc": oof_pr_auc, "oof_roc_auc": oof_roc}


# ══════════════════════════════════════════════════════════════════════════════
# 6. FINAL MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_final_model(X_train, y_train, params, best_iteration) -> xgb.XGBClassifier:
    final_params = {
        **params,
        "n_estimators":          int(best_iteration * 1.1),
        "early_stopping_rounds": None,
    }
    final_params = {k: v for k, v in final_params.items() if v is not None}
    model = xgb.XGBClassifier(**{**final_params, "scale_pos_weight": _capped_spw(y_train)})
    model.fit(X_train, y_train, verbose=False)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 7. TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_on_test(model, calibrator, X_test, y_test, target_name) -> dict:
    # Evaluate both raw (pre-calibration) and calibrated probabilities
    raw_preds = model.predict_proba(X_test)[:, 1]
    cal_preds = calibrator.predict(raw_preds)

    raw_brier = brier_score_loss(y_test, raw_preds)
    cal_brier = brier_score_loss(y_test, cal_preds)
    pr_auc    = average_precision_score(y_test, cal_preds)
    roc       = roc_auc_score(y_test, cal_preds)

    print(f"  [{target_name}]  PR-AUC={pr_auc:.4f}  ROC-AUC={roc:.4f}  "
          f"Brier raw={raw_brier:.4f} → calibrated={cal_brier:.4f}")
    baseline_brier = float(y_test.mean()) * (1 - float(y_test.mean()))
    bss = 1 - cal_brier / max(baseline_brier, 1e-9)
    print(f"           Brier Skill Score (calibrated): {bss:.3f}  "
          f"(baseline={baseline_brier:.4f})")

    return {
        "target": target_name,
        "test_pr_auc":          pr_auc,
        "test_roc_auc":         roc,
        "test_brier_raw":       raw_brier,
        "test_brier_calibrated": cal_brier,
        "test_brier_skill_score": bss,
        "test_positive_rate":   float(y_test.mean()),
        "test_n_rows":          int(len(y_test)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    models_dir  = OUTPUT_DIR / "models"
    metrics_dir = OUTPUT_DIR / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Device: {DEVICE.upper()} ({'GPU' if DEVICE == 'cuda' else 'CPU'} training)")
    print("\n[1] Loading prepared data …")
    features, labels = load_data(OUTPUT_DIR)

    print("\n[2] Facility-held-out train/test split …")
    X_train, X_test, y_train_df, y_test_df, groups, feat_cols = \
        facility_train_test_split(features, labels)

    with open(models_dir / "feature_cols.json", "w") as f:
        json.dump(feat_cols, f)
    y_test_df.to_parquet(metrics_dir  / "y_test.parquet",   index=False)
    X_test.to_parquet(metrics_dir     / "X_test.parquet",   index=False)

    all_cv_results   = {}
    all_test_results = {}

    for target in TARGETS:
        print(f"\n{'═'*60}")
        print(f"  TARGET: {target}")
        print(f"{'═'*60}")

        # ── Drop rows with NaN labels (censored for this target's horizon) ──
        valid_mask   = y_train_df[target].notna()
        y_tr_full    = y_train_df.loc[valid_mask, target].astype(int)
        X_tr_full    = X_train.loc[valid_mask.values].reset_index(drop=True)
        groups_valid = groups[valid_mask.values]
        y_tr_full    = y_tr_full.reset_index(drop=True)

        valid_test   = y_test_df[target].notna()
        y_te         = y_test_df.loc[valid_test, target].astype(int)
        X_te         = X_test.loc[valid_test.values].reset_index(drop=True)

        print(f"  Train rows (non-censored): {len(X_tr_full):,}  "
              f"positive rate: {y_tr_full.mean():.2%}")
        print(f"  Test  rows (non-censored): {len(X_te):,}  "
              f"positive rate: {y_te.mean():.2%}")

        # ── Apply per-target feature exclusions ───────────────────────────
        excluded = EXCLUDE_FEATURES_BY_TARGET.get(target, [])
        if excluded:
            print(f"  Excluding features: {excluded}")
        target_feat_cols = [c for c in feat_cols if c not in excluded]
        X_tr = X_tr_full[target_feat_cols]
        X_te_filtered = X_te[target_feat_cols]

        # ── Save per-target feature list ──────────────────────────────────
        with open(models_dir / f"feature_cols_{target}.json", "w") as f:
            json.dump(target_feat_cols, f)

        # ── Hyperparameter tuning ─────────────────────────────────────────
        print(f"\n  [a] Optuna search ({OPTUNA_N_TRIALS} trials) …")
        best_params = tune_hyperparameters(X_tr, y_tr_full, groups_valid, target)

        # ── Cross-validation ──────────────────────────────────────────────
        print(f"\n  [b] Facility-held-out CV ({N_CV_FOLDS} folds) …")
        cv_results = cross_validate_model(X_tr, y_tr_full, groups_valid,
                                          best_params, target)
        all_cv_results[target] = cv_results

        # ── Isotonic calibration on OOF predictions ───────────────────────
        print(f"\n  [c] Fitting isotonic calibrator on OOF predictions …")
        calibrator = fit_calibrator(cv_results["oof_predictions"], y_tr_full.values)
        cal_oof_bss = 1 - brier_score_loss(y_tr_full, calibrator.predict(cv_results["oof_predictions"])) \
                      / max(y_tr_full.mean() * (1 - y_tr_full.mean()), 1e-9)
        print(f"    OOF Brier Skill Score after calibration: {cal_oof_bss:.3f}")

        # ── Final model ───────────────────────────────────────────────────
        avg_iter = int(np.mean([m["best_iteration"] for m in cv_results["fold_metrics"]]))
        n_final  = int(avg_iter * 1.1)
        print(f"\n  [d] Training final model (n_estimators={n_final}) …")
        final_model = train_final_model(X_tr, y_tr_full, best_params, avg_iter)

        # ── Test evaluation ───────────────────────────────────────────────
        print(f"\n  [e] Test-set evaluation …")
        test_results = evaluate_on_test(final_model, calibrator,
                                        X_te_filtered, y_te, target)
        test_results["oof_pr_auc"]  = cv_results["oof_pr_auc"]
        test_results["oof_roc_auc"] = cv_results["oof_roc_auc"]
        all_test_results[target] = test_results

        # ── Persist model and calibrator ──────────────────────────────────
        with open(models_dir / f"xgb_{target}.pkl", "wb") as f:
            pickle.dump(final_model, f)
        with open(models_dir / f"calibrator_{target}.pkl", "wb") as f:
            pickle.dump(calibrator, f)
        print(f"  ✓ Model + calibrator saved → {models_dir}")

        serialisable = {k: v for k, v in best_params.items()
                        if isinstance(v, (int, float, str, bool, type(None)))}
        with open(models_dir / f"params_{target}.json", "w") as f:
            json.dump(serialisable, f, indent=2)

        oof_df = y_train_df.loc[valid_mask, ["resident_id", "facility_id", "obs_date", target]].copy()
        oof_df["oof_pred"] = cv_results["oof_predictions"]
        oof_df.to_parquet(metrics_dir / f"oof_{target}.parquet", index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  SUMMARY (calibrated test metrics)")
    print(f"{'═'*60}")
    rows = []
    for t in TARGETS:
        r = all_test_results[t]
        rows.append({
            "target":        t,
            "oof_pr_auc":    f"{r['oof_pr_auc']:.4f}",
            "test_pr_auc":   f"{r['test_pr_auc']:.4f}",
            "test_roc_auc":  f"{r['test_roc_auc']:.4f}",
            "brier_raw":     f"{r['test_brier_raw']:.4f}",
            "brier_cal":     f"{r['test_brier_calibrated']:.4f}",
            "bss":           f"{r['test_brier_skill_score']:.3f}",
            "pos_rate":      f"{r['test_positive_rate']:.2%}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    pd.DataFrame(rows).to_csv(metrics_dir / "training_summary.csv", index=False)
    pd.DataFrame([m for t in TARGETS for m in all_cv_results[t]["fold_metrics"]]) \
      .to_csv(metrics_dir / "cv_fold_metrics.csv", index=False)

    return all_test_results


if __name__ == "__main__":
    import time
    t0 = time.time()
    main()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
