"""
model_evaluation.py
──────────────────────────────────────────
Comprehensive model evaluation on held-out test facilities.

XGBoost-specific notes
───────────────────────
• XGBoost's built-in feature_importances_ uses 'weight' (split count) by
  default; we use mean |SHAP| throughout for more reliable importance ranking.

Produces
────────
  outputs/plots/roc_pr_curves.png
  outputs/plots/calibration.png
  outputs/plots/shap_summary_{target}.png
  outputs/plots/shap_importance_{target}.png
  outputs/plots/shap_dependence_{target}.png
  outputs/plots/shap_cross_target.png
  outputs/plots/score_distributions.png
  outputs/plots/facility_risk_scatter.png
  outputs/metrics/evaluation_report.json
  outputs/metrics/facility_scores.csv
  outputs/metrics/shap_values_{target}.parquet
  outputs/metrics/feature_importance_{target}.csv

Run:  python model_evaluation.py
"""

import json
import pickle
import sys
import warnings
from pathlib import Path

import copy as _copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Workaround for matplotlib 3.10.x regression: Path.__deepcopy__ crashes when
# tick_params leaves a MarkerStyle whose _path.__dict__ is not iterable.
# The fix copies via __slots__ when the normal __dict__ iteration fails.
def _safe_path_deepcopy(self, memo):
    cls = type(self)
    memo[id(self)] = p = cls.__new__(cls)
    d = self.__dict__ if isinstance(self.__dict__, dict) else {}
    for k, v in d.items():
        setattr(p, k, _copy.deepcopy(v, memo))
    p._readonly = False
    return p

matplotlib.path.Path.__deepcopy__ = _safe_path_deepcopy
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    roc_curve, precision_recall_curve,
    brier_score_loss, log_loss,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR, TARGETS, LABEL_HORIZONS

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})

TARGET_LABELS = {
    "fall_30d":  "Fall (30-day)",
    "rth_60d":   "Return-to-Hospital (60-day)",
    "wound_60d": "Wound (60-day)",
}
TARGET_COLORS = {
    "fall_30d":  "#5B6ABD",
    "rth_60d":   "#C1453A",
    "wound_60d": "#2A9D8F",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_artifacts(output_dir: Path) -> dict:
    """Load models (xgb_*.pkl), calibrators, test data, and per-target feature column lists."""
    models_dir  = output_dir / "models"
    metrics_dir = output_dir / "metrics"

    with open(models_dir / "feature_cols.json") as f:
        feat_cols = json.load(f)

    X_test = pd.read_parquet(metrics_dir / "X_test.parquet")
    y_test = pd.read_parquet(metrics_dir / "y_test.parquet")

    models, calibrators, oof, target_feat_cols = {}, {}, {}, {}
    for target in TARGETS:
        with open(models_dir / f"xgb_{target}.pkl", "rb") as f:
            models[target] = pickle.load(f)

        cal_path = models_dir / f"calibrator_{target}.pkl"
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                calibrators[target] = pickle.load(f)
        else:
            print(f"  WARNING: no calibrator found for {target} — using raw probabilities")
            calibrators[target] = None

        # Per-target feature list (may exclude facility_id_enc for wound, etc.)
        tfc_path = models_dir / f"feature_cols_{target}.json"
        if tfc_path.exists():
            with open(tfc_path) as f:
                target_feat_cols[target] = json.load(f)
        else:
            target_feat_cols[target] = feat_cols

        oof_path = metrics_dir / f"oof_{target}.parquet"
        if oof_path.exists():
            oof[target] = pd.read_parquet(oof_path)

    print(f"  X_test: {X_test.shape}  |  y_test: {y_test.shape}")
    print(f"  Models: {list(models.keys())}  |  Calibrators: {[t for t,c in calibrators.items() if c]}")
    return {
        "models":           models,
        "calibrators":      calibrators,
        "X_test":           X_test[feat_cols],
        "y_test":           y_test,
        "feat_cols":        feat_cols,
        "target_feat_cols": target_feat_cols,
        "oof":              oof,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLASSIFICATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def _predict(artifacts: dict, target: str) -> np.ndarray:
    """
    Route predictions through the isotonic calibrator when available.
    All evaluation functions use this helper so calibration is applied
    consistently — switching calibration on/off in one place affects everything.
    """
    model      = artifacts["models"][target]
    calibrator = artifacts["calibrators"].get(target)
    feat_cols  = artifacts["target_feat_cols"][target]
    X          = artifacts["X_test"][feat_cols]
    raw = model.predict_proba(X)[:, 1]
    return calibrator.predict(raw) if calibrator is not None else raw


def compute_all_metrics(artifacts: dict) -> dict:
    """Threshold-free and threshold-dependent metrics on calibrated probabilities."""
    models, X_test_full, y_test = \
        artifacts["models"], artifacts["X_test"], artifacts["y_test"]
    results = {}
    for target in TARGETS:
        y_true = y_test[target].dropna().astype(int)
        valid_idx = y_test[target].notna()
        y_prob = _predict(artifacts, target)[valid_idx.values]

        pr_auc  = average_precision_score(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        brier   = brier_score_loss(y_true, y_prob)
        logloss = log_loss(y_true, y_prob)

        prec, rec, thr = precision_recall_curve(y_true, y_prob)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        best_idx = np.argmax(f1[:-1])
        best_thr = thr[best_idx]
        y_pred   = (y_prob >= best_thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        results[target] = {
            "pr_auc":         pr_auc,
            "roc_auc":        roc_auc,
            "brier":          brier,
            "log_loss":       logloss,
            "best_threshold": best_thr,
            "f1_at_best_thr": f1[best_idx],
            "sensitivity":    tp / max(tp + fn, 1),
            "specificity":    tn / max(tn + fp, 1),
            "ppv":            tp / max(tp + fp, 1),
            "npv":            tn / max(tn + fn, 1),
            "positive_rate":  float(y_true.mean()),
            "n_positive":     int(y_true.sum()),
            "n_total":        int(len(y_true)),
            # stored for plot functions
            "_prec": prec, "_rec": rec, "_thr": thr, "_y_prob": y_prob,
        }
        m = results[target]
        print(f"  {target:12s}  PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}  "
              f"Brier={brier:.4f}  Sens={m['sensitivity']:.3f}  Spec={m['specificity']:.3f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. ROC / PR CURVES
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_pr_curves(
    metrics: dict,
    y_test: pd.DataFrame,
    plots_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Random")
    for t in TARGETS:
        y_true = y_test[t]
        fpr, tpr, _ = roc_curve(y_true, metrics[t]["_y_prob"])
        ax.plot(fpr, tpr, lw=2, color=TARGET_COLORS[t],
                label=f"{TARGET_LABELS[t]} (AUC={metrics[t]['roc_auc']:.3f})")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC curves — held-out test facilities")
    ax.legend(fontsize=9, loc="lower right")

    ax = axes[1]
    for t in TARGETS:
        y_true   = y_test[t]
        pos_rate = metrics[t]["positive_rate"]
        ax.plot(metrics[t]["_rec"], metrics[t]["_prec"], lw=2, color=TARGET_COLORS[t],
                label=f"{TARGET_LABELS[t]} (AP={metrics[t]['pr_auc']:.3f})")
        ax.axhline(pos_rate, color=TARGET_COLORS[t], lw=0.6, ls=":", alpha=0.5)
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Precision-Recall curves — held-out test facilities")
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    path = plots_dir / "roc_pr_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_calibration(
    metrics: dict,
    y_test: pd.DataFrame,
    plots_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, t in zip(axes, TARGETS):
        y_true = y_test[t]
        y_prob = metrics[t]["_y_prob"]
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Perfect")
        ax.plot(mean_pred, frac_pos, "o-", lw=2, ms=6,
                color=TARGET_COLORS[t], label="Model")
        ax.fill_between(mean_pred, frac_pos,
                        [np.interp(x, [0, 1], [0, 1]) for x in mean_pred],
                        alpha=0.15, color=TARGET_COLORS[t])
        ax.set(xlabel="Mean predicted probability", ylabel="Observed event rate",
               xlim=(0, 1), ylim=(0, 1),
               title=f"{TARGET_LABELS[t]}\nBrier = {metrics[t]['brier']:.4f}")
        ax.legend(fontsize=8)
    plt.suptitle("Calibration plots — held-out test facilities", y=1.02, fontsize=12)
    plt.tight_layout()
    path = plots_dir / "calibration.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SHAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap_values(
    artifacts: dict,
    metrics_dir: Path,
    plots_dir: Path,
    n_shap_samples: int = 2000,
) -> dict:
    """
    Compute SHAP values via TreeExplainer (exact, fast for XGBoost trees).
    Uses per-target feature columns; SHAP values reflect raw model log-odds
    (calibration is monotone so feature rankings are preserved).
    """
    models     = artifacts["models"]
    X_test_all = artifacts["X_test"]
    feat_cols  = artifacts["feat_cols"]

    rng = np.random.default_rng(42)
    n   = min(n_shap_samples, len(X_test_all))
    idx = rng.choice(len(X_test_all), size=n, replace=False)
    X_sample_all = X_test_all.iloc[idx].reset_index(drop=True)

    shap_results = {}

    for target in TARGETS:
        model = models[target]
        target_cols = artifacts["target_feat_cols"][target]
        X_sample = X_sample_all[target_cols]
        print(f"  Computing SHAP for {target} (n={n}, features={len(target_cols)}) …")

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        assert sv.ndim == 2 and sv.shape[1] == len(target_cols), \
            f"Unexpected SHAP shape {sv.shape} for {target}"

        # Persist raw SHAP matrix (using target-specific feature columns)
        pd.DataFrame(sv, columns=target_cols).to_parquet(
            metrics_dir / f"shap_values_{target}.parquet", index=False
        )

        # Feature importance: mean |SHAP|
        mean_abs = np.abs(sv).mean(axis=0)
        imp_df = (pd.DataFrame({"feature": target_cols, "mean_abs_shap": mean_abs})
                  .sort_values("mean_abs_shap", ascending=False)
                  .reset_index(drop=True))
        imp_df.to_csv(metrics_dir / f"feature_importance_{target}.csv", index=False)

        shap_results[target] = {
            "shap_values": sv,
            "X_sample":    X_sample,
            "explainer":   explainer,
            "feat_cols":   target_cols,   # per-target columns (may differ from global list)
        }

        # ── Plot: beeswarm summary ────────────────────────────────────────
        plt.figure(figsize=(9, 8))
        shap.summary_plot(sv, X_sample, feature_names=target_cols,
                          max_display=25, show=False, plot_type="dot")
        plt.title(f"SHAP summary — {TARGET_LABELS[target]}\n(n={n})", fontsize=11, pad=12)
        plt.tight_layout()
        plt.savefig(plots_dir / f"shap_summary_{target}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # ── Plot: mean |SHAP| bar chart (top 20) ─────────────────────────
        top20 = imp_df.head(20)
        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(top20["feature"][::-1], top20["mean_abs_shap"][::-1],
                       color=TARGET_COLORS[target], alpha=0.85)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Top 20 features — {TARGET_LABELS[target]}", fontsize=11)
        ax.set_xlim(0, top20["mean_abs_shap"].max() * 1.15)
        for bar, val in zip(bars, top20["mean_abs_shap"][::-1]):
            ax.text(val + ax.get_xlim()[1] * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(plots_dir / f"shap_importance_{target}.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"    Top 5: {imp_df['feature'].head(5).tolist()}")

    # ── Cross-target normalised importance bar chart ──────────────────────
    print("  Plotting cross-target feature importance …")
    combined = {}
    for t in TARGETS:
        imp = (pd.read_csv(metrics_dir / f"feature_importance_{t}.csv")
               .set_index("feature")["mean_abs_shap"])
        imp = imp / imp.max()
        for feat, val in imp.items():
            combined.setdefault(feat, {})[t] = val

    cdf = pd.DataFrame(combined).T.fillna(0)
    cdf["total"] = cdf.sum(axis=1)
    cdf = cdf.sort_values("total", ascending=True).tail(25)

    fig, ax = plt.subplots(figsize=(10, 8))
    x = np.arange(len(cdf))
    w = 0.25
    for i, t in enumerate(TARGETS):
        if t in cdf.columns:
            ax.barh(x + (i - 1) * w, cdf[t], height=w,
                    label=TARGET_LABELS[t], color=TARGET_COLORS[t], alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(cdf.index, fontsize=9)
    ax.set_xlabel("Normalised mean |SHAP|")
    ax.set_title("Cross-target feature importance (top features across all outcomes)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_cross_target.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✓ SHAP plots → {plots_dir}")
    return shap_results


# ══════════════════════════════════════════════════════════════════════════════
# 6. SHAP DEPENDENCE PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_shap_dependence(shap_results: dict, feat_cols: list[str],
                         plots_dir: Path, n_top: int = 4) -> None:
    """Partial dependence via SHAP for the top N features per target."""
    for target in TARGETS:
        sv         = shap_results[target]["shap_values"]
        X_sample   = shap_results[target]["X_sample"]
        tcols      = shap_results[target]["feat_cols"]   # per-target column list
        mean_abs   = np.abs(sv).mean(axis=0)
        top_feats  = [tcols[i] for i in np.argsort(mean_abs)[::-1][:n_top]]

        fig, axes = plt.subplots(1, n_top, figsize=(5 * n_top, 4))
        if n_top == 1:
            axes = [axes]

        for ax, feat in zip(axes, top_feats):
            idx    = tcols.index(feat)
            x_vals = X_sample[feat].values
            s_vals = sv[:, idx]
            sc = ax.scatter(x_vals, s_vals, c=x_vals, cmap="RdBu_r",
                            alpha=0.4, s=8, rasterized=True)
            ax.axhline(0, color="black", lw=0.5)
            ax.set_xlabel(feat, fontsize=9)
            ax.set_ylabel("SHAP value", fontsize=9)
            ax.set_title(feat.replace("_", " "), fontsize=9)
            plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)

        fig.suptitle(f"SHAP dependence — {TARGET_LABELS[target]}", fontsize=11)
        plt.tight_layout()
        plt.savefig(plots_dir / f"shap_dependence_{target}.png", dpi=150, bbox_inches="tight")
        plt.close()
    print(f"  ✓ Dependence plots → {plots_dir}")


def facility_level_analysis(
    artifacts: dict,
    y_test: pd.DataFrame,
    metrics_dir: Path,
    plots_dir: Path,
) -> pd.DataFrame:
    """
    Per-facility calibrated predicted risk vs observed event rate.
    Facilities above the diagonal have higher observed rates than predicted.
    """
    rows = []
    for fac_id in y_test["facility_id"].unique():
        mask  = y_test["facility_id"] == fac_id
        fac_y = y_test[mask]
        row   = {"facility_id": fac_id, "n_obs": int(mask.sum())}

        for t in TARGETS:
            y_true = fac_y[t]
            row[f"{t}_obs_rate"] = float(y_true.mean())
            fac_feat_cols = artifacts["target_feat_cols"][t]
            fac_X = artifacts["X_test"].loc[mask.values, fac_feat_cols]
            if fac_X.empty or y_true.sum() == 0:
                row[f"{t}_mean_pred"] = np.nan
                row[f"{t}_roc_auc"]   = np.nan
                continue
            calibrator = artifacts["calibrators"].get(t)
            raw   = artifacts["models"][t].predict_proba(fac_X)[:, 1]
            y_prob = calibrator.predict(raw) if calibrator else raw
            row[f"{t}_mean_pred"] = float(y_prob.mean())
            row[f"{t}_roc_auc"]   = roc_auc_score(y_true, y_prob) \
                if y_true.nunique() == 2 else np.nan

        rows.append(row)

    sort_col = f"{TARGETS[0]}_mean_pred"
    facility_df = (pd.DataFrame(rows).sort_values(sort_col, ascending=False))
    facility_df.to_csv(metrics_dir / "facility_scores.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, t in zip(axes, TARGETS):
        valid = facility_df.dropna(subset=[f"{t}_mean_pred", f"{t}_obs_rate"])
        x, y  = valid[f"{t}_mean_pred"], valid[f"{t}_obs_rate"]
        ax.scatter(x, y, alpha=0.7, s=40, color=TARGET_COLORS[t])
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.4)
        ax.set(xlabel="Mean calibrated predicted risk", ylabel="Observed event rate",
               title=f"{TARGET_LABELS[t]}\n(one dot = one facility)")
        div = (y - x).abs()
        for _, r in valid[div > div.quantile(0.9)].iterrows():
            ax.annotate(str(r["facility_id"])[-4:],
                        (r[f"{t}_mean_pred"], r[f"{t}_obs_rate"]),
                        fontsize=7, alpha=0.8)
    plt.suptitle("Facility risk: calibrated predicted vs observed\n"
                 "(points above diagonal → potentially under-reported)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "facility_risk_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Facility analysis → {metrics_dir / 'facility_scores.csv'}")
    return facility_df


def plot_score_distributions(
    artifacts: dict,
    y_test: pd.DataFrame,
    metrics: dict,
    plots_dir: Path,
) -> None:
    """Score density by class (calibrated) + expected claim cost per risk decile."""
    CLAIM_COSTS = {"fall_30d": 3_500, "rth_60d": 20_000, "wound_60d": 4_000}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for col, t in enumerate(TARGETS):
        valid_mask = y_test[t].notna()
        y_true = y_test.loc[valid_mask, t].astype(int)
        y_prob = _predict(artifacts, t)[valid_mask.values]
        thr    = metrics[t]["best_threshold"]

        # Row 0: density by class
        ax = axes[0, col]
        bins = np.linspace(0, max(y_prob.max(), 0.01), 40)
        ax.hist(y_prob[y_true == 0], bins=bins, alpha=0.6,
                color="steelblue", density=True, label="No event")
        ax.hist(y_prob[y_true == 1], bins=bins, alpha=0.6,
                color=TARGET_COLORS[t], density=True, label="Event")
        ax.axvline(thr, color="black", lw=1.2, ls="--",
                   label=f"Threshold={thr:.3f}")
        ax.set(xlabel="Calibrated predicted probability", ylabel="Density",
               title=TARGET_LABELS[t])
        ax.legend(fontsize=8)

        # Row 1: expected cost per risk decile
        ax = axes[1, col]
        cost_key = t if t in CLAIM_COSTS else t.replace("_60d", "_30d")
        cost = CLAIM_COSTS.get(t, 4_000)
        decile = pd.qcut(y_prob, q=10, labels=False, duplicates="drop")
        agg = (pd.DataFrame({"decile": decile, "prob": y_prob, "event": y_true.values})
               .groupby("decile")
               .agg(mean_prob=("prob", "mean"), event_rate=("event", "mean"),
                    n=("event", "count"))
               .reset_index())
        agg["expected_cost"] = agg["event_rate"] * cost
        bars = ax.bar(agg["decile"] + 1, agg["expected_cost"],
                      color=TARGET_COLORS[t], alpha=0.85)
        ax.set(xlabel="Risk decile (1=lowest, 10=highest)",
               ylabel=f"Exp. cost (${cost:,} × P)",
               title=f"Expected cost by decile\n{TARGET_LABELS[t]}")
        ax.set_xticks(range(1, len(agg) + 1))
        for bar, er in zip(bars, agg["event_rate"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1, f"{er:.1%}",
                    ha="center", fontsize=7)

    plt.suptitle("Score distributions & expected cost by risk decile", fontsize=12)
    plt.tight_layout()
    plt.savefig(plots_dir / "score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Score distributions → {plots_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. EVALUATION REPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_evaluation_report(
    metrics: dict,
    facility_df: pd.DataFrame,
    metrics_dir: Path,
) -> None:
    report = {}
    for t in TARGETS:
        m = metrics[t]
        report[t] = {
            "pr_auc":           round(float(m["pr_auc"]),         4),
            "roc_auc":          round(float(m["roc_auc"]),        4),
            "brier_score":      round(float(m["brier"]),          4),
            "log_loss":         round(float(m["log_loss"]),       4),
            "best_threshold":   round(float(m["best_threshold"]), 4),
            "f1_at_threshold":  round(float(m["f1_at_best_thr"]), 4),
            "sensitivity":      round(float(m["sensitivity"]),    4),
            "specificity":      round(float(m["specificity"]),    4),
            "ppv":              round(float(m["ppv"]),            4),
            "npv":              round(float(m["npv"]),            4),
            "positive_rate":    round(float(m["positive_rate"]),  4),
            "n_positive":       int(m["n_positive"]),
            "n_total":          int(m["n_total"]),
        }
    report["n_test_facilities"] = int(facility_df["facility_id"].nunique())
    report["model_type"] = "XGBoost (binary:logistic, tree_method=hist)"

    path = metrics_dir / "evaluation_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ Evaluation report → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    plots_dir   = OUTPUT_DIR / "plots"
    metrics_dir = OUTPUT_DIR / "metrics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1] Loading models, calibrators, and test data …")
    artifacts = load_artifacts(OUTPUT_DIR)
    y_test    = artifacts["y_test"]
    feat_cols = artifacts["feat_cols"]

    print("\n[2] Classification metrics (calibrated) …")
    metrics = compute_all_metrics(artifacts)

    print("\n[3] ROC and PR curves …")
    plot_roc_pr_curves(metrics, y_test, plots_dir)

    print("\n[4] Calibration plots …")
    plot_calibration(metrics, y_test, plots_dir)

    print("\n[5] SHAP analysis …")
    shap_results = compute_shap_values(artifacts, metrics_dir, plots_dir)

    print("\n[6] SHAP dependence plots …")
    plot_shap_dependence(shap_results, feat_cols, plots_dir)

    print("\n[7] Score distributions …")
    plot_score_distributions(artifacts, y_test, metrics, plots_dir)

    print("\n[8] Facility-level analysis …")
    facility_df = facility_level_analysis(artifacts, y_test, metrics_dir, plots_dir)

    print("\n[9] Saving evaluation report …")
    save_evaluation_report(metrics, facility_df, metrics_dir)

    print(f"\n{'═'*65}")
    print(f"  {'Target':<20} {'PR-AUC':>8} {'ROC-AUC':>9} {'Sens':>7} {'Spec':>7} {'PPV':>7}")
    print(f"{'─'*65}")
    for t in TARGETS:
        m = metrics[t]
        print(f"  {TARGET_LABELS[t]:<20} {m['pr_auc']:>8.4f} {m['roc_auc']:>9.4f} "
              f"{m['sensitivity']:>7.3f} {m['specificity']:>7.3f} {m['ppv']:>7.3f}")
    print(f"{'═'*65}")


if __name__ == "__main__":
    import time
    t0 = time.time()
    main()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
