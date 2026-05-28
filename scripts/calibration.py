"""
calibration.py
─────────────────
Standalone calibration script — fits and saves isotonic calibrators for
already-trained XGBoost models without rerunning the full pipeline.

This is useful when you have existing model artifacts (xgb_*.pkl) and want
to apply the calibration:

  Problem:  scale_pos_weight overcorrection inflated all predicted
            probabilities 5-20× above their true values, making the
            expected-cost formula unreliable and flagging all 612 active
            residents as "High" tier.

  Fix:      IsotonicRegression fitted on out-of-fold (OOF) predictions
            maps raw scores back to calibrated probabilities that match
            the observed event rate.  The calibrator is monotone, so the
            ranking (ROC-AUC, PR-AUC) is unaffected.

When to run this
────────────────────────────────────────────────────
• Run this script if you only want to add calibration to existing models.
  It requires outputs/metrics/oof_{target}.parquet (written by training).

Output
──────
  outputs/models/calibrator_{target}.pkl     isotonic calibrator per target
  outputs/metrics/calibration_report.json    before/after Brier scores
  outputs/plots/calibration_comparison.png   visual comparison

Run:  python calibration.py
"""

import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})


# ══════════════════════════════════════════════════════════════════════════════
# 1. AUTO-DETECT AVAILABLE TARGETS
# ══════════════════════════════════════════════════════════════════════════════

def discover_targets(output_dir: Path) -> list[str]:
    """
    Find all targets for which both a trained model and OOF predictions exist.
    Works with any target naming convention (fall_30d, rth_30d, rth_60d, etc.)
    """
    models_dir  = output_dir / "models"
    metrics_dir = output_dir / "metrics"

    model_targets = {p.stem.replace("xgb_", "") for p in models_dir.glob("xgb_*.pkl")}
    oof_targets   = {p.stem.replace("oof_", "") for p in metrics_dir.glob("oof_*.parquet")}
    available     = sorted(model_targets & oof_targets)

    if not available:
        raise FileNotFoundError(
            f"No matching (model, OOF) pairs found.\n"
            f"  model files : {list(models_dir.glob('xgb_*.pkl'))}\n"
            f"  OOF files   : {list(metrics_dir.glob('oof_*.parquet'))}"
        )
    print(f"  Targets with both model and OOF predictions: {available}")
    return available


# ══════════════════════════════════════════════════════════════════════════════
# 2. FIT CALIBRATORS
# ══════════════════════════════════════════════════════════════════════════════

def fit_and_save_calibrators(targets: list[str], output_dir: Path) -> dict:
    """
    For each target:
      1. Load the XGBoost model and per-target feature column list (if available).
      2. Load OOF predictions and true labels.
      3. Fit an IsotonicRegression on (raw_oof_pred, true_label).
      4. Save the calibrator.
      5. Compute and report before/after Brier scores.

    Returns a dict of calibration metrics for reporting.
    """
    models_dir  = output_dir / "models"
    metrics_dir = output_dir / "metrics"
    report      = {}

    for target in targets:
        print(f"\n  [{target}]")

        # ── Load OOF data ─────────────────────────────────────────────────
        oof_df = pd.read_parquet(metrics_dir / f"oof_{target}.parquet")
        y_true  = oof_df[target].dropna().astype(int)
        oof_raw = oof_df.loc[y_true.index, "oof_pred"].values
        y_true  = y_true.values

        pos_rate     = float(y_true.mean())
        baseline_bs  = pos_rate * (1 - pos_rate)
        raw_bs       = brier_score_loss(y_true, oof_raw)
        raw_bss      = 1 - raw_bs / max(baseline_bs, 1e-9)
        raw_pr_auc   = average_precision_score(y_true, oof_raw)
        raw_roc_auc  = roc_auc_score(y_true, oof_raw)

        print(f"    OOF positive rate   : {pos_rate:.3%}")
        print(f"    OOF mean prediction : {oof_raw.mean():.4f}  "
              f"(true rate {pos_rate:.4f}  — ratio {oof_raw.mean()/max(pos_rate,1e-9):.1f}×)")
        print(f"    Pre-calibration     : Brier={raw_bs:.4f}  BSS={raw_bss:.3f}  "
              f"PR-AUC={raw_pr_auc:.4f}  ROC-AUC={raw_roc_auc:.4f}")

        # ── Fit isotonic calibrator ───────────────────────────────────────
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(oof_raw, y_true)
        cal_pred    = calibrator.predict(oof_raw)

        cal_bs      = brier_score_loss(y_true, cal_pred)
        cal_bss     = 1 - cal_bs / max(baseline_bs, 1e-9)
        cal_pr_auc  = average_precision_score(y_true, cal_pred)
        cal_roc_auc = roc_auc_score(y_true, cal_pred)

        print(f"    Post-calibration    : Brier={cal_bs:.4f}  BSS={cal_bss:.3f}  "
              f"PR-AUC={cal_pr_auc:.4f}  ROC-AUC={cal_roc_auc:.4f}")
        print(f"    Mean pred shift     : {oof_raw.mean():.4f} → {cal_pred.mean():.4f}  "
              f"(true rate: {pos_rate:.4f})")
        brier_improvement_pct = (raw_bs - cal_bs) / max(raw_bs, 1e-9) * 100
        print(f"    Brier improvement   : {brier_improvement_pct:.1f}%")

        # ── Save calibrator ───────────────────────────────────────────────
        cal_path = models_dir / f"calibrator_{target}.pkl"
        with open(cal_path, "wb") as f:
            pickle.dump(calibrator, f)
        print(f"    ✓ Saved → {cal_path}")

        report[target] = {
            "positive_rate":        pos_rate,
            "baseline_brier":       baseline_bs,
            "pre_cal_brier":        raw_bs,
            "pre_cal_bss":          raw_bss,
            "pre_cal_mean_pred":    float(oof_raw.mean()),
            "post_cal_brier":       cal_bs,
            "post_cal_bss":         cal_bss,
            "post_cal_mean_pred":   float(cal_pred.mean()),
            "brier_improvement_pct": brier_improvement_pct,
            "pr_auc":               raw_pr_auc,   # unchanged by monotone calibration
            "roc_auc":              raw_roc_auc,  # unchanged by monotone calibration
            "n_oof":                int(len(y_true)),
        }

    return report


# ══════════════════════════════════════════════════════════════════════════════
# 3. CALIBRATION PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_calibration_comparison(targets: list[str], output_dir: Path) -> None:
    """
    Reliability diagrams (predicted probability vs observed event rate)
    before and after calibration — side by side per target.
    """
    models_dir  = output_dir / "models"
    metrics_dir = output_dir / "metrics"
    plots_dir   = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    colors = {"pre": "#A32D2D", "post": "#185FA5"}

    for ax, target in zip(axes, targets):
        oof_df  = pd.read_parquet(metrics_dir / f"oof_{target}.parquet")
        y_true  = oof_df[target].dropna().astype(int).values
        oof_raw = oof_df.loc[oof_df[target].notna(), "oof_pred"].values

        cal_path = models_dir / f"calibrator_{target}.pkl"
        with open(cal_path, "rb") as f:
            calibrator = pickle.load(f)
        cal_pred = calibrator.predict(oof_raw)

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.35, label="Perfect")

        for label, probs, color in [
            ("Before calibration", oof_raw, colors["pre"]),
            ("After calibration",  cal_pred, colors["post"]),
        ]:
            frac_pos, mean_pred = calibration_curve(
                y_true, probs, n_bins=10, strategy="quantile"
            )
            bs = brier_score_loss(y_true, probs)
            ax.plot(mean_pred, frac_pos, "o-", lw=2, ms=5, color=color,
                    label=f"{label}\n(Brier={bs:.4f})")

        pos_rate = float(y_true.mean())
        ax.axhline(pos_rate, color="gray", lw=0.6, ls=":", alpha=0.6,
                   label=f"Base rate {pos_rate:.2%}")

        ax.set(xlabel="Mean predicted probability", ylabel="Observed event rate",
               xlim=(0, max(oof_raw.max(), 0.01) * 1.05), ylim=(0, 1),
               title=target.replace("_", " ").title())
        ax.legend(fontsize=8)

    plt.suptitle("Reliability diagrams: before vs after isotonic calibration\n"
                 "(OOF predictions — n = all training rows)",
                 fontsize=12)
    plt.tight_layout()
    path = plots_dir / "calibration_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Calibration comparison plot → {path}")


def plot_score_shift(targets: list[str], output_dir: Path) -> None:
    """
    Histogram overlay of raw vs calibrated OOF scores per target.
    Shows how calibration compresses the inflated score distribution.
    """
    models_dir  = output_dir / "models"
    metrics_dir = output_dir / "metrics"
    plots_dir   = output_dir / "plots"

    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        oof_df  = pd.read_parquet(metrics_dir / f"oof_{target}.parquet")
        oof_raw = oof_df.loc[oof_df[target].notna(), "oof_pred"].values
        pos_rate = oof_df[target].dropna().mean()

        cal_path = models_dir / f"calibrator_{target}.pkl"
        with open(cal_path, "rb") as f:
            cal_pred = pickle.load(f).predict(oof_raw)

        bins = np.linspace(0, max(oof_raw.max(), cal_pred.max(), pos_rate * 3, 0.01), 40)
        ax.hist(oof_raw,  bins=bins, alpha=0.6, color="#A32D2D", density=True,
                label=f"Raw (mean={oof_raw.mean():.3f})")
        ax.hist(cal_pred, bins=bins, alpha=0.6, color="#185FA5", density=True,
                label=f"Calibrated (mean={cal_pred.mean():.3f})")
        ax.axvline(pos_rate, color="black", lw=1.5, ls="--",
                   label=f"True rate {pos_rate:.3f}")
        ax.set(xlabel="Predicted probability", ylabel="Density",
               title=target.replace("_", " ").title())
        ax.legend(fontsize=8)

    plt.suptitle("Score distribution shift: raw vs calibrated\n"
                 "(calibration compresses inflated scores toward true event rates)",
                 fontsize=12)
    plt.tight_layout()
    path = plots_dir / "calibration_score_shift.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Score shift plot → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SAVE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_report(report: dict, output_dir: Path) -> None:
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Round floats for readability
    clean = {}
    for target, m in report.items():
        clean[target] = {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in m.items()}

    path = metrics_dir / "calibration_report.json"
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)

    # Print summary table
    print(f"\n{'─'*72}")
    print(f"  {'Target':<14} {'Pre-BSS':>8} {'Post-BSS':>9} {'Pre-mean':>10} "
          f"{'Post-mean':>10} {'True rate':>10}")
    print(f"{'─'*72}")
    for target, m in clean.items():
        print(f"  {target:<14} {m['pre_cal_bss']:>8.3f} {m['post_cal_bss']:>9.3f} "
              f"{m['pre_cal_mean_pred']:>10.4f} {m['post_cal_mean_pred']:>10.4f} "
              f"{m['positive_rate']:>10.4f}")
    print(f"{'─'*72}")
    print(f"\n  ✓ Calibration report → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n[1] Discovering available model/OOF pairs …")
    targets = discover_targets(OUTPUT_DIR)

    print("\n[2] Fitting isotonic calibrators …")
    report = fit_and_save_calibrators(targets, OUTPUT_DIR)

    print("\n[3] Calibration reliability diagrams …")
    plot_calibration_comparison(targets, OUTPUT_DIR)

    print("\n[4] Score distribution shift plots …")
    plot_score_shift(targets, OUTPUT_DIR)

    print("\n[5] Saving calibration report …")
    save_report(report, OUTPUT_DIR)

    print("\nCalibration complete.")
    print("Next steps:")
    print("  • Run python model_evaluation.py to regenerate evaluation metrics")
    print("    and plots using calibrated probabilities.")
    print("  • Run python risk_scoring.py to regenerate the risk dashboard")
    print("    with calibrated scores and percentile-based tiers.")


if __name__ == "__main__":
    import time
    t0 = time.time()
    main()
    print(f"\nDone in {time.time() - t0:.1f}s")
