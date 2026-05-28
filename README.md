# SNF Incident Risk Model

A machine learning system that predicts the probability of three high-cost incident types — **falls**, **unplanned return-to-hospital (RTH)**, and **wound events** — for residents of skilled nursing facilities (SNFs), using longitudinal clinical and operational data.

Built for a liability insurer whose claims exposure is concentrated in resident incidents. The model's output is a ranked risk dashboard that facility staff can act on before incidents occur, reducing both claim frequency and severity.

---

## Table of Contents

1. [How to run](#1-how-to-run)
2. [Business problem](#2-business-problem)
3. [Insurance domain context](#3-insurance-domain-context)
4. [Prediction approach](#4-prediction-approach)
5. [Data](#5-data)
6. [Feature engineering](#6-feature-engineering)
7. [Model design](#7-model-design)
8. [Evaluation metrics and their rationale](#8-evaluation-metrics-and-their-rationale)
9. [Key decisions and tradeoffs](#9-key-decisions-and-tradeoffs)
10. [Repository structure](#10-repository-structure)
11. [Outputs](#11-outputs)
12. [Key findings](#12-key-findings)
13. [Open limitations identified from live scoring](#13-open-limitations-identified-from-live-scoring)
14. [References](#references)

---

## 1. How to run

### Requirements

This project uses a conda environment defined in `environment.yml` (Python 3.11, conda-forge channel).

```bash
conda env create -f environment.yml
conda activate tricura
```

To update an existing environment after changes to `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

### Configuration

Edit `config.py`:

```python
DATA_DIR   = Path("/path/to/your/parquet/files")
OUTPUT_DIR = Path("outputs")
```

Key tunable parameters (all in `config.py`):

| Parameter | Default | Effect |
|---|---|---|
| `SCALE_POS_WEIGHT_MAX` | `10.0` | Cap on class imbalance upweighting |
| `CALIBRATION_METHOD` | `"isotonic"` | `"isotonic"` or `"sigmoid"` |
| `RISK_TIER_PERCENTILES` | `{"high": 75, "medium": 50}` | Percentile cutoffs applied to `expected_cost_30d` |
| `DASHBOARD_ACTIVE_TARGETS` | `["fall_30d", "rth_60d"]` | Targets that contribute to expected cost and tiers; `wound_60d` suppressed |
| `EXCLUDE_FEATURES_BY_TARGET` | `{"wound_60d": ["facility_id_enc"]}` | Feature exclusions per target |
| `OPTUNA_N_TRIALS` | `30` | Reduce to 10 for a quick smoke test |
| `N_CV_FOLDS` | `5` | Reduce to 3 for faster iteration |

### Full pipeline

```bash
# Step 1 — Build feature matrix with per-target label horizons (~15 min)
python data_preparation.py

# Step 2 — Train with calibration (~25 min)
python model_training.py

# Step 3 — Evaluate with calibrated probabilities (~5 min)
python model_evaluation.py

# Step 4 — Score active residents (~3 min)
python risk_scoring.py
```

### Calibrating without retraining

If you already have trained XGBoost models and OOF predictions, you can fit the calibrators without retraining:

```bash
# Requires: outputs/models/xgb_*.pkl
#           outputs/metrics/oof_*.parquet
python calibration.py

# Then regenerate evaluation and dashboard with calibrated probabilities:
python model_evaluation.py
python risk_scoring.py
```

The script auto-detects available target names from OOF file names in `outputs/metrics/`.

---

## 2. Business problem

The company insures skilled nursing facilities against liability arising from resident incidents. The claims portfolio breaks down as follows:

| Incident type | Share of claims | Average cost |
|---|---|---|
| Falls | ~13% | ~$3,500 |
| Medication errors | ~10% | ~$5,000 |
| Wounds / pressure injuries | ~7% | ~$4,000 |
| Return-to-hospital (RTH) | ~7% | ~$20,000 |
| Elopement / wandering | ~5% | ~$2,500 |
| Altercations | ~2% | ~$2,500 |

RTH events dominate dollar exposure despite a modest claim share, because the average cost per event is nearly six times that of a fall. The core business lever is reducing **incident frequency** at insured facilities — raising premiums is not sustainable without losing clients, but helping facilities intervene earlier on high-risk residents directly compresses the loss ratio.

The model is designed to answer a concrete question that facility staff and insurers can act on: *which residents are most likely to generate a high-cost incident in the next 30–60 days, and why?*

> **Why medication errors are not modeled as a target.** Despite representing ~10% of claims at $5,000 average cost, the `incidents` table contains only 44 medication error records across 3,000 residents — a 13:1 shortfall relative to wound incidents. The discrepancy is almost certainly underreporting (facilities have strong incentives not to document errors). Labels derived from a heavily selected, non-representative sample train the model to predict documentation behavior rather than clinical risk. Medication process quality is instead captured as features: miss rate, late-dose rate, polypharmacy count, and psychotropic drug exposure.

---

## 3. Insurance domain context

**Loss ratio** is the ratio of claims paid to premiums collected. A sustained loss ratio above ~70% signals unprofitability after operational costs. [1] The model reduces this by surfacing actionable risk earlier — every prevented incident improves the loss ratio without requiring a premium increase.

**Pure premium** is the expected annual cost per insured unit. Facility-level pricing today relies on historical averages. A resident-level risk score allows decomposition of that aggregate into individual risk contributions, enabling more accurate facility-level pricing.

**Experience rating** adjusts a client's premium against their own loss history relative to the expected baseline. Without risk adjustment, a facility with a genuinely high-acuity resident mix may be penalized even if their care quality is excellent. The model provides the risk-adjusted counterfactual: given *this* population, how many incidents would we expect? [1]

**Calibrated probabilities are essential for financial use.** The expected-cost formula in the dashboard (`score × avg_claim_cost`) only produces correct financial forecasts when `score` is a true probability. An uncalibrated model that inflates all scores 10× inflates the portfolio exposure estimate 10× — material for loss reserving and premium decisions. This is why isotonic calibration is a mandatory step rather than an optional enhancement.

**Adverse selection and moral hazard** are relevant at the facility level. The facility-level scatter plot in evaluation flags facilities where predicted risk substantially exceeds observed event rates, which may indicate systematic incident underreporting.

---

## 4. Prediction approach

### Target definition

Three binary outcomes are predicted independently. Horizons differ by target because RTH and wound events are sparser, requiring a wider window to accumulate enough positive labels for reliable learning.

| Target column | Event definition | Prediction horizon |
|---|---|---|
| `fall_30d` | Any Fall incident (not struck out) in `(obs_date, obs_date + 30d]` | 30 days |
| `rth_60d` | Any unplanned hospital transfer (`planned_flag=False`, outcome contains "Admitted") in `(obs_date, obs_date + 60d]` | 60 days |
| `wound_60d` | Any Wound incident (not struck out) in `(obs_date, obs_date + 60d]` | 60 days |

Rows where `obs_date + horizon > DATA_AVAILABILITY_END` cannot have complete labels and are marked `NaN` for that target. These censored rows are excluded per-target before training — they are retained in the feature matrix so no observations are lost for targets with shorter horizons.

### Observation unit

One row per resident per 7-day step across their active stay. For 3,000 residents over ~18 months, this produces approximately 150,000–300,000 labeled rows.

Each row has strict temporal separation: features use only data before `obs_date`; labels use only data after it. No look-ahead leakage.

### Why three separate models

The three targets have distinct clinical drivers. Falls are driven by mobility decline and psychotropic medication [2, 3]; RTH is driven by vital sign instability and cardiopulmonary diagnoses [4]; wounds are driven by immobility and malnutrition [5]. Separate models are also simpler to update independently — if wound prediction degrades, only that model needs retraining.

---

## 5. Data

3,000 residents across 100 facilities, with longitudinal records spanning approximately 2023–2025.

| Table | Rows | Role |
|---|---|---|
| `residents` | 3,000 | Observation spine |
| `diagnoses` | 60,620 | Comorbidity flags, Charlson CCI |
| `adl_responses` | 480,554 | Functional status trends |
| `gg_responses` | 660,711 | CMS Section GG mobility/self-care |
| `vitals` | 2,517,056 | Rolling clinical signals |
| `medications` | 1,430,877 | Polypharmacy, drug classes, adherence |
| `document_tags` | 562,905 | NLP-derived clinical signals |
| `lab_reports` | 13,334 | Abnormal lab counts |
| `incidents` | 3,578 | **Target labels** (falls, wounds) |
| `hospital_transfers` | 1,816 | **Target labels** (RTH) + prior RTH features |
| `injuries` | 1,219 | Fall severity features |
| `factors` | 190,284 | Incident factor annotations |
| `needs` | 162,762 | Active care plan need flags |
| `therapy_tracks` | 761 | PT/OT enrollment |

---

## 6. Feature engineering

All features computed strictly from data before `obs_date`. Rolling statistics (mean, std, min, max) at 7-, 30-, and 90-day windows using `pandas groupby().rolling()`.

**Static:** age at observation, length of stay, outpatient flag, encoded facility ID.

**Comorbidities:** 26 ICD-10 diagnosis flags and Charlson Comorbidity Index (Quan 2011, age-adjusted) [6]. Active diagnoses only: `onset_at ≤ obs_date` and `resolved_at` null or future.

**ADL:** rolling mean/std across 7 domains (eating, bed mobility, transfer, toilet, locomotion, walking) at 7/30/90-day windows. Trend feature `adl_trend_7v30` captures recent vs medium-term trajectory.

**GG mobility/self-care:** CMS Section GG scores inverted so higher = more dependent (matching ADL direction). Key mobility tasks (sit-to-stand, chair transfer, walking) computed separately.

**Vital signs:** mean, std, min, max for 8 vital types at 7/30-day windows. Derived: BP coefficient of variation (instability predicts falls and RTH), weight % change, eight binary clinical threshold flags (hypoxia, hypertensive urgency, severe pain, fever, etc.).

**Medications:** polypharmacy count, psychotropic/opioid/diuretic/anticoagulant exposure (extracted by regex from description text) [2, 3], miss rate, late dose rate, polypharmacy flag (≥9 drugs) [7].

**Document tags:** 18 high-signal NLP tags from progress notes and physician orders, including `fall_risk`, `uti_symptoms`, `aggressive_behavior`, `altered_mental_status`.

**Lab reports:** abnormal and critical lab counts in the last 30 days.

**Prior incidents:** fall and wound counts at 30/90-day lookbacks, days since last fall, indicator for injury at last fall, altercation count.

**Prior RTH:** unplanned transfer counts at 30/90-day lookbacks, days since last RTH, ever-transferred flag.

**Active care needs:** binary flags for active fall, wound, and nutrition care plan needs.

**Therapy enrollment:** active PT and OT flags.

---

## 7. Model design

### Algorithm: XGBoost with histogram split-finding

`xgboost.XGBClassifier` with `objective="binary:logistic"` and `tree_method="hist"`. Three independent binary classifiers, one per target.

### Class imbalance handling

`scale_pos_weight` is set dynamically per training fold, **capped at 10×** (`SCALE_POS_WEIGHT_MAX`). Without the cap, a 2% positive-rate target receives a raw ratio of 49×, which causes all predicted probabilities to cluster far above their true values regardless of individual resident characteristics — in testing against analogous data, uncapped upweighting produced mean predictions of 0.46 against a true rate of 0.02 (Brier Skill Score = −9.4). Post-training isotonic calibration corrects any residual bias.

### Post-training probability calibration

After cross-validation assembles out-of-fold (OOF) predictions for every training row, an `sklearn.isotonic.IsotonicRegression` is fitted on `(oof_pred, true_label)` [12]. This non-parametric calibrator is saved alongside the XGBoost model and applied transparently at evaluation and scoring time.

Isotonic regression is chosen over Platt scaling (sigmoid) because the miscalibration shape produced by high `scale_pos_weight` values is non-linear and severe — sigmoid calibration assumes the raw scores are a monotone transformation of the true log-odds, which does not hold when `scale_pos_weight` has dramatically shifted the decision boundary.

The calibrator is monotone, so all discrimination metrics (ROC-AUC, PR-AUC, feature importance rankings) are unaffected. Only the numeric probability values change.

### Feature exclusion per target

`facility_id_enc` is excluded from the wound model (`EXCLUDE_FEATURES_BY_TARGET`). Without this exclusion, it ranks 2nd by mean |SHAP| [9, 10] for wound prediction, meaning the model memorizes facility-level effects rather than learning portable clinical signals. Facility-level ROC-AUC variance in the wound model ranged from 0.115 to 0.969 — near-random on some facilities, excellent on others — consistent with facility memorization that fails to generalize.

### Hyperparameter tuning: Optuna TPE

30 trials of Bayesian optimisation (TPE sampler) per target, searching over learning rate, max depth, min child weight, subsample, colsample_bytree, and L1/L2 regularisation. Objective: mean PR-AUC across 5 facility-held-out CV folds.

### Final model training

`n_estimators = int(avg_best_iteration × 1.1)`. No early stopping on the full training set — all 80 training facilities are used without withholding a validation partition.

---

## 8. Evaluation metrics and their rationale

**PR-AUC (primary):** measures discrimination on the minority class. More sensitive than ROC-AUC to class imbalance [13]. Used in tuning, CV, and final evaluation.

**ROC-AUC:** reported for comparability with the clinical prediction literature.

**Brier score and Brier Skill Score [14]:** measures calibration. BSS = `1 − BS_model / BS_baseline` where `BS_baseline = π(1−π)` (always predicting the base rate). A positive BSS means the model beats the no-skill baseline; negative means it is worse. This is the metric that most directly validates whether the expected-cost formula produces reliable financial estimates.

**Sensitivity and specificity at the F1-optimal threshold:** for the operating scenario where a facility reviews a ranked list and flags the top N residents for follow-up.

### Validation strategy: facility-held-out GroupKFold

Cross-validation groups on `facility_id` — all residents from a held-out set of facilities are excluded from each training fold. This tests generalization to new facilities rather than new residents at known ones. The final test set holds out 20% of facilities (20 of 100).

---

## 9. Key decisions and tradeoffs

**Three separate models vs. multi-label.** Separate models per target are easier to calibrate, explain, and update independently. Multi-task learning becomes attractive at >100K residents where joint training provides meaningful regularisation.

**Per-target label horizons (30 vs 60 days).** A 30-day horizon for RTH and wound yields positive rates of approximately 2.0% and 2.5% — too sparse for reliable learning. During cross-validation, the RTH model reached its best iteration after an average of 34 trees (minimum 7 across folds), and the wound model after 17 trees (minimum 2), indicating both stopped learning almost immediately. The 60-day horizon roughly doubles positive label counts while remaining actionable. Target columns are named `rth_60d` and `wound_60d` to make the horizon explicit.

**Rolling window features vs. sequence models.** Hand-engineered rolling statistics are interpretable, produce actionable SHAP explanations, and typically outperform LSTMs/transformers on structured clinical data at <100K patients [15].

**scale_pos_weight capped at 10×, with post-training isotonic calibration.** The raw class-imbalance ratio `n_negative / n_positive` reaches 49× for a 2% positive-rate target. Applying this uncapped inflates all predicted probabilities far above their true values — in testing, uncapped upweighting produced mean predictions of 0.46 against a true rate of 0.02 (Brier Skill Score = −9.4). `SCALE_POS_WEIGHT_MAX = 10.0` limits this distortion. After training, an `IsotonicRegression` calibrator is fitted on the out-of-fold predictions from cross-validation and applied transparently at scoring time. Isotonic regression is chosen over Platt scaling [12] because the miscalibration shape produced by high `scale_pos_weight` values is non-linear; simulation confirms a ~92% Brier score reduction relative to uncapped upweighting.

**Per-target feature exclusion.** `facility_id_enc` is excluded from the wound model via `EXCLUDE_FEATURES_BY_TARGET`. Without this exclusion it ranks 2nd by mean |SHAP| (0.176, versus the top feature at 0.188), indicating the model memorizes facility-level documentation patterns rather than portable clinical signals — confirmed by facility-level wound ROC-AUC ranging from 0.115 to 0.969, near-random on some held-out facilities and high on others. `facility_id_enc` is retained for fall and RTH, where it provides legitimate population-level signal without the same generalization failure.

**Tier assignment on combined expected cost, not per-target percentiles.** The dashboard assigns tiers by ranking residents on their total expected claim cost — `fall_score × $3,500 + rth_score × $20,000` — rather than computing per-target percentile tiers and taking the worst. Applying a 25%-High threshold independently to each active target and taking the worst produces a union probability of approximately `1 − 0.75^N` — roughly 44% for two targets, growing with more — far above the intended 25%. Tiering on the single combined cost column guarantees exactly 25% High, 25% Medium, 50% Low regardless of the number of active models or their score distributions.

**Wound model scored but suppressed from dashboard tiers and expected cost.** All three models are trained and all three scores are written to the output CSV. Only `fall_30d` and `rth_60d` are listed in `DASHBOARD_ACTIVE_TARGETS` and therefore contribute to the expected-cost formula and tier assignment. The wound model's Brier Skill Score of −0.019 (below the no-skill baseline) and extreme score compression — the 90th-percentile calibrated score of 4.75% is only 0.3 percentage points above the 4.53% base rate — mean it provides no useful triage signal. Suppressing it prevents it from inflating tier counts or contaminating the expected-cost estimate. The wound score column is retained in the HTML table in muted/italic style for monitoring and stored in full in the CSV; re-enabling it requires adding `"wound_60d"` back to `DASHBOARD_ACTIVE_TARGETS` in `config.py`.

**Cost-weighted dominant model for SHAP explanations.** The "top drivers" column shows SHAP explanations from whichever `DASHBOARD_ACTIVE_TARGETS` model contributes most to each individual resident's expected cost — typically RTH near the top of the list, since its $20,000 multiplier dominates unless the fall score exceeds 5.7× the RTH score. Wound SHAP drivers are not shown in the HTML dashboard but are saved in the CSV (`wound_60d_top_reasons`). Only 11–18% of individual SHAP values are positive across the three models, which is the correct representation of a population where 93–96% of residents have no incident in any given window — the dashboard uses directional `↑`/`↓` arrows to communicate this distinction to care coordinators.

---

## 10. Repository structure

```
.
├── config.py               # All constants: paths, horizons, XGBoost params,
│                              # calibration config, tier percentiles, exclusions,
│                              # and DASHBOARD_ACTIVE_TARGETS (wound suppressed)
│
├── data_preparation.py     # Feature matrix + per-target labels with horizon-aware
│                              # censoring (NaN for obs_dates with incomplete label windows)
│                              # Input:  DATA_DIR (raw parquets)
│                              # Output: outputs/features.parquet
│                              #         outputs/labels.parquet
│
├── model_training.py       # HPO + facility-held-out CV + isotonic calibration
│                              # + final model training
│                              # Input:  outputs/features.parquet / labels.parquet
│                              # Output: outputs/models/xgb_{target}.pkl
│                              #         outputs/models/calibrator_{target}.pkl
│                              #         outputs/models/feature_cols_{target}.json
│                              #         outputs/metrics/oof_{target}.parquet
│                              #         outputs/metrics/training_summary.csv
│
├── model_evaluation.py     # Calibrated metrics, plots, and SHAP analysis
│                              # Input:  trained models + calibrators + test data
│                              # Output: outputs/plots/*.png  (9 plot files)
│                              #         outputs/metrics/evaluation_report.json
│                              #         outputs/metrics/facility_scores.csv
│                              #         outputs/metrics/shap_values_{target}.parquet
│                              #         outputs/metrics/feature_importance_{target}.csv
│
├── risk_scoring.py         # Scores active residents with calibrated probabilities.
│                              # Tier assigned on combined expected cost (fall + RTH only);
│                              # wound scored and saved to CSV but suppressed from tiers.
│                              # Input:  models + calibrators + raw parquets
│                              # Output: outputs/metrics/risk_dashboard_{date}.csv
│                              #         outputs/metrics/risk_dashboard_{date}.html
│
├── calibration.py          # Standalone: fits calibrators to existing models
│                              # using saved OOF predictions — no retraining needed.
│                              # Useful when models are already trained and only
│                              # the calibration step needs to be (re)applied.
│                              # Input:  outputs/models/xgb_*.pkl
│                              #         outputs/metrics/oof_*.parquet
│                              # Output: outputs/models/calibrator_{target}.pkl
│                              #         outputs/metrics/calibration_report.json
│                              #         outputs/plots/calibration_comparison.png
│                              #         outputs/plots/calibration_score_shift.png
│
└── outputs/
    ├── models/
    │   ├── xgb_{target}.pkl
    │   ├── calibrator_{target}.pkl
    │   ├── feature_cols.json
    │   ├── feature_cols_{target}.json    (per-target, respects exclusions)
    │   └── params_{target}.json
    ├── metrics/
    │   ├── features.parquet
    │   ├── labels.parquet
    │   ├── oof_{target}.parquet
    │   ├── training_summary.csv          (calibrated Brier scores per target)
    │   ├── calibration_report.json       (from calibration.py)
    │   ├── evaluation_report.json
    │   ├── facility_scores.csv
    │   └── risk_dashboard_{date}.html
    └── plots/
        ├── roc_pr_curves.png
        ├── calibration.png
        ├── calibration_comparison.png    (pre/post calibration reliability diagrams)
        ├── calibration_score_shift.png   (score distribution before and after calibration)
        ├── shap_summary_{target}.png
        ├── shap_importance_{target}.png
        ├── shap_dependence_{target}.png
        ├── shap_cross_target.png
        ├── score_distributions.png
        └── facility_risk_scatter.png
```

---

## 11. Outputs

### Risk dashboard (`risk_dashboard_{date}.html`)

Ranks all active residents by expected claim cost, calculated from the two dashboard-active models only:

```
expected cost = fall_score × $3,500 + rth_score × $20,000
```

Wound is excluded from this formula because its Brier Skill Score of −0.019 means it is a no-skill predictor — including it would add noise to the ranking without contributing information. Its calibrated score is still shown in a muted column for monitoring purposes, and all wound data is preserved in the CSV.

Scores are **calibrated probabilities** — they reflect true event probabilities and can be used directly in financial calculations.

Tier assignment is **percentile-ranked on `expected_cost_30d`** directly: top 25% = High, next 25% = Medium, bottom 50% = Low. This guarantees a 25/25/50 split regardless of absolute score levels or number of active models. Earlier per-target tiering (applying the 25% threshold independently to each target and taking the worst) produced ~50% "High" from the compound union probability, which was corrected by this approach.

Each row shows:
- Calibrated risk scores for all three models — fall and RTH as active triage columns, wound as a muted informational column marked with ⚠
- Overall tier, derived from expected cost rank (not per-target worst)
- Expected claim cost (fall + RTH only)
- Top 3 SHAP-driven clinical drivers from the dominant active-model, with `↑` (risk-increasing) or `↓` (risk-reducing) directional arrows. The dominant model is whichever of `fall_30d` and `rth_60d` contributes more to that resident's expected cost.

### `calibration_report.json`

Before/after Brier Skill Scores, mean score vs true event rate, and Brier improvement percentage per target. The primary diagnostic for confirming that calibration is working as expected.

### Evaluation plots

| Plot | What it shows |
|---|---|
| `roc_pr_curves.png` | ROC and PR curves on held-out test facilities |
| `calibration.png` | Reliability diagrams on test set |
| `calibration_comparison.png` | Before vs after calibration (OOF data) |
| `calibration_score_shift.png` | Score distribution before vs after calibration |
| `shap_summary_{target}.png` | SHAP beeswarm per target |
| `shap_importance_{target}.png` | Top 20 features by mean \|SHAP\| |
| `shap_dependence_{target}.png` | Partial dependence for top 4 features |
| `shap_cross_target.png` | Cross-target feature importance comparison |
| `score_distributions.png` | Score density by class + expected cost per decile |
| `facility_risk_scatter.png` | Per-facility calibrated predicted risk vs observed rate |

### Facility risk scatter (underreporting detection)

Facilities above the diagonal have higher observed incident rates than the model predicts given their resident population. After calibration, systematic divergence is more likely to reflect real underreporting or facility-specific care issues rather than model miscalibration.

---

## 12. Key findings

### Model evaluation findings

All findings below are derived from the plots generated by `model_evaluation.py` on the 20 held-out test facilities. Plots are in `outputs/plots/`.

#### Discrimination and ranking

- **Fall** is the only model with strong discrimination: ROC-AUC 0.821, PR-AUC 0.283 (3.8× base-rate lift). The ROC curve separates clearly from the random baseline at every operating point.
- **RTH** (ROC-AUC 0.689, PR-AUC 0.105) shows a brief precision spike above 0.60 at very low recall before collapsing to near-baseline. The model reliably identifies the very highest-risk residents but captures only a small fraction of all RTH events.
- **Wound** (ROC-AUC 0.662, PR-AUC 0.076, 1.7× base-rate lift) adds minimal discrimination beyond chance. Its PR curve hugs the base-rate line throughout — consistent with a Brier Skill Score of −0.019.
- RTH and wound ROC curves overlap substantially across most of their range, confirming they share a similar level of discriminative power despite different clinical drivers.

#### Calibration

- **Fall** calibration is nearly perfect: all decile points lie close to the diagonal from 0 to ~0.35, with only minor over-prediction at the top decile (~0.38 predicted vs ~0.35 observed).
- **RTH** calibration is good within its compressed range (0–0.10): predicted probabilities track observed rates, but all predictions remain below 0.15, which reflects the score discretisation rather than miscalibration.
- **Wound** shows irregular calibration — a visible zig-zag pattern with notable over-prediction near 0.20 predicted probability. This non-monotone reliability diagram is consistent with the wound model's poor discrimination and BSS of −0.019.

#### Score distributions and operating thresholds

- **Fall** scores span 0–0.55 with genuine density separation between event (shifted right) and no-event residents. The optimal threshold of 0.197 sits in a meaningful gap between the two distributions.
- **RTH** scores display a clear step-function pattern with discrete spikes rather than a smooth density. Both event and no-event residents cluster at the same discrete values, confirming the calibrator produces only 12 unique score levels.
- The expected-cost-per-decile chart increases monotonically for fall (top decile: 35.7% event rate, ~$1,250) and RTH (top decile: 9.6%, ~$1,900), confirming the model correctly concentrates risk at the high end. Wound's ordering is non-monotone (decile 3 > decile 4), further evidence of unreliable discrimination.
- RTH produces only 5 distinct decile groups rather than 10, caused by score discretisation collapsing many residents into the same predicted values.

#### Facility-level calibration and underreporting signals

- **Fall scatter**: two facilities are substantially above the diagonal. Facility `a79c` has a predicted rate of 15% but an observed rate of 30%; facility `2b76` predicts 14% against 21% observed. Both are candidates for care-quality audits or genuine underreporting review.
- **RTH scatter**: facilities `a615` and `2b76` both show observed RTH rates of ~13–14% against model predictions of only ~3–4%, a 3.5× under-prediction. These facilities may have resident populations with acute decompensation patterns not fully captured in the structured data.
- **Wound scatter**: facility `c9f8` predicts 9% but observes 22%, the largest single-facility divergence. Several others are moderately above the diagonal.
- **Facility `2b76` appears above the diagonal in both the fall and RTH scatter plots** — a possible signal of a systematically under-served or under-reported facility warranting direct insurer engagement.
- A minority of facilities lie below the diagonal in each plot, indicating the model over-predicts for them; these are likely lower-acuity populations or facilities with genuinely better outcomes than their case-mix profile would suggest.

#### Feature drivers — fall model

- `prior_days_since_last_fall` (mean |SHAP| = 0.720) is the single dominant feature, 3.3× larger than the second-ranked feature (`n_comorbidities` at 0.221). The fall model is fundamentally a recency-of-prior-fall detector with clinical adjustment.
- The SHAP dependence plot shows a sharp step: residents with a fall within the last 0–100 days contribute +1.5 to +1.7 SHAP units; those with no fall history (999 days) contribute −0.6 to −0.8 units. The relationship is highly non-linear, not continuous.
- `n_comorbidities` is monotone: zero comorbidities → SHAP ~−0.8; 10–12 comorbidities → SHAP ~+0.2.
- `vital_pain_min_30d` (rank 3, mean |SHAP| = 0.138) captures residents in chronic pain; the dependence plot shows a cluster of elevated pain observations (>1) with positive SHAP.
- `age_at_obs` contributes positively only for residents above ~80; younger residents push prediction down.
- `is_outpatient` flag generates a hard-negative cluster: outpatient (short-stay rehab) residents have markedly lower predicted fall risk, likely reflecting their shorter length of exposure.
- `need_fall_active` (rank 7, 0.071) and `need_nutrition_active` (rank 6, 0.096) are clinician-entered care plan flags — their presence confirms documented clinical concern, which is itself a validated risk signal.
- `facility_id_enc` ranks 9th (0.050), present but not dominant; the model has learned a legitimate facility-level adjustment without over-relying on it.

#### Feature drivers — RTH model

- `n_comorbidities` (mean |SHAP| = 0.443) is dominant at 2.3× the second feature. The SHAP dependence plot shows the widest range of any feature across all models: zero comorbidities drives SHAP as low as −1.5; high comorbidity burden contributes up to +0.5. Cumulative disease burden is the primary RTH signal.
- `prior_rth_days_since_last` (0.196): recent prior RTH within 0–250 days contributes +0.6 to +0.8 SHAP; beyond 1,000 days (no history) the contribution approaches zero or turns negative.
- `age_at_obs` (0.127) and `los_days` (0.090) contribute broadly, though their SHAP dependence plots show noisy, non-monotone relationships — they capture background frailty rather than a specific acute signal.
- `facility_id_enc` ranks 5th (0.070), above several clinical features. This is a residual risk: the model has learnt that some facilities have systematically higher RTH rates even after controlling for resident-level features, which limits generalization to new facilities.
- Pain variability (`vital_pain_std_30d`, `vital_pain_mean_30d`) and O2 saturation features appear in the top 20, confirming that vital sign instability is a meaningful RTH signal — consistent with the clinical literature on acute deterioration [4].
- The RTH model feature set is broader and more evenly distributed than the fall model (no single feature dominates above 0.443 vs 0.720 for fall), reflecting that RTH risk is driven by a combination of chronic burden and acute signals rather than a single dominant prior-event history.

#### Feature drivers — wound model

- Removing `facility_id_enc` changes the wound model's top features fundamentally: the 30-day model has `facility_id_enc` at rank 2 (0.176); the 60-day model replaces it with `dx_dementia` (0.219), a clinically appropriate driver.
- The wound (60d) top-4 — `dx_dementia` (0.219), `n_comorbidities` (0.214), `age_at_obs` (0.181), `los_days` (0.171) — form a chronic frailty profile. This is consistent with the clinical risk factors for pressure injuries: immobility, cognitive impairment, advanced age, long institutional stay [5].
- The SHAP dependence for `dx_dementia` is a clean binary signal: absent → −0.4 to 0 SHAP; present → 0 to +0.45. It is the clearest single-feature effect across all wound SHAP plots.
- `prior_days_since_last_fall` (rank 5, 0.134) contributes to wound prediction — a recent fall implies immobility, which is a primary wound risk factor. This cross-target signal is clinically plausible.
- Despite the improved feature profile, the wound model's discrimination is limited (ROC-AUC 0.662): the features drive SHAP values but the model cannot reliably separate who will develop a wound within 60 days from those who will not, given the data available.

#### Cross-target patterns

- `n_comorbidities` is the most universally important feature — ranked 1st for RTH and wound, 2nd for fall. Comorbidity burden is the clearest generalized risk signal across all outcome types.
- `prior_days_since_last_fall` cross-contaminates into wound (normalised importance ~0.6) and RTH (~0.1), suggesting a recent fall is a broader signal of acute health deterioration, not just a predictor of subsequent falls.
- `dx_dementia` is nearly exclusively a wound feature (normalised ~1.0 for wound vs ~0.1 for fall and near-zero for RTH), validating that the three models have learned meaningfully distinct clinical signatures rather than duplicating the same signal.
- Vital sign features (pain mean/std/min/max, O2 mean/std, glucose std, respiratory rate) appear in the top 20 of all three models but carry more weight for fall and RTH than wound, reflecting that acute physiological instability drives near-term deterioration events more than wound development.
- The three separate models are validated by their distinct feature profiles: training a single multi-label model would have compressed these differences and likely degraded performance on each individual outcome.  


### Business findings

### The fall model is the only one ready for operational use

With ROC-AUC of 0.822 and a positive Brier Skill Score of +0.144, the fall model discriminates meaningfully and is well-calibrated. A Brier Skill Score above zero means the model beats a naive predictor that always guesses the 7.4% base rate — and at +0.144 the margin is substantial. The calibrated score range of 0.8% to 11.8% across active residents is narrow enough to reflect true probabilities (no resident has a wildly inflated score) and wide enough to rank meaningfully (the top-risk resident is 15× higher than the lowest-risk). At the optimal operating threshold of 19.7%, the model achieves 59.5% sensitivity with 89% specificity and a PPV of 30.1% — meaning 1 in 3 flagged residents will actually fall within 30 days, versus 1 in 14 at random. Facility-level calibration is nearly perfect: the mean predicted rate is 1.10× the observed rate across 20 test facilities.

### The RTH model discriminates but barely beats the no-skill baseline

ROC-AUC of 0.689 confirms the model does rank residents correctly more often than chance — a resident who will experience an RTH event in the next 60 days is ranked above a non-event resident 69% of the time. But the Brier Skill Score of +0.019 is just barely positive, meaning the calibrated probabilities contribute almost nothing beyond the base rate in terms of financial accuracy. The deeper problem is score compression: 88% of residents receive an RTH score below 3%, and the interquartile range spans only 1.3%–1.5%. The model correctly identifies that most residents have low RTH risk, but it provides almost no separation within the high-risk minority. At the optimal threshold of 6.1%, sensitivity reaches 50% — half of all RTH events are captured — but PPV is only 8%, meaning 12 of every 13 flagged residents will not have an RTH event.

### The wound model is a no-skill predictor

A Brier Skill Score of −0.019 means the wound model is marginally worse than simply predicting the base rate for every resident. The score distribution makes this concrete: the 90th percentile of wound scores across all 612 active residents is 4.75%, only 0.3 percentage points above the 4.53% base rate. The model assigns virtually the same score to 90% of the population. The few residents who receive elevated wound scores (maximum 36.7%) tend to have extreme comorbidity profiles — high Charlson score, active dementia, malnutrition diagnosis, long length of stay — but the vast majority who share some of these characteristics receive near-base-rate predictions. There is no reliable clinical signal the model can extract from the current data to separate wound risk within this population.

### Calibrated probabilities are financially usable

The mean predicted probabilities match observed event rates to within reasonable bounds: fall 7.30% vs 7.36% true, RTH 3.05% vs 3.44% true, wound 6.85% vs 4.53% true. The total 30/60-day expected claim exposure for 612 active residents is \$466K, with individual resident estimates ranging from \$277 to \$2,652. These numbers are grounded in true event probabilities rather than inflated scores, which makes the expected-cost formula — `fall_score × $3,500 + rth_score × $20,000 + wound_score × $4,000` — usable for loss reserve purposes. The wound model's slight over-prediction (mean 6.85% vs 4.53% true) adds a conservative buffer to the wound component of the estimate.

### SHAP drivers reflect genuine low-risk population, not model failure

All top features show negative mean SHAP values — they push predicted scores below the base rate on average. This is the correct behavior for a population where 93% of residents have no incident in any given 30-day window. The dominant fall feature, `prior_days_since_last_fall`, is strongly positive (mean SHAP +0.99) for the minority of residents who did fall recently, and strongly negative for the 83% with no recent fall history. `n_comorbidities` is positive 84.5% of the time — meaning the model mostly uses it in the risk-increasing direction, as expected — but because the absolute magnitude is smaller, the feature appears negative at population level. The clinical interpretation of top drivers for each model is sensible: fall risk is anchored to fall history and comorbidity burden; RTH risk to comorbidity load and prior hospitalization; wound risk to chronic disease diagnoses (dementia, malnutrition, stroke) and age.


### Business actions

**Deploy the fall model for daily facility triage.** Ranking the 612 active residents by `fall_30d_score` and flagging the top quartile (approximately 153 residents) for clinical review is justified by the model's discrimination quality. With PPV of 30.1%, 1 in 3 flagged residents will fall without intervention — a rate 4× higher than random screening. The expected cost of a fall-prevention intervention (PT referral, environmental modification, medication review) is materially lower than the $3,500 average claim cost. Run this list every morning, prioritising residents with `prior_days_since_last_fall` below 30 and active `need_fall_active` flags.

**Use RTH scores for watchlisting the extreme top-decile only.** The score compression means the RTH model only reliably identifies outlier residents — those in roughly the top 5–10% of `rth_60d_score`. At a score of 8.8% (the observed maximum), the odds of an RTH event in 60 days are approximately 2.5× the base rate. These residents should be flagged for weekly physician review: medication reconciliation, fluid balance checks, and early escalation protocols. Do not use RTH scores for financial modeling at the individual level given the low PPV of 8% — the model captures the direction of risk but not the magnitude at the individual-resident level.

**Use \$466K as the 30/60-day expected claims estimate.** After removing the wound component (which is unreliable), the fall and RTH contributions alone sum to approximately \$298K (\$64K fall + \$234K RTH). Adding a 25% reserve buffer for model uncertainty yields a working claims provision of \$370K–\$466K for this facility population. This figure should be reviewed at the facility level: the RTH model under-predicts at 10 of the 20 test facilities (mean pred/obs ratio 0.67×), suggesting those facilities may generate higher-than-estimated RTH claims. A targeted clinical audit of those specific facilities — reviewing care protocols, discharge planning practices, and physician-on-call coverage — is warranted regardless of the model's exact score.

**Schedule quarterly recalibration.** The isotonic calibrators can be refitted in minutes using `05_calibration.py` as new OOF data accumulates. Recalibrate when: (a) the mean predicted rate drifts more than 1 percentage point from the observed rate in recent months, or (b) the resident population mix changes substantially (e.g., a new facility onboards). Full model retraining should be triggered if ROC-AUC on held-out facilities drops below 0.78 for fall or 0.65 for RTH in the monitoring dashboard.

---

## 13. Open limitations identified from live scoring

This section documents limitations discovered by running `risk_scoring.py` against the live active-resident population. These are open findings that require either further model development or awareness when interpreting the dashboard output.

**Limitation 1: RTH isotonic calibrator collapses to a small number of discrete score levels**

Running the scorer against 612 active residents produces only 12 unique RTH score values. The most common, 1.30%, is assigned to 359 residents (58.7% of the population). The second most common, 0.90%, covers another 84 residents. The top two values alone account for 73% of all active residents.

This is a direct consequence of isotonic regression's mechanics. The calibrator fits a piecewise constant (step function) mapping from raw XGBoost scores to calibrated probabilities. When the underlying model has limited discrimination — the RTH ROC-AUC is 0.689 — the calibrator finds only a small number of probability levels that meaningfully separate OOF predictions from the true labels. It cannot create calibration distinctions the model did not make. The result is that the RTH score behaves more like a coarse risk category (12 bins) than a continuous probability, which limits its use for anything finer than identifying the extreme high-risk tail.

The practical consequence for clinical use: RTH scores near the median (1.30%) should not be interpreted as precise probability estimates. A resident with an RTH score of 1.30% should be understood as "in the large group with similar risk profiles," not as someone with a precisely measured 1.30% 60-day RTH probability.

*Remediation path:* improving the RTH model's discrimination (by adding better clinical features, extending the label horizon further, or using claims-linked labels as described in Section 1) would give the calibrator more variation to map and would naturally produce a richer score distribution. Alternatively, applying a secondary ranking within the modal score group using fall risk or comorbidity burden could partially restore meaningful differentiation for the majority of residents.

**Limitation 2: the Medium tier is unexpectedly thin (8.7% instead of the intended 25%)**

The tier assignment applies `pd.cut` to `expected_cost_30d` using the population's 50th and 75th percentiles as boundaries. With well-distributed expected costs, this would yield 50% Low, 25% Medium, and 25% High. The actual split on 612 residents is 66.7% Low, 8.7% Medium, and 24.7% High.

The root cause is the RTH score discretization described above. Because 359 residents share the same RTH score of 1.30%, their expected costs cluster tightly. The 50th percentile falls at $401 and the 75th percentile at $444 — a gap of only $43. `pd.cut` uses left-open intervals, so residents with a cost of exactly $401 fall into the Low bin (since Low captures the interval (−0.001, 401]), not Medium. Of the 163 residents whose expected cost is exactly $401, all are classified Low. The Medium band captures only the 53 residents with costs strictly above $401 and at or below $444.

In practical terms, the Medium tier does not form a meaningfully distinct clinical group between High and Low. Facility staff should treat the dashboard as an effective two-category system in its current state: the 151 High-tier residents (top 25% by expected cost, all with costs above $444) are the primary action list; the remaining 461 residents form a monitoring pool regardless of whether they are formally labeled Medium or Low.

*Remediation path:* the same as for Limitation 1 — improving RTH discrimination would spread expected costs more continuously and restore a useful Medium tier. In the interim, the dashboard could be configured to display only High / Not High rather than three tiers, to avoid communicating a false precision in the Medium/Low distinction.

**Limitation 3: RTH SHAP drivers dominate 99.5% of top-driver explanations, including residents for whom fall risk is more immediately actionable**

The "top drivers" column shows SHAP explanations from whichever `DASHBOARD_ACTIVE_TARGETS` model contributes most to that resident's expected cost. Because the RTH multiplier is $20,000 versus $3,500 for falls, fall only dominates when a resident's fall score exceeds 10.9% — the break-even point where `fall_score × $3,500 > rth_score × $20,000` at the population mean RTH score of 1.91%. Only 3 of 612 residents reach this threshold. The remaining 609 receive RTH SHAP drivers, even when their fall score may be substantially elevated and represent the more actionable clinical concern.

A care coordinator reviewing a resident with a fall score of 9% (in the top decile) and an RTH score of 1.3% (modal value) will see RTH drivers in the dashboard — cardiopulmonary diagnoses and prior transfer history — when the clinically useful intervention is fall prevention: PT enrollment, medication review, environmental modification. The financial ranking is correct (RTH cost dominates), but the displayed drivers may not always point to the intervention that is most feasible or most likely to prevent the highest-cost event for that specific resident.

This is not a bug in the dominant-model logic — selecting drivers from the highest-cost model is the right financial framing. But it means the "top drivers" column is most useful for residents near the top of the list (where RTH risk is genuinely elevated, not just modally present) and less useful for residents whose ranking is primarily driven by a high fall score rather than elevated RTH risk.

*Remediation path:* show both models' top drivers when the expected costs are within 20% of each other, allowing care coordinators to see the full clinical picture for residents who are meaningfully elevated on both targets. Alternatively, add a separate "fall risk drivers" tooltip or secondary column that always shows fall SHAP drivers regardless of which model is dominant for cost purposes.

---

## References

[1] Werner, G., & Modlin, C. (2016). *Basic Ratemaking* (5th ed.). Casualty Actuarial Society. Defines loss ratio, pure premium, experience rating, adverse selection, and moral hazard in P&C insurance. https://www.casact.org/sites/default/files/old/studynotes_werner_modlin_ratemaking.pdf

[2] Leipzig, R. M., Cumming, R. G., & Tinetti, M. E. (1999). Drugs and falls in older people: a systematic review and meta-analysis: I. Psychotropic drugs. *Journal of the American Geriatrics Society*, 47(1), 30–39. DOI: 10.1111/j.1532-5415.1999.tb01898.x. PMID: 9920227. Meta-analysis establishing the association between sedative/hypnotic, antidepressant, and neuroleptic drug classes and falls in older adults.

[3] Woolcott, J. C., Richardson, K. J., Wiens, M. O., et al. (2009). Meta-analysis of the impact of 9 medication classes on falls in elderly persons. *Archives of Internal Medicine*, 169(21), 1952–1960. DOI: 10.1001/archinternmed.2009.357. PMID: 19933955. Bayesian meta-analysis confirming sedatives, antidepressants, and benzodiazepines as significant fall risk factors; underpins the psychotropic and opioid exposure features.

[4] Ouslander, J. G., Lamb, G., Perloe, M., et al. (2010). Potentially avoidable hospitalizations of nursing home residents: frequency, causes, and costs. *Journal of the American Geriatrics Society*, 58(4), 627–635. DOI: 10.1111/j.1532-5415.2010.02768.x. PMID: 20398146. Documents that the majority of SNF hospitalizations are potentially avoidable and identifies cardiopulmonary conditions and vital-sign instability as primary drivers of unplanned transfers.

[5] European Pressure Ulcer Advisory Panel, National Pressure Injury Advisory Panel, & Pan Pacific Pressure Injury Alliance. (2019). *Prevention and Treatment of Pressure Ulcers/Injuries: Clinical Practice Guideline* (3rd ed.). EPUAP/NPIAP/PPPIA. Establishes immobility, malnutrition, cognitive impairment, advanced age, and long institutional stay as the principal risk factors for pressure injury development. https://npiap.com/page/2019Guideline

[6] Quan, H., Li, B., Couris, C. M., et al. (2011). Updating and validating the Charlson comorbidity index and score for risk adjustment in hospital discharge abstracts using data from 6 countries. *American Journal of Epidemiology*, 173(6), 676–682. DOI: 10.1093/aje/kwq433. Updated ICD-10 coding rules and age-adjusted weights for the Charlson Comorbidity Index; the specification used in feature engineering.

[7] Maher, R. L., Hanlon, J., & Hajjar, E. R. (2014). Clinical consequences of polypharmacy in elderly. *Expert Opinion on Drug Safety*, 13(1), 57–65. DOI: 10.1517/14740338.2013.827660. PMID: 24073682. Documents that CMS implemented a quality indicator targeting patients on ≥9 medications as a polypharmacy threshold in US nursing homes; source for the ≥9-drug flag.

[9] Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems* 30 (NeurIPS 2017), pp. 4766–4777. arXiv: 1705.07874. Introduces SHAP (SHapley Additive exPlanations) as a unified framework for feature attribution; theoretical foundation for all SHAP outputs in the evaluation and dashboard.

[10] Lundberg, S. M., Erion, G., Chen, H., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67. DOI: 10.1038/s42256-019-0138-9. Introduces TreeSHAP, a polynomial-time algorithm for computing exact SHAP values for tree-based models including XGBoost; the implementation used by `shap.TreeExplainer`.

[12] Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of the 22nd International Conference on Machine Learning (ICML 2005)*, pp. 625–632. DOI: 10.1145/1102351.1102430. Empirically shows that maximum-margin methods (including boosted trees) produce systematically distorted probabilities and that isotonic regression outperforms Platt scaling for correcting non-linear miscalibration; motivates the choice of isotonic over sigmoid calibration.

[13] Davis, J., & Goadrich, M. (2006). The relationship between precision-recall and ROC curves. *Proceedings of the 23rd International Conference on Machine Learning (ICML 2006)*, pp. 233–240. DOI: 10.1145/1143844.1143874. Proves that precision-recall curves are more informative than ROC curves for highly imbalanced datasets; justification for using PR-AUC as the primary tuning and evaluation metric.

[14] Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3. DOI: 10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2. Introduces the Brier score as a proper scoring rule for probabilistic forecasts; the calibration metric used throughout model evaluation and for the Brier Skill Score.

[15] Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on tabular data? *Advances in Neural Information Processing Systems* 36 (NeurIPS 2022), Datasets and Benchmarks Track. arXiv: 2207.08815. https://arxiv.org/abs/2207.08815. Benchmarks tree-based models against deep learning architectures across 45 tabular datasets and finds that gradient-boosted trees remain state-of-the-art on medium-sized datasets (~10K samples); supports the choice of XGBoost over sequence models for this problem.
