"""
config.py
────────────
Central configuration for the SNF Incident Risk Model.
All paths, constants, feature definitions, and hyperparameters live here.
Adjust DATA_DIR / OUTPUT_DIR to match your local environment.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent.parent / "data"     # raw parquet files
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"  # models, plots, metrics

MODELS_DIR  = OUTPUT_DIR / "models"
PLOTS_DIR   = OUTPUT_DIR / "plots"
METRICS_DIR = OUTPUT_DIR / "metrics"

# ── Temporal boundaries ────────────────────────────────────────────────────
DATA_START            = "2023-08-01"
DATA_END              = "2024-12-31"   # latest valid obs_date (see LABEL_HORIZONS)
DATA_AVAILABILITY_END = "2025-01-31"   # last date with reliable records in source tables
MIN_HISTORY_DAYS      = 14
OBS_STEP_DAYS         = 7

# ── Feature look-back windows ──────────────────────────────────────────────
WINDOWS = [7, 30, 90]

# ── Target columns and per-target prediction horizons ─────────────────────
# RTH and wound use a 60-day horizon (instead of 30) to increase positive
# label count from 2% → ~4% and 2.5% → ~5%, giving the model more signal.
# Renaming the targets (rth_60d / wound_60d) makes the horizon explicit.
# DATA_END is set so that all obs_dates have a valid 30-day label window;
# rows where obs_date + horizon > DATA_AVAILABILITY_END are excluded per target.
LABEL_HORIZONS: dict[str, int] = {
    "fall_30d":  30,
    "rth_60d":   60,
    "wound_60d": 60,
}
TARGETS = list(LABEL_HORIZONS.keys())

# ── Comorbidity ICD-10 prefix map ──────────────────────────────────────────
COMORBIDITY_MAP = {
    "dx_dementia":          (["F00","F01","F02","F03","G30"],           2),
    "dx_hemi_paraplegia":   (["G81","G82","G83"],                       2),
    "dx_diabetes_cc":       (["E10","E11","E12","E13","E14"],           1),
    "dx_diabetes_comp":     (["E102","E112","E122","E132","E142"],       2),
    "dx_chf":               (["I50"],                                   1),
    "dx_copd":              (["J43","J44"],                             1),
    "dx_stroke":            (["I60","I61","I62","I63","I64","I65","I66","I69"], 1),
    "dx_parkinsons":        (["G20","G21"],                             0),
    "dx_ckd":               (["N17","N18","N19"],                       2),
    "dx_hypertension":      (["I10","I11","I12","I13"],                 0),
    "dx_osteoporosis":      (["M80","M81"],                             0),
    "dx_depression":        (["F32","F33"],                             0),
    "dx_anxiety":           (["F40","F41"],                             0),
    "dx_malnutrition":      (["E40","E41","E42","E43","E44","E45","E46"], 0),
    "dx_uti":               (["N390"],                                  0),
    "dx_pressure_ulcer":    (["L89"],                                   0),
    "dx_anemia":            (["D50","D51","D52","D53","D64"],           0),
    "dx_afib":              (["I48"],                                   0),
    "dx_pneumonia":         (["J13","J15","J18"],                       0),
    "dx_sepsis":            (["A40","A41"],                             2),
    "dx_cancer":            (["C"],                                     2),
    "dx_obesity":           (["E66"],                                   0),
    "dx_liver_mild":        (["K70","K73","K74"],                       1),
    "dx_liver_severe":      (["K721","K729"],                           3),
    "dx_mi":                (["I21","I22"],                             1),
    "dx_pvd":               (["I70","I71","I739"],                      1),
}

# ── ADL domains ───────────────────────────────────────────────────────────
ADL_DOMAIN_MAP = {
    "eating":          "Eating - Self-Performance",
    "personal_hygiene":"Personal hygiene - Self-Performance",
    "bed_mobility":    "Bed mobility - Self-Performance",
    "transfer":        "Transfer - Self-Performance",
    "toilet_use":      "Toilet use - Self-Performance",
    "locomotion":      "Locomotion on unit - Self-Performance",
    "walking":         "Walk in corridor - Self-Performance",
}

# ── GG task groups ────────────────────────────────────────────────────────
GG_MOBILITY_TASKS = [
    "Sit to Stand", "Chair / Bed-to-Chair Transfer",
    "Lying to Sitting on Side of Bed", "Walk 50 Feet with Two Turns",
    "Walk 150 Feet",
]
GG_SELFCARE_TASKS = [
    "Eating", "Oral Hygiene", "Upper Body Dressing", "Lower Body Dressing",
    "Toileting Hygiene",
]

# ── Vital sign types ──────────────────────────────────────────────────────
VITAL_TYPES = [
    "BP - Systolic", "Pulse", "O2 sats", "Blood Sugar",
    "Temperature", "Respiration", "Weight", "Pain Level",
]
VITAL_COL_NAMES = {
    "BP - Systolic": "bp_sys", "Pulse": "pulse", "O2 sats": "o2",
    "Blood Sugar": "glucose", "Temperature": "temp",
    "Respiration": "resp", "Weight": "weight", "Pain Level": "pain",
}

# ── Clinical threshold flags ──────────────────────────────────────────────
VITAL_FLAGS = {
    "o2_low":       ("O2 sats",       "<",   92.0),
    "pain_high":    ("Pain Level",    ">=",   7.0),
    "bp_high":      ("BP - Systolic", ">=", 160.0),
    "bp_low":       ("BP - Systolic", "<",   90.0),
    "fever":        ("Temperature",   ">=",  38.0),
    "tachycardia":  ("Pulse",         ">=", 100.0),
    "bradycardia":  ("Pulse",         "<",   60.0),
    "hypoglycemia": ("Blood Sugar",   "<",   70.0),
}

# ── Medication drug-class patterns (case-insensitive regex) ──────────────
PSYCHOTROPIC_PATTERNS = [
    r"haloperidol", r"quetiapine", r"risperidone", r"olanzapine",
    r"aripiprazole", r"clozapine", r"ziprasidone",
    r"lorazepam", r"diazepam", r"clonazepam", r"alprazolam",
    r"temazepam", r"oxazepam", r"midazolam", r"triazolam",
    r"zolpidem", r"zaleplon", r"eszopiclone",
    r"amitriptyline", r"nortriptyline", r"imipramine",
    r"trazodone", r"mirtazapine", r"doxepin",
    r"diphenhydramine", r"promethazine",
]
OPIOID_PATTERNS = [
    r"oxycodone", r"hydrocodone", r"morphine", r"fentanyl",
    r"tramadol", r"codeine", r"methadone", r"buprenorphine",
    r"oxymorphone", r"hydromorphone", r"meperidine",
]
DIURETIC_PATTERNS = [
    r"furosemide", r"torsemide", r"bumetanide",
    r"hydrochlorothiazide", r"hctz", r"chlorthalidone",
    r"spironolactone", r"metolazone",
]
ANTICOAGULANT_PATTERNS = [
    r"warfarin", r"coumadin", r"apixaban", r"rivaroxaban",
    r"dabigatran", r"edoxaban", r"enoxaparin", r"heparin",
]

# ── High-signal document tags ─────────────────────────────────────────────
HIGH_RISK_DOC_TAGS = [
    "fall", "fall_risk", "wound_care", "skin_wound_assessment",
    "wound_infection_symptoms", "uti_symptoms", "antibiotic_therapy",
    "aggressive_behavior", "altered_mental_status", "confusion",
    "weight_progress_note", "nutrition_plan", "physician_notification",
    "refusal", "pain_intervention", "pain_progress_note",
    "monitor_side_effects", "family_responsible",
]

# ── XGBoost base hyperparameters ──────────────────────────────────────────
# tree_method="hist" enables the histogram-based split-finding algorithm —
# the same approach LightGBM uses by default, giving comparable speed.
# XGBoost does not have a num_leaves parameter; depth is controlled purely
# by max_depth (each level doubles the max leaves: depth-6 → up to 64 leaves).
XGB_BASE_PARAMS = {
    "objective":          "binary:logistic",
    "eval_metric":        "aucpr",           # PR-AUC monitored during early stopping
    "tree_method":        "hist",            # histogram split-finding (fast, memory-efficient)
    "booster":            "gbtree",
    "n_estimators":       500,
    "learning_rate":      0.05,
    "max_depth":          6,
    "min_child_weight":   30,                # minimum sum of instance weights in a leaf
                                             # (analogous to LightGBM's min_child_samples
                                             #  when all weights = 1)
    "subsample":          0.80,
    "colsample_bytree":   0.80,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "scale_pos_weight":   None,              # set dynamically per target class ratio
    "n_jobs":             -1,
    "random_state":       42,
    "verbosity":          0,
    "early_stopping_rounds": 50,            # set in constructor; active when eval_set given
}

# ── Optuna search space ───────────────────────────────────────────────────
OPTUNA_SEARCH_SPACE = {
    "learning_rate":    (0.02, 0.15),
    "max_depth":        (4, 8),
    "min_child_weight": (10, 100),   # weight-based; ~equivalent to 10-100 min samples
    "subsample":        (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "reg_alpha":        (0.0, 2.0),
    "reg_lambda":       (0.5, 5.0),
}
OPTUNA_N_TRIALS       = 30
N_CV_FOLDS            = 5
EARLY_STOPPING_ROUNDS = 50

# ── scale_pos_weight cap ───────────────────────────────────────────────────
# The raw ratio n_negative/n_positive can reach 49× for a 2% positive-rate
# target, which severely overcorrects class imbalance and inflates all
# predicted probabilities well above their true values (confirmed by the
# BSS of -9.4 for rth_30d in previous version).  Capping at 10× limits this distortion
# while still up-weighting the minority class meaningfully.
# Post-training isotonic calibration corrects any residual bias.
SCALE_POS_WEIGHT_MAX = 10.0

# ── Per-target feature exclusions ─────────────────────────────────────────
# facility_id_enc ranks 2nd by |SHAP| in the wound model but represents
# memorized facility effects that do not generalize to unseen facilities.
# Excluding it forces the model to learn portable clinical signals instead.
EXCLUDE_FEATURES_BY_TARGET: dict[str, list[str]] = {
    "wound_60d": ["facility_id_enc"],
}

# ── Post-training probability calibration ─────────────────────────────────
# Isotonic regression fitted on out-of-fold (OOF) predictions corrects the
# systematic over-prediction caused by scale_pos_weight without discarding
# any training data for a separate calibration holdout.
# The calibrator is saved alongside each model and applied transparently at
# evaluation and scoring time.
CALIBRATION_METHOD = "isotonic"   # "isotonic" or "sigmoid" (Platt scaling)

# ── Risk tier assignment: percentile-based on expected cost ──────────────
# Tiers are assigned by ranking residents on their total expected claim cost
# (fall_score × $3,500 + rth_score × $20,000) rather than per-target score
# percentiles.  The previous per-target approach caused ~58% of residents to
# be flagged "High" (union of three 25%-High targets), defeating triage.
# Tiering on the combined cost column guarantees exactly 25% High, 25%
# Medium, 50% Low regardless of score level or number of active models.
RISK_TIER_PERCENTILES = {"high": 75, "medium": 50}

# ── Dashboard-active targets ──────────────────────────────────────────────
# Only targets listed here contribute to expected cost and tier assignment.
# wound_60d is suppressed: Brier Skill Score = -0.019 (below no-skill
# baseline), and 90% of residents receive a wound score within 0.3 pp of the
# 4.5% base rate — the model predicts nearly everyone at the base rate and
# provides no useful triage signal.  It is still scored and saved to the CSV
# for monitoring; re-add it here once discrimination improves.
DASHBOARD_ACTIVE_TARGETS: list[str] = ["fall_30d", "rth_60d"]


# ── Charlson score age adjustment ─────────────────────────────────────────
def charlson_age_points(age: float) -> int:
    if age < 50: return 0
    if age < 60: return 1
    if age < 70: return 2
    if age < 80: return 3
    return 4
