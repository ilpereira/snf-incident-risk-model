"""
Shared fixtures for all test modules.

Data design
───────────
Two residents (R001, R002) across two facilities (F001, F002).
Two observation dates: OBS1 = 2024-11-01, OBS2 = 2024-11-08.

Events are placed at obs-dates so that rolling-window joins to the spine
return non-NaN values. Future events (after OBS1) drive label columns.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))   # so test modules can import helpers

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from helpers import OBS1, OBS2, ADM, DOB1, DOB2
from config import TARGETS


# ── Residents ──────────────────────────────────────────────────────────────

@pytest.fixture
def residents():
    return pd.DataFrame({
        "resident_id":    ["R001", "R002", "R003"],
        "facility_id":    ["F001", "F001", "F002"],
        "admission_date": [ADM,    ADM,    ADM],
        "discharge_date": [pd.NaT, pd.NaT, pd.Timestamp("2024-11-20")],
        "deceased_date":  [pd.NaT, pd.NaT, pd.NaT],
        "date_of_birth":  [DOB1,   DOB2,   DOB2],
        "outpatient":     [False,  False,  False],
    })


# ── Observation spine (pre-built, 4 rows) ──────────────────────────────────

@pytest.fixture
def spine():
    return pd.DataFrame({
        "resident_id":    ["R001", "R001", "R002", "R002"],
        "facility_id":    ["F001", "F001", "F001", "F001"],
        "obs_date":       [OBS1,   OBS2,   OBS1,   OBS2],
        "admission_date": ADM,
        "date_of_birth":  [DOB1,   DOB1,   DOB2,   DOB2],
        "discharge_date": pd.NaT,
        "deceased_date":  pd.NaT,
        "outpatient":     False,
    })


# ── Diagnoses ──────────────────────────────────────────────────────────────

@pytest.fixture
def diagnoses():
    return pd.DataFrame({
        "resident_id": ["R001", "R001",   "R002"],
        # F00 → dx_dementia (w=2), I50 → dx_chf (w=1), I10 → dx_hypertension (w=0)
        "icd_10_code": ["F00.0", "I50.0", "I10"],
        "onset_at":    [pd.Timestamp("2024-09-01")] * 3,
        # CHF resolved before OBS1 → should NOT flag dx_chf
        "resolved_at": [pd.NaT, pd.Timestamp("2024-10-15"), pd.NaT],
        "strikeout":   [False, False, False],
    })


# ── Incidents ──────────────────────────────────────────────────────────────

@pytest.fixture
def incidents():
    return pd.DataFrame({
        "incident_id":   ["INC001", "INC002", "INC003"],
        "resident_id":   ["R001",    "R001",   "R002"],
        "incident_type": ["Fall",    "Fall",   "Wound"],
        "occurred_at": [
            OBS1,                           # at OBS1 → prior_fall rolling at OBS1
            OBS1 + pd.Timedelta(days=15),   # 15d after OBS1 → fall_30d label = 1
            OBS1,                           # at OBS1 → prior_wound rolling at OBS1
        ],
        "strikeout": [False, False, False],
    })


@pytest.fixture
def injuries():
    return pd.DataFrame({
        "injury_id":   ["INJ001"],
        "incident_id": ["INC001"],
        "description": ["Minor abrasion"],
    })


# ── Hospital transfers ─────────────────────────────────────────────────────

@pytest.fixture
def hospital_transfers():
    return pd.DataFrame({
        "resident_id":      ["R001",  "R001"],
        "effective_date": [
            OBS1,                          # at OBS1 → prior_rth rolling at OBS1
            OBS1 + pd.Timedelta(days=10),  # 10d after OBS1 → rth_30d label = 1
        ],
        "planned_flag":     [False,   False],
        "transfer_outcome": ["Admitted", "Admitted"],
    })


# ── ADL responses ──────────────────────────────────────────────────────────

@pytest.fixture
def adl_responses():
    rows = []
    for res_id in ["R001", "R002"]:
        for obs in [OBS1, OBS2]:
            rows.append({
                "resident_id":    res_id,
                "assessment_date": obs,
                "activity":       "Eating - Self-Performance",
                "category":       "Self-Performance",
                "response":       "2",   # string, as in real data
            })
    return pd.DataFrame(rows)


# ── GG responses ───────────────────────────────────────────────────────────

@pytest.fixture
def gg_responses():
    rows = []
    for res_id in ["R001", "R002"]:
        for obs in [OBS1, OBS2]:
            rows.append({
                "resident_id":   res_id,
                "created_at":    obs,
                "task_group":    "Mobility",
                "task_name":     "Sit to Stand",
                "response_code": 3,
            })
    return pd.DataFrame(rows)


# ── Vitals ─────────────────────────────────────────────────────────────────

@pytest.fixture
def vitals():
    rows = []
    for res_id in ["R001", "R002"]:
        for obs in [OBS1, OBS2]:
            rows.append({
                "resident_id": res_id,
                "measured_at": obs,
                "vital_type":  "BP - Systolic",
                # R002 has high BP (≥160) → flag_bp_high = 1
                "value": 120.0 if res_id == "R001" else 165.0,
                "strikeout": False,
            })
    # O2 reading below 92 → flag_o2_low = 1 for R001 at OBS1
    rows.append({
        "resident_id": "R001",
        "measured_at": OBS1,
        "vital_type":  "O2 sats",
        "value":       89.0,
        "strikeout":   False,
    })
    return pd.DataFrame(rows)


# ── Medications ────────────────────────────────────────────────────────────

@pytest.fixture
def medications():
    rows = []
    for desc in ["morphine 5mg", "warfarin 2mg", "furosemide 20mg",
                 "haloperidol 1mg", "metformin 500mg"]:
        rows.append({
            "resident_id":  "R001",
            "scheduled_at": OBS1,
            "description":  desc,
            "status":       "Given",
        })
    rows.append({
        "resident_id":  "R001",
        "scheduled_at": OBS1,
        "description":  "lisinopril 10mg",
        "status":       "Missed",
    })
    return pd.DataFrame(rows)


# ── Document tags ──────────────────────────────────────────────────────────

@pytest.fixture
def document_tags():
    return pd.DataFrame({
        "resident_id": ["R001",      "R001",              "R002"],
        "tag_id":      ["fall_risk", "pain_intervention", "wound_care"],
        "created_at":  [OBS1,        OBS1,                OBS1],
        "deleted_at":  [pd.NaT,      pd.NaT,              pd.NaT],
    })


# ── Lab reports ────────────────────────────────────────────────────────────

@pytest.fixture
def lab_reports():
    return pd.DataFrame({
        "resident_id":     ["R001",    "R001",    "R002"],
        "reported_at":     [OBS1,      OBS1,      OBS1],
        "severity_status": ["Abnormal", "Critical", "Normal"],
    })


# ── Needs ──────────────────────────────────────────────────────────────────

@pytest.fixture
def needs():
    return pd.DataFrame({
        "resident_id":   ["R001",  "R002"],
        "need_category": ["Fall",  "Wound"],
        "initiated_at":  [pd.Timestamp("2024-10-15")] * 2,
        "resolved_at":   [pd.NaT,  pd.NaT],
        "strikeout":     [False,   False],
    })


# ── Therapy tracks ─────────────────────────────────────────────────────────

@pytest.fixture
def therapy_tracks():
    return pd.DataFrame({
        "resident_id": ["R001"],
        "discipline":  ["PT"],
        "start_at":    [pd.Timestamp("2024-10-10")],
        "end_at":      [pd.NaT],
    })


# ── Full data dict (mirrors what load_raw_data returns) ────────────────────

@pytest.fixture
def data_dict(residents, diagnoses, adl_responses, gg_responses, vitals,
              medications, document_tags, lab_reports, incidents, injuries,
              hospital_transfers, needs, therapy_tracks):
    return {
        "residents":          residents,
        "diagnoses":          diagnoses,
        "adl_responses":      adl_responses,
        "gg_responses":       gg_responses,
        "vitals":             vitals,
        "medications":        medications,
        "document_tags":      document_tags,
        "lab_reports":        lab_reports,
        "incidents":          incidents,
        "injuries":           injuries,
        "hospital_transfers": hospital_transfers,
        "needs":              needs,
        "therapy_tracks":     therapy_tracks,
    }


# ── Training data (4 facilities, numeric features, binary labels) ──────────

@pytest.fixture(scope="session")
def training_data():
    rng = np.random.default_rng(42)
    n_fac, n_per = 4, 60
    n = n_fac * n_per
    fac_ids = np.repeat([f"FAC{i}" for i in range(n_fac)], n_per)
    res_ids = [f"R{i:04d}" for i in range(n)]
    feat_data = {f"feat_{i}": rng.standard_normal(n) for i in range(10)}

    features = pd.DataFrame({
        **feat_data,
        "resident_id": res_ids,
        "facility_id": fac_ids,
        "obs_date":    pd.Timestamp("2024-11-01"),
    })
    labels = pd.DataFrame({
        "resident_id": res_ids,
        "facility_id": fac_ids,
        "obs_date":    pd.Timestamp("2024-11-01"),
        "fall_30d":    (rng.random(n) > 0.85).astype(int),
        "rth_60d":     (rng.random(n) > 0.92).astype(int),
        "wound_60d":   (rng.random(n) > 0.90).astype(int),
    })
    return features, labels


@pytest.fixture(scope="session")
def feat_cols(training_data):
    features, _ = training_data
    return [c for c in features.columns
            if c not in ("resident_id", "facility_id", "obs_date")]


# ── Tiny trained models (one per target, session-scoped for speed) ─────────

@pytest.fixture(scope="session")
def models_dict(training_data, feat_cols):
    features, labels = training_data
    X = features[feat_cols]
    result = {}
    rng = np.random.default_rng(0)
    for target in TARGETS:
        y = labels[target]
        m = xgb.XGBClassifier(
            n_estimators=10, max_depth=2, learning_rate=0.1,
            random_state=42, verbosity=0, eval_metric="logloss",
        )
        m.fit(X, y)
        result[target] = m
    return result


@pytest.fixture(scope="session")
def eval_X(feat_cols):
    rng = np.random.default_rng(1)
    n = 60
    return pd.DataFrame(rng.standard_normal((n, len(feat_cols))), columns=feat_cols)


@pytest.fixture(scope="session")
def eval_y(eval_X):
    rng = np.random.default_rng(2)
    n = len(eval_X)
    facs = np.resize(["F001", "F002", "F003"], n)
    return pd.DataFrame({
        "facility_id": facs,
        "fall_30d":    (rng.random(n) > 0.85).astype(int),
        "rth_60d":     (rng.random(n) > 0.92).astype(int),
        "wound_60d":   (rng.random(n) > 0.90).astype(int),
    })


@pytest.fixture(scope="session")
def calibrators_dict(models_dict, training_data, feat_cols):
    from sklearn.isotonic import IsotonicRegression
    from config import TARGETS
    features, labels = training_data
    X = features[feat_cols]
    result = {}
    for target in TARGETS:
        y = labels[target]
        raw_preds = models_dict[target].predict_proba(X)[:, 1]
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(raw_preds, y.values)
        result[target] = cal
    return result


@pytest.fixture(scope="session")
def artifacts(models_dict, calibrators_dict, eval_X, eval_y, feat_cols):
    from config import TARGETS
    return {
        "models":           models_dict,
        "calibrators":      calibrators_dict,
        "X_test":           eval_X,
        "y_test":           eval_y,
        "feat_cols":        feat_cols,
        "target_feat_cols": {t: feat_cols for t in TARGETS},
    }
