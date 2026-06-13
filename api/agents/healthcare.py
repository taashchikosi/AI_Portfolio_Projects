"""Healthcare Patient-Flow agent: prediction + RAG.

Ports TAB 2 logic from app.py (build_health_feature_row + predict_healthcare).
Uses the LOCAL model artifacts (ignores app.py's GitHub-Release download path).
"""
import json
from functools import lru_cache

import joblib
import pandas as pd

from ..config import REPO_ROOT
from ..rag import rag_answer

HEALTH_DIR = REPO_ROOT / "healthcare_project_artifacts"
HEALTH_VECTOR_DIR = REPO_ROOT / "healthcare_rag_artifacts" / "vector_store"

HEALTH_SYSTEM_PROMPT = """
You are a hospital patient-flow decision-support assistant.

Rules (must follow):
- Use ONLY the provided CONTEXT for factual claims.
- Do NOT invent clinical facts, hospital policies, or staffing rules.
- If the CONTEXT is insufficient, say so and ask a specific follow-up question.
- Always include citations to the retrieved sources.

Return format:
1) Answer
2) Why
3) Evidence (bullets with source + chunk)
4) Confidence (High/Medium/Low)
5) Human Review Trigger (Yes/No + reason)
"""

ARRIVAL_PRESSURE_MAP = {"Low": 0.75, "Normal": 1.00, "High": 1.35}
STAFF_LEVEL_MAP = {"Low": 0.75, "Normal": 1.00, "High": 1.25}


def clamp(x, lo, hi):
    return float(max(lo, min(hi, x)))


@lru_cache(maxsize=1)
def _load():
    models = joblib.load(HEALTH_DIR / "models.joblib")
    meta = json.loads((HEALTH_DIR / "model_meta.json").read_text())
    return models, meta


def build_feature_row(meta, capacity_beds, current_occupancy_pct, arrivals_pressure,
                      staffing_level, current_wait_minutes, dt):
    FEATURES = meta["features"]

    occ_ratio = clamp(current_occupancy_pct / 100.0, 0.05, 1.08)

    base_arrivals = 6 + (capacity_beds / 700) * 19  # ~6..25
    arrivals_now = base_arrivals * ARRIVAL_PRESSURE_MAP[arrivals_pressure]

    staff_now = clamp(STAFF_LEVEL_MAP[staffing_level], 0.4, 1.6)
    wait_now = clamp(current_wait_minutes, 5, 360)

    hour = int(dt.hour)
    dayofweek = int(dt.dayofweek)
    month = int(dt.month)
    is_weekend = 1.0 if dayofweek >= 5 else 0.0

    arrivals_roll_mean_24h = arrivals_now * clamp(0.95 + 0.05 * (1 if 12 <= hour <= 20 else -1), 0.85, 1.05)
    arrivals_roll_max_24h = arrivals_now * (1.15 if arrivals_pressure != "Low" else 1.05)

    occ_roll_mean_24h = clamp(occ_ratio * (0.98 if is_weekend else 1.00), 0.05, 1.08)
    occ_roll_max_24h = clamp(occ_ratio + (0.06 if arrivals_pressure == "High" else 0.03), 0.05, 1.08)

    wait_roll_mean_24h = wait_now * (1.05 if arrivals_pressure == "High" else 1.00)
    wait_roll_max_24h = clamp(wait_now * (1.35 if arrivals_pressure == "High" else 1.15), 5, 360)

    row = {
        "capacity_beds": float(capacity_beds),
        "occupancy_ratio": float(occ_ratio),
        "arrivals": float(arrivals_now),
        "staff_index": float(staff_now),
        "wait_minutes": float(wait_now),
        "hour": float(hour),
        "dayofweek": float(dayofweek),
        "month": float(month),
        "is_weekend": float(is_weekend),
        "arrivals_roll_mean_24h": float(arrivals_roll_mean_24h),
        "arrivals_roll_max_24h": float(arrivals_roll_max_24h),
        "occ_roll_mean_24h": float(occ_roll_mean_24h),
        "occ_roll_max_24h": float(occ_roll_max_24h),
        "wait_roll_mean_24h": float(wait_roll_mean_24h),
        "wait_roll_max_24h": float(wait_roll_max_24h),
    }
    return pd.DataFrame([row])[FEATURES]


def predict(req) -> dict:
    models, meta = _load()
    dt = pd.Timestamp(req.date) + pd.Timedelta(hours=int(req.hour))
    X = build_feature_row(
        meta=meta,
        capacity_beds=req.capacity_beds,
        current_occupancy_pct=req.current_occupancy_pct,
        arrivals_pressure=req.arrivals_pressure,
        staffing_level=req.staffing_level,
        current_wait_minutes=req.current_wait_minutes,
        dt=dt,
    )

    pred_max_occ = float(models["rf_occ"].predict(X)[0])
    pred_mean_wait = float(models["rf_wait"].predict(X)[0])
    risk_high = int((pred_max_occ >= 0.95) or (pred_mean_wait >= 120))

    return {
        "pred_max_occ_ratio_24h": pred_max_occ,
        "pred_mean_wait_24h": pred_mean_wait,
        "risk_high": risk_high,
        "feature_row": {k: float(v) for k, v in X.iloc[0].items()},
    }


def ask(question: str, k: int = 5) -> str:
    return rag_answer(HEALTH_VECTOR_DIR, HEALTH_SYSTEM_PROMPT, question, k)
