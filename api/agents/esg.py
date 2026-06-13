"""ESG Energy & Emissions agent: prediction + RAG.

Ports TAB 1 logic from app.py.
"""
from functools import lru_cache

from ..config import REPO_ROOT          # inserts repo root on sys.path (import first)
from ..rag import rag_answer
from src.ml.predict import EnergySavingsPredictor

ESG_VECTOR_DIR = REPO_ROOT / "rag_project_artifacts" / "vector_store"

ESG_SYSTEM_PROMPT = """
You are an energy-efficiency decision-support assistant.

Rules (must follow):
- Use ONLY the provided CONTEXT for factual claims.
- Do NOT invent facts or standards.
- If the CONTEXT is insufficient, say so and ask a specific follow-up question.
- Always include citations to the retrieved sources.

Return format:
1) Answer
2) Why
3) Evidence (bullets with source + chunk)
4) Confidence (High/Medium/Low)
5) Human Review Trigger (Yes/No + reason)
"""


@lru_cache(maxsize=1)
def _predictor():
    return EnergySavingsPredictor()


def predict(req) -> dict:
    inputs = {
        "floor_area_m2": float(req.floor_area_m2),
        "building_age_years": float(req.building_age_years),
        "hvac_efficiency_score": float(req.hvac_efficiency_score),
        "insulation_quality_score": float(req.insulation_quality_score),
        "occupancy_rate": float(req.occupancy_rate),
        "baseline_energy_kwh": float(req.baseline_energy_kwh),
    }
    savings_pct = float(_predictor().predict(inputs))

    energy_saved_kwh = req.baseline_energy_kwh * savings_pct
    cost_saved = energy_saved_kwh * 0.25            # AUD/yr (indicative)
    tco2_saved = (energy_saved_kwh * 0.7) / 1000    # tCO2e/yr (Scope 2)

    return {
        "savings_pct": savings_pct,
        "energy_saved_kwh": energy_saved_kwh,
        "cost_saved": cost_saved,
        "tco2_saved": tco2_saved,
    }


def ask(question: str, k: int = 5) -> str:
    return rag_answer(ESG_VECTOR_DIR, ESG_SYSTEM_PROMPT, question, k)
