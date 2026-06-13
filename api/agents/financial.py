"""Financial Credit Default Risk agent: defaults + prediction + portfolio + RAG.

Ports TAB 4 logic from app.py.
"""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from ..config import REPO_ROOT
from ..rag import rag_answer

FIN_DIR = REPO_ROOT / "financial_risk_agent"
FIN_MODEL_PATH = FIN_DIR / "models" / "pd_model_CHAMPION.joblib"
FIN_DATA_PATH = FIN_DIR / "data" / "consumer_loans_synthetic_v1.csv"
FIN_DECISIONS_PATH = FIN_DIR / "data" / "loan_decisions_with_reasons_actions_v3.csv"
FIN_VECTOR_DIR = FIN_DIR / "rag_artifacts" / "vector_store"

FIN_SYSTEM_PROMPT = """
You are a credit risk decision-support assistant for consumer lending.

Hard Rules (must follow):
- Use ONLY the provided CONTEXT for factual claims.
- Do NOT invent policy thresholds, model metrics, governance rules, or data facts.
- If the CONTEXT is insufficient, say: "Not enough evidence in the KB to answer that." Then ask ONE targeted follow-up question.
- Always cite evidence using the SOURCE filenames provided in the CONTEXT.
- Treat outputs as decision support, not autonomous lending decisions.

Return format (exact):
1) Answer
2) Why (brief reasoning)
3) Evidence (bullet list; cite SOURCE filenames)
4) Confidence (High/Medium/Low)
5) Human Review Trigger (Yes/No + reason)
"""

PD_APPROVE_MAX = 0.03
PD_REPRICE_MAX = 0.08
PD_REVIEW_MAX = 0.15

REGION_RISK_MAP = {"Inner Metro": 0.95, "Outer Metro": 1.00, "Regional": 1.08, "Remote": 1.18}


@lru_cache(maxsize=1)
def _model():
    return joblib.load(FIN_MODEL_PATH)


@lru_cache(maxsize=1)
def _data():
    return pd.read_csv(FIN_DATA_PATH)


@lru_cache(maxsize=1)
def _decisions():
    if FIN_DECISIONS_PATH.exists():
        return pd.read_csv(FIN_DECISIONS_PATH)
    return None


def decision_bucket(pd_val: float) -> str:
    if pd_val < PD_APPROVE_MAX:
        return "APPROVE"
    if pd_val < PD_REPRICE_MAX:
        return "APPROVE_REPRICE"
    if pd_val < PD_REVIEW_MAX:
        return "MANUAL_REVIEW"
    return "DECLINE"


def calc_interest_income(loan_amount, interest_rate, term_months, avg_balance_factor=0.5):
    term_years = float(term_months) / 12.0
    return float(loan_amount) * float(interest_rate) * term_years * avg_balance_factor


def calc_required_rate(loan_amount, term_months, ecl, operating_cost=150.0, target_profit_pct=0.02, avg_balance_factor=0.5):
    term_years = float(term_months) / 12.0
    denom = max(float(loan_amount) * term_years * avg_balance_factor, 1.0)
    target_profit = target_profit_pct * float(loan_amount)
    req = (target_profit + float(ecl) + float(operating_cost)) / denom
    return float(np.clip(req, 0.05, 0.35))


def defaults() -> dict:
    d = _data()
    median_fields = [
        "age", "annual_income", "dependents", "tenure_months", "credit_score",
        "delinquencies_12m", "inquiries_6m", "revolving_utilization", "total_open_accounts",
        "months_since_last_delinquency", "loan_amount", "term_months", "interest_rate",
        "unemployment_rate", "inflation_rate", "cash_rate_proxy",
    ]
    medians = {f: float(d[f].median()) for f in median_fields if f in d.columns}
    options = {
        "employment_status": sorted(d["employment_status"].unique().tolist()),
        "residence_type": sorted(d["residence_type"].unique().tolist()),
        "purpose": sorted(d["purpose"].unique().tolist()),
        "region": sorted(d["region"].unique().tolist()),
        "term_months": sorted(int(x) for x in d["term_months"].unique().tolist()),
    }
    modes = {
        "employment_status": d["employment_status"].mode()[0],
        "residence_type": d["residence_type"].mode()[0],
        "purpose": d["purpose"].mode()[0],
        "region": d["region"].mode()[0],
    }
    return {"medians": medians, "options": options, "modes": modes}


def _reason_codes(credit_score, payment_to_income_ratio, revolving_utilization,
                  delinquencies_12m, inquiries_6m, employment_status,
                  unemployment_rate, inflation_rate):
    reasons = []
    if credit_score <= 580:
        reasons.append(("Very low credit score", 5, f"credit_score={credit_score}"))
    elif credit_score <= 640:
        reasons.append(("Low credit score", 4, f"credit_score={credit_score}"))

    if payment_to_income_ratio >= 0.45:
        reasons.append(("Very high payment-to-income (affordability stress)", 5, f"PTI={payment_to_income_ratio:.2f}"))
    elif payment_to_income_ratio >= 0.35:
        reasons.append(("High payment-to-income (affordability stress)", 4, f"PTI={payment_to_income_ratio:.2f}"))

    if revolving_utilization >= 0.80:
        reasons.append(("Very high revolving utilization", 5, f"utilization={revolving_utilization:.2f}"))
    elif revolving_utilization >= 0.60:
        reasons.append(("High revolving utilization", 4, f"utilization={revolving_utilization:.2f}"))

    if delinquencies_12m >= 2:
        reasons.append(("Multiple delinquencies in last 12 months", 5, f"delinq_12m={delinquencies_12m}"))
    elif delinquencies_12m >= 1:
        reasons.append(("Recent delinquency in last 12 months", 4, f"delinq_12m={delinquencies_12m}"))

    if inquiries_6m >= 7:
        reasons.append(("Very high recent credit inquiries", 4, f"inquiries_6m={inquiries_6m}"))
    elif inquiries_6m >= 4:
        reasons.append(("High recent credit inquiries", 3, f"inquiries_6m={inquiries_6m}"))

    if employment_status == "Unemployed":
        reasons.append(("Unemployed (high income stability risk)", 5, "employment=Unemployed"))
    elif employment_status in ["Student", "Self-employed"]:
        reasons.append(("Income volatility risk (student/self-employed)", 3, f"employment={employment_status}"))

    macro_score = 0
    if unemployment_rate >= 8.0:
        macro_score += 2
    elif unemployment_rate >= 6.5:
        macro_score += 1
    if inflation_rate >= 7.0:
        macro_score += 2
    elif inflation_rate >= 5.5:
        macro_score += 1
    if macro_score >= 3:
        reasons.append(("Macro environment under high stress (unemployment/inflation)", 4, f"unemp={unemployment_rate:.1f} infl={inflation_rate:.1f}"))
    elif macro_score >= 2:
        reasons.append(("Macro environment stressed (unemployment/inflation)", 3, f"unemp={unemployment_rate:.1f} infl={inflation_rate:.1f}"))

    if not reasons:
        reasons = [("No major risk flags triggered (combined moderate factors)", 1, "—")]

    reasons = sorted(reasons, key=lambda x: x[1], reverse=True)[:4]
    return [{"reason": r, "severity": s, "detail": d} for r, s, d in reasons]


def _next_actions(decision: str):
    if decision == "APPROVE":
        return ["Auto-approve under standard terms", "Monitor early payment performance (first 60 days)"]
    if decision == "APPROVE_REPRICE":
        return [
            "Approve with risk-based pricing (increase rate to target margin)",
            "Offer shorter term or smaller principal to reduce PTI",
        ]
    if decision == "MANUAL_REVIEW":
        return [
            "Route to manual credit review",
            "Request supporting documents (payslips/bank statements)",
            "Counter-offer: reduce amount or require co-applicant/guarantor",
        ]
    return [
        "Decline under automated policy",
        "Offer alternative: secured product or smaller amount after seasoning period",
        "Provide adverse action notice + path to eligibility",
    ]


def predict(req) -> dict:
    model = _model()

    r_m = float(req.interest_rate) / 12.0
    n = int(req.term_months)
    installment_amount = float((req.loan_amount * r_m) / (1 - (1 + r_m) ** (-n))) if r_m > 0 else float(req.loan_amount / max(n, 1))
    monthly_income = float(req.annual_income) / 12.0
    payment_to_income_ratio = float(installment_amount / max(monthly_income, 1.0))
    region_risk_index = float(REGION_RISK_MAP.get(req.region, 1.0))

    # Origination scenario placeholders (mirror app.py)
    months_on_book = 0
    current_balance = float(req.loan_amount)
    missed_payments_3m = 0
    days_past_due = 0

    lgd = float(np.clip(
        0.60
        + 0.05 * (1 if req.delinquencies_12m > 0 else 0)
        + 0.08 * req.revolving_utilization
        - 0.03 * (1 if req.residence_type == "Own" else 0)
        - 0.02 * (1 if req.residence_type == "Mortgage" else 0)
        + 0.03 * (1 if req.employment_status == "Unemployed" else 0),
        0.30, 0.90,
    ))
    ead = float(current_balance)

    row = pd.DataFrame([{
        "loan_id": 0,
        "age": int(req.age),
        "employment_status": str(req.employment_status),
        "tenure_months": int(req.tenure_months),
        "annual_income": float(req.annual_income),
        "residence_type": str(req.residence_type),
        "dependents": int(req.dependents),
        "credit_score": int(req.credit_score),
        "delinquencies_12m": int(req.delinquencies_12m),
        "inquiries_6m": int(req.inquiries_6m),
        "revolving_utilization": float(req.revolving_utilization),
        "total_open_accounts": int(req.total_open_accounts),
        "months_since_last_delinquency": int(req.months_since_last_delinquency),
        "loan_amount": float(req.loan_amount),
        "term_months": int(req.term_months),
        "interest_rate": float(req.interest_rate),
        "installment_amount": float(installment_amount),
        "purpose": str(req.purpose),
        "months_on_book": int(months_on_book),
        "current_balance": float(current_balance),
        "missed_payments_3m": int(missed_payments_3m),
        "days_past_due": int(days_past_due),
        "payment_to_income_ratio": float(payment_to_income_ratio),
        "unemployment_rate": float(req.unemployment_rate),
        "inflation_rate": float(req.inflation_rate),
        "cash_rate_proxy": float(req.cash_rate_proxy),
        "region": str(req.region),
        "region_risk_index": float(region_risk_index),
        "lgd": lgd,
        "ead": ead,
    }])

    pd_hat = float(model.predict_proba(row)[:, 1][0])
    ecl = float(pd_hat * lgd * ead)

    interest_income = calc_interest_income(req.loan_amount, req.interest_rate, req.term_months)
    operating_cost = 150.0
    expected_profit = float(interest_income - ecl - operating_cost)

    decision = decision_bucket(pd_hat)
    req_rate = calc_required_rate(req.loan_amount, req.term_months, ecl, operating_cost=operating_cost)
    recommended_rate = float(max(req.interest_rate, req_rate)) if decision == "APPROVE_REPRICE" else float(req.interest_rate)

    return {
        "pd": pd_hat,
        "ecl": ecl,
        "expected_profit": expected_profit,
        "decision": decision,
        "recommended_rate": recommended_rate,
        "installment_amount": installment_amount,
        "payment_to_income_ratio": payment_to_income_ratio,
        "lgd": lgd,
        "ead": ead,
        "reason_codes": _reason_codes(
            req.credit_score, payment_to_income_ratio, req.revolving_utilization,
            req.delinquencies_12m, req.inquiries_6m, req.employment_status,
            req.unemployment_rate, req.inflation_rate,
        ),
        "next_actions": _next_actions(decision),
    }


def portfolio() -> dict:
    fd = _decisions()
    if fd is None:
        sample = json.loads(_data().head(30).to_json(orient="records"))
        return {"available": False, "sample": sample}

    mix = fd["decision"].value_counts().rename_axis("decision").reset_index(name="count")
    mix["share"] = mix["count"] / mix["count"].sum()
    return {
        "available": True,
        "loans": int(len(fd)),
        "avg_pd": float(fd["pd_model"].mean()),
        "total_ecl": float(fd["ecl_model"].sum()),
        "decline_rate": float(fd["decision"].eq("DECLINE").mean()),
        "decision_mix": json.loads(mix.to_json(orient="records")),
        "sample": json.loads(fd.head(30).to_json(orient="records")),
    }


def ask(question: str, k: int = 5) -> str:
    return rag_answer(FIN_VECTOR_DIR, FIN_SYSTEM_PROMPT, question, k)
