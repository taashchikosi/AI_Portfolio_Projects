# app.py — FULL COPY/PASTE (ESG + Healthcare + Retail tab)
# Source base: your uploaded "app ESG & Healthcare.py" :contentReference[oaicite:0]{index=0}

import os
import json
import joblib
import requests
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.ml.predict import EnergySavingsPredictor

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI Portfolio — Decision Support Agents", layout="wide")
BASE_DIR = Path(__file__).resolve().parent

# -----------------------------
# OpenAI API Key (for RAG)
# -----------------------------
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# ============================================================
# Shared LLM loader (used by both RAG assistants)
# ============================================================
@st.cache_resource
def load_llm():
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0)


# ============================================================
# ESG — RAG Setup (Chroma persist dir)
# ============================================================
ESG_RAG_DB_DIR = BASE_DIR / "rag_project_artifacts" / "vector_store"

@st.cache_resource
def load_esg_vectordb(persist_dir: Path):
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)

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

def esg_rag_answer(vectordb, question: str, k: int = 5) -> str:
    hits = vectordb.similarity_search(question, k=k)
    context = "\n\n".join(
        f"[SOURCE: {h.metadata.get('source')} | CHUNK: {h.metadata.get('chunk')}]\n{h.page_content}"
        for h in hits
    )
    prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"
    llm = load_llm()
    return llm.invoke([
        {"role": "system", "content": ESG_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]).content


# ============================================================
# ESG — ML Predictor
# ============================================================
@st.cache_resource
def load_esg_predictor():
    return EnergySavingsPredictor()

esg_predictor = load_esg_predictor()


# ============================================================
# Healthcare — Auto-download model artifacts from GitHub Release
# ============================================================
HEALTH_ARTIFACTS_DIR = BASE_DIR / "healthcare_project_artifacts"
HEALTH_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

HEALTH_MODELS_PATH = HEALTH_ARTIFACTS_DIR / "models.joblib"
HEALTH_META_PATH   = HEALTH_ARTIFACTS_DIR / "model_meta.json"

# 🔴 PASTE YOUR *DIRECT* GITHUB RELEASE ASSET LINKS HERE:
HEALTH_MODELS_URL = "https://github.com/taashchikosi/esg-energy-efficiency-agent/releases/download/healthcare-v1/models.joblib"
HEALTH_META_URL   = "https://github.com/taashchikosi/esg-energy-efficiency-agent/releases/download/healthcare-v1/model_meta.json"

@st.cache_resource
def ensure_healthcare_artifacts():
    # meta first (small)
    if not HEALTH_META_PATH.exists():
        r = requests.get(HEALTH_META_URL, timeout=60)
        r.raise_for_status()
        HEALTH_META_PATH.write_bytes(r.content)

    # model (bigger)
    if not HEALTH_MODELS_PATH.exists():
        r = requests.get(HEALTH_MODELS_URL, timeout=240)
        r.raise_for_status()
        HEALTH_MODELS_PATH.write_bytes(r.content)

    meta = json.loads(HEALTH_META_PATH.read_text())
    models = joblib.load(HEALTH_MODELS_PATH)
    return meta, models


# ============================================================
# Healthcare — RAG Setup (Chroma persist dir)
# ============================================================
HEALTH_RAG_DB_DIR = BASE_DIR / "healthcare_project_artifacts" / "vector_store"

@st.cache_resource
def load_health_vectordb(persist_dir: Path):
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)

HEALTH_SYSTEM_PROMPT = """
You are a healthcare patient-flow decision-support assistant.

Rules (must follow):
- Use ONLY the provided CONTEXT for factual claims.
- Do NOT invent facts, policies, or operational details.
- If the CONTEXT is insufficient, say so and ask a specific follow-up question.
- Always include citations to the retrieved sources.

Return format:
1) Answer
2) Why
3) Evidence (bullets with source + chunk)
4) Confidence (High/Medium/Low)
5) Human Review Trigger (Yes/No + reason)
"""

def health_rag_answer(vectordb, question: str, k: int = 5) -> str:
    hits = vectordb.similarity_search(question, k=k)
    context = "\n\n".join(
        f"[SOURCE: {h.metadata.get('source')} | CHUNK: {h.metadata.get('chunk')}]\n{h.page_content}"
        for h in hits
    )
    prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"
    llm = load_llm()
    return llm.invoke([
        {"role": "system", "content": HEALTH_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]).content


# ============================================================
# Retail — Simple Helper Functions (from your prior tab)
# ============================================================
def holding_cost_per_unit_per_day(unit_cost, annual_hold_pct):
    return float(unit_cost) * (float(annual_hold_pct) / 365.0)

def stockout_penalty_per_unit(unit_margin, penalty_factor):
    return float(unit_margin) * float(penalty_factor)

def economic_impact(order_qty, unit_cost, annual_hold_pct, unit_margin, penalty_factor, stockout_units_avoided):
    # 14-day holding proxy (simple, explainable)
    hold_cost = float(order_qty) * holding_cost_per_unit_per_day(unit_cost, annual_hold_pct) * 14.0
    stockout_benefit = float(stockout_units_avoided) * stockout_penalty_per_unit(unit_margin, penalty_factor)
    working_capital = float(order_qty) * float(unit_cost)
    net_value = stockout_benefit - hold_cost
    return hold_cost, stockout_benefit, working_capital, net_value


# ============================================================
# App Layout — Tabs
# ============================================================
st.title("AI Portfolio — Decision Support Agents")
st.caption("Select a project tab below.")

tab_esg, tab_health, tab_retail, tab_fin = st.tabs([
    "🌱 ESG Energy & Emissions Optimization Agent",
    "🏥 Healthcare Patient-Flow Optimization Agent",
    "🛒 Retail Inventory Optimization Agent",
    "💳 Financial Credit Default Risk Agent",
])


# ============================================================
# TAB 1 — ESG (unchanged)
# ============================================================
with tab_esg:
    st.title("ESG Energy & Emissions Optimization Agent")
    st.subheader("AI Decision-Support Tool for Building Retrofit Prioritization")

    st.caption(
        "This tool estimates post-intervention energy-savings, cost-savings, and emissions reduction potential "
        "by learning from patterns in retrofit outcomes of buildings with similar pre-intervention characteristics. "
        "Its main goal is to help decision-makers decide where to spend limited retrofit capital first, and how confidently they can justify that choice."
    )

    st.subheader("Examples of Decisions it Supports")
    st.caption("1. Prioritization Decisions - Which buildings have the highest savings potential?")
    st.caption("2. Capital Allocation Decisions - Where should we deploy limited budget or resources?")
    st.caption("3. Scenario Comparison Decisions - How sensitive are outcomes to changes in building characteristics?")
    st.caption("4. Governance & Denfensibility Decisions - Can we responsibly communicate this insight to executives, investors, or ESG reports?")

    st.subheader("How to use this tool - The Decision Workflow")
    st.caption("1. Input baseline characteristics -> this defines the current state of the building")
    st.caption("2. Allow the machine to predict potential ESG outcomes -> Review estimated energy, cost, & emissions impacts.")
    st.caption("3. Use the Assistant to ask questions -> Interogate the rationale, understand assumptions, limitations, & governance implications.")
    st.caption("4. Decide next action —> investigate further, prioritise or deprioritize buildings")

    st.subheader("1) Building Baseline")
    st.caption("Describe the current (pre-retrofit) state of the building. These inputs define the baseline against which potential improvements are estimated.")

    col1, col2, col3 = st.columns(3)
    with col1:
        floor_area = st.number_input("Floor Area (m²)", min_value=100, max_value=100000, value=5000, step=100)
        building_age = st.number_input("Building Age (years)", min_value=0, max_value=200, value=25, step=1)
    with col2:
        hvac_efficiency = st.slider("HVAC Efficiency (0–1)", 0.0, 1.0, 0.65, 0.01)
        occupancy_rate = st.slider("Occupancy Rate (0–1)", 0.0, 1.0, 0.80, 0.01)
    with col3:
        baseline_kwh = st.number_input("Baseline Energy Use (kWh/yr)", min_value=1000, max_value=100000000, value=1200000, step=10000)
        energy_price = st.number_input("Energy Price ($/kWh)", min_value=0.01, max_value=2.0, value=0.32, step=0.01)

    st.subheader("2) Intervention Scenario")
    st.caption("Select which upgrades you are considering. The model estimates potential savings and reductions based on patterns in similar retrofits.")

    retrofit_hvac = st.checkbox("HVAC Upgrade")
    retrofit_lighting = st.checkbox("LED Lighting Upgrade")
    retrofit_controls = st.checkbox("Smart Controls / BMS")
    retrofit_insulation = st.checkbox("Building Envelope / Insulation")

    if st.button("Run ESG Prediction", type="primary"):
        inputs = {
            "floor_area": floor_area,
            "building_age": building_age,
            "hvac_efficiency": hvac_efficiency,
            "occupancy_rate": occupancy_rate,
            "baseline_kwh": baseline_kwh,
            "energy_price": energy_price,
            "retrofit_hvac": retrofit_hvac,
            "retrofit_lighting": retrofit_lighting,
            "retrofit_controls": retrofit_controls,
            "retrofit_insulation": retrofit_insulation,
        }
        pred = esg_predictor.predict(inputs)
        st.success("Prediction complete.")
        st.json(pred)

    st.subheader("3) ESG Assistant (RAG)")
    st.caption("Ask questions about assumptions, governance, and defensibility. Answers are evidence-backed from your embedded project documents.")

    q = st.text_input("Ask the ESG Assistant:")
    if st.button("Ask ESG Assistant"):
        vectordb = load_esg_vectordb(ESG_RAG_DB_DIR)
        st.write(esg_rag_answer(vectordb, q))


# ============================================================
# TAB 2 — Healthcare (unchanged)
# ============================================================
with tab_health:
    st.title("Healthcare Patient-Flow Optimization Agent")
    st.subheader("AI Decision-Support Tool for Occupancy, Length-of-Stay, and Staffing Decisions")

    st.caption(
        "This tool predicts patient-flow bottlenecks and supports operational decisions "
        "around occupancy, length-of-stay, and staffing allocation."
    )

    meta, models = ensure_healthcare_artifacts()

    st.markdown("### Model Artifacts Loaded")
    st.json(meta)

    st.subheader("Healthcare Assistant (RAG)")
    qh = st.text_input("Ask the Healthcare Assistant:")
    if st.button("Ask Healthcare Assistant"):
        vectordb = load_health_vectordb(HEALTH_RAG_DB_DIR)
        st.write(health_rag_answer(vectordb, qh))


# ============================================================
# TAB 3 — Retail (unchanged)
# ============================================================
with tab_retail:
    st.title("Retail Inventory Optimization Agent")
    st.subheader("AI Decision-Support Tool for Reorder Quantity & Stockout Tradeoffs")

    st.caption(
        "This tool supports retail replenishment decisions by translating demand uncertainty "
        "into reorder recommendations, stockout risk, and economic trade-offs."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        forecast_demand = st.number_input("Forecast Demand (next period)", min_value=0, max_value=100000, value=5000, step=100)
        demand_std = st.number_input("Demand Std Dev", min_value=0, max_value=50000, value=800, step=50)
    with col2:
        unit_cost = st.number_input("Unit Cost ($)", min_value=0.01, max_value=10000.0, value=12.0, step=0.5)
        unit_margin = st.number_input("Unit Margin ($)", min_value=0.01, max_value=10000.0, value=6.0, step=0.5)
    with col3:
        annual_hold_pct = st.slider("Annual Holding Cost %", 0.0, 1.0, 0.25, 0.01)
        penalty_factor = st.slider("Stockout Penalty Factor", 0.0, 5.0, 2.0, 0.1)

    service_level = st.slider("Target Service Level", 0.50, 0.99, 0.95, 0.01)

    if st.button("Run Retail Recommendation", type="primary"):
        z = float(norm.ppf(service_level))
        safety_stock = z * demand_std
        reorder_qty = forecast_demand + safety_stock

        stockout_units_avoided = safety_stock  # proxy
        hold_cost, stockout_benefit, working_capital, net_value = economic_impact(
            reorder_qty, unit_cost, annual_hold_pct, unit_margin, penalty_factor, stockout_units_avoided
        )

        st.success("Recommendation complete.")
        st.write(f"**Reorder Quantity:** {reorder_qty:,.0f}")
        st.write(f"**Safety Stock:** {safety_stock:,.0f}")
        st.write(f"**Estimated Holding Cost (14-day proxy):** ${hold_cost:,.0f}")
        st.write(f"**Estimated Stockout Benefit:** ${stockout_benefit:,.0f}")
        st.write(f"**Working Capital Required:** ${working_capital:,.0f}")
        st.write(f"**Net Value (benefit - holding):** ${net_value:,.0f}")


# ============================================================
# TAB 4 — Financial (Consumer Loan Default Risk)
# ============================================================
with tab_fin:
    st.title("Financial Credit Default Risk Agent")
    st.subheader("PD → ECL → Decision + Reason Codes + Next Actions")

    st.caption(
        "This tool predicts 12-month probability of default (PD) for consumer loans, translates that risk into "
        "expected credit loss (ECL), and recommends an action (approve / reprice / manual review / decline) with "
        "bank-style reason codes and next-best actions."
    )

    FIN_BASE_DIR = BASE_DIR / "financial_risk_agent"
    FIN_MODEL_PATH = FIN_BASE_DIR / "models" / "pd_model_CHAMPION.joblib"
    FIN_DATA_PATH = FIN_BASE_DIR / "data" / "consumer_loans_synthetic_v1.csv"
    FIN_DECISIONS_PATH = FIN_BASE_DIR / "data" / "loan_decisions_with_reasons_actions_v3.csv"

    @st.cache_resource
    def load_fin_model():
        if not FIN_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing financial model at: {FIN_MODEL_PATH}. "
                "Add it to the repo under financial_risk_agent/models/."
            )
        return joblib.load(FIN_MODEL_PATH)

    @st.cache_data
    def load_fin_data():
        if not FIN_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Missing financial dataset at: {FIN_DATA_PATH}. "
                "Add it to the repo under financial_risk_agent/data/."
            )
        return pd.read_csv(FIN_DATA_PATH)

    @st.cache_data
    def load_fin_decisions():
        if FIN_DECISIONS_PATH.exists():
            return pd.read_csv(FIN_DECISIONS_PATH)
        return None

    try:
        fin_model = load_fin_model()
        fin_data = load_fin_data()
        fin_decisions = load_fin_decisions()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # --- Policy thresholds (keep consistent with the project narrative)
    PD_APPROVE_MAX = 0.03
    PD_REPRICE_MAX = 0.08
    PD_REVIEW_MAX  = 0.15

    def fin_decision_bucket(pd_val: float) -> str:
        if pd_val < PD_APPROVE_MAX:
            return "APPROVE"
        if pd_val < PD_REPRICE_MAX:
            return "APPROVE_REPRICE"
        if pd_val < PD_REVIEW_MAX:
            return "MANUAL_REVIEW"
        return "DECLINE"

    # --- Simple money layer (consistent with Colab)
    def fin_calc_interest_income(loan_amount, interest_rate, term_months, avg_balance_factor=0.5):
        term_years = float(term_months) / 12.0
        return float(loan_amount) * float(interest_rate) * term_years * avg_balance_factor

    def fin_calc_required_rate(loan_amount, term_months, ecl, operating_cost=150.0, target_profit_pct=0.02, avg_balance_factor=0.5):
        term_years = float(term_months) / 12.0
        denom = max(float(loan_amount) * term_years * avg_balance_factor, 1.0)
        target_profit = target_profit_pct * float(loan_amount)
        req = (target_profit + float(ecl) + float(operating_cost)) / denom
        return float(np.clip(req, 0.05, 0.35))

    # --- Tabs inside financial section
    fin_tab1, fin_tab2 = st.tabs(["📊 Portfolio Overview", "🧮 Decision Simulator"])

    with fin_tab1:
        st.markdown("### Portfolio Overview")
        if fin_decisions is None:
            st.warning(
                "Optional file missing: loan_decisions_with_reasons_actions_v3.csv. "
                "Upload it for full decision mix + explainability browsing."
            )
            st.dataframe(fin_data.head(30), use_container_width=True)
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Loans", f"{len(fin_decisions):,}")
            c2.metric("Avg PD", f"{fin_decisions['pd_model'].mean():.2%}")
            c3.metric("Total ECL", f"${fin_decisions['ecl_model'].sum():,.0f}")
            c4.metric("Decline Rate", f"{(fin_decisions['decision'].eq('DECLINE').mean()):.2%}")

            mix = fin_decisions["decision"].value_counts().rename_axis("decision").reset_index(name="count")
            mix["share"] = mix["count"] / mix["count"].sum()
            st.dataframe(mix, use_container_width=True)

            st.markdown("#### Sample loans (with reason codes + actions)")
            show_cols = [
                "loan_id","pd_model","ecl_model","expected_profit","decision",
                "reason_1","severity_1","reason_2","severity_2",
                "action_1","action_2"
            ]
            existing = [c for c in show_cols if c in fin_decisions.columns]
            st.dataframe(fin_decisions[existing].head(30), use_container_width=True)

    with fin_tab2:
        st.markdown("### Decision Simulator")
        st.caption("Adjust inputs and run PD → ECL → Decision, plus pricing suggestion when applicable.")

        d = fin_data
        # sensible defaults
        default_region = d["region"].mode()[0]
        default_emp = d["employment_status"].mode()[0]
        default_res = d["residence_type"].mode()[0]
        default_purpose = d["purpose"].mode()[0]
        region_risk_map = {"Inner Metro": 0.95, "Outer Metro": 1.00, "Regional": 1.08, "Remote": 1.18}

        cA, cB, cC = st.columns(3)

        with cA:
            age = st.number_input("Age", 18, 75, int(d["age"].median()))
            employment_status = st.selectbox("Employment status", sorted(d["employment_status"].unique().tolist()),
                                            index=sorted(d["employment_status"].unique().tolist()).index(default_emp))
            annual_income = st.number_input("Annual income", 18000, 250000, int(d["annual_income"].median()), step=1000)
            residence_type = st.selectbox("Residence type", sorted(d["residence_type"].unique().tolist()),
                                          index=sorted(d["residence_type"].unique().tolist()).index(default_res))
            dependents = st.number_input("Dependents", 0, 5, int(d["dependents"].median()))
            tenure_months = st.number_input("Tenure (months)", 0, 240, int(d["tenure_months"].median()))

        with cB:
            credit_score = st.number_input("Credit score", 300, 850, int(d["credit_score"].median()))
            delinquencies_12m = st.number_input("Delinquencies (12m)", 0, 6, int(d["delinquencies_12m"].median()))
            inquiries_6m = st.number_input("Inquiries (6m)", 0, 10, int(d["inquiries_6m"].median()))
            revolving_utilization = st.slider("Revolving utilization", 0.0, 1.0, float(d["revolving_utilization"].median()), 0.01)
            total_open_accounts = st.number_input("Total open accounts", 1, 25, int(d["total_open_accounts"].median()))
            months_since_last_delinquency = st.number_input("Months since last delinquency", 0, 120, int(d["months_since_last_delinquency"].median()))

        with cC:
            loan_amount = st.number_input("Loan amount", 2000, 60000, int(d["loan_amount"].median()), step=500)
            term_months = st.selectbox("Term (months)", sorted(d["term_months"].unique().tolist()),
                                       index=sorted(d["term_months"].unique().tolist()).index(int(d["term_months"].median())))
            interest_rate = st.slider("Offered interest rate", 0.05, 0.29, float(d["interest_rate"].median()), 0.005)
            purpose = st.selectbox("Loan purpose", sorted(d["purpose"].unique().tolist()),
                                   index=sorted(d["purpose"].unique().tolist()).index(default_purpose))

            # approx installment + PTI
            r_m = float(interest_rate) / 12.0
            n = int(term_months)
            installment_amount = float((loan_amount * r_m) / (1 - (1 + r_m) ** (-n))) if r_m > 0 else float(loan_amount / max(n, 1))
            monthly_income = float(annual_income) / 12.0
            payment_to_income_ratio = float(installment_amount / max(monthly_income, 1.0))

            st.write(f"Estimated installment: **${installment_amount:,.2f}** / month")
            st.write(f"Payment-to-income ratio (PTI): **{payment_to_income_ratio:.2f}**")

            unemployment_rate = st.slider("Unemployment rate", 3.0, 9.0, float(d['unemployment_rate'].median()), 0.1)
            inflation_rate = st.slider("Inflation rate", 2.0, 8.0, float(d['inflation_rate'].median()), 0.1)
            cash_rate_proxy = st.slider("Cash rate proxy", 3.0, 7.0, float(d['cash_rate_proxy'].median()), 0.1)

            region = st.selectbox("Region", sorted(d["region"].unique().tolist()),
                                  index=sorted(d["region"].unique().tolist()).index(default_region))
            region_risk_index = float(region_risk_map.get(region, 1.0))

        # account-performance placeholders (origination scenario)
        months_on_book = 0
        current_balance = float(loan_amount)
        missed_payments_3m = 0
        days_past_due = 0

        if st.button("Run Financial Decision", type="primary"):
            # -----------------------------
            # FIX: add lgd + ead as features BEFORE predict_proba()
            # -----------------------------

            # Compute LGD/EAD first (so the model receives the same columns it was trained on)
            lgd = float(np.clip(
                0.60
                + 0.05 * (1 if delinquencies_12m > 0 else 0)
                + 0.08 * revolving_utilization
                - 0.03 * (1 if residence_type == "Own" else 0)
                - 0.02 * (1 if residence_type == "Mortgage" else 0)
                + 0.03 * (1 if employment_status == "Unemployed" else 0),
                0.30, 0.90
            ))
            ead = float(current_balance)
            
            row = pd.DataFrame([{
                "loan_id": 0,
                "age": int(age),
                "employment_status": str(employment_status),
                "tenure_months": int(tenure_months),
                "annual_income": float(annual_income),
                "residence_type": str(residence_type),
                "dependents": int(dependents),
            
                "credit_score": int(credit_score),
                "delinquencies_12m": int(delinquencies_12m),
                "inquiries_6m": int(inquiries_6m),
                "revolving_utilization": float(revolving_utilization),
                "total_open_accounts": int(total_open_accounts),
                "months_since_last_delinquency": int(months_since_last_delinquency),
            
                "loan_amount": float(loan_amount),
                "term_months": int(term_months),
                "interest_rate": float(interest_rate),
                "installment_amount": float(installment_amount),
                "purpose": str(purpose),
            
                "months_on_book": int(months_on_book),
                "current_balance": float(current_balance),
                "missed_payments_3m": int(missed_payments_3m),
                "days_past_due": int(days_past_due),
                "payment_to_income_ratio": float(payment_to_income_ratio),
            
                "unemployment_rate": float(unemployment_rate),
                "inflation_rate": float(inflation_rate),
                "cash_rate_proxy": float(cash_rate_proxy),
                "region": str(region),
                "region_risk_index": float(region_risk_index),
            
                # ✅ THESE TWO MUST EXIST because the model was trained with them
                "lgd": lgd,
                "ead": ead,
            }])

            pd_hat = float(fin_model.predict_proba(row)[:, 1][0])


            # LGD approximation (same as Colab)
            lgd = float(np.clip(
                0.60
                + 0.05 * (1 if delinquencies_12m > 0 else 0)
                + 0.08 * revolving_utilization
                - 0.03 * (1 if residence_type == "Own" else 0)
                - 0.02 * (1 if residence_type == "Mortgage" else 0)
                + 0.03 * (1 if employment_status == "Unemployed" else 0),
                0.30, 0.90
            ))
            ead = float(current_balance)
            ecl = float(pd_hat * lgd * ead)

            interest_income = fin_calc_interest_income(loan_amount, interest_rate, term_months)
            operating_cost = 150.0
            expected_profit = float(interest_income - ecl - operating_cost)

            decision = fin_decision_bucket(pd_hat)
            req_rate = fin_calc_required_rate(loan_amount, term_months, ecl, operating_cost=operating_cost)
            recommended_rate = float(max(interest_rate, req_rate)) if decision == "APPROVE_REPRICE" else float(interest_rate)

            st.markdown("## Decision Card")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Predicted PD", f"{pd_hat:.2%}")
            k2.metric("ECL (PD×LGD×EAD)", f"${ecl:,.0f}")
            k3.metric("Expected Profit (approx)", f"${expected_profit:,.0f}")
            k4.metric("Decision", decision)

            if decision == "APPROVE_REPRICE":
                st.info(f"Recommended rate to meet target margin: **{recommended_rate:.2%}** (current offered: {interest_rate:.2%})")

            # Lightweight reason codes (rule-based, aligned to your Phase 6 logic)
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

            # Macro stress
            macro_score = 0
            if unemployment_rate >= 8.0: macro_score += 2
            elif unemployment_rate >= 6.5: macro_score += 1
            if inflation_rate >= 7.0: macro_score += 2
            elif inflation_rate >= 5.5: macro_score += 1
            if macro_score >= 3:
                reasons.append(("Macro environment under high stress (unemployment/inflation)", 4, f"unemp={unemployment_rate:.1f} infl={inflation_rate:.1f}"))
            elif macro_score >= 2:
                reasons.append(("Macro environment stressed (unemployment/inflation)", 3, f"unemp={unemployment_rate:.1f} infl={inflation_rate:.1f}"))

            if not reasons:
                reasons = [("No major risk flags triggered (combined moderate factors)", 1, "—")]

            reasons = sorted(reasons, key=lambda x: x[1], reverse=True)[:4]

            st.markdown("### Top Reason Codes (with severity)")
            for r, s, d_ in reasons:
                st.write(f"- **{r}** | severity **{s}/5** | _{d_}_")

            st.markdown("### Next Best Actions")
            if decision == "APPROVE":
                actions = ["Auto-approve under standard terms", "Monitor early payment performance (first 60 days)"]
            elif decision == "APPROVE_REPRICE":
                actions = [
                    "Approve with risk-based pricing (increase rate to target margin)",
                    "Offer shorter term or smaller principal to reduce PTI"
                ]
            elif decision == "MANUAL_REVIEW":
                actions = [
                    "Route to manual credit review",
                    "Request supporting documents (payslips/bank statements)",
                    "Counter-offer: reduce amount or require co-applicant/guarantor"
                ]
            else:
                actions = [
                    "Decline under automated policy",
                    "Offer alternative: secured product or smaller amount after seasoning period",
                    "Provide adverse action notice + path to eligibility"
                ]

            for i, a in enumerate(actions[:3], start=1):
                st.write(f"{i}. {a}")

    st.divider()
    st.caption(
        "Setup note: ensure you’ve committed the financial_risk_agent folder (data + models). "
        "If you want evidence-backed Q&A, we’ll add the Phase 8 RAG layer next."
    )
