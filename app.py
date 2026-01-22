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

    return True

@st.cache_resource
def load_healthcare_artifacts():
    models = joblib.load(HEALTH_MODELS_PATH)
    with open(HEALTH_META_PATH, "r") as f:
        meta = json.load(f)
    return models, meta


# ============================================================
# Healthcare — RAG Setup (NEW)
# ============================================================
HEALTH_RAG_DB_DIR = BASE_DIR / "healthcare_rag_artifacts" / "vector_store"

@st.cache_resource
def load_health_vectordb(persist_dir: Path):
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)

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
# Retail — RAG Setup (NEW)
# ============================================================
RETAIL_RAG_DB_DIR = BASE_DIR / "retail_rag_artifacts" / "vector_store"

@st.cache_resource
def load_retail_vectordb(persist_dir: Path):
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)

RETAIL_SYSTEM_PROMPT = """
You are a retail inventory decision-support assistant.

Rules (must follow):
- Use ONLY the provided CONTEXT for factual claims.
- Do NOT invent inventory policies, financial figures, supplier rules, or domain facts not present in CONTEXT.
- Treat ALL economic outputs as directional proxies unless CONTEXT explicitly states otherwise.
- If CONTEXT is insufficient, say so and ask a specific follow-up question.
- Always include citations to the retrieved sources.

Return format:
1) Answer
2) Why
3) Evidence (bullets with source + chunk)
4) Confidence (High/Medium/Low)
5) Human Review Trigger (Yes/No + reason)
"""

def retail_rag_answer(vectordb, question: str, k: int = 5) -> str:
    hits = vectordb.similarity_search(question, k=k)
    context = "\n\n".join(
        f"[SOURCE: {h.metadata.get('source')} | CHUNK: {h.metadata.get('chunk')}]\n{h.page_content}"
        for h in hits
    )
    prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"
    llm = load_llm()
    return llm.invoke([
        {"role": "system", "content": RETAIL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]).content


# ============================================================
# Healthcare — Scenario → Feature Builder (Streamlit-ready)
# ============================================================
ARRIVAL_PRESSURE_MAP = {"Low": 0.75, "Normal": 1.00, "High": 1.35}
STAFF_LEVEL_MAP      = {"Low": 0.75, "Normal": 1.00, "High": 1.25}

def clamp(x, lo, hi):
    return float(max(lo, min(hi, x)))

def build_health_feature_row(meta, capacity_beds, current_occupancy_pct, arrivals_pressure, staffing_level, current_wait_minutes, dt):
    FEATURES = meta["features"]

    occ_ratio = clamp(current_occupancy_pct / 100.0, 0.05, 1.08)

    # heuristic baseline arrivals scales with bed capacity (demo-friendly, consistent)
    base_arrivals = 6 + (capacity_beds / 700) * 19  # ~6..25
    arrivals_now = base_arrivals * ARRIVAL_PRESSURE_MAP[arrivals_pressure]

    staff_now = clamp(STAFF_LEVEL_MAP[staffing_level], 0.4, 1.6)
    wait_now  = clamp(current_wait_minutes, 5, 360)

    hour = int(dt.hour)
    dayofweek = int(dt.dayofweek)
    month = int(dt.month)
    is_weekend = 1.0 if dayofweek >= 5 else 0.0

    # conservative “last 24h” proxies for a demo (keeps UI simple)
    arrivals_roll_mean_24h = arrivals_now * clamp(0.95 + 0.05 * (1 if 12 <= hour <= 20 else -1), 0.85, 1.05)
    arrivals_roll_max_24h  = arrivals_now * (1.15 if arrivals_pressure != "Low" else 1.05)

    occ_roll_mean_24h = clamp(occ_ratio * (0.98 if is_weekend else 1.00), 0.05, 1.08)
    occ_roll_max_24h  = clamp(occ_ratio + (0.06 if arrivals_pressure == "High" else 0.03), 0.05, 1.08)

    wait_roll_mean_24h = wait_now * (1.05 if arrivals_pressure == "High" else 1.00)
    wait_roll_max_24h  = clamp(wait_now * (1.35 if arrivals_pressure == "High" else 1.15), 5, 360)

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

    X = pd.DataFrame([row])[FEATURES]
    return X

def predict_healthcare(models, X):
    rf_occ = models["rf_occ"]
    rf_wait = models["rf_wait"]

    pred_max_occ_ratio_24h = float(rf_occ.predict(X)[0])
    pred_mean_wait_24h = float(rf_wait.predict(X)[0])

    # screening-level risk rule
    risk_high = int((pred_max_occ_ratio_24h >= 0.95) or (pred_mean_wait_24h >= 120))
    return pred_max_occ_ratio_24h, pred_mean_wait_24h, risk_high


# ============================================================
# Retail — Artifacts + Helpers (NEW TAB)
# ============================================================
RETAIL_DIR = BASE_DIR / "retail_project_artifacts"

RETAIL_REQUIRED_FILES = [
    "models.joblib",
    "model_meta.json",
    "products.csv",
    "stores.csv",
    "suppliers.csv",
    "daily_sales.csv",
    "inventory_positions.csv",
]

@st.cache_resource
def load_retail_model():
    models_path = RETAIL_DIR / "models.joblib"
    meta_path = RETAIL_DIR / "model_meta.json"
    models = joblib.load(models_path)
    meta = json.loads(meta_path.read_text())
    # saved as {"demand_model": pipeline}
    return models["demand_model"], meta

@st.cache_data
def load_retail_data():
    products = pd.read_csv(RETAIL_DIR / "products.csv")
    stores = pd.read_csv(RETAIL_DIR / "stores.csv")
    suppliers = pd.read_csv(RETAIL_DIR / "suppliers.csv")
    sales = pd.read_csv(RETAIL_DIR / "daily_sales.csv", parse_dates=["date"])
    inv = pd.read_csv(RETAIL_DIR / "inventory_positions.csv", parse_dates=["date"])
    return products, stores, suppliers, sales, inv

def retail_calendar(dates):
    cal = pd.DataFrame({"date": pd.to_datetime(dates)})
    cal["dow"] = cal["date"].dt.dayofweek
    cal["is_weekend"] = (cal["dow"] >= 5).astype(int)
    cal["month"] = cal["date"].dt.month
    cal["is_peak_season"] = cal["month"].isin([11, 12]).astype(int)
    cal["is_summer"] = cal["month"].isin([12, 1, 2]).astype(int)
    return cal

def build_retail_feature_frame(products, suppliers, sales, inv):
    # normalize column names
    for d in (products, suppliers, sales, inv):
        d.columns = [c.strip().lower() for c in d.columns]

    # normalize keys (prevents join drops)
    for d in (products, suppliers, sales, inv):
        if "sku_id" in d.columns:
            d["sku_id"] = d["sku_id"].astype(str).str.strip()
        if "store_id" in d.columns:
            d["store_id"] = d["store_id"].astype(str).str.strip()

    df = sales.merge(inv, on=["date", "sku_id", "store_id"], how="left")
    df = df.merge(
        products[[
            "sku_id",
            "category",
            "base_price",
            "unit_cost",
            "gross_margin_pct",
            "annual_holding_cost_pct",
            "stockout_penalty_factor"
        ]],
        on="sku_id",
        how="left"
    )
    df = df.merge(
        suppliers[["sku_id", "lead_time_mean_days", "lead_time_sd_days"]],
        on="sku_id",
        how="left"
    )

    dates = pd.DatetimeIndex(df["date"].unique()).sort_values()
    df = df.merge(retail_calendar(dates), on="date", how="left")

    df = df.sort_values(["sku_id", "store_id", "date"])

    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df.groupby(["sku_id", "store_id"])["units_sold"].shift(lag)

    g = df.groupby(["sku_id", "store_id"])["units_sold"]
    df["roll7_mean"]  = g.transform(lambda s: s.shift(1).rolling(7).mean())
    df["roll14_mean"] = g.transform(lambda s: s.shift(1).rolling(14).mean())
    df["roll14_std"]  = g.transform(lambda s: s.shift(1).rolling(14).std())

    return df

def safety_stock(z, demand_std, lead_time_days):
    return float(z) * float(demand_std) * float(np.sqrt(max(1.0, float(lead_time_days))))

def reorder_point(mean_demand, lead_time_days, safety_stock_units):
    return float(mean_demand) * float(lead_time_days) + float(safety_stock_units)

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

tab_esg, tab_health, tab_retail = st.tabs([
    "🌱 ESG Energy & Emissions Optimization Agent",
    "🏥 Healthcare Patient-Flow Optimization Agent",
    "🛒 Retail Inventory Optimization Agent",
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
        floor_area_m2 = st.number_input("Gross Floor area (m²)", 100.0, 200000.0, 5000.0, step=100.0)
        building_age_years = st.slider("Building Age (years since construction)", 0, 100, 25)

    with col2:
        hvac_efficiency_score = st.slider("HVAC Efficiency (relative score)", 0.4, 1.0, 0.7, 0.01)
        insulation_quality_score = st.slider("Envelope/Insulation quality (relative score)", 0.3, 1.0, 0.6, 0.01)

    with col3:
        occupancy_rate = st.slider("Average Occupancy Utilisation %", 0.5, 1.0, 0.85, 0.01)
        baseline_energy_kwh = st.number_input("Baseline annual energy consumption (kWh)", 10000.0, 50_000_000.0, 750000.0, step=10000.0)

    inputs = {
        "floor_area_m2": float(floor_area_m2),
        "building_age_years": float(building_age_years),
        "hvac_efficiency_score": float(hvac_efficiency_score),
        "insulation_quality_score": float(insulation_quality_score),
        "occupancy_rate": float(occupancy_rate),
        "baseline_energy_kwh": float(baseline_energy_kwh),
    }

    st.subheader("2) Estimated Post-Intervention Impact")
    st.caption("Estimated outcomes assuming representative retrofit interventions applied to similar buildings.")

    try:
        savings_pct = esg_predictor.predict(inputs)
        st.success(f"Predicted energy savings: **{savings_pct:.2%}**")

        energy_saved_kwh = baseline_energy_kwh * savings_pct
        cost_saved = energy_saved_kwh * 0.25
        tco2_saved = (energy_saved_kwh * 0.7) / 1000

        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated annual energy reduction (kWh/year)", f"{energy_saved_kwh:,.0f}")
        c2.metric("Indicative operating cost reduction (AUD/year)", f"${cost_saved:,.0f}")
        c3.metric("Estimated Scope 2 emissions reduction (tCO₂e/yr)", f"{tco2_saved:,.1f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

    st.divider()

    st.header("🧠 Decision Rationale & Evidence Assistant")
    st.caption("Evidence-grounded explanations, limitations, and governance context. Ask anything!")

    ESG_RAG_READY = bool(os.environ.get("OPENAI_API_KEY")) and ESG_RAG_DB_DIR.exists()
    if not ESG_RAG_READY:
        st.warning(
            "RAG is not ready.\n\n"
            "Fix checklist:\n"
            "1) Add OPENAI_API_KEY in Streamlit Secrets\n"
            "2) Ensure rag_project_artifacts/vector_store exists in this repo"
        )
    else:
        vectordb = load_esg_vectordb(ESG_RAG_DB_DIR)
        q = st.text_input("Ask a question", key="esg_q", placeholder="e.g., Does this tool give NABERS ratings?")
        ask_btn = st.button("Ask", key="esg_ask")

        if ask_btn and q.strip():
            with st.spinner("Retrieving evidence and generating answer..."):
                ans = esg_rag_answer(vectordb, q.strip())
            st.markdown("### Response")
            st.write(ans)

        with st.expander("Suggested demo questions"):
            st.write(
                "1) What decisions does this tool support?\n"
                "2) Does this tool give NABERS ratings?\n"
                "3) How do we avoid greenwashing when communicating results?\n"
                "4) What are the model limitations?\n"
                "5) When should human expert review be triggered?\n"
            )

    st.caption("This demo uses synthetic and simulation-informed data to demonstrate decision-support workflows. Outputs are directional and intended for prioritisation, not certification.")

    st.markdown("---")
    st.markdown("## 📄 Full Consulting Report - Decision-Support Analysis for Retrofit Prioritisation ")
    esg_pdf_file = "ESG_Energy_and_Emissions_Optimization_Agent.pdf"
    try:
        with open(esg_pdf_file, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label="⬇️ Download Report (PDF)",
            data=pdf_bytes,
            file_name=esg_pdf_file,
            mime="application/pdf",
        )
    except FileNotFoundError:
        st.info(f"PDF not found: {esg_pdf_file}\n\nUpload it to the repo root (same folder as app.py).")


# ============================================================
# TAB 2 — Healthcare (NOW with RAG)
# ============================================================
with tab_health:
    st.title("Healthcare Patient-Flow Optimization Agent")
    st.subheader("AI Decision-Support Tool for Next-24h Congestion Risk Screening")

    st.caption(
        "This tool forecasts near-term operational pressure (next 24 hours) from the hospital’s current state. "
        "It is designed for screening and prioritization (early warning), not automated staffing or clinical decisions."
    )

    # Download artifacts from GitHub Release
    try:
        if "PASTE_" in HEALTH_MODELS_URL or "PASTE_" in HEALTH_META_URL:
            st.error("Healthcare Release URLs are not set. Paste the two GitHub Release asset links into HEALTH_MODELS_URL and HEALTH_META_URL at the top of app.py.")
            st.stop()
        ensure_healthcare_artifacts()
    except Exception as e:
        st.error(f"Failed to download Healthcare model artifacts: {e}")
        st.stop()

    # Load artifacts
    try:
        models, meta = load_healthcare_artifacts()
    except Exception as e:
        st.error(f"Failed to load Healthcare model artifacts after download: {e}")
        st.stop()

    st.subheader("1) Current Operational State (Inputs)")
    st.caption("Set a high-level snapshot of current conditions. The app constructs a valid feature vector behind the scenes.")

    c1, c2 = st.columns(2)

    with c1:
        capacity_beds = st.slider("Hospital bed capacity", 150, 700, 450, step=10)
        current_occupancy_pct = st.slider("Current occupancy (%)", 10, 108, 85, step=1)
        arrivals_pressure = st.selectbox("Recent arrivals pressure", ["Low", "Normal", "High"], index=1)

    with c2:
        staffing_level = st.selectbox("Staffing level (relative)", ["Low", "Normal", "High"], index=1)
        current_wait_minutes = st.slider("Current average wait time (minutes)", 5, 240, 60, step=5)
        date = st.date_input("Date", key="health_date")

    hour = st.slider("Hour of day", 0, 23, 18, step=1, key="health_hour")
    dt = pd.Timestamp(date) + pd.Timedelta(hours=hour)

    st.subheader("2) Next-24h Forecast (Outputs)")
    st.caption("Screening-level signals to support earlier operational attention and prioritization.")

    if st.button("Run 24-hour forecast", key="health_run"):
        X = build_health_feature_row(
            meta=meta,
            capacity_beds=capacity_beds,
            current_occupancy_pct=current_occupancy_pct,
            arrivals_pressure=arrivals_pressure,
            staffing_level=staffing_level,
            current_wait_minutes=current_wait_minutes,
            dt=dt
        )

        pred_max_occ, pred_mean_wait, risk_high = predict_healthcare(models, X)

        k1, k2, k3 = st.columns(3)
        k1.metric("Predicted max occupancy ratio (next 24h)", f"{pred_max_occ:.2f}")
        k2.metric("Predicted mean wait (minutes, next 24h)", f"{pred_mean_wait:.0f}")
        k3.metric("Risk flag (screening)", "HIGH" if risk_high else "LOW")

        if risk_high:
            st.error("High congestion risk (screening signal). Use as an early-warning prompt for human review, not as an automated instruction.")
        else:
            st.success("Lower congestion risk (screening signal). Continue monitoring and reassess if conditions change.")

        with st.expander("Show constructed model inputs (feature row)"):
            st.dataframe(X)

    st.divider()

    # -----------------------------
    # Healthcare RAG Assistant (NEW)
    # -----------------------------
    st.header("🧠 Decision Rationale & Evidence Assistant")
    st.caption("Evidence-grounded explanations for how to interpret outputs, limitations, governance triggers, and safe use.")

    HEALTH_RAG_READY = bool(os.environ.get("OPENAI_API_KEY")) and HEALTH_RAG_DB_DIR.exists()
    if not HEALTH_RAG_READY:
        st.warning(
            "Healthcare RAG is not ready.\n\n"
            "Fix checklist:\n"
            "1) Add OPENAI_API_KEY in Streamlit Secrets\n"
            "2) Ensure healthcare_rag_artifacts/vector_store exists in this repo"
        )
    else:
        health_vectordb = load_health_vectordb(HEALTH_RAG_DB_DIR)
        qh = st.text_input("Ask a question", key="health_q", placeholder="e.g., What does HIGH risk mean and what should I do next?")
        askh = st.button("Ask", key="health_ask")

        if askh and qh.strip():
            with st.spinner("Retrieving evidence and generating answer..."):
                ans = health_rag_answer(health_vectordb, qh.strip())
            st.markdown("### Response")
            st.write(ans)

        with st.expander("Suggested demo questions"):
            st.write(
                "1) What does HIGH risk mean in this demo?\n"
                "2) What are the model limitations and what can it NOT be used for?\n"
                "3) What data does the model learn from and what is the time split logic?\n"
                "4) When should a human review be triggered and why?\n"
                "5) How should an operations lead communicate outputs responsibly?\n"
            )

    st.markdown("---")
    st.markdown("## 📄 Full Consulting Report - Decision-Support Analysis for Early Operational Screening ")
    health_pdf_file = "Healthcare_Patient_Flow_Optimization_Agent.pdf"
    try:
        with open(health_pdf_file, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label="⬇️ Download Report (PDF)",
            data=pdf_bytes,
            file_name=health_pdf_file,
            mime="application/pdf",
        )
    except FileNotFoundError:
        st.info(
            f"PDF not found: {health_pdf_file}\n\n"
            "When you generate it at the end, upload it to the repo root (same folder as app.py)."
        )

    st.caption(
        "Governance note: This is operational decision support for screening/prioritization. "
        "It does not provide medical advice, staffing schedules, or clinical decisions."
    )


# ============================================================
# TAB 3 — Retail Inventory Optimization (NEW)
# ============================================================
with tab_retail:
    st.title("Retail Inventory Optimization Agent")
    st.subheader("AI Decision-Support Tool for Weekly Replenishment & SKU Prioritisation")

    st.caption(
        "This tool forecasts next-day demand and translates it into reorder policies (safety stock, reorder point, order qty) "
        "with a simple economic trade-off view (working capital vs holding cost vs stockout penalty proxy). "
        "It is decision support, not automated purchasing."
    )

    # ---- Artifact presence check (friendly error)
    missing_files = [f for f in RETAIL_REQUIRED_FILES if not (RETAIL_DIR / f).exists()]
    if missing_files:
        st.error(
            "Retail artifacts not found in repo.\n\n"
            "Create folder: retail_project_artifacts/ and upload:\n"
            + "\n".join([f"- {x}" for x in missing_files])
        )
        st.stop()

    # ---- Load
    try:
        demand_model, retail_meta = load_retail_model()
        products, stores, suppliers, sales, inv = load_retail_data()
    except Exception as e:
        st.error(f"Failed to load Retail artifacts: {e}")
        st.stop()

    # ---- Build features
    df = build_retail_feature_frame(products, suppliers, sales, inv)

    # ---- Controls
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    default_date = (df["date"].max() - pd.Timedelta(days=7)).date()

    c1, c2, c3 = st.columns(3)
    with c1:
        decision_date = st.date_input("Decision date", value=default_date, min_value=min_date, max_value=max_date)
    with c2:
        service_level = st.slider("Service level target", 0.80, 0.99, 0.95, 0.01)
    with c3:
        lead_time_shock = st.slider("Lead time shock (+ days)", 0, 21, 0, 1)

    snap = df[df["date"].dt.date == decision_date].copy()
    if snap.empty:
        st.warning("No data for the selected decision date.")
        st.stop()

    # ---- Predict demand
    # ---- Robust feature alignment (fixes Category vs category + casing drift)
    snap.columns = [c.strip().lower() for c in snap.columns]

    # --- Fix: resolve category suffixes from merges (category_x/category_y -> category)
    snap.columns = [c.strip().lower() for c in snap.columns]

    if "category" not in snap.columns:
        if "category_x" in snap.columns and "category_y" in snap.columns:
            snap["category"] = snap["category_x"].fillna(snap["category_y"])
        elif "category_x" in snap.columns:
            snap["category"] = snap["category_x"]
        elif "category_y" in snap.columns:
            snap["category"] = snap["category_y"]

    # Optional: drop suffix columns to avoid confusion later
    for c in ["category_x", "category_y"]:
        if c in snap.columns:
            snap.drop(columns=[c], inplace=True)


    X_cols_num = [c.strip().lower() for c in retail_meta["features_num"]]
    X_cols_cat = [c.strip().lower() for c in retail_meta["features_cat"]]

    # If meta mistakenly stored "category" as "Category" earlier, this forces alignment
    if "category" not in snap.columns and "category" in X_cols_cat:
    # (Nothing to do: snap already lowercased; this is here for clarity)
        pass

    # Fail-fast with a readable message if anything still missing
    needed = X_cols_num + X_cols_cat
    missing = [c for c in needed if c not in snap.columns]
    if missing:
        st.error("Retail model features missing from dataset (schema mismatch).")
        st.write("Missing columns:", missing)
        st.write("Available columns (sample):", list(snap.columns)[:40])
        st.stop()

    X_pred = snap[needed].copy()


    snap["pred_next_day_demand"] = np.clip(demand_model.predict(X_pred), 0, None)

    # ---- Inventory policy
    snap["lead_time_days"] = (snap["lead_time_mean_days"].fillna(7) + lead_time_shock).clip(lower=1)
    snap["demand_std_proxy"] = snap["roll14_std"].fillna(snap["pred_next_day_demand"].clip(lower=1))

    Z = float(norm.ppf(service_level))
    snap["safety_stock"] = snap.apply(lambda r: safety_stock(Z, r["demand_std_proxy"], r["lead_time_days"]), axis=1)
    snap["reorder_point"] = snap.apply(lambda r: reorder_point(r["pred_next_day_demand"], r["lead_time_days"], r["safety_stock"]), axis=1)

    snap["inventory_position"] = snap["on_hand"].fillna(0) + snap["on_order_qty"].fillna(0)
    snap["recommended_order_qty"] = np.ceil((snap["reorder_point"] - snap["inventory_position"]).clip(lower=0)).astype(int)

    # ---- Economics (proxy)
    snap["unit_margin"] = (snap["price"] - snap["unit_cost"]).clip(lower=0)
    snap["pred_demand_over_lt"] = snap["pred_next_day_demand"] * snap["lead_time_days"]
    snap["stockout_units_avoided_proxy"] = np.minimum(
        snap["recommended_order_qty"],
        np.ceil(snap["pred_demand_over_lt"]).astype(int)
    )

    econ = snap.apply(lambda r: economic_impact(
        order_qty=r["recommended_order_qty"],
        unit_cost=r["unit_cost"],
        annual_hold_pct=r["annual_holding_cost_pct"],
        unit_margin=r["unit_margin"],
        penalty_factor=r["stockout_penalty_factor"],
        stockout_units_avoided=r["stockout_units_avoided_proxy"]
    ), axis=1)

    econ_df = pd.DataFrame(econ.tolist(), columns=[
        "hold_cost_proxy",
        "stockout_benefit_proxy",
        "working_capital_required",
        "net_value_proxy"
    ])
    snap = pd.concat([snap.reset_index(drop=True), econ_df.reset_index(drop=True)], axis=1)

    # ---- Action list
    action = snap[snap["recommended_order_qty"] > 0].copy()
    action = action.sort_values(["net_value_proxy", "stockout_benefit_proxy"], ascending=False)

    k1, k2, k3 = st.columns(3)
    k1.metric("Recommended actions", int(action.shape[0]))
    k2.metric("Working capital required (proxy)", f"${action['working_capital_required'].sum():,.0f}")
    k3.metric("Net value (proxy)", f"${action['net_value_proxy'].sum():,.0f}")

    st.subheader("📋 Action List (ranked)")
    st.dataframe(
        action[[
            "sku_id", "store_id", "category",
            "on_hand", "on_order_qty", "inventory_position",
            "pred_next_day_demand", "lead_time_days",
            "safety_stock", "reorder_point",
            "recommended_order_qty",
            "hold_cost_proxy", "stockout_benefit_proxy",
            "working_capital_required", "net_value_proxy"
        ]],
        use_container_width=True,
        height=560
    )

    st.download_button(
        "⬇️ Download action list CSV",
        data=action.to_csv(index=False).encode("utf-8"),
        file_name=f"retail_action_list_{decision_date}.csv",
        mime="text/csv"
    )

    st.caption(
        "Governance note: Recommendations are decision support. For high-value or high-capital orders, require buyer review and supplier confirmation."
    )


    st.divider()

    # -----------------------------
    # Retail RAG Assistant (NEW)
    # -----------------------------
    st.header("🧠 Decision Rationale & Evidence Assistant")
    st.caption("Ask about rationale, assumptions, limitations, governance, and safe use. Answers are evidence-grounded from the Retail KB.")

    RETAIL_RAG_READY = bool(os.environ.get("OPENAI_API_KEY")) and RETAIL_RAG_DB_DIR.exists()

    if not RETAIL_RAG_READY:
        st.warning(
            "Retail RAG is not ready.\n\n"
            "Fix checklist:\n"
            "1) Add OPENAI_API_KEY in Streamlit Secrets\n"
            "2) Ensure retail_rag_artifacts/vector_store exists in this repo"
        )
    else:
        retail_vectordb = load_retail_vectordb(RETAIL_RAG_DB_DIR)
        qr = st.text_input(
            "Ask a question",
            key="retail_q",
            placeholder="e.g., Why is this SKU recommended? When should I escalate to buyer review?"
        )
        askr = st.button("Ask", key="retail_ask")

        if askr and qr.strip():
            with st.spinner("Retrieving evidence and generating answer..."):
                ans = retail_rag_answer(retail_vectordb, qr.strip())
            st.markdown("### Response")
            st.write(ans)

        with st.expander("Suggested demo questions"):
            st.write(
                "1) What does service level mean and how does it affect safety stock?\n"
                "2) How should I interpret net value proxy vs working capital required?\n"
                "3) When should I override a recommendation?\n"
                "4) What are the known failure modes (promos, stockouts, new SKUs)?\n"
                "5) What human review triggers should I apply for high-risk orders?\n"
            )

    st.markdown("## 📄 Full Consulting Report - Decision-Grade AI System for Weekly Replenishment & Working-Capital Prioritization")
    retail_pdf_file = "Retail_Inventory_Optimization_Agent.pdf"
    try:
        with open(retail_pdf_file, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label="⬇️ Download Report (PDF)",
            data=pdf_bytes,
            file_name=retail_pdf_file,
            mime="application/pdf",
        )
    except FileNotFoundError:
        st.info(
            f"PDF not found: {retail_pdf_file}\n\n"
            "When you generate it at the end, upload it to the repo root (same folder as app.py)."
        )

    st.caption(
        "Governance note: This is decision support for replenishment screening/prioritization. "
        "It does not automate purchasing decisions; high-capital or high-uncertainty actions require buyer review."
    )

        

