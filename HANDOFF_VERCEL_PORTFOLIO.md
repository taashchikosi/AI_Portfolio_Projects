# Handoff: Port the AI Portfolio Agents to the Vercel Portfolio Site

> **Purpose of this document.** This is a complete, self-contained spec for making the four
> AI agents in the `taashchikosi/ai_portfolio_projects` repo **viewable and usable on the live
> Vercel portfolio site** (`Taash_Chikosi_Portfolio`). Hand this file to a new chat that has
> access to the portfolio repo. It contains the architecture, exact inference contracts, the
> backend API spec, and the frontend build plan.

---

## 0. TL;DR

- **Keep Streamlit as-is.** It stays running independently. We are *not* moving or shutting it down.
- We build **two new things**:
  1. A **Python FastAPI backend** (lives in `ai_portfolio_projects`, deployed to Render) that
     serves predictions + RAG answers for all 4 agents over HTTP/JSON.
  2. A **Next.js frontend** (lives in `Taash_Chikosi_Portfolio`, deployed to Vercel) that
     replicates each agent's UI as native React and calls the backend.
- **The repos stay separate.** Frontend in the portfolio repo; backend in the projects repo.
- **RAG is included** (all four chat assistants), using **DeepSeek** for chat and **OpenAI for
  embeddings** (mandatory — see §3).

---

## 1. Architecture

```
┌─────────────────────────────┐         HTTPS / JSON        ┌──────────────────────────────────┐
│  Vercel (Next.js / React)   │  ───────────────────────▶   │  Render (Python FastAPI)         │
│  repo: Taash_Chikosi_Portfolio │                          │  repo: ai_portfolio_projects/api │
│                             │  ◀───────────────────────   │                                  │
│  - Project pages (4 agents) │                             │  - Loads .joblib models          │
│  - Input forms (native UI)  │                             │  - Loads retail CSVs             │
│  - Results cards            │                             │  - Loads 4 Chroma vector stores  │
│  - RAG chat component       │                             │  - DeepSeek (chat) + OpenAI (emb)│
└─────────────────────────────┘                             └──────────────────────────────────┘
                                                                     │
                                                              ┌──────┴───────┐
                                                              │ DeepSeek API │ (chat LLM)
                                                              │ OpenAI API   │ (embeddings only)
                                                              └──────────────┘
```

**Why this split:** Vercel cannot run heavy Python ML inference (scikit-learn/xgboost models,
Chroma vector DB) reliably within its serverless limits. A dedicated Python service handles
inference; Vercel does what it's great at — the UI.

---

## 2. Repos, boundaries & where code lives

| Concern | Repo | Notes |
|---|---|---|
| Next.js frontend | `Taash_Chikosi_Portfolio` | **The new chat builds this.** |
| FastAPI backend | `ai_portfolio_projects` (new `api/` dir) | Built where the models/data/vector stores already live. |
| ML models, CSVs, vector stores | `ai_portfolio_projects` (already present) | Do **not** copy these into the portfolio repo. |
| Streamlit `app.py` | `ai_portfolio_projects` (unchanged) | Source of truth for all business logic — port from it. |

> **Reusable source logic** to lift from `app.py` and `src/ml/predict.py` is referenced
> by function name throughout §5. The Streamlit app is the canonical implementation — the
> FastAPI handlers should reproduce its math exactly.

---

## 3. Shared RAG / LLM / Embeddings infrastructure (read first)

All four agents share the same RAG pattern. Port this once as a backend utility.

- **Chat LLM — DeepSeek (OpenAI-compatible):**
  ```python
  from langchain_openai import ChatOpenAI
  llm = ChatOpenAI(
      model="deepseek-chat",
      temperature=0,
      base_url="https://api.deepseek.com",
      api_key=os.environ["DEEPSEEK_API_KEY"],
  )
  ```
- **Embeddings — OpenAI (DO NOT CHANGE):**
  ```python
  from langchain_openai import OpenAIEmbeddings
  embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY
  ```
  > ⚠️ The four Chroma vector stores were **built with OpenAI embeddings**. They must be
  > **queried** with OpenAI embeddings too. Swapping to DeepSeek/HuggingFace embeddings would
  > require rebuilding all four stores. Keep `OPENAI_API_KEY` set.
- **Vector store load (per agent):**
  ```python
  from langchain_community.vectorstores import Chroma
  vectordb = Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embeddings)
  # collection name is "langchain" (default)
  ```
- **Answer pattern (shared):** `similarity_search(question, k)` → build a context string of
  `[SOURCE: <source> | CHUNK: <n>]\n<page_content>` blocks → send `system` + `user` messages
  to the LLM → return `.content`. See `esg_rag_answer` / `health_rag_answer` /
  `retail_rag_answer` / `fin_rag_answer` in `app.py`. Each agent has its own **system prompt**
  (`ESG_SYSTEM_PROMPT`, `HEALTH_SYSTEM_PROMPT`, `RETAIL_SYSTEM_PROMPT`, `FIN_SYSTEM_PROMPT`) —
  copy them verbatim.

**Vector stores (paths + sizes + doc counts):**

| Agent | Persist dir | Docs | Size |
|---|---|---|---|
| ESG | `rag_project_artifacts/vector_store` | 41 | 1.5M |
| Healthcare | `healthcare_rag_artifacts/vector_store` | 9 | 964K |
| Retail | `retail_rag_artifacts/vector_store` | 24 | 1.1M |
| Financial | `financial_risk_agent/rag_artifacts/vector_store` | 20 | 1.1M |

---

## 4. Critical dependency pinning (backend `requirements.txt`)

> ⚠️ **scikit-learn version must match the one the models were pickled with**, or
> `joblib.load` may warn or fail. The repo pins `scikit-learn==1.6.1` and `joblib==1.4.2` —
> the backend **must** use the same.

```
fastapi>=0.111
uvicorn[standard]>=0.30
pydantic>=2.8

numpy>=1.26
pandas>=2.2
scipy>=1.11
scikit-learn==1.6.1      # MUST match the pickling version
xgboost>=2.1
joblib==1.4.2

langchain>=0.2.12
langchain-openai>=0.1.20
langchain-community
chromadb>=0.5.5
tiktoken
openai

# Prevents a protobuf 3.x vs 4.x/5.x crash in chromadb's opentelemetry chain
protobuf>=4.21.0
opentelemetry-exporter-otlp-proto-grpc>=1.24.0
```

---

## 5. The four agents — exact inference contracts

For each agent: model files, the input the API accepts, the transform to apply, and the output.
**Port the math from `app.py` exactly** — the line references are approximate.

### 5.1 ESG — Energy & Emissions Optimization

- **Model files:** `models/energy_savings_rf.joblib` (17M) + `models/features.csv`.
  Wrapper already exists: `src/ml/predict.py::EnergySavingsPredictor` (load it directly).
- **Inputs (6 floats):** `floor_area_m2`, `building_age_years`, `hvac_efficiency_score`,
  `insulation_quality_score`, `occupancy_rate`, `baseline_energy_kwh`.
- **Predict:** `EnergySavingsPredictor().predict(inputs)` → `savings_pct` (e.g. `0.26`).
- **Derived outputs (see `app.py` "TAB 1"):**
  - `energy_saved_kwh = baseline_energy_kwh * savings_pct`
  - `cost_saved = energy_saved_kwh * 0.25`  (AUD/yr)
  - `tco2_saved = (energy_saved_kwh * 0.7) / 1000`  (tCO₂e/yr)
- **RAG:** store `rag_project_artifacts/vector_store`, prompt `ESG_SYSTEM_PROMPT`.

### 5.2 Healthcare — Patient-Flow Optimization

- **Model files:** `healthcare_project_artifacts/models.joblib` (19M; dict with keys
  `rf_occ`, `rf_wait`) + `model_meta.json` (15-feature contract).
  > Note: `app.py` *can* download these from a GitHub Release, but the files are present in
  > the repo. The backend should just load the **local** files — ignore the download logic.
- **UI inputs:** `capacity_beds` (150–700), `current_occupancy_pct` (10–108),
  `arrivals_pressure` ∈ {Low, Normal, High}, `staffing_level` ∈ {Low, Normal, High},
  `current_wait_minutes` (5–240), `date`, `hour` (0–23).
- **Transform:** port `build_health_feature_row(meta, ...)` from `app.py` **exactly** — it maps
  the friendly inputs into the 15-feature vector using `ARRIVAL_PRESSURE_MAP`,
  `STAFF_LEVEL_MAP`, `clamp`, and rolling-window heuristics. Feature order comes from
  `meta["features"]`:
  `capacity_beds, occupancy_ratio, arrivals, staff_index, wait_minutes, hour, dayofweek, month,
  is_weekend, arrivals_roll_mean_24h, arrivals_roll_max_24h, occ_roll_mean_24h, occ_roll_max_24h,
  wait_roll_mean_24h, wait_roll_max_24h`.
- **Predict:** `predict_healthcare(models, X)`:
  - `pred_max_occ_ratio_24h = rf_occ.predict(X)[0]`
  - `pred_mean_wait_24h     = rf_wait.predict(X)[0]`
  - `risk_high = (pred_max_occ_ratio_24h >= 0.95) or (pred_mean_wait_24h >= 120)`
- **RAG:** store `healthcare_rag_artifacts/vector_store`, prompt `HEALTH_SYSTEM_PROMPT`.

### 5.3 Retail — Inventory Optimization (data-heavy)

- **Model:** `retail_project_artifacts/models.joblib` (dict key `demand_model`,
  a `HistGradientBoostingRegressor`) + `model_meta.json` (`features_num` [19], `features_cat`
  = `["category"]`).
- **Data (5 CSVs, ~9.5M total):** `products.csv`, `stores.csv`, `suppliers.csv`,
  `daily_sales.csv` (6.7M), `inventory_positions.csv` (2.7M).
- **Transform:** port `build_retail_feature_frame(...)` from `app.py`:
  normalize column names/keys → merge sales+inv+products+suppliers → add calendar features
  (`retail_calendar`) → sort → lags `[1,7,14,28]` → rolling `roll7_mean, roll14_mean, roll14_std`.
  > ⚠️ **Merge collision gotcha:** the product/sales merge produces `category_x`/`category_y`.
  > `app.py` resolves these back to `category` (coalesce, then drop the suffixed columns) before
  > selecting model features. **Replicate this exactly** or prediction will fail with a missing
  > `category` column.
- **UI inputs:** `decision_date` (must fall within the data's date range), `service_level`
  (0.80–0.99), `lead_time_shock` (0–21 days).
- **Compute (per snapshot row of `decision_date`):**
  - `pred_next_day_demand = clip(demand_model.predict(X), 0, None)`
  - `Z = scipy.stats.norm.ppf(service_level)`
  - `safety_stock`, `reorder_point`, `economic_impact` — port the helper functions verbatim.
  - Output a **ranked action list** (rows where `recommended_order_qty > 0`, sorted by
    `net_value_proxy`, `stockout_benefit_proxy`) + 3 KPIs (count, working capital, net value).
  - Row fields: `sku_id, store_id, category, on_hand, on_order_qty, inventory_position,
    pred_next_day_demand, lead_time_days, safety_stock, reorder_point, recommended_order_qty,
    hold_cost_proxy, stockout_benefit_proxy, working_capital_required, net_value_proxy`.
- **Performance:** building the full feature frame from CSVs on every request is expensive.
  **Build it once at startup and cache it**, then filter by `decision_date` per request.
- **RAG:** store `retail_rag_artifacts/vector_store`, prompt `RETAIL_SYSTEM_PROMPT`.

### 5.4 Financial — Credit Default Risk

- **Model:** `financial_risk_agent/models/pd_model_CHAMPION.joblib` (22K,
  `CalibratedClassifierCV`, **30 features**).
- **Data:** `financial_risk_agent/data/consumer_loans_synthetic_v1.csv` (30k rows, 7.7M) — used
  for input default values (medians/modes) and dropdown options.
  Optional: `loan_decisions_with_reasons_actions_v3.csv` (16M) — for the Portfolio Overview.
- **30 model feature columns (exact order):**
  `loan_id, age, employment_status, tenure_months, annual_income, residence_type, dependents,
  credit_score, delinquencies_12m, inquiries_6m, revolving_utilization, total_open_accounts,
  months_since_last_delinquency, loan_amount, term_months, interest_rate, installment_amount,
  purpose, months_on_book, current_balance, missed_payments_3m, days_past_due,
  payment_to_income_ratio, unemployment_rate, inflation_rate, cash_rate_proxy, region,
  region_risk_index, lgd, ead`.
- **Inputs collected from user (~24)** + **derived server-side** (port from `app.py` "TAB 4"):
  - `installment_amount` from amortization formula; `payment_to_income_ratio`;
    `lgd` (clipped heuristic); `ead = current_balance`; `region_risk_index` from `region_risk_map`.
  - Placeholders: `loan_id=0, months_on_book=0, current_balance=loan_amount,
    missed_payments_3m=0, days_past_due=0`.
- **Compute:**
  - `pd_hat = model.predict_proba(row)[:,1][0]`
  - `ecl = pd_hat * lgd * ead`
  - Decision bands: `<0.03 APPROVE`, `<0.08 APPROVE_REPRICE`, `<0.15 MANUAL_REVIEW`, else `DECLINE`.
  - Reason codes (rule-based w/ severity), next actions, recommended rate — port verbatim.
- **Portfolio Overview:** aggregate from `loan_decisions_*.csv` (`pd_model`, `ecl_model`,
  `decision` columns): loans, avg PD, total ECL, decline rate, decision mix.
- **RAG:** store `financial_risk_agent/rag_artifacts/vector_store`, prompt `FIN_SYSTEM_PROMPT`
  (supports a `k` parameter, 3–10).

---

## 6. Backend (FastAPI) build spec

### 6.1 Suggested structure (in `ai_portfolio_projects/api/`)
```
api/
  main.py            # FastAPI app, CORS, routers
  rag.py             # shared LLM + embeddings + vector store loaders (§3)
  agents/
    esg.py           # /esg/predict, /esg/ask
    healthcare.py    # /healthcare/predict, /healthcare/ask
    retail.py        # /retail/predict, /retail/ask  (+ startup feature-frame cache)
    financial.py     # /financial/predict, /financial/portfolio, /financial/ask
  schemas.py         # pydantic request/response models
  requirements.txt   # §4
  Dockerfile         # §6.4
```
Models/CSVs/vector stores are read from their **existing repo paths** (relative to repo root).

### 6.2 Endpoints (contract the frontend codes against)

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | – | `{status:"ok"}` |
| POST | `/esg/predict` | 6 ESG floats | `{savings_pct, energy_saved_kwh, cost_saved, tco2_saved}` |
| POST | `/esg/ask` | `{question}` | `{answer}` |
| POST | `/healthcare/predict` | 7 healthcare inputs | `{pred_max_occ_ratio_24h, pred_mean_wait_24h, risk_high, feature_row}` |
| POST | `/healthcare/ask` | `{question}` | `{answer}` |
| GET | `/retail/meta` | – | `{min_date, max_date, default_date}` |
| POST | `/retail/predict` | `{decision_date, service_level, lead_time_shock}` | `{kpis:{...}, actions:[...]}` |
| POST | `/retail/ask` | `{question}` | `{answer}` |
| GET | `/financial/defaults` | – | medians/modes + dropdown options for the form |
| POST | `/financial/predict` | ~24 financial inputs | `{pd, ecl, decision, expected_profit, recommended_rate, reason_codes:[...], next_actions:[...]}` |
| GET | `/financial/portfolio` | – | `{loans, avg_pd, total_ecl, decline_rate, decision_mix:[...], sample:[...]}` |
| POST | `/financial/ask` | `{question, k?}` | `{answer}` |

> Cache model/vector-store loads at module import (load once, reuse) — equivalent to Streamlit's
> `@st.cache_resource`. Build the retail feature frame once at startup (§5.3).

### 6.3 CORS
Allow the Vercel domain(s) + localhost for dev:
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://<your-portfolio>.vercel.app",
        "https://<custom-domain-if-any>",
        "http://localhost:3000",
    ],
    allow_methods=["*"], allow_headers=["*"],
)
```

### 6.4 Dockerfile (Render-ready)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .                       # brings in models, CSVs, vector stores
ENV PORT=8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
```

### 6.5 Hosting (Render — recommended)
- New **Web Service** → connect `ai_portfolio_projects` → Docker (or Python) environment.
- Set env vars: `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`.
- Total runtime assets ≈ **90 MB** (models + data + vector stores) — fine for Render.
- ⚠️ **Free tier sleeps after ~15 min idle** → first request cold-starts (~30–60s). For a smooth
  portfolio demo, use a paid "Starter" instance or add a keep-warm ping. The frontend should show
  a loading state and a "warming up" hint on first call.
- Alternatives: Railway, Fly.io, Hugging Face Spaces (Docker).

---

## 7. Frontend (Next.js) build spec — *this is what the new chat builds*

### 7.1 In `Taash_Chikosi_Portfolio`
- Confirm framework (Next.js App Router assumed). Add an **AI Agents** section with a landing
  grid (4 cards) + a detail page per agent.
- Suggested routes: `/agents`, `/agents/esg`, `/agents/healthcare`, `/agents/retail`,
  `/agents/financial`.
- **API client:** read base URL from `process.env.NEXT_PUBLIC_API_BASE_URL`. Centralize fetch
  calls (e.g. `lib/api.ts`) with error + loading handling.
- **Reusable components:** `<AgentForm>`, `<ResultCard>` / `<MetricGrid>`, `<DataTable>` (retail
  action list / financial portfolio), `<RagChat>` (question box → `/ask` → render markdown
  answer, with the agent's suggested demo questions).

### 7.2 Per-agent UI (mirror the Streamlit forms)
- **ESG:** 6 numeric inputs (sliders/number fields per `app.py` ranges) → 1 prediction + 3 metric
  cards → RAG chat.
- **Healthcare:** sliders/selects (capacity, occupancy %, arrivals pressure, staffing, wait,
  date, hour) → 3 metric cards + risk flag → RAG chat.
- **Retail:** fetch `/retail/meta` for the date range → date picker + service-level slider +
  lead-time-shock slider → 3 KPI cards + ranked action table (+ CSV download) → RAG chat.
- **Financial:** fetch `/financial/defaults` to prefill fields/dropdowns → decision simulator
  form → Decision Card (PD, ECL, profit, decision) + reason codes + next actions; second tab
  Portfolio Overview from `/financial/portfolio` → RAG chat (with `k` control).

### 7.3 Reports (PDFs)
The repo root has 4 consulting-report PDFs (one per agent). Copy them into the portfolio's
`public/` and add a "Download report" button on each agent page (no backend needed).

---

## 8. Environment variables

| Where | Var | Purpose |
|---|---|---|
| Render (backend) | `DEEPSEEK_API_KEY` | RAG chat LLM |
| Render (backend) | `OPENAI_API_KEY` | RAG **embeddings** (mandatory) |
| Vercel (frontend) | `NEXT_PUBLIC_API_BASE_URL` | Base URL of the Render backend |

---

## 9. Execution phases (checklist)

**Phase A — Backend (in `ai_portfolio_projects`)**
- [ ] Scaffold `api/` (§6.1) + `requirements.txt` (§4) + `Dockerfile` (§6.4).
- [ ] Port shared RAG util (§3).
- [ ] Implement the 4 agents' predict + ask endpoints, lifting math from `app.py`.
- [ ] Add CORS, `/health`, retail startup cache.
- [ ] Test locally (`uvicorn api.main:app --reload`) against each endpoint.
- [ ] Deploy to Render; set `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`; verify `/health`.

**Phase B — Frontend (in `Taash_Chikosi_Portfolio`)**
- [ ] Add API client + `NEXT_PUBLIC_API_BASE_URL`.
- [ ] Build the 4 agent pages + shared components (§7).
- [ ] Wire up RAG chat per agent.
- [ ] Add PDF report downloads.
- [ ] Set env var in Vercel; deploy; smoke-test all 4 agents end-to-end.

**Phase C — Polish**
- [ ] Loading/cold-start states, error toasts, mobile layout.
- [ ] Keep-warm strategy (or paid tier) so demos don't cold-start.

---

## 10. Gotchas & risks (read before coding)

1. **scikit-learn must be `1.6.1`** in the backend (pickle compatibility). Same for `joblib==1.4.2`.
2. **Keep OpenAI embeddings.** Vector stores are in OpenAI's embedding space (§3).
3. **Retail `category_x`/`category_y` merge collision** must be resolved exactly (§5.3).
4. **Healthcare model:** use the **local** joblib; ignore `app.py`'s GitHub-Release download path.
5. **protobuf pin** (`>=4.21.0`) is required or chromadb's opentelemetry import crashes with a
   `TypeError` (this already bit the Streamlit app).
6. **Render free-tier cold starts** (~30–60s). Plan UX around it or upgrade.
7. **Retail feature frame is expensive** — build once at startup, filter per request.
8. **CORS** must list the exact Vercel domain(s).
9. **Big optional file:** `loan_decisions_*.csv` is 16M; only needed for the financial Portfolio
   Overview. Fine to include, or lazy-load.

---

## 11. Acceptance criteria

- All 4 agents render on the live Vercel site and return correct results matching the Streamlit
  app for the same inputs.
- Each agent's RAG chat returns grounded, cited answers (DeepSeek + OpenAI embeddings).
- Backend `/health` is green; cold-start behavior is handled gracefully in the UI.
- The Streamlit app is untouched and still works.

---

## 12. Open items for the new chat to confirm

- Exact Next.js version/router and styling system in `Taash_Chikosi_Portfolio` (Tailwind? CSS
  modules? a UI kit?) — adapt components accordingly.
- Final Render service URL → set as `NEXT_PUBLIC_API_BASE_URL`.
- Whether to gate the agents behind a nav section or feature them on the homepage.
- Custom domain / analytics, if any.

---

### Appendix — source-of-truth files in `ai_portfolio_projects`
- `app.py` — all UI + business logic (canonical).
- `src/ml/predict.py` — ESG predictor wrapper.
- `models/` — ESG model + `features.csv`.
- `healthcare_project_artifacts/` — healthcare models + meta.
- `retail_project_artifacts/` — retail model, meta, 5 CSVs.
- `financial_risk_agent/` — financial model, data, RAG store.
- `*_rag_artifacts/` & `rag_project_artifacts/` — Chroma vector stores.
- `*.pdf` (repo root) — the four consulting reports.
