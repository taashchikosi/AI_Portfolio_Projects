"""FastAPI backend serving the four AI portfolio agents (predictions + RAG).

Run locally:   uvicorn api.main:app --reload
Health check:  GET /health   (lightweight -- used by the keep-warm pinger)
"""
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .rag import keys_ready
from .schemas import (
    AskRequest,
    ESGPredictRequest,
    FinancialPredictRequest,
    HealthcarePredictRequest,
    RetailPredictRequest,
)
from .agents import esg, healthcare, retail, financial

app = FastAPI(title="AI Portfolio Agents API", version="1.0.0")

# CORS: comma-separated list in ALLOWED_ORIGINS env var (set the Vercel domain here).
_origins_env = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_rag():
    if not keys_ready():
        raise HTTPException(
            status_code=503,
            detail="RAG not configured: set DEEPSEEK_API_KEY and OPENAI_API_KEY.",
        )


# ---- health / readiness (lightweight; safe for keep-warm pings) ----
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return {"rag_keys_ready": keys_ready()}


# ---- ESG ----
@app.post("/esg/predict")
def esg_predict(req: ESGPredictRequest):
    try:
        return esg.predict(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ESG prediction failed: {e}")


@app.post("/esg/ask")
def esg_ask(req: AskRequest):
    _require_rag()
    return {"answer": esg.ask(req.question, req.k)}


# ---- Healthcare ----
@app.post("/healthcare/predict")
def healthcare_predict(req: HealthcarePredictRequest):
    try:
        return healthcare.predict(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Healthcare prediction failed: {e}")


@app.post("/healthcare/ask")
def healthcare_ask(req: AskRequest):
    _require_rag()
    return {"answer": healthcare.ask(req.question, req.k)}


# ---- Retail ----
@app.get("/retail/meta")
def retail_meta():
    return retail.meta()


@app.post("/retail/predict")
def retail_predict(req: RetailPredictRequest):
    try:
        return retail.predict(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retail prediction failed: {e}")


@app.post("/retail/ask")
def retail_ask(req: AskRequest):
    _require_rag()
    return {"answer": retail.ask(req.question, req.k)}


# ---- Financial ----
@app.get("/financial/defaults")
def financial_defaults():
    return financial.defaults()


@app.post("/financial/predict")
def financial_predict(req: FinancialPredictRequest):
    try:
        return financial.predict(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Financial prediction failed: {e}")


@app.get("/financial/portfolio")
def financial_portfolio():
    return financial.portfolio()


@app.post("/financial/ask")
def financial_ask(req: AskRequest):
    _require_rag()
    return {"answer": financial.ask(req.question, req.k)}
