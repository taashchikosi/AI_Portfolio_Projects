# AI Portfolio Agents — FastAPI backend

JSON API that serves predictions + RAG answers for the four agents (ESG,
Healthcare, Retail, Financial). The Next.js portfolio frontend calls this.
The Streamlit app (`app.py`) is unaffected and remains the source of truth for
the business logic.

## Run locally

```bash
# from the repo root
pip install -r api/requirements.txt
export DEEPSEEK_API_KEY=sk-...      # RAG chat (DeepSeek)
export OPENAI_API_KEY=sk-...        # RAG embeddings (OpenAI — mandatory)
export ALLOWED_ORIGINS=http://localhost:3000
uvicorn api.main:app --reload
# docs: http://127.0.0.1:8000/docs
```

Predictions work without keys; only the `/ask` (RAG) endpoints need them.

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | – | `{status:"ok"}` (keep-warm target) |
| GET | `/ready` | – | `{rag_keys_ready: bool}` |
| POST | `/esg/predict` | 6 ESG floats | savings_pct + energy/cost/CO₂ |
| POST | `/esg/ask` | `{question,k?}` | `{answer}` |
| POST | `/healthcare/predict` | 7 inputs | occ ratio, wait, risk flag, feature_row |
| POST | `/healthcare/ask` | `{question,k?}` | `{answer}` |
| GET | `/retail/meta` | – | `{min_date,max_date,default_date}` |
| POST | `/retail/predict` | `{decision_date,service_level,lead_time_shock}` | `{kpis,actions}` |
| POST | `/retail/ask` | `{question,k?}` | `{answer}` |
| GET | `/financial/defaults` | – | medians/options/modes for the form |
| POST | `/financial/predict` | ~20 inputs | PD, ECL, decision, reasons, actions |
| GET | `/financial/portfolio` | – | portfolio aggregates |
| POST | `/financial/ask` | `{question,k?}` | `{answer}` |

## Deploy (Render)

1. New **Web Service** → connect this repo → **Docker** (uses `api/Dockerfile`).
2. Env vars: `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `ALLOWED_ORIGINS`
   (your Vercel domain, comma-separated).
3. Verify `GET /health` → `{"status":"ok"}`.

### Keep-warm (avoid cold starts on the free tier)

- **Primary:** an external pinger (UptimeRobot / cron-job.org) hitting
  `https://<service>.onrender.com/health` every 5 min.
- **Backup:** the repo's `.github/workflows/keep-warm.yml` — set repo secret
  `BACKEND_HEALTH_URL` to the same `/health` URL.

## Notes / gotchas

- `scikit-learn==1.6.1` and `joblib==1.4.2` are pinned to match the pickled
  models — do not bump without re-testing `joblib.load`.
- Keep **OpenAI** for embeddings; the Chroma stores live in that embedding space.
- Healthcare uses the **local** model artifacts (not app.py's GitHub-Release
  download path).
- The retail feature frame is built once and cached at first request.
