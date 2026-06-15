# Action Required: Add AI Projects to Portfolio Site

The four AI decision-support agents in this repository need to be integrated into the live portfolio site at **https://taash-portfolio.vercel.app/**.

**Scope: ML prediction only — do not include the RAG chat assistants.**

## Projects to Add

1. **ESG Energy & Emissions Optimization Agent** — predicts energy savings %, cost savings, and CO₂ reduction from building retrofit inputs
2. **Healthcare Patient-Flow Optimization Agent** — forecasts next-24h occupancy ratio, average wait time, and congestion risk flag
3. **Retail Inventory Optimization Agent** — predicts next-day demand and outputs a ranked replenishment action list with safety stock and economic impact
4. **Financial Credit Default Risk Agent** — scores 12-month probability of default, expected credit loss, and an approve/reprice/review/decline decision with reason codes

## What to Build

- 4 new pages on the Vercel site (one per project) matching the existing site style
- Each page: an input form → run prediction → display results (metric cards / action table)
- **No RAG chat component** — exclude the "Decision Rationale & Evidence Assistant" sections entirely

## How to Do This

A full technical handoff document exists in this repo: **`HANDOFF_VERCEL_PORTFOLIO.md`**

When using it, follow **§5 (inference contracts)** and **§7.2 (per-agent UI)** for the ML prediction pieces only. Skip the `/ask` endpoints and `<RagChat>` component — those are RAG and are out of scope here.

## Quick Summary

- The **FastAPI backend** is already built in `api/` — the `/predict` endpoints for all 4 agents are ready. It just needs to be deployed to Render.
- The **Next.js frontend** in `Taash_Chikosi_Portfolio` needs 4 new pages wired to the `/predict` endpoints only.
- No OpenAI or DeepSeek API keys are needed — predictions run entirely from the local ML models, no external API calls.
- The Streamlit app remains live and untouched.
