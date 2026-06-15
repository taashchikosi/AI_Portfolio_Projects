# Action Required: Add AI Projects to Portfolio Site

The four AI decision-support agents in this repository need to be integrated into the live portfolio site at **https://taash-portfolio.vercel.app/**.

## Projects to Add

1. **ESG Energy & Emissions Optimization Agent**
2. **Healthcare Patient-Flow Optimization Agent**
3. **Retail Inventory Optimization Agent**
4. **Financial Credit Default Risk Agent**

## How to Do This

A full technical handoff document already exists in this repo:
**`HANDOFF_VERCEL_PORTFOLIO.md`**

It covers the complete architecture, backend API (already built in `api/`), frontend build spec, deployment steps, and a ready-to-use cold-start UX pattern. Hand that file to a new chat with access to the `Taash_Chikosi_Portfolio` repo and it can execute the integration end-to-end.

## Quick Summary

- The **FastAPI backend** is already built and lives in `api/` — it just needs to be deployed to Render.
- The **Next.js frontend** in `Taash_Chikosi_Portfolio` needs 4 new agent pages added to match the live site's existing style.
- The Streamlit app remains live and untouched — the Vercel site will call the Render backend for predictions and RAG, not Streamlit.
