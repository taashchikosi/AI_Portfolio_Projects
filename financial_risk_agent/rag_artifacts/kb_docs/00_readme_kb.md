# Financial Credit Default Risk Agent — Knowledge Base (KB)

This knowledge base powers the Financial Credit Default Risk Agent’s RAG assistant.  
The assistant must answer using ONLY these KB documents and must cite sources.

## What the system does
- Predicts **12-month Probability of Default (PD)** for consumer consumer loans.
- Translates PD into **Expected Credit Loss (ECL)** using:  
  **ECL = PD × LGD × EAD**
- Produces an operational **decision recommendation**:
  - APPROVE / APPROVE_REPRICE / MANUAL_REVIEW / DECLINE
- Provides **reason codes** + **severity** + **next-best actions**.

## Intended users
- Credit/risk analysts
- Product and portfolio managers
- Operations leads (review queues)
- Governance/compliance stakeholders (for auditability)

## Hard guardrails (must follow)
- Do NOT invent policy thresholds, model metrics, or governance rules.
- If KB evidence is insufficient, respond: “Not enough evidence in KB” and ask a targeted follow-up question.
- Treat outputs as **decision support** — not autonomous lending decisions.
