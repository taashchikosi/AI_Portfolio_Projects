# Governance and Guardrails — Energy Efficiency Decision-Support Model

## 1. Purpose
This document defines the governance principles and guardrails that guide the responsible use of the energy-efficiency decision-support model.

The objective is to:
- Prevent misuse or overreliance on model outputs
- Ensure transparency and accountability
- Support safe, ethical, and defensible decision-making

This model is governed as a **decision-support system**, not an automated decision-maker.

---

## 2. Decision-Support Classification
The model is explicitly classified as:
- Advisory
- Non-deterministic
- Human-in-the-loop

All outputs require interpretation by a qualified human decision-maker prior to action.

---

## 3. Grounding and Evidence Guardrails
To reduce the risk of hallucination or misinterpretation:
- Model explanations must be grounded in documented assumptions and knowledge-base content
- Outputs must clearly state limitations and uncertainty
- Unsupported claims must be explicitly avoided

When sufficient evidence is unavailable, the system should indicate this clearly.

---

## 4. Confidence and Uncertainty Handling
The model incorporates qualitative confidence signalling:
- High confidence does not imply certainty
- Medium confidence indicates partial uncertainty
- Low confidence requires caution and additional review

Confidence levels are intended to **inform judgment**, not replace it.

---

## 5. Human-in-the-Loop Requirements
Human review is required when:
- Confidence is assessed as low
- Results materially influence capital allocation decisions
- Outputs are referenced in ESG or sustainability discussions
- Scenarios involve atypical or extreme assumptions
- Outputs appear inconsistent with domain expectations

Escalation to domain experts is encouraged in these cases.

---

## 6. Misuse Prevention
The model must not be used for:
- Certified energy ratings or compliance claims
- Guaranteed performance assertions
- Public ESG disclosures without verification
- Replacement of accredited assessors or engineers

Clear boundaries help prevent reputational and regulatory risk.

---

## 7. Greenwashing Risk Mitigation
To avoid greenwashing:
- Energy savings and emissions impacts must be framed as indicative
- Assumptions must be stated explicitly
- Ranges and uncertainty should be preferred over point claims
- Marketing or external communications require additional review

Responsible framing is essential for credibility.

---

## 8. Transparency and Explainability
The model is designed to be explainable through:
- Feature-level interpretation
- Scenario-based comparisons
- Documented assumptions and logic

Opaque or unexplained outputs should be treated as insufficient for decision-making.

---

## 9. Governance Ownership
Responsibility for this model includes:
- Maintaining documentation accuracy
- Reviewing updates and changes
- Monitoring use cases and boundaries
- Ensuring alignment with organisational governance standards

Governance ownership should be clearly assigned in any production deployment.

---

## 10. Continuous Review
Governance controls should be revisited when:
- New data sources are introduced
- Model logic or features change
- Use cases expand beyond original intent
- Regulatory or ESG expectations evolve

Ongoing review supports long-term trust and relevance.
