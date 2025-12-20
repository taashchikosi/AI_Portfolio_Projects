# Model Card — Energy Efficiency Savings Decision-Support Model

## 1. Model Overview
This model estimates the **directional impact of energy-efficiency interventions** on building energy consumption.  
It is designed to support **early-stage decision-making**, scenario exploration, and investment prioritisation.

The model does **not** certify performance, guarantee savings, or replace engineering judgment.

---

## 2. Intended Use
This model is intended to support:
- Prioritisation of energy-efficiency investments
- Comparison of intervention scenarios (e.g. HVAC, lighting, controls)
- Pre-assessment analysis prior to engaging engineers or assessors
- Internal ESG and sustainability decision support
- Early-stage portfolio-level screening

Outputs should be used as **decision-support signals**, not final determinations.

---

## 3. Not Intended Use
This model must **not** be used for:
- Certified energy ratings (e.g. NABERS)
- Regulatory or compliance reporting
- Guaranteed savings claims
- Replacement of accredited assessors or engineers
- Public ESG disclosures without additional verification

---

## 4. Input Assumptions
The model assumes:
- Input features reasonably represent typical operating conditions
- Building characteristics are broadly similar to the training data distribution
- Efficiency interventions are implemented to standard industry quality
- Operational behaviour remains broadly consistent post-intervention

Violations of these assumptions may reduce reliability.

---

## 5. Model Outputs
The model produces:
- Estimated percentage energy savings (directional)
- Relative contribution of key drivers (feature importance)
- Scenario-based comparative outputs

All outputs are **indicative**, not deterministic.

---

## 6. Known Limitations
Key limitations include:
- Reduced reliability for out-of-distribution building types
- Inability to capture installation quality or commissioning variance
- No modelling of occupant behaviour changes
- Dependence on assumed baseline conditions

Predictions should be interpreted in context.

---

## 7. Confidence Interpretation
Confidence levels are qualitative and defined as:
- **High**: Inputs well-aligned with training distribution; strong supporting evidence
- **Medium**: Minor extrapolation or partial data uncertainty
- **Low**: Significant extrapolation, missing inputs, or weak evidence support

Low-confidence outputs should trigger additional review.

---

## 8. Human Review Triggers
Human expert review is recommended when:
- Input data is incomplete or estimated
- Predictions fall outside expected ranges
- Confidence is assessed as low
- Results are used to support high-stakes decisions
- Outputs are referenced in ESG or strategic communications

---

## 9. Governance Statement
This model is governed as a **decision-support tool**, not an automated decision-maker.  
Human oversight is required for interpretation, validation, and action.

Responsible use requires adherence to stated limitations and governance controls.
