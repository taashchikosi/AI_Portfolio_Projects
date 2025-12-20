# Evaluation and Confidence — Energy Efficiency Decision-Support Model

## 1. Purpose
This document explains how the model’s outputs should be evaluated and how confidence levels are interpreted.

The objective is to:
- Provide transparency about model reliability
- Support responsible interpretation of results
- Clarify when additional review is required

Evaluation is designed to support **decision confidence**, not to claim predictive certainty.

---

## 2. Evaluation Approach
The model was evaluated using standard machine-learning validation techniques appropriate for exploratory and decision-support use cases.

Evaluation focused on:
- Generalisation to unseen scenarios
- Stability of directional trends
- Consistency of feature influence
- Reasonableness of predicted savings ranges

The emphasis is on **relative accuracy and signal quality**, rather than exact point prediction.

---

## 3. Interpretation of Performance
Model performance should be interpreted as follows:
- The model is effective at identifying **which factors matter most**
- The model supports **comparison between scenarios**
- The model provides **directional insight**, not exact outcomes

Higher apparent numerical precision does not imply higher real-world certainty.

---

## 4. Confidence Levels
Confidence levels are qualitative and assigned based on input quality, similarity to training data, and retrieval support from the knowledge base.

### High Confidence
- Inputs closely resemble typical scenarios seen during training
- Key features are well-defined and complete
- Model behaviour aligns with known energy-efficiency principles

### Medium Confidence
- Minor extrapolation beyond typical conditions
- Partial uncertainty in one or more inputs
- Directional results are still considered informative

### Low Confidence
- Significant extrapolation or unusual building characteristics
- Missing, estimated, or highly uncertain inputs
- Weak or ambiguous supporting evidence

Low-confidence outputs should not be used without additional validation.

---

## 5. Factors That Reduce Confidence
Confidence may be reduced when:
- Buildings fall outside common archetypes
- Inputs are estimated rather than measured
- Multiple interacting interventions are applied simultaneously
- Scenarios involve extreme efficiency assumptions

These conditions increase uncertainty and warrant caution.

---

## 6. Human Review Triggers
Human expert review is recommended when:
- Confidence is assessed as low
- Results materially influence investment decisions
- Outputs are referenced in ESG or strategic communications
- Model behaviour appears inconsistent with domain expectations

Human oversight is a required component of responsible use.

---

## 7. Governance Alignment
This evaluation framework supports:
- Transparent communication of uncertainty
- Avoidance of overconfidence or overclaiming
- Alignment with internal governance and ESG controls

The model is governed as a **decision-support tool**, not an automated decision-maker.

---

## 8. Ongoing Monitoring
As the model evolves, evaluation and confidence criteria should be revisited to reflect:
- New data sources
- Expanded feature sets
- Changing use cases
- Improved calibration techniques

Continuous review helps maintain trust and relevance.
