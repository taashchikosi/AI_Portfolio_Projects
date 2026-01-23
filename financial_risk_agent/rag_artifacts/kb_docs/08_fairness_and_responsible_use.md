# Fairness, Bias, Responsible Use

## Data note
This is a synthetic dataset used for portfolio demonstration.

## Fairness approach for real deployment (recommended)
- Identify protected attributes per jurisdiction (do not infer)
- Evaluate parity:
  - AUC/PR, calibration (Brier), error rates across groups
- Check outcome parity:
  - adverse impact ratios by decision band
- Stress test under macro changes

## Key risks
- Proxy discrimination via correlated features (region/employment)
- Misinterpretation of PD as certainty
- Feedback loops (declines alter future applicant pool)

## Mitigations
- Human oversight in borderline / decline decisions
- Documented thresholds and review processes
- Ongoing monitoring and periodic revalidation
