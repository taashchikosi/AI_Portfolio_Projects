# Governance & Controls

## Auditability requirements
For each decision, store:
- input snapshot (features used)
- PD, LGD, EAD, ECL
- decision band (policy version)
- reason codes + severity
- next actions
- timestamp + model version hash

## Human review triggers
- MANUAL_REVIEW band
- High-severity reasons (severity 5)
- Unusual feature values / missing data
- Macro stress elevated

## Change control
- Version thresholds
- Version model artifact
- Log model training metadata:
  - train/test split strategy
  - metrics
  - calibration notes
  - feature schema

## Model monitoring (recommended)
- PD drift (distribution shift)
- calibration drift (Brier score / calibration curve)
- decision mix drift (approve/decline rate changes)
- segment performance monitoring
