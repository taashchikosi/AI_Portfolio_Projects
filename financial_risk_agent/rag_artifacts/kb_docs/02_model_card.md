# Model Card — PD Model (Champion)

## Task
Binary classification:
- Target = default_12m (1 = default within 12 months)

## Model type
- A calibrated probability model (outputs PD)
- Calibration is required because PD is used for:
  - pricing decisions
  - ECL computation
  - risk banding thresholds

## Training data
- Synthetic consumer loan dataset designed to simulate real lending patterns
- Includes applicant, credit, loan term, and macro features

## Intended use
- Decision support for:
  - risk screening
  - repricing suggestions
  - routing to manual review queues
  - decline recommendations

## Not intended use
- Fully automated adverse action decisions without oversight
- Use on real customer data without:
  - privacy controls
  - governance approval
  - bias testing & monitoring

## Human review triggers
- Borderline PD bands (manual review)
- High-severity reason codes (see reason code spec)
- Unusual inputs or missing data
