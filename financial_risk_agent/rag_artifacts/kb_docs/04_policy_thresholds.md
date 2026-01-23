# Policy Thresholds (Decision Bands)

## PD thresholds (business policy)
- APPROVE: PD < 0.03
- APPROVE_REPRICE: 0.03 ≤ PD < 0.08
- MANUAL_REVIEW: 0.08 ≤ PD < 0.15
- DECLINE: PD ≥ 0.15

## Purpose
- Translate a probability estimate into operational actions
- Maintain consistency with risk appetite and review capacity

## Notes
- Thresholds should be versioned and reviewed regularly
- Threshold selection should consider:
  - expected loss targets
  - operational review capacity
  - fairness & adverse impact analysis
