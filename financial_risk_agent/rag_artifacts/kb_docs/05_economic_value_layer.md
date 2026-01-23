# Economic Value Translation Layer (PD → $)

## Core formula
ECL = PD × LGD × EAD

## LGD (Loss Given Default)
- Severity of loss if default occurs
- In this project: bounded 0.30–0.90 using a simple risk-based heuristic

## EAD (Exposure at Default)
- Outstanding balance at time of default
- In this project: approximated as current_balance for origination

## Expected profit (proxy, simplified)
expected_profit ≈ expected_interest_income − ECL − operating_cost

Where:
- expected_interest_income ≈ loan_amount × interest_rate × term_years × avg_balance_factor
- avg_balance_factor ≈ 0.5
- operating_cost is a fixed proxy cost per loan (e.g., $150)

## Repricing (proxy)
If in APPROVE_REPRICE band:
- compute a required interest rate that covers expected loss + target margin
- clamp to realistic bounds (5%–35%)
