# System Overview — PD → ECL → Decision + Reason Codes

## Inputs (feature families)
**Applicant**
- age, dependents
- employment_status, tenure_months
- annual_income, residence_type

**Credit/Bureau signals**
- credit_score
- delinquencies_12m
- inquiries_6m
- revolving_utilization
- total_open_accounts
- months_since_last_delinquency

**Loan terms**
- loan_amount
- term_months
- interest_rate
- installment_amount
- payment_to_income_ratio (PTI)
- purpose

**Macro & regional context**
- unemployment_rate
- inflation_rate
- cash_rate_proxy
- region
- region_risk_index

## Outputs (what the tool returns)
1) **PD**: probability of default within 12 months  
2) **LGD**: loss severity if default occurs  
3) **EAD**: exposure at default (approx current balance)  
4) **ECL**: expected loss in dollars  
5) **Decision**: approve / reprice / review / decline  
6) **Reason codes** + severity + recommended actions

## Decision bands (policy thresholds)
- APPROVE: PD < 3%
- APPROVE_REPRICE: 3% ≤ PD < 8%
- MANUAL_REVIEW: 8% ≤ PD < 15%
- DECLINE: PD ≥ 15%

## Why it’s useful
- Standardizes risk screening
- Makes PD interpretable as **financial loss**
- Makes decisions explainable with **bank-style reasons**
- Enables governance: review triggers, documentation, monitoring
