# Data Dictionary (Key Features)

## Applicant
- age: integer
- annual_income: numeric (annual)
- dependents: integer (count)
- employment_status: categorical (e.g., Employed, Self-employed, Student, Unemployed)
- tenure_months: numeric (months at current job)
- residence_type: categorical (Rent, Mortgage, Own)

## Credit / Bureau
- credit_score: numeric (typical range 300–850)
- delinquencies_12m: integer (count)
- inquiries_6m: integer (count)
- revolving_utilization: numeric (0–1)
- total_open_accounts: integer
- months_since_last_delinquency: integer

## Loan
- loan_amount: numeric
- term_months: integer
- interest_rate: numeric (0–1)
- installment_amount: numeric (monthly payment estimate)
- payment_to_income_ratio: numeric (installment / monthly income)
- purpose: categorical (e.g., debt_consolidation, auto, home, personal)

## Macro / Region
- unemployment_rate: numeric (%)
- inflation_rate: numeric (%)
- cash_rate_proxy: numeric (%)
- region: categorical
- region_risk_index: numeric multiplier (e.g., 0.95–1.18)

## Loss components
- lgd: numeric (0.30–0.90) — loss severity if default occurs
- ead: numeric — exposure at default (approx outstanding balance)
