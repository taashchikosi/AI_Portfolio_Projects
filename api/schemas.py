"""Pydantic request models for the agent endpoints."""
from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    k: int = 5


class ESGPredictRequest(BaseModel):
    floor_area_m2: float
    building_age_years: float
    hvac_efficiency_score: float
    insulation_quality_score: float
    occupancy_rate: float
    baseline_energy_kwh: float


class HealthcarePredictRequest(BaseModel):
    capacity_beds: int
    current_occupancy_pct: int
    arrivals_pressure: str          # "Low" | "Normal" | "High"
    staffing_level: str             # "Low" | "Normal" | "High"
    current_wait_minutes: int
    date: str                       # ISO date, e.g. "2024-10-18"
    hour: int                       # 0-23


class RetailPredictRequest(BaseModel):
    decision_date: str              # ISO date within the data range
    service_level: float = 0.95     # 0.80 - 0.99
    lead_time_shock: int = 0        # extra days, 0 - 21


class FinancialPredictRequest(BaseModel):
    age: int
    employment_status: str
    annual_income: float
    residence_type: str
    dependents: int
    tenure_months: int
    credit_score: int
    delinquencies_12m: int
    inquiries_6m: int
    revolving_utilization: float
    total_open_accounts: int
    months_since_last_delinquency: int
    loan_amount: float
    term_months: int
    interest_rate: float
    purpose: str
    unemployment_rate: float
    inflation_rate: float
    cash_rate_proxy: float
    region: str
