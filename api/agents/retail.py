"""Retail Inventory Optimization agent: prediction + RAG.

Ports TAB 3 logic from app.py. The full feature frame is expensive to build
from the CSVs, so it is built once and cached; each request just filters it by
decision_date.
"""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import REPO_ROOT
from ..rag import rag_answer

RETAIL_DIR = REPO_ROOT / "retail_project_artifacts"
RETAIL_VECTOR_DIR = REPO_ROOT / "retail_rag_artifacts" / "vector_store"

RETAIL_SYSTEM_PROMPT = """
You are a retail inventory decision-support assistant.

Rules (must follow):
- Use ONLY the provided CONTEXT for factual claims.
- Do NOT invent inventory policies, financial figures, supplier rules, or domain facts not present in CONTEXT.
- Treat ALL economic outputs as directional proxies unless CONTEXT explicitly states otherwise.
- If CONTEXT is insufficient, say so and ask a specific follow-up question.
- Always include citations to the retrieved sources.

Return format:
1) Answer
2) Why
3) Evidence (bullets with source + chunk)
4) Confidence (High/Medium/Low)
5) Human Review Trigger (Yes/No + reason)
"""

ACTION_COLS = [
    "sku_id", "store_id", "category",
    "on_hand", "on_order_qty", "inventory_position",
    "pred_next_day_demand", "lead_time_days",
    "safety_stock", "reorder_point", "recommended_order_qty",
    "hold_cost_proxy", "stockout_benefit_proxy",
    "working_capital_required", "net_value_proxy",
]


# ---- economics helpers (verbatim from app.py) ----
def safety_stock(z, demand_std, lead_time_days):
    return float(z) * float(demand_std) * float(np.sqrt(max(1.0, float(lead_time_days))))


def reorder_point(mean_demand, lead_time_days, safety_stock_units):
    return float(mean_demand) * float(lead_time_days) + float(safety_stock_units)


def holding_cost_per_unit_per_day(unit_cost, annual_hold_pct):
    return float(unit_cost) * (float(annual_hold_pct) / 365.0)


def stockout_penalty_per_unit(unit_margin, penalty_factor):
    return float(unit_margin) * float(penalty_factor)


def economic_impact(order_qty, unit_cost, annual_hold_pct, unit_margin, penalty_factor, stockout_units_avoided):
    hold_cost = float(order_qty) * holding_cost_per_unit_per_day(unit_cost, annual_hold_pct) * 14.0
    stockout_benefit = float(stockout_units_avoided) * stockout_penalty_per_unit(unit_margin, penalty_factor)
    working_capital = float(order_qty) * float(unit_cost)
    net_value = stockout_benefit - hold_cost
    return hold_cost, stockout_benefit, working_capital, net_value


def retail_calendar(dates):
    cal = pd.DataFrame({"date": pd.to_datetime(dates)})
    cal["dow"] = cal["date"].dt.dayofweek
    cal["is_weekend"] = (cal["dow"] >= 5).astype(int)
    cal["month"] = cal["date"].dt.month
    cal["is_peak_season"] = cal["month"].isin([11, 12]).astype(int)
    cal["is_summer"] = cal["month"].isin([12, 1, 2]).astype(int)
    return cal


@lru_cache(maxsize=1)
def _load_model():
    models = joblib.load(RETAIL_DIR / "models.joblib")
    meta = json.loads((RETAIL_DIR / "model_meta.json").read_text())
    return models["demand_model"], meta


@lru_cache(maxsize=1)
def _feature_frame():
    products = pd.read_csv(RETAIL_DIR / "products.csv")
    suppliers = pd.read_csv(RETAIL_DIR / "suppliers.csv")
    sales = pd.read_csv(RETAIL_DIR / "daily_sales.csv", parse_dates=["date"])
    inv = pd.read_csv(RETAIL_DIR / "inventory_positions.csv", parse_dates=["date"])

    for d in (products, suppliers, sales, inv):
        d.columns = [c.strip().lower() for c in d.columns]
    for d in (products, suppliers, sales, inv):
        if "sku_id" in d.columns:
            d["sku_id"] = d["sku_id"].astype(str).str.strip()
        if "store_id" in d.columns:
            d["store_id"] = d["store_id"].astype(str).str.strip()

    df = sales.merge(inv, on=["date", "sku_id", "store_id"], how="left")
    df = df.merge(
        products[[
            "sku_id", "category", "base_price", "unit_cost", "gross_margin_pct",
            "annual_holding_cost_pct", "stockout_penalty_factor",
        ]],
        on="sku_id", how="left",
    )
    df = df.merge(
        suppliers[["sku_id", "lead_time_mean_days", "lead_time_sd_days"]],
        on="sku_id", how="left",
    )

    dates = pd.DatetimeIndex(df["date"].unique()).sort_values()
    df = df.merge(retail_calendar(dates), on="date", how="left")
    df = df.sort_values(["sku_id", "store_id", "date"])

    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df.groupby(["sku_id", "store_id"])["units_sold"].shift(lag)

    g = df.groupby(["sku_id", "store_id"])["units_sold"]
    df["roll7_mean"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    df["roll14_mean"] = g.transform(lambda s: s.shift(1).rolling(14).mean())
    df["roll14_std"] = g.transform(lambda s: s.shift(1).rolling(14).std())

    # Resolve category_x / category_y merge collision back to `category`.
    df.columns = [c.strip().lower() for c in df.columns]
    if "category" not in df.columns:
        if "category_x" in df.columns and "category_y" in df.columns:
            df["category"] = df["category_x"].fillna(df["category_y"])
        elif "category_x" in df.columns:
            df["category"] = df["category_x"]
        elif "category_y" in df.columns:
            df["category"] = df["category_y"]
    for c in ["category_x", "category_y"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    return df


def meta() -> dict:
    df = _feature_frame()
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    default_date = (df["date"].max() - pd.Timedelta(days=7)).date()
    return {
        "min_date": str(min_date),
        "max_date": str(max_date),
        "default_date": str(default_date),
    }


def predict(req) -> dict:
    df = _feature_frame()
    demand_model, retail_meta = _load_model()

    decision_date = pd.to_datetime(req.decision_date).date()
    snap = df[df["date"].dt.date == decision_date].copy()
    if snap.empty:
        raise ValueError("No data for the selected decision date.")

    snap.columns = [c.strip().lower() for c in snap.columns]

    X_cols_num = [c.strip().lower() for c in retail_meta["features_num"]]
    X_cols_cat = [c.strip().lower() for c in retail_meta["features_cat"]]
    needed = X_cols_num + X_cols_cat
    missing = [c for c in needed if c not in snap.columns]
    if missing:
        raise ValueError(f"Retail model features missing from dataset (schema mismatch): {missing}")

    X_pred = snap[needed].copy()
    snap["pred_next_day_demand"] = np.clip(demand_model.predict(X_pred), 0, None)

    snap["lead_time_days"] = (snap["lead_time_mean_days"].fillna(7) + req.lead_time_shock).clip(lower=1)
    snap["demand_std_proxy"] = snap["roll14_std"].fillna(snap["pred_next_day_demand"].clip(lower=1))

    Z = float(norm.ppf(req.service_level))
    snap["safety_stock"] = snap.apply(lambda r: safety_stock(Z, r["demand_std_proxy"], r["lead_time_days"]), axis=1)
    snap["reorder_point"] = snap.apply(lambda r: reorder_point(r["pred_next_day_demand"], r["lead_time_days"], r["safety_stock"]), axis=1)

    snap["inventory_position"] = snap["on_hand"].fillna(0) + snap["on_order_qty"].fillna(0)
    snap["recommended_order_qty"] = np.ceil((snap["reorder_point"] - snap["inventory_position"]).clip(lower=0)).astype(int)

    snap["unit_margin"] = (snap["price"] - snap["unit_cost"]).clip(lower=0)
    snap["pred_demand_over_lt"] = snap["pred_next_day_demand"] * snap["lead_time_days"]
    snap["stockout_units_avoided_proxy"] = np.minimum(
        snap["recommended_order_qty"],
        np.ceil(snap["pred_demand_over_lt"]).astype(int),
    )

    econ = snap.apply(lambda r: economic_impact(
        order_qty=r["recommended_order_qty"],
        unit_cost=r["unit_cost"],
        annual_hold_pct=r["annual_holding_cost_pct"],
        unit_margin=r["unit_margin"],
        penalty_factor=r["stockout_penalty_factor"],
        stockout_units_avoided=r["stockout_units_avoided_proxy"],
    ), axis=1)
    econ_df = pd.DataFrame(econ.tolist(), columns=[
        "hold_cost_proxy", "stockout_benefit_proxy", "working_capital_required", "net_value_proxy",
    ])
    snap = pd.concat([snap.reset_index(drop=True), econ_df.reset_index(drop=True)], axis=1)

    action = snap[snap["recommended_order_qty"] > 0].copy()
    action = action.sort_values(["net_value_proxy", "stockout_benefit_proxy"], ascending=False)

    kpis = {
        "recommended_actions": int(action.shape[0]),
        "working_capital_required": float(action["working_capital_required"].sum()),
        "net_value_proxy": float(action["net_value_proxy"].sum()),
    }
    actions = json.loads(action[ACTION_COLS].round(2).to_json(orient="records"))
    return {"decision_date": str(decision_date), "kpis": kpis, "actions": actions}


def ask(question: str, k: int = 5) -> str:
    return rag_answer(RETAIL_VECTOR_DIR, RETAIL_SYSTEM_PROMPT, question, k)
