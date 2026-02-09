"""
QBO Expense Engine - FX Rate Management
Handles CAD → USD conversion using monthly average rates.
"""
import json
import os
import re
from datetime import datetime

FX_RATES_FILE = "data/fx_rates.json"

# Default monthly average rates (CAD per 1 USD) - user can override
# These are approximate — the app will prompt user to confirm/update
DEFAULT_RATES = {
    "2025-01": 1.4400,
    "2025-02": 1.4300,
    "2025-03": 1.4350,
    "2025-04": 1.4300,
    "2025-05": 1.3900,
    "2025-06": 1.3800,
    "2025-07": 1.3750,
    "2025-08": 1.3600,
    "2025-09": 1.3500,
    "2025-10": 1.3800,
    "2025-11": 1.4000,
    "2025-12": 1.4350,
    "2026-01": 1.3787,
    "2026-02": 1.3647,
}


def load_fx_rates() -> dict:
    if os.path.exists(FX_RATES_FILE):
        with open(FX_RATES_FILE, "r") as f:
            return json.load(f)
    return DEFAULT_RATES.copy()


def save_fx_rates(rates: dict):
    os.makedirs(os.path.dirname(FX_RATES_FILE), exist_ok=True)
    with open(FX_RATES_FILE, "w") as f:
        json.dump(rates, f, indent=2)


def get_month_key(date) -> str:
    """Convert a date to YYYY-MM format."""
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")
    return date.strftime("%Y-%m")


def convert_cad_to_usd(amount_cad: float, date, fx_rates: dict = None) -> tuple:
    """
    Convert CAD amount to USD.
    
    Returns:
        (amount_usd, exchange_rate_used)
    """
    if fx_rates is None:
        fx_rates = load_fx_rates()

    month_key = get_month_key(date)

    if month_key in fx_rates:
        rate = fx_rates[month_key]
        amount_usd = round(amount_cad / rate, 2)
        return amount_usd, rate
    else:
        # Return as-is with rate 0 (flag for user to input)
        return amount_cad, 0


def get_required_months(df) -> list:
    """Get all unique months from CAD transactions that need FX rates."""
    cad_df = df[df["currency"] == "CAD"]
    if cad_df.empty:
        return []
    months = cad_df["date"].apply(get_month_key).unique().tolist()
    return sorted(months)
