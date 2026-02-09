"""
QBO Expense Engine - Apportionment Engine
Manages shared/direct tagging and brand allocation percentages.
"""
import json
import os
from config import PL_BRANDS, ALL_BRANDS

SHARED_RULES_FILE = "data/shared_rules.json"
APPORTION_FILE = "data/apportionment.json"
TXN_APPORTION_FILE = "data/txn_apportionment.json"


def load_shared_rules() -> dict:
    """Load shared/direct rules. Key = description_key, Value = 'shared' or 'direct'."""
    if os.path.exists(SHARED_RULES_FILE):
        with open(SHARED_RULES_FILE, "r") as f:
            return json.load(f)
    return {}


def save_shared_rules(rules: dict):
    os.makedirs(os.path.dirname(SHARED_RULES_FILE), exist_ok=True)
    with open(SHARED_RULES_FILE, "w") as f:
        json.dump(rules, f, indent=2)


def load_apportionment() -> dict:
    """
    Load apportionment percentages.
    Structure: { "head_category": { "brand": percentage, ... }, ... }
    """
    if os.path.exists(APPORTION_FILE):
        with open(APPORTION_FILE, "r") as f:
            return json.load(f)
    return {}


def save_apportionment(data: dict):
    os.makedirs(os.path.dirname(APPORTION_FILE), exist_ok=True)
    with open(APPORTION_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_default_apportionment() -> dict:
    """Return equal split across all PL brands as default."""
    n = len(PL_BRANDS)
    pct = round(100.0 / n, 2)
    return {brand: pct for brand in PL_BRANDS}


def load_txn_apportionment() -> dict:
    """Load per-transaction apportionment. Key = str(df_index), Value = {brand: pct}."""
    if os.path.exists(TXN_APPORTION_FILE):
        with open(TXN_APPORTION_FILE, "r") as f:
            return json.load(f)
    return {}


def save_txn_apportionment(data: dict):
    os.makedirs(os.path.dirname(TXN_APPORTION_FILE), exist_ok=True)
    with open(TXN_APPORTION_FILE, "w") as f:
        json.dump(data, f, indent=2)


def apportion_by_splits(amount: float, splits: dict) -> list:
    """Apportion a transaction using explicit % splits. Returns [(brand, amount)]."""
    results = []
    for brand, pct in splits.items():
        allocated = round(amount * (pct / 100.0), 2)
        if allocated != 0:
            results.append((brand, allocated))
    if results:
        total_allocated = sum(a for _, a in results)
        diff = round(amount - total_allocated, 2)
        if diff != 0:
            b, a = results[0]
            results[0] = (b, round(a + diff, 2))
    return results


def apportion_transaction(amount: float, head: str, source_brand: str,
                           apportionment: dict, account_type: str) -> list:
    """
    Apportion a shared expense across brands.
    
    Rules:
    - Wholesale expenses → shared to PL brands only
    - PL expenses → shared to other PL brands only (never to Wholesale)
    - Originating brand keeps their share
    
    Returns list of (brand, allocated_amount) tuples.
    """
    if head not in apportionment:
        # Use default equal split
        splits = get_default_apportionment()
    else:
        splits = apportionment[head]

    results = []
    
    if account_type == "Wholesale":
        # Wholesale → only PL brands get shares
        pl_splits = {b: s for b, s in splits.items() if b in PL_BRANDS}
        total_pct = sum(pl_splits.values())
        if total_pct > 0:
            for brand, pct in pl_splits.items():
                allocated = round(amount * (pct / 100.0), 2)
                if allocated != 0:
                    results.append((brand, allocated))
    else:
        # Private Label → PL brands only (including originating)
        pl_splits = {b: s for b, s in splits.items() if b in PL_BRANDS}
        total_pct = sum(pl_splits.values())
        if total_pct > 0:
            for brand, pct in pl_splits.items():
                allocated = round(amount * (pct / 100.0), 2)
                if allocated != 0:
                    results.append((brand, allocated))

    # Adjust rounding — ensure amounts sum to original
    if results:
        total_allocated = sum(a for _, a in results)
        diff = round(amount - total_allocated, 2)
        if diff != 0 and results:
            # Add rounding diff to first brand
            brand, amt = results[0]
            results[0] = (brand, round(amt + diff, 2))

    return results
