"""
QBO Expense Engine - Classification Engine
Keyword matching against expense categories with learning capability.
Sub-head must differ from Head — Name searched first, then Memo.
"""
import json
import os
from config import KEYWORD_INDEX, EXPENSE_CATEGORIES

LEARNED_RULES_FILE = "data/learned_rules.json"


def load_learned_rules() -> dict:
    """Load previously learned classification rules."""
    if os.path.exists(LEARNED_RULES_FILE):
        with open(LEARNED_RULES_FILE, "r") as f:
            return json.load(f)
    return {}


def save_learned_rules(rules: dict):
    """Save learned classification rules."""
    os.makedirs(os.path.dirname(LEARNED_RULES_FILE), exist_ok=True)
    with open(LEARNED_RULES_FILE, "w") as f:
        json.dump(rules, f, indent=2)


def _find_specific_subhead(text: str, head: str) -> str | None:
    """Find a keyword in text that belongs to this Head but isn't the Head name itself."""
    head_lower = head.lower()
    for kw_lower, kw_head, kw_sub in KEYWORD_INDEX:
        if kw_head == head and kw_lower != head_lower and kw_lower in text:
            return kw_sub
    return None


def classify_transaction(description: str, name: str, memo: str, qbo_group: str, learned_rules: dict = None, amount: float = 0.0) -> tuple:
    """
    Classify a transaction into Head + Sub-head.

    Priority:
    0. Amount-based rules (e.g. Amazon > $300 → Marketing)
    1. Learned rules (exact match on description)
    2. Keyword matching — Name field first, then Memo, then combined
    3. QBO group name matching
    4. None (unmatched)

    Sub-head must differ from Head. If only Head-level keyword matched,
    scan Name then Memo for a more specific sub-head.

    Returns:
        (head, sub_head, confidence) where confidence is
        'learned', 'keyword', 'qbo_group', 'discrepancy', or 'unmatched'
    """
    if learned_rules is None:
        learned_rules = {}

    name_lower = name.lower()
    memo_lower = memo.lower()
    search_text = f"{name} {memo} {description}".lower()
    qbo_group_lower = qbo_group.lower() if qbo_group else ""

    # Payroll country mapping: person name keywords → sub-head
    _usa_payroll = ["alexa black", "hope syriah", "isaiah joseph", "joshua",
                    "manuel a leo", "varun arora", "elle smith", "tristian",
                    "sharrell"]
    _canada_payroll = ["kateryna rotko", "mark collins", "myroslava", "rihab",
                       "siarhei"]

    def _payroll_subhead(person_name: str) -> str:
        pn = person_name.lower()
        for kw in _usa_payroll:
            if kw in pn:
                return "USA Payroll"
        for kw in _canada_payroll:
            if kw in pn:
                return "Canada Payroll"
        return "USA Payroll"  # default

    # 0. Name-based payroll: if Name matches a known payroll person → Payroll
    for kw in _canada_payroll:
        if kw in name_lower:
            return "Payroll", "Canada Payroll", "keyword"
    for kw in _usa_payroll:
        if kw in name_lower:
            return "Payroll", "USA Payroll", "keyword"

    # 0a. Memo-based rules
    if any(kw in memo_lower for kw in ["employer health ins", "health ins. contribution", "health insurance"]):
        person = name.strip() if name.strip() else ""
        return "Payroll", _payroll_subhead(person), "keyword"

    # Payroll payment by account number
    if "payroll payment" in memo_lower or "pay-file" in memo_lower:
        _payroll_accounts = {
            "2688": "Varun Arora",
            "2962": "Joshua",
            "0340": "Tristian",
            "2876": "Alexa Black",
            "5221": "Manuel A Leo",
            "0611": "Sharrell L Smi",
            "0631": "Isaiah Joseph",
            "8317": "Hope Syriah Erickson",
        }
        for acct_suffix, person in _payroll_accounts.items():
            if memo.rstrip().endswith(acct_suffix):
                return "Payroll", _payroll_subhead(person), "keyword"

    # 0b. Amount-based rules
    if "amazon" in search_text and abs(amount) > 300:
        return "Marketing & Advert", "Amazon", "keyword"
    if "shopify" in search_text and abs(amount) > 100:
        return "Marketing & Advert", "Shopify", "keyword"
    if "walmart" in search_text and abs(amount) > 100:
        return "Marketing & Advert", "Walmart", "keyword"

    # 1. Check learned rules (exact description match)
    desc_key = description.strip().lower()
    if desc_key in learned_rules:
        rule = learned_rules[desc_key]
        return rule["head"], rule["sub_head"], "learned"

    # 2a. Keyword matching on Name only — prefer specific (non-Head) keywords
    for kw_lower, head, sub_head in KEYWORD_INDEX:
        if kw_lower in name_lower and kw_lower != head.lower():
            return head, sub_head, "keyword"

    # 2b. Keyword matching on Memo only — prefer specific keywords
    for kw_lower, head, sub_head in KEYWORD_INDEX:
        if kw_lower in memo_lower and kw_lower != head.lower():
            return head, sub_head, "keyword"

    # 2c. Full-text match (may hit head-level keywords like "Office Exp")
    for kw_lower, head, sub_head in KEYWORD_INDEX:
        if kw_lower in search_text:
            if sub_head.lower() != head.lower():
                return head, sub_head, "keyword"
            # Head-level keyword matched — try to find specific sub-head
            better = _find_specific_subhead(name_lower, head)
            if better:
                return head, better, "keyword"
            better = _find_specific_subhead(memo_lower, head)
            if better:
                return head, better, "keyword"
            # Couldn't find specific sub-head → discrepancy
            return head, head, "discrepancy"

    # 3. Try matching QBO group name to expense categories
    if qbo_group_lower:
        for kw_lower, head, sub_head in KEYWORD_INDEX:
            if kw_lower in qbo_group_lower:
                if sub_head.lower() != head.lower():
                    # Got a specific sub-head from QBO group, but try name/memo first
                    better = _find_specific_subhead(name_lower, head)
                    if better:
                        return head, better, "qbo_group"
                    better = _find_specific_subhead(memo_lower, head)
                    if better:
                        return head, better, "qbo_group"
                    return head, sub_head, "qbo_group"
                else:
                    # QBO group matched head-level keyword — try name/memo
                    better = _find_specific_subhead(name_lower, head)
                    if better:
                        return head, better, "qbo_group"
                    better = _find_specific_subhead(memo_lower, head)
                    if better:
                        return head, better, "qbo_group"
                    return head, head, "discrepancy"

        # Direct head name match on QBO group
        for head_name in EXPENSE_CATEGORIES.keys():
            if head_name.lower() in qbo_group_lower or qbo_group_lower in head_name.lower():
                better = _find_specific_subhead(name_lower, head_name)
                if better:
                    return head_name, better, "qbo_group"
                better = _find_specific_subhead(memo_lower, head_name)
                if better:
                    return head_name, better, "qbo_group"
                return head_name, head_name, "discrepancy"

    # 4. Unmatched
    return None, None, "unmatched"


def classify_dataframe(df):
    """
    Classify all transactions in a DataFrame.
    Adds columns: head, sub_head, match_confidence
    """
    learned_rules = load_learned_rules()

    heads = []
    sub_heads = []
    confidences = []

    for _, row in df.iterrows():
        head, sub_head, confidence = classify_transaction(
            description=str(row.get("description", "")),
            name=str(row.get("name", "")),
            memo=str(row.get("memo", "")),
            qbo_group=str(row.get("qbo_group", "")),
            learned_rules=learned_rules,
            amount=float(row.get("amount_lcy", 0)),
        )
        heads.append(head)
        sub_heads.append(sub_head)
        confidences.append(confidence)

    df = df.copy()
    df["head"] = heads
    df["sub_head"] = sub_heads
    df["match_confidence"] = confidences
    return df
