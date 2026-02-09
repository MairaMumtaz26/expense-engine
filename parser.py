"""
QBO Expense Engine - GL File Parser
Reads QBO General Ledger Excel exports and extracts expense transactions.
"""
import pandas as pd
import re
from config import ACCOUNT_MAP, FILENAME_KEYWORDS, IGNORED_KEYWORDS


def identify_account(filename: str, company_name: str) -> dict:
    """Identify QBO account from filename or company name in the file."""
    # Try exact match on company name first
    if company_name in ACCOUNT_MAP:
        info = ACCOUNT_MAP[company_name].copy()
        info["qbo_account"] = company_name
        return info

    # Try filename keyword matching
    for keyword, account_name in FILENAME_KEYWORDS.items():
        if keyword.lower() in filename.lower():
            info = ACCOUNT_MAP[account_name].copy()
            info["qbo_account"] = account_name
            return info

    # Not found
    return None


def parse_gl_file(filepath: str) -> tuple:
    """
    Parse a QBO General Ledger Excel file.
    
    Returns:
        (account_info: dict, period: str, transactions: list[dict])
    """
    filename = filepath.split("/")[-1] if "/" in filepath else filepath.split("\\")[-1]
    df = pd.read_excel(filepath, header=None)

    # Row 0 = Company name, Row 2 = Period
    company_name = str(df.iloc[0, 0]).strip()
    period = str(df.iloc[2, 0]).strip()

    # Identify account
    account_info = identify_account(filename, company_name)

    # Find header row (row 4 typically)
    header_row = 4
    headers = list(df.iloc[header_row].dropna().values)

    # Determine column indices based on headers
    n_cols = df.shape[1]
    # Standard: col0=group/label, col1=Date, col2=TxnType, col3=Num/#, col4=Name, col5=Memo, col6=Amount
    # WST-style: col0=group, col1=Date, col2=TxnType, col3=Num, col4=Name, col5=Memo, col6=Split, col7=Amount, col8=Balance

    has_balance = "Balance" in headers
    if has_balance:
        # 9-column format
        col_date = 1
        col_txn_type = 2
        col_num = 3
        col_name = 4
        col_memo = 5
        col_amount = 7  # Amount is col 7, Balance is col 8
    else:
        # 7-column format
        col_date = 1
        col_txn_type = 2
        col_num = 3
        col_name = 4
        col_memo = 5
        col_amount = 6

    transactions = []
    current_group = None

    for idx in range(header_row + 1, len(df)):
        row = df.iloc[idx]
        col0 = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        date_val = row.iloc[col_date] if pd.notna(row.iloc[col_date]) else None
        amount_val = row.iloc[col_amount] if pd.notna(row.iloc[col_amount]) else None

        # Skip empty rows, footer rows, beginning balance rows
        if col0 == "" and date_val is None and amount_val is None:
            continue
        if "Accrual Basis" in col0 or "Cash Basis" in col0:
            continue
        if date_val is not None and str(date_val).strip() == "Beginning Balance":
            continue

        # Group header row (expense category from QBO)
        if col0 and not col0.startswith("Total for") and date_val is None:
            current_group = col0
            continue

        # Total row - skip
        if col0.startswith("Total for"):
            continue

        # Transaction row
        if date_val is not None and amount_val is not None:
            # Parse date
            try:
                if isinstance(date_val, str):
                    date_parsed = pd.to_datetime(date_val)
                else:
                    date_parsed = pd.to_datetime(date_val)
            except:
                continue

            name_val = str(row.iloc[col_name]).strip() if pd.notna(row.iloc[col_name]) else ""
            memo_val = str(row.iloc[col_memo]).strip() if pd.notna(row.iloc[col_memo]) else ""
            txn_type = str(row.iloc[col_txn_type]).strip() if pd.notna(row.iloc[col_txn_type]) else ""

            transactions.append({
                "date": date_parsed,
                "txn_type": txn_type,
                "name": name_val,
                "memo": memo_val,
                "description": f"{name_val} {memo_val}".strip(),
                "amount": float(amount_val),
                "qbo_group": current_group or "",
            })

    return account_info, period, transactions


def parse_all_files(file_paths: list) -> pd.DataFrame:
    """Parse multiple GL files and return a combined DataFrame."""
    all_rows = []
    unidentified = []

    for fp in file_paths:
        account_info, period, transactions = parse_gl_file(fp)
        filename = fp.split("/")[-1] if "/" in fp else fp.split("\\")[-1]

        if account_info is None:
            unidentified.append(filename)
            continue

        for txn in transactions:
            all_rows.append({
                "qbo_account": account_info["qbo_account"],
                "brand": account_info["brand"],
                "currency": account_info["currency"],
                "account_type": account_info["type"],
                "period": period,
                "date": txn["date"],
                "txn_type": txn["txn_type"],
                "name": txn["name"],
                "memo": txn["memo"],
                "description": txn["description"],
                "amount_lcy": txn["amount"],
                "qbo_group": txn["qbo_group"],
            })

    df = pd.DataFrame(all_rows)

    # Filter out ignored keywords (China/Pakistan budget)
    if not df.empty and IGNORED_KEYWORDS:
        def should_ignore(row):
            search_text = f"{row.get('name', '')} {row.get('memo', '')} {row.get('description', '')} {row.get('qbo_group', '')}".lower()
            return any(kw in search_text for kw in IGNORED_KEYWORDS)
        mask = df.apply(should_ignore, axis=1)
        ignored_count = mask.sum()
        df = df[~mask].reset_index(drop=True)
        if ignored_count > 0:
            print(f"Filtered out {ignored_count} China/Pakistan budget transactions")

    return df, unidentified
