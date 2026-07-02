import re
import time

import pandas as pd


def normalize_narration(text: str) -> str:
    """Collapse multi-line narrations into one line; segments join with no gap."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = [line.strip() for line in text.split("\n") if line.strip()]
    if parts:
        return "".join(parts)
    return re.sub(r"\s+", " ", text).strip()


def extract_rows_iterrows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        rec = {
            "date": str(row.get("date", "")).strip(),
            "particulars": normalize_narration(str(row.get("particulars", ""))),
            "deposits": str(row.get("deposits", "")).strip(),
            "withdrawals": str(row.get("withdrawals", "")).strip(),
            "balance": str(row.get("balance", "")).strip(),
        }
        if not any(rec.values()):
            continue
        rows.append(rec)
    return rows


def extract_rows_optimized(df: pd.DataFrame) -> list[dict]:
    rows = []
    for row in df.to_dict("records"):
        rec = {
            "date": str(row.get("date", "")).strip(),
            "particulars": normalize_narration(str(row.get("particulars", ""))),
            "deposits": str(row.get("deposits", "")).strip(),
            "withdrawals": str(row.get("withdrawals", "")).strip(),
            "balance": str(row.get("balance", "")).strip(),
        }
        if not any(rec.values()):
            continue
        rows.append(rec)
    return rows


if __name__ == "__main__":
    # Create mock dataframe
    num_rows = 50000
    data = {
        "date": ["01-01-2024"] * num_rows,
        "particulars": ["Amazon \n AWS"] * num_rows,
        "deposits": ["1000"] * num_rows,
        "withdrawals": [""] * num_rows,
        "balance": ["5000"] * num_rows,
    }
    df = pd.DataFrame(data)

    # Benchmark iterrows
    start = time.time()
    res1 = extract_rows_iterrows(df)
    time_iterrows = time.time() - start

    # Benchmark optimized
    start = time.time()
    res2 = extract_rows_optimized(df)
    time_optimized = time.time() - start

    print(f"Iterrows: {time_iterrows:.4f}s")
    print(f"Optimized (to_dict): {time_optimized:.4f}s")
    print(f"Improvement: {(time_iterrows - time_optimized) / time_iterrows * 100:.2f}%")
    print(f"Results match: {res1 == res2}")
