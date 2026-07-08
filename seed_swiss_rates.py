#!/usr/bin/env python3
"""One-time seeding of Swiss government-bond yields into master_prices.csv.

Why this exists: `update_master_prices_wide` derives its ticker universe from
the CSV *header* and only refreshes columns that already exist — it cannot
onboard a brand-new ticker. So a new series (Swiss 2Y/10Y/30Y) must be seeded
once, with full history, as new columns. After this runs, the normal daily
updater keeps them fresh automatically.

Run this ON the Bloomberg machine (needs blpapi + a terminal connection):

    python seed_swiss_rates.py --dry-run     # report coverage, write nothing
    python seed_swiss_rates.py                # back up + seed the columns

CONFIRM THE TICKERS on the terminal first. GSWISS10 is used elsewhere in the
repo; the 2Y/30Y generics are assumed to follow the same pattern. If Bloomberg
returns nothing for one, fix SWISS_TICKERS below (e.g. GSWISS02 / GSWISS30).
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.bloomberg import connect_bloomberg, disconnect_bloomberg, bbg_bdh

# label -> ticker; label only used for logging. Order = column append order.
# 2Y is GSWISS02 (zero-padded); "GSWISS2" resolves to a different series.
SWISS_TICKERS = {
    "2Y": "GSWISS02 Index",
    "10Y": "GSWISS10 Index",
    "30Y": "GSWISS30 Index",
}
# A single 2006->today daily request truncates at ~5100 points and silently
# drops the most recent year. The bond slides only look back 12M, so we seed
# recent history (covers YTD + 12M with buffer) in one safe request.
START = "20230101"


def _fmt(v) -> str:
    """Match update_master_prices_wide formatting exactly."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.4f}" if abs(v) < 100 else f"{v:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/Users/larazanella/Library/CloudStorage/Dropbox/"
                    "Tools_In_Construction/ic/master_prices.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return 1

    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    header, subheader, data = lines[0], lines[1], lines[2:]

    existing = {t.strip() for t in header.split(";")}
    already = [t for t in SWISS_TICKERS.values() if t in existing]
    if already:
        print(f"ERROR: these tickers already have columns, refusing to duplicate: {already}")
        return 1

    # date (DD/MM/YYYY) -> row index, to align pulled history onto existing rows
    date_to_row = {}
    for i, row in enumerate(data):
        d = row.split(";", 1)[0].strip()
        if d:
            date_to_row[d] = i

    today = datetime.today().strftime("%Y%m%d")
    session = connect_bloomberg()
    try:
        pulled = {}  # ticker -> {date_str: (close, low, high)}
        for label, ticker in SWISS_TICKERS.items():
            raw = bbg_bdh(session, [ticker], ["PX_LAST", "PX_LOW", "PX_HIGH"], START, today)
            m = {}
            for _, r in raw.iterrows():
                ds = pd.Timestamp(r["date"]).strftime("%d/%m/%Y")
                m[ds] = (r.get("PX_LAST"), r.get("PX_LOW"), r.get("PX_HIGH"))
            pulled[ticker] = m
            hit = sum(1 for ds in m if ds in date_to_row)
            dates = sorted(pd.to_datetime(list(m), format="%d/%m/%Y")) if m else []
            span = f"first={dates[0].date()} last={dates[-1].date()}" if dates else "first=None last=None"
            print(f"{label:4s} {ticker:16s} pulled={len(m):5d} rows, "
                  f"aligned_to_existing_dates={hit:5d}, {span}")
            if not m:
                print(f"  !! NO DATA for {ticker} — fix the ticker symbol before seeding.")
    finally:
        disconnect_bloomberg(session)

    if any(not pulled[t] for t in pulled):
        print("Aborting: at least one ticker returned no data.")
        return 1

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    # Append 3 columns per ticker to header/subheader and every data row.
    new_header = header
    new_sub = subheader
    for ticker in SWISS_TICKERS.values():
        new_header += ";" + ";".join([ticker] * 3)
        new_sub += ";#price;#low;#high"

    new_data = []
    for i, row in enumerate(data):
        d = row.split(";", 1)[0].strip()
        for ticker in SWISS_TICKERS.values():
            c, lo, hi = pulled[ticker].get(d, (None, None, None))
            row += ";" + ";".join([_fmt(c), _fmt(lo), _fmt(hi)])
        new_data.append(row)

    backup = csv_path.with_suffix(f".csv.bak-{datetime.today():%Y%m%d%H%M%S}")
    shutil.copy2(csv_path, backup)
    csv_path.write_text("\n".join([new_header, new_sub, *new_data]) + "\n", encoding="utf-8-sig")
    print(f"\nSeeded {len(SWISS_TICKERS)} Swiss tickers. Backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
