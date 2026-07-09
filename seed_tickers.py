#!/usr/bin/env python3
"""Seed new ticker columns (with history) into master_prices.csv.

The daily updater (update_master_prices_wide) reads its ticker universe from the
CSV header and only refreshes columns that already exist -- it cannot onboard a
new ticker. So any new series must be seeded once, as new columns, with enough
history for the slide horizons (>=12M). After this runs, the daily updater keeps
them fresh automatically.

Run ON the Bloomberg machine (needs blpapi + a terminal):

    python seed_tickers.py --dry-run "SFC14T Index"
    python seed_tickers.py "SFC14T Index"                 # backs up + seeds

Notes:
- A single very long request (~5100+ daily points) silently drops the most
  recent year, so we default START to 2023 -- enough for YTD + 12M with buffer.
- Existing rows before a ticker's first data are left blank.
"""
from __future__ import annotations
import argparse, shutil
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from pipeline.bloomberg import connect_bloomberg, disconnect_bloomberg, bbg_bdh

DEFAULT_CSV = ("/Users/larazanella/Library/CloudStorage/Dropbox/"
               "Tools_In_Construction/ic/master_prices.csv")


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.4f}" if abs(v) < 100 else f"{v:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="+", help="Bloomberg tickers, e.g. 'SFC14T Index'")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--start", default="20230101")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found"); return 1

    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    header, subheader, data = lines[0], lines[1], lines[2:]
    existing = {t.strip() for t in header.split(";")}
    dup = [t for t in args.tickers if t in existing]
    if dup:
        print(f"ERROR: already present, refusing to duplicate: {dup}"); return 1

    date_to_row = {}
    for i, row in enumerate(data):
        d = row.split(";", 1)[0].strip()
        if d:
            date_to_row[d] = i

    today = datetime.today().strftime("%Y%m%d")
    session = connect_bloomberg()
    try:
        pulled = {}
        for t in args.tickers:
            raw = bbg_bdh(session, [t], ["PX_LAST", "PX_LOW", "PX_HIGH"], args.start, today)
            m = {}
            for _, r in raw.iterrows():
                ds = pd.Timestamp(r["date"]).strftime("%d/%m/%Y")
                m[ds] = (r.get("PX_LAST"), r.get("PX_LOW"), r.get("PX_HIGH"))
            pulled[t] = m
            hit = sum(1 for ds in m if ds in date_to_row)
            dts = sorted(pd.to_datetime(list(m), format="%d/%m/%Y")) if m else []
            span = f"{dts[0].date()}->{dts[-1].date()}" if dts else "none"
            print(f"{t:16s} pulled={len(m):5d} aligned={hit:5d} {span}")
            if not m:
                print(f"  !! NO DATA for {t} -- fix the ticker before seeding.")
    finally:
        disconnect_bloomberg(session)

    if any(not pulled[t] for t in pulled):
        print("Aborting: a ticker returned no data."); return 1
    if args.dry_run:
        print("\n--dry-run: nothing written."); return 0

    new_header, new_sub = header, subheader
    for t in args.tickers:
        new_header += ";" + ";".join([t] * 3)
        new_sub += ";#price;#low;#high"
    new_data = []
    for row in data:
        d = row.split(";", 1)[0].strip()
        for t in args.tickers:
            c, lo, hi = pulled[t].get(d, (None, None, None))
            row += ";" + ";".join([_fmt(c), _fmt(lo), _fmt(hi)])
        new_data.append(row)

    backup = csv_path.with_suffix(f".csv.bak-{datetime.today():%Y%m%d%H%M%S}")
    shutil.copy2(csv_path, backup)
    csv_path.write_text("\n".join([new_header, new_sub, *new_data]) + "\n", encoding="utf-8-sig")
    print(f"\nSeeded {len(args.tickers)} ticker(s). Backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
