"""
Bloomberg data pulling via raw blpapi (HERA pattern).

Connection: read BLOOMBERG_HOST / BLOOMBERG_PORT from env.
Defaults to the Mac's Parallels VM (10.211.55.3:8194). On the intern PC
where Bloomberg runs locally, set BLOOMBERG_HOST=localhost.

DO NOT use xbbg — it has a bug with remote server parameter.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import blpapi
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BBG_HOST = os.environ.get("BLOOMBERG_HOST", "10.211.55.3")
BBG_PORT = int(os.environ.get("BLOOMBERG_PORT", "8194"))


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect_bloomberg() -> blpapi.Session:
    """Connect to Bloomberg DAPI via Parallels."""
    opts = blpapi.SessionOptions()
    opts.setServerHost(BBG_HOST)
    opts.setServerPort(BBG_PORT)
    session = blpapi.Session(opts)
    if not session.start():
        raise ConnectionError(
            f"Cannot connect to Bloomberg at {BBG_HOST}:{BBG_PORT}. "
            "Is Bloomberg Terminal running in Parallels?"
        )
    if not session.openService("//blp/refdata"):
        raise ConnectionError("Cannot open refdata service.")
    log.info("Bloomberg connected at %s:%d", BBG_HOST, BBG_PORT)
    return session


def disconnect_bloomberg(session: blpapi.Session) -> None:
    """Cleanly stop the Bloomberg session."""
    try:
        session.stop()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Reference Data (BDP — current values)
# ---------------------------------------------------------------------------

def bbg_bdp(
    session: blpapi.Session,
    tickers: List[str],
    fields: List[str],
) -> pd.DataFrame:
    """
    Bloomberg Data Point — reference data for multiple securities.

    Returns DataFrame indexed by ticker, columns = fields.
    Missing values are NaN.
    """
    refdata = session.getService("//blp/refdata")
    request = refdata.createRequest("ReferenceDataRequest")

    for ticker in tickers:
        request.getElement("securities").appendValue(ticker)
    for field in fields:
        request.getElement("fields").appendValue(field)

    session.sendRequest(request)

    rows: Dict[str, Dict[str, float]] = {t: {} for t in tickers}

    while True:
        ev = session.nextEvent(500)
        for msg in ev:
            if msg.messageType() == blpapi.Name("ReferenceDataResponse"):
                sec_data = msg.getElement("securityData")
                for i in range(sec_data.numValues()):
                    sec = sec_data.getValueAsElement(i)
                    ticker_val = sec.getElementAsString("security")
                    field_data = sec.getElement("fieldData")
                    for field in fields:
                        try:
                            rows[ticker_val][field] = field_data.getElementAsFloat(field)
                        except Exception:
                            rows[ticker_val][field] = np.nan
        if ev.eventType() == blpapi.Event.RESPONSE:
            break

    df = pd.DataFrame(rows).T
    df.index.name = "ticker"
    return df


# ---------------------------------------------------------------------------
# Historical Data (BDH — time series)
# ---------------------------------------------------------------------------

def bbg_bdh(
    session: blpapi.Session,
    tickers: List[str],
    fields: List[str],
    start_date: str,
    end_date: str,
    period: str = "DAILY",
) -> pd.DataFrame:
    """
    Bloomberg Data History — daily time-series for multiple securities.

    Returns long-format DataFrame: date, ticker, + one column per field.
    """
    refdata = session.getService("//blp/refdata")
    request = refdata.createRequest("HistoricalDataRequest")

    for ticker in tickers:
        request.getElement("securities").appendValue(ticker)
    for field in fields:
        request.getElement("fields").appendValue(field)

    request.set("startDate", start_date)
    request.set("endDate", end_date)
    request.set("periodicitySelection", period)
    request.set("nonTradingDayFillOption", "ACTIVE_DAYS_ONLY")
    request.set("nonTradingDayFillMethod", "PREVIOUS_VALUE")

    session.sendRequest(request)

    records = []
    while True:
        ev = session.nextEvent(500)
        for msg in ev:
            if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                sec_data = msg.getElement("securityData")
                ticker_val = sec_data.getElementAsString("security")
                field_data_arr = sec_data.getElement("fieldData")
                for i in range(field_data_arr.numValues()):
                    point = field_data_arr.getValueAsElement(i)
                    dt_val = point.getElementAsDatetime("date")
                    row = {"date": dt_val if isinstance(dt_val, datetime) else datetime(dt_val.year, dt_val.month, dt_val.day), "ticker": ticker_val}
                    for field in fields:
                        try:
                            row[field] = point.getElementAsFloat(field)
                        except Exception:
                            row[field] = np.nan
                    records.append(row)
        if ev.eventType() == blpapi.Event.RESPONSE:
            break

    if not records:
        return pd.DataFrame(columns=["date", "ticker"] + fields)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# High-level data pulls
# ---------------------------------------------------------------------------

def pull_prices(
    session: blpapi.Session,
    tickers: List[str],
    master_csv_path: str,
) -> pd.DataFrame:
    """
    Pull close/low/high for all tickers (incremental append to master_prices.csv).

    master_prices.csv long format: date, ticker, close, low, high
    Only pulls dates not already in the CSV.
    Returns the full updated DataFrame.
    """
    master_path = Path(master_csv_path)
    existing: pd.DataFrame

    if master_path.exists():
        try:
            existing = pd.read_csv(master_path, parse_dates=["date"])
            existing["date"] = pd.to_datetime(existing["date"])
            if "ticker" not in existing.columns:
                raise ValueError("Not in tidy format")
        except (ValueError, KeyError):
            log.warning("Existing master_prices.csv is not in tidy format — starting fresh")
            existing = pd.DataFrame(columns=["date", "ticker", "close", "low", "high"])
    else:
        existing = pd.DataFrame(columns=["date", "ticker", "close", "low", "high"])

    # Determine start date per ticker
    today = datetime.today().strftime("%Y%m%d")
    new_rows = []

    for ticker in tickers:
        mask = existing["ticker"] == ticker
        if mask.any():
            last_date = existing.loc[mask, "date"].max()
            start = (last_date + timedelta(days=1)).strftime("%Y%m%d")
        else:
            start = "20060101"

        if start > today:
            log.debug("Ticker %s already up to date", ticker)
            continue

        raw = bbg_bdh(
            session,
            [ticker],
            ["PX_LAST", "PX_LOW", "PX_HIGH"],
            start,
            today,
        )

        if raw.empty:
            log.warning("No data returned for ticker %s (start=%s)", ticker, start)
            continue

        raw = raw.rename(columns={"PX_LAST": "close", "PX_LOW": "low", "PX_HIGH": "high"})
        raw["ticker"] = ticker
        new_rows.append(raw[["date", "ticker", "close", "low", "high"]])

    if new_rows:
        appended = pd.concat([existing] + new_rows, ignore_index=True)
        appended = appended.sort_values(["ticker", "date"]).drop_duplicates(
            subset=["date", "ticker"]
        )
        appended.to_csv(master_path, index=False)
        log.info("Appended %d new rows to %s", sum(len(r) for r in new_rows), master_path)
    else:
        appended = existing
        log.info("No new data to append")

    # Validate: warn if any ticker has a gap > 7 trading days
    for ticker in tickers:
        mask = appended["ticker"] == ticker
        if not mask.any():
            log.warning("Ticker %s has NO data at all", ticker)
            continue
        last = appended.loc[mask, "date"].max()
        gap_days = (pd.Timestamp.today() - last).days
        if gap_days > 10:
            log.warning("Ticker %s last date is %s (%d days ago)", ticker, last.date(), gap_days)

    return appended


def pull_breadth(
    session: blpapi.Session,
    indices: List[Dict],
) -> pd.DataFrame:
    """
    Pull breadth data for equity indices using bbg_bdp().

    indices: list of dicts with 'ticker' and 'name' keys
    Returns DataFrame: index=ticker, columns=breadth field names
    """
    tickers = [idx["ticker"] for idx in indices]
    fields = [
        "PCT_MEMB_PX_GT_50D_MOV_AVG",
        "PCT_MEMB_PX_GT_100D_MOV_AVG",
        "PCT_MEMB_PX_BLW_LWR_BOLL_BAND",
        "PCT_MEMB_PX_ABV_UPPER_BOLL_BAND",
        "PCT_MEM_MACD_SL_SIGNAL_LST_10D",
        "PCT_MEMB_MACD_GT_BASE_LINE_0",
        "PCT_MEM_MACD_BUY_SIGNAL_LST_10D",
        "PCT_MEMB_SIGNAL_GT_BASE_LINE_0",
    ]
    df = bbg_bdp(session, tickers, fields)
    # Map ticker → name
    ticker_to_name = {idx["ticker"]: idx["name"] for idx in indices}
    df.index = [ticker_to_name.get(t, t) for t in df.index]
    return df


FUNDAMENTAL_FIELDS = [
    "BEST_PE_RATIO",          # Forward PE
    "PX_TO_BOOK_RATIO",       # Price / Book
    "EV_TO_T12M_EBITDA",      # EV / EBITDA (trailing 12m)
    "EST_LTG_EPS_AGGTE",      # Long-term EPS growth estimate
    "OPER_MARGIN",            # Operating margin
    "PROF_MARGIN",            # Net income margin
    "BEST_ROE",               # Forward ROE
    "NET_DEBT_TO_EBITDA",     # Leverage
    "BEST_DIV_YLD",           # Dividend yield
]


def pull_fundamentals(
    session: blpapi.Session,
    indices: List[Dict],
) -> pd.DataFrame:
    """
    Pull fundamental data for equity indices using bbg_bdp().

    Returns DataFrame: index=name, columns=FUNDAMENTAL_FIELDS.
    Logs a warning for any NaN values but does not fail.
    """
    tickers = [idx["ticker"] for idx in indices]
    df = bbg_bdp(session, tickers, FUNDAMENTAL_FIELDS)
    ticker_to_name = {idx["ticker"]: idx["name"] for idx in indices}
    df.index = [ticker_to_name.get(t, t) for t in df.index]

    # Log warnings for NaN values
    for name in df.index:
        nans = df.loc[name][df.loc[name].isna()]
        if len(nans):
            log.warning("Fundamental NaN for %s: %s", name, ", ".join(nans.index))

    return df


def pull_market_caps(
    session: blpapi.Session,
    tickers: List[str],
) -> Dict[str, Optional[float]]:
    """
    Pull current market cap using bbg_bdp() with field CUR_MKT_CAP.
    Returns dict: {ticker: market_cap_value}
    """
    df = bbg_bdp(session, tickers, ["CUR_MKT_CAP"])
    return df["CUR_MKT_CAP"].to_dict()


def update_master_prices_wide(
    session: blpapi.Session,
    master_csv_path: str,
) -> int:
    """
    Incrementally append missing price rows to master_prices.csv (wide format).

    master_prices.csv format:
      - Semicolon-separated, European dates (DD/MM/YYYY) in column 0
      - Row 0: ticker names (repeats 3x per ticker: close, low, high)
      - Row 1: field labels (#price, #low, #high)
      - Data rows: date ; close ; low ; high ; close ; low ; high ; ...

    Reads the last date, pulls PX_LAST / PX_LOW / PX_HIGH for all tickers
    from last_date+1 through today, then appends the new rows.

    Returns the number of new rows appended.
    """
    master_path = Path(master_csv_path)
    if not master_path.exists():
        log.warning("master_prices.csv not found at %s — skipping price update", master_path)
        return 0

    # ── Read header to extract ticker list and column order ──────────────────
    with open(master_path, encoding="utf-8-sig") as f:
        header_line = f.readline().rstrip("\n")
        _subheader = f.readline().rstrip("\n")  # skip #price/#low/#high row

    header_cols = header_line.split(";")
    # Column 0 is the date column; every 3 columns after is one ticker (close/low/high)
    # Ticker names appear in columns 1, 4, 7, ... (the #price column for each ticker)
    tickers_ordered = []
    for i in range(1, len(header_cols), 3):
        t = header_cols[i].strip()
        if t:
            tickers_ordered.append(t)

    if not tickers_ordered:
        log.error("Could not extract tickers from master_prices.csv header")
        return 0

    # ── Find last date in CSV ─────────────────────────────────────────────────
    df_dates = pd.read_csv(master_path, sep=";", header=0, skiprows=[1], usecols=[0])
    date_col = df_dates.columns[0]
    df_dates["_d"] = pd.to_datetime(df_dates[date_col], format="%d/%m/%Y", errors="coerce")
    last_date = df_dates["_d"].dropna().max()

    today = datetime.today()
    overwrite_today = last_date.date() == today.date()
    if last_date.date() > today.date():
        log.info("master_prices.csv is already up to date (%s)", last_date.date())
        return 0

    if overwrite_today:
        # Last row is today — re-pull today to pick up markets that closed after
        # the previous run (e.g. US/LatAm when the pipeline ran in the morning).
        start_str = last_date.strftime("%Y%m%d")
        log.info(
            "Re-pulling today (%s) to fill any markets that were still open on last run",
            last_date.date(),
        )
    else:
        start_str = (last_date + timedelta(days=1)).strftime("%Y%m%d")
    end_str = today.strftime("%Y%m%d")
    log.info(
        "Pulling Bloomberg prices for %d tickers from %s to %s",
        len(tickers_ordered), start_str, end_str,
    )

    # ── Pull from Bloomberg (batch of 50 tickers at a time to avoid limits) ──
    BATCH = 50
    all_long: list[pd.DataFrame] = []
    for i in range(0, len(tickers_ordered), BATCH):
        batch = tickers_ordered[i : i + BATCH]
        try:
            raw = bbg_bdh(session, batch, ["PX_LAST", "PX_LOW", "PX_HIGH"], start_str, end_str)
            all_long.append(raw)
        except Exception as e:
            log.warning("Bloomberg BDH failed for batch %d-%d: %s", i, i + BATCH, e)

    if not all_long:
        log.warning("No Bloomberg price data returned")
        return 0

    long_df = pd.concat(all_long, ignore_index=True)
    long_df["date"] = pd.to_datetime(long_df["date"])

    def fmt(v):
        if isinstance(v, float) and np.isnan(v):
            return ""
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.2f}"

    # ── Pivot to wide format matching CSV column order ────────────────────────
    new_dates = sorted(long_df["date"].unique())
    append_lines = []
    overwrite_line: str | None = None

    for dt in new_dates:
        day_data = long_df[long_df["date"] == dt].set_index("ticker")
        date_str = pd.Timestamp(dt).strftime("%d/%m/%Y")
        vals = []
        for ticker in tickers_ordered:
            if ticker in day_data.index:
                row = day_data.loc[ticker]
                close = row.get("PX_LAST", np.nan)
                low   = row.get("PX_LOW",  np.nan)
                high  = row.get("PX_HIGH", np.nan)
            else:
                close = low = high = np.nan
            vals.extend([fmt(close), fmt(low), fmt(high)])

        line = date_str + ";" + ";".join(vals)
        if overwrite_today and pd.Timestamp(dt).date() == last_date.date():
            overwrite_line = line
        else:
            append_lines.append(line)

    # ── Overwrite today's row if we re-pulled it ──────────────────────────────
    if overwrite_line is not None:
        with open(master_path, encoding="utf-8-sig") as f:
            all_lines = f.readlines()
        today_str = last_date.strftime("%d/%m/%Y")
        for idx in range(len(all_lines) - 1, -1, -1):
            if all_lines[idx].startswith(today_str):
                all_lines[idx] = overwrite_line + "\n"
                log.info("Overwrote today's row (%s) with fresh Bloomberg data", today_str)
                break
        with open(master_path, "w", encoding="utf-8-sig") as f:
            f.writelines(all_lines)

    # ── Append any genuinely new rows ─────────────────────────────────────────
    if append_lines:
        with open(master_path, "a") as f:
            for line in append_lines:
                f.write("\n" + line)

    n_written = (1 if overwrite_line else 0) + len(append_lines)
    log.info("Updated %d price rows in %s", n_written, master_path)
    return n_written
