"""
Bloomberg data pulling via raw blpapi (HERA pattern).

Connection: Parallels Windows VM at 10.211.55.3:8194
DO NOT use xbbg — it has a bug with remote server parameter.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import blpapi
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BBG_HOST = "10.211.55.3"
BBG_PORT = 8194


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


def pull_fundamentals(
    session: blpapi.Session,
    indices: List[Dict],
) -> pd.DataFrame:
    """
    Pull fundamental data for equity indices using bbg_bdp().

    Returns DataFrame: index=name, columns=fundamental fields
    """
    tickers = [idx["ticker"] for idx in indices]
    fields = [
        "BEST_PE_RATIO",
        "EARN_YLD",
        "EST_LTG_EPS_AGGTE",
        "PROF_MARGIN",
        "RETURN_ON_EQY",
        "TOT_DEBT_TO_TOT_EQY",
        "DVD_YLD",
    ]
    df = bbg_bdp(session, tickers, fields)
    ticker_to_name = {idx["ticker"]: idx["name"] for idx in indices}
    df.index = [ticker_to_name.get(t, t) for t in df.index]
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
