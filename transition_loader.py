"""
Read transition sheet from Excel to pre-populate app fields.
"""
import pandas as pd
from typing import Dict, Optional, Any
from datetime import datetime


def read_transition_sheet(excel_path) -> Dict[str, Dict[str, Any]]:
    """
    Read the 'transition' sheet from Excel to pre-populate app fields.

    Expected structure:
    - Column A (row 2+): Ticker (e.g., "SPX INDEX", "GCA COMDTY")
    - Column B (row 2+): Last week DMAS (float)
    - Column C (row 2+): Anchor date for regression channel (date)
    - Column D (row 2+): Assessment (string)
    - Column E (row 2+): Subtitle (string)

    Parameters
    ----------
    excel_path : str or Path
        Path to the Excel file

    Returns
    -------
    dict
        Dictionary keyed by ticker (uppercase, stripped) with values:
        {
            "last_week_dmas": float or None,
            "anchor_date": pd.Timestamp or None,
            "assessment": str or None,
            "subtitle": str or None
        }
    """
    try:
        df = pd.read_excel(excel_path, sheet_name="transition")
    except Exception as e:
        print(f"Warning: Could not read 'transition' sheet: {e}")
        return {}

    # Drop the first row (header row 1, data starts at row 2)
    if len(df) > 0:
        df = df.iloc[1:]  # Skip row 0 (which is row 1 in Excel)

    result = {}

    for _, row in df.iterrows():
        # Column A: Ticker
        ticker = row.iloc[0] if len(row) > 0 else None
        if pd.isna(ticker) or str(ticker).strip() == "":
            continue

        ticker = str(ticker).strip().upper()

        # Column B: Last week DMAS
        last_week_dmas = None
        if len(row) > 1 and not pd.isna(row.iloc[1]):
            try:
                last_week_dmas = float(row.iloc[1])
            except (ValueError, TypeError):
                pass

        # Column C: Anchor date
        anchor_date = None
        if len(row) > 2 and not pd.isna(row.iloc[2]):
            try:
                anchor_date = pd.to_datetime(row.iloc[2])
            except Exception:
                pass

        # Column D: Assessment
        assessment = None
        if len(row) > 3 and not pd.isna(row.iloc[3]):
            assessment = str(row.iloc[3]).strip()

        # Column E: Subtitle
        subtitle = None
        if len(row) > 4 and not pd.isna(row.iloc[4]):
            subtitle = str(row.iloc[4]).strip()

        result[ticker] = {
            "last_week_dmas": last_week_dmas,
            "anchor_date": anchor_date,
            "assessment": assessment,
            "subtitle": subtitle
        }

    print(f"Loaded transition data for {len(result)} tickers")
    return result


def get_ticker_key_from_ticker(ticker: str) -> Optional[str]:
    """
    Map Excel ticker to app ticker_key.

    Parameters
    ----------
    ticker : str
        Ticker from Excel (e.g., "SPX INDEX", "GCA COMDTY")

    Returns
    -------
    str or None
        The ticker_key used in the app (e.g., "spx", "gold")
    """
    ticker = ticker.upper().strip()

    # Equity tickers
    ticker_map = {
        "SPX INDEX": "spx",
        "SHSZ300 INDEX": "csi",
        "NKY INDEX": "nikkei",
        "SASEIDX INDEX": "tasi",
        "SENSEX INDEX": "sensex",
        "DAX INDEX": "dax",
        "SMI INDEX": "smi",
        "IBOV INDEX": "ibov",
        "MEXBOL INDEX": "mexbol",
        # Commodity tickers
        "GCA COMDTY": "gold",
        "SIA COMDTY": "silver",
        "XPT COMDTY": "platinum",
        "XPD CURNCY": "palladium",
        "CL1 COMDTY": "oil",
        "LP1 COMDTY": "copper",
        # Crypto tickers
        "XBTUSD CURNCY": "bitcoin",
        "XETUSD CURNCY": "ethereum",
        "XRPUSD CURNCY": "ripple",
        "XSOUSD CURNCY": "solana",
        "XBIUSD CURNCY": "binance",
    }

    return ticker_map.get(ticker)


def apply_transition_data_to_session_state(transition_data: Dict[str, Dict[str, Any]], session_state) -> None:
    """
    Apply transition data to Streamlit session state.

    This pre-populates:
    - Last week DMAS
    - Anchor date for regression channel
    - Assessment
    - Subtitle

    Parameters
    ----------
    transition_data : dict
        Dictionary from read_transition_sheet()
    session_state : streamlit.session_state
        Streamlit session state object
    """
    for ticker, data in transition_data.items():
        ticker_key = get_ticker_key_from_ticker(ticker)
        if not ticker_key:
            print(f"Warning: Unknown ticker '{ticker}' in transition sheet")
            continue

        # Apply last week DMAS
        if data["last_week_dmas"] is not None:
            session_state[f"{ticker_key}_last_week_avg"] = data["last_week_dmas"]

        # Apply anchor date (and enable regression channel)
        if data["anchor_date"] is not None:
            session_state[f"{ticker_key}_anchor_date"] = data["anchor_date"].date()
            session_state[f"{ticker_key}_anchor"] = data["anchor_date"]
            session_state[f"{ticker_key}_enable_channel"] = True

        # Apply assessment
        if data["assessment"] is not None:
            session_state[f"{ticker_key}_assessment"] = data["assessment"]

        # Apply subtitle
        if data["subtitle"] is not None:
            session_state[f"{ticker_key}_subtitle"] = data["subtitle"]

    print(f"Applied transition data to session state for {len(transition_data)} tickers")
