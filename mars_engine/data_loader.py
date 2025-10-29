# -*- coding: utf-8 -*-
"""
Data loader for MARS momentum scoring.

Loads and transforms the Excel data_prices sheet into the format expected
by mars_lite_scorer.
"""

from __future__ import annotations
from typing import Union
import pathlib
import pandas as pd


# Mapping from Excel column names to MARS scorer names
TICKER_NAME_MAP = {
    # Equity
    "SPX Index": "SPX",
    "SHSZ300 Index": "CSI",
    "NKY Index": "NKY Index",
    "SASEIDX Index": "SASEIDX Index",
    "SENSEX Index": "SENSEX Index",
    "DAX Index": "DAX Index",
    "SMI Index": "SMI Index",
    "IBOV Index": "IBOV Index",
    "MEXBOL Index": "MEXBOL Index",
    # MARS Peers
    "CCMP Index": "CCMP Index",
    "SXXP Index": "SXXP Index",
    "UKX Index": "UKX Index",
    "SMI Index": "SMI Index",
    "HSI Index": "HSI Index",
    "MXWO Index": "MXWO Index",
    "USGG10YR Index": "USGG10YR Index",
    "GECU10YR Index": "GECU10YR Index",
    # Commodities
    "GCA Comdty": "GCA Comdty",
    "CL1 Comdty": "CL1 Comdty",
    # Currencies
    "DXY Curncy": "DXY Curncy",
    # Crypto
    "XBTUSD Curncy": "XBTUSD Curncy",
}


def load_prices_for_mars(excel_obj_or_path: Union[str, pathlib.Path, pd.ExcelFile]) -> pd.DataFrame:
    """
    Load and transform the data_prices sheet for MARS scoring.

    Parameters
    ----------
    excel_obj_or_path : str, pathlib.Path, or pd.ExcelFile
        Excel file path or already-opened Excel file object

    Returns
    -------
    pd.DataFrame
        DataFrame with Date index and columns for each ticker.
        Column names are transformed to match MARS expectations.
        For SPX, also creates SPX_high and SPX_low columns if not present.
    """
    # Read data_prices sheet
    df = pd.read_excel(excel_obj_or_path, sheet_name="data_prices")

    # Clean: drop first row and filter out "DATES" rows
    df = df.drop(index=0)
    df = df[df[df.columns[0]] != "DATES"]

    # Parse date column
    df["Date"] = pd.to_datetime(df[df.columns[0]], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.set_index("Date")
    df = df.sort_index()

    # Select only ticker columns (skip the original date column)
    ticker_cols = [col for col in df.columns if col != df.columns[0]]
    df = df[ticker_cols]

    # Rename columns to match MARS expectations
    df = df.rename(columns=TICKER_NAME_MAP)

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create high/low columns for targets if they don't exist
    # For SPX: create SPX_high and SPX_low as ±1% of SPX
    if "SPX" in df.columns:
        if "SPX_high" not in df.columns:
            df["SPX_high"] = df["SPX"] * 1.01
        if "SPX_low" not in df.columns:
            df["SPX_low"] = df["SPX"] * 0.99

    # For CSI: create CSI_high and CSI_low as ±1% of CSI
    if "CSI" in df.columns:
        if "CSI_high" not in df.columns:
            df["CSI_high"] = df["CSI"] * 1.01
        if "CSI_low" not in df.columns:
            df["CSI_low"] = df["CSI"] * 0.99

    return df
