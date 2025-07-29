"""Data loader for YTD performance modules.

This module provides a helper to load consolidated price and parameter
data from an Excel workbook.  It is designed to mirror the behaviour
expected by the Streamlit application, returning a pair of
``(prices_df, params_df)``.  The ``prices_df`` DataFrame contains a
``Date`` column and one column per ticker, while ``params_df``
contains the contents of the ``parameters`` sheet.

Although earlier versions of the application loaded data via this
function, the YTD modules now perform their own loading and price
mode adjustments internally.  This loader is retained for
compatibility with the Streamlit ``YTD Update`` page.
"""

from __future__ import annotations

import pandas as pd
from typing import Tuple


def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load consolidated price and parameter data from an Excel file.

    Parameters
    ----------
    file_path : str
        Path to the Excel workbook containing a ``data_prices`` sheet
        and a ``parameters`` sheet.

    Returns
    -------
    tuple
        A two‑tuple ``(prices_df, params_df)``.  ``prices_df`` is a
        DataFrame containing a ``Date`` column and one column per
        ticker.  ``params_df`` is the DataFrame parsed from the
        ``parameters`` sheet.
    """
    params_df = pd.read_excel(file_path, sheet_name="parameters")
    df = pd.read_excel(file_path, sheet_name="data_prices")
    # Drop header row
    df = df.drop(index=0)
    # Remove rows with 'DATES' marker
    df = df[df[df.columns[0]] != "DATES"]
    df.loc[:, "Date"] = pd.to_datetime(df[df.columns[0]], errors="coerce")
    # Drop the first column which held date strings
    df = df.drop(columns=[df.columns[0]])
    # Convert price columns to numeric
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    return df, params_df