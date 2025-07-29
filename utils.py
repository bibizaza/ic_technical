"""
Utility functions for price data adjustments and common helpers.

This module defines helper functions that can be reused across
technical analysis modules (e.g. SPX, other indices) to standardise
the handling of price data, particularly with respect to choosing
between the last intraday price and the previous day's closing price.

Functions
---------
adjust_prices_for_mode(df, price_mode)
    Given a price DataFrame with a 'Date' column, return a copy of the
    DataFrame filtered according to the requested price mode and the
    effective date being used.  "Last Price" uses the most recent row;
    "Last Close" drops the most recent row if the last date equals
    today's date.
"""
from __future__ import annotations

import pandas as pd
from typing import Tuple, Optional

def adjust_prices_for_mode(
    df: pd.DataFrame, price_mode: str = "Last Price"
) -> Tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    """Adjust a price DataFrame according to the specified price mode.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least a 'Date' column.  Rows should be
        sorted by date in ascending order.  The function will not
        mutate the original DataFrame.
    price_mode : str, default "Last Price"
        One of "Last Price" or "Last Close".  "Last Price" returns
        the DataFrame unchanged and sets the used date to the maximum
        date in ``df``.  "Last Close" checks whether the maximum date
        equals today's date; if so, it removes all rows with that date
        and sets the used date to the next most recent date.  If the
        maximum date is not today or there is only one unique date,
        the DataFrame is returned unchanged.

    Returns
    -------
    tuple
        A two-tuple ``(df_out, used_date)``, where ``df_out`` is the
        adjusted DataFrame and ``used_date`` is the maximum date in
        ``df_out`` (or ``None`` if ``df`` is empty).
    """
    if df is None or df.empty or "Date" not in df.columns:
        return df.copy() if df is not None else pd.DataFrame(), None
    df_out = df.copy()
    df_out["Date"] = pd.to_datetime(df_out["Date"], errors="coerce")
    # Drop rows with NaT dates
    df_out = df_out.dropna(subset=["Date"])
    if df_out.empty:
        return df_out, None
    # Sort by date if not already sorted
    df_out = df_out.sort_values("Date").reset_index(drop=True)
    # Determine mode
    if price_mode and price_mode.lower() == "last close":
        # Unique sorted dates
        unique_dates = (
            df_out["Date"].dropna().dt.normalize().sort_values().unique()
        )
        if len(unique_dates) >= 2:
            last_date = unique_dates[-1]
            penultimate_date = unique_dates[-2]
            today = pd.Timestamp.today().normalize()
            # If the last date matches today's date, drop it
            if last_date == today:
                df_out = df_out[df_out["Date"].dt.normalize() < last_date]
    # The used date is the maximum date in the adjusted DataFrame
    used_date = df_out["Date"].max() if not df_out.empty else None
    return df_out, used_date