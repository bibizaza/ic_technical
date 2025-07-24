"""
ytd_perf/loader_update.py

This module contains helper functions for loading raw price data and
parameters from the consolidated Excel workbook.  Keeping data I/O in
one place avoids circular imports and makes it easy to swap out or extend
the loading logic later (for example, to add caching or error handling).
"""

import pandas as pd
from typing import Tuple

def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw price and parameter data from the given Excel file.

    Parameters
    ----------
    file_path : str
        Path to the Excel workbook that contains the 'data_prices' and 'parameters' sheets.

    Returns
    -------
    tuple
        A pair `(prices_df, params_df)` where:

        - `prices_df` is a DataFrame of price data with a 'Date' column
          and one column per ticker.  The first metadata row (`#price`) is
          removed and the date column is parsed to datetime.
        - `params_df` is a DataFrame of ticker metadata (ticker, name,
          asset class, etc.) from the 'parameters' sheet.
    """
    # Read raw price data; data_prices is assumed to have a header row (#price)
    raw_prices = pd.read_excel(file_path, sheet_name="data_prices")
    if raw_prices.empty:
        raise ValueError("The 'data_prices' sheet is empty or missing.")
    # Drop the first row (metadata) and reset index
    prices_df = raw_prices.iloc[1:].copy()
    # Rename the first column to 'Date' and convert it to datetime
    date_col_name = prices_df.columns[0]
    prices_df.rename(columns={date_col_name: "Date"}, inplace=True)
    prices_df["Date"] = pd.to_datetime(prices_df["Date"])
    # Read parameters sheet
    params_df = pd.read_excel(file_path, sheet_name="parameters")
    return prices_df, params_df
