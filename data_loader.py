"""
Data loader for IC Technical presentation.

Handles loading price data from master_prices.csv which has:
- Semicolon separator
- Multi-row header (row 1 = tickers, row 2 = #price/#low/#high)
- European dates (DD/MM/YYYY) in first column
"""

import pandas as pd
from pathlib import Path
from datetime import date
from typing import Optional, Tuple


def load_prices_from_csv(
    csv_path: Path,
    data_as_of: Optional[date] = None
) -> pd.DataFrame:
    """
    Load price data from master_prices.csv.

    Parameters
    ----------
    csv_path : Path
        Path to master_prices.csv
    data_as_of : date, optional
        Filter data up to this date. If None, use all data.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'Date' column and price columns for each ticker.
        Column names are the ticker names from row 1 (e.g., "SPX Index").
    """
    # Read CSV with semicolon separator, skip the second header row
    # Row 0 = tickers, Row 1 = #price/#low/#high indicators
    df = pd.read_csv(csv_path, sep=';', header=0, skiprows=[1])

    # First column is the date
    date_col = df.columns[0]

    # Parse European dates (DD/MM/YYYY)
    df['Date'] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')

    # Drop the original date column if it's not named 'Date'
    if date_col != 'Date':
        df = df.drop(columns=[date_col])

    # Filter by date if specified
    if data_as_of is not None:
        df = df[df['Date'] <= pd.Timestamp(data_as_of)]

    # Sort by date and reset index
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    return df


def get_max_date_from_csv(csv_path: Path) -> date:
    """
    Get the maximum (most recent) date from master_prices.csv.

    Parameters
    ----------
    csv_path : Path
        Path to master_prices.csv

    Returns
    -------
    date
        The most recent date in the CSV.
    """
    # Read just enough to get dates - skip second header row
    df = pd.read_csv(csv_path, sep=';', header=0, skiprows=[1], usecols=[0])

    date_col = df.columns[0]

    # Parse European dates
    df['Date'] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Date'])

    return df['Date'].max().date()


def get_price_series(
    df_prices: pd.DataFrame,
    ticker: str
) -> Optional[pd.Series]:
    """
    Extract a price series for a specific ticker.

    Parameters
    ----------
    df_prices : pd.DataFrame
        DataFrame from load_prices_from_csv()
    ticker : str
        Ticker name (e.g., "SPX Index", "GCA Comdty")

    Returns
    -------
    pd.Series or None
        Price series indexed by position, or None if ticker not found.
    """
    # Try exact match first
    if ticker in df_prices.columns:
        prices = pd.to_numeric(df_prices[ticker], errors='coerce')
        return prices.dropna()

    # Try case-insensitive match
    ticker_upper = ticker.upper()
    for col in df_prices.columns:
        if col.upper() == ticker_upper:
            prices = pd.to_numeric(df_prices[col], errors='coerce')
            return prices.dropna()

    return None


__all__ = [
    'load_prices_from_csv',
    'get_max_date_from_csv',
    'get_price_series',
]
