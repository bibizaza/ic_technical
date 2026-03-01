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
import tempfile
import os


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


def create_temp_excel_from_csv(
    csv_path: Path,
    source_excel_path: Path,
    data_as_of: Optional[date] = None
) -> Path:
    """
    Create a temporary Excel file with data_prices sheet from master_prices.csv.

    The chart generation functions in technical_analysis/ expect an Excel file
    with a 'data_prices' sheet in a specific format:
    - Row 0: header row (gets dropped by df.drop(index=0))
    - First column: dates
    - Other columns: ticker prices

    This also copies other sheets (breadth, fundamental, data_perf) from the
    source Excel file if they exist.

    Parameters
    ----------
    csv_path : Path
        Path to master_prices.csv
    source_excel_path : Path
        Path to the source IC Excel file (for breadth/fundamental sheets)
    data_as_of : date, optional
        Filter data up to this date. If None, use all data.

    Returns
    -------
    Path
        Path to the temporary Excel file.
    """
    # Load price data from CSV
    df_prices = load_prices_from_csv(csv_path, data_as_of)

    # Create data_prices DataFrame in expected format
    # Row 0 will be a header row that gets dropped
    # First column is dates, rest are price columns

    # Create header row (will be dropped by chart functions)
    header_row = {'Date': 'DATES'}  # Will be filtered out by df[df[df.columns[0]] != "DATES"]
    for col in df_prices.columns:
        if col != 'Date':
            header_row[col] = '#price'

    # Create DataFrame with header row first
    df_excel = pd.DataFrame([header_row])

    # Append the actual data
    # Convert Date column to string format expected by chart functions
    df_data = df_prices.copy()
    df_data['Date'] = df_data['Date'].dt.strftime('%Y-%m-%d')
    df_excel = pd.concat([df_excel, df_data], ignore_index=True)

    # Create temporary Excel file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.xlsx', prefix='ic_temp_')
    os.close(temp_fd)

    # Write to Excel with openpyxl engine
    with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
        # Write data_prices sheet
        df_excel.to_excel(writer, sheet_name='data_prices', index=False)

        # Copy other sheets from source Excel if they exist
        try:
            xl = pd.ExcelFile(source_excel_path)
            sheets_to_copy = ['breadth', 'fundamental', 'data_perf', 'mars_score',
                             'data_technical_score', 'data_momentum_score']

            for sheet in sheets_to_copy:
                if sheet in xl.sheet_names:
                    df_sheet = pd.read_excel(xl, sheet_name=sheet)
                    df_sheet.to_excel(writer, sheet_name=sheet, index=False)
        except Exception as e:
            print(f"[data_loader] Warning: Could not copy sheets from {source_excel_path}: {e}")

    return Path(temp_path)


def compute_ytd_performance(df_prices: pd.DataFrame, ticker: str) -> Optional[float]:
    """
    Compute YTD performance for a ticker.

    Parameters
    ----------
    df_prices : pd.DataFrame
        DataFrame from load_prices_from_csv()
    ticker : str
        Ticker name (e.g., "SPX Index")

    Returns
    -------
    float or None
        YTD performance as a percentage, or None if cannot compute.
    """
    if 'Date' not in df_prices.columns:
        return None

    # Find ticker column (case-insensitive)
    ticker_col = None
    if ticker in df_prices.columns:
        ticker_col = ticker
    else:
        ticker_upper = ticker.upper()
        for col in df_prices.columns:
            if col.upper() == ticker_upper:
                ticker_col = col
                break

    if ticker_col is None:
        return None

    # Get price series
    df = df_prices[['Date', ticker_col]].copy()
    df[ticker_col] = pd.to_numeric(df[ticker_col], errors='coerce')
    df = df.dropna()

    if len(df) < 2:
        return None

    # Get current year
    current_year = df['Date'].max().year

    # Find last price of previous year
    df_prev_year = df[df['Date'].dt.year == current_year - 1]
    if df_prev_year.empty:
        # If no previous year data, use first available price
        first_price = df[ticker_col].iloc[0]
    else:
        first_price = df_prev_year[ticker_col].iloc[-1]

    # Get most recent price
    last_price = df[ticker_col].iloc[-1]

    if first_price == 0 or pd.isna(first_price):
        return None

    # Calculate YTD return as percentage
    ytd_return = ((last_price / first_price) - 1) * 100
    return ytd_return


def create_data_perf_sheet(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Create a data_perf DataFrame with YTD performance for all instruments.

    Parameters
    ----------
    df_prices : pd.DataFrame
        DataFrame from load_prices_from_csv()

    Returns
    -------
    pd.DataFrame
        DataFrame with YTD performance values for equity, commodity, and crypto.
    """
    # Map from ticker column names to data_perf column names
    ticker_to_perf = {
        # Equity
        'SPX Index': 'SPX',
        'DAX Index': 'STOXX',  # DAX represents STOXX
        'SASEIDX Index': 'TASI',
        'NKY Index': 'NKY',
        'SENSEX Index': 'Sensex',
        'SHSZ300 Index': 'Hang Seng',  # CSI represents HK
        'IBOV Index': 'IBOV',
        # Commodities
        'GCA Comdty': 'Gold',
        'SIA Comdty': 'Silver',
        'CL1 Comdty': 'Oil',
        'LP1 Comdty': 'Copper',
        # Crypto
        'XBTUSD Curncy': 'Bitcoin',
        'XETUSD Curncy': 'Ethereum',
        'XSOUSD Curncy': 'Solana',
    }

    perf_data = {}
    for ticker, perf_col in ticker_to_perf.items():
        ytd = compute_ytd_performance(df_prices, ticker)
        perf_data[perf_col] = [ytd if ytd is not None else 0]

    return pd.DataFrame(perf_data)


__all__ = [
    'load_prices_from_csv',
    'get_max_date_from_csv',
    'get_price_series',
    'create_temp_excel_from_csv',
    'compute_ytd_performance',
    'create_data_perf_sheet',
]
