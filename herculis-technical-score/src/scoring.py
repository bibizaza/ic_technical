"""
Main scoring module to compute Technical Score from price data.

This module combines all technical indicators into a final score using
configured weights, replicating BQL logic with identified fixes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import sys
sys.path.append('..')

# Import configuration
from ..config import (
    SMA_PERIODS, EMA_PERIODS, RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
    DMI_PERIOD, ADX_THRESHOLD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    STOCH_K_PERIOD, STOCH_D_PERIOD, STOCH_NEUTRAL_LOW, STOCH_NEUTRAL_HIGH,
    MAE_PERIOD, MAE_ENVELOPE_PCT, MA_OSC_PERIOD,
    PSAR_AF_START, PSAR_AF_INCREMENT, PSAR_AF_MAX,
    WEIGHTS, MIN_LOOKBACK_DAYS
)

# Import indicators
from .indicators import (
    compute_sma, compute_ema, score_ma,
    compute_rsi, score_rsi,
    compute_dmi, score_dmi,
    compute_parabolic_sar, score_parabolic,
    compute_macd, score_macd,
    compute_stochastics, score_stochastics,
    compute_mae, score_mae
)

from .utils import get_last_value, get_value_n_periods_ago


def compute_technical_score(
    prices_df: pd.DataFrame,
    ticker: str,
    include_components: bool = True
) -> Dict:
    """
    Compute technical score for a single ticker.

    Parameters
    ----------
    prices_df : pd.DataFrame
        DataFrame with columns ['Date', 'Price', 'High', 'Low']
        Must have at least MIN_LOOKBACK_DAYS rows
    ticker : str
        Asset identifier (for output)
    include_components : bool, default True
        Whether to include component scores and raw indicators in output

    Returns
    -------
    dict
        {
            'ticker': str,
            'technical_score': float,  # 0-100
            'components': dict,        # Component scores (if include_components=True)
            'raw_indicators': dict     # Raw indicator values (if include_components=True)
        }

    Raises
    ------
    ValueError
        If insufficient data or missing required columns
    """
    # Validate input
    required_cols = ['Date', 'Price', 'High', 'Low']
    missing = [col for col in required_cols if col not in prices_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(prices_df) < MIN_LOOKBACK_DAYS:
        raise ValueError(
            f"Insufficient data: need at least {MIN_LOOKBACK_DAYS} days, "
            f"got {len(prices_df)}"
        )

    # Sort by date and reset index
    df = prices_df.sort_values('Date').reset_index(drop=True)

    # Extract series
    prices = df['Price']
    highs = df['High']
    lows = df['Low']

    # Get current and 1-week-ago prices
    current_price = get_last_value(prices)
    price_1w = get_value_n_periods_ago(prices, 5)

    # Dictionary to store component scores and raw indicators
    components = {}
    raw_indicators = {}

    # ========================================================================
    # 1. SIMPLE MOVING AVERAGES
    # ========================================================================
    sma_values = {}
    for period in SMA_PERIODS:
        sma = compute_sma(prices, period)
        sma_val = get_last_value(sma)
        if sma_val is not None:
            sma_values[period] = sma_val

    components['sma'] = score_ma(current_price, sma_values, "SMA")
    raw_indicators['sma'] = sma_values

    # ========================================================================
    # 2. EXPONENTIAL MOVING AVERAGES
    # ========================================================================
    ema_values = {}
    for period in EMA_PERIODS:
        ema = compute_ema(prices, period)
        ema_val = get_last_value(ema)
        if ema_val is not None:
            ema_values[period] = ema_val

    components['ema'] = score_ma(current_price, ema_values, "EMA")
    raw_indicators['ema'] = ema_values

    # ========================================================================
    # 3. RSI (Contrarian)
    # ========================================================================
    rsi = compute_rsi(prices, RSI_PERIOD)
    rsi_value = get_last_value(rsi)
    components['rsi'] = score_rsi(rsi_value, RSI_OVERBOUGHT, RSI_OVERSOLD)
    raw_indicators['rsi'] = rsi_value

    # ========================================================================
    # 4. DMI / ADX
    # ========================================================================
    dmi_data = compute_dmi(highs, lows, prices, DMI_PERIOD)
    components['dmi'] = score_dmi(dmi_data, ADX_THRESHOLD)
    raw_indicators['dmi'] = dmi_data

    # ========================================================================
    # 5. PARABOLIC SAR (Fixed Logic)
    # ========================================================================
    psar = compute_parabolic_sar(highs, lows, prices, PSAR_AF_START, PSAR_AF_INCREMENT, PSAR_AF_MAX)
    sar_current = get_last_value(psar)
    sar_1w = get_value_n_periods_ago(psar, 5)
    components['parabolic'] = score_parabolic(current_price, price_1w, sar_current, sar_1w)
    raw_indicators['parabolic_sar'] = {'current': sar_current, '1w_ago': sar_1w}

    # ========================================================================
    # 6. MACD
    # ========================================================================
    macd_data = compute_macd(prices, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    components['macd'] = score_macd(macd_data)
    raw_indicators['macd'] = macd_data

    # ========================================================================
    # 7. STOCHASTICS
    # ========================================================================
    stoch_data = compute_stochastics(highs, lows, prices, STOCH_K_PERIOD, STOCH_D_PERIOD)
    components['stochastics'] = score_stochastics(stoch_data, STOCH_NEUTRAL_LOW, STOCH_NEUTRAL_HIGH)
    raw_indicators['stochastics'] = stoch_data

    # ========================================================================
    # 8. MAE (Contrarian)
    # ========================================================================
    mae_data = compute_mae(prices, MAE_PERIOD, MAE_ENVELOPE_PCT, MA_OSC_PERIOD)
    components['mae'] = score_mae(current_price, mae_data)
    raw_indicators['mae'] = mae_data

    # ========================================================================
    # COMBINE COMPONENTS INTO FINAL SCORE
    # ========================================================================
    weighted_score = 0.0
    for component, weight in WEIGHTS.items():
        weighted_score += components[component] * weight

    # Convert to 0-100 scale
    technical_score = weighted_score * 100

    # Build result
    result = {
        'ticker': ticker,
        'technical_score': round(technical_score, 2),
    }

    if include_components:
        result['components'] = {k: round(v, 4) for k, v in components.items()}
        result['raw_indicators'] = raw_indicators

    return result


def load_price_data_from_excel(
    excel_path: Path,
    ticker: str,
    sheet_name: str = "data_prices"
) -> pd.DataFrame:
    """
    Load price data for a ticker from Excel file.

    Parameters
    ----------
    excel_path : Path
        Path to Excel file
    ticker : str
        Ticker symbol to load
    sheet_name : str, default "data_prices"
        Name of sheet containing price data

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Date', 'Price', 'High', 'Low']

    Notes
    -----
    Assumes Excel structure:
    - First row is header (text like "DATES", "#price", etc.)
    - First column is dates
    - Subsequent columns are tickers
    """
    # Read the sheet
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Drop first row (text headers)
    df = df.drop(index=0)

    # First column is dates
    date_col = df.columns[0]
    df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['Date'])

    # Find the ticker column
    if ticker not in df.columns:
        raise ValueError(f"Ticker '{ticker}' not found in {sheet_name}")

    # For now, assume Price = Close (if High/Low available, use them)
    # Otherwise duplicate Price for High/Low
    df['Price'] = pd.to_numeric(df[ticker], errors='coerce')

    # Check if High and Low columns exist (format: "ticker_High", "ticker_Low")
    high_col = f"{ticker}_High"
    low_col = f"{ticker}_Low"

    if high_col in df.columns and low_col in df.columns:
        df['High'] = pd.to_numeric(df[high_col], errors='coerce')
        df['Low'] = pd.to_numeric(df[low_col], errors='coerce')
    else:
        # Use Price as High/Low (approximate)
        df['High'] = df['Price']
        df['Low'] = df['Price']

    # Select relevant columns
    result = df[['Date', 'Price', 'High', 'Low']].copy()
    result = result.dropna(subset=['Price'])

    return result


def compute_all_scores(
    excel_path: str,
    tickers: Optional[list] = None
) -> pd.DataFrame:
    """
    Compute technical scores for all (or specified) tickers in Excel file.

    Parameters
    ----------
    excel_path : str
        Path to Excel file
    tickers : list, optional
        List of tickers to process. If None, attempts to read all tickers.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['ticker', 'technical_score', 'sma_score', 'ema_score', 'rsi_score',
         'dmi_score', 'parabolic_score', 'macd_score', 'stochastics_score', 'mae_score']

    Notes
    -----
    This function is designed for Streamlit integration. It computes scores
    for all tickers and returns them in a format ready for display.
    """
    path = Path(excel_path)

    if tickers is None:
        # Try to auto-detect tickers from the Excel file
        # Read first few rows to get column names
        df_peek = pd.read_excel(path, sheet_name="data_prices", nrows=2)
        # Skip first column (dates) and first row (text headers)
        tickers = [col for col in df_peek.columns[1:] if not col.startswith('Unnamed')]

    results = []

    for ticker in tickers:
        try:
            # Load price data
            prices_df = load_price_data_from_excel(path, ticker)

            # Compute score
            score_data = compute_technical_score(prices_df, ticker, include_components=True)

            # Flatten for DataFrame
            row = {
                'ticker': ticker,
                'technical_score': score_data['technical_score'],
            }

            # Add component scores
            for component, score in score_data['components'].items():
                row[f'{component}_score'] = score

            results.append(row)

        except Exception as e:
            print(f"Warning: Failed to compute score for {ticker}: {e}")
            # Add placeholder row
            results.append({
                'ticker': ticker,
                'technical_score': None,
            })

    return pd.DataFrame(results)
