"""Commodity market cap calculation using reserves × price."""

from typing import Dict, Optional
import pandas as pd

from .config import (
    COMMODITY_RESERVES_TONNES,
    TROY_OZ_PER_TONNE,
    COMMODITY_SPOT_TICKERS,
)


def _format_market_cap(value: float) -> str:
    """Format market cap to readable string (e.g., '17.1 T', '850 B')."""
    if value >= 1e12:
        return f"{value / 1e12:.1f} T"
    elif value >= 1e9:
        return f"{value / 1e9:.1f} B"
    elif value >= 1e6:
        return f"{value / 1e6:.1f} M"
    else:
        return f"{value:,.0f}"


def calculate_commodity_market_cap(name: str, price_per_oz: float) -> str:
    """
    Calculate market cap for a precious metal commodity.

    Market Cap = Above-Ground Reserves (tonnes) × Price per tonne

    Parameters
    ----------
    name : str
        Commodity name (e.g., "Gold", "Silver")
    price_per_oz : float
        Current spot price per troy ounce in USD

    Returns
    -------
    str
        Formatted market cap (e.g., "17.1 T") or "—" if not applicable
    """
    reserves = COMMODITY_RESERVES_TONNES.get(name)

    if reserves is None or price_per_oz is None or price_per_oz <= 0:
        return "—"

    # Convert price per oz to price per tonne
    price_per_tonne = price_per_oz * TROY_OZ_PER_TONNE

    # Calculate market cap
    market_cap = reserves * price_per_tonne

    return _format_market_cap(market_cap)


def get_commodity_spot_prices(prices_df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract spot prices for precious metals from Bloomberg price data.

    Parameters
    ----------
    prices_df : pd.DataFrame
        DataFrame with price columns (from Excel)

    Returns
    -------
    Dict[str, float]
        Mapping of commodity name to spot price per oz
    """
    prices = {}

    for name, ticker in COMMODITY_SPOT_TICKERS.items():
        # Try to find the ticker column (case-insensitive)
        matching_col = None
        ticker_upper = ticker.upper()

        for col in prices_df.columns:
            if isinstance(col, str) and col.upper() == ticker_upper:
                matching_col = col
                break

        if matching_col is not None:
            # Get the last valid price
            series = pd.to_numeric(prices_df[matching_col], errors="coerce")
            last_price = series.dropna().iloc[-1] if len(series.dropna()) > 0 else None

            if last_price is not None and last_price > 0:
                prices[name] = float(last_price)
                print(f"[Commodity] {name}: ${prices[name]:.2f}/oz")
            else:
                print(f"[Commodity] {name}: No valid price found")
        else:
            print(f"[Commodity] {name}: Ticker {ticker} not found in data")

    return prices


def calculate_all_commodity_market_caps(prices_df: pd.DataFrame) -> Dict[str, str]:
    """
    Calculate market caps for all precious metals from Excel price data.

    Parameters
    ----------
    prices_df : pd.DataFrame
        DataFrame with price columns (from Excel)

    Returns
    -------
    Dict[str, str]
        Mapping of commodity name to formatted market cap
    """
    # Get spot prices from Excel
    spot_prices = get_commodity_spot_prices(prices_df)

    # Calculate market caps
    result = {}
    for name in COMMODITY_RESERVES_TONNES.keys():
        price = spot_prices.get(name)
        if price:
            result[name] = calculate_commodity_market_cap(name, price)
            print(f"[Commodity] {name} market cap: {result[name]}")
        else:
            result[name] = "—"

    return result
