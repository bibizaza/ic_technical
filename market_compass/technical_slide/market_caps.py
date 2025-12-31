"""Market cap data sources for Technical Analysis slide."""

from typing import Dict, Union
import pandas as pd

# =============================================================================
# STATIC MARKET CAPS (update quarterly)
# =============================================================================

COMMO_MARKET_CAPS = {
    "Gold": "17.1 T",
    "Silver": "1.4 T",
    "Platinum": "32 B",
    "Palladium": "14 B",
    "Oil (WTI)": "—",
    "Copper": "—",
}

# Placeholder for crypto (Phase 2 will add API)
CRYPTO_MARKET_CAPS = {
    "Bitcoin": "—",
    "Ethereum": "—",
    "Ripple": "—",
    "Solana": "—",
    "Binance": "—",
}


def format_market_cap(value: Union[float, int, str]) -> str:
    """
    Format market cap as readable string.

    Examples:
        58510000000000 -> "58.51 T"
        2260000000 -> "2.26 B"
        500000000 -> "500 M"
    """
    if isinstance(value, str):
        return value

    if pd.isna(value):
        return "—"

    value = float(value)

    if value >= 1e12:
        return f"{value / 1e12:.2f} T"
    elif value >= 1e9:
        return f"{value / 1e9:.2f} B"
    elif value >= 1e6:
        return f"{value / 1e6:.0f} M"
    else:
        return f"{value:,.0f}"


def get_equity_market_caps(excel_path: str) -> Dict[str, str]:
    """
    Read equity market caps from Excel.

    Parameters
    ----------
    excel_path : str
        Path to Excel file with market_cap sheet

    Returns
    -------
    Dict[str, str]
        Mapping of asset name to formatted market cap
    """
    try:
        df = pd.read_excel(excel_path, sheet_name="market_cap")

        # Find the market cap column
        mkt_cap_col = None
        for col in df.columns:
            if "#market_cap" in str(col).lower() or "market_cap" in str(col).lower():
                mkt_cap_col = col
                break

        if mkt_cap_col is None:
            # Try column B (index 1)
            if len(df.columns) > 1:
                mkt_cap_col = df.columns[1]
            else:
                print("Warning: Could not find market_cap column")
                return {}

        # Build mapping
        market_caps = {}
        name_col = df.columns[0]  # First column = asset names

        for _, row in df.iterrows():
            asset_name = str(row[name_col]).strip()
            if pd.isna(asset_name) or asset_name == "":
                continue

            mkt_cap_value = row[mkt_cap_col]
            market_caps[asset_name] = format_market_cap(mkt_cap_value)

        return market_caps

    except Exception as e:
        print(f"Warning: Could not read market caps from Excel: {e}")
        return {}


# =============================================================================
# CRYPTO API (Phase 2 - Future)
# =============================================================================

def get_crypto_market_caps_from_api(api_key: str = None) -> Dict[str, str]:
    """
    Fetch market caps from CoinMarketCap API.

    NOTE: This is a placeholder for Phase 2 implementation.
    Requires a CoinMarketCap API key.

    Parameters
    ----------
    api_key : str, optional
        CoinMarketCap API key

    Returns
    -------
    Dict[str, str]
        Mapping of asset name to formatted market cap
    """
    if api_key is None:
        # Return static placeholders
        return CRYPTO_MARKET_CAPS.copy()

    try:
        import requests

        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"

        headers = {
            "X-CMC_PRO_API_KEY": api_key,
            "Accept": "application/json"
        }

        params = {
            "symbol": "BTC,ETH,XRP,SOL,BNB",
            "convert": "USD"
        }

        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()

        # Map to our asset names
        symbol_to_name = {
            "BTC": "Bitcoin",
            "ETH": "Ethereum",
            "XRP": "Ripple",
            "SOL": "Solana",
            "BNB": "Binance"
        }

        market_caps = {}
        for symbol, name in symbol_to_name.items():
            if symbol in data.get("data", {}):
                mkt_cap = data["data"][symbol]["quote"]["USD"]["market_cap"]
                market_caps[name] = format_market_cap(mkt_cap)
            else:
                market_caps[name] = "—"

        return market_caps

    except Exception as e:
        print(f"Warning: Could not fetch crypto market caps: {e}")
        return CRYPTO_MARKET_CAPS.copy()
