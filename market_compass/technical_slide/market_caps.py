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

        print(f"[Market Cap] Columns found: {df.columns.tolist()}")
        print(f"[Market Cap] First 3 rows:\n{df.head(3)}")

        # Find the market cap column (flexible matching)
        mkt_cap_col = None
        for col in df.columns:
            col_str = str(col).lower()
            if "market" in col_str or "cap" in col_str or col_str == "#market_cap":
                mkt_cap_col = col
                print(f"[Market Cap] Found market cap column: '{col}'")
                break

        if mkt_cap_col is None:
            # Try column B (index 1) as fallback
            if len(df.columns) > 1:
                mkt_cap_col = df.columns[1]
                print(f"[Market Cap] Using column B as fallback: '{mkt_cap_col}'")
            else:
                print("[Market Cap] ❌ Could not find market_cap column")
                return {}

        # Find the asset name column (first column or has "name"/"index"/"ticker" in it)
        name_col = df.columns[0]  # Default to first column
        for col in df.columns:
            col_str = str(col).lower()
            if "name" in col_str or "index" in col_str or "ticker" in col_str:
                name_col = col
                break

        print(f"[Market Cap] Using name column: '{name_col}', market cap column: '{mkt_cap_col}'")

        # Build mapping with flexible name matching
        market_caps = {}

        # Name mapping for display names
        name_mapping = {
            "spx index": "S&P 500",
            "s&p 500": "S&P 500",
            "shsz300 index": "CSI 300",
            "csi 300": "CSI 300",
            "nky index": "Nikkei 225",
            "nikkei 225": "Nikkei 225",
            "saseidx index": "TASI",
            "tasi": "TASI",
            "sensex index": "Sensex",
            "sensex": "Sensex",
            "dax index": "DAX",
            "dax": "DAX",
            "smi index": "SMI",
            "smi": "SMI",
            "mexbol index": "Mexbol",
            "mexbol": "Mexbol",
            "ibov index": "IBOV",
            "ibov": "IBOV",
        }

        for _, row in df.iterrows():
            asset_name = str(row[name_col]).strip()
            if pd.isna(asset_name) or asset_name == "" or asset_name == "nan":
                continue

            mkt_cap_value = row[mkt_cap_col]
            formatted_cap = format_market_cap(mkt_cap_value)

            # Store both original name and mapped display name
            market_caps[asset_name] = formatted_cap

            # Also store with display name mapping
            asset_lower = asset_name.lower()
            if asset_lower in name_mapping:
                display_name = name_mapping[asset_lower]
                market_caps[display_name] = formatted_cap

        print(f"[Market Cap] ✅ Loaded {len(market_caps)} market caps: {list(market_caps.keys())[:5]}...")
        return market_caps

    except Exception as e:
        print(f"[Market Cap] ❌ Error reading market caps: {e}")
        import traceback
        traceback.print_exc()
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
