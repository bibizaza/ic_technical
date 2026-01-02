"""CoinMarketCap API integration for crypto market cap data."""

import os
from pathlib import Path
import requests
from typing import Dict, Optional
from dotenv import load_dotenv

# Find project root and load .env from there
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent  # market_compass/technical_slide -> market_compass -> ic_technical
_ENV_PATH = _PROJECT_ROOT / ".env"

CMC_BASE_URL = "https://pro-api.coinmarketcap.com/v1"


def _get_api_key() -> Optional[str]:
    """Get API key fresh from .env file each time."""
    load_dotenv(_ENV_PATH, override=True)
    return os.getenv("COINMARKETCAP_API_KEY")

# Mapping of our asset names to CoinMarketCap symbols
CRYPTO_SYMBOLS = {
    "Bitcoin": "BTC",
    "Ethereum": "ETH",
    "Ripple": "XRP",
    "Solana": "SOL",
    "Binance": "BNB",
}


def _format_market_cap(value: float) -> str:
    """Format market cap to readable string (e.g., '1.2 T', '850 B')."""
    if value >= 1e12:
        return f"{value / 1e12:.1f} T"
    elif value >= 1e9:
        return f"{value / 1e9:.1f} B"
    elif value >= 1e6:
        return f"{value / 1e6:.1f} M"
    else:
        return f"{value:,.0f}"


def fetch_crypto_market_caps() -> Dict[str, str]:
    """
    Fetch market caps for all crypto assets from CoinMarketCap.

    Returns
    -------
    Dict[str, str]
        Mapping of asset name to formatted market cap (e.g., {"Bitcoin": "1.2 T"})
    """
    api_key = _get_api_key()
    if not api_key:
        print(f"[CoinMarketCap] WARNING: API key not found in {_ENV_PATH}, using placeholder")
        return {name: "—" for name in CRYPTO_SYMBOLS.keys()}

    # Build comma-separated symbol list
    symbols = ",".join(CRYPTO_SYMBOLS.values())

    url = f"{CMC_BASE_URL}/cryptocurrency/quotes/latest"
    headers = {
        "X-CMC_PRO_API_KEY": api_key,
        "Accept": "application/json",
    }
    params = {
        "symbol": symbols,
        "convert": "USD",
    }

    try:
        print(f"[CoinMarketCap] Fetching market caps for: {symbols}")
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status", {}).get("error_code") != 0:
            error_msg = data.get("status", {}).get("error_message", "Unknown error")
            print(f"[CoinMarketCap] API Error: {error_msg}")
            return {name: "—" for name in CRYPTO_SYMBOLS.keys()}

        # Extract market caps
        result = {}
        for name, symbol in CRYPTO_SYMBOLS.items():
            try:
                crypto_data = data["data"][symbol]
                quote = crypto_data["quote"]["USD"]

                # Try market_cap first
                market_cap = quote.get("market_cap")

                # Fallback: circulating_supply * price
                if not market_cap or market_cap == 0:
                    circulating = crypto_data.get("circulating_supply", 0)
                    price = quote.get("price", 0)
                    market_cap = circulating * price
                    print(f"[CoinMarketCap] {name}: Using calculated market cap (supply * price)")

                result[name] = _format_market_cap(market_cap) if market_cap else "—"
                print(f"[CoinMarketCap] {name}: {result[name]}")

            except KeyError as e:
                print(f"[CoinMarketCap] {name}: Data not found ({e})")
                result[name] = "—"

        return result

    except requests.exceptions.RequestException as e:
        print(f"[CoinMarketCap] Request failed: {e}")
        return {name: "—" for name in CRYPTO_SYMBOLS.keys()}


def get_crypto_market_cap(name: str) -> str:
    """
    Get market cap for a single crypto asset.

    Parameters
    ----------
    name : str
        Asset name (e.g., "Bitcoin", "Ethereum")

    Returns
    -------
    str
        Formatted market cap or "—" if unavailable
    """
    symbol = CRYPTO_SYMBOLS.get(name)
    if not symbol:
        return "—"

    api_key = _get_api_key()
    if not api_key:
        return "—"

    url = f"{CMC_BASE_URL}/cryptocurrency/quotes/latest"
    headers = {
        "X-CMC_PRO_API_KEY": api_key,
        "Accept": "application/json",
    }
    params = {
        "symbol": symbol,
        "convert": "USD",
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        crypto_data = data["data"][symbol]
        quote = crypto_data["quote"]["USD"]

        market_cap = quote.get("market_cap")
        if not market_cap or market_cap == 0:
            circulating = crypto_data.get("circulating_supply", 0)
            price = quote.get("price", 0)
            market_cap = circulating * price

        return _format_market_cap(market_cap) if market_cap else "—"

    except Exception as e:
        print(f"[CoinMarketCap] Error fetching {name}: {e}")
        return "—"
