#!/usr/bin/env python3
"""
import os; os.environ["PYTHONIOENCODING"] = "utf-8"
Shadow IC - Side-by-side comparison of Haiku vs DeepSeek subtitles.

Replays the same subtitle prompts used by the production IC pipeline
to both Claude Haiku and a local DeepSeek model via Ollama.

Usage:
    python shadow_ic.py

Output:
    - shadow_comparison.html: Side-by-side HTML table
    - shadow_comparison.json: Raw results for further analysis
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import requests

# Import prompt construction from production code (no duplication)
from market_compass.subtitle_generator.claude_generator import (
    build_prompt,
    SYSTEM_PROMPT,
    EXAMPLES,
    MODELS,
    DEFAULT_MODEL_KEY,
    get_client,
)

# Import helper functions
from assessment_integration import (
    get_default_assessment_from_dmas,
    _calculate_ma,
    _calculate_ma_pct,
    detect_ath,
    detect_52w_low,
    detect_ma_cross,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Default paths (Dropbox location)
IC_DROPBOX = Path.home() / "Library/CloudStorage/Dropbox/Tools_In_Construction/ic"
DEFAULT_EXCEL = IC_DROPBOX / "ic_file.xlsx"
DEFAULT_MASTER = IC_DROPBOX / "master_prices.csv"
OUTPUT_DIR = Path(__file__).parent

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
DEEPSEEK_MODEL = "deepseek-r1:32b"

# Asset configuration (same as cli_generate.py)
ASSET_MAP = {
    # Equity
    "spx": ("SPX Index", "S&P 500"),
    "csi": ("SHSZ300 Index", "CSI 300"),
    "nikkei": ("NKY Index", "Nikkei 225"),
    "tasi": ("SASEIDX Index", "TASI"),
    "sensex": ("SENSEX Index", "Sensex"),
    "dax": ("DAX Index", "Dax"),
    "smi": ("SMI Index", "SMI"),
    "ibov": ("IBOV Index", "IBOV"),
    "mexbol": ("MEXBOL Index", "MEXBOL"),
    # Commodity
    "gold": ("GCA Comdty", "Gold"),
    "silver": ("SIA Comdty", "Silver"),
    "platinum": ("XPT Comdty", "Platinum"),
    "palladium": ("XPD Curncy", "Palladium"),
    "oil": ("CL1 Comdty", "Oil"),
    "copper": ("LP1 Comdty", "Copper"),
    # Crypto
    "bitcoin": ("XBTUSD Curncy", "Bitcoin"),
    "ethereum": ("XETUSD Curncy", "Ethereum"),
    "ripple": ("XRPUSD Curncy", "Ripple"),
    "solana": ("XSOUSD Curncy", "Solana"),
    "binance": ("XBIUSD Curncy", "Binance"),
}


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_prices(master_csv: Path, data_as_of: datetime = None) -> pd.DataFrame:
    """Load price data from master CSV."""
    df = pd.read_csv(master_csv, sep=';', header=[0, 1])

    # Flatten multi-index columns
    df.columns = ['_'.join(col).strip() for col in df.columns.values]

    # Find date column
    date_col = [c for c in df.columns if 'Unnamed' in c or 'date' in c.lower()][0]
    df['Date'] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Date'])

    if data_as_of:
        df = df[df['Date'] <= data_as_of]

    df = df.sort_values('Date')
    return df


def get_price_series(df: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    """Extract price series for a specific ticker."""
    # Find the #price column for this ticker
    price_col = None
    for col in df.columns:
        if ticker in col and '#price' in col.lower():
            price_col = col
            break

    if price_col is None:
        # Try alternative format
        for col in df.columns:
            if ticker.replace(' ', '') in col.replace(' ', '') and 'price' in col.lower():
                price_col = col
                break

    if price_col is None:
        return None

    series = pd.to_numeric(df[price_col], errors='coerce')
    series.index = df['Date']
    return series.dropna()


def load_scores_from_excel(excel_path: Path) -> Dict[str, Dict]:
    """Load DMAS scores from mars_score sheet."""
    try:
        df = pd.read_excel(excel_path, sheet_name="mars_score")
        scores = {}

        # Find ticker column
        ticker_col = df.columns[0]

        for _, row in df.iterrows():
            ticker = str(row[ticker_col]).strip() if pd.notna(row[ticker_col]) else ""
            if not ticker:
                continue

            # Find matching asset
            for key, (bbg, name) in ASSET_MAP.items():
                if ticker.upper() == bbg.upper():
                    # Get scores from columns
                    tech = row.iloc[1] if len(row) > 1 and pd.notna(row.iloc[1]) else 50
                    mom = row.iloc[2] if len(row) > 2 and pd.notna(row.iloc[2]) else 50
                    dmas = (float(tech) + float(mom)) / 2

                    scores[key] = {
                        "technical_score": float(tech),
                        "momentum_score": float(mom),
                        "dmas": dmas,
                    }
                    break

        return scores
    except Exception as e:
        print(f"Warning: Could not load scores from Excel: {e}")
        return {}


# ==============================================================================
# PROMPT BUILDING
# ==============================================================================

def build_asset_data(
    ticker_key: str,
    asset_name: str,
    scores: Dict,
    prices: pd.Series
) -> Dict[str, Any]:
    """Build asset data dict for prompt construction."""
    dmas = scores.get("dmas", 50)
    technical = scores.get("technical_score", dmas)
    momentum = scores.get("momentum_score", dmas)

    # Calculate MA percentages
    price_vs_50ma_pct = 0.0
    price_vs_100ma_pct = 0.0
    price_vs_200ma_pct = 0.0
    price_change_1w_pct = 0.0

    if prices is not None and len(prices) >= 200:
        current_price = prices.iloc[-1]
        ma_50 = _calculate_ma(prices, 50)
        ma_100 = _calculate_ma(prices, 100)
        ma_200 = _calculate_ma(prices, 200)

        price_vs_50ma_pct = _calculate_ma_pct(current_price, ma_50)
        price_vs_100ma_pct = _calculate_ma_pct(current_price, ma_100)
        price_vs_200ma_pct = _calculate_ma_pct(current_price, ma_200)

        # Weekly price change
        if len(prices) >= 5:
            prev_price = prices.iloc[-6] if len(prices) > 5 else prices.iloc[0]
            if prev_price > 0:
                price_change_1w_pct = ((current_price / prev_price) - 1) * 100

    return {
        "asset_name": asset_name,
        "dmas": int(dmas),
        "technical_score": int(technical),
        "momentum_score": int(momentum),
        "dmas_prev_week": int(dmas),  # Simplified for shadow comparison
        "price_vs_50ma_pct": price_vs_50ma_pct,
        "price_vs_100ma_pct": price_vs_100ma_pct,
        "price_vs_200ma_pct": price_vs_200ma_pct,
        "price_change_1w_pct": price_change_1w_pct,
    }


# ==============================================================================
# MODEL CALLS
# ==============================================================================

def call_haiku(prompt: str, system_prompt: str) -> str:
    """Call Claude Haiku API."""
    try:
        client = get_client()
        model = MODELS[DEFAULT_MODEL_KEY]

        message = client.messages.create(
            model=model,
            max_tokens=50,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {"role": "user", "content": prompt}
            ],
            extra_headers={
                "anthropic-beta": "prompt-caching-2024-07-31"
            }
        )

        subtitle = message.content[0].text.strip()
        subtitle = subtitle.strip('"\'').rstrip('.')
        return subtitle

    except Exception as e:
        return f"[error: {str(e)[:50]}]"


def call_deepseek(prompt: str, system_prompt: str) -> str:
    """Call DeepSeek via Ollama."""
    try:
        # Combine system and user prompt for Ollama
        full_prompt = f"{system_prompt}\n\n{prompt}"

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": DEEPSEEK_MODEL,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "num_predict": 50,
                    "temperature": 0.7,
                }
            },
            timeout=300
        )

        if response.status_code != 200:
            return f"[HTTP {response.status_code}]"

        result = response.json()
        subtitle = result.get("response", "").strip()

        # Clean up response (remove thinking tags if present)
        if "<think>" in subtitle:
            # Extract content after </think>
            parts = subtitle.split("</think>")
            if len(parts) > 1:
                subtitle = parts[-1].strip()

        subtitle = subtitle.strip('"\'').rstrip('.')

        # Truncate to ~12 words
        words = subtitle.split()
        if len(words) > 15:
            subtitle = ' '.join(words[:12])

        return subtitle

    except requests.exceptions.ConnectionError:
        return "[unavailable - Ollama not running]"
    except requests.exceptions.Timeout:
        return "[timeout]"
    except Exception as e:
        return f"[error: {str(e)[:50]}]"


# ==============================================================================
# COMPARISON GENERATION
# ==============================================================================

def generate_comparison(
    excel_path: Path = DEFAULT_EXCEL,
    master_path: Path = DEFAULT_MASTER,
) -> List[Dict]:
    """Generate side-by-side comparison for all assets."""

    print("=" * 60)
    print("SHADOW IC - Haiku vs DeepSeek Comparison")
    print("=" * 60)

    # Load data
    print(f"\nLoading prices from: {master_path}")
    df_prices = load_prices(master_path)
    print(f"  Loaded {len(df_prices)} rows")

    print(f"\nLoading scores from: {excel_path}")
    scores = load_scores_from_excel(excel_path)
    print(f"  Loaded scores for {len(scores)} assets")

    # Prepare system prompt (same as production)
    system_prompt = SYSTEM_PROMPT + "\n\n" + EXAMPLES

    results = []
    previous_subtitles = []

    print(f"\nGenerating comparisons for {len(ASSET_MAP)} assets...")
    print("-" * 60)

    for i, (ticker_key, (bbg_ticker, display_name)) in enumerate(ASSET_MAP.items()):
        print(f"\n[{i+1}/{len(ASSET_MAP)}] {display_name}")

        # Get price series
        prices = get_price_series(df_prices, bbg_ticker)

        # Get scores (use defaults if not in Excel)
        asset_scores = scores.get(ticker_key, {
            "dmas": 50,
            "technical_score": 50,
            "momentum_score": 50,
        })

        # Build asset data
        asset_data = build_asset_data(
            ticker_key,
            display_name,
            asset_scores,
            prices
        )

        # Get rating
        dmas = asset_data["dmas"]
        if dmas >= 70:
            rating = "Bullish"
        elif dmas >= 55:
            rating = "Constructive"
        elif dmas >= 45:
            rating = "Neutral"
        elif dmas >= 30:
            rating = "Cautious"
        else:
            rating = "Bearish"

        # Build prompt (using production function)
        prompt = build_prompt(asset_data, previous_subtitles)

        # Call both models
        print(f"  Calling Haiku...")
        haiku_subtitle = call_haiku(prompt, system_prompt)
        print(f"    -> {haiku_subtitle[:60]}...")

        print(f"  Calling DeepSeek...")
        deepseek_subtitle = call_deepseek(prompt, system_prompt)
        print(f"    -> {deepseek_subtitle[:60]}...")

        # Store result
        result = {
            "asset": display_name,
            "ticker_key": ticker_key,
            "dmas": dmas,
            "rating": rating,
            "haiku_subtitle": haiku_subtitle,
            "deepseek_subtitle": deepseek_subtitle,
            "prompt": prompt,  # For debugging
        }
        results.append(result)

        # Track for deduplication
        previous_subtitles.append(haiku_subtitle)

    return results


def generate_html(results: List[Dict], output_path: Path) -> None:
    """Generate HTML comparison table."""

    html = """<!DOCTYPE html>
<html>
<head>
    <title>Shadow IC - Haiku vs DeepSeek Comparison</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #1a1a2e;
            border-bottom: 2px solid #4a4e69;
            padding-bottom: 10px;
        }
        .timestamp {
            color: #666;
            font-size: 14px;
            margin-bottom: 20px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        th {
            background: #1a1a2e;
            color: white;
            padding: 12px 16px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 12px 16px;
            border-bottom: 1px solid #eee;
        }
        tr:hover {
            background: #f8f9fa;
        }
        .rating-bullish { color: #22c55e; font-weight: 600; }
        .rating-constructive { color: #84cc16; font-weight: 600; }
        .rating-neutral { color: #eab308; font-weight: 600; }
        .rating-cautious { color: #f97316; font-weight: 600; }
        .rating-bearish { color: #ef4444; font-weight: 600; }
        .subtitle {
            font-style: italic;
            color: #333;
        }
        .error {
            color: #ef4444;
            font-style: italic;
        }
        .dmas {
            font-weight: 600;
            color: #4a4e69;
        }
    </style>
</head>
<body>
    <h1>Shadow IC - Subtitle Comparison</h1>
    <p class="timestamp">Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>

    <table>
        <thead>
            <tr>
                <th>Instrument</th>
                <th>DMAS</th>
                <th>Rating</th>
                <th>Haiku Subtitle</th>
                <th>DeepSeek Subtitle</th>
            </tr>
        </thead>
        <tbody>
"""

    for r in results:
        rating_class = f"rating-{r['rating'].lower()}"
        haiku_class = "error" if r['haiku_subtitle'].startswith('[') else "subtitle"
        deepseek_class = "error" if r['deepseek_subtitle'].startswith('[') else "subtitle"

        html += f"""            <tr>
                <td><strong>{r['asset']}</strong></td>
                <td class="dmas">{r['dmas']}</td>
                <td class="{rating_class}">{r['rating']}</td>
                <td class="{haiku_class}">{r['haiku_subtitle']}</td>
                <td class="{deepseek_class}">{r['deepseek_subtitle']}</td>
            </tr>
"""

    html += """        </tbody>
    </table>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"\nHTML saved to: {output_path}")


def generate_json(results: List[Dict], output_path: Path) -> None:
    """Save raw results to JSON."""
    # Remove prompts for cleaner JSON (they're long)
    clean_results = []
    for r in results:
        clean_results.append({
            "asset": r["asset"],
            "ticker_key": r["ticker_key"],
            "dmas": r["dmas"],
            "rating": r["rating"],
            "haiku_subtitle": r["haiku_subtitle"],
            "deepseek_subtitle": r["deepseek_subtitle"],
        })

    with open(output_path, 'w') as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "haiku_model": MODELS[DEFAULT_MODEL_KEY],
            "deepseek_model": DEEPSEEK_MODEL,
            "results": clean_results,
        }, f, indent=2)

    print(f"JSON saved to: {output_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Run shadow comparison."""

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key'")
        sys.exit(1)

    # Check input files
    if not DEFAULT_EXCEL.exists():
        print(f"Error: Excel file not found: {DEFAULT_EXCEL}")
        sys.exit(1)

    if not DEFAULT_MASTER.exists():
        print(f"Error: Master prices not found: {DEFAULT_MASTER}")
        sys.exit(1)

    # Generate comparison
    results = generate_comparison()

    # Output files
    html_path = OUTPUT_DIR / "shadow_comparison.html"
    json_path = OUTPUT_DIR / "shadow_comparison.json"

    generate_html(results, html_path)
    generate_json(results, json_path)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  HTML: {html_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
