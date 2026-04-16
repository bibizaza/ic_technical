"""Diagnostic: test whether the Technical Nutshell data pipeline works."""
import os, sys

os.environ.setdefault(
    "IC_DROPBOX_PATH",
    r"C:\Users\User3\Dropbox\Tools_In_Construction\ic"
    if sys.platform == "win32"
    else os.path.expanduser("~/Library/CloudStorage/Dropbox/Tools_In_Construction/ic"),
)

from pathlib import Path
from data_loader import load_prices_from_csv
import pandas as pd

dp = os.environ["IC_DROPBOX_PATH"]
csv = str(Path(dp) / "master_prices.csv")

print(f"1. Loading {csv}")
df = load_prices_from_csv(Path(csv))
print(f"   Shape: {df.shape}")
print(f"   First 5 cols: {list(df.columns[:5])}")

from market_compass.technical_slide.data_prep import (
    _get_price_series, EQUITY_ASSETS, COMMO_ASSETS, CRYPTO_ASSETS,
)

print("\n2. Testing price extraction:")
for group, assets in [("EQ", EQUITY_ASSETS), ("CO", COMMO_ASSETS), ("CR", CRYPTO_ASSETS)]:
    for item in assets:
        name, ticker = item[0], item[1]
        p = _get_price_series(df, ticker)
        n = len(p) if p is not None else 0
        status = "OK" if n >= 50 else "SKIP"
        print(f"   [{group}] {name:15s} ({ticker:20s}) -> {n:5d} rows  {status}")

print("\n3. Testing market caps:")
try:
    from market_compass.technical_slide.crypto_data import fetch_crypto_market_caps, _get_api_key
    key = _get_api_key()
    print(f"   CMC API key: {'set (' + key[:8] + '...)' if key else 'MISSING'}")
    caps = fetch_crypto_market_caps()
    print(f"   Crypto caps: {caps}")
except Exception as e:
    print(f"   Crypto caps ERROR: {e}")

try:
    from data_loader import create_temp_excel_from_csv
    excel_path = str(Path(dp) / "ic_file.xlsx")
    data_as_of = pd.Timestamp("2026-04-15").date()
    tmp = create_temp_excel_from_csv(Path(csv), Path(excel_path), data_as_of)
    from market_compass.technical_slide.market_caps import get_equity_market_caps
    eq_caps = get_equity_market_caps(str(tmp))
    print(f"   Equity caps: {len(eq_caps)} entries, sample: {dict(list(eq_caps.items())[:3])}")
except Exception as e:
    print(f"   Equity caps ERROR: {e}")

print("\n4. Testing prepare_slide_data (full function):")
try:
    from utils import adjust_prices_for_mode
    from market_compass.technical_slide import prepare_slide_data
    df_adj, used_date = adjust_prices_for_mode(df, "Last Price")

    # Build dmas_scores from a dummy set (same keys the pipeline uses)
    dmas_scores = {
        "spx": 65, "csi": 65, "nikkei": 86, "tasi": 66, "sensex": 11,
        "dax": 47, "smi": 66, "mexbol": 80, "ibov": 96,
        "gold": 60, "silver": 74, "platinum": 80, "palladium": 66,
        "oil": 93, "copper": 88,
        "bitcoin": 24, "ethereum": 22, "ripple": 17, "solana": 9, "binance": 19,
    }
    rows = prepare_slide_data(df_adj, dmas_scores, str(tmp), price_mode="Last Price")
    print(f"   Rows: {len(rows)}")
    for r in rows[:3]:
        print(f"   {r.name}: mktcap={r.market_cap}, dmas={r.dmas}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n5. Testing Playwright rendering:")
try:
    from market_compass.technical_slide import insert_technical_analysis_slide
    from pptx import Presentation as Prs
    # Create a minimal test: just render the HTML to PNG
    from market_compass.technical_slide.slide_generator import _generate_tables_html, _render_html_to_png
    html = _generate_tables_html(rows, used_date, "Last Price")
    print(f"   HTML length: {len(html)} chars")
    png_bytes = _render_html_to_png(html)
    print(f"   PNG rendered: {len(png_bytes)} bytes")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n6. Testing Playwright render via _html_to_png (exact pipeline path):")
try:
    from market_compass.technical_slide.slide_generator import _html_to_png
    html = _generate_tables_html(rows)
    import tempfile as _tf
    with _tf.NamedTemporaryFile(suffix=".png", delete=False) as _f:
        _img = _f.name
    _html_to_png(html, _img)
    _sz = os.path.getsize(_img)
    print(f"   PNG rendered: {_sz} bytes")
    os.unlink(_img)
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nDone.")
