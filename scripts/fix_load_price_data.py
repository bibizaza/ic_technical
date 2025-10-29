#!/usr/bin/env python3
"""
Fix: Restore _load_price_data function that was incorrectly removed in Phase 2

The function _load_price_data(path, ticker) is different from _load_price_data_from_obj(obj, ticker).
It needs to be restored as a simple wrapper in each instrument that calls the generic version.
"""

from pathlib import Path
import re

BASE_DIR = Path("technical_analysis")

# Instrument configurations with their tickers
INSTRUMENTS = {
    'equity': [
        {'name': 'spx', 'ticker': 'SPX Index'},
        {'name': 'csi', 'ticker': 'SHSZ300 INDEX'},
        {'name': 'dax', 'ticker': 'DAX INDEX'},
        {'name': 'ibov', 'ticker': 'IBOV INDEX'},
        {'name': 'mexbol', 'ticker': 'MEXBOL INDEX'},
        {'name': 'nikkei', 'ticker': 'NKY INDEX'},
        {'name': 'sensex', 'ticker': 'SENSEX INDEX'},
        {'name': 'smi', 'ticker': 'SMI INDEX'},
        {'name': 'tasi', 'ticker': 'SASEIDX INDEX'},
    ],
    'commodity': [
        {'name': 'gold', 'ticker': 'GCA COMDTY'},
        {'name': 'silver', 'ticker': 'SI1 COMDTY'},
        {'name': 'copper', 'ticker': 'HG1 COMDTY'},
        {'name': 'oil', 'ticker': 'CL1 COMDTY'},
        {'name': 'palladium', 'ticker': 'PA1 COMDTY'},
        {'name': 'platinum', 'ticker': 'PL1 COMDTY'},
    ],
    'crypto': [
        {'name': 'bitcoin', 'ticker': 'XBTUSD CURNCY'},
        {'name': 'ethereum', 'ticker': 'XETUSD CURNCY'},
        {'name': 'binance', 'ticker': 'XBNCUR CURNCY'},
        {'name': 'solana', 'ticker': 'SOLUSD CURNCY'},
        {'name': 'ripple', 'ticker': 'XRPUSD CURNCY'},
    ],
}


def fix_instrument(category: str, name: str, ticker: str):
    """Add back _load_price_data as a wrapper function."""

    file_path = BASE_DIR / category / f"{name}.py"

    if not file_path.exists():
        print(f"⚠️  {category}/{name}.py not found")
        return False

    with open(file_path, 'r') as f:
        content = f.read()

    # Check if _load_price_data already exists
    if re.search(r'^def _load_price_data\(', content, re.MULTILINE):
        print(f"✓ {category:10s}/{name:10s}: Already has _load_price_data")
        return False

    # Find where to insert (after imports, before first function)
    # Look for the end of imports section (after the common_helpers import)
    insert_pattern = r'(from technical_analysis\.common_helpers import.*?\))'

    match = re.search(insert_pattern, content, re.DOTALL)
    if not match:
        print(f"⚠️  {category}/{name}.py: Could not find import section")
        return False

    insert_pos = match.end()

    # Create the wrapper function
    wrapper_function = f'''


def _load_price_data(
    excel_path: pathlib.Path,
    ticker: str = "{ticker}",
    price_mode: str = "Last Price",
) -> pd.DataFrame:
    """
    Read the raw price sheet and return a tidy Date‑Price DataFrame.

    This is a wrapper around _load_price_data_generic with the instrument-specific
    default ticker.

    Parameters
    ----------
    excel_path : pathlib.Path
        Path to the Excel workbook containing price data.
    ticker : str, default "{ticker}"
        Column name corresponding to the desired ticker in the Excel sheet.
    price_mode : str, default "Last Price"
        One of "Last Price" or "Last Close".  If ``adjust_prices_for_mode``
        is available and the mode is "Last Close", rows with the last
        recorded date (if equal to today's date) will be dropped.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``Date`` and ``Price``.  The data are
        sorted by date and any rows with missing values are removed.
    """
    return _load_price_data_generic(excel_path, ticker, price_mode)
'''

    # Insert the function
    new_content = content[:insert_pos] + wrapper_function + content[insert_pos:]

    # Update imports to include _load_price_data_generic
    old_import = """from technical_analysis.common_helpers import (
    _get_run_font_attributes,
    _apply_run_font_attributes,
    _add_mas,
    _get_technical_score_generic,
    _get_momentum_score_generic,
    _interpolate_color,
    _load_price_data_from_obj,
    _compute_range_bounds,
)"""

    new_import = """from technical_analysis.common_helpers import (
    _get_run_font_attributes,
    _apply_run_font_attributes,
    _add_mas,
    _get_technical_score_generic,
    _get_momentum_score_generic,
    _interpolate_color,
    _load_price_data_from_obj,
    _load_price_data_generic,
    _compute_range_bounds,
)"""

    new_content = new_content.replace(old_import, new_import)

    # Write back
    with open(file_path, 'w') as f:
        f.write(new_content)

    print(f"✓ {category:10s}/{name:10s}: Added _load_price_data wrapper")
    return True


def main():
    """Fix all instruments."""

    print("=" * 80)
    print("FIXING: Restore _load_price_data function")
    print("=" * 80)
    print()

    fixed_count = 0

    for category, instruments in INSTRUMENTS.items():
        print(f"\n{category.upper()}:")
        for config in instruments:
            if fix_instrument(category, config['name'], config['ticker']):
                fixed_count += 1

    print()
    print("=" * 80)
    print(f"Fixed {fixed_count} files")
    print("=" * 80)


if __name__ == '__main__':
    main()
