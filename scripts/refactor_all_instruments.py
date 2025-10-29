#!/usr/bin/env python3
"""
Refactor all remaining instruments to use common_helpers.py

This script applies the same refactoring pattern that worked for SPX to all other instruments:
1. Add import of common helpers
2. Remove duplicate helper functions (_get_run_font_attributes, _apply_run_font_attributes, _add_mas)
3. Replace score functions with calls to generic versions

Expected reduction: ~170-180 lines per instrument
Total expected reduction: ~3,400 lines across 19 instruments
"""

from pathlib import Path
import re
import shutil

BASE_DIR = Path("technical_analysis")

# Instrument configurations (ticker names for score lookups)
INSTRUMENTS = {
    'equity': [
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

COMMON_IMPORT = """
# Import common helpers (eliminates code duplication)
from technical_analysis.common_helpers import (
    _get_run_font_attributes,
    _apply_run_font_attributes,
    _add_mas,
    _get_technical_score_generic,
    _get_momentum_score_generic,
)"""


def refactor_instrument(category: str, name: str, ticker: str):
    """Refactor a single instrument file."""

    file_path = BASE_DIR / category / f"{name}.py"
    backup_path = BASE_DIR / category / f"{name}_before_refactor_backup.py"

    if not file_path.exists():
        print(f"⚠️  {category}/{name}.py not found, skipping")
        return 0

    # Read original file
    with open(file_path, 'r') as f:
        content = f.read()
        original_lines = len(content.split('\n'))

    # Create backup
    shutil.copy(file_path, backup_path)

    # Split into lines for import insertion
    lines = content.split('\n')

    # Find where to insert import (after "from pptx import Presentation")
    import_insert_line = None
    for i, line in enumerate(lines):
        if 'from pptx import Presentation' in line:
            import_insert_line = i + 1
            break

    if import_insert_line is None:
        print(f"⚠️  Could not find import section in {category}/{name}.py")
        return 0

    # Insert common helpers import
    lines.insert(import_insert_line, COMMON_IMPORT)
    content = '\n'.join(lines)

    # Remove duplicate helper functions using regex patterns
    # These patterns match the function definition until the next function/class

    # Remove _get_run_font_attributes
    pattern_get_font = r'def _get_run_font_attributes\(run\):.*?(?=\n\ndef |\nclass |\Z)'
    content = re.sub(pattern_get_font, '', content, flags=re.DOTALL)

    # Remove _apply_run_font_attributes
    pattern_apply_font = r'def _apply_run_font_attributes\(new_run.*?\):.*?(?=\n\ndef |\nclass |\Z)'
    content = re.sub(pattern_apply_font, '', content, flags=re.DOTALL)

    # Remove _add_mas
    pattern_add_mas = r'def _add_mas\(df: pd\.DataFrame\).*?(?=\n\ndef |\nclass |\Z)'
    content = re.sub(pattern_add_mas, '', content, flags=re.DOTALL)

    # Replace technical score function
    old_tech_pattern = rf'def _get_{name}_technical_score\(excel_obj_or_path\).*?(?=\n\ndef )'
    new_tech_score = f'''def _get_{name}_technical_score(excel_obj_or_path) -> Optional[float]:
    """
    Retrieve the technical score for {name.upper()}.
    Uses common helper with instrument-specific ticker.
    """
    return _get_technical_score_generic(excel_obj_or_path, "{ticker}")

'''
    content = re.sub(old_tech_pattern, new_tech_score, content, flags=re.DOTALL)

    # Replace momentum score function
    old_mom_pattern = rf'def _get_{name}_momentum_score\(excel_obj_or_path\).*?(?=\n\ndef )'
    new_mom_score = f'''def _get_{name}_momentum_score(excel_obj_or_path) -> Optional[float]:
    """
    Retrieve the momentum score for {name.upper()}.
    Uses common helper with instrument-specific ticker.
    """
    return _get_momentum_score_generic(excel_obj_or_path, "{ticker}")

'''
    content = re.sub(old_mom_pattern, new_mom_score, content, flags=re.DOTALL)

    # Write refactored content
    with open(file_path, 'w') as f:
        f.write(content)

    # Calculate reduction
    new_lines = len(content.split('\n'))
    reduction = original_lines - new_lines

    print(f"✓ {category:10s}/{name:10s}: {original_lines:4d} → {new_lines:4d} lines (-{reduction:3d}, {reduction/original_lines*100:.1f}%)")

    return reduction


def main():
    """Refactor all instruments."""

    print("=" * 80)
    print("REFACTORING ALL INSTRUMENTS")
    print("=" * 80)
    print()

    total_reduction = 0
    instruments_processed = 0

    for category, instruments in INSTRUMENTS.items():
        print(f"\n{category.upper()}:")
        for config in instruments:
            reduction = refactor_instrument(category, config['name'], config['ticker'])
            total_reduction += reduction
            if reduction > 0:
                instruments_processed += 1

    print()
    print("=" * 80)
    print(f"REFACTORING COMPLETE!")
    print("=" * 80)
    print(f"Instruments processed: {instruments_processed}")
    print(f"Total lines removed:   {total_reduction:,}")
    print()
    print("Including SPX (already refactored): 174 lines")
    print(f"Grand total reduction: {total_reduction + 174:,} lines")
    print()
    print("All backups saved as *_before_refactor_backup.py")
    print()
    print("NEXT: Test with 'streamlit run app.py' to verify everything works!")


if __name__ == '__main__':
    main()
