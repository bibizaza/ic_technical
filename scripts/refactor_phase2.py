#!/usr/bin/env python3
"""
Phase 2 Refactoring: Extract additional common utility functions

This script removes 3 more duplicate functions from all instruments:
1. _interpolate_color (17 lines each)
2. _load_price_data_from_obj (49 lines each)
3. _compute_range_bounds (66 lines each)

Expected reduction: ~2,300 lines total (115 lines × 20 instruments)

Combined with Phase 1: ~4,600 lines total reduction
"""

from pathlib import Path
import re
import shutil

BASE_DIR = Path("technical_analysis")

# All instruments (same as Phase 1)
INSTRUMENTS = {
    'equity': ['spx', 'csi', 'dax', 'ibov', 'mexbol', 'nikkei', 'sensex', 'smi', 'tasi'],
    'commodity': ['gold', 'silver', 'copper', 'oil', 'palladium', 'platinum'],
    'crypto': ['bitcoin', 'ethereum', 'binance', 'solana', 'ripple'],
}


def refactor_instrument_phase2(category: str, name: str):
    """Apply Phase 2 refactoring to a single instrument."""

    file_path = BASE_DIR / category / f"{name}.py"
    backup_path = BASE_DIR / category / f"{name}_phase2_backup.py"

    if not file_path.exists():
        print(f"⚠️  {category}/{name}.py not found, skipping")
        return 0

    # Read original file
    with open(file_path, 'r') as f:
        content = f.read()
        original_lines = len(content.split('\n'))

    # Create backup (overwrite Phase 1 backup)
    shutil.copy(file_path, backup_path)

    # Update imports to include the 3 new functions
    old_import = """from technical_analysis.common_helpers import (
    _get_run_font_attributes,
    _apply_run_font_attributes,
    _add_mas,
    _get_technical_score_generic,
    _get_momentum_score_generic,
)"""

    new_import = """from technical_analysis.common_helpers import (
    _get_run_font_attributes,
    _apply_run_font_attributes,
    _add_mas,
    _get_technical_score_generic,
    _get_momentum_score_generic,
    _interpolate_color,
    _load_price_data_from_obj,
    _compute_range_bounds,
)"""

    content = content.replace(old_import, new_import)

    # Remove _interpolate_color function
    pattern_interpolate = r'def _interpolate_color\(value: float\).*?(?=\n\ndef |\nclass |\Z)'
    content = re.sub(pattern_interpolate, '', content, flags=re.DOTALL)

    # Remove _load_price_data_from_obj function
    pattern_load_obj = r'def _load_price_data_from_obj\(.*?(?=\n\ndef |\n###|\nclass |\Z)'
    content = re.sub(pattern_load_obj, '', content, flags=re.DOTALL)

    # Remove _compute_range_bounds function
    pattern_compute = r'def _compute_range_bounds\(.*?(?=\n\ndef |\nclass |\Z)'
    content = re.sub(pattern_compute, '', content, flags=re.DOTALL)

    # Write refactored content
    with open(file_path, 'w') as f:
        f.write(content)

    # Calculate reduction
    new_lines = len(content.split('\n'))
    reduction = original_lines - new_lines

    print(f"✓ {category:10s}/{name:10s}: {original_lines:4d} → {new_lines:4d} lines (-{reduction:3d}, {reduction/original_lines*100:.1f}%)")

    return reduction


def main():
    """Apply Phase 2 refactoring to all instruments."""

    print("=" * 80)
    print("PHASE 2 REFACTORING - Extract Additional Utility Functions")
    print("=" * 80)
    print()
    print("Extracting 3 additional functions:")
    print("  - _interpolate_color (gauge color calculations)")
    print("  - _load_price_data_from_obj (data loading from file objects)")
    print("  - _compute_range_bounds (volatility-based range estimation)")
    print()

    total_reduction = 0
    instruments_processed = 0

    for category, instruments in INSTRUMENTS.items():
        print(f"\n{category.upper()}:")
        for name in instruments:
            reduction = refactor_instrument_phase2(category, name)
            total_reduction += reduction
            if reduction > 0:
                instruments_processed += 1

    print()
    print("=" * 80)
    print(f"PHASE 2 COMPLETE!")
    print("=" * 80)
    print(f"Instruments processed:  {instruments_processed}")
    print(f"Phase 2 lines removed:  {total_reduction:,}")
    print()
    print(f"COMBINED TOTALS:")
    print(f"  Phase 1: 2,322 lines")
    print(f"  Phase 2: {total_reduction:,} lines")
    print(f"  Total:   {2322 + total_reduction:,} lines removed")
    print()
    print("Backups saved as *_phase2_backup.py")
    print()
    print("NEXT: Test with 'streamlit run app.py'")


if __name__ == '__main__':
    main()
