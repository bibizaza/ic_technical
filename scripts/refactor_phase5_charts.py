#!/usr/bin/env python3
"""
Phase 5 Refactoring: Move more identical chart generation functions

This script moves functions that are 99.9% identical (only docstring differences)
from all instruments to common_helpers.py.

Functions to move:
1. generate_range_gauge_only_image (117 lines × 20 = 2,223 lines)
2. generate_range_gauge_chart_image (252 lines × 20 = 4,788 lines)
3. generate_range_callout_chart_image (277 lines × 20 = 5,263 lines)

Total expected savings: ~12,000 lines
"""

from pathlib import Path
import re
import shutil

BASE_DIR = Path("technical_analysis")

# All instruments
instruments = []
for category in ['equity', 'commodity', 'crypto']:
    category_path = BASE_DIR / category
    for file in category_path.glob('*.py'):
        if not file.name.endswith('_backup.py') and file.name != 'spx_refactored.py':
            instruments.append(file)

# Functions that are identical (except docstrings)
IDENTICAL_FUNCTIONS = [
    'generate_range_gauge_only_image',
    'generate_range_gauge_chart_image',
    'generate_range_callout_chart_image',
]


def refactor_instrument_phase5(file_path: Path):
    """Remove identical chart functions and add import."""

    # Read file
    with open(file_path) as f:
        content = f.read()
        original_lines = len(content.split('\n'))

    # Create backup
    backup_path = file_path.with_name(file_path.stem + '_phase5_backup.py')
    shutil.copy(file_path, backup_path)

    # Update imports to include the chart functions
    old_imports = """from technical_analysis.common_helpers import (
    _get_run_font_attributes,
    _apply_run_font_attributes,
    _add_mas,
    _get_technical_score_generic,
    _get_momentum_score_generic,
    _interpolate_color,
    _load_price_data_from_obj,
    _load_price_data_generic,
    _compute_range_bounds,
    generate_average_gauge_image,
)"""

    new_imports = """from technical_analysis.common_helpers import (
    _get_run_font_attributes,
    _apply_run_font_attributes,
    _add_mas,
    _get_technical_score_generic,
    _get_momentum_score_generic,
    _interpolate_color,
    _load_price_data_from_obj,
    _load_price_data_generic,
    _compute_range_bounds,
    generate_average_gauge_image,
    generate_range_gauge_only_image,
    generate_range_gauge_chart_image,
    generate_range_callout_chart_image,
)"""

    content = content.replace(old_imports, new_imports)

    # Remove the identical functions
    for func_name in IDENTICAL_FUNCTIONS:
        # Match function from def to next def at column 0
        pattern = rf'def {func_name}\(.*?(?=\n\ndef [a-z_]|\nclass |\Z)'
        content = re.sub(pattern, '', content, flags=re.DOTALL)

    # Write back
    with open(file_path, 'w') as f:
        f.write(content)

    # Calculate reduction
    new_lines = len(content.split('\n'))
    reduction = original_lines - new_lines

    category = file_path.parent.name
    name = file_path.stem
    if reduction > 0:
        print(f"✓ {category:10s}/{name:10s}: {original_lines:4d} → {new_lines:4d} lines (-{reduction:3d}, {reduction/original_lines*100:.1f}%)")

    return reduction


def main():
    """Refactor all instruments."""

    print("=" * 80)
    print("PHASE 5 REFACTORING - Move Identical Chart Functions")
    print("=" * 80)
    print()
    print("Moving 3 chart generation functions to common_helpers.py:")
    for func in IDENTICAL_FUNCTIONS:
        print(f"  - {func}")
    print()

    total_reduction = 0

    for file_path in sorted(instruments):
        reduction = refactor_instrument_phase5(file_path)
        total_reduction += reduction

    print()
    print("=" * 80)
    print(f"PHASE 5 COMPLETE!")
    print("=" * 80)
    print(f"Phase 5 lines removed:  {total_reduction:,}")
    print()
    print(f"CUMULATIVE TOTALS:")
    print(f"  Phase 1: 2,322 lines")
    print(f"  Phase 2: 2,400 lines")
    print(f"  Phase 3: 2,795 lines")
    print(f"  Phase 4: 2,887 lines")
    print(f"  Phase 5: {total_reduction:,} lines")
    print(f"  Total:   {10404 + total_reduction:,} lines removed")
    print()

    # Calculate current avg lines per instrument
    avg_lines = sum(len(open(f).readlines()) for f in instruments) / len(instruments)
    print(f"Current average lines per instrument: {avg_lines:.0f}")


if __name__ == '__main__':
    main()
