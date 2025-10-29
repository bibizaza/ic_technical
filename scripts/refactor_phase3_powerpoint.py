#!/usr/bin/env python3
"""
Phase 3 Refactoring: Replace PowerPoint functions with generic versions

This script replaces all instrument-specific PowerPoint manipulation functions
with calls to generic functions from powerpoint_utils.py.

Expected savings: ~7,500 lines
"""

from pathlib import Path
import re
import shutil

BASE_DIR = Path("technical_analysis")

# Instrument configurations
INSTRUMENTS = {
    'equity': [
        {'name': 'spx', 'display': 'S&P 500'},
        {'name': 'csi', 'display': 'CSI 300'},
        {'name': 'dax', 'display': 'DAX'},
        {'name': 'ibov', 'display': 'IBOVESPA'},
        {'name': 'mexbol', 'display': 'MEXBOL'},
        {'name': 'nikkei', 'display': 'NIKKEI'},
        {'name': 'sensex', 'display': 'SENSEX'},
        {'name': 'smi', 'display': 'SMI'},
        {'name': 'tasi', 'display': 'TASI'},
    ],
    'commodity': [
        {'name': 'gold', 'display': 'Gold'},
        {'name': 'silver', 'display': 'Silver'},
        {'name': 'copper', 'display': 'Copper'},
        {'name': 'oil', 'display': 'Oil'},
        {'name': 'palladium', 'display': 'Palladium'},
        {'name': 'platinum', 'display': 'Platinum'},
    ],
    'crypto': [
        {'name': 'bitcoin', 'display': 'Bitcoin'},
        {'name': 'ethereum', 'display': 'Ethereum'},
        {'name': 'binance', 'display': 'Binance'},
        {'name': 'solana', 'display': 'Solana'},
        {'name': 'ripple', 'display': 'Ripple'},
    ],
}


def refactor_powerpoint_functions(category: str, name: str, display: str):
    """Replace PowerPoint functions with generic versions."""

    file_path = BASE_DIR / category / f"{name}.py"
    backup_path = BASE_DIR / category / f"{name}_phase3_backup.py"

    if not file_path.exists():
        print(f"⚠️  {category}/{name}.py not found")
        return 0

    # Read file
    with open(file_path, 'r') as f:
        content = f.read()
        original_lines = len(content.split('\n'))

    # Create backup
    shutil.copy(file_path, backup_path)

    # Add powerpoint_utils import
    old_imports = f"""from technical_analysis.common_helpers import (
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

    new_imports = f"""from technical_analysis.common_helpers import (
    _get_run_font_attributes,
    _apply_run_font_attributes,
    _add_mas,
    _get_technical_score_generic,
    _get_momentum_score_generic,
    _interpolate_color,
    _load_price_data_from_obj,
    _load_price_data_generic,
    _compute_range_bounds,
)
from technical_analysis.powerpoint_utils import (
    find_slide_by_placeholder,
    insert_score_number,
    insert_chart_image,
    insert_subtitle,
    insert_technical_assessment,
    insert_source,
)"""

    content = content.replace(old_imports, new_imports)

    # Replace _find_*_slide with wrapper
    old_find = rf'def _find_{name}_slide\(prs: Presentation\).*?(?=\n\ndef |\nclass |\Z)'
    new_find = f'''def _find_{name}_slide(prs: Presentation) -> Optional[int]:
    """Find the {display} slide by placeholder."""
    return find_slide_by_placeholder(prs, "{name}")

'''
    content = re.sub(old_find, new_find, content, flags=re.DOTALL)

    # Replace insert_*_technical_score_number
    old_tech_score = rf'def insert_{name}_technical_score_number\(prs: Presentation.*?(?=\n\ndef |\n###|\nclass |\Z)'
    new_tech_score = f'''def insert_{name}_technical_score_number(prs: Presentation, excel_file) -> Presentation:
    """Insert the {display} technical score into the slide."""
    score = _get_{name}_technical_score(excel_file)
    return insert_score_number(prs, score, "{name}", "tech_score")

'''
    content = re.sub(old_tech_score, new_tech_score, content, flags=re.DOTALL)

    # Replace insert_*_momentum_score_number
    old_mom_score = rf'def insert_{name}_momentum_score_number\(prs: Presentation.*?(?=\n\ndef |\n###|\nclass |\Z)'
    new_mom_score = f'''def insert_{name}_momentum_score_number(prs: Presentation, excel_file) -> Presentation:
    """Insert the {display} momentum score into the slide."""
    score = _get_{name}_momentum_score(excel_file)
    return insert_score_number(prs, score, "{name}", "momentum_score")

'''
    content = re.sub(old_mom_score, new_mom_score, content, flags=re.DOTALL)

    # Replace insert_*_subtitle
    old_subtitle = rf'def insert_{name}_subtitle\(prs: Presentation.*?(?=\n\ndef |\n###|\nclass |\Z)'
    new_subtitle = f'''def insert_{name}_subtitle(prs: Presentation, subtitle: str) -> Presentation:
    """Insert subtitle into the {display} slide."""
    return insert_subtitle(prs, subtitle, "{name}")

'''
    content = re.sub(old_subtitle, new_subtitle, content, flags=re.DOTALL)

    # Replace insert_*_technical_assessment
    old_assessment = rf'def insert_{name}_technical_assessment\(prs: Presentation.*?(?=\n\ndef |\n###|\nclass |\Z)'
    new_assessment = f'''def insert_{name}_technical_assessment(prs: Presentation, view_text: str) -> Presentation:
    """Insert technical assessment into the {display} slide."""
    return insert_technical_assessment(prs, view_text, "{name}")

'''
    content = re.sub(old_assessment, new_assessment, content, flags=re.DOTALL)

    # Replace insert_*_source
    old_source = rf'def insert_{name}_source\(prs: Presentation.*?(?=\n\ndef |\n###|\nclass |\Z)'
    new_source = f'''def insert_{name}_source(prs: Presentation) -> Presentation:
    """Insert source attribution into the {display} slide."""
    return insert_source(prs, "{name}", "Bloomberg")

'''
    content = re.sub(old_source, new_source, content, flags=re.DOTALL)

    # Write back
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
    print("PHASE 3 REFACTORING - PowerPoint Functions")
    print("=" * 80)
    print()
    print("Replacing 6 PowerPoint functions per instrument with generic wrappers:")
    print("  - _find_*_slide")
    print("  - insert_*_technical_score_number")
    print("  - insert_*_momentum_score_number")
    print("  - insert_*_subtitle")
    print("  - insert_*_technical_assessment")
    print("  - insert_*_source")
    print()

    total_reduction = 0
    instruments_processed = 0

    for category, instruments in INSTRUMENTS.items():
        print(f"\n{category.upper()}:")
        for config in instruments:
            reduction = refactor_powerpoint_functions(category, config['name'], config['display'])
            total_reduction += reduction
            if reduction > 0:
                instruments_processed += 1

    print()
    print("=" * 80)
    print(f"PHASE 3 COMPLETE!")
    print("=" * 80)
    print(f"Instruments processed:  {instruments_processed}")
    print(f"Phase 3 lines removed:  {total_reduction:,}")
    print()
    print(f"CUMULATIVE TOTALS:")
    print(f"  Phase 1: 2,322 lines")
    print(f"  Phase 2: 2,400 lines")
    print(f"  Phase 3: {total_reduction:,} lines")
    print(f"  Total:   {2322 + 2400 + total_reduction:,} lines removed")
    print()


if __name__ == '__main__':
    main()
