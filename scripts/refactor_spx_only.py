#!/usr/bin/env python3
"""
Refactor SPX to use common_helpers.py

This script surgically replaces duplicate code in spx.py with imports from
common_helpers.py. It keeps all other logic identical.

Steps:
1. Add import of common helpers
2. Remove duplicate _get_run_font_attributes
3. Remove duplicate _apply_run_font_attributes
4. Remove duplicate _add_mas
5. Replace _get_spx_technical_score with call to generic version
6. Replace _get_spx_momentum_score with call to generic version

This should reduce SPX from ~2232 lines to ~2000 lines while keeping exact behavior.
"""

import re
from pathlib import Path

SPX_FILE = Path("technical_analysis/equity/spx.py")
BACKUP_FILE = Path("technical_analysis/equity/spx_before_refactor.py")


def refactor_spx():
    """Refactor SPX file to use common helpers."""

    print("Reading original SPX file...")
    with open(SPX_FILE, 'r') as f:
        content = f.read()
        lines = content.split('\n')

    print(f"Original: {len(lines)} lines")

    # Backup original
    with open(BACKUP_FILE, 'w') as f:
        f.write(content)
    print(f"✓ Backed up to {BACKUP_FILE}")

    # Find where to add imports (after other imports)
    import_insert_line = None
    for i, line in enumerate(lines):
        if line.startswith('from pptx import Presentation'):
            import_insert_line = i + 1
            break

    if import_insert_line is None:
        print("ERROR: Couldn't find import section")
        return

    # Add common helpers import
    new_import = """
# Import common helpers (eliminates code duplication)
from technical_analysis.common_helpers import (
    _get_run_font_attributes,
    _apply_run_font_attributes,
    _add_mas,
    _get_technical_score_generic,
    _get_momentum_score_generic,
)
"""

    lines.insert(import_insert_line, new_import)

    # Find and remove _get_run_font_attributes function
    content = '\n'.join(lines)

    # Pattern to find the function
    pattern_get_font = r'def _get_run_font_attributes\(run\):.*?(?=\n\ndef |\nclass |\Z)'
    content = re.sub(pattern_get_font, '', content, flags=re.DOTALL)

    # Remove _apply_run_font_attributes
    pattern_apply_font = r'def _apply_run_font_attributes\(new_run.*?\):.*?(?=\n\ndef |\nclass |\Z)'
    content = re.sub(pattern_apply_font, '', content, flags=re.DOTALL)

    # Remove _add_mas
    pattern_add_mas = r'def _add_mas\(df: pd\.DataFrame\).*?(?=\n\ndef |\nclass |\Z)'
    content = re.sub(pattern_add_mas, '', content, flags=re.DOTALL)

    # Replace _get_spx_technical_score
    old_tech_score = r'def _get_spx_technical_score\(excel_obj_or_path\).*?(?=\n\ndef )'
    new_tech_score = '''def _get_spx_technical_score(excel_obj_or_path) -> Optional[float]:
    """
    Retrieve the technical score for S&P 500.
    Uses common helper with SPX-specific ticker.
    """
    return _get_technical_score_generic(excel_obj_or_path, "SPX INDEX")

'''
    content = re.sub(old_tech_score, new_tech_score, content, flags=re.DOTALL)

    # Replace _get_spx_momentum_score
    old_mom_score = r'def _get_spx_momentum_score\(excel_obj_or_path\).*?(?=\n\ndef )'
    new_mom_score = '''def _get_spx_momentum_score(excel_obj_or_path) -> Optional[float]:
    """
    Retrieve the momentum score for S&P 500.
    Uses common helper with SPX-specific ticker.
    """
    return _get_momentum_score_generic(excel_obj_or_path, "SPX INDEX")

'''
    content = re.sub(old_mom_score, new_mom_score, content, flags=re.DOTALL)

    # Write refactored version
    with open(SPX_FILE, 'w') as f:
        f.write(content)

    new_lines = len(content.split('\n'))
    reduction = len(lines) - new_lines

    print(f"✓ Refactored SPX file")
    print(f"  Before: {len(lines)} lines")
    print(f"  After: {new_lines} lines")
    print(f"  Reduction: {reduction} lines")

    print()
    print("=" * 70)
    print("SPX REFACTORED!")
    print("=" * 70)
    print()
    print("NEXT STEP: Test SPX thoroughly before applying to other instruments")
    print()
    print("Test with: streamlit run app.py")
    print("  1. Select SPX")
    print("  2. Check technical score displays correctly")
    print("  3. Check momentum score displays correctly")
    print("  4. Check chart looks correct")
    print()
    print(f"If issues, restore with: cp {BACKUP_FILE} {SPX_FILE}")


if __name__ == '__main__':
    refactor_spx()
