#!/usr/bin/env python3
"""
Analyze how much code can be consolidated by creating generic versions.

This script identifies all the duplicate patterns across instruments and
estimates the potential line reduction.
"""

from pathlib import Path
import re

def count_lines(file_path, func_name):
    """Count lines in a specific function."""
    with open(file_path) as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        if f'def {func_name}(' in line:
            start = i
            break

    if start is None:
        return 0

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith('def '):
            end = i
            break

    return end - start


# Function patterns that appear in all instruments
COMMON_PATTERNS = [
    ('make_*_figure', 'Chart generation'),
    ('_generate_*_image_from_df', 'Image generation from DataFrame'),
    ('insert_*_technical_score_number', 'Insert technical score'),
    ('insert_*_momentum_score_number', 'Insert momentum score'),
    ('insert_*_technical_chart', 'Insert technical chart'),
    ('insert_*_technical_chart_with_callout', 'Insert chart with callout'),
    ('insert_*_technical_chart_with_range', 'Insert chart with range'),
    ('insert_*_subtitle', 'Insert subtitle'),
    ('insert_*_average_gauge', 'Insert average gauge'),
    ('insert_*_technical_assessment', 'Insert technical assessment'),
    ('insert_*_source', 'Insert source'),
    ('generate_average_gauge_image', 'Generate gauge image'),
    ('generate_range_gauge_chart_image', 'Generate range gauge'),
    ('generate_range_callout_chart_image', 'Generate callout chart'),
    ('_find_*_slide', 'Find slide by placeholder'),
]

# All instruments
instruments = {
    'equity': ['spx', 'csi', 'dax', 'ibov', 'mexbol', 'nikkei', 'sensex', 'smi', 'tasi'],
    'commodity': ['gold', 'silver', 'copper', 'oil', 'palladium', 'platinum'],
    'crypto': ['bitcoin', 'ethereum', 'binance', 'solana', 'ripple'],
}

print("=" * 80)
print("CONSOLIDATION POTENTIAL ANALYSIS")
print("=" * 80)
print()

total_potential = 0

# Sample SPX to get baseline line counts
spx_path = Path('technical_analysis/equity/spx.py')

for pattern, description in COMMON_PATTERNS:
    # Get SPX version
    spx_func = pattern.replace('*', 'spx')
    lines = count_lines(spx_path, spx_func)

    if lines > 0:
        # This pattern exists - calculate potential savings
        # We keep 1 generic version, remove from 19 others
        potential_savings = lines * 19  # 20 instruments - 1 generic
        total_potential += potential_savings

        print(f"{description:40s}: {lines:3d} lines × 19 = {potential_savings:5d} lines")

print()
print("=" * 80)
print(f"TOTAL POTENTIAL REDUCTION: {total_potential:,} lines")
print("=" * 80)
print()
print("Current instrument average: ~2,000 lines")
print(f"After consolidation: ~{2000 - (total_potential // 20):,} lines per instrument")
print()
print("This would be in ADDITION to the 4,722 lines already removed in Phases 1 & 2")
