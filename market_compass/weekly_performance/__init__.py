"""Weekly Performance chart generator for all asset classes."""

from .slide_generator import (
    generate_weekly_performance_png,
    insert_weekly_performance,
    prepare_performance_rows,
    calculate_scale,
    PerformanceRow,
    EQUITY_INDEX_MAP,
)

__all__ = [
    "generate_weekly_performance_png",
    "insert_weekly_performance",
    "prepare_performance_rows",
    "calculate_scale",
    "PerformanceRow",
    "EQUITY_INDEX_MAP",
]
