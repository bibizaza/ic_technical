"""Performance module package.

This package aggregates functions for generating and inserting performance
dashboards across asset classes.  See ``equity_perf.py`` for equity
performance charts and slide insertion helpers.
"""

from .equity_perf import (
    create_weekly_performance_chart,
    create_historical_performance_table,
    insert_equity_performance_bar_slide,
    insert_equity_performance_histo_slide,
)

__all__ = [
    "create_weekly_performance_chart",
    "create_historical_performance_table",
    "insert_equity_performance_bar_slide",
    "insert_equity_performance_histo_slide",
]