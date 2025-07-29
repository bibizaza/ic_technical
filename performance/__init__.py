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
from .fx_perf import (
    create_weekly_performance_chart as create_weekly_fx_performance_chart,
    create_historical_performance_table as create_historical_fx_performance_table,
    insert_fx_performance_bar_slide,
    insert_fx_performance_histo_slide,
)

__all__ = [
    # Equity functions
    "create_weekly_performance_chart",
    "create_historical_performance_table",
    "insert_equity_performance_bar_slide",
    "insert_equity_performance_histo_slide",
    # FX functions (aliased to avoid name clash)
    "create_weekly_fx_performance_chart",
    "create_historical_fx_performance_table",
    "insert_fx_performance_bar_slide",
    "insert_fx_performance_histo_slide",
]