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

from .crypto_perf import (
    create_weekly_performance_chart as create_weekly_crypto_performance_chart,
    create_historical_performance_table as create_historical_crypto_performance_table,
    insert_crypto_performance_bar_slide,
    insert_crypto_performance_histo_slide,
)

# Import rates performance functions (aliased to avoid name clash)
try:
    from .rates_perf import (
        create_weekly_performance_chart as create_weekly_rates_performance_chart,
        create_historical_performance_table as create_historical_rates_performance_table,
        insert_rates_performance_bar_slide,
        insert_rates_performance_histo_slide,
    )
except Exception:
    # Provide no-op fallbacks if the rates module cannot be imported
    def create_weekly_rates_performance_chart(*args, **kwargs):  # type: ignore
        return (b"", None)
    def create_historical_rates_performance_table(*args, **kwargs):  # type: ignore
        return (b"", None)
    def insert_rates_performance_bar_slide(prs, image_bytes, *args, **kwargs):  # type: ignore
        return prs
    def insert_rates_performance_histo_slide(prs, image_bytes, *args, **kwargs):  # type: ignore
        return prs

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
    # Crypto performance functions
    "create_weekly_crypto_performance_chart",
    "create_historical_crypto_performance_table",
    "insert_crypto_performance_bar_slide",
    "insert_crypto_performance_histo_slide",
    # Rates performance functions
    "create_weekly_rates_performance_chart",
    "create_historical_rates_performance_table",
    "insert_rates_performance_bar_slide",
    "insert_rates_performance_histo_slide",
]