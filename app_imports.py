"""
Centralized imports for app.py - replaces 1500+ lines of repetitive try/except blocks.

This module imports all necessary functions from instrument modules with proper
fallbacks, eliminating the need for repetitive code in app.py.
"""

import plotly.graph_objects as go
from typing import Optional

# Import all instrument functions
# SPX (Equity)
try:
    from technical_analysis.equity.spx import (
        make_spx_figure,
        insert_spx_technical_chart_with_callout,
        insert_spx_technical_chart,
        insert_spx_technical_score_number,
        insert_spx_momentum_score_number,
        insert_spx_subtitle,
        insert_spx_average_gauge,
        insert_spx_technical_assessment,
        insert_spx_source,
        _get_spx_technical_score,
        _get_spx_momentum_score,
        generate_range_gauge_only_image as generate_range_gauge_only_image_spx,
        _compute_range_bounds as _compute_range_bounds_spx,
    )
except Exception:
    make_spx_figure = lambda *args, **kwargs: go.Figure()
    insert_spx_technical_chart_with_callout = lambda prs, *args, **kwargs: prs
    insert_spx_technical_chart = lambda prs, *args, **kwargs: prs
    insert_spx_technical_score_number = lambda prs, *args, **kwargs: prs
    insert_spx_momentum_score_number = lambda prs, *args, **kwargs: prs
    insert_spx_subtitle = lambda prs, *args, **kwargs: prs
    insert_spx_average_gauge = lambda prs, *args, **kwargs: prs
    insert_spx_technical_assessment = lambda prs, *args, **kwargs: prs
    insert_spx_source = lambda prs, *args, **kwargs: prs
    _get_spx_technical_score = lambda *args, **kwargs: None
    _get_spx_momentum_score = lambda *args, **kwargs: None
    generate_range_gauge_only_image_spx = lambda *args, **kwargs: b""
    _compute_range_bounds_spx = lambda *args, **kwargs: (None, None, None)

# Helper function to create imports for all instruments with same pattern
def _create_instrument_imports(instrument_name: str, module_path: str):
    """Create import dictionary for an instrument."""
    imports = {}
    func_names = [
        f'make_{instrument_name}_figure',
        f'insert_{instrument_name}_technical_chart_with_callout',
        f'insert_{instrument_name}_technical_chart',
        f'insert_{instrument_name}_technical_score_number',
        f'insert_{instrument_name}_momentum_score_number',
        f'insert_{instrument_name}_subtitle',
        f'insert_{instrument_name}_average_gauge',
        f'insert_{instrument_name}_technical_assessment',
        f'insert_{instrument_name}_source',
        f'_get_{instrument_name}_technical_score',
        f'_get_{instrument_name}_momentum_score',
        f'_compute_range_bounds',
    ]

    try:
        module = __import__(module_path, fromlist=func_names)
        for func_name in func_names:
            if func_name == '_compute_range_bounds':
                imports[f'_compute_range_bounds_{instrument_name}'] = getattr(module, func_name, lambda *args, **kwargs: (None, None, None))
            else:
                imports[func_name] = getattr(module, func_name, None)
    except Exception:
        # Create no-op functions
        for func_name in func_names:
            if 'make_' in func_name and '_figure' in func_name:
                imports[func_name] = lambda *args, **kwargs: go.Figure()
            elif func_name.startswith('insert_') or func_name.startswith('_compute'):
                if func_name == '_compute_range_bounds':
                    imports[f'_compute_range_bounds_{instrument_name}'] = lambda *args, **kwargs: (None, None, None)
                else:
                    imports[func_name] = lambda prs, *args, **kwargs: prs
            elif func_name.startswith('_get_'):
                imports[func_name] = lambda *args, **kwargs: None
            else:
                imports[func_name] = lambda *args, **kwargs: None

    return imports

# Equity instruments
EQUITY_INSTRUMENTS = {
    'csi': 'technical_analysis.equity.csi',
    'dax': 'technical_analysis.equity.dax',
    'ibov': 'technical_analysis.equity.ibov',
    'mexbol': 'technical_analysis.equity.mexbol',
    'nikkei': 'technical_analysis.equity.nikkei',
    'sensex': 'technical_analysis.equity.sensex',
    'smi': 'technical_analysis.equity.smi',
    'tasi': 'technical_analysis.equity.tasi',
}

# Commodity instruments
COMMODITY_INSTRUMENTS = {
    'gold': 'technical_analysis.commodity.gold',
    'silver': 'technical_analysis.commodity.silver',
    'platinum': 'technical_analysis.commodity.platinum',
    'palladium': 'technical_analysis.commodity.palladium',
    'copper': 'technical_analysis.commodity.copper',
    'oil': 'technical_analysis.commodity.oil',
}

# Crypto instruments
CRYPTO_INSTRUMENTS = {
    'bitcoin': 'technical_analysis.crypto.bitcoin',
    'ethereum': 'technical_analysis.crypto.ethereum',
    'ripple': 'technical_analysis.crypto.ripple',
    'solana': 'technical_analysis.crypto.solana',
    'binance': 'technical_analysis.crypto.binance',
}

# Import all instruments
all_instruments = {}
all_instruments.update(EQUITY_INSTRUMENTS)
all_instruments.update(COMMODITY_INSTRUMENTS)
all_instruments.update(CRYPTO_INSTRUMENTS)

for inst_name, module_path in all_instruments.items():
    instrument_imports = _create_instrument_imports(inst_name, module_path)
    globals().update(instrument_imports)

# Performance modules
try:
    from performance.equity_perf import (
        create_weekly_equity_performance_chart,
        create_historical_equity_performance_table,
        insert_equity_performance_bar_slide,
        insert_equity_performance_histo_slide,
    )
except Exception:
    create_weekly_equity_performance_chart = lambda *args, **kwargs: (b"", None)
    create_historical_equity_performance_table = lambda *args, **kwargs: (b"", None)
    insert_equity_performance_bar_slide = lambda prs, *args, **kwargs: prs
    insert_equity_performance_histo_slide = lambda prs, *args, **kwargs: prs

try:
    from performance.commodity_perf import (
        create_weekly_commodity_performance_chart,
        create_historical_commodity_performance_table,
        insert_commodity_performance_bar_slide,
        insert_commodity_performance_histo_slide,
    )
except Exception:
    create_weekly_commodity_performance_chart = lambda *args, **kwargs: (b"", None)
    create_historical_commodity_performance_table = lambda *args, **kwargs: (b"", None)
    insert_commodity_performance_bar_slide = lambda prs, *args, **kwargs: prs
    insert_commodity_performance_histo_slide = lambda prs, *args, **kwargs: prs

try:
    from performance.crypto_perf import (
        create_weekly_crypto_performance_chart,
        create_historical_crypto_performance_table,
        insert_crypto_performance_bar_slide,
        insert_crypto_performance_histo_slide,
    )
except Exception:
    create_weekly_crypto_performance_chart = lambda *args, **kwargs: (b"", None)
    create_historical_crypto_performance_table = lambda *args, **kwargs: (b"", None)
    insert_crypto_performance_bar_slide = lambda prs, *args, **kwargs: prs
    insert_crypto_performance_histo_slide = lambda prs, *args, **kwargs: prs

try:
    from performance.credit_perf import (
        create_weekly_credit_performance_chart,
        create_historical_credit_performance_table,
        insert_credit_performance_bar_slide,
        insert_credit_performance_histo_slide,
    )
except Exception:
    create_weekly_credit_performance_chart = lambda *args, **kwargs: (b"", None)
    create_historical_credit_performance_table = lambda *args, **kwargs: (b"", None)
    insert_credit_performance_bar_slide = lambda prs, *args, **kwargs: prs
    insert_credit_performance_histo_slide = lambda prs, *args, **kwargs: prs

try:
    from performance.fx_perf import (
        create_weekly_fx_performance_chart,
        create_historical_fx_performance_table,
        insert_fx_performance_bar_slide,
        insert_fx_performance_histo_slide,
    )
except Exception:
    create_weekly_fx_performance_chart = lambda *args, **kwargs: (b"", None)
    create_historical_fx_performance_table = lambda *args, **kwargs: (b"", None)
    insert_fx_performance_bar_slide = lambda prs, *args, **kwargs: prs
    insert_fx_performance_histo_slide = lambda prs, *args, **kwargs: prs

try:
    from performance.rates_perf import (
        create_weekly_rates_performance_chart,
        create_historical_rates_performance_table,
        insert_rates_performance_bar_slide,
        insert_rates_performance_histo_slide,
    )
except Exception:
    create_weekly_rates_performance_chart = lambda *args, **kwargs: (b"", None)
    create_historical_rates_performance_table = lambda *args, **kwargs: (b"", None)
    insert_rates_performance_bar_slide = lambda prs, *args, **kwargs: prs
    insert_rates_performance_histo_slide = lambda prs, *args, **kwargs: prs

# YTD modules
try:
    from ytd_perf.equity_ytd import handle_equity_ytd_page
except Exception:
    handle_equity_ytd_page = lambda *args, **kwargs: None

try:
    from ytd_perf.commodity_ytd import handle_commodity_ytd_page
except Exception:
    handle_commodity_ytd_page = lambda *args, **kwargs: None

try:
    from ytd_perf.crypto_ytd import handle_crypto_ytd_page
except Exception:
    handle_crypto_ytd_page = lambda *args, **kwargs: None
