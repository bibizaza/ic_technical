"""
Compatibility layer for app.py refactoring.

This module provides a drop-in replacement for all the try/except import blocks
in app.py. Instead of importing from individual instrument modules, app.py can
import from this single module.

Usage in app.py:
    from technical_analysis.compatibility_layer import (
        make_spx_figure, insert_spx_technical_chart,
        make_gold_figure, insert_gold_technical_chart,
        ...
    )
"""

import plotly.graph_objects as go
from technical_analysis.instrument_factory import InstrumentFactory

# Create global factory instance
_factory = InstrumentFactory()

# ============================================================================
# Auto-generate all instrument functions
# ============================================================================

def _create_no_op_functions():
    """Create no-op fallback functions for all instruments."""
    noop_funcs = {}

    all_instruments = ['spx', 'csi', 'dax', 'ibov', 'mexbol', 'nikkei', 'sensex', 'smi', 'tasi',
                       'gold', 'silver', 'copper', 'oil', 'palladium', 'platinum',
                       'bitcoin', 'ethereum', 'binance', 'solana', 'ripple']

    for inst_name in all_instruments:
        # Get the instrument
        inst = _factory.get_instrument(inst_name)

        if inst is None:
            # Create no-op functions
            noop_funcs[f'make_{inst_name}_figure'] = lambda *args, **kwargs: go.Figure()
            noop_funcs[f'insert_{inst_name}_technical_chart'] = lambda prs, *args, **kwargs: prs
            noop_funcs[f'insert_{inst_name}_technical_chart_with_callout'] = lambda prs, *args, **kwargs: prs
            noop_funcs[f'insert_{inst_name}_technical_chart_with_range'] = lambda prs, *args, **kwargs: prs
            noop_funcs[f'insert_{inst_name}_technical_score_number'] = lambda prs, *args, **kwargs: prs
            noop_funcs[f'insert_{inst_name}_momentum_score_number'] = lambda prs, *args, **kwargs: prs
            noop_funcs[f'insert_{inst_name}_subtitle'] = lambda prs, *args, **kwargs: prs
            noop_funcs[f'insert_{inst_name}_average_gauge'] = lambda prs, *args, **kwargs: prs
            noop_funcs[f'insert_{inst_name}_technical_assessment'] = lambda prs, *args, **kwargs: prs
            noop_funcs[f'insert_{inst_name}_source'] = lambda prs, *args, **kwargs: prs
            noop_funcs[f'_get_{inst_name}_technical_score'] = lambda *args, **kwargs: None
            noop_funcs[f'_get_{inst_name}_momentum_score'] = lambda *args, **kwargs: None
            noop_funcs[f'_compute_range_bounds_{inst_name}'] = lambda *args, **kwargs: (None, None, None)
        else:
            # Create wrapper functions that delegate to the instrument
            noop_funcs[f'make_{inst_name}_figure'] = inst.make_figure
            noop_funcs[f'insert_{inst_name}_technical_chart'] = inst.insert_technical_chart
            noop_funcs[f'insert_{inst_name}_technical_score_number'] = inst.insert_technical_score_number
            noop_funcs[f'insert_{inst_name}_momentum_score_number'] = inst.insert_momentum_score_number
            noop_funcs[f'insert_{inst_name}_subtitle'] = inst.insert_subtitle
            noop_funcs[f'_get_{inst_name}_technical_score'] = inst._get_technical_score
            noop_funcs[f'_get_{inst_name}_momentum_score'] = inst._get_momentum_score

            # For functions not yet implemented, use no-ops
            noop_funcs[f'insert_{inst_name}_technical_chart_with_callout'] = lambda prs, *args, **kwargs: prs
            noop_funcs[f'insert_{inst_name}_technical_chart_with_range'] = lambda prs, *args, **kwargs: prs
            noop_funcs[f'insert_{inst_name}_average_gauge'] = lambda prs, *args, **kwargs: prs
            noop_funcs[f'insert_{inst_name}_technical_assessment'] = lambda prs, *args, **kwargs: prs
            noop_funcs[f'insert_{inst_name}_source'] = lambda prs, *args, **kwargs: prs
            noop_funcs[f'_compute_range_bounds_{inst_name}'] = lambda *args, **kwargs: (None, None, None)

    return noop_funcs


# Generate all functions
_all_functions = _create_no_op_functions()

# ============================================================================
# Export all functions to module namespace
# ============================================================================

# This makes them importable: from compatibility_layer import make_spx_figure
globals().update(_all_functions)

# Also provide __all__ for explicit exports
__all__ = list(_all_functions.keys())


# ============================================================================
# Utility functions for app.py
# ============================================================================

def set_all_instruments_lookback_days(days: int):
    """
    Set the lookback days for all instruments.

    This replaces the repetitive module configuration loop in app.py.
    """
    _factory.set_lookback_days(days)


def get_factory():
    """Get the global factory instance."""
    return _factory
