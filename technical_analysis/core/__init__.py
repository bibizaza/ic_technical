"""
Core module for Technical Analysis V2 slides.

This module provides shared components for generating technical analysis charts
in a config-driven architecture.

Components:
- config: Shared configuration (colors, dimensions, instrument configs)
- indicators: Technical indicator calculations (RSI, MA, Fibonacci)
- chart_builder: Universal chart generation logic

Usage:
    from technical_analysis.core import INSTRUMENT_CONFIGS, create_v2_chart
"""

from technical_analysis.core.config import (
    INSTRUMENT_CONFIGS,
    DEFAULT_COLORS,
    DEFAULT_MA_PERIODS,
)
from technical_analysis.core.indicators import (
    compute_rsi,
    compute_fibonacci_levels,
    get_score_status,
    get_rsi_interpretation,
)
