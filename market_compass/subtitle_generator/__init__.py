"""
Market Compass Subtitle Generator

Automatically generates chart commentary based on DMAS scores,
technical indicators, and moving average positions.

Rating Vocabulary:
- Positive: Bullish, strong momentum and/or technical (DMAS >= 70)
- Constructive: Positive outlook, favorable technical setup (DMAS 55-69)
- Neutral: Mixed signals, no clear direction (DMAS 45-54)
- Cautious: Negative bias, warning signs present (DMAS 30-44)
- Negative: Bearish, weak technical and/or momentum (DMAS < 30)
"""

from .generator import SubtitleGenerator, SubtitleTracker, generate_subtitle, validate_subtitle_language
from .patterns import PATTERNS, get_rating, get_ma_dynamics, get_high_low_dynamics, add_context_if_needed

__all__ = [
    'SubtitleGenerator',
    'SubtitleTracker',
    'generate_subtitle',
    'validate_subtitle_language',
    'PATTERNS',
    'get_rating',
    'get_ma_dynamics',
    'get_high_low_dynamics',
    'add_context_if_needed',
]

__version__ = '3.0.0'
