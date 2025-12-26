"""
Market Compass Subtitle Generator

Automatically generates chart commentary based on DMAS scores,
technical indicators, and moving average positions.
"""

from .generator import SubtitleGenerator, generate_subtitle
from .patterns import PATTERNS, get_rating

__all__ = [
    'SubtitleGenerator',
    'generate_subtitle',
    'PATTERNS',
    'get_rating'
]

__version__ = '1.0.0'
