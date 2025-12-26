"""
Herculis Assessment - Automatic asset classification system.

This package provides automatic classification of assets into 5 assessment levels
(Bearish → Cautious → Neutral → Constructive → Bullish) based on:
- DMAS score (base classification)
- Price structure relative to moving averages
- Momentum direction (week-over-week DMAS change)
"""

from .config import Assessment, ASSESSMENT_LABELS, ASSESSMENT_COLORS
from .classifier import classify, classify_all, ClassificationResult
from .structure import StructureAnalysis, analyze_structure

__version__ = "1.0.0"
__all__ = [
    "Assessment",
    "ASSESSMENT_LABELS",
    "ASSESSMENT_COLORS",
    "classify",
    "classify_all",
    "ClassificationResult",
    "StructureAnalysis",
    "analyze_structure",
]
