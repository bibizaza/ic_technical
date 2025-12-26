"""
Herculis Technical Score - Compute technical indicators from price data.

This package provides Python implementations of technical indicators used
to compute the Technical Score, replicating Bloomberg BQL logic with fixes.
"""

from .scoring import compute_technical_score, compute_all_scores

__version__ = "1.0.0"
__all__ = ["compute_technical_score", "compute_all_scores"]
