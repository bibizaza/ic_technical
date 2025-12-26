"""
Wrapper to import herculis-technical-score module with proper path handling.

This wrapper handles the relative imports in the herculis-technical-score module
by setting up the correct Python path before importing.
"""

import sys
from pathlib import Path

# Add herculis-technical-score to path
_TECH_SCORE_PATH = Path(__file__).parent / "herculis-technical-score"
sys.path.insert(0, str(_TECH_SCORE_PATH))

# Now import the module - this will handle the relative imports
from src.scoring import compute_dmas_scores

__all__ = ['compute_dmas_scores']
