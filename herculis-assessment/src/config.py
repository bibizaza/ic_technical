"""
Assessment classification configuration.

This module defines the 5-level assessment scale and all thresholds used
for classifying assets based on DMAS scores, price structure, and momentum.
"""

from enum import IntEnum


class Assessment(IntEnum):
    """
    5-level assessment scale from Bearish to Bullish.

    Values are integers to allow arithmetic operations for upgrades/downgrades.
    """
    BEARISH = -2
    CAUTIOUS = -1
    NEUTRAL = 0
    CONSTRUCTIVE = 1
    BULLISH = 2


# Display names for output
ASSESSMENT_LABELS = {
    Assessment.BEARISH: "Bearish",
    Assessment.CAUTIOUS: "Cautious",
    Assessment.NEUTRAL: "Neutral",
    Assessment.CONSTRUCTIVE: "Constructive",
    Assessment.BULLISH: "Bullish"
}

# Short names for compact display
ASSESSMENT_SHORT_LABELS = {
    Assessment.BEARISH: "Bear",
    Assessment.CAUTIOUS: "Caut",
    Assessment.NEUTRAL: "Neut",
    Assessment.CONSTRUCTIVE: "Cons",
    Assessment.BULLISH: "Bull"
}

# Colors for visualization (Streamlit/matplotlib)
ASSESSMENT_COLORS = {
    Assessment.BEARISH: "#FF0000",      # Red
    Assessment.CAUTIOUS: "#FF8C00",     # Dark Orange
    Assessment.NEUTRAL: "#FFD700",      # Gold
    Assessment.CONSTRUCTIVE: "#90EE90", # Light Green
    Assessment.BULLISH: "#009951"       # Green
}

# ============================================================================
# DMAS THRESHOLDS FOR BASE CLASSIFICATION
# ============================================================================
# These thresholds determine the starting assessment based solely on DMAS score
# before any adjustments are applied.

DMAS_THRESHOLDS = {
    'bullish': 70,        # DMAS >= 70 → Bullish
    'constructive': 55,   # DMAS 55-69 → Constructive
    'neutral': 45,        # DMAS 45-54 → Neutral
    'cautious': 30,       # DMAS 30-44 → Cautious
    # DMAS < 30 → Bearish
}

# ============================================================================
# MOVING AVERAGE PERIODS
# ============================================================================
# Key moving averages used for structure analysis

MA_PERIODS = {
    'short': 50,
    'medium': 100,
    'long': 200
}

# ============================================================================
# ADJUSTMENT THRESHOLDS
# ============================================================================
# Thresholds that trigger upgrades or downgrades from the base assessment

ADJUSTMENT_CONFIG = {
    # Week-over-week DMAS change threshold (absolute points)
    'wow_change_threshold': 10,      # ±10 points triggers adjustment

    # Minimum DMAS required for structure-based upgrade to Bullish
    'structure_upgrade_min_dmas': 65,

    # Distance from MA thresholds (percentage)
    'close_to_ma_threshold': 2.0,    # Within 2% = "close to MA"
}

# ============================================================================
# ADJUSTMENT RULES DOCUMENTATION
# ============================================================================
"""
DOWNGRADE RULES (applied in order):
1. Price < 200d MA: Hard cap at Cautious (cannot be above Cautious)
2. Price < 100d MA: Downgrade 1 level
3. Price < 50d MA: Downgrade 1 level (stacks with #2)
4. DMAS dropped ≥10 pts WoW: Downgrade 1 level

UPGRADE RULES (applied after downgrades):
1. Perfect structure (Price > 50d > 100d > 200d) AND DMAS ≥65: Upgrade to Bullish
2. DMAS gained ≥10 pts WoW: Upgrade 1 level

Final assessment is clamped to [-2, +2] range (Bearish to Bullish).
"""
