"""
Configuration parameters for Technical Score calculation.

This module defines all periods, thresholds, and weights used in computing
the technical score. Centralizing these values makes it easy to tune the
scoring system.
"""

# ============================================================================
# MOVING AVERAGES
# ============================================================================
# Periods for Simple Moving Average (SMA)
SMA_PERIODS = [5, 10, 20, 50, 100, 200]

# Periods for Exponential Moving Average (EMA)
# Note: Removed shorter period (5) compared to SMA
EMA_PERIODS = [10, 20, 50, 100, 200]

# ============================================================================
# RSI (Relative Strength Index)
# ============================================================================
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70  # Above this = overbought (contrarian bearish)
RSI_OVERSOLD = 30    # Below this = oversold (contrarian bullish)

# ============================================================================
# DMI (Directional Movement Index) and ADX
# ============================================================================
DMI_PERIOD = 14
ADX_THRESHOLD = 25  # ADX/ADXR above this indicates strong trend

# ============================================================================
# PARABOLIC SAR
# ============================================================================
PSAR_AF_START = 0.02   # Initial acceleration factor
PSAR_AF_INCREMENT = 0.02
PSAR_AF_MAX = 0.2      # Maximum acceleration factor

# ============================================================================
# MACD (Moving Average Convergence Divergence)
# ============================================================================
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ============================================================================
# STOCHASTICS
# ============================================================================
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
STOCH_NEUTRAL_LOW = 30   # Below this = oversold
STOCH_NEUTRAL_HIGH = 70  # Above this = overbought

# ============================================================================
# MAE (Moving Average Envelope)
# ============================================================================
MAE_PERIOD = 15
MAE_ENVELOPE_PCT = 2.5  # Envelope % above/below MA
MA_OSC_PERIOD = 9       # Period for MA oscillator signal

# ============================================================================
# COMPONENT WEIGHTS
# ============================================================================
# These weights determine how much each indicator contributes to the final
# technical score. They must sum to 1.0.
WEIGHTS = {
    'sma': 0.15,          # Simple Moving Averages
    'ema': 0.15,          # Exponential Moving Averages
    'rsi': 0.10,          # Relative Strength Index (contrarian)
    'dmi': 0.15,          # Directional Movement Index
    'parabolic': 0.10,    # Parabolic SAR (trend-following)
    'macd': 0.15,         # MACD
    'stochastics': 0.10,  # Stochastic Oscillator
    'mae': 0.10,          # Moving Average Envelope (contrarian)
}

# Validate weights sum to 1.0
assert abs(sum(WEIGHTS.values()) - 1.0) < 0.001, "Weights must sum to 1.0"

# ============================================================================
# LOOKBACK PERIOD
# ============================================================================
# Minimum number of data points required for calculation
# This should be at least max(all periods) + some buffer for derivatives
MIN_LOOKBACK_DAYS = 250
