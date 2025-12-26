"""
Test suite for technical score calculations.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scoring import compute_technical_score
from src.indicators import (
    compute_sma, compute_ema, score_ma,
    compute_rsi, score_rsi,
    score_dmi, score_macd, score_stochastics
)


def generate_sample_data(length=300, trend='bullish'):
    """
    Generate sample price data for testing.

    Parameters
    ----------
    length : int
        Number of data points
    trend : str
        'bullish', 'bearish', or 'neutral'

    Returns
    -------
    pd.DataFrame
        Sample price data with Date, Price, High, Low
    """
    dates = pd.date_range('2020-01-01', periods=length, freq='D')

    if trend == 'bullish':
        # Uptrending prices with noise
        base = np.linspace(100, 150, length)
        noise = np.random.normal(0, 2, length)
        prices = base + noise
    elif trend == 'bearish':
        # Downtrending prices with noise
        base = np.linspace(150, 100, length)
        noise = np.random.normal(0, 2, length)
        prices = base + noise
    else:  # neutral
        # Sideways prices with noise
        base = np.full(length, 125)
        noise = np.random.normal(0, 5, length)
        prices = base + noise

    # Ensure positive prices
    prices = np.maximum(prices, 50)

    # Generate High/Low
    highs = prices * 1.01  # 1% higher
    lows = prices * 0.99   # 1% lower

    return pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'High': highs,
        'Low': lows
    })


class TestIndicators:
    """Test individual indicator calculations."""

    def test_sma_calculation(self):
        """Test SMA calculation."""
        prices = pd.Series([100, 102, 104, 106, 108, 110])
        sma = compute_sma(prices, period=3)

        # First two values should be NaN (not enough data)
        assert pd.isna(sma.iloc[0])
        assert pd.isna(sma.iloc[1])

        # Third value should be average of first 3
        assert sma.iloc[2] == pytest.approx(102.0, abs=0.01)

    def test_ema_calculation(self):
        """Test EMA calculation."""
        prices = pd.Series([100, 102, 104, 106, 108, 110])
        ema = compute_ema(prices, period=3)

        # First two values should be NaN
        assert pd.isna(ema.iloc[0])
        assert pd.isna(ema.iloc[1])

        # EMA values should exist after period
        assert not pd.isna(ema.iloc[2])

    def test_rsi_contrarian_scoring(self):
        """Test RSI contrarian scoring logic."""
        # Overbought (>70) should return 0.0 (bearish)
        assert score_rsi(75) == 0.0

        # Oversold (<30) should return 1.0 (bullish)
        assert score_rsi(25) == 1.0

        # Neutral (30-70) should return 0.5
        assert score_rsi(50) == 0.5

    def test_ma_scoring(self):
        """Test moving average scoring."""
        # Price above all MAs = bullish
        ma_values = {5: 95, 10: 93, 20: 90, 50: 85, 100: 80, 200: 75}
        score = score_ma(100, ma_values)
        assert score == 1.0

        # Price below all MAs = bearish
        score = score_ma(70, ma_values)
        assert score == 0.0

        # Price above half of MAs = neutral
        ma_values_mixed = {5: 105, 10: 103, 20: 98, 50: 97}
        score = score_ma(100, ma_values_mixed)
        assert 0.4 <= score <= 0.6  # Approximately neutral


class TestScoring:
    """Test complete scoring system."""

    def test_bullish_scenario(self):
        """Test that bullish data produces high score."""
        np.random.seed(42)  # For reproducibility
        df = generate_sample_data(length=300, trend='bullish')

        result = compute_technical_score(df, 'TEST', include_components=True)

        # Bullish trend should produce score > 50
        assert result['technical_score'] > 50, \
            f"Bullish scenario should score > 50, got {result['technical_score']}"

        # Check that we got all components
        assert 'components' in result
        assert len(result['components']) == 8  # All 8 indicators

    def test_bearish_scenario(self):
        """Test that bearish data produces low score."""
        np.random.seed(42)
        df = generate_sample_data(length=300, trend='bearish')

        result = compute_technical_score(df, 'TEST', include_components=True)

        # Bearish trend should produce score < 50
        assert result['technical_score'] < 50, \
            f"Bearish scenario should score < 50, got {result['technical_score']}"

    def test_neutral_scenario(self):
        """Test that sideways data produces neutral score."""
        np.random.seed(42)
        df = generate_sample_data(length=300, trend='neutral')

        result = compute_technical_score(df, 'TEST', include_components=True)

        # Neutral trend should produce score around 50
        assert 30 < result['technical_score'] < 70, \
            f"Neutral scenario should score 30-70, got {result['technical_score']}"

    def test_insufficient_data(self):
        """Test that insufficient data raises error."""
        df = generate_sample_data(length=50, trend='bullish')  # Too short

        with pytest.raises(ValueError, match="Insufficient data"):
            compute_technical_score(df, 'TEST')

    def test_missing_columns(self):
        """Test that missing columns raise error."""
        df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=300),
            'Price': np.random.randn(300) + 100
            # Missing High and Low
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            compute_technical_score(df, 'TEST')

    def test_score_range(self):
        """Test that scores are in valid range [0, 100]."""
        np.random.seed(42)
        df = generate_sample_data(length=300, trend='bullish')

        result = compute_technical_score(df, 'TEST')

        assert 0 <= result['technical_score'] <= 100, \
            f"Score must be 0-100, got {result['technical_score']}"

    def test_component_weights_sum(self):
        """Test that component weights sum to 1.0."""
        from config import WEIGHTS

        total_weight = sum(WEIGHTS.values())
        assert abs(total_weight - 1.0) < 0.001, \
            f"Weights must sum to 1.0, got {total_weight}"


class TestDMIScoring:
    """Test DMI scoring logic."""

    def test_dmi_all_bullish(self):
        """Test DMI scoring when all conditions are bullish."""
        dmi_data = {
            'plus_di': 30,
            'minus_di': 20,
            'adx': 28,
            'adx_1w': 25,
            'adxr': 27,
            'adxr_1w': 24
        }
        score = score_dmi(dmi_data, adx_threshold=25)
        assert score == 1.0

    def test_dmi_all_bearish(self):
        """Test DMI scoring when all conditions are bearish."""
        dmi_data = {
            'plus_di': 20,
            'minus_di': 30,
            'adx': 20,
            'adx_1w': 25,
            'adxr': 20,
            'adxr_1w': 24
        }
        score = score_dmi(dmi_data, adx_threshold=25)
        assert score == 0.0


class TestMACDScoring:
    """Test MACD scoring logic."""

    def test_macd_bullish(self):
        """Test MACD scoring with bullish crossover and improving histogram."""
        macd_data = {
            'macd': 1.5,
            'signal': 1.0,
            'histogram': 0.5,
            'histogram_1w': 0.3
        }
        score = score_macd(macd_data)
        assert score == 1.0

    def test_macd_bearish(self):
        """Test MACD scoring with bearish crossover and weakening histogram."""
        macd_data = {
            'macd': 1.0,
            'signal': 1.5,
            'histogram': -0.5,
            'histogram_1w': -0.3
        }
        score = score_macd(macd_data)
        assert score == 0.0


class TestStochasticsScoring:
    """Test Stochastic scoring logic."""

    def test_stochastics_all_bullish(self):
        """Test stochastics with all bullish conditions."""
        stoch_data = {
            'k': 55,      # In neutral zone
            'k_1w': 25,   # Was oversold
            'd': 50,
            'd_1w': 22
        }
        score = score_stochastics(stoch_data, neutral_low=30, neutral_high=70)
        assert score == 1.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])
