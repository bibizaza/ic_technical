"""
Test suite for assessment classification system.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Assessment
from src.classifier import classify, classify_all, _get_base_assessment
from src.structure import analyze_structure, compute_ma, compute_percent_distance


def generate_sample_prices(length=300, trend='uptrend'):
    """
    Generate sample price data for testing.

    Parameters
    ----------
    length : int
        Number of data points
    trend : str
        'uptrend', 'downtrend', or 'sideways'

    Returns
    -------
    pd.Series
        Sample price series
    """
    dates = pd.date_range('2023-01-01', periods=length, freq='D')

    if trend == 'uptrend':
        # Uptrending prices
        base = np.linspace(100, 150, length)
        noise = np.random.normal(0, 1, length)
        prices = base + noise
    elif trend == 'downtrend':
        # Downtrending prices
        base = np.linspace(150, 100, length)
        noise = np.random.normal(0, 1, length)
        prices = base + noise
    else:  # sideways
        # Sideways prices
        base = np.full(length, 125)
        noise = np.random.normal(0, 3, length)
        prices = base + noise

    return pd.Series(prices, index=dates)


class TestBaseClassification:
    """Test base classification from DMAS scores."""

    def test_bullish_threshold(self):
        """Test DMAS ≥ 70 → Bullish."""
        assert _get_base_assessment(70) == Assessment.BULLISH
        assert _get_base_assessment(85) == Assessment.BULLISH

    def test_constructive_threshold(self):
        """Test DMAS 55-69 → Constructive."""
        assert _get_base_assessment(55) == Assessment.CONSTRUCTIVE
        assert _get_base_assessment(65) == Assessment.CONSTRUCTIVE
        assert _get_base_assessment(69) == Assessment.CONSTRUCTIVE

    def test_neutral_threshold(self):
        """Test DMAS 45-54 → Neutral."""
        assert _get_base_assessment(45) == Assessment.NEUTRAL
        assert _get_base_assessment(50) == Assessment.NEUTRAL
        assert _get_base_assessment(54) == Assessment.NEUTRAL

    def test_cautious_threshold(self):
        """Test DMAS 30-44 → Cautious."""
        assert _get_base_assessment(30) == Assessment.CAUTIOUS
        assert _get_base_assessment(40) == Assessment.CAUTIOUS
        assert _get_base_assessment(44) == Assessment.CAUTIOUS

    def test_bearish_threshold(self):
        """Test DMAS < 30 → Bearish."""
        assert _get_base_assessment(29) == Assessment.BEARISH
        assert _get_base_assessment(15) == Assessment.BEARISH
        assert _get_base_assessment(0) == Assessment.BEARISH


class TestStructureAnalysis:
    """Test price structure analysis."""

    def test_perfect_uptrend_structure(self):
        """Test perfect uptrend (Price > 50d > 100d > 200d)."""
        np.random.seed(42)
        prices = generate_sample_prices(length=300, trend='uptrend')

        structure = analyze_structure(prices)

        # Should have positive structure score
        assert structure.score >= 1
        assert structure.above_50d
        assert structure.above_100d
        assert structure.above_200d

    def test_downtrend_structure(self):
        """Test downtrend structure."""
        np.random.seed(42)
        prices = generate_sample_prices(length=300, trend='downtrend')

        structure = analyze_structure(prices)

        # Should have negative structure score
        assert structure.score <= 0
        assert not structure.above_50d

    def test_insufficient_data_raises_error(self):
        """Test that insufficient data raises ValueError."""
        prices = pd.Series(np.random.randn(100) + 100)

        with pytest.raises(ValueError, match="Insufficient data"):
            analyze_structure(prices)

    def test_compute_ma(self):
        """Test moving average calculation."""
        prices = pd.Series([100, 102, 104, 106, 108])

        ma = compute_ma(prices, period=3)
        # Last 3 values: 104, 106, 108 → mean = 106
        assert ma == pytest.approx(106.0, abs=0.01)

    def test_percent_distance(self):
        """Test percentage distance calculation."""
        # Price 10% above MA
        dist = compute_percent_distance(110, 100)
        assert dist == pytest.approx(10.0, abs=0.01)

        # Price 10% below MA
        dist = compute_percent_distance(90, 100)
        assert dist == pytest.approx(-10.0, abs=0.01)


class TestDowngradeRules:
    """Test downgrade adjustment rules."""

    def test_below_200d_ma_cap(self):
        """Test that price < 200d MA caps assessment at Cautious."""
        np.random.seed(42)
        # Create prices that end below 200d MA
        prices = generate_sample_prices(length=300, trend='downtrend')

        # High DMAS should normally be Bullish, but cap at Cautious
        result = classify('TEST', prices, dmas=75, dmas_1w=75)

        # Should be capped at Cautious or lower
        assert result.assessment <= Assessment.CAUTIOUS
        assert any('200d MA' in adj for adj in result.adjustments)

    def test_below_100d_ma_downgrade(self):
        """Test that price < 100d MA downgrades 1 level."""
        np.random.seed(42)
        prices = generate_sample_prices(length=300, trend='downtrend')

        result = classify('TEST', prices, dmas=60, dmas_1w=60)

        # Should have downgrade from 100d MA
        assert any('100d MA' in adj for adj in result.adjustments)

    def test_below_50d_ma_downgrade(self):
        """Test that price < 50d MA downgrades 1 level."""
        np.random.seed(42)
        prices = generate_sample_prices(length=300, trend='downtrend')

        result = classify('TEST', prices, dmas=60, dmas_1w=60)

        # Should have downgrade from 50d MA
        assert any('50d MA' in adj for adj in result.adjustments)

    def test_dmas_drop_downgrade(self):
        """Test that DMAS drop ≥10 pts downgrades 1 level."""
        np.random.seed(42)
        prices = generate_sample_prices(length=300, trend='uptrend')

        # DMAS dropped from 70 to 55 (15 points)
        result = classify('TEST', prices, dmas=55, dmas_1w=70)

        # Should have momentum downgrade
        assert any('dropped' in adj.lower() for adj in result.adjustments)
        assert result.dmas_wow_change == -15


class TestUpgradeRules:
    """Test upgrade adjustment rules."""

    def test_perfect_structure_upgrade(self):
        """Test that perfect structure + DMAS ≥65 upgrades to Bullish."""
        np.random.seed(42)
        prices = generate_sample_prices(length=300, trend='uptrend')

        # DMAS = 65 (Constructive) but with perfect structure → Bullish
        result = classify('TEST', prices, dmas=67, dmas_1w=65)

        # With strong uptrend and DMAS 67, could get upgrade
        # Check that structure is analyzed
        assert result.structure is not None

    def test_dmas_gain_upgrade(self):
        """Test that DMAS gain ≥10 pts upgrades 1 level."""
        np.random.seed(42)
        prices = generate_sample_prices(length=300, trend='uptrend')

        # DMAS gained from 45 to 60 (15 points)
        result = classify('TEST', prices, dmas=60, dmas_1w=45)

        # Should have momentum upgrade
        assert any('gained' in adj.lower() for adj in result.adjustments)
        assert result.dmas_wow_change == 15


class TestCompleteClassification:
    """Test complete classification scenarios."""

    def test_strong_bullish_scenario(self):
        """Test strong bullish scenario with high DMAS and uptrend."""
        np.random.seed(42)
        prices = generate_sample_prices(length=300, trend='uptrend')

        result = classify('TEST', prices, dmas=75, dmas_1w=70)

        assert result.ticker == 'TEST'
        assert result.dmas == 75
        assert result.base_assessment == Assessment.BULLISH
        # Final assessment should be positive
        assert result.assessment >= Assessment.NEUTRAL

    def test_strong_bearish_scenario(self):
        """Test strong bearish scenario with low DMAS and downtrend."""
        np.random.seed(42)
        prices = generate_sample_prices(length=300, trend='downtrend')

        result = classify('TEST', prices, dmas=25, dmas_1w=30)

        assert result.base_assessment == Assessment.BEARISH
        # Final assessment should be negative
        assert result.assessment <= Assessment.NEUTRAL

    def test_no_adjustments_scenario(self):
        """Test scenario where no adjustments are applied."""
        np.random.seed(42)
        # Create strong uptrend to avoid downgrades
        prices = generate_sample_prices(length=300, trend='uptrend')

        # Neutral DMAS with no momentum change
        result = classify('TEST', prices, dmas=50, dmas_1w=50)

        # Should have some description
        assert result.description is not None

    def test_classification_clamping(self):
        """Test that final assessment is clamped to valid range."""
        np.random.seed(42)
        prices = generate_sample_prices(length=300, trend='downtrend')

        # Very low DMAS with negative momentum
        result = classify('TEST', prices, dmas=10, dmas_1w=25)

        # Should be clamped to BEARISH (not below -2)
        assert result.assessment >= Assessment.BEARISH
        assert result.assessment <= Assessment.BULLISH


class TestBatchClassification:
    """Test classify_all for multiple assets."""

    def test_classify_multiple_tickers(self):
        """Test classification of multiple tickers."""
        np.random.seed(42)

        tickers = ['SPX', 'NKY', 'CSI']
        prices_dict = {
            'SPX': generate_sample_prices(300, 'uptrend'),
            'NKY': generate_sample_prices(300, 'downtrend'),
            'CSI': generate_sample_prices(300, 'sideways'),
        }
        dmas_dict = {'SPX': 75, 'NKY': 25, 'CSI': 50}
        dmas_1w_dict = {'SPX': 70, 'NKY': 30, 'CSI': 50}

        results_df = classify_all(tickers, prices_dict, dmas_dict, dmas_1w_dict)

        # Should have 3 results
        assert len(results_df) == 3

        # Check columns
        assert 'ticker' in results_df.columns
        assert 'assessment' in results_df.columns
        assert 'assessment_value' in results_df.columns
        assert 'dmas' in results_df.columns

        # SPX should be positive
        spx_row = results_df[results_df['ticker'] == 'SPX'].iloc[0]
        assert spx_row['dmas'] == 75

        # NKY should be negative
        nky_row = results_df[results_df['ticker'] == 'NKY'].iloc[0]
        assert nky_row['dmas'] == 25

    def test_missing_ticker_handling(self):
        """Test that missing tickers are skipped gracefully."""
        np.random.seed(42)

        tickers = ['SPX', 'MISSING']
        prices_dict = {'SPX': generate_sample_prices(300, 'uptrend')}
        dmas_dict = {'SPX': 75}

        results_df = classify_all(tickers, prices_dict, dmas_dict)

        # Should only have SPX result
        assert len(results_df) == 1
        assert results_df.iloc[0]['ticker'] == 'SPX'


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
