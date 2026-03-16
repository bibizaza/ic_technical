"""
Test suite for Market Compass subtitle generator.

Tests all major scenarios and edge cases.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from subtitle_generator import SubtitleGenerator, generate_subtitle
from subtitle_generator.patterns import get_rating, get_ma_number, get_ordinal


def create_asset_data(
    asset_name="Test Asset",
    dmas=50,
    technical=50,
    momentum=50,
    **kwargs
):
    """Helper to create asset data dict with defaults."""
    data = {
        "asset_name": asset_name,
        "asset_class": "equity",
        "dmas": dmas,
        "technical_score": technical,
        "momentum_score": momentum,
        "rating": get_rating(dmas),
        "price_vs_50ma": "above",
        "price_vs_100ma": "above",
        "price_vs_200ma": "above",
        "dmas_prev_week": dmas,
        "rating_prev_week": get_rating(dmas),
        "ma_cross_event": None,
        "channel_color": "green",
        "near_support": False,
        "near_resistance": False,
        "at_ath": False,
        "price_target": None
    }
    data.update(kwargs)
    return data


class TestRatingFunction:
    """Test rating calculation."""

    def test_strongly_bearish(self):
        assert get_rating(0) == "Strongly Bearish"
        assert get_rating(20) == "Strongly Bearish"

    def test_bearish(self):
        assert get_rating(21) == "Bearish"
        assert get_rating(35) == "Bearish"

    def test_slightly_bearish(self):
        assert get_rating(36) == "Slightly Bearish"
        assert get_rating(45) == "Slightly Bearish"

    def test_neutral(self):
        assert get_rating(46) == "Neutral"
        assert get_rating(55) == "Neutral"

    def test_slightly_bullish(self):
        assert get_rating(56) == "Slightly Bullish"
        assert get_rating(65) == "Slightly Bullish"

    def test_bullish(self):
        assert get_rating(66) == "Bullish"
        assert get_rating(80) == "Bullish"

    def test_strongly_bullish(self):
        assert get_rating(81) == "Strongly Bullish"
        assert get_rating(100) == "Strongly Bullish"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_ma_number(self):
        assert get_ma_number("crossed_above_50") == 50
        assert get_ma_number("crossed_below_100") == 100
        assert get_ma_number("crossed_above_200") == 200

    def test_get_ordinal(self):
        assert get_ordinal(1) == "1st"
        assert get_ordinal(2) == "2nd"
        assert get_ordinal(3) == "3rd"
        assert get_ordinal(4) == "4th"
        assert get_ordinal(11) == "11th"
        assert get_ordinal(21) == "21st"
        assert get_ordinal(22) == "22nd"


class TestScenario1_StrongBullish:
    """Test Scenario 1: Strong bullish (DMAS 85, Tech 70, Mom 100)."""

    def test_strong_bullish(self):
        data = create_asset_data(
            asset_name="S&P 500",
            dmas=85,
            technical=70,
            momentum=100
        )

        result = generate_subtitle(data)

        assert result["rating"] == "Strongly Bullish"
        assert len(result["subtitle"]) > 0
        assert result["pattern_used"] in [
            "extreme_momentum_high",
            "bullish_strong",
            "bullish_ath"
        ]
        print(f"✓ Scenario 1: {result['subtitle']}")


class TestScenario2_BullishCaution:
    """Test Scenario 2: Bullish with caution (DMAS 75, Tech 50, Mom 100)."""

    def test_bullish_caution(self):
        data = create_asset_data(
            asset_name="Nasdaq",
            dmas=75,
            technical=50,
            momentum=100
        )

        result = generate_subtitle(data)

        assert result["rating"] == "Bullish"
        # Should route to bullish_caution due to low tech, high mom
        assert result["pattern_used"] in [
            "extreme_momentum_high",
            "bullish_caution"
        ]
        print(f"✓ Scenario 2: {result['subtitle']}")


class TestScenario3_NeutralHighTech:
    """Test Scenario 3: Neutral high-tech (DMAS 55, Tech 70, Mom 0)."""

    def test_neutral_high_tech(self):
        data = create_asset_data(
            asset_name="Gold",
            dmas=55,
            technical=70,
            momentum=0
        )

        result = generate_subtitle(data)

        assert result["rating"] == "Neutral"
        # Should route to neutral_tech_offset
        assert result["pattern_used"] in [
            "extreme_momentum_low",
            "neutral_tech_offset"
        ]
        print(f"✓ Scenario 3: {result['subtitle']}")


class TestScenario4_NeutralHighMom:
    """Test Scenario 4: Neutral high-mom (DMAS 55, Tech 40, Mom 100)."""

    def test_neutral_high_mom(self):
        data = create_asset_data(
            asset_name="Bitcoin",
            dmas=55,
            technical=40,
            momentum=100
        )

        result = generate_subtitle(data)

        assert result["rating"] == "Neutral"
        # Should route to neutral_mom_offset
        assert result["pattern_used"] in [
            "extreme_momentum_high",
            "neutral_mom_offset"
        ]
        print(f"✓ Scenario 4: {result['subtitle']}")


class TestScenario5_BearishDeep:
    """Test Scenario 5: Bearish deep (DMAS 20, Tech 30, Mom 0)."""

    def test_bearish_deep(self):
        data = create_asset_data(
            asset_name="CSI 300",
            dmas=20,
            technical=30,
            momentum=0
        )

        result = generate_subtitle(data)

        assert result["rating"] == "Strongly Bearish"
        assert result["pattern_used"] in [
            "extreme_momentum_low",
            "bearish_deep"
        ]
        print(f"✓ Scenario 5: {result['subtitle']}")


class TestScenario6_MACrossUp:
    """Test Scenario 6: MA crossover up (crossed_above_50, DMAS 65)."""

    def test_ma_cross_up(self):
        data = create_asset_data(
            asset_name="EUR/USD",
            dmas=65,
            technical=60,
            momentum=70,
            ma_cross_event="crossed_above_50"
        )

        result = generate_subtitle(data)

        assert result["pattern_used"] == "ma_cross_up"
        assert "50" in result["subtitle"]
        print(f"✓ Scenario 6: {result['subtitle']}")


class TestScenario7_MACrossDown:
    """Test Scenario 7: MA crossover down (crossed_below_200, DMAS 35)."""

    def test_ma_cross_down(self):
        data = create_asset_data(
            asset_name="FTSE 100",
            dmas=35,
            technical=40,
            momentum=30,
            ma_cross_event="crossed_below_200",
            price_vs_200ma="below"
        )

        result = generate_subtitle(data)

        assert result["pattern_used"] == "ma_cross_down"
        assert "200" in result["subtitle"]
        print(f"✓ Scenario 7: {result['subtitle']}")


class TestScenario8_DramaticImprovement:
    """Test Scenario 8: Dramatic improvement (DMAS +20 WoW)."""

    def test_dramatic_surge(self):
        data = create_asset_data(
            asset_name="Tesla",
            dmas=70,
            technical=65,
            momentum=75,
            dmas_prev_week=50
        )

        result = generate_subtitle(data)

        # DMAS change is 70 - 50 = +20, should trigger dramatic_surge
        assert result["pattern_used"] == "dramatic_surge"
        print(f"✓ Scenario 8: {result['subtitle']}")


class TestScenario9_DramaticDeterioration:
    """Test Scenario 9: Dramatic deterioration (DMAS -20 WoW)."""

    def test_dramatic_collapse(self):
        data = create_asset_data(
            asset_name="Crude Oil",
            dmas=30,
            technical=35,
            momentum=25,
            dmas_prev_week=50
        )

        result = generate_subtitle(data)

        # DMAS change is 30 - 50 = -20, should trigger dramatic_collapse
        assert result["pattern_used"] == "dramatic_collapse"
        print(f"✓ Scenario 9: {result['subtitle']}")


class TestScenario10_AllTimeHigh:
    """Test Scenario 10: At all-time high (at_ath=True, DMAS 80+)."""

    def test_at_ath(self):
        data = create_asset_data(
            asset_name="Apple",
            dmas=85,
            technical=80,
            momentum=90,
            at_ath=True
        )

        result = generate_subtitle(data)

        assert result["rating"] == "Strongly Bullish"
        assert result["pattern_used"] in ["bullish_ath", "extreme_momentum_high"]
        print(f"✓ Scenario 10: {result['subtitle']}")


class TestScenario11_GoldenCross:
    """Test Scenario 11: Golden cross event."""

    def test_golden_cross_strong(self):
        data = create_asset_data(
            asset_name="DAX",
            dmas=70,
            technical=65,
            momentum=75,
            ma_cross_event="golden_cross"
        )

        result = generate_subtitle(data)

        assert result["pattern_used"] == "golden_cross"
        print(f"✓ Scenario 11a (strong): {result['subtitle']}")

    def test_golden_cross_weak(self):
        data = create_asset_data(
            asset_name="Nikkei",
            dmas=45,
            technical=50,
            momentum=40,
            ma_cross_event="golden_cross"
        )

        result = generate_subtitle(data)

        assert result["pattern_used"] == "golden_cross_weak"
        print(f"✓ Scenario 11b (weak): {result['subtitle']}")


class TestScenario12_DeathCross:
    """Test Scenario 12: Death cross event."""

    def test_death_cross(self):
        data = create_asset_data(
            asset_name="Hang Seng",
            dmas=35,
            technical=40,
            momentum=30,
            ma_cross_event="death_cross"
        )

        result = generate_subtitle(data)

        assert result["pattern_used"] == "death_cross"
        print(f"✓ Scenario 12: {result['subtitle']}")


class TestScenario13_BearishStuck:
    """Test Scenario 13: Bearish stuck below MA."""

    def test_stuck_below_50ma(self):
        data = create_asset_data(
            asset_name="Ethereum",
            dmas=40,
            technical=45,
            momentum=35,
            price_vs_50ma="below"
        )

        result = generate_subtitle(data)

        assert result["pattern_used"] == "bearish_stuck"
        assert "50" in result["subtitle"]
        print(f"✓ Scenario 13: {result['subtitle']}")


class TestAntiRepetition:
    """Test anti-repetition logic."""

    def test_no_consecutive_repetition(self):
        generator = SubtitleGenerator()

        data = create_asset_data(
            asset_name="S&P 500",
            dmas=85,
            technical=70,
            momentum=90
        )

        # Generate 5 subtitles for same asset
        results = []
        for _ in range(5):
            result = generator.generate(data)
            results.append(result["subtitle"])

        # Check that we got some variety (not all identical)
        unique_subtitles = set(results)
        assert len(unique_subtitles) > 1, "Should have variety in subtitles"
        print(f"✓ Anti-repetition: Generated {len(unique_subtitles)} unique subtitles")


class TestBatchGeneration:
    """Test batch generation."""

    def test_batch_generation(self):
        generator = SubtitleGenerator()

        assets = [
            create_asset_data("S&P 500", dmas=85, technical=70, momentum=90),
            create_asset_data("Gold", dmas=55, technical=60, momentum=50),
            create_asset_data("Bitcoin", dmas=30, technical=35, momentum=25),
        ]

        results = generator.generate_batch(assets)

        assert len(results) == 3
        assert all("subtitle" in r for r in results)
        assert all("rating" in r for r in results)
        assert results[0]["asset_name"] == "S&P 500"
        assert results[1]["asset_name"] == "Gold"
        assert results[2]["asset_name"] == "Bitcoin"
        print(f"✓ Batch generation: {len(results)} subtitles generated")


class TestEdgeCases:
    """Test edge cases."""

    def test_max_length_truncation(self):
        data = create_asset_data(
            asset_name="Very Long Asset Name That Would Generate Long Subtitle",
            dmas=85,
            technical=70,
            momentum=90
        )

        result = generate_subtitle(data, max_length=60)

        assert len(result["subtitle"]) <= 60
        if result["truncated"]:
            print(f"✓ Truncation: {result['subtitle']}")
        else:
            print(f"✓ No truncation needed: {result['subtitle']}")

    def test_missing_optional_fields(self):
        # Minimal data
        data = {
            "asset_name": "Test",
            "dmas": 50,
            "technical_score": 50,
            "momentum_score": 50
        }

        result = generate_subtitle(data)

        assert "subtitle" in result
        assert "rating" in result
        print(f"✓ Missing fields handled: {result['subtitle']}")

    def test_extreme_dmas_values(self):
        # DMAS = 0
        data_zero = create_asset_data(dmas=0, technical=0, momentum=0)
        result_zero = generate_subtitle(data_zero)
        assert result_zero["rating"] == "Strongly Bearish"

        # DMAS = 100
        data_hundred = create_asset_data(dmas=100, technical=100, momentum=100)
        result_hundred = generate_subtitle(data_hundred)
        assert result_hundred["rating"] == "Strongly Bullish"

        print(f"✓ DMAS 0: {result_zero['subtitle']}")
        print(f"✓ DMAS 100: {result_hundred['subtitle']}")


class TestGeneratorStats:
    """Test generator statistics."""

    def test_stats(self):
        generator = SubtitleGenerator()

        assets = [
            create_asset_data("Asset1", dmas=85, technical=70, momentum=90),
            create_asset_data("Asset2", dmas=55, technical=60, momentum=50),
            create_asset_data("Asset3", dmas=30, technical=35, momentum=25),
        ]

        generator.generate_batch(assets)

        stats = generator.get_stats()

        assert stats["assets_tracked"] == 3
        assert "category_distribution" in stats
        print(f"✓ Stats: {stats}")

    def test_reset_history(self):
        generator = SubtitleGenerator()

        data = create_asset_data("Test", dmas=85, technical=70, momentum=90)
        generator.generate(data)

        assert len(generator.last_pattern_used) == 1

        generator.reset_history()

        assert len(generator.last_pattern_used) == 0
        print("✓ History reset successful")


def run_all_tests():
    """Run all test scenarios."""
    print("=" * 60)
    print("MARKET COMPASS SUBTITLE GENERATOR - TEST SUITE")
    print("=" * 60)

    test_classes = [
        TestRatingFunction,
        TestUtilityFunctions,
        TestScenario1_StrongBullish,
        TestScenario2_BullishCaution,
        TestScenario3_NeutralHighTech,
        TestScenario4_NeutralHighMom,
        TestScenario5_BearishDeep,
        TestScenario6_MACrossUp,
        TestScenario7_MACrossDown,
        TestScenario8_DramaticImprovement,
        TestScenario9_DramaticDeterioration,
        TestScenario10_AllTimeHigh,
        TestScenario11_GoldenCross,
        TestScenario12_DeathCross,
        TestScenario13_BearishStuck,
        TestAntiRepetition,
        TestBatchGeneration,
        TestEdgeCases,
        TestGeneratorStats
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 60)

        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith("test_")]

        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                passed_tests += 1
            except AssertionError as e:
                print(f"✗ {method_name} FAILED: {e}")
            except Exception as e:
                print(f"✗ {method_name} ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print("=" * 60)

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
