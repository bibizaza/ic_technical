"""
Main subtitle generator class.

Coordinates pattern selection, anti-repetition logic, and
subtitle generation for Market Compass assets.
"""

import random
from typing import Optional, Dict

from .patterns import PATTERNS, get_rating
from .decision_tree import route_subtitle


class SubtitleGenerator:
    """
    Generates Market Compass subtitles with anti-repetition logic.

    Tracks last pattern used per asset to avoid consecutive repetition
    and provides variety in subtitle generation.

    Attributes
    ----------
    last_pattern_used : dict
        Maps asset_name to last pattern string used
    last_category_used : dict
        Maps asset_name to last pattern category used
    """

    def __init__(self):
        """Initialize subtitle generator."""
        self.last_pattern_used: Dict[str, str] = {}
        self.last_category_used: Dict[str, str] = {}

    def get_pattern(self, asset: str, category: str) -> str:
        """
        Select pattern from category with anti-repetition.

        Filters out the last used pattern for this asset if possible,
        ensuring variety across weeks.

        Parameters
        ----------
        asset : str
            Asset name for tracking
        category : str
            Pattern category key (e.g., "bullish_strong")

        Returns
        -------
        str
            Selected pattern template
        """
        if category not in PATTERNS:
            # Fallback to neutral default if category not found
            category = "neutral_default"

        patterns = PATTERNS[category]
        last_used = self.last_pattern_used.get(asset)

        # Filter out last used pattern if possible
        available = [p for p in patterns if p != last_used]
        if not available:
            # If all patterns were filtered (or only 1 pattern), use all
            available = patterns

        # Select random pattern from available
        selected = random.choice(available)

        # Track for next time
        self.last_pattern_used[asset] = selected
        self.last_category_used[asset] = category

        return selected

    def generate(
        self,
        asset_data: dict,
        last_week_data: Optional[dict] = None,
        max_length: int = 120
    ) -> dict:
        """
        Generate subtitle for an asset.

        Parameters
        ----------
        asset_data : dict
            Current week asset data with keys:
            - asset_name (str)
            - asset_class (str)
            - dmas (int): 0-100
            - technical_score (int): 0-100
            - momentum_score (int): 0-100
            - rating (str)
            - price_vs_50ma (str): "above" | "below" | "at"
            - price_vs_100ma (str): "above" | "below" | "at"
            - price_vs_200ma (str): "above" | "below" | "at"
            - dmas_prev_week (int): Previous week DMAS
            - rating_prev_week (str): Previous week rating
            - ma_cross_event (str | None)
            - channel_color (str): "green" | "red"
            - near_support (bool)
            - near_resistance (bool)
            - at_ath (bool)
            - price_target (float | None)

        last_week_data : dict, optional
            Previous week data for comparison (not currently used,
            included for future enhancement)

        max_length : int, default=120
            Maximum subtitle length in characters

        Returns
        -------
        dict
            Result with keys:
            - subtitle (str): Generated commentary
            - pattern_used (str): Pattern category for debugging
            - rating (str): Rating label
            - truncated (bool): Whether subtitle was truncated
        """
        asset = asset_data["asset_name"]

        # Create pattern selector closure that uses this instance
        def pattern_selector(category: str) -> str:
            return self.get_pattern(asset, category)

        # Route to appropriate subtitle
        subtitle, pattern_category = route_subtitle(asset_data, pattern_selector)

        # Calculate rating
        rating = get_rating(asset_data["dmas"])

        # Truncate if needed (preserve complete sentences)
        truncated = False
        if len(subtitle) > max_length:
            # Try to truncate at sentence boundary
            truncation_point = max_length
            for delimiter in ['. ', '! ', '? ']:
                idx = subtitle[:max_length].rfind(delimiter)
                if idx > 0:
                    truncation_point = idx + 1
                    break

            subtitle = subtitle[:truncation_point].rstrip()
            truncated = True

        return {
            "subtitle": subtitle,
            "pattern_used": pattern_category,
            "rating": rating,
            "truncated": truncated
        }

    def generate_batch(
        self,
        assets_data: list[dict],
        max_length: int = 120
    ) -> list[dict]:
        """
        Generate subtitles for multiple assets.

        Parameters
        ----------
        assets_data : list[dict]
            List of asset data dictionaries

        max_length : int, default=120
            Maximum subtitle length in characters

        Returns
        -------
        list[dict]
            List of results with subtitle, pattern_used, rating for each asset
        """
        results = []

        for asset_data in assets_data:
            try:
                result = self.generate(asset_data, max_length=max_length)
                result["asset_name"] = asset_data["asset_name"]
                results.append(result)
            except Exception as e:
                # Handle errors gracefully
                results.append({
                    "asset_name": asset_data.get("asset_name", "Unknown"),
                    "subtitle": "Technical analysis unavailable",
                    "pattern_used": "error",
                    "rating": "Neutral",
                    "truncated": False,
                    "error": str(e)
                })

        return results

    def reset_history(self):
        """
        Reset pattern usage history.

        Useful for testing or when starting a new reporting period.
        """
        self.last_pattern_used.clear()
        self.last_category_used.clear()

    def get_stats(self) -> dict:
        """
        Get usage statistics.

        Returns
        -------
        dict
            Statistics with keys:
            - assets_tracked (int): Number of assets with history
            - category_distribution (dict): Count per category used
        """
        category_counts = {}
        for category in self.last_category_used.values():
            category_counts[category] = category_counts.get(category, 0) + 1

        return {
            "assets_tracked": len(self.last_pattern_used),
            "category_distribution": category_counts
        }


# Convenience function for single-shot usage
def generate_subtitle(
    asset_data: dict,
    last_week_data: Optional[dict] = None,
    last_pattern: Optional[str] = None,
    max_length: int = 120
) -> dict:
    """
    Generate subtitle for a single asset (convenience function).

    Creates a temporary SubtitleGenerator instance for one-time use.
    For repeated use across multiple weeks, use SubtitleGenerator class
    directly to maintain anti-repetition state.

    Parameters
    ----------
    asset_data : dict
        Asset data dictionary (see SubtitleGenerator.generate)

    last_week_data : dict, optional
        Previous week data (not currently used)

    last_pattern : str, optional
        Last pattern used for this asset (for anti-repetition)

    max_length : int, default=120
        Maximum subtitle length

    Returns
    -------
    dict
        Result dictionary with subtitle, pattern_used, rating, truncated
    """
    generator = SubtitleGenerator()

    # If last pattern provided, pre-populate to avoid repetition
    if last_pattern:
        asset = asset_data["asset_name"]
        generator.last_pattern_used[asset] = last_pattern

    return generator.generate(asset_data, last_week_data, max_length)
