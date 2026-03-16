"""
Main subtitle generator class.

Coordinates pattern selection, anti-repetition logic, and
subtitle generation for Market Compass assets.

Implements the directive V3 specifications:
- Processing order: momentum → technical → DMAS → rating → subtitle
- One sentence only, maximum 15 words
- Pattern library usage (no generic boilerplate)
- Correct rating vocabulary: Positive, Constructive, Neutral, Cautious, Negative
- Anti-repetition: Track last 3 subtitles per asset
- BATCH DEDUPLICATION: No two assets in same report can have identical subtitles
- RATING-APPROPRIATE LANGUAGE: No "bullish" for Constructive, etc.
"""

import random
from typing import Optional, Dict, List, Set

from .patterns import PATTERNS, get_rating, get_ma_dynamics, add_context_if_needed


class SubtitleTracker:
    """
    Tracks subtitle history per asset to avoid repetition.

    Maintains last 3 subtitles for each asset to ensure variety
    across consecutive weeks.
    """

    def __init__(self):
        """Initialize tracker."""
        self.history: Dict[str, List[str]] = {}

    def is_repetitive(self, asset: str, new_subtitle: str) -> bool:
        """
        Check if subtitle was used recently.

        Parameters
        ----------
        asset : str
            Asset name
        new_subtitle : str
            Proposed subtitle

        Returns
        -------
        bool
            True if subtitle is in last 3 used
        """
        history = self.history.get(asset, [])
        return new_subtitle in history

    def record(self, asset: str, subtitle: str):
        """
        Record subtitle usage.

        Parameters
        ----------
        asset : str
            Asset name
        subtitle : str
            Subtitle used
        """
        if asset not in self.history:
            self.history[asset] = []
        self.history[asset].append(subtitle)
        # Keep only last 3
        self.history[asset] = self.history[asset][-3:]

    def clear(self, asset: str = None):
        """
        Clear history.

        Parameters
        ----------
        asset : str, optional
            If provided, clear only this asset. Otherwise clear all.
        """
        if asset:
            self.history.pop(asset, None)
        else:
            self.history.clear()


def validate_subtitle_language(subtitle: str, rating: str) -> bool:
    """
    Validate that subtitle uses rating-appropriate language.

    Word restrictions by rating:
    - Positive: Can use "bullish", "strong"
    - Constructive: NO "bullish" - use "constructive", "favorable", "positive"
    - Neutral: NO "bullish" or "bearish"
    - Cautious: NO "bullish"
    - Negative: NO "bullish" or "constructive"

    Parameters
    ----------
    subtitle : str
        Generated subtitle text
    rating : str
        Current rating (Positive, Constructive, Neutral, Cautious, Negative)

    Returns
    -------
    bool
        True if language is appropriate for the rating
    """
    subtitle_lower = subtitle.lower()

    if rating == "Constructive":
        # Should not use "bullish"
        if "bullish" in subtitle_lower:
            return False

    if rating in ["Neutral", "Cautious", "Negative"]:
        # Should not use "bullish"
        if "bullish" in subtitle_lower:
            return False

    if rating in ["Positive", "Constructive"]:
        # Should not use "bearish" or "negative"
        if "bearish" in subtitle_lower:
            return False
        # Allow "negative" only if it's about territory (e.g., "negative territory")
        if "negative" in subtitle_lower and "territory" not in subtitle_lower:
            return False

    if rating == "Negative":
        # Should not use "constructive"
        if "constructive" in subtitle_lower:
            return False

    return True


class SubtitleGenerator:
    """
    Generates Market Compass subtitles with anti-repetition logic.

    Tracks last 3 patterns used per asset to avoid repetition
    and provides variety in subtitle generation.

    NEW IN V3: Batch-level deduplication to prevent same subtitle
    appearing for multiple assets in the same report.

    Attributes
    ----------
    tracker : SubtitleTracker
        Tracks subtitle history
    last_pattern_used : dict
        Maps asset_name to last pattern string used
    last_category_used : dict
        Maps asset_name to last pattern category used
    current_batch_subtitles : set
        Tracks all subtitles used in current batch (normalized)
    current_batch_patterns : dict
        Tracks pattern category usage counts in current batch
    """

    def __init__(self):
        """Initialize subtitle generator."""
        self.tracker = SubtitleTracker()
        self.last_pattern_used: Dict[str, str] = {}
        self.last_category_used: Dict[str, str] = {}
        # NEW: Batch-level tracking
        self.current_batch_subtitles: Set[str] = set()
        self.current_batch_patterns: Dict[str, int] = {}

    def start_batch(self):
        """
        Call before generating subtitles for a new report.

        Clears batch-level tracking to start fresh for each report.
        Per-asset history (tracker) is preserved across batches.
        """
        self.current_batch_subtitles.clear()
        self.current_batch_patterns.clear()

    def _normalize_subtitle(self, subtitle: str) -> str:
        """Normalize subtitle for comparison."""
        return subtitle.lower().strip()

    def _is_used_in_batch(self, subtitle: str) -> bool:
        """
        Check if subtitle already used in current batch.

        Parameters
        ----------
        subtitle : str
            Subtitle to check

        Returns
        -------
        bool
            True if subtitle (normalized) already used in this batch
        """
        normalized = self._normalize_subtitle(subtitle)
        return normalized in self.current_batch_subtitles

    def _record_batch_usage(self, subtitle: str, pattern_category: str):
        """
        Record subtitle usage in current batch.

        Parameters
        ----------
        subtitle : str
            Subtitle used
        pattern_category : str
            Pattern category used
        """
        normalized = self._normalize_subtitle(subtitle)
        self.current_batch_subtitles.add(normalized)
        self.current_batch_patterns[pattern_category] = \
            self.current_batch_patterns.get(pattern_category, 0) + 1

    def get_pattern(self, asset: str, category: str) -> str:
        """
        Select pattern from category with BOTH per-asset AND batch-level anti-repetition.

        Filters out:
        1. Last pattern used for this asset
        2. Patterns in this asset's history
        3. Patterns already heavily used in current batch

        Parameters
        ----------
        asset : str
            Asset name for tracking
        category : str
            Pattern category key (e.g., "positive_strong")

        Returns
        -------
        str
            Selected pattern template
        """
        if category not in PATTERNS:
            # Fallback to neutral default if category not found
            category = "neutral_turning"

        patterns = PATTERNS[category]
        last_used = self.last_pattern_used.get(asset)
        history = self.tracker.history.get(asset, [])

        # Collect patterns used by OTHER assets in this batch
        batch_used_patterns = set()
        for other_asset, pattern in self.last_pattern_used.items():
            if other_asset != asset:
                batch_used_patterns.add(pattern)

        # Filter patterns with multiple criteria
        available = []
        for p in patterns:
            # Skip last used for this asset
            if p == last_used:
                continue
            # Skip patterns in this asset's history
            if p in history:
                continue
            # Skip patterns used by other assets in this batch (if category used > 0)
            if self.current_batch_patterns.get(category, 0) > 0 and p in batch_used_patterns:
                continue
            available.append(p)

        if not available:
            # Fallback: just avoid last used for this asset
            available = [p for p in patterns if p != last_used]

        if not available:
            # If still none, use all patterns
            available = patterns

        # Select random pattern from available
        selected = random.choice(available)

        # Track for next time
        self.last_pattern_used[asset] = selected
        self.last_category_used[asset] = category

        return selected

    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())

    def _validate_scores(self, asset_data: dict) -> bool:
        """
        Validate all required scores are present and valid.

        Parameters
        ----------
        asset_data : dict
            Asset data dictionary

        Returns
        -------
        bool
            True if all scores are valid
        """
        try:
            dmas = asset_data.get("dmas")
            tech = asset_data.get("technical_score")
            mom = asset_data.get("momentum_score")

            if dmas is None or tech is None or mom is None:
                return False

            if not (0 <= dmas <= 100):
                return False
            if not (0 <= tech <= 100):
                return False
            if not (0 <= mom <= 100):
                return False

            return True
        except (TypeError, ValueError):
            return False

    def _generate_unique_fallback(self, asset: str, dmas: int) -> str:
        """
        Generate a unique asset-specific subtitle as last resort.

        Used when all pattern-based attempts result in duplicates.

        Parameters
        ----------
        asset : str
            Asset name
        dmas : int
            DMAS score

        Returns
        -------
        str
            Unique fallback subtitle
        """
        if dmas >= 70:
            return f"{asset} maintains strong technical positioning"
        elif dmas >= 55:
            return f"{asset} holds constructive outlook with mixed signals"
        elif dmas >= 45:
            return f"Technical picture for {asset} remains balanced"
        elif dmas >= 30:
            return f"{asset} shows cautious technical setup"
        else:
            return f"Weak technical backdrop persists for {asset}"

    def generate(
        self,
        asset_data: dict,
        last_week_data: Optional[dict] = None,
        max_words: int = 15
    ) -> dict:
        """
        Generate subtitle for an asset with batch deduplication.

        Processing order (per directive):
        1. Read momentum_score from input
        2. Read/calculate technical_score
        3. Calculate DMAS
        4. Validate all scores
        5. Determine rating
        6. Calculate MA dynamics
        7. Generate subtitle with batch deduplication

        Parameters
        ----------
        asset_data : dict
            Current week asset data with keys:
            - asset_name (str)
            - asset_class (str): "equity" | "commodity" | "crypto"
            - dmas (int): 0-100
            - technical_score (int): 0-100
            - momentum_score (int): 0-100
            - price_vs_50ma_pct (float): e.g., 2.5 means 2.5% above 50MA
            - price_vs_100ma_pct (float): e.g., -1.2 means 1.2% below 100MA
            - price_vs_200ma_pct (float): e.g., 5.0 means 5% above 200MA
            - dmas_prev_week (int): Previous week DMAS
            - rating_prev_week (str): Previous week rating
            - ma_cross_event (str | None): golden_cross, death_cross, crossed_above_*, crossed_below_*
            - near_support (bool): Price within 2% of support
            - near_resistance (bool): Price within 2% of resistance
            - at_ath (bool): At or near all-time high
            - price_target (float | None): Optional price target

        last_week_data : dict, optional
            Previous week data for comparison

        max_words : int, default=15
            Maximum words per subtitle (per directive)

        Returns
        -------
        dict
            Result with keys:
            - subtitle (str): Generated commentary (max 15 words, one sentence)
            - rating (str): One of: Positive, Constructive, Neutral, Cautious, Negative
            - pattern_used (str): Pattern category for debugging
            - validated (bool): True if all scores were valid
            - word_count (int): Number of words in subtitle
        """
        # Import here to avoid circular import
        from .decision_tree import route_subtitle

        asset = asset_data["asset_name"]

        # Validate scores
        validated = self._validate_scores(asset_data)
        if not validated:
            return {
                "subtitle": "Technical analysis unavailable",
                "rating": "Neutral",
                "pattern_used": "error",
                "validated": False,
                "word_count": 3
            }

        # Calculate rating from DMAS
        rating = get_rating(
            asset_data["dmas"],
            asset_data.get("technical_score"),
            asset_data.get("momentum_score")
        )

        # Create pattern selector closure that uses this instance
        def pattern_selector(category: str) -> str:
            return self.get_pattern(asset, category)

        # Route to appropriate subtitle with retry logic for deduplication
        max_attempts = 5
        attempts = 0
        subtitle = None
        pattern_category = None

        while attempts < max_attempts:
            subtitle, pattern_category = route_subtitle(asset_data, pattern_selector)

            # Validate language is appropriate for rating
            if not validate_subtitle_language(subtitle, rating):
                attempts += 1
                continue

            # Check batch duplication
            if not self._is_used_in_batch(subtitle):
                break

            attempts += 1

        # If still duplicate after max attempts, use unique fallback
        if subtitle and self._is_used_in_batch(subtitle):
            subtitle = self._generate_unique_fallback(asset, asset_data["dmas"])
            pattern_category = "unique_fallback"

        # Add contextual suffix if conditions warrant (FIX 5)
        subtitle = add_context_if_needed(subtitle, asset_data, max_words)

        # Ensure one sentence only and max 15 words
        # Split into sentences and take first one
        for delimiter in ['. ', '! ', '? ']:
            if delimiter in subtitle:
                subtitle = subtitle.split(delimiter)[0] + delimiter[0]
                break

        # Check word count and truncate if needed
        word_count = self._count_words(subtitle)
        if word_count > max_words:
            words = subtitle.split()[:max_words]
            subtitle = ' '.join(words)
            # Clean up trailing punctuation if mid-sentence
            if not subtitle[-1] in '.!?':
                subtitle = subtitle.rstrip(',;:') + '.'
            word_count = max_words

        # Record for anti-repetition (both per-asset and batch)
        self.tracker.record(asset, subtitle)
        self._record_batch_usage(subtitle, pattern_category)

        return {
            "subtitle": subtitle,
            "rating": rating,
            "pattern_used": pattern_category,
            "validated": True,
            "word_count": word_count
        }

    def generate_batch(
        self,
        assets_data: list,
        max_words: int = 15
    ) -> list:
        """
        Generate subtitles for multiple assets with deduplication.

        IMPORTANT: Starts a new batch before generating, ensuring no
        two assets in the same report have identical subtitles.

        Parameters
        ----------
        assets_data : list[dict]
            List of asset data dictionaries

        max_words : int, default=15
            Maximum words per subtitle

        Returns
        -------
        list[dict]
            List of results with subtitle, pattern_used, rating for each asset
        """
        # START NEW BATCH - clear batch tracking
        self.start_batch()

        results = []

        for asset_data in assets_data:
            try:
                result = self.generate(asset_data, max_words=max_words)
                result["asset_name"] = asset_data["asset_name"]
                results.append(result)
            except Exception as e:
                # Handle errors gracefully
                results.append({
                    "asset_name": asset_data.get("asset_name", "Unknown"),
                    "subtitle": "Technical analysis unavailable",
                    "pattern_used": "error",
                    "rating": "Neutral",
                    "validated": False,
                    "word_count": 3,
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
        self.tracker.clear()
        self.current_batch_subtitles.clear()
        self.current_batch_patterns.clear()

    def get_stats(self) -> dict:
        """
        Get usage statistics.

        Returns
        -------
        dict
            Statistics with keys:
            - assets_tracked (int): Number of assets with history
            - category_distribution (dict): Count per category used
            - batch_subtitle_count (int): Unique subtitles in current batch
            - batch_pattern_distribution (dict): Pattern category counts in batch
        """
        category_counts = {}
        for category in self.last_category_used.values():
            category_counts[category] = category_counts.get(category, 0) + 1

        return {
            "assets_tracked": len(self.last_pattern_used),
            "category_distribution": category_counts,
            "batch_subtitle_count": len(self.current_batch_subtitles),
            "batch_pattern_distribution": dict(self.current_batch_patterns)
        }


# Convenience function for single-shot usage
def generate_subtitle(
    asset_data: dict,
    last_week_data: Optional[dict] = None,
    last_pattern: Optional[str] = None,
    max_words: int = 15
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

    max_words : int, default=15
        Maximum words per subtitle

    Returns
    -------
    dict
        Result dictionary with subtitle, pattern_used, rating, validated, word_count
    """
    generator = SubtitleGenerator()

    # If last pattern provided, pre-populate to avoid repetition
    if last_pattern:
        asset = asset_data["asset_name"]
        generator.last_pattern_used[asset] = last_pattern

    return generator.generate(asset_data, last_week_data, max_words)
