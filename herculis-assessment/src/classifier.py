"""
Core classification logic for assessment ratings.

This module implements the complete classification system that combines
DMAS scores, price structure, and momentum to produce final assessments.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd

from .config import Assessment, DMAS_THRESHOLDS, ADJUSTMENT_CONFIG
from .structure import analyze_structure, StructureAnalysis


@dataclass
class ClassificationResult:
    """
    Complete classification result for an asset.

    Attributes
    ----------
    ticker : str
        Asset identifier

    assessment : Assessment
        Final assessment level (BEARISH to BULLISH)

    base_assessment : Assessment
        Initial assessment from DMAS score (before adjustments)

    dmas : float
        Current DMAS score (0-100)

    dmas_wow_change : float
        Week-over-week DMAS change (points)

    structure : StructureAnalysis
        Price structure analysis

    adjustments : list[str]
        List of adjustment reasons applied

    description : str
        Human-readable summary
    """
    ticker: str
    assessment: Assessment
    base_assessment: Assessment
    dmas: float
    dmas_wow_change: float
    structure: StructureAnalysis
    adjustments: list[str]
    description: str


def _get_base_assessment(dmas: float) -> Assessment:
    """
    Get base assessment from DMAS score using thresholds.

    Parameters
    ----------
    dmas : float
        DMAS score (0-100)

    Returns
    -------
    Assessment
        Base assessment level
    """
    if dmas >= DMAS_THRESHOLDS['bullish']:
        return Assessment.BULLISH
    elif dmas >= DMAS_THRESHOLDS['constructive']:
        return Assessment.CONSTRUCTIVE
    elif dmas >= DMAS_THRESHOLDS['neutral']:
        return Assessment.NEUTRAL
    elif dmas >= DMAS_THRESHOLDS['cautious']:
        return Assessment.CAUTIOUS
    else:
        return Assessment.BEARISH


def _apply_downgrades(
    assessment: int,
    structure: StructureAnalysis,
    dmas_wow_change: float,
    adjustments: list[str]
) -> int:
    """
    Apply downgrade rules based on price structure and momentum.

    Downgrades are applied in order:
    1. Price < 200d MA: Hard cap at Cautious
    2. Price < 100d MA: Downgrade 1 level
    3. Price < 50d MA: Downgrade 1 level (stacks with #2)
    4. DMAS dropped ≥10 pts WoW: Downgrade 1 level

    Parameters
    ----------
    assessment : int
        Current assessment level (as integer)
    structure : StructureAnalysis
        Price structure analysis
    dmas_wow_change : float
        Week-over-week DMAS change
    adjustments : list[str]
        List to append adjustment reasons

    Returns
    -------
    int
        Adjusted assessment level
    """
    # Rule 1: Price < 200d MA → Hard cap at Cautious
    if not structure.above_200d:
        if assessment > Assessment.CAUTIOUS:
            adjustments.append("Price < 200d MA → Capped at Cautious")
            assessment = Assessment.CAUTIOUS

    # Rule 2: Price < 100d MA → Downgrade 1 level
    if not structure.above_100d:
        adjustments.append("Price < 100d MA → Downgrade 1 level")
        assessment -= 1

    # Rule 3: Price < 50d MA → Downgrade 1 level (stacks)
    if not structure.above_50d:
        adjustments.append("Price < 50d MA → Downgrade 1 level")
        assessment -= 1

    # Rule 4: DMAS dropped ≥10 pts WoW → Downgrade 1 level
    if dmas_wow_change <= -ADJUSTMENT_CONFIG['wow_change_threshold']:
        adjustments.append(
            f"DMAS dropped {abs(dmas_wow_change):.1f} pts WoW → Downgrade 1 level"
        )
        assessment -= 1

    return assessment


def _apply_upgrades(
    assessment: int,
    dmas: float,
    structure: StructureAnalysis,
    dmas_wow_change: float,
    adjustments: list[str]
) -> int:
    """
    Apply upgrade rules based on perfect structure and momentum.

    Upgrades are applied after downgrades:
    1. Perfect structure AND DMAS ≥65 → Upgrade to Bullish
    2. DMAS gained ≥10 pts WoW → Upgrade 1 level

    Parameters
    ----------
    assessment : int
        Current assessment level (as integer, after downgrades)
    dmas : float
        Current DMAS score
    structure : StructureAnalysis
        Price structure analysis
    dmas_wow_change : float
        Week-over-week DMAS change
    adjustments : list[str]
        List to append adjustment reasons

    Returns
    -------
    int
        Adjusted assessment level
    """
    # Rule 1: Perfect structure AND DMAS ≥65 → Upgrade to Bullish
    if structure.perfect_structure and dmas >= ADJUSTMENT_CONFIG['structure_upgrade_min_dmas']:
        if assessment < Assessment.BULLISH:
            adjustments.append(
                f"Perfect structure + DMAS {dmas:.1f} → Upgrade to Bullish"
            )
            assessment = Assessment.BULLISH

    # Rule 2: DMAS gained ≥10 pts WoW → Upgrade 1 level
    if dmas_wow_change >= ADJUSTMENT_CONFIG['wow_change_threshold']:
        adjustments.append(
            f"DMAS gained {dmas_wow_change:.1f} pts WoW → Upgrade 1 level"
        )
        assessment += 1

    return assessment


def classify(
    ticker: str,
    prices: pd.Series,
    dmas: float,
    dmas_1w: Optional[float] = None
) -> ClassificationResult:
    """
    Classify an asset into 5-level assessment.

    Parameters
    ----------
    ticker : str
        Asset identifier

    prices : pd.Series
        Price series (needs at least 200 data points)

    dmas : float
        Current DMAS score (0-100)

    dmas_1w : float, optional
        DMAS score from 1 week ago (for momentum calculation)

    Returns
    -------
    ClassificationResult
        Complete classification with reasoning

    Raises
    ------
    ValueError
        If insufficient data for analysis
    """
    # Get base assessment from DMAS
    base_assessment = _get_base_assessment(dmas)

    # Analyze structure
    structure = analyze_structure(prices)

    # Calculate momentum
    dmas_wow_change = (dmas - dmas_1w) if dmas_1w is not None else 0.0

    # Start with base assessment
    assessment = int(base_assessment)
    adjustments = []

    # Apply downgrades first
    assessment = _apply_downgrades(assessment, structure, dmas_wow_change, adjustments)

    # Then apply upgrades
    assessment = _apply_upgrades(assessment, dmas, structure, dmas_wow_change, adjustments)

    # Clamp to valid range
    assessment = max(Assessment.BEARISH, min(Assessment.BULLISH, assessment))

    # Build description
    if not adjustments:
        description = f"Base assessment from DMAS {dmas:.1f}"
    else:
        description = f"Base: {base_assessment.name} (DMAS {dmas:.1f}) → {len(adjustments)} adjustment(s)"

    return ClassificationResult(
        ticker=ticker,
        assessment=Assessment(assessment),
        base_assessment=base_assessment,
        dmas=dmas,
        dmas_wow_change=dmas_wow_change,
        structure=structure,
        adjustments=adjustments,
        description=description
    )


def classify_all(
    tickers: list[str],
    prices_dict: dict[str, pd.Series],
    dmas_dict: dict[str, float],
    dmas_1w_dict: Optional[dict[str, float]] = None
) -> pd.DataFrame:
    """
    Classify multiple assets into assessment levels.

    Parameters
    ----------
    tickers : list[str]
        List of asset identifiers

    prices_dict : dict[str, pd.Series]
        Dictionary mapping ticker to price series

    dmas_dict : dict[str, float]
        Dictionary mapping ticker to current DMAS score

    dmas_1w_dict : dict[str, float], optional
        Dictionary mapping ticker to DMAS from 1 week ago

    Returns
    -------
    pd.DataFrame
        Classification results with columns:
        - ticker
        - assessment (string label)
        - assessment_value (integer -2 to +2)
        - base_assessment
        - dmas
        - dmas_wow_change
        - price_vs_50d
        - price_vs_100d
        - price_vs_200d
        - adjustments_count
        - description
    """
    results = []

    for ticker in tickers:
        if ticker not in prices_dict or ticker not in dmas_dict:
            continue

        prices = prices_dict[ticker]
        dmas = dmas_dict[ticker]
        dmas_1w = dmas_1w_dict.get(ticker) if dmas_1w_dict else None

        try:
            result = classify(ticker, prices, dmas, dmas_1w)

            results.append({
                'ticker': result.ticker,
                'assessment': result.assessment.name.title(),
                'assessment_value': int(result.assessment),
                'base_assessment': result.base_assessment.name.title(),
                'dmas': result.dmas,
                'dmas_wow_change': result.dmas_wow_change,
                'price_vs_50d': result.structure.price_vs_50d,
                'price_vs_100d': result.structure.price_vs_100d,
                'price_vs_200d': result.structure.price_vs_200d,
                'adjustments_count': len(result.adjustments),
                'description': result.description
            })
        except Exception as e:
            print(f"Error classifying {ticker}: {e}")
            continue

    return pd.DataFrame(results)
