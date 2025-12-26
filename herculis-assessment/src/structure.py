"""
Moving average structure analysis.

This module analyzes the price structure relative to key moving averages
to determine trend strength and provide scoring adjustments.
"""

from dataclasses import dataclass
import pandas as pd
from .config import MA_PERIODS


@dataclass
class StructureAnalysis:
    """
    Result of price structure analysis relative to moving averages.

    Attributes
    ----------
    score : int
        Structure score from -1 to +2
        +2 = Perfect uptrend (Price > 50d > 100d > 200d)
        +1 = Healthy (Price > most MAs)
         0 = Mixed structure
        -1 = Broken (Price below key MAs)

    price_vs_50d : float
        Percentage distance from 50-day MA (positive = above)

    price_vs_100d : float
        Percentage distance from 100-day MA

    price_vs_200d : float
        Percentage distance from 200-day MA

    above_50d : bool
        True if price is above 50-day MA

    above_100d : bool
        True if price is above 100-day MA

    above_200d : bool
        True if price is above 200-day MA

    perfect_structure : bool
        True if Price > 50d > 100d > 200d (all aligned)

    description : str
        Human-readable description of the structure
    """
    score: int
    price_vs_50d: float
    price_vs_100d: float
    price_vs_200d: float
    above_50d: bool
    above_100d: bool
    above_200d: bool
    perfect_structure: bool
    description: str


def compute_ma(prices: pd.Series, period: int) -> float:
    """
    Compute simple moving average for the most recent period.

    Parameters
    ----------
    prices : pd.Series
        Price series (must have at least 'period' data points)

    period : int
        Number of periods for MA calculation

    Returns
    -------
    float
        Most recent MA value, or None if insufficient data
    """
    if len(prices) < period:
        return None

    return prices.tail(period).mean()


def compute_percent_distance(price: float, ma_value: float) -> float:
    """
    Compute percentage distance of price from MA.

    Parameters
    ----------
    price : float
        Current price

    ma_value : float
        Moving average value

    Returns
    -------
    float
        Percentage distance (positive = price above MA)
    """
    if ma_value is None or ma_value == 0:
        return 0.0

    return ((price - ma_value) / ma_value) * 100


def analyze_structure(prices: pd.Series) -> StructureAnalysis:
    """
    Analyze price structure relative to key moving averages.

    Parameters
    ----------
    prices : pd.Series
        Price series (must have at least 200 data points for full analysis)

    Returns
    -------
    StructureAnalysis
        Complete structure analysis with score and details

    Raises
    ------
    ValueError
        If insufficient data for analysis
    """
    if len(prices) < MA_PERIODS['long']:
        raise ValueError(
            f"Insufficient data for structure analysis. "
            f"Need at least {MA_PERIODS['long']} periods, got {len(prices)}"
        )

    # Get current price
    current_price = prices.iloc[-1]

    # Compute MAs
    ma_50 = compute_ma(prices, MA_PERIODS['short'])
    ma_100 = compute_ma(prices, MA_PERIODS['medium'])
    ma_200 = compute_ma(prices, MA_PERIODS['long'])

    # Compute distances
    dist_50 = compute_percent_distance(current_price, ma_50)
    dist_100 = compute_percent_distance(current_price, ma_100)
    dist_200 = compute_percent_distance(current_price, ma_200)

    # Check position flags
    above_50 = current_price > ma_50 if ma_50 is not None else False
    above_100 = current_price > ma_100 if ma_100 is not None else False
    above_200 = current_price > ma_200 if ma_200 is not None else False

    # Check perfect structure (Price > 50d > 100d > 200d)
    perfect = False
    if all([ma_50, ma_100, ma_200]):
        perfect = (current_price > ma_50 > ma_100 > ma_200)

    # Compute structure score
    score = 0
    description = ""

    if perfect:
        score = 2
        description = "Perfect uptrend structure (Price > 50d > 100d > 200d)"
    elif above_50 and above_100 and above_200:
        score = 1
        description = "Healthy structure (Price above all key MAs)"
    elif above_50 and above_100:
        score = 0
        description = "Mixed structure (Price above 50d/100d, below 200d)"
    elif above_50:
        score = -1
        description = "Broken structure (Price below 100d MA)"
    else:
        score = -1
        description = "Weak structure (Price below 50d MA)"

    return StructureAnalysis(
        score=score,
        price_vs_50d=dist_50,
        price_vs_100d=dist_100,
        price_vs_200d=dist_200,
        above_50d=above_50,
        above_100d=above_100,
        above_200d=above_200,
        perfect_structure=perfect,
        description=description
    )
