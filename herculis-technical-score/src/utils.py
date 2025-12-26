"""
Utility functions for technical indicator calculations.
"""

import pandas as pd
import numpy as np
from typing import Optional


def get_value_n_periods_ago(series: pd.Series, n_periods: int) -> Optional[float]:
    """
    Get the value from n periods ago.

    Parameters
    ----------
    series : pd.Series
        Time series data
    n_periods : int
        Number of periods to look back

    Returns
    -------
    float or None
        Value from n periods ago, or None if not available
    """
    if len(series) <= n_periods:
        return None
    return float(series.iloc[-(n_periods + 1)])


def get_last_value(series: pd.Series) -> Optional[float]:
    """
    Get the last value from a series.

    Parameters
    ----------
    series : pd.Series
        Time series data

    Returns
    -------
    float or None
        Last value, or None if series is empty
    """
    if len(series) == 0:
        return None
    return float(series.iloc[-1])


def ensure_numeric(value: any) -> float:
    """
    Ensure a value is numeric, handling None and NaN.

    Parameters
    ----------
    value : any
        Value to convert

    Returns
    -------
    float
        Numeric value, or 0.0 if invalid
    """
    if value is None or pd.isna(value):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def validate_series(series: pd.Series, min_length: int, name: str = "Series") -> None:
    """
    Validate that a series has sufficient data.

    Parameters
    ----------
    series : pd.Series
        Series to validate
    min_length : int
        Minimum required length
    name : str
        Name of the series (for error messages)

    Raises
    ------
    ValueError
        If series is too short
    """
    if len(series) < min_length:
        raise ValueError(
            f"{name} must have at least {min_length} data points, "
            f"got {len(series)}"
        )


def clip_score(score: float) -> float:
    """
    Clip a score to valid range [0.0, 1.0].

    Parameters
    ----------
    score : float
        Raw score

    Returns
    -------
    float
        Clipped score between 0.0 and 1.0
    """
    return max(0.0, min(1.0, score))
