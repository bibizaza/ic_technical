"""Pre-registered trailing volatility sizing and combined-weight trading."""
import numpy as np
import pandas as pd


def sigma(returns):
    return returns.rolling(21, min_periods=21).std(ddof=1) * np.sqrt(252)


def target_weight(returns):
    return (.15 / sigma(returns)).clip(upper=1.)


def daily_weights(index, dates, target, sleeve=None):
    """Decision inputs indexed at t; resulting weights earn returns from t+2.

    Sleeve is R1's scheduled decision-date weight, before the t+2 shift.
    State changes override the thresholds. Initial allocation is uncharged,
    consistent with the existing gauge.
    """
    changes = pd.Series(np.nan, index=index)
    held, previous_sleeve, last_month = np.nan, None, None
    for date in dates:
        month = date.to_period('M')
        base = 1. if sleeve is None else sleeve.loc[date]
        wanted = base * target.loc[date]
        threshold = .10 if month != last_month else .25
        forced = previous_sleeve is not None and base != previous_sleeve
        delta = abs(wanted - held)
        if pd.isna(wanted):
            raise ValueError(f'Missing volatility target at {date}')
        # isclose only handles binary representation at the exact boundary.
        if pd.isna(held) or forced or delta >= threshold or np.isclose(delta, threshold, rtol=0, atol=1e-14):
            held = wanted
        effective = index.get_loc(date) + 2
        if effective < len(index):
            changes.iloc[effective] = held
        previous_sleeve, last_month = base, month
    return changes.ffill().rename('equity_weight')


def turnover_per_year(weights, index=None):
    index = weights.dropna().index if index is None else index
    return weights.diff().abs().reindex(index).sum() / (len(index) / 252)
