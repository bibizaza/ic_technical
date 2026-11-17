"""Delphi monthly exposure setting and deep-trend early exits."""
import numpy as np
import pandas as pd


def decision_dates(trading_days: pd.DatetimeIndex, start: str) -> pd.DatetimeIndex:
    if trading_days.empty:
        return pd.DatetimeIndex([])
    tuesdays = pd.date_range(start, trading_days[-1], freq='W-TUE')
    positions = trading_days.searchsorted(tuesdays, side='right') - 1
    return trading_days[positions[positions >= 0]].drop_duplicates()


def sleeve_weights(trading_days, dates, cash_signal: pd.Series, deep: pd.Series, cash_weight: float) -> pd.Series:
    """Weights earn t+2 returns after a decision at t and execution at close t+1.

    No position is assumed before the first scheduled position becomes effective.
    """
    changes = pd.Series(np.nan, index=trading_days, dtype=float)
    current = np.nan
    last_month = None
    for d in dates:
        month = d.to_period('M')
        if month != last_month:
            current = cash_weight if cash_signal.loc[d] else 1.
        elif current == 1. and cash_signal.loc[d] and deep.loc[d]:
            current = cash_weight
        last_month = month
        effective = trading_days.get_loc(d) + 2
        if effective < len(trading_days):
            changes.iloc[effective] = current
    return changes.ffill().rename('equity_weight')


def net_returns(w: pd.Series, eq_ret: pd.Series, cash_ret: pd.Series, cost_bp=10.0) -> pd.Series:
    """Task-4 convention: debit turnover on the new-weight return day.

    Initial allocation has no prior weight, so is not counted as a switch.
    """
    cost = w.diff().abs().fillna(0) * cost_bp / 1e4
    return (w * eq_ret + (1 - w) * cash_ret - cost).rename('net_return')
