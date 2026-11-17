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


def r2_positions(dates: pd.DatetimeIndex, dmas: pd.Series, rev_z: pd.Series) -> pd.DataFrame:
    """Step 1b state after each decision; only current/past observations are read.

    Normal trend re-entry leaves armed unchanged, as specified in 6b. Every
    subsequent exit resets it; only earnings re-entry explicitly clears it.
    """
    rows = []
    last_month = None
    pos, mode, armed = 'Cash', 'trend', False
    for d in dates:
        month = d.to_period('M')
        monthly = month != last_month
        trend, rev = dmas.loc[d], rev_z.loc[d]
        negative = bool(pd.notna(rev) and rev < 0)
        recovered = bool(pd.notna(rev) and rev >= 0)
        if last_month is None:
            pos = 'Invested' if trend >= 45 else 'Cash'
            armed = pos == 'Cash' and negative
        elif pos == 'Invested':
            if mode == 'earnings' and trend >= 45:
                mode = 'trend'
            elif (mode == 'trend' or negative) and (
                (monthly and trend < 45) or (not monthly and trend < 30)
            ):
                pos, mode, armed = 'Cash', 'trend', negative
        else:
            if negative:
                armed = True
            if monthly and trend >= 45:
                pos, mode = 'Invested', 'trend'
            elif armed and recovered:
                pos, mode, armed = 'Invested', 'earnings', False
        rows.append((pos, mode, armed))
        last_month = month
    return pd.DataFrame(rows, index=dates, columns=['pos', 'mode', 'armed'])


def weights_from_positions(trading_days, positions: pd.Series, cash_weight) -> pd.Series:
    """Map invested booleans to weights earning from t+2, like sleeve_weights."""
    changes = pd.Series(np.nan, index=trading_days, dtype=float)
    for d, invested in positions.items():
        effective = trading_days.get_loc(d) + 2
        if effective < len(trading_days):
            changes.iloc[effective] = 1. if invested else cash_weight
    return changes.ffill().rename('equity_weight')
