"""
Composite breadth score computation.

Three-pillar model:
  TREND (40%)    : average(pct_gt_50d, pct_gt_100d)
  MOMENTUM (35%) : 0.50*macd_gt_0 + 0.25*signal_gt_0 + 0.25*net_signal_rescaled
  EXTENSION (25%): 100 - (pct_above_upper_boll + pct_below_lower_boll)

COMPOSITE = 0.40*TREND + 0.35*MOMENTUM + 0.25*EXTENSION
"""
from __future__ import annotations

import pandas as pd


def compute_composite_breadth(raw_breadth: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite breadth scores from raw Bloomberg breadth fields.

    Parameters
    ----------
    raw_breadth : pd.DataFrame
        Index = index name
        Columns = breadth field names (percentages 0-100)

    Returns
    -------
    pd.DataFrame
        Columns: name, composite, trend, momentum, extension, rank
    """
    df = raw_breadth.copy()

    # --- TREND ---
    trend = df[["PCT_MEMB_PX_GT_50D_MOV_AVG", "PCT_MEMB_PX_GT_100D_MOV_AVG"]].mean(axis=1)

    # --- MOMENTUM ---
    macd_gt_0 = df["PCT_MEMB_MACD_GT_BASE_LINE_0"].fillna(50)
    signal_gt_0 = df["PCT_MEMB_SIGNAL_GT_BASE_LINE_0"].fillna(50)

    # Net signal: buy - sell (PCT_MEM_MACD_BUY_SIGNAL_LST_10D - PCT_MEM_MACD_SL_SIGNAL_LST_10D)
    buy_pct = df["PCT_MEM_MACD_BUY_SIGNAL_LST_10D"].fillna(50)
    sell_pct = df["PCT_MEM_MACD_SL_SIGNAL_LST_10D"].fillna(50)
    net_signal_raw = buy_pct - sell_pct           # range roughly -100 to +100
    net_signal_rescaled = (net_signal_raw + 100) / 2  # rescale to 0-100 (50=neutral)

    momentum = (
        0.50 * macd_gt_0
        + 0.25 * signal_gt_0
        + 0.25 * net_signal_rescaled
    )

    # --- EXTENSION (inverted) ---
    above_upper = df["PCT_MEMB_PX_ABV_UPPER_BOLL_BAND"].fillna(0)
    below_lower = df["PCT_MEMB_PX_BLW_LWR_BOLL_BAND"].fillna(0)
    extension = (100 - (above_upper + below_lower)).clip(0, 100)

    # --- COMPOSITE ---
    composite = (
        0.40 * trend
        + 0.35 * momentum
        + 0.25 * extension
    ).clip(0, 100)

    result = pd.DataFrame(
        {
            "name": df.index,
            "composite": composite.round(1),
            "trend": trend.round(1),
            "momentum": momentum.round(1),
            "extension": extension.round(1),
        }
    ).reset_index(drop=True)

    result["rank"] = result["composite"].rank(ascending=False, method="min").astype(int)
    result = result.sort_values("rank").reset_index(drop=True)

    return result
