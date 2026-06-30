"""
Composite breadth score computation.

Three-pillar model:
  TREND (40%)      : average(pct_gt_50d, pct_gt_100d)
  CONVICTION (35%) : (pct_gt_20d - pct_above_200d + 100) / 2  — MA spread rescaled 0-100
  SENTIMENT (25%)  : 50 - (pct_rsi_lt_30 - pct_rsi_gt_70), clamped 0-100

COMPOSITE = 0.40*TREND + 0.35*CONVICTION + 0.25*SENTIMENT
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
        Columns: name, composite, trend, conviction, sentiment, rank
    """
    df = raw_breadth.copy()

    # --- TREND ---
    trend = df[["PCT_MEMB_PX_GT_50D_MOV_AVG", "PCT_MEMB_PX_GT_100D_MOV_AVG"]].mean(axis=1)

    # --- CONVICTION ---
    # Short-term vs long-term MA breadth spread: >0 means more members heating up short-term
    gt_20d = df["PCT_MEMB_PX_GT_20D_MOV_AVG"].fillna(50)
    above_200d = df["PCT_MEMB_ABOVE_MOV_AVG_200D"].fillna(50)
    conviction = (gt_20d - above_200d + 100) / 2  # rescaled to 0-100, 50=neutral

    # --- SENTIMENT ---
    # RSI extremes: more overbought than oversold members → positive sentiment
    rsi_gt_70 = df["PCT_MEMB_WITH_14D_RSI_GT_70"].fillna(0)
    rsi_lt_30 = df["PCT_MEMB_WITH_14D_RSI_LT_30"].fillna(0)
    sentiment = (50 - (rsi_lt_30 - rsi_gt_70)).clip(0, 100)

    # --- COMPOSITE ---
    composite = (
        0.40 * trend
        + 0.35 * conviction
        + 0.25 * sentiment
    ).clip(0, 100)

    result = pd.DataFrame(
        {
            "name": df.index,
            "composite": composite.round(1),
            "trend": trend.round(1),
            "conviction": conviction.round(1),
            "sentiment": sentiment.round(1),
        }
    ).reset_index(drop=True)

    result["rank"] = result["composite"].rank(ascending=False, method="min").astype(int)
    result = result.sort_values("rank").reset_index(drop=True)

    return result
