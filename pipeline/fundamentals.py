"""
Cross-sectional fundamental ranking for 9 equity indices.

Replaces the Excel-based ic_file.xlsx data_fundamental sheet with a
direct Bloomberg pull + Python ranking (same pattern as HERA score_v2).
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Pillar definitions ────────────────────────────────────────────────────
# Each pillar: list of (bloomberg_field, direction)
#   "lower" → lower value = rank 1 (best)
#   "higher" → higher value = rank 1 (best)

PILLARS = {
    "Value": [
        ("BEST_PE_RATIO",     "lower"),
        ("PX_TO_BOOK_RATIO",  "lower"),
        ("EV_TO_T12M_EBITDA", "lower"),
    ],
    "Growth": [
        ("EST_LTG_EPS_AGGTE", "higher"),
    ],
    "Profitability": [
        ("OPER_MARGIN",       "higher"),
        ("PROF_MARGIN",       "higher"),
    ],
    "Quality": [
        ("BEST_ROE",          "higher"),
    ],
    "Leverage": [
        ("NET_DEBT_TO_EBITDA", "lower"),
    ],
    "Dividend": [
        ("BEST_DIV_YLD",      "higher"),
    ],
}

# Short column name for output (matches history.json keys)
PILLAR_RANK_KEYS = {
    "Value":         "value_rank",
    "Growth":        "growth_rank",
    "Profitability": "profitability_rank",
    "Quality":       "quality_rank",
    "Leverage":      "leverage_rank",
    "Dividend":      "dividend_rank",
}


def compute_fundamental_ranks(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional ranking of 9 equity indices across 6 pillars.

    Parameters
    ----------
    raw_data : pd.DataFrame
        Index = instrument name (e.g. "S&P 500"), columns = Bloomberg fields.
        Output of pull_fundamentals().

    Returns
    -------
    pd.DataFrame
        Columns: index_name, fundamental_rank, value_rank, growth_rank,
                 profitability_rank, quality_rank, leverage_rank, dividend_rank
        Sorted by fundamental_rank ascending (rank 1 = best).
    """
    pillar_ranks = {}

    for pillar_name, metrics in PILLARS.items():
        metric_ranks = []

        for field, direction in metrics:
            if field not in raw_data.columns:
                log.warning("Field %s missing from data — skipping", field)
                continue

            values = raw_data[field]
            if values.isna().all():
                log.warning("All NaN for %s — skipping", field)
                continue

            ascending = direction == "lower"
            ranks = values.rank(ascending=ascending, method="min", na_option="bottom")
            metric_ranks.append(ranks)

        if not metric_ranks:
            # All metrics missing — assign neutral rank
            pillar_ranks[pillar_name] = pd.Series(5, index=raw_data.index, dtype=float)
            continue

        # Average sub-metric ranks, then re-rank 1–9
        avg_ranks = pd.concat(metric_ranks, axis=1).mean(axis=1)
        pillar_ranks[pillar_name] = avg_ranks.rank(method="min", ascending=True)

    # Composite: average of 6 pillar ranks, then re-rank 1–9
    pillar_df = pd.DataFrame(pillar_ranks)
    composite_score = pillar_df.mean(axis=1)
    composite_rank = composite_score.rank(method="min", ascending=True).astype(int)

    # Build output
    result = pd.DataFrame({
        "index_name":          raw_data.index,
        "fundamental_rank":    composite_rank.values,
        "value_rank":          pillar_df["Value"].astype(int).values,
        "growth_rank":         pillar_df["Growth"].astype(int).values,
        "profitability_rank":  pillar_df["Profitability"].astype(int).values,
        "quality_rank":        pillar_df["Quality"].astype(int).values,
        "leverage_rank":       pillar_df["Leverage"].astype(int).values,
        "dividend_rank":       pillar_df["Dividend"].astype(int).values,
    })
    result = result.sort_values("fundamental_rank").reset_index(drop=True)

    # Log rankings
    log.info("Fundamental ranks computed:")
    for _, row in result.iterrows():
        log.info(
            "  #%d %s — Val=%d Grw=%d Prof=%d Qual=%d Lev=%d Div=%d",
            row["fundamental_rank"], row["index_name"],
            row["value_rank"], row["growth_rank"],
            row["profitability_rank"], row["quality_rank"],
            row["leverage_rank"], row["dividend_rank"],
        )

    return result
