"""
Momentum scoring wrapper around the MARS engine.

Computes DMAS, technical, momentum, RSI, MA distances for all 20 IC instruments.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from mars_engine.mars_lite_scorer import (
    generate_spx_score_history,
    generate_csi_score_history,
    _generate_score_history,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MARS peer groups (from mars_lite_scorer)
# ---------------------------------------------------------------------------
MARS_PEERS = [
    "CCMP Index", "IBOV Index", "MEXBOL Index", "SXXP Index",
    "UKX Index", "SMI Index", "HSI Index", "SHSZ300 Index",
    "NKY Index", "SENSEX Index", "DAX Index", "MXWO Index",
    "USGG10YR Index", "GECU10YR Index", "CL1 Comdty",
    "GCA Comdty", "DXY Curncy", "XBTUSD Curncy",
]

# Map from instrument name → (mars_col, mars_hi, mars_lo, bloomberg_ticker)
INSTRUMENT_MAP = {
    "S&P 500":    ("SPX",      "SPX_high",       "SPX_low",       "SPX Index"),
    "CSI 300":    ("CSI",      "CSI_high",        "CSI_low",       "SHSZ300 Index"),
    "Nikkei 225": ("NKY",      "NKY_high",        "NKY_low",       "NKY Index"),
    "TASI":       ("TASI",     "TASI_high",       "TASI_low",      "SASEIDX Index"),
    "Sensex":     ("SENSEX",   "SENSEX_high",     "SENSEX_low",    "SENSEX Index"),
    "DAX":        ("DAX",      "DAX_high",        "DAX_low",       "DAX Index"),
    "SMI":        ("SMI",      "SMI_high",        "SMI_low",       "SMI Index"),
    "IBOV":       ("IBOV",     "IBOV_high",       "IBOV_low",      "IBOV Index"),
    "MEXBOL":     ("MEXBOL",   "MEXBOL_high",     "MEXBOL_low",    "MEXBOL Index"),
    "Gold":       ("GOLD",     "GOLD_high",       "GOLD_low",      "GCA Comdty"),
    "Silver":     ("SILVER",   "SILVER_high",     "SILVER_low",    "SIA Comdty"),
    "Platinum":   ("PLATINUM", "PLATINUM_high",   "PLATINUM_low",  "XPT Comdty"),
    "Palladium":  ("PALLADIUM","PALLADIUM_high",  "PALLADIUM_low", "XPD Curncy"),
    "Oil":        ("OIL",      "OIL_high",        "OIL_low",       "CL1 Comdty"),
    "Copper":     ("COPPER",   "COPPER_high",     "COPPER_low",    "LP1 Comdty"),
    "Bitcoin":    ("BTC",      "BTC_high",        "BTC_low",       "XBTUSD Curncy"),
    "Ethereum":   ("ETH",      "ETH_high",        "ETH_low",       "XETUSD Curncy"),
    "Ripple":     ("XRP",      "XRP_high",        "XRP_low",       "XRPUSD Curncy"),
    "Solana":     ("SOL",      "SOL_high",        "SOL_low",       "XSOUSD Curncy"),
    "Binance":    ("BNB",      "BNB_high",        "BNB_low",       "XBIUSD Curncy"),
}

# Bloomberg ticker → MARS column name (for peer data in the wide DataFrame)
TICKER_TO_MARS_COL = {
    "SPX Index":      "SPX",
    "SHSZ300 Index":  "CSI",
    "NKY Index":      "NKY",
    "SASEIDX Index":  "TASI",
    "SENSEX Index":   "SENSEX",
    "DAX Index":      "DAX",
    "SMI Index":      "SMI",
    "IBOV Index":     "IBOV",
    "MEXBOL Index":   "MEXBOL",
    "GCA Comdty":     "GOLD",
    "SIA Comdty":     "SILVER",
    "XPT Comdty":     "PLATINUM",
    "XPD Curncy":     "PALLADIUM",
    "CL1 Comdty":     "OIL",
    "LP1 Comdty":     "COPPER",
    "XBTUSD Curncy":  "BTC",
    "XETUSD Curncy":  "ETH",
    "XRPUSD Curncy":  "XRP",
    "XSOUSD Curncy":  "SOL",
    "XBIUSD Curncy":  "BNB",
    # Peers kept as-is
    "CCMP Index":     "CCMP Index",
    "SXXP Index":     "SXXP Index",
    "UKX Index":      "UKX Index",
    "HSI Index":      "HSI Index",
    "MXWO Index":     "MXWO Index",
    "USGG10YR Index": "USGG10YR Index",
    "GECU10YR Index": "GECU10YR Index",
    "DXY Curncy":     "DXY Curncy",
}


def _build_wide_df(master_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-format master_prices.csv to wide format for MARS engine.

    Long: date, ticker, close, low, high
    Wide: index=date, columns = [MARS_COL, MARS_COL_high, MARS_COL_low, ...]
    """
    # Pivot close prices
    close = master_prices.pivot(index="date", columns="ticker", values="close")
    high  = master_prices.pivot(index="date", columns="ticker", values="high")
    low   = master_prices.pivot(index="date", columns="ticker", values="low")

    # Rename to MARS column names
    close = close.rename(columns=TICKER_TO_MARS_COL)
    high  = high.rename(columns={t: f"{TICKER_TO_MARS_COL.get(t, t)}_high" for t in high.columns})
    low   = low.rename(columns={t: f"{TICKER_TO_MARS_COL.get(t, t)}_low"  for t in low.columns})

    wide = pd.concat([close, high, low], axis=1)
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    return wide


def _compute_technical_score(close: pd.Series, target_date: pd.Timestamp) -> int:
    """
    Simple technical score based on MA positioning (0-100 scale).

    Components (equal weight):
    - Price vs 50d MA  : 100 if above, 0 if below
    - Price vs 100d MA : 100 if above, 0 if below
    - Price vs 200d MA : 100 if above, 0 if below
    - RSI (14d) in neutral zone: scaled

    Returns integer 0-100.
    """
    close_up_to = close[close.index <= target_date].dropna()
    if len(close_up_to) < 200:
        return 50  # insufficient history

    price = close_up_to.iloc[-1]
    ma50  = close_up_to.iloc[-50:].mean()
    ma100 = close_up_to.iloc[-100:].mean()
    ma200 = close_up_to.iloc[-200:].mean()

    score_ma = (
        (100 if price > ma50 else 0)
        + (100 if price > ma100 else 0)
        + (100 if price > ma200 else 0)
    ) / 3

    return int(round(score_ma))


def _compute_rsi(close: pd.Series, target_date: pd.Timestamp, period: int = 14) -> int:
    """Compute RSI(14) at target_date."""
    close_up_to = close[close.index <= target_date].dropna()
    if len(close_up_to) < period + 1:
        return 50

    delta = close_up_to.diff().dropna()
    gains = delta.clip(lower=0).rolling(period).mean()
    losses = (-delta.clip(upper=0)).rolling(period).mean()

    rs = gains / losses.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))

    return int(round(float(rsi.iloc[-1])))


def _ma_distance_pct(close: pd.Series, target_date: pd.Timestamp, window: int) -> str:
    """Distance of last price from moving average, as formatted percentage string."""
    close_up_to = close[close.index <= target_date].dropna()
    if len(close_up_to) < window:
        return "N/A"
    price = float(close_up_to.iloc[-1])
    ma    = float(close_up_to.iloc[-window:].mean())
    pct   = (price - ma) / ma * 100
    sign  = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def _default_rating(dmas: int) -> str:
    """Convert DMAS score to rating label (5-tier scale)."""
    if dmas >= 70:   return "Bullish"
    if dmas >= 60:   return "Constructive"
    if dmas >= 40:   return "Neutral"
    if dmas >= 30:   return "Cautious"
    return "Bearish"


def compute_scores(
    master_prices: pd.DataFrame,
    instruments: List[str],
    target_date: Optional[pd.Timestamp] = None,
) -> Dict[str, Dict]:
    """
    Compute DMAS, technical, momentum, RSI, and MA distances for each instrument.

    Parameters
    ----------
    master_prices : pd.DataFrame
        Long format: date, ticker, close, low, high
    instruments : list of str
        Instrument names (keys in INSTRUMENT_MAP)
    target_date : pd.Timestamp, optional
        Date to score. Defaults to latest available date.

    Returns
    -------
    dict : {instrument_name: {dmas, technical, momentum, rsi, rating,
                               vs_50d, vs_100d, vs_200d, price, ...}}
    """
    wide = _build_wide_df(master_prices)

    if target_date is None:
        target_date = wide.index.max()

    wide_up_to = wide[wide.index <= target_date]

    results = {}

    for name in instruments:
        if name not in INSTRUMENT_MAP:
            log.warning("Unknown instrument: %s", name)
            continue

        mars_col, mars_hi, mars_lo, bbg_ticker = INSTRUMENT_MAP[name]

        if mars_col not in wide_up_to.columns:
            log.warning("Column %s not found in wide DataFrame for %s", mars_col, name)
            continue

        close_series = wide_up_to[mars_col].dropna()
        if close_series.empty:
            log.warning("No price data for %s", name)
            continue

        # Compute MARS momentum score
        try:
            if name == "S&P 500":
                score_series = generate_spx_score_history(wide_up_to)
            elif name == "CSI 300":
                score_series = generate_csi_score_history(wide_up_to)
            else:
                score_series = _generate_score_history(
                    wide_up_to,
                    target_col=mars_col,
                    hi_col=mars_hi if mars_hi in wide_up_to.columns else mars_col,
                    lo_col=mars_lo if mars_lo in wide_up_to.columns else mars_col,
                    peer_universe=MARS_PEERS,
                    bench_candidates=["MXWO Index", "SPX"],
                )
        except Exception as e:
            log.warning("MARS scoring failed for %s: %s", name, e)
            score_series = pd.Series(dtype=float)

        momentum_score = int(round(float(score_series.iloc[-1]))) if not score_series.empty else 50

        # Technical score
        technical = _compute_technical_score(close_series, target_date)

        # DMAS = average of technical and momentum
        dmas = int(round((technical + momentum_score) / 2))

        # RSI
        rsi = _compute_rsi(close_series, target_date)

        # MA distances
        vs_50d  = _ma_distance_pct(close_series, target_date, 50)
        vs_100d = _ma_distance_pct(close_series, target_date, 100)
        vs_200d = _ma_distance_pct(close_series, target_date, 200)

        # Current price
        price = float(close_series.iloc[-1])

        results[name] = {
            "ticker":       bbg_ticker,
            "dmas":         dmas,
            "technical":    technical,
            "momentum":     momentum_score,
            "rsi":          rsi,
            "rating":       _default_rating(dmas),
            "vs_50d":       vs_50d,
            "vs_100d":      vs_100d,
            "vs_200d":      vs_200d,
            "price":        price,
            "score_date":   str(target_date.date()),
        }

        log.info(
            "%-12s  DMAS=%3d  Tech=%3d  Mom=%3d  RSI=%3d  vs50d=%s",
            name, dmas, technical, momentum_score, rsi, vs_50d,
        )

    return results
