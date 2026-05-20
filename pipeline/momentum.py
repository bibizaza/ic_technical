"""
Momentum scoring — matches production technical_score_wrapper exactly.

Computes DMAS, technical, momentum, RSI, MA distances for all 20 IC instruments.
- Technical: graduated MA-distance formula (same as compute_technical_score_only)
- Momentum: reads from mars_score sheet in ic_file.xlsx (same as production)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
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
    Convert long-format master_prices DataFrame to wide format indexed by date.

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


def _compute_technical_score(close: pd.Series, target_date: pd.Timestamp) -> float:
    """
    Technical score based on MA positioning (0-100 scale).

    Graduated scoring (matches production compute_technical_score_only):
    - Price vs 50d MA  : 0-30 pts (below: 0-15, above: 15-30)
    - Price vs 100d MA : 0-30 pts (below: 0-15, above: 15-30)
    - Price vs 200d MA : 0-40 pts (below: 0-20, above: 20-40)
    Total weight = 100 pts → score is already 0-100.

    Returns float so that DMAS averaging uses the untruncated value (matching
    production compute_technical_score_only which also returns a float).
    """
    close_up_to = close[close.index <= target_date].dropna()
    if len(close_up_to) < 200:
        return 50  # insufficient history

    cp   = close_up_to.iloc[-1]
    s50  = close_up_to.iloc[-50:].mean()
    s100 = close_up_to.iloc[-100:].mean()
    s200 = close_up_to.iloc[-200:].mean()

    score = 0.0
    # vs 50-day MA (30% weight)
    if cp > s50:
        score += min(30, 15 + ((cp - s50) / s50 * 100) * 3)
    else:
        score += max(0, 15 - ((s50 - cp) / s50 * 100) * 3)
    # vs 100-day MA (30% weight)
    if cp > s100:
        score += min(30, 15 + ((cp - s100) / s100 * 100) * 3)
    else:
        score += max(0, 15 - ((s100 - cp) / s100 * 100) * 3)
    # vs 200-day MA (40% weight)
    if cp > s200:
        score += min(40, 20 + ((cp - s200) / s200 * 100) * 4)
    else:
        score += max(0, 20 - ((s200 - cp) / s200 * 100) * 4)

    return max(0.0, min(100.0, score))


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


def _compute_mars_momentum(close: pd.Series, target_date: pd.Timestamp) -> Optional[float]:
    """
    In-script MARS-style momentum score (0-100).

    Reproduces the Absolute Score from the MARS Engine spec (Pure Momentum,
    Trend Smoothness, Sharpe, ADX). The 10% Idiosyncratic-Momentum factor is
    skipped (requires a per-asset benchmark regression); its weight is
    redistributed pro-rata across the remaining four factors so the weights
    sum to 100%.

    Each factor is converted to a 0-100 percentile rank against its own
    rolling history (up to 5 years of trading days) with winsorization at
    the 2nd/98th percentile to neutralize one-off anomalies.

    Returns None if there isn't enough history to compute a stable score.
    """
    close = close.sort_index().dropna()
    asof = close[close.index <= target_date]
    if len(asof) < 260:  # need at least ~12 months for the 12M return
        return None

    # ---------- Build daily history of each factor over the full series ----
    rets = asof.pct_change()

    # 1) Pure Momentum (weighted avg of 12M, 6M, 3M total returns, 40% weight)
    pm_series = (
        0.50 * (asof / asof.shift(252) - 1)
        + 0.30 * (asof / asof.shift(126) - 1)
        + 0.20 * (asof / asof.shift(63)  - 1)
    )

    # 2) Trend Smoothness (positive-day ratio over 6M, 20% weight)
    pos_day = (rets > 0).astype(float)
    smooth_series = pos_day.rolling(126, min_periods=80).mean()

    # 3) Risk-Adjusted Return (6M annualized Sharpe, 20% weight)
    mean_6m = rets.rolling(126, min_periods=80).mean()
    std_6m  = rets.rolling(126, min_periods=80).std()
    sharpe_series = (mean_6m / std_6m.replace(0, np.nan)) * (252 ** 0.5)

    # 4) Trend Strength (14-day ADX proxy, 10% weight)
    # ADX needs OHL data; this code only has close. Use a robust proxy:
    # |EMA(returns, 14)| / EMA(|returns|, 14) — scaled 0-100. Captures the
    # same trend-vs-chop concept as ADX without needing high/low ranges.
    abs_rets = rets.abs()
    adx_proxy = (
        rets.ewm(span=14, min_periods=14).mean().abs()
        / abs_rets.ewm(span=14, min_periods=14).mean().replace(0, np.nan)
    ) * 100.0

    factor_specs = [
        ("pm",     pm_series,     50.0),
        ("smooth", smooth_series, 22.2),
        ("sharpe", sharpe_series, 22.2),
        ("adx",    adx_proxy,     5.6),
    ]
    # Pro-rated weights (40/20/20/10 normalized to sum=100 after dropping 10%
    # idiosyncratic): 40→50, 20→22.2, 20→22.2, 10→5.6 — sum to 100%.

    total_score = 0.0
    total_weight = 0.0
    history_window = 252 * 5  # 5-year rolling distribution

    for _name, series, weight in factor_specs:
        # History distribution for percentile rank (winsorized 2nd/98th).
        # Use only history strictly prior to target_date for rank context.
        hist = series[series.index < target_date].dropna()
        if len(hist) < 60:
            continue
        hist = hist.iloc[-history_window:]
        lo, hi = np.nanpercentile(hist, [2.0, 98.0])
        hist_w = hist.clip(lo, hi)
        current = series.loc[asof.index[asof.index <= target_date]].dropna()
        if current.empty:
            continue
        cur_val = float(current.iloc[-1])
        cur_clipped = max(lo, min(hi, cur_val))
        # Percentile rank within winsorized history (0-100)
        pct = (hist_w <= cur_clipped).mean() * 100.0
        total_score += pct * weight
        total_weight += weight

    if total_weight == 0:
        return None
    return total_score / total_weight


def _lookup_mars_score(mars_df: "pd.DataFrame", ticker: str) -> Optional[float]:
    """
    Look up momentum score for a ticker in a pre-loaded mars_score DataFrame.
    Mirrors the matching logic of _get_momentum_score_generic.
    """
    if mars_df is None or mars_df.empty:
        return None
    try:
        score_col = None
        for col in mars_df.columns:
            if str(col).lower() in ('score', 'mars', 'momentum', 'mars_score', 'momentum_score'):
                score_col = col
                break
        if score_col is None and len(mars_df.columns) >= 2:
            score_col = mars_df.columns[1]
        if score_col is None:
            return None
        ticker_col = mars_df.columns[0]
        ticker_upper = ticker.strip().upper()
        first_part = ticker_upper.split()[0] if ticker_upper.split() else ticker_upper
        for _, row in mars_df.iterrows():
            cell = str(row[ticker_col]).strip().upper()
            if cell == ticker_upper or first_part in cell or cell in ticker_upper:
                val = row[score_col]
                if pd.notna(val):
                    return float(val)
    except Exception:
        pass
    return None


def _default_rating(dmas: int) -> str:
    """Convert DMAS score to rating label (5-tier scale).
    Thresholds match production get_outlook() in data_prep.py: 70/55/45/30.
    """
    if dmas >= 70:   return "Bullish"
    if dmas >= 55:   return "Constructive"
    if dmas >= 45:   return "Neutral"
    if dmas >= 30:   return "Cautious"
    return "Bearish"


def compute_scores(
    master_prices: pd.DataFrame,
    instruments: List[str],
    target_date: Optional[pd.Timestamp] = None,
    excel_path: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Compute DMAS, technical, momentum, RSI, and MA distances for each instrument.

    Matches production scoring exactly:
    - Technical: graduated MA-distance formula (same as compute_technical_score_only)
    - Momentum: reads from mars_score sheet in Excel if excel_path provided;
                falls back to technical score as proxy (same as production)

    Parameters
    ----------
    master_prices : pd.DataFrame
        Long format: date, ticker, close, low, high
    instruments : list of str
        Instrument names (keys in INSTRUMENT_MAP)
    target_date : pd.Timestamp, optional
        Date to score. Defaults to latest available date.
    excel_path : str, optional
        Path to ic_file.xlsx to read pre-computed momentum scores from mars_score sheet.

    Returns
    -------
    dict : {instrument_name: {dmas, technical, momentum, rsi, rating,
                               vs_50d, vs_100d, vs_200d, price, ...}}
    """
    wide = _build_wide_df(master_prices)

    if target_date is None:
        target_date = wide.index.max()

    wide_up_to = wide[wide.index <= target_date]

    # Pre-load mars_score sheet once for all instruments
    _mars_df = None
    if excel_path:
        try:
            _mars_df = pd.read_excel(excel_path, sheet_name="mars_score")
        except Exception as _e:
            log.warning("Could not read mars_score sheet from %s: %s", excel_path, _e)

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

        # Technical score (graduated formula, matches production).
        # Keep as float for DMAS computation; truncate only for display.
        technical_float = _compute_technical_score(close_series, target_date)
        technical = int(technical_float)   # truncate to match production

        # Momentum score: compute in-script using MARS-style factors (Pure
        # Momentum + Trend Smoothness + Sharpe + ADX proxy, percentile-ranked
        # against the asset's own 5-year history). Falls back to the stale
        # mars_score Excel sheet only if the in-script computation fails (e.g.
        # insufficient history); falls back to technical as a last resort.
        momentum_float = _compute_mars_momentum(close_series, target_date)
        if momentum_float is None and _mars_df is not None:
            _ex = _lookup_mars_score(_mars_df, bbg_ticker)
            if _ex is not None:
                momentum_float = _ex
        if momentum_float is None:
            log.warning("MARS momentum unavailable for %s — using technical as proxy", name)
            momentum_float = technical_float

        momentum_score = int(momentum_float)   # truncate to match production

        # DMAS = average of raw floats, then truncate (matches production int(dmas))
        dmas = int((technical_float + momentum_float) / 2)

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
