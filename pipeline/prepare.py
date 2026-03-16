"""
Stage: prepare

Pulls Bloomberg data, computes all scores, renders charts,
and saves draft_state.json (subtitles left null for Claude Code to fill).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

log = logging.getLogger(__name__)


def _parse_bloomberg_raw_csv(csv_path: str) -> pd.DataFrame:
    """
    Parse raw Bloomberg Excel-export CSV (semicolon-separated, multi-row header)
    into long format: date, ticker, close, low, high.

    Row 0: empty ; ticker ; ticker ; ticker ; ticker ; ticker ; ticker ; ...
            (each ticker repeated 3 times for price/low/high)
    Row 1: DATES ; #price ; #low ; #high ; #price ; #low ; #high ; ...
    Row 2+: DD/MM/YYYY ; val ; val ; val ; ...
    """
    with open(csv_path, encoding="utf-8-sig") as f:
        lines = f.readlines()

    # Parse header rows
    ticker_row = lines[0].strip().split(";")
    field_row = lines[1].strip().split(";")

    # Build (ticker, field) pairs — skip first column (DATES)
    columns = []
    for i in range(1, len(ticker_row)):
        ticker = ticker_row[i].strip()
        field = field_row[i].strip().lstrip("#") if i < len(field_row) else ""
        columns.append((ticker, field))

    # Parse data rows
    records = []
    for line in lines[2:]:
        parts = line.strip().split(";")
        if not parts or not parts[0]:
            continue
        date_str = parts[0].strip()
        try:
            date = pd.to_datetime(date_str, dayfirst=True)
        except Exception:
            continue

        # Group values by ticker
        ticker_data: dict = {}
        for i, (ticker, field) in enumerate(columns):
            if not ticker or not field:
                continue
            val_str = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if val_str in ("", "#N/A", "N/A"):
                val = float("nan")
            else:
                try:
                    val = float(val_str)
                except ValueError:
                    val = float("nan")

            if ticker not in ticker_data:
                ticker_data[ticker] = {}
            ticker_data[ticker][field] = val

        for ticker, fields in ticker_data.items():
            close = fields.get("price", float("nan"))
            low = fields.get("low", float("nan"))
            high = fields.get("high", float("nan"))
            if pd.notna(close):
                records.append({
                    "date": date,
                    "ticker": ticker,
                    "close": close,
                    "low": low,
                    "high": high,
                })

    df = pd.DataFrame(records)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    log.info("Parsed Bloomberg raw CSV: %d rows, %d tickers", len(df), df["ticker"].nunique())
    return df


def _read_master_csv(csv_path: str) -> pd.DataFrame:
    """
    Read master_prices.csv, auto-detecting whether it's in tidy long format
    (date,ticker,close,low,high) or raw Bloomberg export format (semicolons).
    """
    with open(csv_path, encoding="utf-8-sig") as f:
        first_line = f.readline()

    if ";" in first_line:
        # Raw Bloomberg format
        log.info("Detected raw Bloomberg CSV format — parsing...")
        return _parse_bloomberg_raw_csv(csv_path)
    else:
        # Already in tidy long format
        return pd.read_csv(csv_path, parse_dates=["date"])


def _load_config(config_path: str = "config/tickers.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _format_market_cap(value: Optional[float]) -> str:
    """Format market cap value to human-readable string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    if value >= 1e12:
        return f"{value/1e12:.2f}T"
    if value >= 1e9:
        return f"{value/1e9:.1f}B"
    return f"{value:.0f}M"


def _build_enriched_context(
    name: str,
    scores: dict,
    market_cap: str,
    breadth_rank: int,
    fundamental_rank: int,
    history: list,
) -> str:
    """
    Build the enriched context block for subtitle generation.
    Mirrors the logic of _build_enriched_context() in claude_generator.py.
    """
    lines = []
    lines.append(f"=== {name} ===")

    # Current scores
    lines.append(f"DMAS: {scores['dmas']} | Technical: {scores['technical']} | Momentum: {scores['momentum']} | RSI: {scores['rsi']}")
    lines.append(f"Rating: {scores['rating']}")
    lines.append(f"Price vs 50d MA: {scores['vs_50d']} | vs 100d MA: {scores['vs_100d']} | vs 200d MA: {scores['vs_200d']}")

    if market_cap != "N/A":
        lines.append(f"Market Cap: {market_cap}")

    lines.append(f"Breadth rank: {breadth_rank}/9 | Fundamental rank: {fundamental_rank}/9")

    # Historical trend (last 4 weeks from history.json)
    if history and len(history) >= 2:
        recent = sorted(history, key=lambda x: x["date"])[-4:]
        trend_parts = [f"{h['date'][:10]}: DMAS={h.get('dmas','?')} ({h.get('rating','?')})" for h in recent]
        lines.append("Recent history: " + " | ".join(trend_parts))

        # Rating streak
        last_rating = recent[-1].get("rating", "")
        streak = sum(1 for h in reversed(recent) if h.get("rating") == last_rating)
        if streak > 1:
            lines.append(f"Rating streak: {last_rating} for {streak} consecutive weeks")

    # Correction depth (if below 200d MA)
    vs_200d_str = scores.get("vs_200d", "N/A")
    if vs_200d_str.startswith("-"):
        lines.append(f"Correction: {vs_200d_str} below 200d MA")

    return "\n".join(lines)


def run_prepare(
    date: Optional[str] = None,
    skip_bloomberg: bool = False,
    output_dir: str = ".",
    draft_path: str = "draft_state.json",
    master_csv: Optional[str] = None,
    config_path: str = "config/tickers.yaml",
    charts_dir: str = "charts_cache",
) -> None:
    """
    Main prepare stage:
    1. Pull Bloomberg data (unless --skip-bloomberg)
    2. Compute scores
    3. Render charts
    4. Save draft_state.json
    """
    cfg = _load_config(config_path)

    # Resolve paths
    dropbox_path = os.environ.get(
        "IC_DROPBOX_PATH",
        "/Users/larazanella/Library/CloudStorage/Dropbox/Tools_In_Construction/ic",
    )
    if master_csv is None:
        master_csv = str(Path(dropbox_path) / "master_prices.csv")

    history_path = Path("market_compass/data/history.json")

    # Load history
    if history_path.exists():
        with open(history_path) as f:
            history_data = json.load(f)
    else:
        history_data = {}

    # -----------------------------------------------------------------------
    # Step 1: Bloomberg data pull
    # -----------------------------------------------------------------------
    if not skip_bloomberg:
        log.info("Pulling Bloomberg data...")
        try:
            from pipeline.bloomberg import connect_bloomberg, pull_prices, pull_breadth, pull_fundamentals, pull_market_caps
            session = connect_bloomberg()
            try:
                all_tickers = _collect_all_tickers(cfg)
                master_df = pull_prices(session, all_tickers, master_csv)

                breadth_indices = cfg["breadth_indices"]
                raw_breadth = pull_breadth(session, breadth_indices)

                equity_indices = breadth_indices  # same indices for fundamentals
                raw_fundamentals = pull_fundamentals(session, equity_indices)

                ic_tickers = _get_ic_tickers(cfg)
                market_caps_raw = pull_market_caps(session, ic_tickers)
            finally:
                from pipeline.bloomberg import disconnect_bloomberg
                disconnect_bloomberg(session)
        except Exception as e:
            log.error("Bloomberg pull failed: %s", e)
            log.info("Falling back to existing master_prices.csv (no new data)")
            master_df = _read_master_csv(master_csv)
            raw_breadth = pd.DataFrame()
            raw_fundamentals = pd.DataFrame()
            market_caps_raw = {}
    else:
        log.info("Skipping Bloomberg (--skip-bloomberg flag set)")
        master_df = _read_master_csv(master_csv)
        raw_breadth = pd.DataFrame()
        raw_fundamentals = pd.DataFrame()
        market_caps_raw = {}

    # -----------------------------------------------------------------------
    # Step 2: Determine target date
    # -----------------------------------------------------------------------
    master_df["date"] = pd.to_datetime(master_df["date"])
    if date:
        target_date = pd.Timestamp(date)
    else:
        target_date = master_df["date"].max()

    log.info("Computing scores for date: %s", target_date.date())

    # -----------------------------------------------------------------------
    # Step 3: Compute momentum scores
    # -----------------------------------------------------------------------
    from pipeline.momentum import compute_scores, INSTRUMENT_MAP
    instrument_names = list(INSTRUMENT_MAP.keys())
    scores_dict = compute_scores(master_df, instrument_names, target_date)

    # -----------------------------------------------------------------------
    # Step 4: Compute breadth
    # -----------------------------------------------------------------------
    from pipeline.breadth import compute_composite_breadth
    if not raw_breadth.empty:
        breadth_df = compute_composite_breadth(raw_breadth)
        breadth_records = breadth_df.to_dict(orient="records")
    else:
        breadth_records = []

    # -----------------------------------------------------------------------
    # Step 5: Fundamental rankings
    # -----------------------------------------------------------------------
    if not raw_fundamentals.empty:
        fundamental_records = _compute_fundamental_ranks(raw_fundamentals)
    else:
        fundamental_records = {}

    # -----------------------------------------------------------------------
    # Step 6: Market caps
    # -----------------------------------------------------------------------
    ticker_to_name = {}
    for section in ["equity", "commodity", "crypto"]:
        for instr in cfg["ic_instruments"].get(section, []):
            ticker_to_name[instr["ticker"]] = instr["name"]

    market_caps = {}
    for ticker, val in market_caps_raw.items():
        name = ticker_to_name.get(ticker, ticker)
        market_caps[name] = _format_market_cap(val)

    # -----------------------------------------------------------------------
    # Step 7: Build breadth/fundamental ranks per instrument
    # -----------------------------------------------------------------------
    # Map equity index names to breadth ranks
    breadth_rank_map = {}
    if breadth_records:
        for rec in breadth_records:
            breadth_rank_map[rec["name"]] = rec["rank"]

    fundamental_rank_map = {}
    if fundamental_records:
        for nm, rank in fundamental_records.items():
            fundamental_rank_map[nm] = rank

    # -----------------------------------------------------------------------
    # Step 8: Render charts
    # -----------------------------------------------------------------------
    charts_cache = Path(charts_dir)
    charts_cache.mkdir(exist_ok=True)

    chart_paths = _render_charts(
        master_df=master_df,
        scores_dict=scores_dict,
        target_date=target_date,
        charts_dir=charts_cache,
        cfg=cfg,
    )

    # -----------------------------------------------------------------------
    # Step 9: Build draft_state.json
    # -----------------------------------------------------------------------
    instruments_out = {}
    for name, scores in scores_dict.items():
        mc = market_caps.get(name, "N/A")
        b_rank = breadth_rank_map.get(name, 5)  # default mid-rank
        f_rank = fundamental_rank_map.get(name, 5)
        hist = history_data.get(name, [])

        enriched = _build_enriched_context(name, scores, mc, b_rank, f_rank, hist)

        instruments_out[name] = {
            **scores,
            "market_cap":       mc,
            "breadth_rank":     b_rank,
            "fundamental_rank": f_rank,
            "enriched_context": enriched,
            "subtitle":         None,
            "chart_path":       chart_paths.get(name, ""),
        }

    draft_state = {
        "date":           str(target_date.date()),
        "generated_at":   datetime.now().isoformat(),
        "instruments":    instruments_out,
        "breadth":        breadth_records,
        "fundamentals":   fundamental_records,
        "market_caps":    market_caps,
    }

    draft_out = Path(output_dir) / draft_path
    with open(draft_out, "w") as f:
        json.dump(draft_state, f, indent=2, default=str)

    log.info("draft_state.json saved to %s", draft_out)
    print(f"\n✓ Prepare complete. Draft saved to: {draft_out}")
    print(f"  Date: {target_date.date()}")
    print(f"  Instruments: {len(instruments_out)}")
    print(f"  Charts: {len(chart_paths)}")
    print("\nNext step: Claude Code reads draft_state.json and writes subtitles.")


def _collect_all_tickers(cfg: dict) -> list:
    """Collect all unique tickers from config."""
    tickers = set()
    for section in ["equity", "commodity", "crypto"]:
        for instr in cfg["ic_instruments"].get(section, []):
            tickers.add(instr["ticker"])
    for section in ["equity", "rates", "credit", "commodity", "fx", "crypto"]:
        for item in cfg["performance_tickers"].get(section, []):
            tickers.add(item["ticker"])
    for idx in cfg.get("breadth_indices", []):
        tickers.add(idx["ticker"])
    for peer in cfg.get("mars_peers", []):
        tickers.add(peer)
    return list(tickers)


def _get_ic_tickers(cfg: dict) -> list:
    """Get tickers for 20 IC instruments only."""
    tickers = []
    for section in ["equity", "commodity", "crypto"]:
        for instr in cfg["ic_instruments"].get(section, []):
            tickers.append(instr["ticker"])
    return tickers


def _compute_fundamental_ranks(raw_fundamentals: pd.DataFrame) -> dict:
    """
    Compute composite fundamental rank for each index.
    Higher PE/debt is worse; higher earnings yield/ROE/growth is better.
    Returns {name: rank} where rank=1 is best.
    """
    df = raw_fundamentals.copy()

    scores = pd.Series(0.0, index=df.index)

    # Higher is better
    for col in ["EARN_YLD", "EST_LTG_EPS_AGGTE", "PROF_MARGIN", "RETURN_ON_EQY", "DVD_YLD"]:
        if col in df.columns:
            ranked = df[col].rank(ascending=True, na_option="bottom")
            scores += ranked

    # Lower is better (invert)
    for col in ["BEST_PE_RATIO", "TOT_DEBT_TO_TOT_EQY"]:
        if col in df.columns:
            ranked = df[col].rank(ascending=False, na_option="bottom")
            scores += ranked

    final_rank = scores.rank(ascending=False, method="min").astype(int)
    return dict(final_rank)


def _render_charts(
    master_df: pd.DataFrame,
    scores_dict: dict,
    target_date: pd.Timestamp,
    charts_dir: Path,
    cfg: dict,
) -> dict:
    """
    Render technical charts for each instrument using existing technical_analysis modules.
    Returns {instrument_name: chart_path_str}
    """
    from pipeline.momentum import INSTRUMENT_MAP

    chart_paths = {}

    # Module mapping: instrument name → (module_path, function_prefix)
    chart_module_map = {
        "S&P 500":    ("technical_analysis.equity.spx",      "spx"),
        "CSI 300":    ("technical_analysis.equity.csi",      "csi"),
        "Nikkei 225": ("technical_analysis.equity.nikkei",   "nikkei"),
        "TASI":       ("technical_analysis.equity.tasi",     "tasi"),
        "Sensex":     ("technical_analysis.equity.sensex",   "sensex"),
        "DAX":        ("technical_analysis.equity.dax",      "dax"),
        "SMI":        ("technical_analysis.equity.smi",      "smi"),
        "IBOV":       ("technical_analysis.equity.ibov",     "ibov"),
        "MEXBOL":     ("technical_analysis.equity.mexbol",   "mexbol"),
        "Gold":       ("technical_analysis.commodity.gold",  "gold"),
        "Silver":     ("technical_analysis.commodity.silver","silver"),
        "Platinum":   ("technical_analysis.commodity.platinum","platinum"),
        "Palladium":  ("technical_analysis.commodity.palladium","palladium"),
        "Oil":        ("technical_analysis.commodity.oil",   "oil"),
        "Copper":     ("technical_analysis.commodity.copper","copper"),
        "Bitcoin":    ("technical_analysis.crypto.bitcoin",  "bitcoin"),
        "Ethereum":   ("technical_analysis.crypto.ethereum", "ethereum"),
        "Ripple":     ("technical_analysis.crypto.ripple",   "ripple"),
        "Solana":     ("technical_analysis.crypto.solana",   "solana"),
        "Binance":    ("technical_analysis.crypto.binance",  "binance"),
    }

    for name, (mod_path, prefix) in chart_module_map.items():
        chart_file = charts_dir / f"{prefix}_chart.png"

        try:
            import importlib
            mod = importlib.import_module(mod_path)
            make_fn = getattr(mod, f"make_{prefix}_figure", None)

            if make_fn is None:
                log.warning("make_%s_figure not found in %s", prefix, mod_path)
                chart_paths[name] = ""
                continue

            # Build the DataFrame expected by the chart function
            mars_col, _, _, bbg_ticker = INSTRUMENT_MAP[name]
            df_chart = _build_chart_df(master_df, bbg_ticker, mars_col, target_date, master_df)

            fig = make_fn(df_chart)

            # Export to PNG via kaleido
            img_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
            with open(chart_file, "wb") as f:
                f.write(img_bytes)

            chart_paths[name] = str(chart_file)
            log.info("Chart saved: %s", chart_file)

        except Exception as e:
            log.warning("Chart generation failed for %s: %s", name, e)
            chart_paths[name] = ""

    return chart_paths


def _build_chart_df(
    master_df: pd.DataFrame,
    bbg_ticker: str,
    mars_col: str,
    target_date: pd.Timestamp,
    full_master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the DataFrame expected by make_*_figure() functions.
    Returns DataFrame with Date, Price columns (+ peer columns where available).
    """
    mask = (master_df["ticker"] == bbg_ticker) & (master_df["date"] <= target_date)
    sub = master_df[mask].sort_values("date").tail(200)

    if sub.empty:
        return pd.DataFrame(columns=["Date", "Price"])

    df = sub[["date", "close"]].rename(columns={"date": "Date", "close": "Price"})
    df = df.set_index("Date")

    # Add high/low if available
    if "high" in sub.columns:
        df[f"{mars_col}_high"] = sub["high"].values
    if "low" in sub.columns:
        df[f"{mars_col}_low"] = sub["low"].values

    return df.reset_index()
