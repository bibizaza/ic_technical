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

from pipeline._atomic import atomic_write_text

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


def _compute_streak_weeks(history: list, current_rating: str) -> int:
    """
    Count consecutive ISO weeks (from most recent, backwards) where the
    latest entry in each week matches current_rating.

    history.json has multiple entries per week (one per pipeline run).
    We deduplicate to one entry per ISO week (latest by date) before counting.
    """
    from datetime import date as _date

    if not history or not current_rating:
        return 1

    # Build week → latest-entry map
    week_latest: dict = {}
    for entry in history:
        try:
            d = _date.fromisoformat(entry["date"][:10])
        except (KeyError, ValueError):
            continue
        yw = d.isocalendar()[:2]  # (year, week)
        if yw not in week_latest or entry["date"] > week_latest[yw]["date"]:
            week_latest[yw] = entry

    if not week_latest:
        return 1

    # Sort weeks descending
    sorted_weeks = sorted(week_latest.keys(), reverse=True)

    streak = 0
    for yw in sorted_weeks:
        rating = week_latest[yw].get("rating", "")
        if rating == current_rating:
            streak += 1
        else:
            break

    return max(streak, 1)


def _build_enriched_context(
    name: str,
    scores: dict,
    market_cap: str,
    breadth_rank: int,
    fundamental_rank: int,
    history: list,
    streak_weeks: int = 1,
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

    # Streak (pre-computed, week-deduplicated)
    if streak_weeks > 1:
        lines.append(f"Rating streak: {scores['rating']} for {streak_weeks} consecutive weeks (pre-verified)")

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

    # history.json lives in Dropbox so all runners (Mac, intern PC) share it.
    # Falls back to the legacy in-repo path if Dropbox is missing.
    _dropbox_hist = Path(dropbox_path) / "history.json"
    _legacy_hist = Path("market_compass/data/history.json")
    history_path = _dropbox_hist if _dropbox_hist.parent.exists() else _legacy_hist

    # Load history
    if history_path.exists():
        with open(history_path) as f:
            history_data = json.load(f)
    else:
        history_data = {}

    # -----------------------------------------------------------------------
    # Step 1: Bloomberg data pull
    # -----------------------------------------------------------------------
    # Always read prices from the raw Bloomberg CSV (maintained separately)
    master_df = _read_master_csv(master_csv)

    if not skip_bloomberg:
        log.info("Pulling Bloomberg prices + reference data (breadth, fundamentals, market caps)...")
        try:
            from pipeline.bloomberg import (
                connect_bloomberg, disconnect_bloomberg,
                update_master_prices_wide,
                pull_breadth, pull_fundamentals, pull_market_caps,
                BloombergPartialFailure,
            )
            session = connect_bloomberg()
            try:
                # 1. Pull new price rows into master_prices.csv
                n_new = update_master_prices_wide(session, master_csv)
                if n_new:
                    log.info("Appended %d new price rows — reloading master_df", n_new)
                    master_df = _read_master_csv(master_csv)

                # 2. Pull reference data
                breadth_indices = cfg["breadth_indices"]
                raw_breadth = pull_breadth(session, breadth_indices)

                equity_indices = breadth_indices
                raw_fundamentals = pull_fundamentals(session, equity_indices)

                ic_tickers = _get_ic_tickers(cfg)
                market_caps_raw = pull_market_caps(session, ic_tickers)
            finally:
                disconnect_bloomberg(session)
        except BloombergPartialFailure as e:
            # Bloomberg was reachable but one or more BDH batches failed —
            # distinct from a connection problem. Don't claim "not reachable".
            log.error("Bloomberg partial-batch failure: %s", e)
            try:
                import os as _os, urllib.parse as _up, urllib.request as _ur
                _token = _os.environ.get("TELEGRAM_BOT_TOKEN", "")
                _chat = _os.environ.get("TELEGRAM_CHAT_ID", "979257663")
                if _token:
                    _url = f"https://api.telegram.org/bot{_token}/sendMessage"
                    _data = _up.urlencode({
                        "chat_id": _chat,
                        "text": f"Bloomberg BDH partial failure (price data NOT updated), IC pipeline aborted: {e}",
                    }).encode()
                    _ur.urlopen(_ur.Request(_url, data=_data), timeout=10)
            except Exception:
                pass
            # Re-raise unchanged so the original message + traceback survive.
            raise
        except Exception as e:
            log.error("Bloomberg pull failed: %s", e)
            # Notify immediately — before any further processing
            try:
                import os as _os, urllib.parse as _up, urllib.request as _ur
                _token = _os.environ.get("TELEGRAM_BOT_TOKEN", "")
                _chat = _os.environ.get("TELEGRAM_CHAT_ID", "979257663")
                if _token:
                    _url = f"https://api.telegram.org/bot{_token}/sendMessage"
                    from pipeline.bloomberg import BBG_HOST as _bh, BBG_PORT as _bp
                    _data = _up.urlencode({
                        "chat_id": _chat,
                        "text": f"Bloomberg not reachable at {_bh}:{_bp}, IC pipeline aborted",
                    }).encode()
                    _ur.urlopen(_ur.Request(_url, data=_data), timeout=10)
            except Exception:
                pass
            raise RuntimeError(
                f"Bloomberg is not available ({e}). "
                "Ensure Bloomberg Terminal is open and blpapi is running. "
                "To run on stale data intentionally, use --skip-bloomberg."
            )
    else:
        log.info("Skipping Bloomberg (--skip-bloomberg flag set)")
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
    # Step 2.5: Refresh mars_score sheet in ic_file.xlsx
    # -----------------------------------------------------------------------
    # pipeline.momentum.compute_scores reads pre-computed momentum from the
    # mars_score sheet. Without this refresh, scores stay frozen at whatever
    # the last momentum-repo run wrote — and weekly deltas (triangle arrows)
    # collapse to zero on every IC slide.
    excel_path = str(Path(dropbox_path) / "ic_file.xlsx")
    momentum_repo = os.environ.get(
        "IC_MOMENTUM_REPO",
        "/Users/larazanella/Desktop/GitHub/Projects/momentum",
    )
    momentum_script = Path(momentum_repo) / "run_momentum.py"
    if momentum_script.exists():
        log.info("Refreshing mars_score sheet via %s ...", momentum_script)
        import subprocess
        _mom_proc = subprocess.run(
            [
                sys.executable, str(momentum_script),
                "--prices", master_csv,
                "--output", excel_path,
                "--history-years", "1",
                "--quiet",
            ],
            cwd=momentum_repo,
            capture_output=True,
            text=True,
        )
        if _mom_proc.returncode != 0:
            log.warning(
                "Momentum scoring failed (exit %d) — falling back to stale mars_score. stderr: %s",
                _mom_proc.returncode, (_mom_proc.stderr or "")[:500],
            )
        else:
            log.info("mars_score sheet refreshed")
    else:
        log.warning(
            "Momentum script not found at %s — skipping refresh (set IC_MOMENTUM_REPO to override)",
            momentum_script,
        )

    # -----------------------------------------------------------------------
    # Step 3: Compute momentum scores
    # -----------------------------------------------------------------------
    from pipeline.momentum import compute_scores, INSTRUMENT_MAP
    instrument_names = list(INSTRUMENT_MAP.keys())
    scores_dict = compute_scores(master_df, instrument_names, target_date, excel_path=excel_path)

    # -----------------------------------------------------------------------
    # Step 4: Compute breadth (sticky weekly snapshot)
    # -----------------------------------------------------------------------
    # Once a successful run writes breadth for target_date, subsequent runs
    # for the SAME target_date reuse those records instead of recomputing.
    # This protects against Bloomberg returning slightly different values
    # mid-week (which would otherwise re-rank instruments). New target_date
    # → fresh pull.
    import json as _json
    _breadth_cache_path = Path(draft_path).parent / "breadth_cache.json"
    _target_date_str = str(target_date.date())

    def _load_sticky_cache(path: Path) -> tuple[Optional[str], Optional[list]]:
        """Return (cached_target_date, records) from a sticky cache file.
        Backward-compat: legacy format (a bare list) → (None, list)."""
        if not path.exists():
            return None, None
        try:
            with open(path, encoding="utf-8") as _f:
                _data = _json.load(_f)
        except Exception as _e:
            log.warning("Could not parse cache %s: %s", path.name, _e)
            return None, None
        if isinstance(_data, list):  # legacy
            return None, _data
        if isinstance(_data, dict):
            return _data.get("target_date"), _data.get("records") or []
        return None, None

    _cache_target, _cache_records = _load_sticky_cache(_breadth_cache_path)
    _cache_hit = _cache_target == _target_date_str and _cache_records

    from pipeline.breadth import compute_composite_breadth
    # raw_breadth is "usable" when it's non-empty AND at least one row has a
    # numeric value for the trend pillar (otherwise composite is NaN and rank
    # is None — which silently corrupts history.json + the quadrant arrows).
    _trend_cols = ["PCT_MEMB_PX_GT_50D_MOV_AVG", "PCT_MEMB_PX_GT_100D_MOV_AVG"]
    _raw_usable = (
        not raw_breadth.empty
        and all(c in raw_breadth.columns for c in _trend_cols)
        and raw_breadth[_trend_cols].notna().any(axis=None)
    )
    if _cache_hit:
        breadth_records = _cache_records
        log.info(
            "Breadth: re-using sticky cache for target_date=%s (%d records). "
            "Skipping recompute to keep ranks stable across re-runs.",
            _target_date_str, len(breadth_records),
        )
    elif _raw_usable:
        breadth_df = compute_composite_breadth(raw_breadth)
        breadth_records = breadth_df.to_dict(orient="records")
        # Persist with target_date stamp so future same-week re-runs hit cache
        try:
            atomic_write_text(
                _breadth_cache_path,
                _json.dumps(
                    {"target_date": _target_date_str, "records": breadth_records},
                    indent=2,
                ),
            )
            log.info(
                "Breadth: computed fresh for target_date=%s (%d records). Cache updated.",
                _target_date_str, len(breadth_records),
            )
        except Exception as _be:
            log.warning("Could not save breadth_cache.json: %s", _be)
    elif skip_bloomberg:
        # Dev path: --skip-bloomberg was explicitly set. Fall back to whatever
        # cache exists so layout-only runs work without a live Bloomberg.
        breadth_records = _cache_records or []
        if breadth_records:
            log.info(
                "Breadth: --skip-bloomberg + cache miss (cache target=%s, run target=%s) "
                "— using stale cache records (%d).",
                _cache_target, _target_date_str, len(breadth_records),
            )
        else:
            log.warning("Breadth: --skip-bloomberg with no usable cache; breadth slide will be empty")
    else:
        # Bloomberg was attempted but pull_breadth came back unusable — either
        # an empty DataFrame or one where every breadth field is NaN. Silently
        # falling back to breadth_cache.json poisons history.json (same ranks
        # repeated for weeks) and breaks the quadrant month-over-month arrows.
        # Hard-fail with a Telegram alert instead.
        _msg = (
            "Bloomberg pull_breadth returned no usable data "
            f"(rows={len(raw_breadth)}, all-NaN trend cols=True) — refusing "
            "to silently reuse breadth_cache.json. Likely an entitlement or "
            "timing issue with the PCT_MEMB_PX_GT_*_MOV_AVG fields. "
            "Investigate, then re-run prepare. To bypass intentionally, use "
            "--skip-bloomberg."
        )
        log.error(_msg)
        try:
            import os as _os, urllib.parse as _up, urllib.request as _ur
            _token = _os.environ.get("TELEGRAM_BOT_TOKEN", "")
            _chat = _os.environ.get("TELEGRAM_CHAT_ID", "979257663")
            if _token:
                _url = f"https://api.telegram.org/bot{_token}/sendMessage"
                _data = _up.urlencode({
                    "chat_id": _chat,
                    "text": f"IC pipeline aborted: {_msg}",
                }).encode()
                _ur.urlopen(_ur.Request(_url, data=_data), timeout=10)
        except Exception:
            pass
        raise RuntimeError(_msg)

    # -----------------------------------------------------------------------
    # Step 5: Fundamental rankings (sticky weekly snapshot)
    # -----------------------------------------------------------------------
    # Same idempotent pattern as breadth: once written for target_date,
    # subsequent same-week re-runs reuse the cached ranks + raw values
    # instead of recomputing from a fresh Bloomberg pull.
    _fund_cache_path = Path(draft_path).parent / "fundamental_cache.json"
    fundamental_df = pd.DataFrame()
    fundamental_records = {}

    _fund_cache_target = None
    _fund_cache_df_rows: list = []
    _fund_cache_raw: dict = {}
    if _fund_cache_path.exists():
        try:
            with open(_fund_cache_path, encoding="utf-8") as _ff:
                _fc = _json.load(_ff)
            if isinstance(_fc, dict):
                _fund_cache_target = _fc.get("target_date")
                _fund_cache_df_rows = _fc.get("fundamental_df") or []
                _fund_cache_raw = _fc.get("raw") or {}
        except Exception as _e:
            log.warning("Could not parse fundamental_cache.json: %s", _e)

    _fund_cache_hit = (
        _fund_cache_target == _target_date_str
        and bool(_fund_cache_df_rows)
    )

    if _fund_cache_hit:
        fundamental_df = pd.DataFrame(_fund_cache_df_rows)
        fundamental_records = dict(zip(
            fundamental_df["index_name"], fundamental_df["fundamental_rank"]
        ))
        # Rebuild raw_fundamentals (DataFrame indexed by name) so downstream
        # slide rendering + draft_state population see identical raw values.
        if _fund_cache_raw:
            raw_fundamentals = pd.DataFrame.from_dict(
                _fund_cache_raw, orient="index"
            )
        log.info(
            "Fundamentals: re-using sticky cache for target_date=%s (%d indices).",
            _target_date_str, len(fundamental_df),
        )
    elif not raw_fundamentals.empty:
        from pipeline.fundamentals import compute_fundamental_ranks as _compute_fund_ranks
        fundamental_df = _compute_fund_ranks(raw_fundamentals)
        # Build {name: rank} map for backward compat
        fundamental_records = dict(zip(
            fundamental_df["index_name"], fundamental_df["fundamental_rank"]
        ))
        # Persist a stamped cache for this target_date
        try:
            _raw_dict = {
                str(name): {
                    str(col): (None if pd.isna(v) else float(v) if isinstance(v, (int, float)) else v)
                    for col, v in raw_fundamentals.loc[name].items()
                }
                for name in raw_fundamentals.index
            }
            atomic_write_text(
                _fund_cache_path,
                _json.dumps(
                    {
                        "target_date": _target_date_str,
                        "fundamental_df": fundamental_df.to_dict(orient="records"),
                        "raw": _raw_dict,
                    },
                    indent=2,
                    default=str,
                ),
            )
            log.info(
                "Fundamentals: computed fresh for target_date=%s (%d indices). Cache updated.",
                _target_date_str, len(fundamental_df),
            )
        except Exception as _e:
            log.warning("Could not save fundamental_cache.json: %s", _e)
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
        scores_dict=scores_dict,
        target_date=target_date,
        charts_dir=charts_cache,
        master_csv=master_csv,
        excel_path=excel_path,
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
        streak_weeks = _compute_streak_weeks(hist, scores.get("rating", ""))

        enriched = _build_enriched_context(name, scores, mc, b_rank, f_rank, hist, streak_weeks)

        instruments_out[name] = {
            **scores,
            "market_cap":       mc,
            "breadth_rank":     b_rank,
            "fundamental_rank": f_rank,
            "streak_weeks":     streak_weeks,
            "enriched_context": enriched,
            "subtitle":         None,
            "chart_path":       chart_paths.get(name, ""),
        }

    # Build full fundamental data for draft_state
    fund_full = []
    if not fundamental_df.empty:
        for _, row in fundamental_df.iterrows():
            fund_full.append(row.to_dict())

    # Store raw Bloomberg values per index for history tracking
    fund_raw_by_name = {}
    if not raw_fundamentals.empty:
        for name in raw_fundamentals.index:
            fund_raw_by_name[name] = {
                col: (float(v) if pd.notna(v) else None)
                for col, v in raw_fundamentals.loc[name].items()
            }

    draft_state = {
        "date":           str(target_date.date()),
        "generated_at":   datetime.now().isoformat(),
        "instruments":    instruments_out,
        "breadth":        breadth_records,
        "fundamentals":   fundamental_records,
        "fundamental_ranks": fund_full,
        "fundamental_raw": fund_raw_by_name,
        "market_caps":    market_caps,
    }

    draft_out = Path(output_dir) / draft_path
    atomic_write_text(draft_out, json.dumps(draft_state, indent=2, default=str))

    log.info("draft_state.json saved to %s", draft_out)
    print(f"\nOK Prepare complete. Draft saved to: {draft_out}")
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
    scores_dict: dict,
    target_date: pd.Timestamp,
    charts_dir: Path,
    master_csv: str,
    excel_path: str,
) -> dict:
    """
    Render technical charts using the existing create_technical_analysis_v2_chart.

    Uses create_temp_excel_from_csv to build a temp Excel file that the chart
    functions expect, then calls create_technical_analysis_v2_chart for each
    instrument.

    Returns {instrument_name: chart_path_str}
    """
    from pipeline.momentum import INSTRUMENT_MAP
    from data_loader import create_temp_excel_from_csv
    from technical_analysis.equity.spx import create_technical_analysis_v2_chart
    from technical_analysis.common_helpers import clear_excel_cache

    clear_excel_cache()

    # Create temp Excel with data_prices sheet from CSV
    data_as_of = target_date.date()
    temp_excel = create_temp_excel_from_csv(
        Path(master_csv), Path(excel_path), data_as_of
    )
    log.info("Created temp Excel for chart generation: %s", temp_excel)

    chart_paths = {}

    for name, scores in scores_dict.items():
        if name not in INSTRUMENT_MAP:
            continue

        mars_col, _, _, bbg_ticker = INSTRUMENT_MAP[name]
        prefix = name.lower().replace(" ", "_").replace("&", "").replace("__", "_")
        # Use the standard prefix naming
        prefix_map = {
            "S&P 500": "spx", "CSI 300": "csi", "Nikkei 225": "nikkei",
            "TASI": "tasi", "Sensex": "sensex", "DAX": "dax", "SMI": "smi",
            "IBOV": "ibov", "MEXBOL": "mexbol", "Gold": "gold", "Silver": "silver",
            "Platinum": "platinum", "Palladium": "palladium", "Oil": "oil",
            "Copper": "copper", "Bitcoin": "bitcoin", "Ethereum": "ethereum",
            "Ripple": "ripple", "Solana": "solana", "Binance": "binance",
        }
        prefix = prefix_map.get(name, name.lower())
        chart_file = charts_dir / f"{prefix}_chart.png"

        try:
            chart_bytes, used_date = create_technical_analysis_v2_chart(
                str(temp_excel),
                ticker=bbg_ticker,
                price_mode="Last Price",
                dmas_score=scores.get("dmas"),
                dmas_prev_week=None,
                technical_score=scores.get("technical"),
                momentum_score=scores.get("momentum"),
            )

            if chart_bytes:
                with open(chart_file, "wb") as f:
                    f.write(chart_bytes)
                chart_paths[name] = str(chart_file)
                log.info("Chart saved: %s", chart_file)
            else:
                log.warning("No chart bytes returned for %s", name)
                chart_paths[name] = ""

        except Exception as e:
            log.warning("Chart generation failed for %s: %s", name, e)
            chart_paths[name] = ""

    # Clean up temp Excel
    try:
        import os
        os.remove(temp_excel)
    except Exception:
        pass

    return chart_paths
