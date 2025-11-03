"""
Pre-warm all instrument charts in parallel for Phase 2 optimization.

This module generates all chart images concurrently before PowerPoint insertion,
dramatically reducing total generation time on multi-core systems.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, Optional
import traceback


def prewarm_all_instrument_charts(
    excel_path: Path,
    config: Dict[str, Any],
    max_workers: int = 8,
    progress_callback: Optional[callable] = None
) -> Dict[str, bool]:
    """
    Pre-generate all instrument charts in parallel.

    This function generates all main charts and gauges for all instruments
    concurrently, storing them in the image cache. Subsequent calls to
    insert_XXX_technical_chart_with_callout() will use the cached images.

    Parameters
    ----------
    excel_path : Path
        Path to the Excel data file
    config : Dict
        Configuration containing:
        - All anchor dates (spx_anchor_dt, csi_anchor_dt, etc.)
        - Price mode
        - Last week averages for gauges
    max_workers : int, default 8
        Number of parallel workers (8 is good for M4 Max)
    progress_callback : callable, optional
        Function to call: callback(completed, total, instrument_name)

    Returns
    -------
    Dict[str, bool]
        Success status for each chart
    """
    from technical_analysis.common_helpers import (
        _load_price_data_generic,
        generate_range_callout_chart_image,
        generate_average_gauge_image,
        _get_technical_score_generic,
        _get_momentum_score_generic,
    )

    # Price mode
    pmode = config.get('price_mode', 'Last Price')

    # Build list of all chart generation tasks
    tasks = []

    # Helper to load vol index
    def get_vol_value(ticker):
        try:
            import pandas as pd
            df_vol = pd.read_excel(excel_path, sheet_name="data_prices")
            df_vol = df_vol.drop(index=0)
            if ticker in df_vol.columns:
                return float(df_vol[ticker].iloc[-1])
        except:
            pass
        return None

    # ========== EQUITY INSTRUMENTS ==========
    equity_instruments = [
        ("SPX", "SPX Index", config.get('spx_anchor_dt'), config.get('spx_last_week_avg', 50), "VIX Index"),
        ("CSI", "SHSZ300 Index", config.get('csi_anchor_dt'), config.get('csi_last_week_avg', 50), None),
        ("Nikkei", "NKY Index", config.get('nikkei_anchor_dt'), config.get('nikkei_last_week_avg', 50), None),
        ("TASI", "SASEIDX Index", config.get('tasi_anchor_dt'), config.get('tasi_last_week_avg', 50), None),
        ("Sensex", "SENSEX Index", config.get('sensex_anchor_dt'), config.get('sensex_last_week_avg', 50), None),
        ("DAX", "DAX Index", config.get('dax_anchor_dt'), config.get('dax_last_week_avg', 50), None),
        ("SMI", "SMI Index", config.get('smi_anchor_dt'), config.get('smi_last_week_avg', 50), None),
        ("IBOV", "IBOV Index", config.get('ibov_anchor_dt'), config.get('ibov_last_week_avg', 50), None),
        ("Mexbol", "MEXBOL Index", config.get('mexbol_anchor_dt'), config.get('mexbol_last_week_avg', 50), None),
    ]

    for name, ticker, anchor, last_week, vol_ticker in equity_instruments:
        tasks.append({
            'name': f"{name}_main_chart",
            'type': 'callout_chart',
            'ticker': ticker,
            'anchor': anchor,
            'vol_ticker': vol_ticker,
            'cache_key': f"{name.lower()}_main_callout",
        })
        tasks.append({
            'name': f"{name}_gauge",
            'type': 'average_gauge',
            'ticker': ticker,
            'last_week_avg': last_week,
            'cache_key': f"{name.lower()}_avg_gauge",
        })

    # ========== COMMODITIES ==========
    commodities = [
        ("Gold", "GCA Comdty", config.get('gold_anchor_dt'), config.get('gold_last_week_avg', 50)),
        ("Silver", "SI1 Comdty", config.get('silver_anchor_dt'), config.get('silver_last_week_avg', 50)),
        ("Platinum", "PL1 Comdty", config.get('platinum_anchor_dt'), config.get('platinum_last_week_avg', 50)),
        ("Palladium", "PA1 Comdty", config.get('palladium_anchor_dt'), config.get('palladium_last_week_avg', 50)),
        ("Oil", "CL1 Comdty", config.get('oil_anchor_dt'), config.get('oil_last_week_avg', 50)),
        ("Copper", "HG1 Comdty", config.get('copper_anchor_dt'), config.get('copper_last_week_avg', 50)),
    ]

    for name, ticker, anchor, last_week in commodities:
        tasks.append({
            'name': f"{name}_main_chart",
            'type': 'callout_chart',
            'ticker': ticker,
            'anchor': anchor,
            'vol_ticker': None,
            'cache_key': f"{name.lower()}_main_callout",
        })
        tasks.append({
            'name': f"{name}_gauge",
            'type': 'average_gauge',
            'ticker': ticker,
            'last_week_avg': last_week,
            'cache_key': f"{name.lower()}_avg_gauge",
        })

    # ========== CRYPTO ==========
    crypto = [
        ("Bitcoin", "XBTUSD Curncy", config.get('bitcoin_anchor_dt'), config.get('bitcoin_last_week_avg', 50)),
        ("Ethereum", "ETHUSD Curncy", config.get('ethereum_anchor_dt'), config.get('ethereum_last_week_avg', 50)),
        ("Ripple", "XRPUSD Curncy", config.get('ripple_anchor_dt'), config.get('ripple_last_week_avg', 50)),
        ("Solana", "SOLUSD Curncy", config.get('solana_anchor_dt'), config.get('solana_last_week_avg', 50)),
        ("Binance", "BNBUSD Curncy", config.get('binance_anchor_dt'), config.get('binance_last_week_avg', 50)),
    ]

    for name, ticker, anchor, last_week in crypto:
        tasks.append({
            'name': f"{name}_main_chart",
            'type': 'callout_chart',
            'ticker': ticker,
            'anchor': anchor,
            'vol_ticker': None,
            'cache_key': f"{name.lower()}_main_callout",
        })
        tasks.append({
            'name': f"{name}_gauge",
            'type': 'average_gauge',
            'ticker': ticker,
            'last_week_avg': last_week,
            'cache_key': f"{name.lower()}_avg_gauge",
        })

    total_tasks = len(tasks)
    completed = 0
    results = {}

    print(f"🚀 Pre-generating {total_tasks} charts in parallel with {max_workers} workers...")

    def generate_single_chart(task):
        """Generate a single chart based on task specification."""
        try:
            if task['type'] == 'callout_chart':
                # Load price data
                df = _load_price_data_generic(excel_path, task['ticker'], pmode)

                # Get volatility index if specified
                vol_val = None
                if task.get('vol_ticker'):
                    vol_val = get_vol_value(task['vol_ticker'])

                # Generate chart with cache key
                img_bytes = generate_range_callout_chart_image(
                    df,
                    anchor_date=task['anchor'],
                    lookback_days=90,
                    width_cm=24.2,
                    height_cm=6.52,
                    vol_index_value=vol_val,
                    show_legend=False,
                    cache_key=task['cache_key']
                )
                return (task['name'], True, len(img_bytes) if img_bytes else 0)

            elif task['type'] == 'average_gauge':
                # Get scores
                tech_score = _get_technical_score_generic(excel_path, task['ticker']) or 50.0
                mom_score = _get_momentum_score_generic(excel_path, task['ticker']) or 50.0

                # Generate gauge with cache key
                gauge_bytes = generate_average_gauge_image(
                    tech_score,
                    mom_score,
                    task['last_week_avg'],
                    cache_key=task['cache_key']
                )
                return (task['name'], True, len(gauge_bytes) if gauge_bytes else 0)

        except Exception as e:
            error_msg = f"Failed {task['name']}: {str(e)}"
            print(f"  ⚠️  {error_msg}")
            traceback.print_exc()
            return (task['name'], False, str(e))

    # Execute all tasks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(generate_single_chart, task): task for task in tasks}

        # Collect results as they complete
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                name, success, result = future.result()
                results[name] = success
                completed += 1

                if progress_callback:
                    progress_callback(completed, total_tasks, name)
                else:
                    status = "✅" if success else "❌"
                    print(f"  [{completed}/{total_tasks}] {status} {name}")

            except Exception as e:
                results[task['name']] = False
                completed += 1
                print(f"  [{completed}/{total_tasks}] ❌ {task['name']}: {str(e)}")

    successful = sum(1 for v in results.values() if v)
    print(f"\n✅ Pre-generation complete: {successful}/{total_tasks} charts generated")

    return results
