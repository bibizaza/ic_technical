"""Rates performance dashboard generation with price‑mode awareness and source footnotes.

This module produces charts summarising recent yield movements for a selection
of government bond rates.  Unlike price‑based instruments, yield increases
correspond to price declines (and vice versa).  Therefore, positive changes
in yields are displayed as negative performance (red bars/cells) and
negative changes as positive performance (green).  Users can choose
to base calculations on either the most recent intraday yield ("Last
Price") or the previous day's closing yield ("Last Close").  The
resulting figures include a weekly change bar chart and a heatmap of
longer horizons (1M, 3M, 6M, 12M and YTD).  When the charts are
inserted into a PowerPoint presentation, a source footnote is added to
the designated text boxes on each slide, reflecting the effective
date used in the computations and the chosen price mode.  Text
formatting from the placeholders is preserved.

Key functions
-------------

* ``create_weekly_performance_chart`` – Build the weekly bar chart and
  return both the image bytes and the effective date used for
  computation.
* ``create_historical_performance_table`` – Build the heatmap of
  yield changes for multiple horizons and return the image bytes and
  the effective date.
* ``insert_rates_performance_bar_slide`` – Insert the weekly bar
  chart and source footnote into its slide.
* ``insert_rates_performance_histo_slide`` – Insert the historical
  performance heatmap and source footnote into its slide.

Usage example::

    from performance.rates_perf import (
        create_weekly_performance_chart,
        create_historical_performance_table,
        insert_rates_performance_bar_slide,
        insert_rates_performance_histo_slide,
    )

    bar_bytes, used_date = create_weekly_performance_chart(excel_path, price_mode="Last Close")
    histo_bytes, _ = create_historical_performance_table(excel_path, price_mode="Last Close")
    prs = insert_rates_performance_bar_slide(prs, bar_bytes, used_date, price_mode)
    prs = insert_rates_performance_histo_slide(prs, histo_bytes, used_date, price_mode)

"""

from __future__ import annotations

import io
import pathlib
from typing import Dict, List, Tuple, Union, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.util import Cm

from utils import adjust_prices_for_mode

try:
    # Reuse font attribute helpers from the SPX module if available
    from technical_analysis.equity.spx import (
        _get_run_font_attributes as _capture_font_attrs,  # type: ignore
        _apply_run_font_attributes as _apply_font_attrs,  # type: ignore
    )
except Exception:
    # Fallback helpers: only support basic size and RGB; ignore theme colours
    def _capture_font_attrs(run):  # type: ignore
        if run is None:
            return (None, None, None, None, None, None)
        size = run.font.size
        colour = run.font.color
        rgb = None
        try:
            rgb = colour.rgb
        except Exception:
            rgb = None
        return (size, rgb, None, None, run.font.bold, run.font.italic)

    def _apply_font_attrs(new_run, size, rgb, theme_color, brightness, bold, italic):  # type: ignore
        if size is not None:
            new_run.font.size = size
        if rgb is not None:
            try:
                new_run.font.color.rgb = rgb
            except Exception:
                pass
        if bold is not None:
            new_run.font.bold = bold
        if italic is not None:
            new_run.font.italic = italic

###############################################################################
# Data loading and return calculations
###############################################################################

def _load_price_data(excel_path: Union[str, pathlib.Path], tickers: List[str]) -> pd.DataFrame:
    """Load a DataFrame of dates and yields for the specified rate tickers.

    Parameters
    ----------
    excel_path : str or pathlib.Path
        Path to the Excel workbook containing a sheet named ``data_prices``.
    tickers : list of str
        Column names in the sheet corresponding to the desired rate indices.

    Returns
    -------
    DataFrame
        A tidy DataFrame with columns ``Date`` and one column per ticker,
        sorted by date and with numeric yields.
    """
    df = pd.read_excel(excel_path, sheet_name="data_prices")
    df = df.drop(index=0)
    df = df[df[df.columns[0]] != "DATES"].copy()
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    out = df[["Date"] + tickers].copy()
    for t in tickers:
        out[t] = pd.to_numeric(out[t], errors="coerce")
    return out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def _compute_horizon_returns(
    df: pd.DataFrame,
    ticker: str,
    today: pd.Timestamp,
    days: int,
) -> float:
    """Compute the change in yield over the given lookback, expressed in basis points.

    Unlike price series, yield changes are computed as simple differences
    rather than percentage changes.  For example, a move from 1.00 %
    to 1.25 % over a one‑week horizon is reported as +25 bps (positive
    values correspond to rising yields and thus negative price action).

    Parameters
    ----------
    df : DataFrame
        A tidy DataFrame with ``Date`` and yield columns.
    ticker : str
        Column name for which to compute the return.
    today : Timestamp
        Anchor date; typically the last date in the dataset.
    days : int
        Lookback period in days.

    Returns
    -------
    float
        Yield change over the lookback horizon in basis points (bps), or
        NaN if not computable.
    """
    past_date = today - pd.Timedelta(days=days)
    past_series = df.loc[df["Date"] <= past_date, ticker]
    if len(past_series) == 0:
        return float("nan")
    past_yield = past_series.iloc[-1]
    current_yield = df[ticker].iloc[-1]
    if pd.isna(past_yield) or pd.isna(current_yield):
        return float("nan")
    # The raw yield values in the Excel file are expressed in percentage points
    # (e.g. 1.941485 represents 1.941485%).  A one‑basis‑point move equals
    # 0.01 percentage points.  Therefore, to convert the difference between
    # two percentage yields into basis points, multiply by 100.  This yields
    # e.g. (1.94 – 1.84) * 100 ≈ 9.7 bps.  Do not multiply by 10,000, which
    # would treat the inputs as decimals rather than percentages.
    return (current_yield - past_yield) * 100.0  # convert to basis points


def _build_returns_table(
    df: pd.DataFrame,
    mapping: Dict[str, str],
) -> pd.DataFrame:
    """Calculate yield changes (in bps) for each ticker across multiple horizons.

    Returns a DataFrame indexed by the ticker code with the following
    columns: ``Name``, ``1W``, ``1M``, ``3M``, ``6M``, ``12M`` and ``YTD``.

    Parameters
    ----------
    df : DataFrame
        A tidy DataFrame as returned by ``_load_price_data``.
    mapping : dict
        Mapping from ticker codes to human‑friendly names.

    Returns
    -------
    DataFrame
        Table of yield changes in basis points.
    """
    today = df["Date"].max()
    results: Dict[str, Dict[str, float]] = {}
    for ticker in mapping:
        results[ticker] = {
            "1W": _compute_horizon_returns(df, ticker, today, 7),
            "1M": _compute_horizon_returns(df, ticker, today, 30),
            "3M": _compute_horizon_returns(df, ticker, today, 90),
            "6M": _compute_horizon_returns(df, ticker, today, 180),
            "12M": _compute_horizon_returns(df, ticker, today, 365),
        }
        # Year‑to‑date: yield change since the start of the calendar year
        start_of_year = pd.Timestamp(year=today.year, month=1, day=1)
        past_series = df.loc[df["Date"] <= start_of_year, ticker]
        if len(past_series) > 0:
            past_yield = past_series.iloc[-1]
            current_yield = df[ticker].iloc[-1]
            if pd.notna(past_yield) and pd.notna(current_yield):
                # Convert the yield difference (expressed in percentage points) to bps.
                # See _compute_horizon_returns for rationale: multiply by 100.
                results[ticker]["YTD"] = (current_yield - past_yield) * 100.0
            else:
                results[ticker]["YTD"] = float("nan")
        else:
            results[ticker]["YTD"] = float("nan")
    table = pd.DataFrame.from_dict(results, orient="index")
    table["Name"] = table.index.map(mapping.get)
    return table[["Name", "1W", "1M", "3M", "6M", "12M", "YTD"]]


###############################################################################
# Figure generation functions
###############################################################################

def create_weekly_performance_chart(
    excel_path: Union[str, pathlib.Path],
    ticker_mapping: Dict[str, str] | None = None,
    *,
    width: float = 14.0,
    height: float = 5.0,
    price_mode: str = "Last Price",
) -> Tuple[bytes, Optional[pd.Timestamp]]:
    """Generate a bar chart of 1‑week yield changes with price‑mode adjustment.

    This helper reads the specified Excel file, optionally adjusts the yield
    history according to the selected ``price_mode`` (``"Last Price"`` vs
    ``"Last Close"``), computes 1‑week changes for the configured rate
    tickers, sorts them in descending order (most positive yield increase
    at the top) and constructs a horizontal bar chart.  A positive
    change (rising yields) is depicted in red; a negative change (falling
    yields) is depicted in green.  The effective date used for the
    calculations is returned along with the PNG data.

    Parameters
    ----------
    excel_path : str or pathlib.Path
        Path to the Excel workbook containing the yield series.
    ticker_mapping : dict, optional
        Mapping from ticker codes to display names.  If not provided,
        sensible defaults are used.
    width, height : float
        Dimensions of the generated figure in inches.
    price_mode : str, default ``"Last Price"``
        One of ``"Last Price"`` or ``"Last Close"``.  Determines whether
        intraday yields are used or the most recent row is dropped if it
        corresponds to today's date.

    Returns
    -------
    tuple
        A two‑tuple ``(image_bytes, used_date)`` where ``image_bytes`` is
        the PNG data for the bar chart and ``used_date`` is the effective
        date in the adjusted yield series.
    """
    default_mapping = {
        "USGG2YR Index": "US - 2Y",
        "USGG10YR Index": "US - 10Y",
        "USGG30YR Index": "US - 30Y",
        "GECU2YR Index": "EUR - 2Y",
        "GECU10YR Index": "EUR - 10Y",
        "GECU30YR Index": "EUR - 30Y",
        "GCNY2YR Index": "CN - 2Y",
        "GCNY10YR Index": "CN - 10Y",
        "GCNY30YR Index": "CN - 30Y",
        "GJGB2 Index": "JP - 2Y",
        "GJGB10 Index": "JP - 10Y",
        "GJGB30 Index": "JP - 30Y",
    }
    mapping = ticker_mapping or default_mapping
    tickers = list(mapping.keys())
    df = _load_price_data(excel_path, tickers)
    df_adj, used_date = adjust_prices_for_mode(df, price_mode)
    perf = _build_returns_table(df_adj, mapping)
    bar_df = perf[["Name", "1W"]].dropna().sort_values("1W", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(width, height))
    # Colour mapping: positive yield change → red, negative → green
    bar_colors = ["#C70039" if x > 0 else "#2E8B57" for x in bar_df["1W"]]
    bars = ax.barh(bar_df["Name"], bar_df["1W"], color=bar_colors)
    # Add labels in bps with sign
    # Determine a small offset for label placement relative to bar length.  A fixed
    # offset (e.g. ±5) was causing labels to drift far from the bar for small
    # values.  Use a modest constant (e.g. 0.2) to keep the label close to the
    # bar end, with the sign matching the direction of the bar.
    label_offset = 0.2
    for bar in bars:
        width_val = bar.get_width()
        x_pos = width_val + (label_offset if width_val > 0 else -label_offset)
        # Display one decimal place for bps to capture partial basis points (e.g. +9.7 bps)
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2.0,
            f"{width_val:+.1f} bps",
            va="center",
            ha="left" if width_val > 0 else "right",
            fontweight="bold",
            color="black",
        )
    ax.axvline(0.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xticks([])
    ax.set_yticks(range(len(bar_df)))
    ax.set_yticklabels(bar_df["Name"], fontsize=10)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0)
    # Provide extra space; values in bps may be large
    min_val = min(bar_df["1W"].min(), 0)
    max_val = bar_df["1W"].max()
    margin = (max_val - min_val) * 0.2
    ax.set_xlim(min_val - margin, max_val + margin)
    fig.subplots_adjust(left=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue(), used_date


def create_historical_performance_table(
    excel_path: Union[str, pathlib.Path],
    ticker_mapping: Dict[str, str] | None = None,
    *,
    width: float = 14.0,
    height: float = 6.0,
    price_mode: str = "Last Price",
) -> Tuple[bytes, Optional[pd.Timestamp]]:
    """Generate a heatmap table of yield changes with price‑mode adjustment.

    This helper reads the yield data, optionally adjusts it according to
    ``price_mode``, computes changes for multiple horizons, sorts the
    table by YTD change (descending) and constructs a heatmap where
    positive values map to red and negative values to green.  The
    effective date used for the computations is returned along with the
    PNG data.

    Parameters
    ----------
    excel_path : str or pathlib.Path
        Path to the Excel workbook containing the yield series.
    ticker_mapping : dict, optional
        Mapping from ticker codes to display names.  If not provided,
        sensible defaults are used.
    width, height : float
        Dimensions of the generated figure in inches.
    price_mode : str, default ``"Last Price"``
        Either ``"Last Price"`` or ``"Last Close"``.  Determines how
        yield data is adjusted prior to computing changes and which
        effective date appears in the source footnote.

    Returns
    -------
    tuple
        A two‑tuple ``(image_bytes, used_date)`` where ``image_bytes`` is
        the PNG data for the heatmap and ``used_date`` is the effective
        date in the adjusted yield series.
    """
    default_mapping = {
        "USGG2YR Index": "US - 2Y",
        "USGG10YR Index": "US - 10Y",
        "USGG30YR Index": "US - 30Y",
        "GECU2YR Index": "EUR - 2Y",
        "GECU10YR Index": "EUR - 10Y",
        "GECU30YR Index": "EUR - 30Y",
        "GCNY2YR Index": "CN - 2Y",
        "GCNY10YR Index": "CN - 10Y",
        "GCNY30YR Index": "CN - 30Y",
        "GJGB2 Index": "JP - 2Y",
        "GJGB10 Index": "JP - 10Y",
        "GJGB30 Index": "JP - 30Y",
    }
    mapping = ticker_mapping or default_mapping
    tickers = list(mapping.keys())
    df = _load_price_data(excel_path, tickers)
    df_adj, used_date = adjust_prices_for_mode(df, price_mode)
    perf = _build_returns_table(df_adj, mapping)
    heat_df = perf[["Name", "YTD", "1M", "3M", "6M", "12M"]].dropna().sort_values("YTD", ascending=False).reset_index(drop=True)
    n_rows = len(heat_df)
    cols = ["YTD", "1M", "3M", "6M", "12M"]
    # Build colour array: positive values map to red, negative to green
    color_array = np.zeros((n_rows, len(cols), 4))
    for j, col in enumerate(cols):
        col_vals = heat_df[col].astype(float).values
        pos_vals = col_vals[col_vals > 0]
        neg_vals = col_vals[col_vals < 0]
        max_pos = pos_vals.max() if len(pos_vals) > 0 else 0.0
        min_neg = neg_vals.min() if len(neg_vals) > 0 else 0.0
        for i, val in enumerate(col_vals):
            color_array[i, j] = _colour_for_value_rates(float(val), max_pos, min_neg)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.imshow(color_array, aspect="auto")
    # Add text: show change in bps with sign
    for i in range(n_rows):
        for j, col in enumerate(cols):
            val = heat_df.iloc[i][col]
            ax.text(
                j,
                i,
                f"{val:+.1f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="black",
            )
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=12, fontweight="bold")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(heat_df["Name"], fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue(), used_date


###############################################################################
# Colour helper for rates (invert green/red)
###############################################################################

def _colour_for_value_rates(val: float, max_pos: float, min_neg: float) -> Tuple[float, float, float, float]:
    """Return an RGBA colour for a single yield change value.

    Positive changes (rising yields) produce a gradient from white to red;
    negative changes (falling yields) produce a gradient from white to green.
    """
    white = np.array([1.0, 1.0, 1.0, 1.0])
    red = np.array(mcolors.to_rgba("#C70039"))
    green = np.array(mcolors.to_rgba("#2E8B57"))
    if val > 0 and max_pos > 0:
        ratio = float(val) / max_pos
        return tuple(white + ratio * (red - white))
    if val < 0 and min_neg < 0:
        ratio = float(val) / min_neg  # both negative → positive ratio
        return tuple(white + ratio * (green - white))
    return tuple(white)


###############################################################################
# Slide insertion helpers
###############################################################################

def _insert_dashboard_to_placeholder(
    prs: Presentation,
    image_bytes: bytes,
    placeholder_names: List[str],
    *,
    left_cm: float,
    top_cm: float,
    width_cm: float,
    height_cm: float,
    used_date: Optional[pd.Timestamp],
    price_mode: str,
    source_placeholder_names: List[str],
) -> Presentation:
    # Reuse the helper from other modules but local to this file for clarity.
    target_slide = None
    placeholder_box = None
    name_candidates = [n.lower() for n in placeholder_names]
    pattern_candidates = [f"[{n}]" for n in name_candidates]
    for slide in prs.slides:
        for shape in slide.shapes:
            name_attr = getattr(shape, "name", "").lower()
            if name_attr in name_candidates:
                target_slide = slide
                placeholder_box = shape
                break
            if shape.has_text_frame:
                text_lower = (shape.text or "").strip().lower()
                if text_lower in [p.lower() for p in pattern_candidates]:
                    target_slide = slide
                    placeholder_box = shape
                    break
        if target_slide:
            break
    if target_slide is None:
        target_slide = prs.slides[min(11, len(prs.slides) - 1)]
    left = Cm(left_cm)
    top = Cm(top_cm)
    width = Cm(width_cm)
    height = Cm(height_cm)
    stream = io.BytesIO(image_bytes)
    target_slide.shapes.add_picture(stream, left, top, width=width, height=height)
    if used_date is not None:
        date_str = used_date.strftime("%d/%m/%Y")
        suffix = " Close" if price_mode.lower() == "last close" else ""
        source_text = f"Source: Bloomberg, Herculis Group, Data as of {date_str}{suffix}"
        source_candidates = [n.lower() for n in source_placeholder_names]
        source_patterns = [f"[{n}]" for n in source_candidates]
        for shape in target_slide.shapes:
            name_attr2 = getattr(shape, "name", "").lower()
            if name_attr2 in source_candidates:
                if shape.has_text_frame:
                    runs = shape.text_frame.paragraphs[0].runs
                    attrs = _capture_font_attrs(runs[0]) if runs else (None, None, None, None, None, None)
                    shape.text_frame.clear()
                    p = shape.text_frame.paragraphs[0]
                    new_run = p.add_run()
                    new_run.text = source_text
                    _apply_font_attrs(new_run, *attrs)
                break
            if shape.has_text_frame:
                for pattern in source_patterns:
                    if pattern.lower() in (shape.text or "").lower():
                        runs = shape.text_frame.paragraphs[0].runs
                        attrs = _capture_font_attrs(runs[0]) if runs else (None, None, None, None, None, None)
                        try:
                            new_text = shape.text.replace(pattern, source_text)
                        except Exception:
                            new_text = source_text
                        shape.text_frame.clear()
                        p = shape.text_frame.paragraphs[0]
                        new_run = p.add_run()
                        new_run.text = new_text
                        _apply_font_attrs(new_run, *attrs)
                        break
                else:
                    continue
                break
    return prs


def insert_rates_performance_bar_slide(
    prs: Presentation,
    image_bytes: bytes,
    used_date: Optional[pd.Timestamp] = None,
    price_mode: str = "Last Price",
    *,
    left_cm: float = 0.0,
    top_cm: float = 3.22,
    width_cm: float = 25.0,
    height_cm: float = 11.66,
) -> Presentation:
    """Insert the weekly rates bar chart and source footnote into its designated slide.

    The slide is identified by a shape named ``rates_perf_1week`` or
    ``rates_perf_1w`` (or containing ``[rates_perf_1week]``).  The chart
    is inserted at the specified coordinates and a source footnote is
    written into ``rates_1w_source`` or a shape containing ``[rates_1w_source]``.
    """
    return _insert_dashboard_to_placeholder(
        prs,
        image_bytes,
        placeholder_names=["rates_perf_1w", "rates_perf_1week"],
        left_cm=left_cm,
        top_cm=top_cm,
        width_cm=width_cm,
        height_cm=height_cm,
        used_date=used_date,
        price_mode=price_mode,
        source_placeholder_names=["rates_1w_source"],
    )


def insert_rates_performance_histo_slide(
    prs: Presentation,
    image_bytes: bytes,
    used_date: Optional[pd.Timestamp] = None,
    price_mode: str = "Last Price",
    *,
    left_cm: float = 0.0,
    top_cm: float = 3.22,
    width_cm: float = 25.0,
    height_cm: float = 11.66,
) -> Presentation:
    """Insert the rates historical performance heatmap and source footnote into its slide.

    The slide is identified by a shape named ``rates_perf_histo`` (or
    containing ``[rates_perf_histo]``).  A source footnote is written
    into ``rates_1w_source2`` or a shape containing ``[rates_1w_source2]``.
    """
    return _insert_dashboard_to_placeholder(
        prs,
        image_bytes,
        placeholder_names=["rates_perf_histo"],
        left_cm=left_cm,
        top_cm=top_cm,
        width_cm=width_cm,
        height_cm=height_cm,
        used_date=used_date,
        price_mode=price_mode,
        source_placeholder_names=["rates_1w_source2"],
    )
