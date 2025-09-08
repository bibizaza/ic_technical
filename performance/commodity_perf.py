"""Commodity performance dashboard generation with price‑mode awareness and source footnotes.

This module produces charts summarising recent performance for a selection
of commodity indices grouped into Energy, Green Transition and Precious Metals.
Users can choose to base calculations on either the most recent intraday
price ("Last Price") or the previous day's closing price ("Last Close").
The resulting figures include a weekly returns bar chart and a heatmap
of longer horizons.  When the charts are inserted into a PowerPoint
presentation, a source footnote is added to the designated text boxes
on each slide, reflecting the effective date used in the computations
and the chosen price mode.  Text formatting from the placeholders
is preserved.

The commodities are grouped into three categories:

* **Energy:** Oil (WTI), Oil (Brent), Natural Gas (TTF)
* **Green Transition:** Uranium, Lithium, Manganese, Copper, Cobalt
* **Precious Metals:** Gold, Silver, Platinum

Key functions
-------------

* ``create_weekly_performance_chart`` – Build the weekly bar chart and
  return both the image bytes and the effective date used for
  computation.  Bars are ordered within each category and separated
  visually by inserting a small gap between categories.
* ``create_historical_performance_table`` – Build the heatmap of
  returns for multiple horizons and return the image bytes and the
  effective date.
* ``insert_commodity_performance_bar_slide`` – Insert the weekly bar
  chart and source footnote into its slide.
* ``insert_commodity_performance_histo_slide`` – Insert the historical
  performance heatmap and source footnote into its slide.
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
    """Load a DataFrame of dates and prices for the specified commodity tickers."""
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
    """Compute the percentage return of a series over the given lookback."""
    past_date = today - pd.Timedelta(days=days)
    past_series = df.loc[df["Date"] <= past_date, ticker]
    if len(past_series) == 0:
        return float("nan")
    past_price = past_series.iloc[-1]
    current_price = df[ticker].iloc[-1]
    if past_price == 0 or pd.isna(past_price):
        return float("nan")
    return (current_price - past_price) / past_price * 100.0


def _build_returns_table(
    df: pd.DataFrame,
    mapping: Dict[str, str],
) -> pd.DataFrame:
    """Calculate performance metrics for each commodity ticker."""
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
        start_of_year = pd.Timestamp(year=today.year, month=1, day=1)
        past_series = df.loc[df["Date"] <= start_of_year, ticker]
        if len(past_series) > 0:
            past_price = past_series.iloc[-1]
            current_price = df[ticker].iloc[-1]
            if past_price and not pd.isna(past_price):
                results[ticker]["YTD"] = (current_price - past_price) / past_price * 100.0
            else:
                results[ticker]["YTD"] = float("nan")
        else:
            results[ticker]["YTD"] = float("nan")
    table = pd.DataFrame.from_dict(results, orient="index")
    table["Name"] = table.index.map(mapping.get)
    return table[["Name", "1W", "1M", "3M", "6M", "12M", "YTD"]]


###############################################################################
# Colour helper (green/red)
###############################################################################

def _colour_for_value(val: float, max_pos: float, min_neg: float) -> Tuple[float, float, float, float]:
    """Return an RGBA colour for a single value using independent scales."""
    white = np.array([1.0, 1.0, 1.0, 1.0])
    red = np.array(mcolors.to_rgba("#C70039"))
    green = np.array(mcolors.to_rgba("#2E8B57"))
    if val > 0 and max_pos > 0:
        ratio = float(val) / max_pos
        return tuple(white + ratio * (green - white))
    if val < 0 and min_neg < 0:
        ratio = float(val) / min_neg  # both negative → positive ratio
        return tuple(white + ratio * (red - white))
    return tuple(white)


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
    """Generate a bar chart of 1‑week returns grouped by commodity category."""
    default_mapping = {
        # Energy
        "CL1 Comdty": "Oil (WTI)",
        "CO1 Comdty": "Oil (Brent)",
        "TTFG1MON OECM Index": "Natural Gas (TTF)",
        # Green Transition
        "UXA1 Comdty": "Uranium",
        "LMY1 Comdty": "Lithium",
        "IMRU5 Comdty": "Manganese",
        "LP1 Comdty": "Copper",
        "CVT1 Comdty": "Cobalt",
        # Precious Metals
        "GCA Comdty": "Gold",
        "SIA Comdty": "Silver",
        "XPT Comdty": "Platinum",
        "XPD Curncy": "Palladium",
    }
    mapping = ticker_mapping or default_mapping
    tickers = list(mapping.keys())
    df = _load_price_data(excel_path, tickers)
    df_adj, used_date = adjust_prices_for_mode(df, price_mode)
    today = df_adj["Date"].max()
    # Compute 1W returns directly for each ticker
    returns_dict = {}
    for t in tickers:
        returns_dict[t] = _compute_horizon_returns(df_adj, t, today, 7)
    # Define category order and mapping ticker → category
    category_map = {
        "CL1 Comdty": "Energy",
        "CO1 Comdty": "Energy",
        "TTFG1MON OECM Index": "Energy",
        "UXA1 Comdty": "Green Transition",
        "LMY1 Comdty": "Green Transition",
        "IMRU5 Comdty": "Green Transition",
        "LP1 Comdty": "Green Transition",
        "CVT1 Comdty": "Green Transition",
        "GCA Comdty": "Precious Metals",
        "SIA Comdty": "Precious Metals",
        "XPT Comdty": "Precious Metals",
        "XPD Curncy": "Precious Metals",
    }
    categories = ["Energy", "Green Transition", "Precious Metals"]
    # Build a list of (category, human name, return) and sort within categories by return descending
    perf_list: List[Tuple[str, str, float]] = []
    for t in tickers:
        cat = category_map.get(t, "Other")
        perf_list.append((cat, mapping[t], returns_dict.get(t, float("nan"))))
    # Sort by category order then return descending (NaNs last)
    perf_list.sort(key=lambda x: (categories.index(x[0]) if x[0] in categories else 99, float("inf") if pd.isna(x[2]) else -x[2]))
    names = [item[1] for item in perf_list]
    returns = [item[2] for item in perf_list]
    bar_df = pd.DataFrame({"Name": names, "Return": returns})
    fig, ax = plt.subplots(figsize=(width, height))
    # Colour bars based on sign
    bar_colors = ["#2E8B57" if x > 0 else "#C70039" for x in bar_df["Return"]]
    bars = ax.barh(bar_df["Name"], bar_df["Return"], color=bar_colors)
    # Compute dynamic offset for labels
    min_val = min(bar_df["Return"].min(), 0)
    max_val = bar_df["Return"].max()
    range_val = max_val - min_val if max_val > min_val else abs(max_val)
    label_offset = range_val * 0.05
    for bar, val in zip(bars, bar_df["Return"]):
        width_val = bar.get_width()
        x_pos = width_val + (label_offset if width_val > 0 else -label_offset)
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:+.1f}%",
            va="center",
            ha="left" if width_val > 0 else "right",
            fontweight="bold",
            color="black",
        )
    # Add category separators: draw horizontal lines between categories
    # Determine index boundaries for categories
    cat_indices: List[int] = []
    idx_count = 0
    for cat in categories:
        count_in_cat = sum(1 for item in perf_list if item[0] == cat)
        if count_in_cat == 0:
            continue
        idx_count += count_in_cat
        cat_indices.append(idx_count)
    # Draw horizontal lines to separate categories
    for boundary in cat_indices[:-1]:  # skip last
        ax.axhline(boundary - 0.5, color="#BFBFBF", linewidth=0.8, linestyle="--")
    ax.axvline(0.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xticks([])
    ax.set_yticks(range(len(bar_df)))
    ax.set_yticklabels(bar_df["Name"], fontsize=10)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0)
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
    """Generate a heatmap table of returns for commodities."""
    default_mapping = {
        # Energy
        "CL1 Comdty": "Oil (WTI)",
        "CO1 Comdty": "Oil (Brent)",
        "TTFG1MON OECM Index": "Natural Gas (TTF)",
        # Green Transition
        "UXA1 Comdty": "Uranium",
        "LMY1 Comdty": "Lithium",
        "IMRU5 Comdty": "Manganese",
        "LP1 Comdty": "Copper",
        "CVT1 Comdty": "Cobalt",
        # Precious Metals
        "GCA Comdty": "Gold",
        "SIA Comdty": "Silver",
        "XPT Comdty": "Platinum",
        "XPD Curncy": "Palladium",
    }
    mapping = ticker_mapping or default_mapping
    tickers = list(mapping.keys())
    df = _load_price_data(excel_path, tickers)
    df_adj, used_date = adjust_prices_for_mode(df, price_mode)
    perf = _build_returns_table(df_adj, mapping)
    heat_df = perf[["Name", "YTD", "1M", "3M", "6M", "12M"]].dropna().sort_values("YTD", ascending=False).reset_index(drop=True)
    n_rows = len(heat_df)
    cols = ["YTD", "1M", "3M", "6M", "12M"]
    color_array = np.zeros((n_rows, len(cols), 4))
    for j, col in enumerate(cols):
        col_vals = heat_df[col].astype(float).values
        pos_vals = col_vals[col_vals > 0]
        neg_vals = col_vals[col_vals < 0]
        max_pos = pos_vals.max() if len(pos_vals) > 0 else 0.0
        min_neg = neg_vals.min() if len(neg_vals) > 0 else 0.0
        for i, val in enumerate(col_vals):
            color_array[i, j] = _colour_for_value(float(val), max_pos, min_neg)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.imshow(color_array, aspect="auto")
    # Add text
    for i in range(n_rows):
        for j, col in enumerate(cols):
            val = heat_df.iloc[i][col]
            ax.text(
                j,
                i,
                f"{val:+.1f}%",
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
    # Insert source footnote if a date is available
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


def insert_commodity_performance_bar_slide(
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
    """Insert the weekly commodity bar chart and source footnote into its slide."""
    return _insert_dashboard_to_placeholder(
        prs,
        image_bytes,
        # Use only commo placeholders for consistency
        placeholder_names=["commo_perf_1w", "commo_perf_1week"],
        left_cm=left_cm,
        top_cm=top_cm,
        width_cm=width_cm,
        height_cm=height_cm,
        used_date=used_date,
        price_mode=price_mode,
        source_placeholder_names=["commo_1w_source"],
    )


def insert_commodity_performance_histo_slide(
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
    """Insert the commodity historical performance heatmap and source footnote."""
    return _insert_dashboard_to_placeholder(
        prs,
        image_bytes,
        # Use only commo placeholder for consistency
        placeholder_names=["commo_perf_histo"],
        left_cm=left_cm,
        top_cm=top_cm,
        width_cm=width_cm,
        height_cm=height_cm,
        used_date=used_date,
        price_mode=price_mode,
        source_placeholder_names=["commo_1w_source2"],
    )