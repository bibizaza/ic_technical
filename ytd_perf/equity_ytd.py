"""Equity YTD performance chart generation and insertion.

This module computes year‑to‑date (YTD) performance for a set of
equity indices, builds a matplotlib chart with connectors and
annotations, and provides a helper to insert the chart and its
subtitle into a PowerPoint slide.  Unlike earlier implementations
that used fixed slide indices, the insertion helper locates the
appropriate slide via a placeholder named ``ytd_eq_perf`` and
inserts the chart at user‑specified coordinates.  The subtitle is
written into a separate textbox named ``ytd_eq_subtitle`` using
formatting preserved from a ``XXX`` placeholder.

Functions
---------

``get_equity_ytd_series``
    Compute percentage change from the start of the current year for
    each equity ticker.
``create_equity_chart``
    Build a YTD line chart with connectors and annotations.
``insert_equity_chart``
    Insert the chart and subtitle into a slide identified by
    placeholders.
"""

from __future__ import annotations

from datetime import datetime
import os
import tempfile
from typing import List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from pptx import Presentation
from pptx.util import Cm

from .loader_update import load_data

# Colour mapping for equities
EQUITY_COLOURS = {
    "S&P 500": "#203864", "SPX Index": "#203864",
    "Shenzen CSI 300": "#00B0F0", "CSI 300": "#00B0F0", "SHSZ300 Index": "#00B0F0",
    "Dax": "#0070C0", "DAX": "#0070C0", "DAX Index": "#0070C0",
    "Ibov": "#BF9000", "IBOV": "#BF9000", "IBOV Index": "#BF9000",
    "Sensex": "#B4C7E7", "Sensex Index": "#B4C7E7", "SENSEX Index": "#B4C7E7",
    "TASI": "#A6A6A6", "SASEIDX Index": "#A6A6A6",
    "Nikkei 225": "#FFC000", "Nikkei": "#FFC000", "NKY Index": "#FFC000",
    "SMI": "#E4AAF4", "SMI Index": "#E4AAF4",
}


def get_equity_ytd_series(file_path: str, tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute YTD performance for equity tickers.  If ``tickers`` is
    ``None``, all equity tickers from the parameters sheet are
    included.  YTD is calculated from 1 January of the current year.

    Parameters
    ----------
    file_path : str
        Path to the Excel workbook containing price and parameter
        sheets.
    tickers : list of str, optional
        Specific tickers to include; if omitted, all equities are
        used.

    Returns
    -------
    DataFrame
        A DataFrame with a ``Date`` column and one column per
        equity name, containing percentage changes from the start of
        the year.
    """
    prices_df, params_df = load_data(file_path)
    now = datetime.now()
    year_start = datetime(now.year, 1, 1)
    # Filter data from the start of the year
    current_year_df = prices_df[prices_df["Date"] >= year_start].reset_index(drop=True)
    eq_params = params_df[params_df["Asset Class"] == "Equity"].copy()
    if tickers is not None:
        eq_params = eq_params[eq_params["Tickers"].isin(tickers)]
    result = pd.DataFrame()
    result["Date"] = current_year_df["Date"]
    for _, row in eq_params.iterrows():
        ticker = str(row["Tickers"]).strip()
        name = str(row["Name"]).strip()
        if ticker not in current_year_df.columns:
            continue
        series = current_year_df[ticker].astype(float)
        base_series = series.dropna()
        if base_series.empty:
            continue
        base_val = base_series.iloc[0]
        if base_val == 0 or pd.isna(base_val):
            continue
        ytd = (series / base_val - 1.0) * 100.0
        result[name] = ytd.reset_index(drop=True)
    return result


def create_equity_chart(df: pd.DataFrame) -> plt.Figure:
    """Create a YTD equity chart with connectors and bold labels.

    The chart plots one line per equity index.  At the right edge,
    connectors lead to bold labels indicating the final value of each
    series.  Colours are assigned according to ``EQUITY_COLOURS``.

    Parameters
    ----------
    df : DataFrame
        The DataFrame returned by ``get_equity_ytd_series``.

    Returns
    -------
    Figure
        A matplotlib Figure object ready for saving.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    # Plot each equity line
    for col in df.columns:
        if col == "Date":
            continue
        colour = EQUITY_COLOURS.get(col, None)
        ax.plot(df["Date"], df[col], color=colour, linewidth=2)
    # Determine y-range and sort by final values
    y_min = df[[c for c in df.columns if c != "Date"]].min().min()
    y_max = df[[c for c in df.columns if c != "Date"]].max().max()
    y_range = y_max - y_min if y_max > y_min else 1.0
    series_cols = [c for c in df.columns if c != "Date"]
    sorted_cols = sorted(series_cols, key=lambda c: df[c].iloc[-1], reverse=True)
    # Annotate with connectors
    for idx, col in enumerate(sorted_cols):
        colour = EQUITY_COLOURS.get(col, None)
        last_x = df["Date"].iloc[-1]
        last_y = df[col].iloc[-1]
        offset_y = -idx * 0.02 * y_range
        target_y = last_y + offset_y
        perf_text = f"{last_y:+.1f}%"
        annotation = f"{col}: {perf_text}"
        x_offset = pd.Timedelta(days=5)
        ax.plot([last_x, last_x + x_offset], [last_y, target_y], color="#BFBFBF", linewidth=0.5)
        ax.text(
            last_x + x_offset,
            target_y,
            annotation,
            color=colour,
            fontsize=8,
            fontweight="bold",
            va="center",
            ha="left",
        )
    # Style axes
    ax.set_title("YTD Performance of Equity Indices (%)", fontsize=14, color="#0A1F44")
    ax.set_xlabel("")
    ax.set_ylabel("Performance")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    fig.autofmt_xdate()
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def insert_equity_chart(
    prs: Presentation,
    file_path: str,
    subtitle: str = "",
    tickers: Optional[List[str]] = None,
    *,
    left_cm: float = 1.87,
    top_cm: float = 5.49,
    width_cm: float = 20.64,
    height_cm: float = 9.57,
) -> Presentation:
    """Insert the YTD equity chart and subtitle into a PowerPoint slide.

    This helper searches for a shape named ``ytd_eq_perf`` or containing
    ``[ytd_eq_perf]`` to locate the correct slide.  It then inserts
    the chart image at the specified coordinates, leaving the
    placeholder text intact.  The subtitle is written into the shape
    named ``ytd_eq_subtitle`` by replacing the placeholder ``XXX``
    while preserving the original formatting of the text run.

    Parameters
    ----------
    prs : Presentation
        PowerPoint presentation to modify.
    file_path : str
        Path to the Excel workbook.
    subtitle : str, optional
        Subtitle text to insert in place of ``XXX``.
    tickers : list of str, optional
        Equity tickers to include; if omitted, all equities are
        included.
    left_cm, top_cm, width_cm, height_cm : float
        Coordinates and dimensions (in centimetres) for the chart
        placement.

    Returns
    -------
    Presentation
        The modified presentation.
    """
    # Compute YTD data and chart
    df_eq = get_equity_ytd_series(file_path, tickers)
    fig = create_equity_chart(df_eq)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_png:
        fig.savefig(tmp_png.name, dpi=200)
        chart_path = tmp_png.name
    try:
        # Locate slide by ytd_eq_perf placeholder
        target_slide = None
        for slide in prs.slides:
            for shape in slide.shapes:
                name_attr = getattr(shape, "name", "").lower()
                if name_attr == "ytd_eq_perf":
                    target_slide = slide
                    break
                if shape.has_text_frame:
                    text_lower = (shape.text or "").strip().lower()
                    if text_lower == "[ytd_eq_perf]":
                        target_slide = slide
                        break
            if target_slide:
                break
        # If the placeholder slide is not found, skip inserting the chart.
        # This avoids placing the equity YTD chart on an unrelated slide
        # when the target slide has been removed from the template.
        if target_slide is None:
            return prs
        # Insert subtitle
        if subtitle:
            for shape in target_slide.shapes:
                name_attr = getattr(shape, "name", "")
                if name_attr and name_attr.lower() == "ytd_eq_subtitle" and shape.has_text_frame:
                    tf = shape.text_frame
                    for paragraph in tf.paragraphs:
                        for run in paragraph.runs:
                            if "XXX" in run.text:
                                original_font = run.font
                                run.text = run.text.replace("XXX", subtitle)
                                run.font.name = original_font.name
                                run.font.size = original_font.size
                                run.font.bold = original_font.bold
                                run.font.italic = original_font.italic
                                run.font.color.rgb = original_font.color.rgb
                                break
                    break
        # Convert coordinates to EMU
        left = Cm(left_cm)
        top = Cm(top_cm)
        width = Cm(width_cm)
        height = Cm(height_cm)
        # Insert chart picture
        picture = target_slide.shapes.add_picture(chart_path, left, top, width, height)
        # Bring to front
        sp_tree = target_slide.shapes._spTree
        sp_tree.remove(picture._element)
        sp_tree.insert(1, picture._element)
        return prs
    finally:
        os.remove(chart_path)