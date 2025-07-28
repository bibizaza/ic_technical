"""Equity performance dashboard generation.

This module produces a two–panel dashboard summarising recent
performance for a selection of major equity indices.  The left
panel shows a horizontal bar chart of 1‑week returns, sorted from
best to worst.  The right panel displays a heatmap of longer
horizons (YTD, 1‑month, 3‑month, 6‑month and 12‑month) with
independent colour scales per column.  Positive values are mapped
to greens and negative values to reds, making it easy to see
relative strength across markets and timeframes.  The functions
exposed here allow the dashboard to be generated as a high‑
resolution PNG and inserted into a PowerPoint slide.

Key functions
-------------

* ``create_equity_performance_figure`` – Build the matplotlib figure
  for the dashboard and return its binary PNG data.  The caller can
  control the set of tickers and their display names via an
  optional mapping.
* ``insert_equity_performance_slide`` – Insert the generated
  dashboard into a slide.  The function looks for a shape named
  ``equity_perf`` (or containing ``[equity_perf]``) and uses that
  location if present.  Otherwise it inserts at a default
  coordinate.  Dimensions and positions can be overridden by the
  caller.

Example usage::

    from performance.equity_perf import (
        create_equity_performance_figure,
        insert_equity_performance_slide,
    )
    fig_bytes = create_equity_performance_figure(excel_path)
    prs = Presentation(template_path)
    prs = insert_equity_performance_slide(prs, fig_bytes)
    prs.save("updated.pptx")

"""

from __future__ import annotations

import io
import pathlib
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.util import Cm

###############################################################################
# Data loading and return calculations
###############################################################################

def _load_price_data(excel_path: Union[str, pathlib.Path], tickers: List[str]) -> pd.DataFrame:
    """Load a DataFrame of dates and prices for the specified tickers.

    Parameters
    ----------
    excel_path : str or pathlib.Path
        Path to the Excel workbook containing a sheet named ``data_prices``.
    tickers : list of str
        Column names in the sheet corresponding to the desired indices.

    Returns
    -------
    DataFrame
        A tidy DataFrame with columns ``Date`` and one column per ticker,
        sorted by date and with numeric prices.
    """
    df = pd.read_excel(excel_path, sheet_name="data_prices")
    # Drop the first row which usually contains metadata and rows with the
    # sentinel ``DATES`` value.
    df = df.drop(index=0)
    df = df[df[df.columns[0]] != "DATES"].copy()
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Restrict to requested tickers and coerce to numeric
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
    """Compute the percentage return of a series over the given lookback.

    This helper finds the last available price on or before ``today`` minus
    ``days`` and compares it to the most recent price.  If insufficient
    history exists, NaN is returned.

    Parameters
    ----------
    df : DataFrame
        A tidy DataFrame with ``Date`` and price columns.
    ticker : str
        Column name for which to compute the return.
    today : Timestamp
        Anchor date; typically the last date in the dataset.
    days : int
        Lookback period in days.

    Returns
    -------
    float
        Percent return over the lookback horizon, or NaN if not computable.
    """
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
    """Calculate performance metrics for each ticker.

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
        Performance table.
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
        # Year‑to‑date: look back to the start of the calendar year
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
# Figure generation
###############################################################################

def _colour_for_value(val: float, max_pos: float, min_neg: float) -> Tuple[float, float, float, float]:
    """Return an RGBA colour for a single value using independent scales.

    Positive values produce a gradient from white to green; negative
    values from white to red.  Zeros (or undefined scales) return
    white.  The input ``max_pos`` should be the maximum positive value
    in the column, and ``min_neg`` the minimum (most negative) value.

    Parameters
    ----------
    val : float
        The value to colour.
    max_pos : float
        The maximum positive value in the column.
    min_neg : float
        The minimum (most negative) value in the column.

    Returns
    -------
    tuple of float
        RGBA quadruple in the range 0–1.
    """
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


def create_equity_performance_figure(
    excel_path: Union[str, pathlib.Path],
    ticker_mapping: Dict[str, str] | None = None,
    *,
    width: float = 14.0,
    height: float = 9.0,
) -> bytes:
    """Generate the equity performance dashboard as a PNG.

    The function reads price data from the supplied Excel workbook,
    computes recent returns for a set of major indices and builds a
    two‑panel figure.  By default the following tickers are used:

    ``SPX Index``, ``CCMP Index``, ``RTY Index``, ``IBOV Index``,
    ``MEXBOL Index``, ``SX5E Index``, ``SXXP Index``, ``UKX Index``,
    ``MXME Index``, ``SMI Index``, ``HSI Index``, ``SHSZ300 Index``,
    ``NKY Index``, ``VNINDEX Index``, ``SKMQAXJN Index``, ``SENSEX Index``,
    ``DAX Index``, ``SASEIDX Index`` and ``MXWO Index``.

    Parameters
    ----------
    excel_path : str or pathlib.Path
        Path to the Excel workbook containing the price series.
    ticker_mapping : dict, optional
        Mapping from ticker codes to display names.  If not provided,
        sensible defaults are used.
    width : float, default 14.0
        Figure width in inches.
    height : float, default 9.0
        Figure height in inches.

    Returns
    -------
    bytes
        PNG data for the generated dashboard.
    """
    # Default mapping of tickers to human‑friendly names
    default_mapping = {
        "SPX Index": "S&P 500",
        "CCMP Index": "Nasdaq Composite",
        "RTY Index": "Russell 2000",
        "IBOV Index": "Bovespa",
        "MEXBOL Index": "Mexican Bolsa",
        "SX5E Index": "Eurostoxx 50",
        "SXXP Index": "Stoxx 600",
        "UKX Index": "FTSE 100",
        "MXME Index": "MSCI EM Eastern Europe",
        "SMI Index": "Swiss Market Index",
        "HSI Index": "Hang Seng",
        "SHSZ300 Index": "Shenzhen CSI 300",
        "NKY Index": "Nikkei 225",
        "VNINDEX Index": "Ho Chi Minh",
        "SKMQAXJN Index": "Solactive Macquarie Asia ex JP",
        "SENSEX Index": "Sensex",
        "DAX Index": "DAX 30",
        "SASEIDX Index": "TASI (Saudi Index)",
        "MXWO Index": "MSCI World",
    }
    mapping = ticker_mapping or default_mapping
    tickers = list(mapping.keys())

    # Load and compute returns
    df = _load_price_data(excel_path, tickers)
    perf = _build_returns_table(df, mapping)

    # Build bar chart data (1W) sorted by performance descending
    bar_df = perf[["Name", "1W"]].dropna().sort_values("1W", ascending=False).reset_index(drop=True)
    # Build heatmap data sorted by YTD descending
    heat_df = perf[["Name", "YTD", "1M", "3M", "6M", "12M"]].dropna().sort_values("YTD", ascending=False).reset_index(drop=True)

    # Precompute colour array per column with independent scaling
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

    # Create figure and axes
    fig, (ax_bar, ax_heat) = plt.subplots(
        2,
        1,
        figsize=(width, height),
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # Bar chart
    bar_colors = ["#2E8B57" if x > 0 else "#C70039" for x in bar_df["1W"]]
    bars = ax_bar.barh(bar_df["Name"], bar_df["1W"], color=bar_colors)
    # Add labels to bars
    for bar in bars:
        width_val = bar.get_width()
        x_pos = width_val + (0.1 if width_val > 0 else -0.1)
        ax_bar.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2.0,
            f"{width_val:.1f}%",
            va="center",
            ha="left" if width_val > 0 else "right",
            fontweight="bold",
            color="black",
        )
    # Style bar chart
    ax_bar.axvline(0.0, color="grey", linewidth=0.8, linestyle="--")
    ax_bar.set_xticks([])
    ax_bar.set_yticks(range(len(bar_df)))
    ax_bar.set_yticklabels(bar_df["Name"], fontsize=10)
    ax_bar.invert_yaxis()
    for spine in ax_bar.spines.values():
        spine.set_visible(False)
    ax_bar.tick_params(axis="y", length=0)
    # Provide extra horizontal space so bars do not overlap names
    min_val = min(bar_df["1W"].min(), 0)
    max_val = bar_df["1W"].max()
    margin = (max_val - min_val) * 0.2
    ax_bar.set_xlim(min_val - margin, max_val + margin)

    # Heatmap
    ax_heat.imshow(color_array, aspect="auto")
    for i in range(n_rows):
        for j, col in enumerate(cols):
            val = heat_df.iloc[i][col]
            # Always render text in black for legibility; the colour cell
            # carries the sign/intensity
            ax_heat.text(
                j,
                i,
                f"{val:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="black",
            )
    ax_heat.set_xticks(range(len(cols)))
    ax_heat.set_xticklabels(cols, fontsize=12, fontweight="bold")
    ax_heat.xaxis.set_ticks_position("top")
    ax_heat.xaxis.set_label_position("top")
    ax_heat.set_yticks(range(n_rows))
    ax_heat.set_yticklabels(heat_df["Name"], fontsize=9)
    ax_heat.tick_params(axis="y", length=0)
    ax_heat.tick_params(axis="x", length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    # Adjust layout to create a large left margin for names
    fig.subplots_adjust(left=0.5, hspace=0.4)

    # Render to PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


###############################################################################
# Separate figure generation for weekly bar and historical heatmap
###############################################################################

def create_weekly_performance_chart(
    excel_path: Union[str, pathlib.Path],
    ticker_mapping: Dict[str, str] | None = None,
    *,
    width: float = 14.0,
    height: float = 5.0,
) -> bytes:
    """Generate a bar chart of 1‑week returns.

    This helper reads the specified Excel file, computes 1‑week returns
    for the configured tickers, sorts them in descending order and
    constructs a horizontal bar chart.  Names are used instead of
    ticker codes.  The caller can control the figure size via
    ``width`` and ``height``.

    Parameters
    ----------
    excel_path : str or pathlib.Path
        Path to the Excel workbook containing the price series.
    ticker_mapping : dict, optional
        Mapping from ticker codes to display names.  If not provided,
        sensible defaults are used.
    width, height : float
        Dimensions of the generated figure in inches.

    Returns
    -------
    bytes
        PNG data for the bar chart.
    """
    default_mapping = {
        "SPX Index": "S&P 500",
        "CCMP Index": "Nasdaq Composite",
        "RTY Index": "Russell 2000",
        "IBOV Index": "Bovespa",
        "MEXBOL Index": "Mexican Bolsa",
        "SX5E Index": "Eurostoxx 50",
        "SXXP Index": "Stoxx 600",
        "UKX Index": "FTSE 100",
        "MXME Index": "MSCI EM Eastern Europe",
        "SMI Index": "Swiss Market Index",
        "HSI Index": "Hang Seng",
        "SHSZ300 Index": "Shenzhen CSI 300",
        "NKY Index": "Nikkei 225",
        "VNINDEX Index": "Ho Chi Minh",
        "SKMQAXJN Index": "Solactive Macquarie Asia ex JP",
        "SENSEX Index": "Sensex",
        "DAX Index": "DAX 30",
        "SASEIDX Index": "TASI (Saudi Index)",
        "MXWO Index": "MSCI World",
    }
    mapping = ticker_mapping or default_mapping
    tickers = list(mapping.keys())
    # Load data and compute returns
    df = _load_price_data(excel_path, tickers)
    perf = _build_returns_table(df, mapping)
    # Build bar chart data sorted by performance descending
    bar_df = perf[["Name", "1W"]].dropna().sort_values("1W", ascending=False).reset_index(drop=True)
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(width, height))
    bar_colors = ["#2E8B57" if x > 0 else "#C70039" for x in bar_df["1W"]]
    bars = ax.barh(bar_df["Name"], bar_df["1W"], color=bar_colors)
    # Add labels
    for bar in bars:
        width_val = bar.get_width()
        x_pos = width_val + (0.1 if width_val > 0 else -0.1)
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2.0,
            f"{width_val:.1f}%",
            va="center",
            ha="left" if width_val > 0 else "right",
            fontweight="bold",
            color="black",
        )
    # Style chart
    ax.axvline(0.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xticks([])
    ax.set_yticks(range(len(bar_df)))
    ax.set_yticklabels(bar_df["Name"], fontsize=10)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0)
    # Provide extra horizontal space so bars do not overlap names
    min_val = min(bar_df["1W"].min(), 0)
    max_val = bar_df["1W"].max()
    margin = (max_val - min_val) * 0.2
    ax.set_xlim(min_val - margin, max_val + margin)
    fig.subplots_adjust(left=0.4)
    # Export to PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def create_historical_performance_table(
    excel_path: Union[str, pathlib.Path],
    ticker_mapping: Dict[str, str] | None = None,
    *,
    width: float = 14.0,
    height: float = 6.0,
) -> bytes:
    """Generate a heatmap table of historical returns.

    This helper reads the price data, computes returns for multiple
    horizons, sorts the table by YTD performance (descending) and
    constructs a heatmap where each column is coloured independently.
    Positive values map to greens and negative values to reds.  The
    columns appear in the order: YTD, 1M, 3M, 6M, 12M.

    Parameters
    ----------
    excel_path : str or pathlib.Path
        Path to the Excel workbook containing the price series.
    ticker_mapping : dict, optional
        Mapping from ticker codes to display names.  If not provided,
        sensible defaults are used.
    width, height : float
        Dimensions of the generated figure in inches.

    Returns
    -------
    bytes
        PNG data for the heatmap table.
    """
    default_mapping = {
        "SPX Index": "S&P 500",
        "CCMP Index": "Nasdaq Composite",
        "RTY Index": "Russell 2000",
        "IBOV Index": "Bovespa",
        "MEXBOL Index": "Mexican Bolsa",
        "SX5E Index": "Eurostoxx 50",
        "SXXP Index": "Stoxx 600",
        "UKX Index": "FTSE 100",
        "MXME Index": "MSCI EM Eastern Europe",
        "SMI Index": "Swiss Market Index",
        "HSI Index": "Hang Seng",
        "SHSZ300 Index": "Shenzhen CSI 300",
        "NKY Index": "Nikkei 225",
        "VNINDEX Index": "Ho Chi Minh",
        "SKMQAXJN Index": "Solactive Macquarie Asia ex JP",
        "SENSEX Index": "Sensex",
        "DAX Index": "DAX 30",
        "SASEIDX Index": "TASI (Saudi Index)",
        "MXWO Index": "MSCI World",
    }
    mapping = ticker_mapping or default_mapping
    tickers = list(mapping.keys())
    # Load data and compute returns
    df = _load_price_data(excel_path, tickers)
    perf = _build_returns_table(df, mapping)
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
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(width, height))
    ax.imshow(color_array, aspect="auto")
    # Add text
    for i in range(n_rows):
        for j, col in enumerate(cols):
            val = heat_df.iloc[i][col]
            ax.text(
                j,
                i,
                f"{val:.1f}%",
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
    return buf.getvalue()


###############################################################################
# Slide insertion
###############################################################################

def insert_equity_performance_slide(
    prs: Presentation,
    image_bytes: bytes,
    *,
    left_cm: float = 0.93,
    top_cm: float = 4.4,
    width_cm: float = 25.0,
    height_cm: float = 7.3,
) -> Presentation:
    """Insert the equity performance dashboard into a slide.

    The function searches for a shape named ``equity_perf`` or a
    textbox containing the text ``[equity_perf]`` to determine where
    to place the image.  If no such placeholder is found, the image
    is inserted at the default position specified by the ``left_cm``,
    ``top_cm``, ``width_cm`` and ``height_cm`` parameters.

    Parameters
    ----------
    prs : Presentation
        The PowerPoint presentation into which the image should be
        inserted.
    image_bytes : bytes
        PNG data for the dashboard generated by
        ``create_equity_performance_figure``.
    left_cm, top_cm, width_cm, height_cm : float
        Position and size (in centimetres) to use if a placeholder is
        not found.  These values can be tuned to match your template.

    Returns
    -------
    Presentation
        The modified presentation.
    """
    placeholder_name = "equity_perf"
    placeholder_patterns = ["[equity_perf]", "equity_perf"]
    target_slide = None
    placeholder_box = None
    for slide in prs.slides:
        for shape in slide.shapes:
            name_attr = getattr(shape, "name", "").lower()
            if name_attr == placeholder_name:
                target_slide = slide
                placeholder_box = shape
                break
            if shape.has_text_frame:
                text_lower = (shape.text or "").strip().lower()
                if text_lower in [p.lower() for p in placeholder_patterns]:
                    target_slide = slide
                    placeholder_box = shape
                    break
        if target_slide:
            break

    # Use the found slide or fall back to the last slide
    if target_slide is None:
        target_slide = prs.slides[min(11, len(prs.slides) - 1)]

    # Determine insertion bounds
    # If a placeholder is found, we use it only to identify the slide
    # but not its dimensions.  Always insert at the caller‑specified
    # position and size.  This approach allows the template to act
    # merely as a marker without constraining the dashboard's
    # dimensions.
    if placeholder_box is not None:
        # Clear the placeholder text if present
        if placeholder_box.has_text_frame:
            placeholder_box.text = ""

    # Always use the provided coordinates and dimensions.  If the
    # caller supplies custom values, they override the defaults.
    left = Cm(left_cm)
    top = Cm(top_cm)
    width = Cm(width_cm)
    height = Cm(height_cm)

    # Insert the image
    stream = io.BytesIO(image_bytes)
    target_slide.shapes.add_picture(stream, left, top, width=width, height=height)
    return prs


def _insert_dashboard_to_placeholder(
    prs: Presentation,
    image_bytes: bytes,
    placeholder_names: List[str],
    *,
    left_cm: float,
    top_cm: float,
    width_cm: float,
    height_cm: float,
) -> Presentation:
    """Helper to insert an image into a slide identified by placeholders.

    This internal helper searches for shapes whose names match any of
    ``placeholder_names`` or whose text contains any of those names in
    square brackets.  Once found, it uses the slide but ignores the
    shape's dimensions, inserting the provided image at the specified
    coordinates and size.  If no matching placeholder is found, the
    image is inserted on the last slide.

    Parameters
    ----------
    prs : Presentation
        The presentation into which the image should be inserted.
    image_bytes : bytes
        The PNG data to insert.
    placeholder_names : list of str
        Names of shapes to look for (lowercased).  The function also
        recognises text placeholders like ``[equity_perf_1week]``.
    left_cm, top_cm, width_cm, height_cm : float
        Desired position and size in centimetres.

    Returns
    -------
    Presentation
        The modified presentation.
    """
    target_slide = None
    placeholder_box = None
    # Normalise names for comparison
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
    # Do not modify or remove the placeholder box; it serves only to locate
    # the slide.  Leave its text intact so the placeholder remains in
    # the template for future reference.
    left = Cm(left_cm)
    top = Cm(top_cm)
    width = Cm(width_cm)
    height = Cm(height_cm)
    stream = io.BytesIO(image_bytes)
    target_slide.shapes.add_picture(stream, left, top, width=width, height=height)
    return prs


def insert_equity_performance_bar_slide(
    prs: Presentation,
    image_bytes: bytes,
    *,
    left_cm: float = 0.0,
    top_cm: float = 3.22,
    width_cm: float = 25.0,
    height_cm: float = 11.66,
) -> Presentation:
    """Insert the weekly performance bar chart into its designated slide.

    The slide is identified by a shape named ``equity_perf_1week`` or
    containing the text ``[equity_perf_1week]``.  The image is
    inserted at the specified coordinates regardless of the
    placeholder's original size.

    Returns the modified presentation.
    """
    # Accept both singular and abbreviated placeholder names for the weekly bar
    return _insert_dashboard_to_placeholder(
        prs,
        image_bytes,
        placeholder_names=["equity_perf_1w", "equity_perf_1week"],
        left_cm=left_cm,
        top_cm=top_cm,
        width_cm=width_cm,
        height_cm=height_cm,
    )


def insert_equity_performance_histo_slide(
    prs: Presentation,
    image_bytes: bytes,
    *,
    left_cm: float = 0.0,
    top_cm: float = 3.22,
    width_cm: float = 25.0,
    height_cm: float = 11.66,
) -> Presentation:
    """Insert the historical performance heatmap into its designated slide.

    The slide is identified by a shape named ``equity_perf_histo`` or
    containing the text ``[equity_perf_histo]``.  The image is
    inserted at the specified coordinates regardless of the
    placeholder's original size.

    Returns the modified presentation.
    """
    return _insert_dashboard_to_placeholder(
        prs,
        image_bytes,
        placeholder_names=["equity_perf_histo"],
        left_cm=left_cm,
        top_cm=top_cm,
        width_cm=width_cm,
        height_cm=height_cm,
    )