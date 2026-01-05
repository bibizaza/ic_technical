"""Equity performance dashboard generation with price‐mode awareness and source footnotes.

This module produces charts summarising recent performance for a selection of major
equity indices.  Users can choose to base calculations on either the most
recent intraday price ("Last Price") or the previous day's closing price
("Last Close").  The resulting figures include a weekly returns bar chart and
a heatmap of longer horizons.  When the charts are inserted into a PowerPoint
presentation, a source footnote is added to the designated text boxes on
each slide, reflecting the effective date used in the computations and the
chosen price mode.  Text formatting from the placeholders is preserved.

Key functions
-------------

* ``create_weekly_performance_chart`` – Build the weekly bar chart and return
  both the image bytes and the effective date used for computation.
* ``create_historical_performance_table`` – Build the heatmap of returns for
  multiple horizons and return the image bytes and the effective date.
* ``insert_equity_performance_bar_slide`` – Insert the weekly bar chart and
  source footnote into its slide.
* ``insert_equity_performance_histo_slide`` – Insert the historical performance
  heatmap and source footnote into its slide.

Usage example::

    from performance.equity_perf import (
        create_weekly_performance_chart,
        create_historical_performance_table,
        insert_equity_performance_bar_slide,
        insert_equity_performance_histo_slide,
    )

    # Generate charts with price-mode awareness
    bar_bytes, used_date = create_weekly_performance_chart(excel_path, price_mode="Last Close")
    histo_bytes, _ = create_historical_performance_table(excel_path, price_mode="Last Close")
    # Insert into PPT, supplying used_date and price_mode for source footnote
    prs = insert_equity_performance_bar_slide(prs, bar_bytes, used_date, price_mode)
    prs = insert_equity_performance_histo_slide(prs, histo_bytes, used_date, price_mode)

"""

from __future__ import annotations

import io
import os
import json
import pathlib
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.util import Cm
from jinja2 import Template, Environment
from html2image import Html2Image

from utils import adjust_prices_for_mode
from market_compass.weekly_performance.html_template import EQUITY_YTD_EVOLUTION_HTML_TEMPLATE

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
    """Generate a bar chart of 1‑week returns with price‑mode adjustment.

    This helper reads the specified Excel file, optionally adjusts the price
    history according to the selected ``price_mode`` (``"Last Price"`` vs
    ``"Last Close"``), computes 1‑week returns for the configured tickers,
    sorts them in descending order and constructs a horizontal bar chart.
    The effective date used for the calculations is returned along with
    the PNG data.

    Parameters
    ----------
    excel_path : str or pathlib.Path
        Path to the Excel workbook containing the price series.
    ticker_mapping : dict, optional
        Mapping from ticker codes to display names.  If not provided,
        sensible defaults are used.
    width, height : float
        Dimensions of the generated figure in inches.
    price_mode : str, default ``"Last Price"``
        One of ``"Last Price"`` or ``"Last Close"``.  Determines whether
        intraday prices are used or the most recent row is dropped if it
        corresponds to today's date.

    Returns
    -------
    tuple
        A two‑tuple ``(image_bytes, used_date)`` where ``image_bytes`` is
        the PNG data for the bar chart and ``used_date`` is the effective
        date in the adjusted price series.
    """
    import tempfile
    from pathlib import Path
    from jinja2 import Template
    from html2image import Html2Image
    from market_compass.weekly_performance.html_template import WEEKLY_PERFORMANCE_HTML_TEMPLATE

    # Index mapping with display names and flags
    INDEX_CONFIG = {
        "SPX Index": {"name": "S&P 500", "flag": "\U0001F1FA\U0001F1F8"},
        "DAX Index": {"name": "Dax", "flag": "\U0001F1E9\U0001F1EA"},
        "SMI Index": {"name": "SMI", "flag": "\U0001F1E8\U0001F1ED"},
        "NKY Index": {"name": "Nikkei 225", "flag": "\U0001F1EF\U0001F1F5"},
        "SHSZ300 Index": {"name": "CSI 300", "flag": "\U0001F1E8\U0001F1F3"},
        "SENSEX Index": {"name": "Sensex", "flag": "\U0001F1EE\U0001F1F3"},
        "IBOV Index": {"name": "Bovespa", "flag": "\U0001F1E7\U0001F1F7"},
        "MEXBOL Index": {"name": "Mexbol", "flag": "\U0001F1F2\U0001F1FD"},
        "SASEIDX Index": {"name": "TASI", "flag": "\U0001F1F8\U0001F1E6"},
    }

    tickers = list(INDEX_CONFIG.keys())

    # Load data and adjust according to price mode
    df = _load_price_data(excel_path, tickers)
    df_adj, used_date = adjust_prices_for_mode(df, price_mode)

    # Build returns
    default_mapping = {k: v["name"] for k, v in INDEX_CONFIG.items()}
    perf = _build_returns_table(df_adj, default_mapping)

    # Get 1W returns and sort descending
    bar_df = perf[["Name", "1W"]].dropna().sort_values("1W", ascending=False).reset_index(drop=True)

    # Prepare rows for HTML template
    rows = []
    max_abs_value = max(abs(bar_df["1W"].max()), abs(bar_df["1W"].min())) if len(bar_df) > 0 else 1
    max_abs_value = max(max_abs_value, 1.0)  # At least 1%

    for i, row in bar_df.iterrows():
        name = row["Name"]
        value = row["1W"] / 100  # Convert percentage to decimal

        # Find flag for this name
        flag = ""
        for ticker, config in INDEX_CONFIG.items():
            if config["name"] == name:
                flag = config["flag"]
                break

        # Determine highlight class
        highlight_class = ""
        if i == 0 and value > 0:
            highlight_class = "top-performer"
        elif i == len(bar_df) - 1 and value < 0:
            highlight_class = "worst-performer"

        # Bar width as percentage of half the chart
        bar_width = abs(value) / (max_abs_value / 100) * 50
        bar_width = min(bar_width, 48)

        # Value classes
        if value > 0:
            bar_class = "positive"
            value_class = "positive"
            formatted_value = f"+{value * 100:.1f}%"
        elif value < 0:
            bar_class = "negative"
            value_class = "negative"
            formatted_value = f"{value * 100:.1f}%"
        else:
            bar_class = ""
            value_class = "zero"
            formatted_value = "0.0%"

        rows.append({
            "name": name,
            "flag": flag,
            "value": value,
            "highlight_class": highlight_class,
            "bar_class": bar_class,
            "bar_width": bar_width,
            "value_class": value_class,
            "formatted_value": formatted_value,
        })

    # Calculate scale
    if max_abs_value <= 1:
        scale_max = 1
    elif max_abs_value <= 2:
        scale_max = 2
    elif max_abs_value <= 5:
        scale_max = 5
    elif max_abs_value <= 10:
        scale_max = 10
    else:
        scale_max = int(max_abs_value) + 5

    scale_values = {
        "scale_min": f"-{scale_max}%",
        "scale_mid_low": f"-{scale_max // 2}%",
        "scale_mid_high": f"+{scale_max // 2}%",
        "scale_max": f"+{scale_max}%",
    }

    # Generate HTML - PNG must be 3x scale for crisp rendering
    # Target: 17.31cm × 10cm in PowerPoint
    SCALE_FACTOR = 3
    width_px = int(17.31 * 37.8 * SCALE_FACTOR)  # = 1963 px
    height_px = int(10 * 37.8 * SCALE_FACTOR)     # = 1134 px

    template = Template(WEEKLY_PERFORMANCE_HTML_TEMPLATE)
    html = template.render(
        rows=rows,
        width=width_px,
        height=height_px,
        scale=SCALE_FACTOR,
        **scale_values,
    )

    # Convert to PNG
    with tempfile.TemporaryDirectory() as tmpdir:
        hti = Html2Image(output_path=tmpdir, size=(width_px, height_px))
        hti.screenshot(html_str=html, save_as="weekly_perf.png")

        img_path = Path(tmpdir) / "weekly_perf.png"
        with open(img_path, "rb") as f:
            png_bytes = f.read()

    return png_bytes, used_date


def create_historical_performance_table(
    excel_path: Union[str, pathlib.Path],
    ticker_mapping: Dict[str, str] | None = None,
    *,
    width: float = 14.0,
    height: float = 6.0,
    price_mode: str = "Last Price",
) -> Tuple[bytes, Optional[pd.Timestamp]]:
    """Generate a heatmap table of historical returns with price‑mode adjustment.

    This helper reads the price data, optionally adjusts it according to
    ``price_mode``, computes returns for multiple horizons, sorts the
    table by YTD performance (descending) and constructs a heatmap where
    each column is coloured independently.  The effective date used
    for the computations is returned along with the PNG data.

    Parameters
    ----------
    excel_path : str or pathlib.Path
        Path to the Excel workbook containing the price series.
    ticker_mapping : dict, optional
        Mapping from ticker codes to display names.  If not provided,
        sensible defaults are used.
    width, height : float
        Dimensions of the generated figure in inches.
    price_mode : str, default ``"Last Price"``
        Either ``"Last Price"`` or ``"Last Close"``.  Determines how
        price data is adjusted prior to computing returns and which
        effective date appears in the source footnote.

    Returns
    -------
    tuple
        A two‑tuple ``(image_bytes, used_date)`` where ``image_bytes`` is
        the PNG data for the heatmap and ``used_date`` is the effective
        date in the adjusted price series.
    """
    import tempfile
    from pathlib import Path
    from jinja2 import Template
    from html2image import Html2Image
    from market_compass.weekly_performance.html_template import HISTORICAL_PERFORMANCE_HTML_TEMPLATE

    # Index mapping with display names and flags
    INDEX_CONFIG = {
        "SPX Index": {"name": "S&P 500", "flag": "\U0001F1FA\U0001F1F8"},
        "DAX Index": {"name": "Dax", "flag": "\U0001F1E9\U0001F1EA"},
        "SMI Index": {"name": "SMI", "flag": "\U0001F1E8\U0001F1ED"},
        "NKY Index": {"name": "Nikkei 225", "flag": "\U0001F1EF\U0001F1F5"},
        "SHSZ300 Index": {"name": "CSI 300", "flag": "\U0001F1E8\U0001F1F3"},
        "SENSEX Index": {"name": "Sensex", "flag": "\U0001F1EE\U0001F1F3"},
        "IBOV Index": {"name": "Bovespa", "flag": "\U0001F1E7\U0001F1F7"},
        "MEXBOL Index": {"name": "Mexbol", "flag": "\U0001F1F2\U0001F1FD"},
        "SASEIDX Index": {"name": "TASI", "flag": "\U0001F1F8\U0001F1E6"},
    }

    tickers = list(INDEX_CONFIG.keys())

    # Load data and adjust according to price mode
    df = _load_price_data(excel_path, tickers)
    df_adj, used_date = adjust_prices_for_mode(df, price_mode)

    # Build returns
    default_mapping = {k: v["name"] for k, v in INDEX_CONFIG.items()}
    perf = _build_returns_table(df_adj, default_mapping)

    # Get returns and sort by YTD descending
    heat_df = perf[["Name", "YTD", "1M", "3M", "6M", "12M"]].dropna().sort_values("YTD", ascending=False).reset_index(drop=True)

    def get_color_class(value: float) -> str:
        """Determine color class based on value magnitude. All values show green or red."""
        val = value / 100  # Convert from percentage to decimal
        # Always green (positive) or red (negative) - no neutral
        prefix = "positive" if val >= 0 else "negative"
        abs_val = abs(val)
        if abs_val <= 0.05:
            level = 1
        elif abs_val <= 0.10:
            level = 2
        elif abs_val <= 0.20:
            level = 3
        elif abs_val <= 0.30:
            level = 4
        else:
            level = 5
        return f"{prefix}-{level}"

    def format_percentage(value: float) -> str:
        """Format value as percentage string."""
        return f"{value:.1f}%"

    # Prepare rows for HTML template
    rows = []
    for _, row in heat_df.iterrows():
        name = row["Name"]

        # Find flag for this name
        flag = ""
        for ticker, config in INDEX_CONFIG.items():
            if config["name"] == name:
                flag = config["flag"]
                break

        rows.append({
            "name": name,
            "flag": flag,
            "ytd_formatted": format_percentage(row["YTD"]),
            "ytd_class": get_color_class(row["YTD"]),
            "m1_formatted": format_percentage(row["1M"]),
            "m1_class": get_color_class(row["1M"]),
            "m3_formatted": format_percentage(row["3M"]),
            "m3_class": get_color_class(row["3M"]),
            "m6_formatted": format_percentage(row["6M"]),
            "m6_class": get_color_class(row["6M"]),
            "m12_formatted": format_percentage(row["12M"]),
            "m12_class": get_color_class(row["12M"]),
        })

    # Generate HTML
    SCALE_FACTOR = 3
    width_px = 2200  # ~19.4cm at 3x
    height_px = 1200  # ~10.6cm at 3x

    template = Template(HISTORICAL_PERFORMANCE_HTML_TEMPLATE)
    html = template.render(
        rows=rows,
        width=width_px,
        height=height_px,
        scale=SCALE_FACTOR,
    )

    # Convert to PNG
    with tempfile.TemporaryDirectory() as tmpdir:
        hti = Html2Image(output_path=tmpdir, size=(width_px, height_px))
        hti.screenshot(html_str=html, save_as="historical_perf.png")

        img_path = Path(tmpdir) / "historical_perf.png"
        with open(img_path, "rb") as f:
            png_bytes = f.read()

    return png_bytes, used_date


###############################################################################
# Colour helper
###############################################################################

def _colour_for_value(val: float, max_pos: float, min_neg: float) -> Tuple[float, float, float, float]:
    """Return an RGBA colour for a single value using independent scales.

    Positive values produce a gradient from white to green; negative
    values from white to red.  Zeros (or undefined scales) return
    white.  The input ``max_pos`` should be the maximum positive value
    in the column, and ``min_neg`` the minimum (most negative) value.
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
    height_cm: Optional[float] = None,
    used_date: Optional[pd.Timestamp],
    price_mode: str,
    source_placeholder_names: List[str],
) -> Presentation:
    """Helper to insert an image and source footnote into a slide identified by placeholders.

    This internal helper searches for shapes whose names match any of
    ``placeholder_names`` or whose text contains any of those names in
    square brackets.  Once found, it uses the slide but ignores the
    shape's dimensions, inserting the provided image at the specified
    coordinates and size.  It also searches for a text box named
    ``source_placeholder_names`` (or containing the pattern) and
    replaces its contents with a source footnote based on ``used_date``
    and ``price_mode`` while preserving the original formatting.  If no
    matching placeholder is found, the image is inserted on the last
    slide.  Footnote insertion is skipped if ``used_date`` is ``None``.

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
    used_date : pandas.Timestamp or None
        The effective date for the source footnote.  If ``None`` the
        source is not inserted.
    price_mode : str
        Either ``"Last Price"`` or ``"Last Close"``, used to suffix
        ``" Close"`` in the footnote.
    source_placeholder_names : list of str
        Names or patterns (without brackets) for the source text box.

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
    stream = io.BytesIO(image_bytes)
    # Only specify width; let height auto-scale to maintain aspect ratio
    if height_cm is not None:
        pic = target_slide.shapes.add_picture(stream, left, top, width=width, height=Cm(height_cm))
    else:
        pic = target_slide.shapes.add_picture(stream, left, top, width=width)

    # Send to back (behind other elements like footnote)
    spTree = target_slide.shapes._spTree
    sp = pic._element
    spTree.remove(sp)
    spTree.insert(2, sp)  # Index 2 = back (0 and 1 are reserved)

    # Insert source footnote if a date is available
    if used_date is not None:
        date_str = used_date.strftime("%d/%m/%Y")
        suffix = " Close" if price_mode.lower() == "last close" else ""
        source_text = f"Source: Bloomberg, Herculis Group, Data as of {date_str}{suffix}"
        # Look for source placeholder
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


def insert_equity_performance_bar_slide(
    prs: Presentation,
    image_bytes: bytes,
    used_date: Optional[pd.Timestamp] = None,
    price_mode: str = "Last Price",
    *,
    left_cm: float = 3.47,
    top_cm: float = 5.28,
    width_cm: float = 17.31,
    height_cm: Optional[float] = 10.0,
) -> Presentation:
    """Insert the weekly performance bar chart and source footnote into its designated slide.

    The slide is identified by a shape named ``equity_perf_1week`` or
    ``equity_perf_1w`` (or containing ``[equity_perf_1week]`` etc.).  The
    chart is inserted at the specified coordinates regardless of the
    placeholder's original size.  A source footnote is written into
    ``equity_1w_source`` or a text box containing ``[equity_1w_source]``
    reflecting the ``used_date`` and ``price_mode``.

    Parameters
    ----------
    prs : Presentation
        The PowerPoint presentation to modify.
    image_bytes : bytes
        PNG data for the weekly bar chart.
    used_date : pandas.Timestamp or None, optional
        Effective date used for performance calculations.  If ``None``,
        the source footnote is not inserted.
    price_mode : str, default ``"Last Price"``
        Either ``"Last Price"`` or ``"Last Close"``.  Determines the
        suffix appended to the date in the source footnote.
    left_cm, top_cm, width_cm, height_cm : float
        Position and size for inserting the chart.

    Returns
    -------
    Presentation
        The modified presentation.
    """
    return _insert_dashboard_to_placeholder(
        prs,
        image_bytes,
        placeholder_names=["equity_perf_1w", "equity_perf_1week"],
        left_cm=left_cm,
        top_cm=top_cm,
        width_cm=width_cm,
        height_cm=height_cm,
        used_date=used_date,
        price_mode=price_mode,
        source_placeholder_names=["equity_1w_source"],
    )


def insert_equity_performance_histo_slide(
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
    """Insert the historical performance heatmap and source footnote into its designated slide.

    The slide is identified by a shape named ``equity_perf_histo`` (or
    containing ``[equity_perf_histo]``).  The heatmap is inserted at the
    specified coordinates.  A source footnote is written into
    ``equity_1w_source2`` or a text box containing ``[equity_1w_source2]``
    based on the provided ``used_date`` and ``price_mode``.

    Parameters
    ----------
    prs : Presentation
        The PowerPoint presentation to modify.
    image_bytes : bytes
        PNG data for the heatmap.
    used_date : pandas.Timestamp or None, optional
        Effective date used for performance calculations.  If ``None``,
        the source footnote is not inserted.
    price_mode : str, default ``"Last Price"``
        Either ``"Last Price"`` or ``"Last Close"``.  Determines the
        suffix appended to the date in the source footnote.
    left_cm, top_cm, width_cm, height_cm : float
        Position and size for inserting the heatmap.

    Returns
    -------
    Presentation
        The modified presentation.
    """
    return _insert_dashboard_to_placeholder(
        prs,
        image_bytes,
        placeholder_names=["equity_perf_histo"],
        left_cm=left_cm,
        top_cm=top_cm,
        width_cm=width_cm,
        height_cm=height_cm,
        used_date=used_date,
        price_mode=price_mode,
        source_placeholder_names=["equity_1w_source2"],
    )


# =============================================================================
# EQUITY YTD EVOLUTION CHART (Chart.js Line Chart)
# =============================================================================

# Index configuration with Bloomberg tickers and colors from directive
EQUITY_YTD_CONFIG = {
    "IBOV Index": {"name": "Ibov", "color": "#9333EA"},
    "MEXBOL Index": {"name": "Mexbol", "color": "#06B6D4"},
    "NKY Index": {"name": "Nikkei 225", "color": "#10B981"},
    "DAX Index": {"name": "Dax", "color": "#DC2626"},
    "SHSZ300 Index": {"name": "CSI 300", "color": "#F97316"},
    "SPX Index": {"name": "S&P 500", "color": "#3B82F6"},
    "SMI Index": {"name": "SMI", "color": "#1B3A5A"},
    "SENSEX Index": {"name": "Sensex", "color": "#22D3EE"},
    "SASEIDX Index": {"name": "TASI", "color": "#9CA3AF"},
}

# Chart dimensions (hardcoded)
EQUITY_YTD_PNG_WIDTH_PX = 2700
EQUITY_YTD_PNG_HEIGHT_PX = 1500
EQUITY_YTD_PPT_WIDTH_CM = 23.0
EQUITY_YTD_PPT_LEFT_CM = 0.5
EQUITY_YTD_PPT_TOP_CM = 4.0
EQUITY_YTD_HTML_SCALE = 3


def _get_ytd_year(data_end_date: pd.Timestamp) -> int:
    """Determine which year to show as YTD based on data end date.

    The YTD year is simply the year of the last data point.
    """
    return data_end_date.year


def _get_chart_title(data_end_date: pd.Timestamp) -> str:
    """Generate chart title with correct year."""
    year = _get_ytd_year(data_end_date)
    return f"YTD {year} Performance of Equity Indices (%)"


def _compute_ytd_series(
    df: pd.DataFrame,
    ticker: str,
) -> Tuple[List[str], List[float]]:
    """Compute weekly YTD performance series for a ticker.

    Samples data to one point per week for clean chart rendering.

    Returns:
        Tuple of (week labels, YTD performance values as cumulative returns)
    """
    if ticker not in df.columns:
        print(f"[DEBUG] {ticker} not in columns")
        return [], []

    # Get the year of the last data point
    last_date = df["Date"].max()
    year = last_date.year

    print(f"[DEBUG] {ticker}: last_date={last_date}, year={year}")

    # Get start of year - use Dec 31 of previous year to get baseline
    start_of_prev_year = pd.Timestamp(year=year-1, month=12, day=31)
    start_of_year = pd.Timestamp(year=year, month=1, day=1)

    # Get baseline price (last price of previous year or first price of current year)
    baseline_df = df[df["Date"] <= start_of_prev_year]
    if len(baseline_df) > 0:
        baseline_price = baseline_df[ticker].iloc[-1]
    else:
        # Fall back to first available price of the year
        year_df = df[df["Date"] >= start_of_year]
        if len(year_df) == 0:
            print(f"[DEBUG] {ticker}: no data for year {year}")
            return [], []
        baseline_price = year_df[ticker].iloc[0]

    if pd.isna(baseline_price) or baseline_price == 0:
        print(f"[DEBUG] {ticker}: invalid baseline price {baseline_price}")
        return [], []

    print(f"[DEBUG] {ticker}: baseline_price={baseline_price}")

    # Filter to current year data and sort by date
    df_year = df[df["Date"] >= start_of_year].copy().sort_values("Date")

    if len(df_year) == 0:
        print(f"[DEBUG] {ticker}: no data for year {year}")
        return [], []

    # Sample to WEEKLY data (one point per week) for clean chart
    # Group by week number and take the last value of each week
    df_year["Week"] = df_year["Date"].dt.isocalendar().week
    df_year["Month"] = df_year["Date"].dt.month

    labels = []
    values = []
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Group by week and get last price of each week
    for week_num in sorted(df_year["Week"].unique()):
        week_data = df_year[df_year["Week"] == week_num]
        if len(week_data) > 0:
            last_row = week_data.iloc[-1]
            price = last_row[ticker]
            month = int(last_row["Month"])
            if not pd.isna(price):
                ytd_return = (price - baseline_price) / baseline_price * 100
                # Use month name as label (will show once per ~4 weeks)
                labels.append(month_names[month - 1])
                values.append(round(ytd_return, 1))

    print(f"[DEBUG] {ticker}: {len(values)} weekly points, first 5={values[:5]}, last 5={values[-5:]}")
    return labels, values


def create_equity_ytd_evolution_chart(
    excel_path: Union[str, pathlib.Path],
    *,
    price_mode: str = "Last Price",
) -> Tuple[bytes, Optional[pd.Timestamp]]:
    """Generate HTML-based Equity YTD Evolution line chart.

    Features:
    - 9 equity indices with smooth lines
    - Dynamic year in title based on data
    - Label collision avoidance at end of lines
    - Emphasized zero line
    - Connector lines for adjusted labels

    Returns:
        Tuple of (PNG bytes, effective date used for computation)
    """
    # Get all tickers from configuration
    tickers = list(EQUITY_YTD_CONFIG.keys())

    # Load and adjust price data
    df = _load_price_data(excel_path, tickers)
    df_adj, used_date = adjust_prices_for_mode(df, price_mode)

    print(f"[DEBUG] Raw data date range: {df['Date'].min()} to {df['Date'].max()}, rows={len(df)}")
    print(f"[DEBUG] Adjusted data date range: {df_adj['Date'].min()} to {df_adj['Date'].max()}, rows={len(df_adj)}")
    print(f"[DEBUG] used_date from adjust_prices_for_mode: {used_date}")

    if used_date is None:
        used_date = df_adj["Date"].max()

    # Get chart title with correct year
    chart_title = _get_chart_title(used_date)

    # Compute YTD series for each index
    all_labels = []
    datasets = []

    for ticker, config in EQUITY_YTD_CONFIG.items():
        labels, values = _compute_ytd_series(df_adj, ticker)
        if labels and values:
            # Track the longest label list
            if len(labels) > len(all_labels):
                all_labels = labels

            datasets.append({
                "label": config["name"],
                "data": values,
                "borderColor": config["color"],
                "backgroundColor": "transparent",
            })

    # Ensure all datasets have the same length (pad with None if needed)
    max_len = len(all_labels)
    for dataset in datasets:
        while len(dataset["data"]) < max_len:
            dataset["data"].append(None)

    # Debug output
    print(f"[DEBUG] chart_title={chart_title}")
    print(f"[DEBUG] all_labels={all_labels} (len={len(all_labels)})")
    for ds in datasets:
        print(f"[DEBUG] {ds['label']}: data={ds['data']} (len={len(ds['data'])})")

    # Render HTML template with proper JSON serialization
    # Create environment with tojson filter
    env = Environment()
    env.filters['tojson'] = json.dumps
    template = env.from_string(EQUITY_YTD_EVOLUTION_HTML_TEMPLATE)
    html_content = template.render(
        width=EQUITY_YTD_PNG_WIDTH_PX,
        height=EQUITY_YTD_PNG_HEIGHT_PX,
        scale=EQUITY_YTD_HTML_SCALE,
        chart_title=chart_title,
        labels=all_labels,
        datasets=datasets,
    )

    # Save HTML for debugging
    with open("equity_ytd_debug.html", "w") as f:
        f.write(html_content)
    print(f"[DEBUG] HTML saved to equity_ytd_debug.html")

    # Convert HTML to PNG
    hti = Html2Image(size=(EQUITY_YTD_PNG_WIDTH_PX, EQUITY_YTD_PNG_HEIGHT_PX))
    hti.screenshot(html_str=html_content, save_as="equity_ytd_temp.png")

    # Read the generated PNG
    with open("equity_ytd_temp.png", "rb") as f:
        png_bytes = f.read()

    # Clean up temp file
    try:
        os.remove("equity_ytd_temp.png")
    except Exception:
        pass

    return png_bytes, used_date


def insert_equity_ytd_evolution_slide(
    prs: Presentation,
    image_bytes: bytes,
    used_date: Optional[pd.Timestamp] = None,
    price_mode: str = "Last Price",
) -> Presentation:
    """Insert the Equity YTD Evolution chart into PowerPoint."""
    if not image_bytes:
        return prs

    # Find target slide by placeholder name
    target_slide = None
    placeholder_names = ["ytd_eq_perf"]
    name_candidates = [n.lower() for n in placeholder_names]
    pattern_candidates = [f"[{n}]" for n in name_candidates]

    for slide in prs.slides:
        for shape in slide.shapes:
            name_attr = getattr(shape, "name", "").lower()
            if name_attr in name_candidates:
                target_slide = slide
                break
            if shape.has_text_frame:
                text_lower = (shape.text or "").strip().lower()
                if text_lower in [p.lower() for p in pattern_candidates]:
                    target_slide = slide
                    break
        if target_slide:
            break

    if target_slide is None:
        print("[Equity YTD Evolution] ERROR: Slide not found")
        return prs

    # Insert chart image with exact hardcoded dimensions
    left = Cm(EQUITY_YTD_PPT_LEFT_CM)
    top = Cm(EQUITY_YTD_PPT_TOP_CM)
    width = Cm(EQUITY_YTD_PPT_WIDTH_CM)

    stream = io.BytesIO(image_bytes)
    picture = target_slide.shapes.add_picture(stream, left, top, width=width)

    # Send picture to back
    spTree = target_slide.shapes._spTree
    pic_element = picture._element
    spTree.remove(pic_element)
    spTree.insert(2, pic_element)

    print(f"[Equity YTD Evolution] Chart inserted at ({EQUITY_YTD_PPT_LEFT_CM}, {EQUITY_YTD_PPT_TOP_CM}) cm")

    # Update source placeholder if date available
    if used_date is not None:
        date_str = used_date.strftime("%d/%m/%Y")
        suffix = " Close" if price_mode.lower() == "last close" else ""
        source_text = f"Source: Bloomberg, Herculis Group, Data as of {date_str}{suffix}"
        source_candidates = ["ytd_eq_perf_source"]
        source_patterns = [f"[{n}]" for n in source_candidates]

        for shape in target_slide.shapes:
            name_attr = getattr(shape, "name", "").lower()
            if name_attr in [n.lower() for n in source_candidates]:
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
                        shape.text_frame.clear()
                        p = shape.text_frame.paragraphs[0]
                        new_run = p.add_run()
                        new_run.text = source_text
                        _apply_font_attrs(new_run, *attrs)
                        break
                else:
                    continue
                break

    return prs
