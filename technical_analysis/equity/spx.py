"""
Utility functions for S&P 500 technical analysis and high‑resolution export.

This module provides tools to build interactive and static charts for the
S&P 500 index, calculate and insert technical and momentum scores into
PowerPoint presentations, generate horizontal and vertical gauges that
visualise the average of the technical and momentum scores, as well as
contextual trading ranges (higher and lower range bounds).  Functions
fall back to sensible defaults when placeholders are not found.

Key functions include:

* ``make_spx_figure`` – interactive Plotly chart for Streamlit.
* ``insert_spx_technical_chart`` – insert a static SPX chart into a PPTX.
* ``insert_spx_technical_score_number`` – insert the technical score (integer).
* ``insert_spx_momentum_score_number`` – insert the momentum score (integer).
* ``insert_spx_subtitle`` – insert a user‑defined subtitle into the SPX slide.
* ``generate_average_gauge_image`` – create a horizontal gauge image.
* ``insert_spx_average_gauge`` – insert the gauge into a PPT slide.
* ``insert_spx_technical_assessment`` – insert a descriptive “view” text.
* ``generate_range_gauge_chart_image`` – create a combined price chart with
  a vertical range gauge on the right hand side, including a horizontal line
  connecting the last price to the gauge.  This function is used by
  ``insert_spx_technical_chart_with_range``.
* ``insert_spx_technical_chart_with_range`` – insert the SPX technical
  analysis chart with the higher/lower range gauge into the PPT.

The range gauge illustrates the recent trading range for the S&P 500.
By default the higher range is the maximum closing price over the last
90 trading days; the lower range is the 50‑day moving average if the
current price is above its 50‑day average, otherwise the minimum
closing price over the same lookback window.  A horizontal line
continues the last price through to the gauge so that the viewer can
quickly assess where the index sits relative to its recent extremes.
"""

from __future__ import annotations

from datetime import timedelta
import pathlib
from typing import Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

from pptx import Presentation
from pptx.util import Cm
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _load_price_data(excel_path: pathlib.Path, ticker: str = "SPX Index") -> pd.DataFrame:
    """Read the raw price sheet and return tidy Date‑Price DataFrame."""
    df = pd.read_excel(excel_path, sheet_name="data_prices")
    df = df.drop(index=0)
    df = df[df[df.columns[0]] != "DATES"]
    df["Date"] = pd.to_datetime(df[df.columns[0]], errors="coerce")
    df["Price"] = pd.to_numeric(df[ticker], errors="coerce")
    return (
        df.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)[
            ["Date", "Price"]
        ]
    )


def _add_mas(df: pd.DataFrame) -> pd.DataFrame:
    """Add 50/100/200‑day moving‑average columns."""
    out = df.copy()
    for w in (50, 100, 200):
        out[f"MA_{w}"] = out["Price"].rolling(w, min_periods=1).mean()
    return out


# ---------------------------------------------------------------------------
# Plotly interactive chart for Streamlit
# ---------------------------------------------------------------------------
def make_spx_figure(
    excel_path: str | pathlib.Path, anchor_date: Optional[pd.Timestamp] = None
) -> go.Figure:
    """
    Build an interactive SPX chart for Streamlit.

    Parameters
    ----------
    excel_path : str or pathlib.Path
        Path to the Excel file containing SPX price data.
    anchor_date : pandas.Timestamp or None
        If provided, a regression channel is drawn from anchor_date to the latest date.

    Returns
    -------
    go.Figure
        A Plotly figure with price, moving averages, Fibonacci lines and optional trend channel.
    """
    excel_path = pathlib.Path(excel_path)
    df_full = _add_mas(_load_price_data(excel_path, "SPX Index"))

    today = df_full["Date"].max().normalize()
    start = today - timedelta(days=365)
    df = df_full[df_full["Date"].between(start, today)].reset_index(drop=True)

    last_price = df["Price"].iloc[-1]
    last_price_str = f"{last_price:,.2f}"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Price"],
            mode="lines",
            name=f"S&P 500 Price (last: {last_price_str})",
            line=dict(color="#153D64", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MA_50"],
            mode="lines",
            name="50‑day MA",
            line=dict(color="#008000", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MA_100"],
            mode="lines",
            name="100‑day MA",
            line=dict(color="#FFA500", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MA_200"],
            mode="lines",
            name="200‑day MA",
            line=dict(color="#FF0000", width=1.5),
        )
    )

    hi, lo = df["Price"].max(), df["Price"].min()
    span = hi - lo
    for lvl in [
        hi,
        hi - 0.236 * span,
        hi - 0.382 * span,
        hi - 0.5 * span,
        hi - 0.618 * span,
        lo,
    ]:
        fig.add_hline(
            y=lvl, line=dict(color="grey", dash="dash", width=1), opacity=0.6
        )

    if anchor_date is not None:
        per = df_full[df_full["Date"].between(anchor_date, today)].copy()
        X = per["Date"].map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
        y_vals = per["Price"].to_numpy()
        model = LinearRegression().fit(X, y_vals)
        trend = model.predict(X)
        resid = y_vals - trend
        upper = trend + resid.max()
        lower = trend + resid.min()

        uptrend = model.coef_[0] > 0
        lineclr = "green" if uptrend else "red"
        fillclr = "rgba(0,150,0,0.25)" if uptrend else "rgba(200,0,0,0.25)"

        fig.add_trace(
            go.Scatter(
                x=per["Date"],
                y=upper,
                mode="lines",
                line=dict(color=lineclr, dash="dash"),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=per["Date"],
                y=lower,
                mode="lines",
                line=dict(color=lineclr, dash="dash"),
                fill="tonexty",
                fillcolor=fillclr,
                showlegend=False,
            )
        )

    fig.update_layout(
        margin=dict(l=30, r=30, t=60, b=40),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        xaxis_title=None,
        yaxis_title=None,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )
    return fig


# ---------------------------------------------------------------------------
# High‑resolution chart export (PNG)
# ---------------------------------------------------------------------------
def _generate_spx_image_from_df(
    df_full: pd.DataFrame,
    anchor_date: Optional[pd.Timestamp],
    width_cm: float = 21.41,
    height_cm: float = 7.53,
) -> bytes:
    """
    Create a high‑resolution (dpi=300) transparent PNG chart from the DataFrame.
    Includes price, moving averages, Fibonacci lines and optional regression channel.
    """
    today = df_full["Date"].max().normalize()
    start = today - timedelta(days=365)
    df = df_full[df_full["Date"].between(start, today)].reset_index(drop=True)

    df_ma = df.copy()
    for w in (50, 100, 200):
        df_ma[f"MA_{w}"] = df_ma["Price"].rolling(w, min_periods=1).mean()

    uptrend = False
    upper = lower = None
    if anchor_date is not None:
        subset = df_full[df_full["Date"].between(anchor_date, today)].copy()
        if not subset.empty:
            X = subset["Date"].map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
            y_vals = subset["Price"].to_numpy()
            model = LinearRegression().fit(X, y_vals)
            trend = model.predict(X)
            resid = y_vals - trend
            uptrend = model.coef_[0] > 0
            upper = trend + resid.max()
            lower = trend + resid.min()

    last_price = df["Price"].iloc[-1]
    last_price_str = f"{last_price:,.2f}"

    fig_width_in, fig_height_in = width_cm / 2.54, height_cm / 2.54
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))

    ax.plot(
        df["Date"],
        df["Price"],
        color="#153D64",
        linewidth=2.5,
        label=f"S&P 500 Price (last: {last_price_str})",
    )
    ax.plot(
        df_ma["Date"],
        df_ma["MA_50"],
        color="#008000",
        linewidth=1.5,
        label="50‑day MA",
    )
    ax.plot(
        df_ma["Date"],
        df_ma["MA_100"],
        color="#FFA500",
        linewidth=1.5,
        label="100‑day MA",
    )
    ax.plot(
        df_ma["Date"],
        df_ma["MA_200"],
        color="#FF0000",
        linewidth=1.5,
        label="200‑day MA",
    )

    hi, lo = df["Price"].max(), df["Price"].min()
    span = hi - lo
    fib_levels = [
        hi,
        hi - 0.236 * span,
        hi - 0.382 * span,
        hi - 0.5 * span,
        hi - 0.618 * span,
        lo,
    ]
    for lvl in fib_levels:
        ax.axhline(y=lvl, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    if anchor_date is not None and upper is not None and lower is not None:
        fill_color = (0, 0.6, 0, 0.25) if uptrend else (0.78, 0, 0, 0.25)
        line_color = "#008000" if uptrend else "#C00000"
        subset = (
            df_full[df_full["Date"].between(anchor_date, today)].copy().reset_index(drop=True)
        )
        ax.plot(subset["Date"], upper, color=line_color, linestyle="--")
        ax.plot(subset["Date"], lower, color=line_color, linestyle="--")
        ax.fill_between(subset["Date"], lower, upper, color=fill_color)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize=8, frameon=False
    )
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------
def _get_spx_technical_score(excel_obj_or_path) -> Optional[float]:
    """
    Retrieve the technical score for SPX from 'data_technical_score' (col A, B).
    Returns None if the sheet or score is unavailable.
    """
    try:
        df = pd.read_excel(excel_obj_or_path, sheet_name="data_technical_score")
    except Exception:
        return None
    df = df.dropna(subset=[df.columns[0], df.columns[1]])
    for _, row in df.iterrows():
        if str(row[df.columns[0]]).strip().upper() == "SPX INDEX":
            try:
                return float(row[df.columns[1]])
            except Exception:
                return None
    return None


def insert_spx_technical_score_number(prs: Presentation, excel_file) -> Presentation:
    """
    Insert the SPX technical score (integer) into a shape named 'tech_score_spx'
    or into any shape containing the placeholder '[XXX]' or 'XXX'.  Original
    formatting (font size, colour, bold, italic) is preserved.
    """
    score = _get_spx_technical_score(excel_file)
    score_text = "N/A" if score is None else f"{int(round(float(score)))}"

    placeholder_name = "tech_score_spx"
    placeholder_patterns = ["[XXX]", "XXX"]

    for slide in prs.slides:
        for shape in slide.shapes:
            if getattr(shape, "name", "").lower() == placeholder_name:
                if shape.has_text_frame:
                    runs = shape.text_frame.paragraphs[0].runs
                    saved_size = runs[0].font.size if runs else None
                    saved_color = runs[0].font.color.rgb if runs else None
                    saved_bold = runs[0].font.bold if runs else None
                    saved_italic = runs[0].font.italic if runs else None
                    shape.text_frame.clear()
                    p = shape.text_frame.paragraphs[0]
                    new_run = p.add_run()
                    new_run.text = score_text
                    if saved_size:
                        new_run.font.size = saved_size
                    if saved_color:
                        new_run.font.color.rgb = saved_color
                    if saved_bold is not None:
                        new_run.font.bold = saved_bold
                    if saved_italic is not None:
                        new_run.font.italic = saved_italic
                return prs
            if shape.has_text_frame:
                for pattern in placeholder_patterns:
                    if pattern in shape.text:
                        runs = shape.text_frame.paragraphs[0].runs
                        saved_size = runs[0].font.size if runs else None
                        saved_color = runs[0].font.color.rgb if runs else None
                        saved_bold = runs[0].font.bold if runs else None
                        saved_italic = runs[0].font.italic if runs else None
                        new_text = shape.text.replace(pattern, score_text)
                        shape.text_frame.clear()
                        p = shape.text_frame.paragraphs[0]
                        new_run = p.add_run()
                        new_run.text = new_text
                        if saved_size:
                            new_run.font.size = saved_size
                        if saved_color:
                            new_run.font.color.rgb = saved_color
                        if saved_bold is not None:
                            new_run.font.bold = saved_bold
                        if saved_italic is not None:
                            new_run.font.italic = saved_italic
                        return prs
    return prs

# ---------------------------------------------------------------------------
# Call‑out range helpers and insertion
# ---------------------------------------------------------------------------
def generate_range_callout_chart_image(
    df_full: pd.DataFrame,
    anchor_date: Optional[pd.Timestamp] = None,
    lookback_days: int = 90,
    width_cm: float = 21.41,
    height_cm: float = 7.53,
    callout_width_cm: float = 3.5,
) -> bytes:
    """
    Create a PNG image of the SPX price chart with a textual call‑out on the
    right summarising the recent trading range.  The call‑out lists the
    higher and lower range values (with ±% changes relative to the last
    price) and draws small coloured markers aligned with those levels on
    the y‑axis.  This design preserves the full chart width and avoids
    overlapping the price plot with additional graphics.

    Parameters
    ----------
    df_full : pandas.DataFrame
        Full SPX price history with 'Date' and 'Price' columns.
    anchor_date : pandas.Timestamp or None, optional
        Optional anchor date for a regression channel; if provided, the
        channel is drawn on the price chart.
    lookback_days : int, default 90
        The lookback window for computing the high and low bounds.
    width_cm : float, default 21.41
        Overall width of the output image in centimetres.  This should
        correspond to the template placeholder width.
    height_cm : float, default 7.53
        Height of the output image in centimetres.
    callout_width_cm : float, default 3.5
        Width of the call‑out area on the right where the range summary
        appears.  The remaining width is used for the chart.

    Returns
    -------
    bytes
        PNG image bytes with transparency.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if df_full.empty:
        return b""

    # Restrict to the last year of data for plotting
    today = df_full["Date"].max().normalize()
    start = today - timedelta(days=365)
    df = df_full[df_full["Date"].between(start, today)].reset_index(drop=True)

    # Calculate moving averages on the 1‑year subset
    df_ma = _add_mas(df)

    # Optional regression channel
    uptrend = False
    upper_channel = lower_channel = None
    if anchor_date is not None:
        subset_full = df_full[df_full["Date"].between(anchor_date, today)].copy()
        if not subset_full.empty:
            X = subset_full["Date"].map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
            y_vals = subset_full["Price"].to_numpy()
            model = LinearRegression().fit(X, y_vals)
            trend = model.predict(X)
            resid = y_vals - trend
            uptrend = model.coef_[0] > 0
            upper_channel = trend + resid.max()
            lower_channel = trend + resid.min()

    # Compute high/low bounds and current price
    upper_bound, lower_bound = _compute_range_bounds(df_full, lookback_days=lookback_days)
    last_price = df["Price"].iloc[-1]
    # Format difference percentages
    up_pct = (upper_bound - last_price) / last_price * 100 if last_price else 0.0
    down_pct = (last_price - lower_bound) / last_price * 100 if last_price else 0.0

    # Determine y‑axis limits: ensure the axis includes the entire trading
    # range and the observed price range.  We add a small margin so the
    # labels and markers do not overlap the top or bottom edges.
    hi = df["Price"].max()
    lo = df["Price"].min()
    y_max = max(hi, upper_bound) * 1.02
    y_min = min(lo, lower_bound) * 0.98

    # Compute widths for chart and call‑out.  Reserve a small margin on the
    # left of the chart to ensure that y‑axis tick labels and the legend
    # remain visible when the image is inserted into a PowerPoint slide.
    callout_width_cm = min(callout_width_cm, width_cm)
    chart_width_cm = max(width_cm - callout_width_cm, 0.0)
    fig_w_in, fig_h_in = width_cm / 2.54, height_cm / 2.54
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    # Relative widths as fractions of the full figure width
    chart_rel_width = chart_width_cm / width_cm if width_cm > 0 else 0.0
    callout_rel_width = callout_width_cm / width_cm if width_cm > 0 else 0.0

    # Define a margin fraction for the left side of the chart.  Without
    # this margin, tick labels and the legend can be clipped when the
    # combined image is saved at high DPI.  Use up to 4% of the figure
    # width or 10% of the chart portion, whichever is smaller.
    margin_rel = min(0.04, 0.10 * chart_rel_width)

    # Axes for chart and call‑out; share the y‑axis so that the call‑out
    # markers align with the same price levels as the chart.  The chart
    # occupies the left portion of the figure starting at margin_rel; the
    # call‑out uses the remaining width starting at chart_rel_width.
    ax_chart = fig.add_axes([margin_rel, 0.0, chart_rel_width - margin_rel, 1.0])
    # Create a separate y‑axis for the call‑out so that hiding its ticks
    # does not remove the ticks from the main chart.  We will manually
    # synchronise the y‑limits below.
    ax_callout = fig.add_axes([chart_rel_width, 0.0, callout_rel_width, 1.0])

    # Set y‑limits before plotting so that shared axes align properly
    ax_chart.set_ylim(y_min, y_max)
    ax_callout.set_ylim(y_min, y_max)

    # Plot price and moving averages on the main chart
    ax_chart.plot(df["Date"], df["Price"], color="#153D64", linewidth=2.5,
                  label=f"S&P 500 Price (last: {last_price:,.2f})")
    ax_chart.plot(df_ma["Date"], df_ma["MA_50"], color="#008000", linewidth=1.5, label="50‑day MA")
    ax_chart.plot(df_ma["Date"], df_ma["MA_100"], color="#FFA500", linewidth=1.5, label="100‑day MA")
    ax_chart.plot(df_ma["Date"], df_ma["MA_200"], color="#FF0000", linewidth=1.5, label="200‑day MA")
    # Fibonacci levels on the subset
    sub_hi, sub_lo = df["Price"].max(), df["Price"].min()
    sub_span = sub_hi - sub_lo
    for lvl in [sub_hi, sub_hi - 0.236 * sub_span, sub_hi - 0.382 * sub_span,
                sub_hi - 0.5 * sub_span, sub_hi - 0.618 * sub_span, sub_lo]:
        ax_chart.axhline(lvl, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    # Regression channel shading
    if anchor_date is not None and upper_channel is not None and lower_channel is not None:
        subset = df_full[df_full["Date"].between(anchor_date, today)].copy().reset_index(drop=True)
        fill_color = (0, 0.6, 0, 0.25) if uptrend else (0.78, 0, 0, 0.25)
        line_color = "#008000" if uptrend else "#C00000"
        ax_chart.plot(subset["Date"], upper_channel, color=line_color, linestyle="--")
        ax_chart.plot(subset["Date"], lower_channel, color=line_color, linestyle="--")
        ax_chart.fill_between(subset["Date"], lower_channel, upper_channel, color=fill_color)

    # Style the main chart: remove spines and configure ticks.  We set the
    # y‑axis tick length to zero so that the small horizontal tick marks
    # next to the axis labels are not visible, while keeping the labels
    # themselves.  The x‑axis ticks retain their default length.
    for spine in ax_chart.spines.values():
        spine.set_visible(False)
    ax_chart.tick_params(axis="y", which="both", length=0)
    ax_chart.tick_params(axis="x", which="both", length=2)
    ax_chart.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    # Legend: place the legend just above the main chart, aligned to the
    # left so that it does not overlap the call‑out panel.  Use a
    # multi‑column layout to fit all entries on a single line.  The
    # bounding box is anchored slightly above the axes (y=1.05).
    ax_chart.legend(loc="upper left", bbox_to_anchor=(0.0, 1.05), ncol=4,
                    fontsize=8, frameon=False)

    # Configure call‑out axis: remove ticks and spines; set background white
    ax_callout.set_xlim(0, 1)
    ax_callout.set_xticks([])
    ax_callout.set_yticks([])
    for spine in ax_callout.spines.values():
        spine.set_visible(False)
    ax_callout.set_facecolor("white")

    # Determine x positions for markers and text in relative coordinates
    marker_start_x = 0.02
    marker_end_x = 0.08
    text_x = 0.15

    # Draw small horizontal bars as markers aligned with the high/low bounds
    ax_callout.hlines(upper_bound, xmin=marker_start_x, xmax=marker_end_x,
                      colors="#009951", linewidth=2, transform=ax_callout.transData)
    ax_callout.hlines(lower_bound, xmin=marker_start_x, xmax=marker_end_x,
                      colors="#C00000", linewidth=2, transform=ax_callout.transData)

    # Helper to format values with apostrophes for thousands separators
    def _fmt(val: float) -> str:
        try:
            return f"{val:,.0f}".replace(",", "'")
        except Exception:
            return f"{val:.0f}"

    # Compose label strings with percentage differences.  Use bold weight
    # for the label and value to emphasise the trading range; the
    # percentage change is left with normal weight.
    upper_text = f"Higher Range\n{_fmt(upper_bound)} $\n(+{up_pct:.1f}%)"
    lower_text = f"Lower Range\n{_fmt(lower_bound)} $\n(-{down_pct:.1f}%)"

    # Add the text labels at the appropriate y positions.  Set
    # fontweight='bold' so that the label and value stand out.
    ax_callout.text(text_x, upper_bound, upper_text, color="#009951",
                    ha="left", va="center", fontsize=8, fontweight='bold',
                    transform=ax_callout.transData)
    ax_callout.text(text_x, lower_bound, lower_text, color="#C00000",
                    ha="left", va="center", fontsize=8, fontweight='bold',
                    transform=ax_callout.transData)

    # Export to transparent PNG.  Use bbox_inches='tight' so that the
    # entire figure (including legends and tick labels) is saved without
    # cropping.  A small padding is added to provide breathing room.
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, transparent=True,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def insert_spx_technical_chart_with_callout(
    prs: Presentation,
    excel_file,
    anchor_date: Optional[pd.Timestamp] = None,
    lookback_days: int = 90,
) -> Presentation:
    """
    Insert the SPX technical analysis chart with the trading range call‑out
    into the PowerPoint.  This function mirrors the behaviour of
    ``insert_spx_technical_chart_with_range`` but uses the call‑out style to
    display the high and low bounds instead of a vertical gauge.

    The image is placed at the fixed coordinates (0.93 cm left, 4.40 cm top)
    with dimensions 21.41 cm wide by 7.53 cm high, matching the template.

    Parameters
    ----------
    prs : Presentation
        The PowerPoint presentation to modify.
    excel_file : file‑like object or path
        Excel workbook containing SPX price data.
    anchor_date : pandas.Timestamp or None, optional
        Optional anchor date for a regression channel.
    lookback_days : int, default 90
        Lookback window for computing the high/low range.

    Returns
    -------
    Presentation
        The presentation with the updated slide.
    """
    # Load the price data from the Excel file
    try:
        df_full = _load_price_data_from_obj(excel_file, "SPX Index")
    except Exception:
        df_full = _load_price_data(pathlib.Path(excel_file), "SPX Index")

    # Generate the image with the call‑out.  Use an extended width of
    # 24.47 cm (matching the user’s requested dimensions) while keeping
    # the height at 7.53 cm.  The call‑out width is left at its default
    # value unless overridden.
    img_bytes = generate_range_callout_chart_image(
        df_full,
        anchor_date=anchor_date,
        lookback_days=lookback_days,
        width_cm=25.0,
        height_cm=7.3,
    )

    # Locate the slide containing the 'spx' placeholder or text
    target_slide = None
    for slide in prs.slides:
        for shape in slide.shapes:
            name_attr = getattr(shape, "name", "").lower()
            if name_attr == "spx":
                target_slide = slide
                break
            if shape.has_text_frame:
                if (shape.text or "").strip().lower() == "[spx]":
                    target_slide = slide
                    break
        if target_slide:
            break
    if target_slide is None:
        target_slide = prs.slides[min(11, len(prs.slides) - 1)]

    # Insert the image at the requested coordinates.  The user specified
    # dimensions of 25 cm wide and 7.3 cm high, positioned 0.93 cm
    # from the left and 4.80 cm from the top.
    left = Cm(0.93)
    top = Cm(4.80)
    width = Cm(25.0)
    height = Cm(7.3)
    stream = BytesIO(img_bytes)
    target_slide.shapes.add_picture(stream, left, top, width=width, height=height)
    return prs


def _get_spx_momentum_score(excel_obj_or_path) -> Optional[float]:
    """
    Retrieve the momentum score for SPX from 'data_trend_rating' (col A, D).
    Returns None if unavailable.
    """
    try:
        df = pd.read_excel(excel_obj_or_path, sheet_name="data_trend_rating")
    except Exception:
        return None
    df = df.dropna(subset=[df.columns[0], df.columns[3]])
    for _, row in df.iterrows():
        ticker = str(row[df.columns[0]]).strip().upper()
        if ticker == "SPX INDEX":
            try:
                return float(row[df.columns[3]])
            except Exception:
                return None
    return None


def insert_spx_momentum_score_number(prs: Presentation, excel_file) -> Presentation:
    """
    Insert the SPX momentum score (integer) into a shape named 'mom_score_spx'
    or into any shape containing '[XXX]' or 'XXX'.  Formatting is preserved.
    """
    score = _get_spx_momentum_score(excel_file)
    score_text = "N/A" if score is None else f"{int(round(float(score)))}"

    placeholder_name = "mom_score_spx"
    placeholder_patterns = ["[XXX]", "XXX"]

    for slide in prs.slides:
        for shape in slide.shapes:
            if getattr(shape, "name", "").lower() == placeholder_name:
                if shape.has_text_frame:
                    runs = shape.text_frame.paragraphs[0].runs
                    saved_size = runs[0].font.size if runs else None
                    saved_color = runs[0].font.color.rgb if runs else None
                    saved_bold = runs[0].font.bold if runs else None
                    saved_italic = runs[0].font.italic if runs else None
                    shape.text_frame.clear()
                    p = shape.text_frame.paragraphs[0]
                    new_run = p.add_run()
                    new_run.text = score_text
                    if saved_size:
                        new_run.font.size = saved_size
                    if saved_color:
                        new_run.font.color.rgb = saved_color
                    if saved_bold is not None:
                        new_run.font.bold = saved_bold
                    if saved_italic is not None:
                        new_run.font.italic = saved_italic
                return prs
            if shape.has_text_frame:
                for pattern in placeholder_patterns:
                    if pattern in shape.text:
                        runs = shape.text_frame.paragraphs[0].runs
                        saved_size = runs[0].font.size if runs else None
                        saved_color = runs[0].font.color.rgb if runs else None
                        saved_bold = runs[0].font.bold if runs else None
                        saved_italic = runs[0].font.italic if runs else None
                        new_text = shape.text.replace(pattern, score_text)
                        shape.text_frame.clear()
                        p = shape.text_frame.paragraphs[0]
                        new_run = p.add_run()
                        new_run.text = new_text
                        if saved_size:
                            new_run.font.size = saved_size
                        if saved_color:
                            new_run.font.color.rgb = saved_color
                        if saved_bold is not None:
                            new_run.font.bold = saved_bold
                        if saved_italic is not None:
                            new_run.font.italic = saved_italic
                        return prs
    return prs


# ---------------------------------------------------------------------------
# Chart insertion
# ---------------------------------------------------------------------------
def insert_spx_technical_chart(
    prs: Presentation, excel_file, anchor_date: Optional[pd.Timestamp] = None
) -> Presentation:
    """
    Insert the SPX technical‑analysis chart into the PPT.

    We only use the textbox named ``spx`` (or containing “[spx]”) to locate
    the correct slide; the chart itself is always pasted at the fixed
    coordinates (0.93 cm left, 4.39 cm top, 21.41 cm wide, 7.53 cm high).
    """
    # Load data and generate image
    try:
        df_full = _load_price_data_from_obj(excel_file, "SPX Index")
    except Exception:
        df_full = _load_price_data(pathlib.Path(excel_file), "SPX Index")
    img_bytes = _generate_spx_image_from_df(df_full, anchor_date)

    # Find the slide containing the 'spx' placeholder
    target_slide = None
    for slide in prs.slides:
        for shape in slide.shapes:
            name_attr = getattr(shape, "name", "").lower()
            if name_attr == "spx":
                target_slide = slide
                break
            if shape.has_text_frame:
                if (shape.text or "").strip().lower() == "[spx]":
                    target_slide = slide
                    break
        if target_slide:
            break

    if target_slide is None:
        target_slide = prs.slides[min(11, len(prs.slides) - 1)]

    # Always paste the chart at fixed coordinates
    left = Cm(0.93)
    top = Cm(4.39)
    width = Cm(21.41)
    height = Cm(7.53)
    stream = BytesIO(img_bytes)
    target_slide.shapes.add_picture(stream, left, top, width=width, height=height)
    return prs


# ---------------------------------------------------------------------------
# Subtitle insertion
# ---------------------------------------------------------------------------
def insert_spx_subtitle(prs: Presentation, subtitle: str) -> Presentation:
    """
    Replace the placeholder ('XXX' or '[XXX]') in a textbox named 'spx_text'
    (or containing those patterns) with the provided subtitle, preserving
    original formatting.
    """
    placeholder_name = "spx_text"
    placeholder_patterns = ["[XXX]", "XXX"]

    subtitle_text = subtitle or ""

    for slide in prs.slides:
        for shape in slide.shapes:
            if getattr(shape, "name", "").lower() == placeholder_name:
                if shape.has_text_frame:
                    runs = shape.text_frame.paragraphs[0].runs
                    saved_size = runs[0].font.size if runs else None
                    saved_color = runs[0].font.color.rgb if runs else None
                    saved_bold = runs[0].font.bold if runs else None
                    saved_italic = runs[0].font.italic if runs else None
                    shape.text_frame.clear()
                    p = shape.text_frame.paragraphs[0]
                    new_run = p.add_run()
                    new_run.text = subtitle_text
                    if saved_size:
                        new_run.font.size = saved_size
                    if saved_color:
                        new_run.font.color.rgb = saved_color
                    if saved_bold is not None:
                        new_run.font.bold = saved_bold
                    if saved_italic is not None:
                        new_run.font.italic = saved_italic
                return prs
            if shape.has_text_frame:
                for pattern in placeholder_patterns:
                    if pattern in shape.text:
                        runs = shape.text_frame.paragraphs[0].runs
                        saved_size = runs[0].font.size if runs else None
                        saved_color = runs[0].font.color.rgb if runs else None
                        saved_bold = runs[0].font.bold if runs else None
                        saved_italic = runs[0].font.italic if runs else None
                        new_text = shape.text.replace(pattern, subtitle_text)
                        shape.text_frame.clear()
                        p = shape.text_frame.paragraphs[0]
                        new_run = p.add_run()
                        new_run.text = new_text
                        if saved_size:
                            new_run.font.size = saved_size
                        if saved_color:
                            new_run.font.color.rgb = saved_color
                        if saved_bold is not None:
                            new_run.font.bold = saved_bold
                        if saved_italic is not None:
                            new_run.font.italic = saved_italic
                        return prs
    return prs


# ---------------------------------------------------------------------------
# Colour interpolation for gauge
# ---------------------------------------------------------------------------
def _interpolate_color(value: float) -> Tuple[float, float, float]:
    """
    Interpolate from red→yellow→green for a 0–100 value.  Pure red at 0,
    bright yellow at 40 and rich green at 70.
    """
    red = (1.0, 0.0, 0.0)
    yellow = (1.0, 204 / 255, 0.0)
    green = (0.0, 153 / 255, 81 / 255)
    if value <= 40:
        t = value / 40.0
        return tuple(red[i] + t * (yellow[i] - red[i]) for i in range(3))
    elif value <= 70:
        t = (value - 40) / 30.0
        return tuple(yellow[i] + t * (green[i] - yellow[i]) for i in range(3))
    return green


def generate_average_gauge_image(
    tech_score: float,
    mom_score: float,
    last_week_avg: float,
    date_text: str | None = None,
    last_label_text: str = "Last Week",
    width_cm: float = 15.15,
    height_cm: float = 3.13,
) -> bytes:
    """
    Create a horizontal gauge with a red→yellow→green gradient, marking the
    average of technical and momentum scores against last week’s average.
    """
    def clamp100(x: float) -> float:
        return max(0.0, min(100.0, float(x)))

    curr = (clamp100(tech_score) + clamp100(mom_score)) / 2.0
    prev = clamp100(last_week_avg)

    cmap = LinearSegmentedColormap.from_list(
        "gauge_gradient", ["#FF0000", "#FFCC00", "#009951"], N=256
    )

    fig_w, fig_h = width_cm / 2.54, height_cm / 2.54
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    gradient = np.linspace(0, 1, 500).reshape(1, -1)
    bar_thickness = 0.4
    bar_bottom_y = -bar_thickness / 2.0
    bar_top_y = bar_thickness / 2.0
    ax.imshow(
        gradient,
        extent=[0, 100, bar_bottom_y, bar_top_y],
        aspect="auto",
        cmap=cmap,
        origin="lower",
    )

    # Marker dimensions and spacing
    marker_width = 3.0
    marker_height = 0.15
    gap = 0.10
    number_space = 0.25
    top_label_offset = 0.40
    bottom_label_offset = 0.40

    # Y positions for current (top) marker and labels
    top_apex_y = bar_top_y + gap
    top_base_y = top_apex_y + marker_height
    top_number_y = top_base_y + number_space
    top_label_y = top_number_y + top_label_offset

    # Y positions for previous (bottom) marker and labels
    bottom_apex_y = bar_bottom_y - gap
    bottom_base_y = bottom_apex_y - marker_height
    bottom_number_y = bottom_base_y - number_space
    bottom_label_y = bottom_number_y - bottom_label_offset

    curr_colour = _interpolate_color(curr)
    prev_colour = _interpolate_color(prev)

    # Draw triangles and numbers
    ax.add_patch(
        patches.Polygon(
            [
                (curr - marker_width / 2, top_base_y),
                (curr + marker_width / 2, top_base_y),
                (curr, top_apex_y),
            ],
            color=curr_colour,
        )
    )
    ax.add_patch(
        patches.Polygon(
            [
                (prev - marker_width / 2, bottom_base_y),
                (prev + marker_width / 2, bottom_base_y),
                (prev, bottom_apex_y),
            ],
            color=prev_colour,
        )
    )
    ax.text(
        curr,
        top_number_y,
        f"{curr:.0f}",
        color=curr_colour,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
    )
    ax.text(
        prev,
        bottom_number_y,
        f"{prev:.0f}",
        color=prev_colour,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
    )

    if date_text:
        ax.text(
            curr,
            top_label_y,
            date_text,
            color="#0063B0",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
        )
    ax.text(
        prev,
        bottom_label_y,
        last_label_text,
        color="#133C74",
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(bottom_label_y - 0.35, top_label_y + 0.35)
    ax.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Helpers for reading Excel from a file-like object
# ---------------------------------------------------------------------------
def _load_price_data_from_obj(excel_obj, ticker: str = "SPX Index") -> pd.DataFrame:
    df = pd.read_excel(excel_obj, sheet_name="data_prices")
    df = df.drop(index=0)
    df = df[df[df.columns[0]] != "DATES"]
    df["Date"] = pd.to_datetime(df[df.columns[0]], errors="coerce")
    df["Price"] = pd.to_numeric(df[ticker], errors="coerce")
    return (
        df.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)[
            ["Date", "Price"]
        ]
    )


# ---------------------------------------------------------------------------
# Gauge insertion
# ---------------------------------------------------------------------------
def insert_spx_average_gauge(
    prs: Presentation, excel_file, last_week_avg: float
) -> Presentation:
    """
    Insert the average gauge into the SPX slide.  Looks for a shape named
    'gauge_spx' or text containing '[GAUGE]', 'GAUGE', or 'gauge_spx'.  If found,
    the gauge uses that position; otherwise it is inserted below the chart at
    the default coordinates (8.97 cm left, 12.13 cm top, 15.15 cm wide, 3.13 cm high).
    """
    tech_score = _get_spx_technical_score(excel_file)
    mom_score = _get_spx_momentum_score(excel_file)
    if tech_score is None or mom_score is None:
        return prs

    try:
        gauge_bytes = generate_average_gauge_image(
            tech_score,
            mom_score,
            last_week_avg,
            date_text="Last",
            last_label_text="Previous Week",
            width_cm=15.15,
            height_cm=3.13,
        )
    except Exception:
        return prs

    placeholder_name = "gauge_spx"
    placeholder_patterns = ["[GAUGE]", "GAUGE", "gauge_spx"]

    for slide in prs.slides:
        for shape in slide.shapes:
            if getattr(shape, "name", "").lower() == placeholder_name:
                left, top, width, height = (
                    shape.left,
                    shape.top,
                    shape.width,
                    shape.height,
                )
                if shape.has_text_frame:
                    shape.text = ""
                stream = BytesIO(gauge_bytes)
                slide.shapes.add_picture(
                    stream, left, top, width=width, height=height
                )
                return prs
            if shape.has_text_frame:
                for pattern in placeholder_patterns:
                    if pattern.lower() in shape.text.lower():
                        left, top, width, height = (
                            shape.left,
                            shape.top,
                            shape.width,
                            shape.height,
                        )
                        shape.text = shape.text.replace(pattern, "")
                        stream = BytesIO(gauge_bytes)
                        slide.shapes.add_picture(
                            stream, left, top, width=width, height=height
                        )
                        return prs

    # Fallback: fixed coordinates below the chart
    idx = min(11, len(prs.slides) - 1)
    slide = prs.slides[idx]
    left = Cm(8.97)
    top = Cm(12.13)
    width = Cm(15.15)
    height = Cm(3.13)
    stream = BytesIO(gauge_bytes)
    slide.shapes.add_picture(stream, left, top, width=width, height=height)
    return prs


# ---------------------------------------------------------------------------
# Technical assessment insertion
# ---------------------------------------------------------------------------
def insert_spx_technical_assessment(prs: Presentation, excel_file) -> Presentation:
    """
    Insert a descriptive assessment text into a shape named 'spx_view'
    (or containing '[spx_view]'), based on the average of technical and
    momentum scores.  The assessment is:
      ≥80: Strongly Bullish
      70–79.99: Bullish
      60–69.99: Slightly Bullish
      40–59.99: Neutral
      30–39.99: Slightly Bearish
      20–29.99: Bearish
      <20: Strongly Bearish.
    """
    tech_score = _get_spx_technical_score(excel_file)
    mom_score = _get_spx_momentum_score(excel_file)
    if tech_score is None or mom_score is None:
        return prs

    avg = (float(tech_score) + float(mom_score)) / 2.0

    if avg >= 80:
        desc = "S&P 500: Strongly Bullish"
    elif avg >= 70:
        desc = "S&P 500: Bullish"
    elif avg >= 60:
        desc = "S&P 500: Slightly Bullish"
    elif avg >= 40:
        desc = "S&P 500: Neutral"
    elif avg >= 30:
        desc = "S&P 500: Slightly Bearish"
    elif avg >= 20:
        desc = "S&P 500: Bearish"
    else:
        desc = "S&P 500: Strongly Bearish"

    target_name = "spx_view"
    placeholder_patterns = ["[spx_view]", "spx_view"]

    for slide in prs.slides:
        for shape in slide.shapes:
            name_attr = getattr(shape, "name", "")
            if name_attr and name_attr.lower() == target_name:
                if shape.has_text_frame:
                    runs = shape.text_frame.paragraphs[0].runs
                    saved_size = runs[0].font.size if runs else None
                    saved_color = runs[0].font.color.rgb if runs else None
                    saved_bold = runs[0].font.bold if runs else None
                    saved_italic = runs[0].font.italic if runs else None
                    shape.text_frame.clear()
                    p = shape.text_frame.paragraphs[0]
                    new_run = p.add_run()
                    new_run.text = desc
                    if saved_size:
                        new_run.font.size = saved_size
                    if saved_color:
                        new_run.font.color.rgb = saved_color
                    if saved_bold is not None:
                        new_run.font.bold = saved_bold
                    if saved_italic is not None:
                        new_run.font.italic = saved_italic
                return prs
            if shape.has_text_frame:
                for pattern in placeholder_patterns:
                    if pattern.lower() in shape.text.lower():
                        runs = shape.text_frame.paragraphs[0].runs
                        saved_size = runs[0].font.size if runs else None
                        saved_color = runs[0].font.color.rgb if runs else None
                        saved_bold = runs[0].font.bold if runs else None
                        saved_italic = runs[0].font.italic if runs else None
                        new_text = shape.text
                        try:
                            new_text = new_text.replace(pattern, desc)
                        except Exception:
                            new_text = desc
                        shape.text_frame.clear()
                        p = shape.text_frame.paragraphs[0]
                        new_run = p.add_run()
                        new_run.text = new_text
                        if saved_size:
                            new_run.font.size = saved_size
                        if saved_color:
                            new_run.font.color.rgb = saved_color
                        if saved_bold is not None:
                            new_run.font.bold = saved_bold
                        if saved_italic is not None:
                            new_run.font.italic = saved_italic
                        return prs
    return prs


# ---------------------------------------------------------------------------
# Range gauge helpers and insertion
# ---------------------------------------------------------------------------
def _compute_range_bounds(
    df_full: pd.DataFrame, lookback_days: int = 90
) -> Tuple[float, float]:
    """
    Compute the higher and lower range bounds for the S&P 500.

    The higher range is defined as the maximum closing price over the
    ``lookback_days`` window ending at the latest date in the dataset.
    The lower range is selected as follows:

      * If the current price is above its 50‑day moving average, use that
        moving average as the lower bound.
      * Otherwise, use the minimum closing price over the same window as
        the lower bound.  This ensures that the lower bound always lies
        below the current price.

    Parameters
    ----------
    df_full : pandas.DataFrame
        DataFrame containing at least 'Date' and 'Price' columns, sorted by
        date ascending.
    lookback_days : int, default 90
        Number of trading days to look back when computing high/low range.

    Returns
    -------
    Tuple[float, float]
        A two‑tuple (upper_bound, lower_bound) representing the recent high
        and support levels.
    """
    if df_full.empty:
        return (np.nan, np.nan)

    today = df_full["Date"].max().normalize()
    window_start = today - timedelta(days=lookback_days)
    window_data = df_full[df_full["Date"].between(window_start, today)]
    if window_data.empty:
        window_data = df_full

    current_price = df_full["Price"].iloc[-1]
    # compute 50‑day moving average for entire series
    ma_50_series = df_full["Price"].rolling(50, min_periods=1).mean()
    ma_50 = ma_50_series.iloc[-1]

    # Determine the highest closing price in the lookback window
    upper_bound = window_data["Price"].max()
    # Lower bound based on current price vs MA_50
    if current_price >= ma_50:
        lower_bound = ma_50
    else:
        lower_bound = window_data["Price"].min()
    # Ensure the upper bound is above the current price.  If the highest
    # price in the lookback window equals or lies below the last price, we
    # extend the upper bound so that it always exceeds the last price.  A
    # conservative uplift of 2 % of the current price is applied.  This
    # prevents the “Higher Range” from coinciding with the current price,
    # especially when momentum is bullish.
    if upper_bound <= current_price:
        upper_bound = current_price * 1.02
    return (float(upper_bound), float(lower_bound))


def generate_range_gauge_chart_image(
    df_full: pd.DataFrame,
    anchor_date: Optional[pd.Timestamp] = None,
    lookback_days: int = 90,
    width_cm: float = 21.41,
    height_cm: float = 7.53,
    chart_width_cm: float = None,
    gauge_width_cm: float = 4.0,
) -> bytes:
    """
    Create a PNG image of the SPX price chart with a vertical range gauge
    appended on the right.  The gauge shows a green–to–red gradient between
    recent high and support levels, with labels for the upper and lower
    bounds.  A horizontal line continues the last price into the gauge so
    that viewers can assess relative positioning.  This function is used by
    ``insert_spx_technical_chart_with_range``.

    Parameters
    ----------
    df_full : pandas.DataFrame
        Full SPX price history as returned by ``_load_price_data``.
    anchor_date : pandas.Timestamp or None, optional
        Optional anchor date for the regression channel.  If ``None`` no
        channel will be drawn.
    lookback_days : int, default 90
        Number of trading days to look back when computing high/low range.
    width_cm : float, default 21.41
        Width of the output image in centimetres.  This should match the
        template placeholder size in PowerPoint.
    height_cm : float, default 7.53
        Height of the output image in centimetres.

    Returns
    -------
    bytes
        A byte array containing the PNG image data with transparency.
    """
    if df_full.empty:
        return b""

    # Compute bounds for the last year of data
    today = df_full["Date"].max().normalize()
    start = today - timedelta(days=365)
    df = df_full[df_full["Date"].between(start, today)].reset_index(drop=True)
    df_ma = _add_mas(df)

    # Regression channel (optional)
    uptrend = False
    upper_channel = lower_channel = None
    if anchor_date is not None:
        subset_full = df_full[df_full["Date"].between(anchor_date, today)].copy()
        if not subset_full.empty:
            X = subset_full["Date"].map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
            y_vals = subset_full["Price"].to_numpy()
            model = LinearRegression().fit(X, y_vals)
            trend = model.predict(X)
            resid = y_vals - trend
            uptrend = model.coef_[0] > 0
            upper_channel = trend + resid.max()
            lower_channel = trend + resid.min()

    # Determine recent high and support levels
    upper_bound, lower_bound = _compute_range_bounds(df_full, lookback_days=lookback_days)
    last_price = df["Price"].iloc[-1]
    last_price_str = f"{last_price:,.2f}"

    # Determine overall width.  If ``width_cm`` is not provided, derive it
    # by adding the chart and gauge widths.  This allows callers to adjust
    # the gauge width independently of the slide size.  The chart width
    # defaults to ~21.41 cm and the gauge to 4.0 cm.
    # Determine overall width.  If ``chart_width_cm`` is not provided,
    # derive it by subtracting the gauge width from the total width.  This
    # ensures that the combined chart and gauge fit within the fixed slide
    # placeholder width (typically ~21.41 cm).  Callers may supply
    # ``chart_width_cm`` explicitly to override this behaviour.
    if chart_width_cm is None:
        chart_width_cm = max(width_cm - gauge_width_cm, 0.0)

    fig_w_in, fig_h_in = width_cm / 2.54, height_cm / 2.54
    plt.style.use("default")
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    # Determine relative widths for the chart and gauge.  The chart occupies
    # ``chart_width_cm`` cm of the total width while the gauge occupies
    # ``gauge_width_cm`` cm.  These ratios control how much of the figure is
    # devoted to each element.
    chart_rel_width = chart_width_cm / width_cm
    gauge_rel_width = gauge_width_cm / width_cm

    # Create main chart axis using add_axes to occupy the left portion of the
    # figure.  We leave the full height (0→1) for the chart; legend
    # positioning is handled later.
    ax = fig.add_axes([0.0, 0.0, chart_rel_width, 1.0])
    # Placeholder for gauge axis; we will add it after plotting on ax so we can
    # align it vertically with the plotted area of the chart.
    ax_gauge = None

    # Plot main price series and MAs
    ax.plot(
        df["Date"], df["Price"], color="#153D64", linewidth=2.5, label=f"S&P 500 Price (last: {last_price_str})"
    )
    ax.plot(df_ma["Date"], df_ma["MA_50"], color="#008000", linewidth=1.5, label="50‑day MA")
    ax.plot(df_ma["Date"], df_ma["MA_100"], color="#FFA500", linewidth=1.5, label="100‑day MA")
    ax.plot(df_ma["Date"], df_ma["MA_200"], color="#FF0000", linewidth=1.5, label="200‑day MA")

    # Fibonacci levels
    hi, lo = df["Price"].max(), df["Price"].min()
    span = hi - lo
    fib_levels = [hi, hi - 0.236 * span, hi - 0.382 * span, hi - 0.5 * span, hi - 0.618 * span, lo]
    for lvl in fib_levels:
        ax.axhline(lvl, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    # Regression channel shading
    if anchor_date is not None and upper_channel is not None and lower_channel is not None:
        subset = df_full[df_full["Date"].between(anchor_date, today)].copy().reset_index(drop=True)
        fill_color = (0, 0.6, 0, 0.25) if uptrend else (0.78, 0, 0, 0.25)
        line_color = "#008000" if uptrend else "#C00000"
        ax.plot(subset["Date"], upper_channel, color=line_color, linestyle="--")
        ax.plot(subset["Date"], lower_channel, color=line_color, linestyle="--")
        ax.fill_between(subset["Date"], lower_channel, upper_channel, color=fill_color)

    # Hide spines and style ticks on the main chart axis
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    # Add legend for main chart
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=4,
        fontsize=8,
        frameon=False,
    )

    # ---------------------------------------------------------------------
    # Create and draw the range gauge.  We first create the gauge axis so
    # that it shares the y‑limits of the main chart.  Sharing the y‑axis
    # ensures that the gradient and markers align with the same numeric
    # scale as the price chart.  The gauge occupies the remaining
    # horizontal width on the right of the figure.
    # ---------------------------------------------------------------------
    # Determine the left position and width (in figure coordinates) for the
    # gauge axis.  It begins immediately after the main chart and uses
    # ``gauge_rel_width`` as its width.
    gauge_left = chart_rel_width
    gauge_width = gauge_rel_width
    # Create the gauge axis, sharing its y‑axis with the main chart.  This
    # ensures that y‑coordinates on the gauge correspond to price levels on
    # the chart.  The x‑axis range (0→1) will represent the width of the
    # gauge; we do not display ticks on this axis.
    ax_gauge = fig.add_axes([gauge_left, 0.0, gauge_width, 1.0], sharey=ax)
    # Hide tick marks and labels for the gauge axis
    ax_gauge.set_xticks([])
    ax_gauge.set_yticks([])

    # Build a vertical gradient (red → white → green) and draw it only
    # within the computed trading range.  The gradient is drawn using
    # ``imshow`` with an extent that maps the gradient onto the segment
    # between ``lower_bound`` and ``upper_bound`` on the y‑axis.  Areas
    # outside this extent are left blank by setting the axis facecolour.
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    cmap = LinearSegmentedColormap.from_list(
        "range_gauge", ["#FF0000", "#FFFFFF", "#009951"], N=256
    )
    ax_gauge.imshow(
        gradient,
        extent=[0, 1, lower_bound, upper_bound],
        aspect="auto",
        origin="lower",
        cmap=cmap,
    )
    # Fill background outside the gradient with opaque white so that
    # regions above the upper bound and below the lower bound remain
    # neutral.
    ax_gauge.set_facecolor((1, 1, 1, 1))

    # Draw a marker indicating the last price.  The marker is a thin
    # horizontal rectangle spanning the entire width of the gauge.  Its
    # height is set to 1 % of the trading range to remain subtle yet
    # visible.  If the trading range is zero, no marker is drawn.
    full_range = upper_bound - lower_bound
    marker_height = full_range * 0.01 if full_range > 0 else 0
    if marker_height > 0:
        ax_gauge.add_patch(
            patches.Rectangle(
                (0.0, last_price - marker_height / 2.0),
                1.0,
                marker_height,
                color="#153D64",
            )
        )

    # Helper to format numeric values with apostrophe separators.
    def _format_value(val: float) -> str:
        try:
            return f"{val:,.0f}".replace(",", "'")
        except Exception:
            return f"{val:.0f}"
    upper_label = _format_value(upper_bound)
    lower_label = _format_value(lower_bound)
    # Compute percentage differences relative to the last price
    up_pct = (upper_bound - last_price) / last_price * 100 if last_price else 0.0
    down_pct = (last_price - lower_bound) / last_price * 100 if last_price else 0.0
    # Compose label strings for the upper and lower bounds.  The
    # percentage differences are shown with a sign and one decimal place.
    upper_text = f"Higher Range\n{upper_label} $\n(+{up_pct:.1f}\%)"
    lower_text = f"Lower Range\n{lower_label} $\n(-{down_pct:.1f}\%)"
    # Position the labels just outside the gauge to the right.  We use
    # data coordinates (``transData``) so that the text aligns with the
    # actual price levels.  The x‑coordinate 1.05 places the text slightly
    # to the right of the gauge.
    ax_gauge.text(
        1.05,
        upper_bound,
        upper_text,
        color="#009951",
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        transform=ax_gauge.transData,
    )
    ax_gauge.text(
        1.05,
        lower_bound,
        lower_text,
        color="#C00000",
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        transform=ax_gauge.transData,
    )

    # Final styling for the gauge axis: hide all spines and fix x‑limits.
    ax_gauge.set_xlim(0, 1)
    for side in ["left", "right", "top", "bottom"]:
        ax_gauge.spines[side].set_visible(False)


    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_range_gauge_only_image(
    df_full: pd.DataFrame,
    lookback_days: int = 90,
    width_cm: float = 2.00,
    height_cm: float = 7.53,
) -> bytes:
    """
    Create a standalone vertical gauge image without the price chart.

    This function is intended for interactive environments (e.g. Streamlit) where
    users want to visualise the recent trading range alongside a separate
    interactive plot.  The gauge shows a green–to–red gradient between the
    computed upper and lower bounds, with labels at the extremes and a marker
    indicating the current price’s position within the range.

    Parameters
    ----------
    df_full : pandas.DataFrame
        Full SPX price history as returned by ``_load_price_data``.
    lookback_days : int, default 90
        Number of trading days to look back when computing high/low range.
    width_cm : float, default 2.00
        Width of the output image in centimetres.  A narrow bar suffices for
        embedding alongside an interactive chart in Streamlit.
    height_cm : float, default 7.53
        Height of the gauge in centimetres.  This should match the height of
        your interactive chart for consistent alignment.

    Returns
    -------
    bytes
        PNG image data for the standalone range gauge.
    """
    if df_full.empty:
        return b""
    # Compute bounds and current price
    upper_bound, lower_bound = _compute_range_bounds(df_full, lookback_days=lookback_days)
    current_price = df_full["Price"].iloc[-1]
    # Normalise current position within the range
    if upper_bound == lower_bound:
        rel_pos = 0.5
    else:
        rel_pos = (current_price - lower_bound) / (upper_bound - lower_bound)
        rel_pos = max(0.0, min(1.0, rel_pos))

    # Prepare figure
    fig_w_in, fig_h_in = width_cm / 2.54, height_cm / 2.54
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
    # Build vertical gradient: red → white → green
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    cmap = LinearSegmentedColormap.from_list(
        "range_gauge_only", ["#FF0000", "#FFFFFF", "#009951"], N=256
    )
    ax.imshow(
        gradient,
        extent=[0, 1, lower_bound, upper_bound],
        aspect="auto",
        origin="lower",
        cmap=cmap,
    )
    ax.set_facecolor((1, 1, 1, 0))
    # Draw marker for current price as a horizontal bar spanning the gauge width
    marker_y = lower_bound + rel_pos * (upper_bound - lower_bound)
    marker_height = (upper_bound - lower_bound) * 0.01  # 1% of range height
    ax.add_patch(
        patches.Rectangle(
            (0.0, marker_y - marker_height / 2),
            1.0,
            marker_height,
            color="#153D64",
        )
    )
    # Draw labels for bounds (centre aligned)
    def _fmt(val):
        try:
            return f"{val:,.0f}".replace(",", "'")
        except Exception:
            return f"{val:.0f}"
    upper_label = _fmt(upper_bound)
    lower_label = _fmt(lower_bound)
    ax.text(
        0.5,
        upper_bound,
        f"Higher Range\n{upper_label} $",
        color="#009951",
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        transform=ax.transData,
    )
    ax.text(
        0.5,
        lower_bound,
        f"Lower Range\n{lower_label} $",
        color="#C00000",
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        transform=ax.transData,
    )
    # Format axes: hide ticks and spines
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def insert_spx_technical_chart_with_range(
    prs: Presentation,
    excel_file,
    anchor_date: Optional[pd.Timestamp] = None,
    lookback_days: int = 90,
) -> Presentation:
    """
    Insert the SPX technical analysis chart with the vertical range gauge into the PPT.

    This function behaves similarly to ``insert_spx_technical_chart`` but uses
    ``generate_range_gauge_chart_image`` to draw a combined chart and gauge.
    It attempts to find a shape named 'spx' or containing '[spx]' to locate the
    slide for insertion.  The image is placed at fixed coordinates matching the
    original template (0.93 cm left, 4.39 cm top, 21.41 cm wide, 7.53 cm high).

    Parameters
    ----------
    prs : Presentation
        The PowerPoint presentation into which the chart should be inserted.
    excel_file : file‑like object or path
        Excel workbook containing SPX price data.
    anchor_date : pandas.Timestamp or None, optional
        Optional anchor date for the regression channel.
    lookback_days : int, default 90
        Lookback window for computing the high and low range bounds.

    Returns
    -------
    Presentation
        The presentation with the updated slide.
    """
    # Load data
    try:
        df_full = _load_price_data_from_obj(excel_file, "SPX Index")
    except Exception:
        df_full = _load_price_data(pathlib.Path(excel_file), "SPX Index")
    img_bytes = generate_range_gauge_chart_image(
        df_full, anchor_date=anchor_date, lookback_days=lookback_days
    )

    # Locate target slide
    target_slide = None
    for slide in prs.slides:
        for shape in slide.shapes:
            name_attr = getattr(shape, "name", "").lower()
            if name_attr == "spx":
                target_slide = slide
                break
            if shape.has_text_frame:
                if (shape.text or "").strip().lower() == "[spx]":
                    target_slide = slide
                    break
        if target_slide:
            break
    if target_slide is None:
        target_slide = prs.slides[min(11, len(prs.slides) - 1)]

    # Position and dimensions tailored to the original placeholder size.
    # The SPX slide in the template allocates ~21.41 cm for the chart area
    # and reserves the remaining width for the chart title, subtitle and
    # margins.  We therefore insert the combined chart‑and‑gauge image
    # using the original dimensions (21.41 cm × 7.53 cm) and rely on the
    # gauge function to include the gauge within that width.  This avoids
    # cropping the chart when the image is inserted into the slide.
    left = Cm(0.93)
    top = Cm(4.40)
    width = Cm(21.41)
    height = Cm(7.53)
    stream = BytesIO(img_bytes)
    target_slide.shapes.add_picture(stream, left, top, width=width, height=height)
    return prs
