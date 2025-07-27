"""
Utility functions for SPX technical analysis and high‑resolution export.

This module provides tools to build interactive and static charts for the
S&P 500 index, calculate and insert technical and momentum scores into
PowerPoint presentations, generate a horizontal gauge that visualises the
average of the technical and momentum scores, and insert that gauge into
a slide. Functions fall back to sensible defaults when placeholders are
not found.

Key functions:
  * make_spx_figure – interactive Plotly chart for Streamlit.
  * insert_spx_technical_chart – insert a static SPX chart into the PPT.
  * insert_spx_technical_score_number – insert the technical score (integer).
  * insert_spx_momentum_score_number – insert the momentum score (integer).
  * insert_spx_subtitle – insert a user‑defined subtitle into the SPX slide.
  * generate_average_gauge_image – create a horizontal gauge image.
  * insert_spx_average_gauge – insert the gauge into a PPT slide.
  * insert_spx_technical_assessment – insert a descriptive “view” text.
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
    return df.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)[["Date","Price"]]

def _add_mas(df: pd.DataFrame) -> pd.DataFrame:
    """Add 50/100/200‑day moving‑average columns."""
    out = df.copy()
    for w in (50, 100, 200):
        out[f"MA_{w}"] = out["Price"].rolling(w, min_periods=1).mean()
    return out

# ---------------------------------------------------------------------------
# Plotly interactive chart for Streamlit
# ---------------------------------------------------------------------------
def make_spx_figure(excel_path: str | pathlib.Path,
                    anchor_date: Optional[pd.Timestamp] = None) -> go.Figure:
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
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Price"], mode="lines",
        name=f"S&P 500 Price (last: {last_price_str})",
        line=dict(color="#153D64", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["MA_50"],
        mode="lines", name="50‑day MA", line=dict(color="#008000", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["MA_100"],
        mode="lines", name="100‑day MA", line=dict(color="#FFA500", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["MA_200"],
        mode="lines", name="200‑day MA", line=dict(color="#FF0000", width=1.5)
    ))

    hi, lo = df["Price"].max(), df["Price"].min()
    span = hi - lo
    for lvl in [hi, hi - 0.236*span, hi - 0.382*span, hi - 0.5*span,
                hi - 0.618*span, lo]:
        fig.add_hline(y=lvl, line=dict(color="grey", dash="dash", width=1),
                      opacity=0.6)

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

        fig.add_trace(go.Scatter(x=per["Date"], y=upper,
                                 mode="lines", line=dict(color=lineclr, dash="dash"),
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=per["Date"], y=lower,
                                 mode="lines", line=dict(color=lineclr, dash="dash"),
                                 fill="tonexty", fillcolor=fillclr, showlegend=False))

    fig.update_layout(
        margin=dict(l=30, r=30, t=60, b=40),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.12,
            xanchor="center", x=0.5, font=dict(size=12)
        ),
        xaxis_title=None, yaxis_title=None,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, zeroline=False)
    )
    return fig

# ---------------------------------------------------------------------------
# High‑resolution chart export (PNG)
# ---------------------------------------------------------------------------
def _generate_spx_image_from_df(df_full: pd.DataFrame,
                                anchor_date: Optional[pd.Timestamp],
                                width_cm: float = 21.41,
                                height_cm: float = 7.53) -> bytes:
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

    fig_width_in, fig_height_in = width_cm/2.54, height_cm/2.54
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))

    ax.plot(df["Date"], df["Price"],
            color="#153D64", linewidth=2.5,
            label=f"S&P 500 Price (last: {last_price_str})")
    ax.plot(df_ma["Date"], df_ma["MA_50"], color="#008000", linewidth=1.5, label="50‑day MA")
    ax.plot(df_ma["Date"], df_ma["MA_100"], color="#FFA500", linewidth=1.5, label="100‑day MA")
    ax.plot(df_ma["Date"], df_ma["MA_200"], color="#FF0000", linewidth=1.5, label="200‑day MA")

    hi, lo = df["Price"].max(), df["Price"].min()
    span = hi - lo
    fib_levels = [hi, hi - 0.236*span, hi - 0.382*span,
                  hi - 0.5*span, hi - 0.618*span, lo]
    for lvl in fib_levels:
        ax.axhline(y=lvl, color="grey", linestyle="--",
                   linewidth=0.8, alpha=0.6)

    if anchor_date is not None and upper is not None and lower is not None:
        fill_color = (0,0.6,0,0.25) if uptrend else (0.78,0,0,0.25)
        line_color = "#008000" if uptrend else "#C00000"
        subset = df_full[df_full["Date"].between(anchor_date, today)].copy().reset_index(drop=True)
        ax.plot(subset["Date"], upper, color=line_color, linestyle="--")
        ax.plot(subset["Date"], lower, color=line_color, linestyle="--")
        ax.fill_between(subset["Date"], lower, upper, color=fill_color)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4,
              fontsize=8, frameon=False)
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
                    saved_size  = runs[0].font.size  if runs else None
                    saved_color = runs[0].font.color.rgb if runs else None
                    saved_bold  = runs[0].font.bold if runs else None
                    saved_italic= runs[0].font.italic if runs else None
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
                        saved_size  = runs[0].font.size  if runs else None
                        saved_color = runs[0].font.color.rgb if runs else None
                        saved_bold  = runs[0].font.bold if runs else None
                        saved_italic= runs[0].font.italic if runs else None
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
                    saved_size  = runs[0].font.size  if runs else None
                    saved_color = runs[0].font.color.rgb if runs else None
                    saved_bold  = runs[0].font.bold if runs else None
                    saved_italic= runs[0].font.italic if runs else None
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
                        saved_size  = runs[0].font.size  if runs else None
                        saved_color = runs[0].font.color.rgb if runs else None
                        saved_bold  = runs[0].font.bold if runs else None
                        saved_italic= runs[0].font.italic if runs else None
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
def insert_spx_technical_chart(prs: Presentation,
                               excel_file,
                               anchor_date: Optional[pd.Timestamp] = None) -> Presentation:
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
                    saved_size  = runs[0].font.size  if runs else None
                    saved_color = runs[0].font.color.rgb if runs else None
                    saved_bold  = runs[0].font.bold if runs else None
                    saved_italic= runs[0].font.italic if runs else None
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
                        saved_size  = runs[0].font.size  if runs else None
                        saved_color = runs[0].font.color.rgb if runs else None
                        saved_bold  = runs[0].font.bold if runs else None
                        saved_italic= runs[0].font.italic if runs else None
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
    red    = (1.0, 0.0, 0.0)
    yellow = (1.0, 204/255, 0.0)
    green  = (0.0, 153/255, 81/255)
    if value <= 40:
        t = value / 40.0
        return tuple(red[i] + t*(yellow[i]-red[i]) for i in range(3))
    elif value <= 70:
        t = (value - 40) / 30.0
        return tuple(yellow[i] + t*(green[i]-yellow[i]) for i in range(3))
    return green

def generate_average_gauge_image(tech_score: float,
                                 mom_score: float,
                                 last_week_avg: float,
                                 date_text: str | None = None,
                                 last_label_text: str = "Last Week",
                                 width_cm: float = 15.15,
                                 height_cm: float = 3.13) -> bytes:
    """
    Create a horizontal gauge with a red→yellow→green gradient, marking the
    average of technical and momentum scores against last week’s average.
    """
    def clamp100(x: float) -> float:
        return max(0.0, min(100.0, float(x)))

    curr = (clamp100(tech_score) + clamp100(mom_score)) / 2.0
    prev = clamp100(last_week_avg)

    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    cmap = LinearSegmentedColormap.from_list(
        "gauge_gradient", ["#FF0000", "#FFCC00", "#009951"], N=256
    )

    fig_w, fig_h = width_cm/2.54, height_cm/2.54
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    gradient = np.linspace(0, 1, 500).reshape(1, -1)
    bar_thickness = 0.4
    bar_bottom_y = -bar_thickness / 2.0
    bar_top_y    =  bar_thickness / 2.0
    ax.imshow(
        gradient,
        extent=[0, 100, bar_bottom_y, bar_top_y],
        aspect="auto",
        cmap=cmap,
        origin="lower",
    )

    # Marker dimensions and spacing
    marker_width  = 3.0
    marker_height = 0.15
    gap           = 0.10
    number_space  = 0.25
    top_label_offset = 0.40
    bottom_label_offset = 0.40

    # Y positions for current (top) marker and labels
    top_apex_y    = bar_top_y + gap
    top_base_y    = top_apex_y + marker_height
    top_number_y  = top_base_y + number_space
    top_label_y   = top_number_y + top_label_offset

    # Y positions for previous (bottom) marker and labels
    bottom_apex_y   = bar_bottom_y - gap
    bottom_base_y   = bottom_apex_y - marker_height
    bottom_number_y = bottom_base_y - number_space
    bottom_label_y  = bottom_number_y - bottom_label_offset

    curr_colour = _interpolate_color(curr)
    prev_colour = _interpolate_color(prev)

    # Draw triangles and numbers
    ax.add_patch(patches.Polygon([
        (curr - marker_width/2, top_base_y),
        (curr + marker_width/2, top_base_y),
        (curr, top_apex_y)
    ], color=curr_colour))
    ax.add_patch(patches.Polygon([
        (prev - marker_width/2, bottom_base_y),
        (prev + marker_width/2, bottom_base_y),
        (prev, bottom_apex_y)
    ], color=prev_colour))
    ax.text(curr, top_number_y, f"{curr:.0f}",
            color=curr_colour, ha="center", va="center",
            fontsize=8, fontweight="bold")
    ax.text(prev, bottom_number_y, f"{prev:.0f}",
            color=prev_colour, ha="center", va="center",
            fontsize=8, fontweight="bold")

    if date_text:
        ax.text(curr, top_label_y, date_text, color="#0063B0",
                ha="center", va="center", fontsize=7, fontweight="bold")
    ax.text(prev, bottom_label_y, last_label_text, color="#133C74",
            ha="center", va="center", fontsize=7, fontweight="bold")

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
    df["Date"]  = pd.to_datetime(df[df.columns[0]], errors="coerce")
    df["Price"] = pd.to_numeric(df[ticker], errors="coerce")
    return df.dropna(subset=["Date","Price"]).sort_values("Date").reset_index(drop=True)[["Date","Price"]]

# ---------------------------------------------------------------------------
# Gauge insertion
# ---------------------------------------------------------------------------
def insert_spx_average_gauge(prs: Presentation,
                             excel_file,
                             last_week_avg: float) -> Presentation:
    """
    Insert the average gauge into the SPX slide.  Looks for a shape named
    'gauge_spx' or text containing '[GAUGE]', 'GAUGE', or 'gauge_spx'.  If found,
    the gauge uses that position; otherwise it is inserted below the chart at
    the default coordinates (8.97 cm left, 12.13 cm top, 15.15 cm wide, 3.13 cm high).
    """
    tech_score = _get_spx_technical_score(excel_file)
    mom_score  = _get_spx_momentum_score(excel_file)
    if tech_score is None or mom_score is None:
        return prs

    try:
        gauge_bytes = generate_average_gauge_image(
            tech_score, mom_score, last_week_avg,
            date_text="Last", last_label_text="Previous Week",
            width_cm=15.15, height_cm=3.13
        )
    except Exception:
        return prs

    placeholder_name = "gauge_spx"
    placeholder_patterns = ["[GAUGE]","GAUGE","gauge_spx"]

    for slide in prs.slides:
        for shape in slide.shapes:
            if getattr(shape,"name","").lower() == placeholder_name:
                left, top, width, height = shape.left, shape.top, shape.width, shape.height
                if shape.has_text_frame:
                    shape.text = ""
                stream = BytesIO(gauge_bytes)
                slide.shapes.add_picture(stream, left, top, width=width, height=height)
                return prs
            if shape.has_text_frame:
                for pattern in placeholder_patterns:
                    if pattern.lower() in shape.text.lower():
                        left, top, width, height = shape.left, shape.top, shape.width, shape.height
                        shape.text = shape.text.replace(pattern,"")
                        stream = BytesIO(gauge_bytes)
                        slide.shapes.add_picture(stream, left, top, width=width, height=height)
                        return prs

    # Fallback: fixed coordinates below the chart
    idx = min(11, len(prs.slides) - 1)
    slide = prs.slides[idx]
    left = Cm(8.97)
    top  = Cm(12.13)
    width = Cm(15.15)
    height= Cm(3.13)
    stream = BytesIO(gauge_bytes)
    slide.shapes.add_picture(stream, left, top, width=width, height=height)
    return prs

# ---------------------------------------------------------------------------
# Technical assessment insertion
# ---------------------------------------------------------------------------
def insert_spx_technical_assessment(prs: Presentation,
                                    excel_file) -> Presentation:
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
    mom_score  = _get_spx_momentum_score(excel_file)
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
    placeholder_patterns = ["[spx_view]","spx_view"]

    for slide in prs.slides:
        for shape in slide.shapes:
            name_attr = getattr(shape,"name","")
            if name_attr and name_attr.lower() == target_name:
                if shape.has_text_frame:
                    runs = shape.text_frame.paragraphs[0].runs
                    saved_size  = runs[0].font.size  if runs else None
                    saved_color = runs[0].font.color.rgb if runs else None
                    saved_bold  = runs[0].font.bold if runs else None
                    saved_italic= runs[0].font.italic if runs else None
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
                        saved_size  = runs[0].font.size  if runs else None
                        saved_color = runs[0].font.color.rgb if runs else None
                        saved_bold  = runs[0].font.bold if runs else None
                        saved_italic= runs[0].font.italic if runs else None
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