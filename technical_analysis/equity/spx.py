"""
technical_analysis/equity/spx.py
================================

Utility functions for SPX technical analysis and high-resolution export.

Functions:
- make_spx_figure: Interactive Plotly chart for Streamlit.
- insert_spx_technical_chart: Insert an SPX technical chart into PPT.
- insert_spx_technical_score_number: Insert the technical score (integer, no decimals).
- insert_spx_momentum_score_number: Insert the momentum score (integer, no decimals).
- insert_spx_subtitle: Insert a user-defined subtitle into the appropriate textbox.
- [Optional] insert_spx_technical_score (gauge) and related helpers (not used here).
"""

from __future__ import annotations
from datetime import timedelta
import pathlib
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# For PPT insertion
from pptx import Presentation
from pptx.util import Cm
from io import BytesIO
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _load_price_data(
    excel_path: pathlib.Path,
    ticker: str = "SPX Index",
) -> pd.DataFrame:
    """Read the raw price sheet and return tidy Date-Price DataFrame."""
    df = pd.read_excel(excel_path, sheet_name="data_prices")
    df = df.drop(index=0)                         # first row = '#price'
    df = df[df[df.columns[0]] != "DATES"]         # second header row
    df["Date"]  = pd.to_datetime(df[df.columns[0]], errors="coerce")
    df["Price"] = pd.to_numeric(df[ticker], errors="coerce")
    return (
        df.dropna(subset=["Date", "Price"])
          .sort_values("Date")
          .reset_index(drop=True)[["Date", "Price"]]
    )

def _add_mas(df: pd.DataFrame) -> pd.DataFrame:
    """Add 50/100/200-day moving-average columns."""
    out = df.copy()
    for w in (50, 100, 200):
        out[f"MA_{w}"] = out["Price"].rolling(w, min_periods=1).mean()
    return out

# ---------------------------------------------------------------------------
# Plotly interactive chart for Streamlit
# ---------------------------------------------------------------------------
def make_spx_figure(
    excel_path: str | pathlib.Path,
    anchor_date: Optional[pd.Timestamp] = None,
) -> go.Figure:
    """
    Build the interactive SPX chart for Streamlit.

    Parameters
    ----------
    excel_path : str or pathlib.Path
        Path to the Excel file containing SPX price data.
    anchor_date : pandas.Timestamp or None
        If provided, a regression channel is drawn from anchor_date to the latest date.

    Returns
    -------
    go.Figure
        A Plotly figure with price, moving averages, Fibonacci lines, and optional trend channel.
    """
    excel_path = pathlib.Path(excel_path)
    df_full = _add_mas(_load_price_data(excel_path, "SPX Index"))

    today  = df_full["Date"].max().normalize()
    start  = today - timedelta(days=365)
    df     = df_full[df_full["Date"].between(start, today)].reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Price"],
        mode="lines", name="S&P 500 Price", line=dict(color="#153D64", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["MA_50"],
        mode="lines", name="50-day MA", line=dict(color="#008000", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["MA_100"],
        mode="lines", name="100-day MA", line=dict(color="#FFA500", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["MA_200"],
        mode="lines", name="200-day MA", line=dict(color="#FF0000", width=1.5)
    ))

    # Fibonacci levels
    hi, lo = df["Price"].max(), df["Price"].min()
    span   = hi - lo
    for lvl in [hi, hi-0.236*span, hi-0.382*span, hi-0.5*span, hi-0.618*span, lo]:
        fig.add_hline(
            y=lvl,
            line=dict(color="grey", dash="dash", width=1),
            opacity=0.6,
        )

    # Regression channel
    if anchor_date is not None:
        per = df_full[df_full["Date"].between(anchor_date, today)].copy()
        X = per["Date"].map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
        y = per["Price"].to_numpy()
        model  = LinearRegression().fit(X, y)
        trend  = model.predict(X)
        resid  = y - trend
        upper  = trend + resid.max()
        lower  = trend + resid.min()

        uptrend = model.coef_[0] > 0
        lineclr = "green" if uptrend else "red"
        fillclr = "rgba(0,150,0,0.25)" if uptrend else "rgba(200,0,0,0.25)"

        fig.add_trace(go.Scatter(
            x=per["Date"], y=upper,
            mode="lines", line=dict(color=lineclr, dash="dash"),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=per["Date"], y=lower,
            mode="lines", line=dict(color=lineclr, dash="dash"),
            fill="tonexty", fillcolor=fillclr,
            showlegend=False
        ))

    fig.update_layout(
        margin=dict(l=30, r=30, t=60, b=40),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.12,
            xanchor="center", x=0.5,
            font=dict(size=12),
        ),
        xaxis_title=None,
        yaxis_title=None,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )
    return fig

# ---------------------------------------------------------------------------
# High-resolution chart export (PNG)
# ---------------------------------------------------------------------------
def _generate_spx_image_from_df(
    df_full: pd.DataFrame,
    anchor_date: Optional[pd.Timestamp],
    width_cm: float = 21.41,
    height_cm: float = 7.53,
) -> bytes:
    """
    Create a high-resolution (dpi=300) transparent PNG chart from the DataFrame.
    The image includes the price, moving averages, Fibonacci lines, and optional
    regression channel, with no border and ticks visible.
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
            residuals = y_vals - trend
            uptrend = model.coef_[0] > 0
            upper = trend + residuals.max()
            lower = trend + residuals.min()

    fig_width_in = width_cm / 2.54
    fig_height_in = height_cm / 2.54

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))

    ax.plot(df["Date"], df["Price"], color="#153D64", linewidth=2.5, label="S&P 500 Price")
    ax.plot(df_ma["Date"], df_ma["MA_50"], color="#008000", linewidth=1.5, label="50-day MA")
    ax.plot(df_ma["Date"], df_ma["MA_100"], color="#FFA500", linewidth=1.5, label="100-day MA")
    ax.plot(df_ma["Date"], df_ma["MA_200"], color="#FF0000", linewidth=1.5, label="200-day MA")

    hi = df["Price"].max()
    lo = df["Price"].min()
    span = hi - lo
    fib_levels = [hi, hi - 0.236 * span, hi - 0.382 * span, hi - 0.5 * span,
                  hi - 0.618 * span, lo]
    for lvl in fib_levels:
        ax.axhline(y=lvl, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    if anchor_date is not None and upper is not None and lower is not None:
        fill_color = (0, 0.6, 0, 0.25) if uptrend else (0.78, 0, 0, 0.25)
        line_color = "#008000" if uptrend else "#C00000"
        subset = df_full[df_full["Date"].between(anchor_date, today)].copy().reset_index(drop=True)
        ax.plot(subset["Date"], upper, color=line_color, linestyle="--")
        ax.plot(subset["Date"], lower, color=line_color, linestyle="--")
        ax.fill_between(subset["Date"], lower, upper, color=fill_color)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize=8, frameon=False)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ---------------------------------------------------------------------------
# Technical score helpers
# ---------------------------------------------------------------------------
def _get_spx_technical_score(excel_obj_or_path) -> Optional[float]:
    """
    Retrieve the technical score for SPX from 'data_technical_score' (col A: ticker, col B: score).
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

def insert_spx_technical_score_number(
    prs: Presentation,
    excel_file,
) -> Presentation:
    """
    Insert the SPX technical score (integer) into the shape named 'tech_score_spx'
    or a shape containing '[XXX]' or 'XXX'. Preserves original formatting.
    """
    score = _get_spx_technical_score(excel_file)
    if score is None:
        score_text = "N/A"
    else:
        try:
            score_text = f"{int(round(float(score)))}"
        except Exception:
            score_text = str(score)

    placeholder_name = "tech_score_spx"
    placeholder_patterns = ["[XXX]", "XXX"]

    for slide in prs.slides:
        for shape in slide.shapes:
            # Match by name first
            if hasattr(shape, "name") and shape.name.lower() == placeholder_name:
                if shape.has_text_frame:
                    # Save formatting from first run
                    if shape.text_frame.paragraphs and shape.text_frame.paragraphs[0].runs:
                        orig_run = shape.text_frame.paragraphs[0].runs[0]
                        saved_size = orig_run.font.size
                        saved_color = orig_run.font.color.rgb
                        saved_bold = orig_run.font.bold
                        saved_italic = orig_run.font.italic
                    else:
                        saved_size = saved_color = saved_bold = saved_italic = None
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

            # Otherwise match by placeholder patterns in text
            if shape.has_text_frame:
                for pattern in placeholder_patterns:
                    if pattern in shape.text:
                        # Save formatting
                        para = shape.text_frame.paragraphs[0]
                        if para.runs:
                            orig_run = para.runs[0]
                            saved_size = orig_run.font.size
                            saved_color = orig_run.font.color.rgb
                            saved_bold = orig_run.font.bold
                            saved_italic = orig_run.font.italic
                        else:
                            saved_size = saved_color = saved_bold = saved_italic = None

                        new_text = shape.text.replace(pattern, score_text)
                        shape.text_frame.clear()
                        new_para = shape.text_frame.paragraphs[0]
                        new_run = new_para.add_run()
                        new_run.text = new_text
                        # Apply formatting to the number portion
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
# Momentum score helpers
# ---------------------------------------------------------------------------
def _get_spx_momentum_score(excel_obj_or_path) -> Optional[float]:
    """
    Retrieve the momentum score for SPX from 'data_trend_rating' (col A: ticker, col D: score).
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

def insert_spx_momentum_score_number(
    prs: Presentation,
    excel_file,
) -> Presentation:
    """
    Insert the SPX momentum score (integer) into the shape named 'mom_score_spx'
    or a shape containing '[XXX]' or 'XXX'. Preserves original formatting.
    """
    score = _get_spx_momentum_score(excel_file)
    if score is None:
        score_text = "N/A"
    else:
        try:
            score_text = f"{int(round(float(score)))}"
        except Exception:
            score_text = str(score)

    placeholder_name = "mom_score_spx"
    placeholder_patterns = ["[XXX]", "XXX"]

    for slide in prs.slides:
        for shape in slide.shapes:
            # Match by name
            if hasattr(shape, "name") and shape.name.lower() == placeholder_name:
                if shape.has_text_frame:
                    if shape.text_frame.paragraphs and shape.text_frame.paragraphs[0].runs:
                        orig_run = shape.text_frame.paragraphs[0].runs[0]
                        saved_size = orig_run.font.size
                        saved_color = orig_run.font.color.rgb
                        saved_bold = orig_run.font.bold
                        saved_italic = orig_run.font.italic
                    else:
                        saved_size = saved_color = saved_bold = saved_italic = None
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

            # Otherwise match by placeholder patterns in text
            if shape.has_text_frame:
                for pattern in placeholder_patterns:
                    if pattern in shape.text:
                        if shape.text_frame.paragraphs and shape.text_frame.paragraphs[0].runs:
                            orig_run = shape.text_frame.paragraphs[0].runs[0]
                            saved_size = orig_run.font.size
                            saved_color = orig_run.font.color.rgb
                            saved_bold = orig_run.font.bold
                            saved_italic = orig_run.font.italic
                        else:
                            saved_size = saved_color = saved_bold = saved_italic = None

                        new_text = shape.text.replace(pattern, score_text)
                        shape.text_frame.clear()
                        new_para = shape.text_frame.paragraphs[0]
                        new_run = new_para.add_run()
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
    prs: Presentation,
    excel_file,
    anchor_date: Optional[pd.Timestamp] = None
) -> Presentation:
    """
    Insert the SPX technical-analysis chart into the PPT. Looks for 'tech_spx'
    placeholder or falls back to a default slide and position.
    """
    try:
        df_full = _load_price_data_from_obj(excel_file, "SPX Index")
    except Exception:
        df_full = _load_price_data(pathlib.Path(excel_file), "SPX Index")

    img_bytes = _generate_spx_image_from_df(df_full, anchor_date)

    placeholder_text = "tech_spx"
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame and placeholder_text.lower() in shape.text.lower():
                left = shape.left
                top = shape.top
                width = shape.width
                height = shape.height
                if shape.has_text_frame:
                    shape.text = ""
                stream = BytesIO(img_bytes)
                slide.shapes.add_picture(stream, left, top, width=width, height=height)
                return prs

    # Fallback: slide 12 or last slide, fixed position and size
    idx = min(11, len(prs.slides) - 1)
    slide = prs.slides[idx]
    left = Cm(0.93)
    top = Cm(4.39)
    width = Cm(21.41)
    height = Cm(7.53)
    stream = BytesIO(img_bytes)
    slide.shapes.add_picture(stream, left, top, width=width, height=height)
    return prs

# ---------------------------------------------------------------------------
# Subtitle insertion
# ---------------------------------------------------------------------------
def insert_spx_subtitle(
    prs: Presentation,
    subtitle: str,
) -> Presentation:
    """
    Replace the placeholder text ('XXX' or '[XXX]') in the textbox named
    'spx_text' (or containing those placeholders) with the provided subtitle,
    preserving the original formatting.
    """
    placeholder_name = "spx_text"
    placeholder_patterns = ["[XXX]", "XXX"]

    subtitle_text = subtitle if subtitle is not None else ""

    for slide in prs.slides:
        for shape in slide.shapes:
            # Match by shape name
            if hasattr(shape, "name") and shape.name.lower() == placeholder_name:
                if shape.has_text_frame:
                    # Extract formatting
                    if shape.text_frame.paragraphs and shape.text_frame.paragraphs[0].runs:
                        orig_run = shape.text_frame.paragraphs[0].runs[0]
                        saved_size = orig_run.font.size
                        saved_color = orig_run.font.color.rgb
                        saved_bold = orig_run.font.bold
                        saved_italic = orig_run.font.italic
                    else:
                        saved_size = saved_color = saved_bold = saved_italic = None

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

            # Otherwise match by placeholder patterns in text
            if shape.has_text_frame:
                for pattern in placeholder_patterns:
                    if pattern in shape.text:
                        if shape.text_frame.paragraphs and shape.text_frame.paragraphs[0].runs:
                            orig_run = shape.text_frame.paragraphs[0].runs[0]
                            saved_size = orig_run.font.size
                            saved_color = orig_run.font.color.rgb
                            saved_bold = orig_run.font.bold
                            saved_italic = orig_run.font.italic
                        else:
                            saved_size = saved_color = saved_bold = saved_italic = None

                        new_text = shape.text.replace(pattern, subtitle_text)
                        shape.text_frame.clear()
                        new_para = shape.text_frame.paragraphs[0]
                        new_run = new_para.add_run()
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
# Helpers to load from file-like object
# ---------------------------------------------------------------------------
def _load_price_data_from_obj(excel_obj, ticker: str = "SPX Index") -> pd.DataFrame:
    df = pd.read_excel(excel_obj, sheet_name="data_prices")
    df = df.drop(index=0)
    df = df[df[df.columns[0]] != "DATES"]
    df["Date"] = pd.to_datetime(df[df.columns[0]], errors="coerce")
    df["Price"] = pd.to_numeric(df[ticker], errors="coerce")
    return (
        df.dropna(subset=["Date", "Price"])
          .sort_values("Date")
          .reset_index(drop=True)[["Date", "Price"]]
    )