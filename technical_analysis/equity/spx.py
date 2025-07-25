"""
technical_analysis/equity/spx.py
================================

Utility functions for SPX technical analysis and high-resolution export.

Functions:
- make_spx_figure: Interactive Plotly chart for Streamlit.
- insert_spx_technical_chart: Insert an SPX technical chart into PPT, using a high-quality PNG
  with transparent background, fixed dimensions (21.41 cm × 7.53 cm), and positioned at
  0.93 cm from the left and 4.39 cm from the top (if no placeholder).
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
    Build the interactive SPX chart (Plotly) for Streamlit.

    Parameters
    ----------
    excel_path   : Path to the Excel file containing SPX prices.
    anchor_date  : pandas.Timestamp or None. If provided, a regression channel
                   is drawn from `anchor_date` to the most recent date.

    Returns
    -------
    go.Figure
    """
    excel_path = pathlib.Path(excel_path)
    df_full = _add_mas(_load_price_data(excel_path, "SPX Index"))

    today  = df_full["Date"].max().normalize()
    start  = today - timedelta(days=365)  # 1-year window
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
        trend  = model.predict(X)                 # fitted values
        resid  = y - trend                        # residuals
        upper  = trend + resid.max()              # trend + max residual
        lower  = trend + resid.min()              # trend + min residual

        uptrend = model.coef_[0] > 0              # slope sign
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
# Matplotlib image generation for PPT export
# ---------------------------------------------------------------------------
def _generate_spx_image_from_df(
    df_full: pd.DataFrame,
    anchor_date: Optional[pd.Timestamp],
    width_cm: float = 21.41,
    height_cm: float = 7.53,
) -> bytes:
    """
    Create a high-resolution, transparent-background Matplotlib chart image.
    It includes price, moving averages, Fibonacci levels, and optional regression
    channel. Returns PNG bytes.
    """
    # Restrict to 1-year window
    today = df_full["Date"].max().normalize()
    start = today - timedelta(days=365)
    df = df_full[df_full["Date"].between(start, today)].reset_index(drop=True)

    # Compute moving averages
    df_ma = df.copy()
    for w in (50, 100, 200):
        df_ma[f"MA_{w}"] = df_ma["Price"].rolling(w, min_periods=1).mean()

    # Prepare regression channel if needed
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
            subset = subset.reset_index(drop=True)
        else:
            subset = None

    # Figure dimensions (convert cm to inches)
    fig_width_in = width_cm / 2.54
    fig_height_in = height_cm / 2.54

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))

    # Price line
    ax.plot(df["Date"], df["Price"], color="#153D64", linewidth=2.5, label="S&P 500 Price")
    # Moving averages
    ax.plot(df_ma["Date"], df_ma["MA_50"], color="#008000", linewidth=1.5, label="50-day MA")
    ax.plot(df_ma["Date"], df_ma["MA_100"], color="#FFA500", linewidth=1.5, label="100-day MA")
    ax.plot(df_ma["Date"], df_ma["MA_200"], color="#FF0000", linewidth=1.5, label="200-day MA")

    # Fibonacci levels
    hi = df["Price"].max()
    lo = df["Price"].min()
    span = hi - lo
    fib_levels = [hi, hi - 0.236 * span, hi - 0.382 * span, hi - 0.5 * span,
                  hi - 0.618 * span, lo]
    for lvl in fib_levels:
        ax.axhline(y=lvl, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    # Regression channel shading
    if anchor_date is not None and upper is not None and lower is not None:
        fill_color = (0, 0.6, 0, 0.25) if uptrend else (0.78, 0, 0, 0.25)
        line_color = "#008000" if uptrend else "#C00000"
        subset = df_full[df_full["Date"].between(anchor_date, today)].copy().reset_index(drop=True)
        ax.plot(subset["Date"], upper, color=line_color, linestyle="--")
        ax.plot(subset["Date"], lower, color=line_color, linestyle="--")
        ax.fill_between(subset["Date"], lower, upper, color=fill_color)

    # Remove spines but keep tick labels
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Retain tick labels
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    # Legend
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize=8, frameon=False)
    plt.tight_layout()

    # Save as transparent PNG
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ---------------------------------------------------------------------------
# PPT insertion functions
# ---------------------------------------------------------------------------
def _load_price_data_from_obj(excel_obj, ticker: str = "SPX Index") -> pd.DataFrame:
    """
    Load SPX price data directly from a file-like object (such as an uploaded
    file from Streamlit). Returns a DataFrame with 'Date' and 'Price' columns.
    """
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

def insert_spx_technical_chart(
    prs: Presentation,
    excel_file,
    anchor_date: Optional[pd.Timestamp] = None
) -> Presentation:
    """
    Insert the SPX technical-analysis chart into the given PowerPoint presentation.
    It searches for a slide containing a shape with the text 'tech_spx' (case-insensitive).
    If found, the chart is inserted at that location; otherwise, it defaults to slide 12
    (or the last slide). It uses fixed dimensions: width=21.41 cm, height=7.53 cm,
    left=0.93 cm, top=4.39 cm. The chart has a transparent background.
    """
    # Load price data from Excel object or path
    try:
        df_full = _load_price_data_from_obj(excel_file, ticker="SPX Index")
    except Exception:
        path = pathlib.Path(excel_file)
        df_full = _load_price_data(path, "SPX Index")

    # Generate image bytes (transparent background)
    img_bytes = _generate_spx_image_from_df(df_full, anchor_date)

    # Search for placeholder slide
    placeholder_text = "tech_spx"
    slide_index = None
    placeholder_shape = None

    for idx, slide in enumerate(prs.slides):
        for shape in slide.shapes:
            if shape.has_text_frame and placeholder_text.lower() in shape.text.lower():
                slide_index = idx
                placeholder_shape = shape
                break
        if slide_index is not None:
            break

    # Determine slide and position
    if slide_index is not None:
        slide = prs.slides[slide_index]
        # Use the placeholder's position and size to place the image
        left = placeholder_shape.left
        top = placeholder_shape.top
        width = placeholder_shape.width
        height = placeholder_shape.height
        if placeholder_shape.has_text_frame:
            placeholder_shape.text = ""
    else:
        slide_index = min(11, len(prs.slides) - 1)
        slide = prs.slides[slide_index]
        left = Cm(0.93)
        top = Cm(4.39)
        width = Cm(21.41)
        height = Cm(7.53)

    # Insert the image
    stream = BytesIO(img_bytes)
    slide.shapes.add_picture(stream, left, top, width=width, height=height)

    return prs