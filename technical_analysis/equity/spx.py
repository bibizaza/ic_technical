"""
technical_analysis/equity/spx.py
================================

Utility functions for SPX technical analysis and high-resolution export.

Functions:
- make_spx_figure: Interactive Plotly chart for Streamlit.
- insert_spx_technical_chart: Insert an SPX technical chart into PPT.
- insert_spx_technical_score: Insert a score gauge into the corresponding slide.
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
    Build the interactive SPX chart for Streamlit. (Unchanged from prior version.)
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
        trend  = model.predict(X)                 # fitted values
        resid  = y - trend                        # residuals
        upper  = trend + resid.max()              # trend + max residual
        lower  = trend + resid.min()              # trend + min residual

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
# Matplotlib image generation for PPT export (unchanged from prior revision)
# ---------------------------------------------------------------------------
def _generate_spx_image_from_df(
    df_full: pd.DataFrame,
    anchor_date: Optional[pd.Timestamp],
    width_cm: float = 21.41,
    height_cm: float = 7.53,
) -> bytes:
    # (Same as previous version with transparent background and ticks retained)
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
            subset = subset.reset_index(drop=True)
        else:
            subset = None

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
    Retrieve the technical score for SPX from the sheet 'data_technical_score'.
    The sheet is expected to have the ticker in column A and the score in column B.
    Returns None if not found or on error.
    """
    try:
        df = pd.read_excel(excel_obj_or_path, sheet_name="data_technical_score")
    except Exception:
        return None

    # Expect columns A: ticker, B: score
    df = df.dropna(subset=[df.columns[0], df.columns[1]])
    for _, row in df.iterrows():
        if str(row[df.columns[0]]).strip().upper() == "SPX INDEX":
            try:
                return float(row[df.columns[1]])
            except Exception:
                return None
    return None

def _generate_score_gauge_image(score: Optional[float], width_cm=5.0, height_cm=1.0) -> bytes:
    """
    Create a simple horizontal gauge bar for the technical score.
    The gauge has red (0-33%), yellow (33-66%), and green (66-100%) segments.
    The score is normalized from 0 to 100 for positioning the indicator.
    Returns PNG bytes with transparent background.
    """
    # Normalize the score if it's not None; assume score in 0-10 range; clamp 0-10 then to 0-100
    if score is None:
        normalized = 50  # neutral mid-point
    else:
        try:
            val = float(score)
            val = max(0.0, min(10.0, val))  # clamp to 0-10
            normalized = val * 10.0  # map 0-10 -> 0-100
        except Exception:
            normalized = 50

    # Convert cm to inches
    fig_w = width_cm / 2.54
    fig_h = height_cm / 2.54

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Draw colored segments (red, yellow, green)
    ax.barh(0, 33, color="#C00000")     # red segment
    ax.barh(0, 33, left=33, color="#F6B26B")  # yellow segment
    ax.barh(0, 34, left=66, color="#6AA84F")  # green segment

    # Draw the indicator line
    ax.axvline(x=normalized, ymin=-0.2, ymax=0.2, color="black", linewidth=3)

    # No axis spines or ticks; but show value as text on top
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 33, 66, 100])
    ax.set_xticklabels([])
    # Show numeric value at bottom center
    ax.text(50, -0.4, f"Score: {score:.2f}" if score is not None else "Score: N/A",
            ha="center", va="center", fontsize=8)

    ax.set_facecolor((0, 0, 0, 0))  # transparent
    fig.patch.set_alpha(0.0)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def insert_spx_technical_score(
    prs: Presentation,
    excel_file
) -> Presentation:
    """
    Insert the SPX technical score gauge into the slide containing 'tech_score_spx'.
    If not found, this function does nothing. The gauge is created based on the
    'data_technical_score' sheet and replaces any placeholder text.
    """
    score = _get_spx_technical_score(excel_file)

    placeholder_text = "tech_score_spx"
    slide_index = None
    placeholder_shape = None

    # Find the placeholder shape
    for idx, slide in enumerate(prs.slides):
        for shape in slide.shapes:
            if shape.has_text_frame and placeholder_text.lower() in shape.text.lower():
                slide_index = idx
                placeholder_shape = shape
                break
        if slide_index is not None:
            break

    if slide_index is None:
        # Do nothing if not found
        return prs

    slide = prs.slides[slide_index]
    left = placeholder_shape.left
    top = placeholder_shape.top
    width = placeholder_shape.width
    height = placeholder_shape.height

    # Clear placeholder text
    if placeholder_shape.has_text_frame:
        placeholder_shape.text = ""

    # Generate gauge image using placeholder dimensions (convert EMU -> cm)
    width_cm = width.cm
    height_cm = height.cm
    gauge_img_bytes = _generate_score_gauge_image(score, width_cm=width_cm, height_cm=height_cm)

    # Insert gauge image into the same position/size
    stream = BytesIO(gauge_img_bytes)
    slide.shapes.add_picture(stream, left, top, width=width, height=height)

    return prs

# ---------------------------------------------------------------------------
# Chart insertion functions
# ---------------------------------------------------------------------------
def insert_spx_technical_chart(
    prs: Presentation,
    excel_file,
    anchor_date: Optional[pd.Timestamp] = None
) -> Presentation:
    """
    Insert the SPX technical-analysis chart into the given PowerPoint presentation.
    Searches for 'tech_spx' placeholder, or uses default slide 12. Fixed size and position.
    """
    # Load price data from Excel object or path
    try:
        df_full = _load_price_data_from_obj(excel_file, "SPX Index")
    except Exception:
        path = pathlib.Path(excel_file)
        df_full = _load_price_data(path, "SPX Index")

    img_bytes = _generate_spx_image_from_df(df_full, anchor_date)

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

    if slide_index is not None:
        slide = prs.slides[slide_index]
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

    stream = BytesIO(img_bytes)
    slide.shapes.add_picture(stream, left, top, width=width, height=height)

    return prs

def insert_spx_technical_score_number(
    prs: Presentation,
    excel_file,
) -> Presentation:
    """
    Insert the SPX technical score (integer, no decimals) into the shape named
    'tech_score_spx', or into a shape containing '[XXX]' or 'XXX', preserving
    the formatting of the original text.

    Parameters
    ----------
    prs : Presentation
        The PowerPoint Presentation object.
    excel_file : file-like or path-like
        The Excel file containing the 'data_technical_score' sheet.

    Returns
    -------
    Presentation
        The updated presentation with the score number inserted.
    """
    score = _get_spx_technical_score(excel_file)
    # Convert to integer (no decimals) if possible
    if score is None:
        score_text = "N/A"
    else:
        try:
            score_text = f"{int(round(float(score)))}"
        except Exception:
            score_text = str(score)

    placeholder_name = "tech_score_spx"
    placeholder_patterns = ["[XXX]", "XXX"]

    found = False

    for slide in prs.slides:
        for shape in slide.shapes:
            # Prefer matching by shape name
            if hasattr(shape, "name") and shape.name.lower() == placeholder_name:
                if shape.has_text_frame:
                    # Save original formatting from first run (if exists)
                    if shape.text_frame.paragraphs and shape.text_frame.paragraphs[0].runs:
                        orig_run = shape.text_frame.paragraphs[0].runs[0]
                        orig_font = orig_run.font
                        saved_size = orig_font.size
                        saved_color = orig_font.color.rgb
                        saved_bold = orig_font.bold
                        saved_italic = orig_font.italic
                    else:
                        # Fallback defaults
                        saved_size = None
                        saved_color = None
                        saved_bold = None
                        saved_italic = None

                    # Clear existing text and insert new number
                    shape.text_frame.clear()
                    paragraph = shape.text_frame.paragraphs[0]
                    run = paragraph.add_run()
                    run.text = score_text

                    # Apply saved formatting
                    if saved_size:
                        run.font.size = saved_size
                    if saved_color:
                        run.font.color.rgb = saved_color
                    if saved_bold is not None:
                        run.font.bold = saved_bold
                    if saved_italic is not None:
                        run.font.italic = saved_italic

                    found = True
                break

            # Otherwise, search for placeholder in text content
            if shape.has_text_frame:
                for pattern in placeholder_patterns:
                    if pattern in shape.text:
                        # Capture formatting from the first run
                        para = shape.text_frame.paragraphs[0]
                        if para.runs:
                            orig_run = para.runs[0]
                            orig_font = orig_run.font
                            saved_size = orig_font.size
                            saved_color = orig_font.color.rgb
                            saved_bold = orig_font.bold
                            saved_italic = orig_font.italic
                        else:
                            saved_size = None
                            saved_color = None
                            saved_bold = None
                            saved_italic = None

                        # Replace only the placeholder with score_text
                        new_text = shape.text.replace(pattern, score_text)
                        shape.text_frame.clear()
                        new_para = shape.text_frame.paragraphs[0]
                        new_run = new_para.add_run()
                        new_run.text = new_text

                        # Apply saved formatting to the number portion
                        if saved_size:
                            new_run.font.size = saved_size
                        if saved_color:
                            new_run.font.color.rgb = saved_color
                        if saved_bold is not None:
                            new_run.font.bold = saved_bold
                        if saved_italic is not None:
                            new_run.font.italic = saved_italic

                        found = True
                        break
            if found:
                break
        if found:
            break

    return prs

# Helper to load from file-like object
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