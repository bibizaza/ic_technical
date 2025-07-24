"""
ytd_perf/equity_ytd.py

This module provides functions to compute YTD performance for equities,
create a chart with connectors and bold labels, and insert it into
slide 11 of a PowerPoint file.  It loads data via loader_update.py.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pptx import Presentation
from pptx.util import Inches
import tempfile
import os
from typing import List, Optional
from datetime import datetime
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
    Compute YTD performance for equity tickers.  If `tickers` is None, all equity tickers
    from the parameters sheet are included.  YTD is calculated from 1 January of the
    current year.
    """
    prices_df, params_df = load_data(file_path)
    # Determine year start as naive datetime
    now = datetime.now()
    year_start = datetime(now.year, 1, 1)
    # Filter current year data
    current_year_df = prices_df[prices_df["Date"] >= year_start].reset_index(drop=True)
    # Find equity tickers from parameters
    eq_params = params_df[params_df["Asset Class"] == "Equity"]
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
        # Use first non-NaN value of the year as base
        base_series = series.dropna()
        if base_series.empty:
            continue
        base_val = base_series.iloc[0]
        if base_val == 0 or pd.isna(base_val):
            continue
        ytd = (series / base_val - 1.0) * 100.0
        result[name] = ytd.reset_index(drop=True)
    return result

def create_equity_chart(df: pd.DataFrame):
    """Create a YTD equity chart with connectors and bold labels."""
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
    # Style
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

def insert_equity_chart(prs: Presentation, file_path: str, subtitle: str = "", tickers: Optional[List[str]] = None) -> Presentation:
    """
    Compute equity YTD data, create the chart, insert it into slide 11 and update its subtitle.
    """
    df_eq = get_equity_ytd_series(file_path, tickers)
    fig = create_equity_chart(df_eq)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_png:
        fig.savefig(tmp_png.name, dpi=200)
        chart_path = tmp_png.name
    try:
        slide = prs.slides[10]
        if subtitle:
            for shape in slide.shapes:
                if shape.name == "YTD_EQ_Perf" and shape.has_text_frame:
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
        left = Inches(1.87 / 2.54)
        top = Inches(5.49 / 2.54)
        width = Inches(20.64 / 2.54)
        height = Inches(9.57 / 2.54)
        picture = slide.shapes.add_picture(chart_path, left, top, width, height)
        sp_tree = slide.shapes._spTree
        sp_tree.remove(picture._element)
        sp_tree.insert(1, picture._element)
        return prs
    finally:
        os.remove(chart_path)
