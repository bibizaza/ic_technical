"""Corporate Bonds Weekly Performance chart generation.

This module generates a weekly performance bar chart for corporate bonds,
showing Investment Grade (IG) and High Yield (HY) bonds sorted by performance.
Each bond displays a flag, credit badge (IG/HY), and performance bar.

Functions
---------
``create_weekly_performance_chart``
    Generate the corporate bonds weekly performance PNG.
``insert_corp_bonds_performance_slide``
    Insert the chart into a PowerPoint slide.
"""

from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from jinja2 import Template
from html2image import Html2Image
from pptx import Presentation
from pptx.util import Cm

from utils import adjust_prices_for_mode
from market_compass.weekly_performance.html_template import CORP_BONDS_WEEKLY_HTML_TEMPLATE


# Scale factor for PNG generation
SCALE_FACTOR = 3

# Corporate bond configuration - uses Bloomberg index tickers from data_prices sheet
# Maps ticker to (display_name, flag, credit_type)
CORP_BOND_CONFIG = {
    # Investment Grade
    "LUACTRUU Index": {"name": "US IG Corp", "flag": "\U0001F1FA\U0001F1F8", "credit": "IG"},
    "LECPTREU Index": {"name": "EUR IG Corp", "flag": "\U0001F1EA\U0001F1FA", "credit": "IG"},
    "JPEIESGE Index": {"name": "EM IG Corp", "flag": "\U0001F30D", "credit": "IG"},
    # High Yield
    "LF98TRUU Index": {"name": "US HY Corp", "flag": "\U0001F1FA\U0001F1F8", "credit": "HY"},
    "IBOXXMJA Index": {"name": "EUR HY Corp", "flag": "\U0001F1EA\U0001F1FA", "credit": "HY"},
    "JPEIEMHY Index": {"name": "EM HY Corp", "flag": "\U0001F30D", "credit": "HY"},
}


@dataclass
class CorpBondRow:
    """Data class for a corporate bond row."""
    name: str           # e.g., "US IG Corp"
    flag: str           # e.g., "🇺🇸"
    credit_type: str    # "IG" or "HY"
    value: float        # Weekly return as decimal (e.g., 0.0032 for 0.32%)


def _load_price_data(
    excel_path: Union[str, Path],
    tickers: List[str],
) -> pd.DataFrame:
    """Load price data for the specified tickers.

    Parameters
    ----------
    excel_path : str or Path
        Path to the Excel workbook containing ``data_prices`` sheet.
    tickers : list of str
        Column names for the desired bonds.

    Returns
    -------
    DataFrame
        A DataFrame with ``Date`` column and one column per ticker.
    """
    df = pd.read_excel(excel_path, sheet_name="data_prices")
    df = df.drop(index=0)
    df = df[df[df.columns[0]] != "DATES"].copy()
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Get available tickers
    available_tickers = [t for t in tickers if t in df.columns]
    if not available_tickers:
        return pd.DataFrame(columns=["Date"])

    out = df[["Date"] + available_tickers].copy()
    for t in available_tickers:
        out[t] = pd.to_numeric(out[t], errors="coerce")

    return out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def _compute_weekly_return(
    df: pd.DataFrame,
    ticker: str,
) -> float:
    """Compute 1-week return for a ticker.

    Parameters
    ----------
    df : DataFrame
        Price data with Date column.
    ticker : str
        Column name for the ticker.

    Returns
    -------
    float
        Weekly return as decimal, or NaN if not computable.
    """
    if ticker not in df.columns:
        return float("nan")

    today = df["Date"].max()
    past_date = today - pd.Timedelta(days=7)
    past_series = df.loc[df["Date"] <= past_date, ticker]

    if len(past_series) == 0:
        return float("nan")

    past_price = past_series.iloc[-1]
    current_price = df[ticker].iloc[-1]

    if past_price == 0 or pd.isna(past_price) or pd.isna(current_price):
        return float("nan")

    return (current_price - past_price) / past_price


def create_weekly_performance_chart(
    excel_path: Union[str, Path],
    *,
    price_mode: str = "Last Price",
    width_cm: float = 17.31,
    height_cm: float = 10.0,
) -> Tuple[bytes, Optional[pd.Timestamp]]:
    """Generate the Corporate Bonds Weekly Performance chart.

    Parameters
    ----------
    excel_path : str or Path
        Path to the Excel workbook.
    price_mode : str, default "Last Price"
        Either "Last Price" or "Last Close".
    width_cm, height_cm : float
        Dimensions for the chart in centimeters.

    Returns
    -------
    tuple
        (png_bytes, used_date) where png_bytes is the chart image and
        used_date is the effective date used.
    """
    tickers = list(CORP_BOND_CONFIG.keys())

    # Load and adjust prices
    df = _load_price_data(excel_path, tickers)
    df_adj, used_date = adjust_prices_for_mode(df, price_mode)

    # Build rows with weekly returns
    rows: List[CorpBondRow] = []

    for ticker, config in CORP_BOND_CONFIG.items():
        if ticker not in df_adj.columns:
            print(f"[Corp Bonds Weekly] DEBUG: Ticker {ticker} not found in data")
            continue

        weekly_ret = _compute_weekly_return(df_adj, ticker)
        if pd.isna(weekly_ret):
            print(f"[Corp Bonds Weekly] DEBUG: Ticker {ticker} has NaN weekly return")
            continue

        rows.append(CorpBondRow(
            name=config["name"],
            flag=config["flag"],
            credit_type=config["credit"],
            value=weekly_ret,
        ))
        print(f"[Corp Bonds Weekly] DEBUG: {config['name']}: {weekly_ret * 100:.2f}%")

    print(f"[Corp Bonds Weekly] DEBUG: Total bonds found: {len(rows)}")

    if not rows:
        # Return empty image if no data
        return b"", used_date

    # Sort by value descending (best to worst)
    rows.sort(key=lambda x: x.value, reverse=True)

    # Calculate max for bar scaling
    max_abs_value = max(abs(r.value) for r in rows)
    max_abs_value = max(max_abs_value, 0.005)  # At least 0.5%

    # Prepare rows for template
    prepared_rows = []
    for i, row in enumerate(rows):
        # Bar width as percentage of half the chart
        bar_width = abs(row.value) / max_abs_value * 50
        bar_width = min(bar_width, 48)

        # Determine highlight class
        if i == 0 and row.value > 0:
            highlight_class = "top-performer"
        elif i == len(rows) - 1 and row.value < 0:
            highlight_class = "worst-performer"
        else:
            highlight_class = ""

        prepared_rows.append({
            "name": row.name,
            "flag": row.flag,
            "credit_type": row.credit_type,
            "credit_class": "ig" if row.credit_type == "IG" else "hy",
            "value": row.value,
            "bar_class": "positive" if row.value >= 0 else "negative",
            "bar_width": bar_width,
            "value_class": "positive" if row.value >= 0 else "negative",
            "formatted_value": f"+{row.value * 100:.2f}%" if row.value >= 0 else f"{row.value * 100:.2f}%",
            "highlight_class": highlight_class,
        })

    # Calculate scale values
    scale_pct = max_abs_value * 100
    if scale_pct <= 0.5:
        scale_max = 0.5
    elif scale_pct <= 1:
        scale_max = 1
    elif scale_pct <= 2:
        scale_max = 2
    elif scale_pct <= 5:
        scale_max = 5
    else:
        scale_max = round(scale_pct) + 1

    scale_values = {
        "scale_min": f"-{scale_max:.1f}%",
        "scale_mid_low": f"-{scale_max / 2:.1f}%",
        "scale_mid_high": f"+{scale_max / 2:.1f}%",
        "scale_max": f"+{scale_max:.1f}%",
    }

    # Generate HTML - use EXACT hardcoded dimensions
    # DO NOT recalculate these values
    PNG_WIDTH_PX = 1963
    PNG_HEIGHT_PX = 1134

    template = Template(CORP_BONDS_WEEKLY_HTML_TEMPLATE)
    html = template.render(
        rows=prepared_rows,
        width=PNG_WIDTH_PX,
        height=PNG_HEIGHT_PX,
        scale=SCALE_FACTOR,
        **scale_values,
    )

    # Convert to PNG at EXACT size
    with tempfile.TemporaryDirectory() as tmpdir:
        hti = Html2Image(output_path=tmpdir, size=(1963, 1134))
        hti.screenshot(html_str=html, save_as="corp_bonds_weekly.png")

        img_path = Path(tmpdir) / "corp_bonds_weekly.png"
        with open(img_path, "rb") as f:
            png_bytes = f.read()

    return png_bytes, used_date


def insert_corp_bonds_performance_slide(
    prs: Presentation,
    image_bytes: bytes,
    used_date: Optional[pd.Timestamp] = None,
    price_mode: str = "Last Price",
    *,
    left_cm: float = 3.35,
    top_cm: float = 4.6,
    width_cm: float = 17.31,
    height_cm: float = 10.0,
) -> Presentation:
    """Insert the Corporate Bonds Weekly Performance chart into a slide.

    The slide is identified by a shape named ``corp_bonds_perf_1w`` or
    containing ``[corp_bonds_perf_1w]``.

    Parameters
    ----------
    prs : Presentation
        PowerPoint presentation to modify.
    image_bytes : bytes
        PNG data for the chart.
    used_date : Timestamp, optional
        Effective date for the source footnote.
    price_mode : str
        Either "Last Price" or "Last Close".
    left_cm, top_cm, width_cm, height_cm : float
        Position and size for the chart.

    Returns
    -------
    Presentation
        The modified presentation.
    """
    # Find target slide
    target_slide = None
    placeholder_names = ["credit_perf_1w", "corp_bonds_perf_1w", "corp_bonds_weekly"]
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
        print("[Corp Bonds Weekly] WARNING: Slide not found")
        return prs

    # Insert picture at EXACT hardcoded PowerPoint dimensions
    # DO NOT modify these values
    stream = io.BytesIO(image_bytes)
    pic = target_slide.shapes.add_picture(
        stream,
        left=Cm(3.35),
        top=Cm(4.6),
        width=Cm(17.31),
        height=Cm(10.0),
    )

    # Send to back
    spTree = target_slide.shapes._spTree
    sp = pic._element
    spTree.remove(sp)
    spTree.insert(2, sp)

    # Insert source footnote if date available
    if used_date is not None:
        date_str = used_date.strftime("%d/%m/%Y")
        suffix = " Close" if price_mode.lower() == "last close" else ""
        source_text = f"Source: Bloomberg, Herculis Group, Data as of {date_str}{suffix}"

        source_names = ["credit_1w_source", "corp_bonds_1w_source"]
        source_candidates = [n.lower() for n in source_names]
        source_patterns = [f"[{n}]" for n in source_candidates]

        for shape in target_slide.shapes:
            name_attr = getattr(shape, "name", "").lower()
            if name_attr in source_candidates and shape.has_text_frame:
                tf = shape.text_frame
                if tf.paragraphs:
                    tf.paragraphs[0].runs[0].text = source_text if tf.paragraphs[0].runs else source_text
                break
            if shape.has_text_frame:
                for pattern in source_patterns:
                    if pattern.lower() in (shape.text or "").lower():
                        shape.text_frame.paragraphs[0].runs[0].text = source_text
                        break

    print(f"[Corp Bonds Weekly] Chart inserted")
    return prs
