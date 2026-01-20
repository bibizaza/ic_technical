"""Breadth Rank slide generator."""

from dataclasses import dataclass
from typing import List, Optional
import tempfile
from pathlib import Path

import pandas as pd
from jinja2 import Template
from html2image import Html2Image
from pptx import Presentation
from pptx.util import Cm

from .html_template import BREADTH_HTML_TEMPLATE
from helpers.flag_utils import get_flag_html


# =============================================================================
# RESOLUTION SETTINGS (Full-width standalone slide)
# =============================================================================

SCALE_FACTOR = 4
BREADTH_BASE_WIDTH = 750  # Wider for full-width display
BREADTH_BASE_HEIGHT = 360  # Taller to fit all 9 rows
BREADTH_WIDTH_PX = BREADTH_BASE_WIDTH * SCALE_FACTOR   # 3000
BREADTH_HEIGHT_PX = BREADTH_BASE_HEIGHT * SCALE_FACTOR  # 1440

# PowerPoint placement (cm)
BREADTH_LEFT_CM = 2.7    # Horizontal position
BREADTH_TOP_CM = 7.25    # Vertical position
BREADTH_WIDTH_CM = 20.0  # Full width
BREADTH_HEIGHT_CM = 9.5  # Taller to fit all 9 rows


# =============================================================================
# INDEX NAME MAPPING
# =============================================================================

# Map tickers to (display_name, flag_code)
INDEX_NAME_MAP = {
    "SPX Index": ("U.S.", "us"),
    "SHSZ300 Index": ("China", "cn"),
    "NKY Index": ("Japan", "jp"),
    "SASEIDX Index": ("Saudi Arabia", "sa"),
    "SENSEX Index": ("India", "in"),
    "DAX Index": ("Germany", "de"),
    "SMI Index": ("Switzerland", "ch"),
    "MEXBOL Index": ("Mexico", "mx"),
    "IBOV Index": ("Brazil", "br"),
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BreadthRow:
    """Data for a single row in the Breadth Rank table."""
    index_name: str
    flag_code: str  # ISO country code for flag
    rank: int
    pct_both: int   # % above both MAs
    pct_20d: int    # % above 20D MA
    pct_50d: int    # % above 50D MA


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_breadth_data(excel_path: str) -> List[BreadthRow]:
    """
    Prepare Breadth Rank data from Excel sheet.

    Parameters
    ----------
    excel_path : str
        Path to Excel file

    Returns
    -------
    List[BreadthRow]
        List of breadth data rows, sorted by % above both MAs
    """
    try:
        # Read the helper_breadth sheet
        df = pd.read_excel(excel_path, sheet_name="helper_breadth")
        print(f"[Breadth Rank] Loaded sheet with {len(df)} rows, columns: {list(df.columns)[:10]}")

        # Get column by position (0-indexed, so C=2, H=7, I=8, J=9)
        # But let's be flexible and try to find them
        cols = df.columns.tolist()

        # Column C (index 2) = Ticker
        # Column H (index 7) = % above both MA
        # Column I (index 8) = % above 20D MA
        # Column J (index 9) = % above 50D MA

        if len(cols) < 10:
            print(f"[Breadth Rank] Warning: Expected at least 10 columns, got {len(cols)}")
            return []

        ticker_col = cols[2]  # Column C
        pct_both_col = cols[7]  # Column H
        pct_20d_col = cols[8]  # Column I
        pct_50d_col = cols[9]  # Column J

        print(f"[Breadth Rank] Using columns: ticker={ticker_col}, both={pct_both_col}, 20d={pct_20d_col}, 50d={pct_50d_col}")

        # Extract data
        rows = []
        for _, row in df.iterrows():
            ticker = str(row[ticker_col]).strip() if pd.notna(row[ticker_col]) else ""

            # Skip empty or header rows
            if not ticker or ticker.lower() in ["ticker", "index", ""]:
                continue

            # Get display name and flag code
            name_flag = INDEX_NAME_MAP.get(ticker)
            if name_flag:
                display_name, flag_code = name_flag
            else:
                display_name = ticker.replace(" Index", "")
                flag_code = ""

            # Get percentages (convert to int, handle NaN)
            try:
                pct_both = int(round(float(row[pct_both_col]) * 100)) if pd.notna(row[pct_both_col]) else 0
                pct_20d = int(round(float(row[pct_20d_col]) * 100)) if pd.notna(row[pct_20d_col]) else 0
                pct_50d = int(round(float(row[pct_50d_col]) * 100)) if pd.notna(row[pct_50d_col]) else 0
            except (ValueError, TypeError):
                # Values might already be percentages (0-100)
                pct_both = int(row[pct_both_col]) if pd.notna(row[pct_both_col]) else 0
                pct_20d = int(row[pct_20d_col]) if pd.notna(row[pct_20d_col]) else 0
                pct_50d = int(row[pct_50d_col]) if pd.notna(row[pct_50d_col]) else 0

            # Clamp to 0-100
            pct_both = max(0, min(100, pct_both))
            pct_20d = max(0, min(100, pct_20d))
            pct_50d = max(0, min(100, pct_50d))

            rows.append(BreadthRow(
                index_name=display_name,
                flag_code=flag_code,
                rank=0,  # Will be set after sorting
                pct_both=pct_both,
                pct_20d=pct_20d,
                pct_50d=pct_50d,
            ))

        # Sort by % above both MAs (descending)
        rows.sort(key=lambda r: r.pct_both, reverse=True)

        # Assign ranks
        for i, row in enumerate(rows):
            row.rank = i + 1

        print(f"[Breadth Rank] Prepared {len(rows)} rows")
        for row in rows:
            print(f"  {row.rank}. {row.index_name}: {row.pct_both}% / {row.pct_20d}% / {row.pct_50d}%")

        return rows

    except Exception as e:
        print(f"[Breadth Rank] Error reading data: {e}")
        import traceback
        traceback.print_exc()
        return []


# =============================================================================
# HTML GENERATION
# =============================================================================

def _get_pct_class(value: int) -> str:
    """Get CSS class for percentage value."""
    if value >= 55:
        return "high"
    elif value >= 45:
        return "med-high"
    elif value >= 35:
        return "med"
    elif value >= 25:
        return "med-low"
    else:
        return "low"


def _prepare_row_for_template(row: BreadthRow) -> dict:
    """Prepare row for Jinja template."""
    # Generate flag HTML (size 22 to match other slides)
    flag_html = get_flag_html(row.flag_code, size=22) if row.flag_code else ""

    return {
        "index_name": row.index_name,
        "flag_html": flag_html,
        "rank": row.rank,
        "pct_both": row.pct_both,
        "pct_both_class": _get_pct_class(row.pct_both),
        "pct_20d": row.pct_20d,
        "pct_20d_class": _get_pct_class(row.pct_20d),
        "pct_50d": row.pct_50d,
        "pct_50d_class": _get_pct_class(row.pct_50d),
    }


def _generate_html(rows: List[BreadthRow]) -> str:
    """Generate HTML from data."""
    template = Template(BREADTH_HTML_TEMPLATE)
    return template.render(
        rows=[_prepare_row_for_template(r) for r in rows],
        width=BREADTH_WIDTH_PX,
        height=BREADTH_HEIGHT_PX,
        scale=SCALE_FACTOR,
    )


def _html_to_png(html: str, output_path: str) -> str:
    """Convert HTML to PNG image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hti = Html2Image(
            output_path=tmpdir,
            size=(BREADTH_WIDTH_PX, BREADTH_HEIGHT_PX)
        )
        hti.screenshot(html_str=html, save_as="breadth.png")

        import shutil
        shutil.move(str(Path(tmpdir) / "breadth.png"), output_path)

    return output_path


# =============================================================================
# SLIDE INSERTION
# =============================================================================

def insert_breadth_rank(
    prs: Presentation,
    rows: List[BreadthRow],
    slide_title: str = "Market Breadth"
) -> Presentation:
    """
    Insert Breadth Rank table into PowerPoint slide.

    Finds the slide by title text and inserts the table at a fixed position.
    Does NOT remove any shapes - just adds the image.

    Parameters
    ----------
    prs : Presentation
        PowerPoint presentation object
    rows : List[BreadthRow]
        List of breadth data rows
    slide_title : str
        Title text to search for to find the correct slide

    Returns
    -------
    Presentation
        Modified presentation
    """
    if not rows:
        print("[Breadth Rank] No data to display")
        return prs

    print(f"[Breadth Rank] Generating table with {len(rows)} rows...")
    print(f"[Breadth Rank] Resolution: {BREADTH_WIDTH_PX}x{BREADTH_HEIGHT_PX}px (4x)")

    # Generate HTML
    html = _generate_html(rows)

    # Convert to PNG
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name

    _html_to_png(html, img_path)

    # Find slide by title text (safer than placeholder removal)
    target_slide = None

    for slide_idx, slide in enumerate(prs.slides):
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                if slide_title.lower() in shape.text.lower():
                    target_slide = slide
                    print(f"[Breadth Rank] Found slide with '{slide_title}' at index {slide_idx + 1}")
                    break
        if target_slide:
            break

    if not target_slide:
        print(f"[Breadth Rank] No slide with title '{slide_title}' found")
        Path(img_path).unlink(missing_ok=True)
        return prs

    # Insert image at FIXED position - don't remove any shapes
    target_slide.shapes.add_picture(
        img_path,
        left=Cm(BREADTH_LEFT_CM),
        top=Cm(BREADTH_TOP_CM),
        width=Cm(BREADTH_WIDTH_CM),
        height=Cm(BREADTH_HEIGHT_CM)
    )

    # Cleanup
    Path(img_path).unlink(missing_ok=True)

    print(f"[Breadth Rank] ✅ Table inserted at ({BREADTH_LEFT_CM}, {BREADTH_TOP_CM}) cm")

    return prs


def generate_breadth_slide(
    prs: Presentation,
    excel_path: str,
    slide_title: str = "Market Breadth"
) -> Presentation:
    """
    Generate Breadth Rank slide from Excel data.

    Parameters
    ----------
    prs : Presentation
        PowerPoint presentation object
    excel_path : str
        Path to Excel file with helper_breadth sheet
    slide_title : str
        Title text to search for to find the correct slide

    Returns
    -------
    Presentation
        Modified presentation
    """
    # Prepare data
    rows = prepare_breadth_data(excel_path)

    if not rows:
        print("[Breadth Rank] No data found in helper_breadth sheet")
        return prs

    # Insert into slide
    return insert_breadth_rank(prs, rows, slide_title)
