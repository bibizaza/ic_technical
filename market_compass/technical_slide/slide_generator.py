"""Technical Analysis slide generator - HTML tables to image.

Renders the 3 tables (Equity, Commodity, Crypto) as an image and inserts
into the existing slide at the technical_nutshell placeholder position.

The slide template already has: title, subtitle, logo, header bar, footer.
We just need the tables.
"""

import tempfile
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from jinja2 import Template
from html2image import Html2Image
from pptx import Presentation
from pptx.util import Cm

from .html_template import TABLES_HTML_TEMPLATE
from .data_prep import AssetRow


# Image dimensions (pixels) - tables only
IMAGE_WIDTH = 1180
IMAGE_HEIGHT = 480


def _get_rsi_class(rsi: int) -> str:
    """Get CSS class for RSI value."""
    if rsi > 70:
        return "rsi-overbought"
    elif rsi < 30:
        return "rsi-oversold"
    return "neutral"


def _get_ma_class(vs_50d: float) -> str:
    """Get CSS class for vs 50d MA value."""
    return "positive" if vs_50d >= 0 else "negative"


def _get_dmas_class(dmas: int) -> str:
    """Get CSS class for DMAS value."""
    if dmas >= 55:
        return "positive"
    elif dmas < 45:
        return "negative"
    return "neutral"


def _prepare_row(row: AssetRow) -> dict:
    """Convert AssetRow to template dict."""
    dmas = int(row.dmas)
    rsi = int(row.rsi)
    vs_50d = row.vs_50d_ma

    return {
        "name": row.name,
        "market_cap": row.market_cap,
        "rsi": rsi,
        "rsi_class": _get_rsi_class(rsi),
        "vs_50d_ma_fmt": f"{vs_50d:+.1f}%",
        "ma_class": _get_ma_class(vs_50d),
        "dmas": dmas,
        "dmas_class": _get_dmas_class(dmas),
        "outlook": row.outlook,
        "outlook_lower": row.outlook.lower(),
    }


def _generate_tables_html(rows: List[AssetRow]) -> str:
    """Generate HTML for tables only."""
    equity = [_prepare_row(r) for r in rows if r.asset_class == "equity"]
    commodity = [_prepare_row(r) for r in rows if r.asset_class == "commodities"]
    crypto = [_prepare_row(r) for r in rows if r.asset_class == "crypto"]

    print(f"[Technical Nutshell] Prepared rows - Equity: {len(equity)}, Commodity: {len(commodity)}, Crypto: {len(crypto)}")

    template = Template(TABLES_HTML_TEMPLATE)
    return template.render(
        equity_rows=equity,
        commodity_rows=commodity,
        crypto_rows=crypto,
    )


def _html_to_png(html: str, output_path: str) -> str:
    """Convert HTML to PNG using html2image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hti = Html2Image(output_path=tmpdir, size=(IMAGE_WIDTH, IMAGE_HEIGHT))
        hti.screenshot(html_str=html, save_as="tables.png")

        shutil.move(str(Path(tmpdir) / "tables.png"), output_path)

    return output_path


def insert_technical_analysis_slide(
    prs: Presentation,
    rows: List[AssetRow],
    placeholder_name: str = "technical_nutshell",
    used_date: Optional[datetime] = None,
    price_mode: str = "Last Price"
) -> Presentation:
    """
    Generate tables as image and insert at placeholder position.

    Parameters
    ----------
    prs : Presentation
        The PowerPoint presentation object
    rows : List[AssetRow]
        List of asset rows with data
    placeholder_name : str
        Name of the placeholder shape to find and replace
    used_date : datetime, optional
        Date to display (not used in tables-only mode)
    price_mode : str
        Price mode (not used in tables-only mode)

    Returns
    -------
    Presentation
        The modified presentation
    """
    print(f"[Technical Nutshell] Generating HTML tables with {len(rows)} assets...")

    # Generate HTML
    html = _generate_tables_html(rows)

    # Convert to PNG
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name

    print(f"[Technical Nutshell] Converting to image...")
    _html_to_png(html, img_path)

    # Find placeholder
    target_slide = None
    placeholder_shape = None

    print(f"[Technical Nutshell] Searching for placeholder '{placeholder_name}'...")

    for slide_idx, slide in enumerate(prs.slides):
        for shape in slide.shapes:
            name = getattr(shape, "name", "")
            if placeholder_name.lower() in name.lower():
                target_slide = slide
                placeholder_shape = shape
                print(f"[Technical Nutshell] Found on slide {slide_idx + 1}")
                break
        if target_slide:
            break

    if not target_slide:
        print(f"[Technical Nutshell] No placeholder found, creating new slide")
        target_slide = prs.slides.add_slide(prs.slide_layouts[6])
        # Default position
        left, top, width, height = Cm(1), Cm(4), Cm(23), Cm(12)
    else:
        # Get placeholder position and size
        left = placeholder_shape.left
        top = placeholder_shape.top
        width = placeholder_shape.width
        height = placeholder_shape.height

        # Remove placeholder
        sp = placeholder_shape._element
        sp.getparent().remove(sp)

    # Insert image at placeholder position
    target_slide.shapes.add_picture(
        img_path,
        left=left,
        top=top,
        width=width,
        height=height
    )

    # Cleanup
    Path(img_path).unlink(missing_ok=True)

    print(f"[Technical Nutshell] ✅ Tables inserted successfully")

    return prs


# Backwards compatibility
def generate_technical_analysis_slide(
    prs: Presentation,
    rows: List[AssetRow],
    used_date: Optional[datetime] = None,
    price_mode: str = "Last Price"
) -> Presentation:
    """Generate slide (backwards compatibility wrapper)."""
    return insert_technical_analysis_slide(prs, rows, "technical_nutshell", used_date, price_mode)
