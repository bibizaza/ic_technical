"""PowerPoint slide generation for Technical Analysis - Two Column Layout."""

from datetime import datetime
from typing import List, Optional

from pptx import Presentation
from pptx.util import Cm, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

from .config import (
    COLORS, OTHER_COL_RATIOS, HEADERS,
    SLIDE_LAYOUT, TABLE_DIMS, HEADER_FONT_SIZE, DATA_FONT_SIZE, OUTLOOK_FONT_SIZE
)
from .data_prep import AssetRow


def _format_cell_text(cell, text: str, align: str = "center", size: int = 8,
                      color: RGBColor = None, bold: bool = False):
    """Helper to format cell text with vertical centering."""
    cell.text = str(text)

    # Vertical centering
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE

    p = cell.text_frame.paragraphs[0]

    if align == "center":
        p.alignment = PP_ALIGN.CENTER
    elif align == "left":
        p.alignment = PP_ALIGN.LEFT
    elif align == "right":
        p.alignment = PP_ALIGN.RIGHT

    p.space_before = Pt(0)
    p.space_after = Pt(0)

    if p.runs:
        run = p.runs[0]
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.name = "Calibri"
        if color:
            run.font.color.rgb = color
        else:
            run.font.color.rgb = COLORS["neutral_text"]


def _create_table(slide, rows: List[AssetRow], asset_class: str):
    """
    Create a single table for an asset class using exact cm dimensions.

    Parameters
    ----------
    slide : pptx.slide.Slide
        The slide to add the table to
    rows : List[AssetRow]
        Asset rows (already filtered by asset class)
    asset_class : str
        "equity", "commodities", or "crypto"
    """
    if not rows:
        print(f"[Technical Nutshell] No rows for asset class '{asset_class}'")
        return

    # Get dimensions from config
    dims = TABLE_DIMS.get(asset_class)
    if not dims:
        print(f"[Technical Nutshell] No dimensions for asset class '{asset_class}'")
        return

    n_rows = len(rows) + 1  # +1 for header
    n_cols = 6

    # Create table with exact cm dimensions
    table_shape = slide.shapes.add_table(
        n_rows, n_cols,
        Cm(dims["left"]), Cm(dims["top"]),
        Cm(dims["width"]), Cm(dims["height"])
    )
    table = table_shape.table

    # Set first column width (from config)
    first_col_width = dims.get("first_col_width", 2.98)
    table.columns[0].width = Cm(first_col_width)

    # Distribute remaining width to other columns based on ratios
    remaining_width = dims["width"] - first_col_width
    for i, ratio in enumerate(OTHER_COL_RATIOS):
        table.columns[i + 1].width = Cm(remaining_width * ratio)

    # Set row heights: header row uses header_height, data rows split remaining
    header_height = dims.get("header_height", 0.68)
    data_row_height = (dims["height"] - header_height) / (n_rows - 1) if n_rows > 1 else header_height

    for i, row in enumerate(table.rows):
        if i == 0:
            row.height = Cm(header_height)
        else:
            row.height = Cm(data_row_height)

    # ----- HEADER ROW -----
    headers = HEADERS.get(asset_class, HEADERS["equity"])

    for col_idx, header in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLORS["header_bg"]
        _format_cell_text(
            cell, header,
            align="left" if col_idx == 0 else "center",
            size=HEADER_FONT_SIZE,
            color=COLORS["header_text"],
            bold=True
        )

    # ----- DATA ROWS -----
    for row_idx, asset_row in enumerate(rows, start=1):
        # Alternating background
        bg_color = COLORS["row_white"] if row_idx % 2 == 1 else COLORS["row_grey"]

        # Column 0: Asset Name
        cell = table.cell(row_idx, 0)
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg_color
        _format_cell_text(cell, asset_row.name, align="left", size=DATA_FONT_SIZE)

        # Column 1: Market Cap
        cell = table.cell(row_idx, 1)
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg_color
        _format_cell_text(cell, asset_row.market_cap, align="center", size=DATA_FONT_SIZE)

        # Column 2: RSI (color coded)
        cell = table.cell(row_idx, 2)
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg_color
        rsi_val = int(asset_row.rsi)
        rsi_color = COLORS["neutral_text"]
        if rsi_val > 70:
            rsi_color = COLORS["negative"]  # Overbought
        elif rsi_val < 30:
            rsi_color = COLORS["positive"]  # Oversold
        _format_cell_text(cell, str(rsi_val), align="center", size=DATA_FONT_SIZE, color=rsi_color)

        # Column 3: vs 50d MA (color coded)
        cell = table.cell(row_idx, 3)
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg_color
        ma_val = asset_row.vs_50d_ma
        ma_text = f"{ma_val:+.1f}%"
        ma_color = COLORS["positive"] if ma_val >= 0 else COLORS["negative"]
        _format_cell_text(cell, ma_text, align="center", size=DATA_FONT_SIZE, color=ma_color)

        # Column 4: DMAS (INTEGER, color coded)
        cell = table.cell(row_idx, 4)
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg_color
        dmas_val = int(asset_row.dmas)
        if dmas_val >= 55:
            dmas_color = COLORS["positive"]
        elif dmas_val < 45:
            dmas_color = COLORS["negative"]
        else:
            dmas_color = COLORS["neutral_text"]
        _format_cell_text(cell, str(dmas_val), align="center", size=DATA_FONT_SIZE,
                          color=dmas_color, bold=True)

        # Column 5: Outlook (colored background)
        cell = table.cell(row_idx, 5)
        outlook_lower = asset_row.outlook.lower()
        bg_key = f"outlook_{outlook_lower}_bg"
        text_key = f"outlook_{outlook_lower}_text"

        if bg_key in COLORS:
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLORS[bg_key]
        else:
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg_color

        text_color = COLORS.get(text_key, COLORS["neutral_text"])
        _format_cell_text(cell, asset_row.outlook, align="center",
                          size=OUTLOOK_FONT_SIZE, color=text_color, bold=True)


def _add_content_to_slide(
    slide,
    rows: List[AssetRow],
    used_date: Optional[datetime] = None,
    price_mode: str = "Last Price"
):
    """
    Add Technical Analysis content to an existing slide using TWO-COLUMN layout.

    Left column: Equity (9 rows)
    Right column: Commodities (6 rows) + Crypto (5 rows)
    """
    layout = SLIDE_LAYOUT

    # Filter rows by asset class
    equity_rows = [r for r in rows if r.asset_class == "equity"]
    commo_rows = [r for r in rows if r.asset_class == "commodities"]
    crypto_rows = [r for r in rows if r.asset_class == "crypto"]

    print(f"[Technical Nutshell] Equity: {len(equity_rows)}, Commo: {len(commo_rows)}, Crypto: {len(crypto_rows)}")

    # ----- LEFT COLUMN: EQUITY -----
    _create_table(slide, equity_rows, asset_class="equity")

    # ----- RIGHT COLUMN: COMMODITIES -----
    _create_table(slide, commo_rows, asset_class="commodities")

    # ----- RIGHT COLUMN: CRYPTO (below commodities) -----
    _create_table(slide, crypto_rows, asset_class="crypto")

    # ----- FOOTER -----
    footer = slide.shapes.add_textbox(
        Cm(1.0), Cm(layout["footer_y"]),
        Cm(20), Cm(0.5)
    )
    tf = footer.text_frame
    p = tf.paragraphs[0]

    if used_date:
        date_str = used_date.strftime("%B %d, %Y")
    else:
        date_str = datetime.now().strftime("%B %d, %Y")

    suffix = " Close" if price_mode.lower() == "last close" else ""
    p.text = f"Source: Bloomberg, Herculis Group | Data as of {date_str}{suffix}"

    if p.runs:
        run = p.runs[0]
        run.font.size = Pt(8)
        run.font.name = "Calibri"
        run.font.color.rgb = COLORS["light_gray"]


def generate_technical_analysis_slide(
    prs: Presentation,
    rows: List[AssetRow],
    used_date: Optional[datetime] = None,
    price_mode: str = "Last Price"
):
    """
    Generate the Technical Analysis In A Nutshell slide (creates new slide).
    Uses TWO-COLUMN layout with exact cm dimensions.
    """
    # Use blank layout
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)

    layout = SLIDE_LAYOUT

    # Gold accent bar
    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Cm(1.0), Cm(layout["title_top"]),
        Cm(0.15), Cm(1.8)
    )
    accent.fill.solid()
    accent.fill.fore_color.rgb = COLORS["gold_accent"]
    accent.line.fill.background()

    # Title
    title_box = slide.shapes.add_textbox(
        Cm(layout["title_left"]), Cm(layout["title_top"]),
        Cm(15), Cm(1.2)
    )
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "Technical Analysis In A Nutshell"
    if p.runs:
        run = p.runs[0]
        run.font.size = Pt(28)
        run.font.italic = True
        run.font.name = "Calibri"
        run.font.color.rgb = COLORS["neutral_text"]

    # Subtitle
    subtitle_top = layout.get("subtitle_top", layout["title_top"] + 1.5)
    subtitle_box = slide.shapes.add_textbox(
        Cm(layout["title_left"]), Cm(subtitle_top),
        Cm(15), Cm(0.8)
    )
    tf = subtitle_box.text_frame
    p = tf.paragraphs[0]
    p.text = "HERA Score and Herculis' view for the main market indexes"
    if p.runs:
        run = p.runs[0]
        run.font.size = Pt(14)
        run.font.name = "Calibri"
        run.font.color.rgb = COLORS["gray_text"]

    # Add content (tables)
    _add_content_to_slide(slide, rows, used_date, price_mode)

    return slide


def insert_technical_analysis_slide(
    prs: Presentation,
    rows: List[AssetRow],
    placeholder_name: str = "technical_nutshell",
    used_date: Optional[datetime] = None,
    price_mode: str = "Last Price"
) -> Presentation:
    """
    Insert the Technical Analysis content into an existing presentation.

    Finds the slide with the matching placeholder (by shape name) and
    adds the tables to that slide. If no placeholder is found, creates
    a new slide at the end.
    """
    # Try to find existing placeholder slide by shape name
    target_slide = None
    found_shape_name = None

    print(f"[Technical Nutshell] Searching for placeholder '{placeholder_name}' in {len(prs.slides)} slides...")

    for slide_idx, slide in enumerate(prs.slides):
        for shape in slide.shapes:
            name_attr = getattr(shape, "name", "")
            if placeholder_name.lower() in name_attr.lower():
                target_slide = slide
                found_shape_name = name_attr
                print(f"[Technical Nutshell] Found placeholder shape '{name_attr}' on slide {slide_idx + 1}")
                # Remove the placeholder shape
                sp = shape._element
                sp.getparent().remove(sp)
                break
        if target_slide:
            break

    if target_slide:
        # Add content to the existing slide
        print(f"[Technical Nutshell] Adding two-column tables to slide with placeholder '{found_shape_name}'")
        _add_content_to_slide(target_slide, rows, used_date, price_mode)
    else:
        # No placeholder found - create new slide at the end
        print(f"[Technical Nutshell] No placeholder '{placeholder_name}' found - creating new slide at end")
        generate_technical_analysis_slide(prs, rows, used_date, price_mode)

    return prs
