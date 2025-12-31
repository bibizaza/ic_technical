"""PowerPoint slide generation for Technical Analysis - Two Column Layout."""

from datetime import datetime
from typing import List, Optional

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

from .config import (
    COLORS, COL_WIDTHS_LEFT, COL_WIDTHS_RIGHT, HEADERS,
    SLIDE_LAYOUT, ROW_HEIGHT, HEADER_FONT_SIZE, DATA_FONT_SIZE, OUTLOOK_FONT_SIZE
)
from .data_prep import AssetRow


def _format_cell_text(cell, text: str, align: str = "center", size: int = 8,
                      color: RGBColor = None, bold: bool = False):
    """Helper to format cell text."""
    cell.text = str(text)
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


def _create_table(slide, rows: List[AssetRow], left: float, top: float,
                  col_widths: List[float], asset_class: str):
    """
    Create a single table for an asset class.

    Parameters
    ----------
    slide : pptx.slide.Slide
        The slide to add the table to
    rows : List[AssetRow]
        Asset rows (already filtered by asset class)
    left, top : float
        Position in inches
    col_widths : List[float]
        Column widths in inches
    asset_class : str
        "equity", "commodities", or "crypto"

    Returns
    -------
    float
        The Y position of the bottom of this table (for stacking)
    """
    if not rows:
        print(f"[Technical Nutshell] No rows for asset class '{asset_class}'")
        return top

    n_rows = len(rows) + 1  # +1 for header
    n_cols = 6
    table_width = sum(col_widths)

    # Create table
    table_shape = slide.shapes.add_table(
        n_rows, n_cols,
        Inches(left), Inches(top),
        Inches(table_width), Inches(ROW_HEIGHT * n_rows)
    )
    table = table_shape.table

    # Set column widths
    for i, w in enumerate(col_widths):
        table.columns[i].width = Inches(w)

    # Set all row heights
    for row in table.rows:
        row.height = Inches(ROW_HEIGHT)

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

    # Return the bottom Y position of this table
    table_bottom = top + (ROW_HEIGHT * n_rows)
    return table_bottom


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
    _create_table(
        slide, equity_rows,
        left=layout["left_table_x"],
        top=layout["left_table_y"],
        col_widths=COL_WIDTHS_LEFT,
        asset_class="equity"
    )

    # ----- RIGHT COLUMN: COMMODITIES -----
    commo_bottom = _create_table(
        slide, commo_rows,
        left=layout["right_table_x"],
        top=layout["right_commo_y"],
        col_widths=COL_WIDTHS_RIGHT,
        asset_class="commodities"
    )

    # ----- RIGHT COLUMN: CRYPTO (below commodities) -----
    crypto_top = commo_bottom + layout["right_crypto_gap"]
    _create_table(
        slide, crypto_rows,
        left=layout["right_table_x"],
        top=crypto_top,
        col_widths=COL_WIDTHS_RIGHT,
        asset_class="crypto"
    )

    # ----- FOOTER -----
    footer = slide.shapes.add_textbox(
        Inches(0.3), Inches(layout["footer_y"]),
        Inches(8), Inches(0.2)
    )
    tf = footer.text_frame
    p = tf.paragraphs[0]

    if used_date:
        date_str = used_date.strftime("%d/%m/%Y")
    else:
        date_str = datetime.now().strftime("%d/%m/%Y")

    suffix = " Close" if price_mode.lower() == "last close" else ""
    p.text = f"Source: Bloomberg, Herculis Group | Data as of {date_str}{suffix}"

    if p.runs:
        run = p.runs[0]
        run.font.size = Pt(7)
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
    Uses TWO-COLUMN layout.
    """
    # Use blank layout
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)

    layout = SLIDE_LAYOUT

    # Gold accent bar
    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.3), Inches(layout["title_top"]),
        Inches(0.05), Inches(0.55)
    )
    accent.fill.solid()
    accent.fill.fore_color.rgb = COLORS["gold_accent"]
    accent.line.fill.background()

    # Title
    title_box = slide.shapes.add_textbox(
        Inches(layout["title_left"]), Inches(layout["title_top"]),
        Inches(6), Inches(0.35)
    )
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "Technical Analysis In A Nutshell"
    if p.runs:
        run = p.runs[0]
        run.font.size = Pt(20)
        run.font.italic = True
        run.font.name = "Calibri"
        run.font.color.rgb = COLORS["neutral_text"]

    # Subtitle
    subtitle_box = slide.shapes.add_textbox(
        Inches(layout["title_left"]), Inches(layout["title_top"] + 0.35),
        Inches(6), Inches(0.25)
    )
    tf = subtitle_box.text_frame
    p = tf.paragraphs[0]
    p.text = "HERA Score and Herculis' view for the main market indexes"
    if p.runs:
        run = p.runs[0]
        run.font.size = Pt(10)
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
