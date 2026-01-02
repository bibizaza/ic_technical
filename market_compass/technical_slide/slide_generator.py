"""PowerPoint slide generation for Technical Analysis - XML FONT FIX VERSION.

This version removes the table style and sets fonts via direct XML manipulation
to guarantee 9pt font size (python-pptx table styles override cell-level settings).
"""

from datetime import datetime
from typing import List, Optional

from pptx import Presentation
from pptx.util import Cm, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from lxml import etree

from .config import (
    COLORS, HEADERS, SLIDE_LAYOUT, TABLE_DIMS,
    FONT_SIZE, FONT_SIZE_HEADER, FONT_SIZE_OUTLOOK
)
from .data_prep import AssetRow


# ============================================================
# FONT SIZE IN EMUs (English Metric Units)
# 1 point = 12700 EMUs
# In XML 'sz' attribute, we use 100ths of a point
# ============================================================
FONT_SIZE_9PT_EMU = 114300   # 9 * 12700
FONT_SIZE_8PT_EMU = 101600   # 8 * 12700


def _remove_table_style(table):
    """
    Remove the table style so it doesn't override our font settings.
    This is the KEY fix for the font size problem.
    """
    tbl = table._tbl
    tblPr = tbl.tblPr

    if tblPr is not None:
        # Remove tableStyleId if it exists
        for child in list(tblPr):
            if 'tableStyleId' in child.tag:
                tblPr.remove(child)
                break


def _rgb_to_hex(rgb_color: RGBColor) -> str:
    """Convert RGBColor to hex string for XML."""
    # RGBColor stores RGB as an integer
    val = int(rgb_color)
    r = (val >> 16) & 0xFF
    g = (val >> 8) & 0xFF
    b = val & 0xFF
    return f'{r:02X}{g:02X}{b:02X}'


def _set_cell_font_xml(cell, font_size_emu: int, bold: bool = False,
                       color_hex: str = None, font_name: str = "Calibri"):
    """
    Set font size using direct XML manipulation.
    This GUARANTEES the font size is set correctly.

    Parameters
    ----------
    cell : pptx table cell
    font_size_emu : int - Font size in EMUs (9pt = 114300)
    bold : bool
    color_hex : str - Color as hex string (e.g., "1A1A2E")
    font_name : str - Font family name
    """
    # Size in 100ths of a point for XML 'sz' attribute
    sz_val = str(font_size_emu // 100)

    # Get the text body XML
    txBody = cell._tc.txBody

    for p in txBody.iterchildren(qn('a:p')):
        # Set paragraph-level defaults
        pPr = p.find(qn('a:pPr'))
        if pPr is None:
            pPr = etree.SubElement(p, qn('a:pPr'))

        # Add default run properties to paragraph
        defRPr = pPr.find(qn('a:defRPr'))
        if defRPr is None:
            defRPr = etree.SubElement(pPr, qn('a:defRPr'))

        defRPr.set('sz', sz_val)
        if bold:
            defRPr.set('b', '1')

        # Set font name
        latin = defRPr.find(qn('a:latin'))
        if latin is None:
            latin = etree.SubElement(defRPr, qn('a:latin'))
        latin.set('typeface', font_name)

        # Set color on default run properties
        if color_hex:
            solidFill = defRPr.find(qn('a:solidFill'))
            if solidFill is not None:
                defRPr.remove(solidFill)
            solidFill = etree.SubElement(defRPr, qn('a:solidFill'))
            srgbClr = etree.SubElement(solidFill, qn('a:srgbClr'))
            srgbClr.set('val', color_hex)

        # Also set on each run (r element)
        for r in p.iterchildren(qn('a:r')):
            rPr = r.find(qn('a:rPr'))
            if rPr is None:
                rPr = etree.Element(qn('a:rPr'))
                rPr.set('lang', 'en-US')
                r.insert(0, rPr)

            rPr.set('sz', sz_val)
            if bold:
                rPr.set('b', '1')

            # Set font name on run
            latin = rPr.find(qn('a:latin'))
            if latin is None:
                latin = etree.SubElement(rPr, qn('a:latin'))
            latin.set('typeface', font_name)

            if color_hex:
                solidFill = rPr.find(qn('a:solidFill'))
                if solidFill is not None:
                    rPr.remove(solidFill)
                solidFill = etree.SubElement(rPr, qn('a:solidFill'))
                srgbClr = etree.SubElement(solidFill, qn('a:srgbClr'))
                srgbClr.set('val', color_hex)


def _normalize_widths(widths: list, total_width: float) -> list:
    """Normalize column widths to match total table width exactly."""
    current_sum = sum(widths)
    factor = total_width / current_sum
    return [w * factor for w in widths]


def _create_and_format_cell(table, row_idx: int, col_idx: int, text: str,
                            bg_color: RGBColor, text_color: RGBColor,
                            font_size_emu: int, bold: bool = False,
                            align: str = "center"):
    """
    Create and format a cell with XML-level font control.

    This function:
    1. Sets background color
    2. Sets text content
    3. Sets alignment and vertical anchor
    4. Forces font size via XML manipulation
    """
    cell = table.cell(row_idx, col_idx)

    # Background color
    cell.fill.solid()
    cell.fill.fore_color.rgb = bg_color

    # Set text
    cell.text = str(text)

    # Alignment
    para = cell.text_frame.paragraphs[0]
    if align == "center":
        para.alignment = PP_ALIGN.CENTER
    else:
        para.alignment = PP_ALIGN.LEFT

    # Vertical centering
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE

    # Remove paragraph spacing
    para.space_before = Pt(0)
    para.space_after = Pt(0)

    # FORCE FONT SIZE VIA XML - This is the critical fix
    color_hex = _rgb_to_hex(text_color)
    _set_cell_font_xml(cell, font_size_emu, bold, color_hex)

    return cell


def _create_table(slide, rows: List[AssetRow], asset_class: str):
    """
    Create a single table for an asset class with XML font fix.

    Parameters
    ----------
    slide : pptx.slide.Slide
        The slide to add the table to
    rows : List[AssetRow]
        Asset rows (already filtered by asset class)
    asset_class : str
        "equity", "commodities", or "crypto"
    """
    print(f"[DEBUG _create_table] Called with asset_class='{asset_class}', rows count={len(rows)}")

    if not rows:
        print(f"[Technical Nutshell] No rows for asset class '{asset_class}'")
        return

    # Get dimensions from config
    dims = TABLE_DIMS.get(asset_class)
    if not dims:
        print(f"[Technical Nutshell] No dimensions for asset class '{asset_class}'")
        return

    print(f"[DEBUG _create_table] dims={dims}")

    n_rows = len(rows) + 1  # +1 for header
    n_cols = 6

    print(f"[DEBUG _create_table] Creating table: {n_rows} rows x {n_cols} cols")

    # Create table with exact cm dimensions
    table_shape = slide.shapes.add_table(
        n_rows, n_cols,
        Cm(dims["left"]), Cm(dims["top"]),
        Cm(dims["width"]), Cm(dims["height"])
    )
    table = table_shape.table

    # *** CRITICAL: REMOVE TABLE STYLE ***
    _remove_table_style(table)

    # Set column widths - normalize to fit exact table width
    col_widths = _normalize_widths(dims["col_widths"], dims["width"])
    print(f"[DEBUG _create_table] col_widths={col_widths}")
    for i, w in enumerate(col_widths):
        table.columns[i].width = Cm(w)

    # Set row heights
    header_height = dims.get("header_height", 0.68)
    data_row_height = (dims["height"] - header_height) / (n_rows - 1) if n_rows > 1 else header_height

    for i, row in enumerate(table.rows):
        if i == 0:
            row.height = Cm(header_height)
        else:
            row.height = Cm(data_row_height)

    # ----- HEADER ROW -----
    headers = HEADERS.get(asset_class, HEADERS["equity"])
    print(f"[DEBUG _create_table] HEADERS for '{asset_class}': {headers} (len={len(headers)})")

    for col_idx, header in enumerate(headers):
        print(f"[DEBUG _create_table] Creating header cell [{col_idx}]: '{header}'")
        _create_and_format_cell(
            table, 0, col_idx,
            text=header,
            bg_color=COLORS["header_bg"],
            text_color=COLORS["header_text"],
            font_size_emu=FONT_SIZE_9PT_EMU,
            bold=True,
            align="left" if col_idx == 0 else "center"
        )

    # ----- DATA ROWS -----
    for row_idx, asset_row in enumerate(rows, start=1):
        # Alternating background
        bg_color = COLORS["row_white"] if row_idx % 2 == 1 else COLORS["row_grey"]

        # Column 0: Asset Name
        _create_and_format_cell(
            table, row_idx, 0,
            text=asset_row.name,
            bg_color=bg_color,
            text_color=COLORS["neutral_text"],
            font_size_emu=FONT_SIZE_9PT_EMU,
            align="left"
        )

        # Column 1: Market Cap
        _create_and_format_cell(
            table, row_idx, 1,
            text=asset_row.market_cap,
            bg_color=bg_color,
            text_color=COLORS["neutral_text"],
            font_size_emu=FONT_SIZE_9PT_EMU
        )

        # Column 2: RSI (color coded)
        rsi_val = int(asset_row.rsi)
        if rsi_val > 70:
            rsi_color = COLORS["negative"]  # Overbought
        elif rsi_val < 30:
            rsi_color = COLORS["positive"]  # Oversold
        else:
            rsi_color = COLORS["neutral_text"]

        _create_and_format_cell(
            table, row_idx, 2,
            text=str(rsi_val),
            bg_color=bg_color,
            text_color=rsi_color,
            font_size_emu=FONT_SIZE_9PT_EMU
        )

        # Column 3: vs 50d MA (color coded)
        ma_val = asset_row.vs_50d_ma
        ma_text = f"{ma_val:+.1f}%"
        ma_color = COLORS["positive"] if ma_val >= 0 else COLORS["negative"]

        _create_and_format_cell(
            table, row_idx, 3,
            text=ma_text,
            bg_color=bg_color,
            text_color=ma_color,
            font_size_emu=FONT_SIZE_9PT_EMU
        )

        # Column 4: DMAS (INTEGER, color coded)
        dmas_val = int(asset_row.dmas)
        if dmas_val >= 55:
            dmas_color = COLORS["positive"]
        elif dmas_val < 45:
            dmas_color = COLORS["negative"]
        else:
            dmas_color = COLORS["neutral_text"]

        _create_and_format_cell(
            table, row_idx, 4,
            text=str(dmas_val),
            bg_color=bg_color,
            text_color=dmas_color,
            font_size_emu=FONT_SIZE_9PT_EMU,
            bold=True
        )

        # Column 5: Outlook (colored background)
        outlook_lower = asset_row.outlook.lower()
        bg_key = f"outlook_{outlook_lower}_bg"
        text_key = f"outlook_{outlook_lower}_text"

        outlook_bg = COLORS.get(bg_key, bg_color)
        outlook_text = COLORS.get(text_key, COLORS["neutral_text"])

        _create_and_format_cell(
            table, row_idx, 5,
            text=asset_row.outlook,
            bg_color=outlook_bg,
            text_color=outlook_text,
            font_size_emu=FONT_SIZE_8PT_EMU,
            bold=True
        )


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

    print(f"[DEBUG _add_content_to_slide] Total rows received: {len(rows)}")
    for i, r in enumerate(rows):
        print(f"[DEBUG _add_content_to_slide] Row {i}: name='{r.name}', asset_class='{r.asset_class}'")

    # Filter rows by asset class
    equity_rows = [r for r in rows if r.asset_class == "equity"]
    commo_rows = [r for r in rows if r.asset_class == "commodities"]
    crypto_rows = [r for r in rows if r.asset_class == "crypto"]

    print(f"[Technical Nutshell] Equity: {len(equity_rows)}, Commo: {len(commo_rows)}, Crypto: {len(crypto_rows)}")
    print(f"[DEBUG] HEADERS keys available: {list(HEADERS.keys())}")
    print(f"[DEBUG] TABLE_DIMS keys available: {list(TABLE_DIMS.keys())}")

    # ----- CREATE TABLES -----
    _create_table(slide, equity_rows, asset_class="equity")
    _create_table(slide, commo_rows, asset_class="commodities")
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

    # Set footer font
    p.font.size = Pt(8)
    p.font.name = "Calibri"
    p.font.color.rgb = COLORS["light_gray"]

    for run in p.runs:
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
    Uses TWO-COLUMN layout with XML font fix.
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
    p.font.size = Pt(28)
    p.font.italic = True
    p.font.name = "Calibri"
    p.font.color.rgb = COLORS["neutral_text"]

    # Subtitle
    subtitle_top = layout.get("subtitle_top", layout["title_top"] + 1.5)
    subtitle_box = slide.shapes.add_textbox(
        Cm(layout["title_left"]), Cm(subtitle_top),
        Cm(15), Cm(0.8)
    )
    tf = subtitle_box.text_frame
    p = tf.paragraphs[0]
    p.text = "DMAS Score and Technical Outlook for the main market indexes"
    p.font.size = Pt(14)
    p.font.name = "Calibri"
    p.font.color.rgb = COLORS["gray_text"]

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
