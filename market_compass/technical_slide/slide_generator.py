"""PowerPoint slide generation for Technical Analysis - AGGRESSIVE FONT FIX VERSION.

This version uses aggressive XML manipulation to GUARANTEE:
- 9pt font size on all cells
- White text on headers
- Proper colors on all cells
- Clean borders
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
# XML HELPER FUNCTIONS
# ============================================================

def _set_font_element(parent, font_name: str):
    """Add font element to parent XML node."""
    # Remove existing font elements
    for child in list(parent):
        if 'latin' in child.tag or 'cs' in child.tag or 'ea' in child.tag:
            parent.remove(child)

    # Add new font elements
    latin = etree.SubElement(parent, qn('a:latin'))
    latin.set('typeface', font_name)

    cs = etree.SubElement(parent, qn('a:cs'))
    cs.set('typeface', font_name)


def _set_color_element(parent, color_hex: str):
    """Add solid fill color to parent XML node."""
    # Remove existing solidFill
    for child in list(parent):
        if 'solidFill' in child.tag:
            parent.remove(child)

    # Add new solidFill with srgbClr
    solidFill = etree.SubElement(parent, qn('a:solidFill'))
    srgbClr = etree.SubElement(solidFill, qn('a:srgbClr'))
    srgbClr.set('val', color_hex)


def _remove_table_style(table):
    """
    AGGRESSIVELY remove table style to prevent font override.
    Removes all style children and disables banding/special formatting.
    """
    tbl = table._tbl
    tblPr = tbl.tblPr

    if tblPr is not None:
        # Remove all children that could affect styling
        for child in list(tblPr):
            tblPr.remove(child)

        # Set first row and banding off
        tblPr.set('firstRow', '0')
        tblPr.set('bandRow', '0')
        tblPr.set('firstCol', '0')
        tblPr.set('lastRow', '0')
        tblPr.set('lastCol', '0')
        tblPr.set('bandCol', '0')


def _rgb_to_hex(rgb_color: RGBColor) -> str:
    """Convert RGBColor to hex string for XML."""
    r = rgb_color[0]
    g = rgb_color[1]
    b = rgb_color[2]
    return f'{r:02X}{g:02X}{b:02X}'


def _set_cell_font_xml(cell, font_size_pt: int, bold: bool = False,
                       color_hex: str = None, font_name: str = "Calibri"):
    """
    Set font using AGGRESSIVE XML manipulation.
    Sets font on: defRPr, endParaRPr, AND each rPr.

    Parameters
    ----------
    cell : pptx table cell
    font_size_pt : int - Font size in points (e.g., 9)
    bold : bool
    color_hex : str - Color as hex string (e.g., "FFFFFF")
    font_name : str - Font family name
    """
    # Size in 100ths of a point for XML 'sz' attribute
    sz_val = str(font_size_pt * 100)  # 9pt -> '900'

    txBody = cell._tc.txBody

    for p in txBody.iterchildren(qn('a:p')):
        # === 1. Paragraph properties (pPr) ===
        pPr = p.find(qn('a:pPr'))
        if pPr is None:
            pPr = etree.SubElement(p, qn('a:pPr'))

        # === 2. Default run properties (defRPr) ===
        defRPr = pPr.find(qn('a:defRPr'))
        if defRPr is None:
            defRPr = etree.SubElement(pPr, qn('a:defRPr'))

        # Set size and bold
        defRPr.set('sz', sz_val)
        defRPr.set('b', '1' if bold else '0')

        # Set font
        _set_font_element(defRPr, font_name)

        # Set color
        if color_hex:
            _set_color_element(defRPr, color_hex)

        # === 3. End paragraph run properties (endParaRPr) ===
        endParaRPr = p.find(qn('a:endParaRPr'))
        if endParaRPr is None:
            endParaRPr = etree.SubElement(p, qn('a:endParaRPr'))

        endParaRPr.set('lang', 'en-US')
        endParaRPr.set('sz', sz_val)
        endParaRPr.set('b', '1' if bold else '0')
        _set_font_element(endParaRPr, font_name)
        if color_hex:
            _set_color_element(endParaRPr, color_hex)

        # === 4. Each run (r) ===
        for r in p.iterchildren(qn('a:r')):
            rPr = r.find(qn('a:rPr'))
            if rPr is None:
                rPr = etree.Element(qn('a:rPr'))
                r.insert(0, rPr)

            rPr.set('lang', 'en-US')
            rPr.set('sz', sz_val)
            rPr.set('b', '1' if bold else '0')
            rPr.set('dirty', '0')

            _set_font_element(rPr, font_name)
            if color_hex:
                _set_color_element(rPr, color_hex)


def _set_thin_borders(cell, color_hex: str = "E5E5E5"):
    """Set thin gray borders on cell."""
    tc = cell._tc
    tcPr = tc.tcPr
    if tcPr is None:
        tcPr = etree.SubElement(tc, qn('a:tcPr'))

    for border in ['lnL', 'lnR', 'lnT', 'lnB']:
        ln = tcPr.find(qn(f'a:{border}'))
        if ln is not None:
            tcPr.remove(ln)

        ln = etree.SubElement(tcPr, qn(f'a:{border}'))
        ln.set('w', '6350')  # 0.5pt in EMUs

        solidFill = etree.SubElement(ln, qn('a:solidFill'))
        srgbClr = etree.SubElement(solidFill, qn('a:srgbClr'))
        srgbClr.set('val', color_hex)


def _normalize_widths(widths: list, total_width: float) -> list:
    """Normalize column widths to match total table width exactly."""
    current_sum = sum(widths)
    factor = total_width / current_sum
    return [w * factor for w in widths]


def _create_and_format_cell(table, row_idx: int, col_idx: int, text: str,
                            bg_color: RGBColor, text_color: RGBColor,
                            font_size_pt: int = 9, bold: bool = False,
                            align: str = "center"):
    """
    Create and format a cell with GUARANTEED font settings.

    Uses aggressive XML manipulation to ensure font size and color
    are applied correctly, overriding any table style defaults.
    """
    cell = table.cell(row_idx, col_idx)

    # Background color
    cell.fill.solid()
    cell.fill.fore_color.rgb = bg_color

    # Set text
    cell.text = str(text)

    # Alignment
    para = cell.text_frame.paragraphs[0]
    para.alignment = PP_ALIGN.CENTER if align == "center" else PP_ALIGN.LEFT

    # Vertical centering
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE

    # Remove margins for tighter layout
    cell.text_frame.margin_left = Pt(2)
    cell.text_frame.margin_right = Pt(2)
    cell.text_frame.margin_top = Pt(0)
    cell.text_frame.margin_bottom = Pt(0)

    # Remove paragraph spacing
    para.space_before = Pt(0)
    para.space_after = Pt(0)

    # FORCE FONT VIA XML - This is the critical fix
    color_hex = _rgb_to_hex(text_color)
    _set_cell_font_xml(cell, font_size_pt, bold, color_hex)

    # Set thin borders for clean look
    _set_thin_borders(cell, "E5E5E5")

    return cell


def _create_table(slide, rows: List[AssetRow], asset_class: str):
    """
    Create a single table for an asset class with aggressive XML font fix.
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

    # *** CRITICAL: REMOVE TABLE STYLE FIRST ***
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
            text_color=COLORS["header_text"],  # WHITE
            font_size_pt=9,
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
            font_size_pt=9,
            align="left"
        )

        # Column 1: Market Cap
        _create_and_format_cell(
            table, row_idx, 1,
            text=asset_row.market_cap,
            bg_color=bg_color,
            text_color=COLORS["neutral_text"],
            font_size_pt=9
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
            font_size_pt=9
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
            font_size_pt=9
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
            font_size_pt=9,
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
            font_size_pt=8,
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
    Uses TWO-COLUMN layout with aggressive XML font fix.
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
