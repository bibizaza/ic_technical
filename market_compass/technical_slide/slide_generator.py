"""PowerPoint slide generation for Technical Analysis."""

from datetime import datetime
from typing import List, Optional

from pptx import Presentation
from pptx.util import Inches, Pt, Cm
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

from .config import COLORS, COLUMN_WIDTHS, HEADERS, SLIDE_LAYOUT
from .data_prep import AssetRow


def _set_cell_fill(cell, color: RGBColor):
    """Set cell background color."""
    cell.fill.solid()
    cell.fill.fore_color.rgb = color


def _set_cell_text(
    cell,
    text: str,
    font_size: int = 9,
    bold: bool = False,
    color: RGBColor = None,
    align: str = "center"
):
    """Set cell text with formatting."""
    cell.text = str(text)
    paragraph = cell.text_frame.paragraphs[0]

    if align == "center":
        paragraph.alignment = PP_ALIGN.CENTER
    elif align == "left":
        paragraph.alignment = PP_ALIGN.LEFT
    elif align == "right":
        paragraph.alignment = PP_ALIGN.RIGHT

    if paragraph.runs:
        run = paragraph.runs[0]
        run.font.size = Pt(font_size)
        run.font.bold = bold
        run.font.name = "Calibri"
        if color:
            run.font.color.rgb = color


def _create_table_section(
    slide,
    rows: List[AssetRow],
    asset_class: str,
    left: float,
    top: float,
    width: float
):
    """
    Create a compact table section for an asset class with alternating row colors.
    """
    # Filter rows for this asset class
    class_rows = [r for r in rows if r.asset_class == asset_class]

    if not class_rows:
        print(f"[Technical Nutshell] No rows for asset class '{asset_class}'")
        return None

    # Table dimensions
    n_rows = len(class_rows) + 1  # +1 for header
    n_cols = 6

    # Row height from config
    row_height = SLIDE_LAYOUT.get("row_height", 0.22)

    # Add table
    table_shape = slide.shapes.add_table(
        n_rows, n_cols,
        Inches(left), Inches(top),
        Inches(width), Inches(row_height * n_rows)
    )
    table = table_shape.table

    # Set column widths
    for i, w in enumerate(COLUMN_WIDTHS):
        table.columns[i].width = Inches(w)

    # Set row heights
    for row in table.rows:
        row.height = Inches(row_height)

    # Header row
    headers = HEADERS.get(asset_class, HEADERS["equity"])
    header_font_size = SLIDE_LAYOUT.get("header_font_size", 8)

    for i, header in enumerate(headers):
        cell = table.cell(0, i)
        _set_cell_fill(cell, COLORS["header_bg"])
        _set_cell_text(
            cell, header,
            font_size=header_font_size,
            bold=True,
            color=COLORS["header_text"],
            align="left" if i == 0 else "center"
        )

    # Data rows with alternating white/grey background
    data_font_size = SLIDE_LAYOUT.get("data_font_size", 9)

    for row_idx, asset_row in enumerate(class_rows, start=1):
        # Alternating background color
        bg_color = COLORS["row_white"] if row_idx % 2 == 1 else COLORS["row_grey"]

        # Column 0: Asset Name
        cell = table.cell(row_idx, 0)
        _set_cell_fill(cell, bg_color)
        _set_cell_text(cell, asset_row.name, font_size=data_font_size, align="left",
                       color=COLORS["neutral_text"])

        # Column 1: Market Cap
        cell = table.cell(row_idx, 1)
        _set_cell_fill(cell, bg_color)
        _set_cell_text(cell, asset_row.market_cap, font_size=data_font_size,
                       color=COLORS["neutral_text"])

        # Column 2: RSI (color coded)
        cell = table.cell(row_idx, 2)
        _set_cell_fill(cell, bg_color)
        rsi_val = int(asset_row.rsi)
        rsi_color = COLORS["neutral_text"]
        if rsi_val > 70:
            rsi_color = COLORS["negative"]  # Overbought
        elif rsi_val < 30:
            rsi_color = COLORS["positive"]  # Oversold
        _set_cell_text(cell, str(rsi_val), font_size=data_font_size, color=rsi_color)

        # Column 3: vs 50d MA (color coded)
        cell = table.cell(row_idx, 3)
        _set_cell_fill(cell, bg_color)
        ma_val = asset_row.vs_50d_ma
        ma_text = f"{ma_val:+.1f}%"
        ma_color = COLORS["positive"] if ma_val >= 0 else COLORS["negative"]
        _set_cell_text(cell, ma_text, font_size=data_font_size, color=ma_color)

        # Column 4: DMAS (INTEGER, color coded)
        cell = table.cell(row_idx, 4)
        _set_cell_fill(cell, bg_color)
        dmas_val = int(asset_row.dmas)  # Ensure integer
        if dmas_val >= 55:
            dmas_color = COLORS["positive"]
        elif dmas_val < 45:
            dmas_color = COLORS["negative"]
        else:
            dmas_color = COLORS["neutral_text"]
        _set_cell_text(cell, str(dmas_val), font_size=data_font_size,
                       bold=True, color=dmas_color)

        # Column 5: Outlook (colored background)
        cell = table.cell(row_idx, 5)
        outlook_lower = asset_row.outlook.lower()
        bg_key = f"outlook_{outlook_lower}_bg"
        text_key = f"outlook_{outlook_lower}_text"

        if bg_key in COLORS:
            _set_cell_fill(cell, COLORS[bg_key])
        else:
            _set_cell_fill(cell, bg_color)

        text_color = COLORS.get(text_key, COLORS["neutral_text"])
        _set_cell_text(
            cell, asset_row.outlook,
            font_size=8, bold=True,
            color=text_color
        )

    return table


def _add_section_label(slide, text: str, left: float, top: float):
    """Add an underlined section label."""
    label = slide.shapes.add_textbox(
        Inches(left), Inches(top - 0.25),
        Inches(2), Inches(0.22)
    )
    tf = label.text_frame
    p = tf.paragraphs[0]
    p.text = text
    if p.runs:
        run = p.runs[0]
        run.font.size = Pt(10)
        run.font.italic = True
        run.font.underline = True
        run.font.name = "Calibri"
        run.font.color.rgb = COLORS["neutral_text"]


def _add_content_to_slide(
    slide,
    rows: List[AssetRow],
    used_date: Optional[datetime] = None,
    price_mode: str = "Last Price"
):
    """
    Add Technical Analysis content to an existing slide.
    """
    layout = SLIDE_LAYOUT

    # Equity section
    _add_section_label(slide, "Equity", layout["title_left"], layout["equity_top"])
    _create_table_section(
        slide, rows, "equity",
        layout["title_left"], layout["equity_top"],
        layout["table_width"]
    )

    # Commodities section
    _add_section_label(slide, "Commodities", layout["title_left"], layout["commodities_top"])
    _create_table_section(
        slide, rows, "commodities",
        layout["title_left"], layout["commodities_top"],
        layout["table_width"]
    )

    # Crypto section
    _add_section_label(slide, "Crypto", layout["title_left"], layout["crypto_top"])
    _create_table_section(
        slide, rows, "crypto",
        layout["title_left"], layout["crypto_top"],
        layout["table_width"]
    )

    # Footer
    footer = slide.shapes.add_textbox(
        Inches(layout["title_left"]), Inches(layout["footer_top"]),
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
    """
    # Use blank layout
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)

    layout = SLIDE_LAYOUT

    # Gold accent bar
    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(layout["title_left"] - 0.08), Inches(layout["title_top"]),
        Inches(0.04), Inches(0.5)
    )
    accent.fill.solid()
    accent.fill.fore_color.rgb = COLORS["gold_accent"]
    accent.line.fill.background()

    # Title
    title_box = slide.shapes.add_textbox(
        Inches(layout["title_left"]), Inches(layout["title_top"]),
        Inches(5), Inches(0.3)
    )
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "Technical Analysis In A Nutshell"
    if p.runs:
        run = p.runs[0]
        run.font.size = Pt(18)
        run.font.italic = True
        run.font.name = "Calibri"
        run.font.color.rgb = COLORS["neutral_text"]

    # Subtitle
    subtitle_box = slide.shapes.add_textbox(
        Inches(layout["title_left"]), Inches(layout["title_top"] + 0.32),
        Inches(5), Inches(0.2)
    )
    tf = subtitle_box.text_frame
    p = tf.paragraphs[0]
    p.text = "HERA Score and Herculis' view for the main market indexes"
    if p.runs:
        run = p.runs[0]
        run.font.size = Pt(10)
        run.font.name = "Calibri"
        run.font.color.rgb = COLORS["gray_text"]

    # Add content (tables, etc.)
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
        print(f"[Technical Nutshell] Adding tables to slide with placeholder '{found_shape_name}'")
        _add_content_to_slide(target_slide, rows, used_date, price_mode)
    else:
        # No placeholder found - create new slide at the end
        print(f"[Technical Nutshell] No placeholder '{placeholder_name}' found - creating new slide at end")
        generate_technical_analysis_slide(prs, rows, used_date, price_mode)

    return prs
