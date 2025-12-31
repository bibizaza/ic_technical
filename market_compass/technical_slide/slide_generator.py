"""PowerPoint slide generation for Technical Analysis."""

from datetime import datetime
from typing import List, Optional

from pptx import Presentation
from pptx.util import Inches, Pt, Cm
from pptx.dml.color import RgbColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

from .config import COLORS, COLUMN_WIDTHS, HEADERS, SLIDE_LAYOUT
from .data_prep import AssetRow


def _set_cell_fill(cell, color: RgbColor):
    """Set cell background color."""
    cell.fill.solid()
    cell.fill.fore_color.rgb = color


def _set_cell_text(
    cell,
    text: str,
    font_size: int = 10,
    bold: bool = False,
    color: RgbColor = None,
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
    Create a table section for an asset class.

    Parameters
    ----------
    slide : pptx.slide.Slide
        The slide to add the table to
    rows : List[AssetRow]
        All asset rows (will be filtered by asset_class)
    asset_class : str
        "equity", "commodities", or "crypto"
    left : float
        Left position in inches
    top : float
        Top position in inches
    width : float
        Table width in inches

    Returns
    -------
    pptx.table.Table
        The created table
    """
    # Filter rows for this asset class
    class_rows = [r for r in rows if r.asset_class == asset_class]

    if not class_rows:
        return None

    # Table dimensions
    n_rows = len(class_rows) + 1  # +1 for header
    n_cols = 6

    # Row height
    row_height = 0.28

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

    # Header row
    headers = HEADERS.get(asset_class, HEADERS["equity"])

    for i, header in enumerate(headers):
        cell = table.cell(0, i)
        _set_cell_fill(cell, COLORS["header_bg"])
        _set_cell_text(
            cell, header,
            font_size=9,
            bold=True,
            color=COLORS["header_text"],
            align="left" if i == 0 else "center"
        )

    # Data rows
    for row_idx, asset_row in enumerate(class_rows, start=1):
        # Column 0: Asset Name
        cell = table.cell(row_idx, 0)
        _set_cell_text(cell, asset_row.name, font_size=10, align="left")

        # Column 1: Market Cap
        cell = table.cell(row_idx, 1)
        _set_cell_text(cell, asset_row.market_cap, font_size=10)

        # Column 2: RSI
        cell = table.cell(row_idx, 2)
        rsi_color = COLORS["neutral_text"]
        if asset_row.rsi > 70:
            rsi_color = COLORS["negative"]  # Overbought
        elif asset_row.rsi < 30:
            rsi_color = COLORS["positive"]  # Oversold
        _set_cell_text(cell, str(int(asset_row.rsi)), font_size=10, color=rsi_color)

        # Column 3: vs 50d MA
        cell = table.cell(row_idx, 3)
        ma_text = f"{asset_row.vs_50d_ma:+.2f}%"
        ma_color = COLORS["positive"] if asset_row.vs_50d_ma >= 0 else COLORS["negative"]
        _set_cell_text(cell, ma_text, font_size=10, color=ma_color)

        # Column 4: DMAS
        cell = table.cell(row_idx, 4)
        if asset_row.dmas >= 55:
            dmas_color = COLORS["positive"]
        elif asset_row.dmas < 45:
            dmas_color = COLORS["negative"]
        else:
            dmas_color = COLORS["neutral_text"]
        _set_cell_text(cell, str(asset_row.dmas), font_size=10, bold=True, color=dmas_color)

        # Column 5: Outlook
        cell = table.cell(row_idx, 5)
        outlook_lower = asset_row.outlook.lower()
        bg_key = f"outlook_{outlook_lower}_bg"
        text_key = f"outlook_{outlook_lower}_text"

        if bg_key in COLORS:
            _set_cell_fill(cell, COLORS[bg_key])
        if text_key in COLORS:
            _set_cell_text(
                cell, asset_row.outlook,
                font_size=9, bold=True,
                color=COLORS[text_key]
            )
        else:
            _set_cell_text(cell, asset_row.outlook, font_size=9, bold=True)

    return table


def _add_section_label(slide, text: str, left: float, top: float):
    """Add an underlined section label."""
    label = slide.shapes.add_textbox(
        Inches(left), Inches(top - 0.28),
        Inches(2), Inches(0.25)
    )
    tf = label.text_frame
    p = tf.paragraphs[0]
    p.text = text
    if p.runs:
        run = p.runs[0]
        run.font.size = Pt(11)
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

    Parameters
    ----------
    slide : pptx.slide.Slide
        The slide to add content to
    rows : List[AssetRow]
        Prepared asset data
    used_date : datetime, optional
        Date for the data source footnote
    price_mode : str
        "Last Price" or "Last Close"
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

    Parameters
    ----------
    prs : Presentation
        PowerPoint presentation object
    rows : List[AssetRow]
        Prepared asset data
    used_date : datetime, optional
        Date for the data source footnote
    price_mode : str
        "Last Price" or "Last Close"

    Returns
    -------
    pptx.slide.Slide
        The generated slide
    """
    # Use blank layout
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)

    layout = SLIDE_LAYOUT

    # Gold accent bar
    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(layout["title_left"] - 0.1), Inches(layout["title_top"]),
        Inches(0.05), Inches(0.6)
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
        run.font.size = Pt(22)
        run.font.italic = True
        run.font.name = "Calibri"
        run.font.color.rgb = COLORS["neutral_text"]

    # Subtitle
    subtitle_box = slide.shapes.add_textbox(
        Inches(layout["title_left"]), Inches(layout["title_top"] + 0.38),
        Inches(6), Inches(0.25)
    )
    tf = subtitle_box.text_frame
    p = tf.paragraphs[0]
    p.text = "HERA Score and Herculis' view for the main market indexes"
    if p.runs:
        run = p.runs[0]
        run.font.size = Pt(11)
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

    Parameters
    ----------
    prs : Presentation
        PowerPoint presentation object
    rows : List[AssetRow]
        Prepared asset data
    placeholder_name : str
        Name of placeholder shape to find in template (default: "technical_nutshell")
    used_date : datetime, optional
        Date for data source footnote
    price_mode : str
        "Last Price" or "Last Close"

    Returns
    -------
    Presentation
        Modified presentation
    """
    # Try to find existing placeholder slide by shape name
    target_slide = None

    for slide in prs.slides:
        for shape in slide.shapes:
            name_attr = getattr(shape, "name", "").lower()
            if placeholder_name.lower() in name_attr:
                target_slide = slide
                # Remove the placeholder shape
                sp = shape._element
                sp.getparent().remove(sp)
                break
        if target_slide:
            break

    if target_slide:
        # Add content to the existing slide
        print(f"Found slide with placeholder '{placeholder_name}' - adding tables")
        _add_content_to_slide(target_slide, rows, used_date, price_mode)
    else:
        # No placeholder found - create new slide at the end
        print(f"No placeholder '{placeholder_name}' found - creating new slide")
        generate_technical_analysis_slide(prs, rows, used_date, price_mode)

    return prs
