"""
Streamlit application for technical dashboard and presentation generation.

This application allows users to upload data, configure year‑to‑date (YTD)
charts for various asset classes, perform technical analysis on the S&P 500
index (including a selectable assessment title and a table of scores) and
generate a customised PowerPoint presentation.  The app persists
configuration selections in the session state and leverages helper functions
from the ``technical_analysis.equity.spx`` module for chart creation and
PowerPoint editing.

Key modifications relative to the original application:

* The SPX “view” title is no longer automatically derived from the average
  of technical and momentum scores.  Instead, users can select a view
  (e.g., “Strongly Bullish”) via a dropdown.  The chosen view is
  prepended with “S&P 500:” and inserted into the PowerPoint slide.
* The Streamlit interface no longer displays an average gauge for the SPX
  scores.  Instead, a simple table shows the technical score, momentum
  score and their average (DMAS), helping users judge the trend.
* The selected view is stored in ``st.session_state["spx_selected_view"]``
  and passed to ``insert_spx_technical_assessment`` when generating the
  presentation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from io import BytesIO
from pptx import Presentation
import tempfile
from pathlib import Path

# Import SPX functions from the dedicated module.  The SPX module
# resides in ``technical_analysis/equity/spx.py`` and provides all helper
# functions for building charts, inserting data into slides, and
# computing scores.  Note that ``insert_spx_technical_assessment``
# accepts a manual description and ``insert_spx_source`` inserts the
# source footnote based on the selected price mode.
from technical_analysis.equity.spx import (
    make_spx_figure,
    insert_spx_technical_chart_with_callout,
    insert_spx_technical_chart,
    insert_spx_technical_score_number,
    insert_spx_momentum_score_number,
    insert_spx_subtitle,
    insert_spx_average_gauge,
    insert_spx_technical_assessment,
    insert_spx_source,
    _get_spx_technical_score,
    _get_spx_momentum_score,
    generate_range_gauge_only_image,
    _compute_range_bounds as _compute_range_bounds_spx,
)

from importlib import reload
try:
    import funda_breadth.breadth_page as _breadth
except ModuleNotFoundError:
    import funda_breadth.breadth_page as _breadth
_breadth = reload(_breadth)

_load_breadth_page_data = _breadth._load_and_prepare
_style_breadth_page     = _breadth._apply_matrix_style
_debug_breadth_rows     = _breadth.debug_first_rows

# ─── Breadth‑picture helper (Excel → EMF → PPT) ───────────────────────
from pathlib import Path
import tempfile
from importlib import reload

try:
    import win32com.client           # pywin32, needs Excel on Windows
except ImportError:
    win32com = None                  # fallback handled later

# ─────────────────────────────────────────────────────────────────────
# Breadth helper  –  sort ▸ copy ▸ PDF ▸ Ghostscript ▸ crop ▸ PNG path
# ─────────────────────────────────────────────────────────────────────
from pathlib import Path

def _copy_breadth_test_range(xlsx_path: Path) -> Path:
    """
    1.  Sort helper_breadth!C5:J13 by column F ascending.
    2.  Copy helper_breadth!E4:J13 (now sorted) as
        VALUES + FORMATS + COLUMN WIDTHS into a one‑sheet temp workbook.
    3.  Export that sheet to PDF, convert to PNG with Ghostscript,
        cropping away transparent margins.  The bitmap width is capped
        at 5 000 px so it displays ≈ 550–600 dpi when inserted at
        19.55 cm on the slide.

    Returns the PNG file path.
    """
    import os, uuid, math, tempfile, shutil, subprocess, pythoncom, traceback
    from win32com.client import DispatchEx
    from PIL import Image

    # ---------------- user‑tuneables ----------------------------------
    SRC_SHEET      = "helper_breadth"
    SORT_RANGE     = "C5:J13"
    SORT_KEY       = "F5"           # first cell of the sort column
    COPY_RANGE     = "E4:J13"
    SLIDE_WIDTH_CM = 19.55
    MAX_PX         = 5000
    # ------------------------------------------------------------------

    # Excel paste constants
    xlPasteValues       = -4163
    xlPasteFormats      = -4122
    xlPasteColumnWidths = 8
    xlAscending         = 1
    xlSortOnValues      = 0
    xlNo                = 2

    tmp_dir  = Path(tempfile.gettempdir())
    pdf_path = tmp_dir / f"breadth_{uuid.uuid4().hex}.pdf"
    png_path = pdf_path.with_suffix(".png")

    # ---- locate Ghostscript -----------------------------------------
    gs = (os.environ.get("GSWIN")
          or shutil.which("gswin64c")
          or shutil.which("gswin32c"))
    if not gs:
        default = Path(r"C:\Program Files\gs")
        if default.exists():
            for ver in sorted(default.glob("gs*"), reverse=True):
                exe = ver / "bin" / "gswin64c.exe"
                if exe.exists():
                    gs = str(exe)
                    break
    if not gs:
        raise RuntimeError("Ghostscript not found. "
                           "Add its bin folder to PATH or set env var GSWIN.")

    pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)
    excel = None
    try:
        excel = DispatchEx("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False

        # 1 · Open workbook and sort -----------------------------------
        src_wb = excel.Workbooks.Open(str(xlsx_path))
        ws     = src_wb.Worksheets(SRC_SHEET)

        rng = ws.Range(SORT_RANGE)
        ws.Sort.SortFields.Clear()
        ws.Sort.SortFields.Add(
            Key=ws.Range(SORT_KEY),
            SortOn=xlSortOnValues,
            Order=xlAscending,
            DataOption=0
        )
        ws.Sort.SetRange(rng)
        ws.Sort.Header = xlNo
        ws.Sort.Apply()

        # 2 · Copy sorted range as static table ------------------------
        ws.Range(COPY_RANGE).Copy()
        tmp_wb = excel.Workbooks.Add()
        tws    = tmp_wb.Worksheets(1)
        tws.Range("A1").PasteSpecial(Paste=xlPasteValues)
        tws.Range("A1").PasteSpecial(Paste=xlPasteFormats)
        tws.Range("A1").PasteSpecial(Paste=xlPasteColumnWidths)

        tws.PageSetup.Zoom = False
        tws.PageSetup.FitToPagesWide = 1
        tws.PageSetup.FitToPagesTall = False

        # 3 · Export to PDF -------------------------------------------
        tmp_wb.ExportAsFixedFormat(Type=0, Filename=str(pdf_path))
        pdf_width_in = tws.UsedRange.Width / 72.0
        tmp_wb.Close(False)
        src_wb.Close(False)

        # 4 · Rasterise with Ghostscript ------------------------------
        target_px  = min(MAX_PX, (SLIDE_WIDTH_CM / 2.54) * 600)  # ~600 dpi cap
        dpi_needed = math.ceil(target_px / pdf_width_in)

        subprocess.run(
            [gs, "-dSAFER", "-dBATCH", "-dNOPAUSE",
             "-sDEVICE=pngalpha",
             f"-r{dpi_needed}",
             f"-sOutputFile={png_path}",
             str(pdf_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if not png_path.exists() or png_path.stat().st_size == 0:
            raise RuntimeError("Ghostscript succeeded but PNG is empty.")

        # 5 · Crop transparent margins --------------------------------
        with Image.open(png_path) as im:
            bbox = im.getbbox()
            if bbox:
                im.crop(bbox).save(png_path)

        return png_path

    except Exception:
        traceback.print_exc()
        raise
    finally:
        if excel:
            excel.Quit()
        pythoncom.CoUninitialize()




def _paste_breadth_picture(
    prs, img_path: Path,
    *, left_cm=1.7, top_cm=5.42, width_cm=21.1, height_cm=4.79
):
    """Replace textbox 'funda_breadth' with the EMF/PNG picture."""
    from pptx.util import Cm

    for slide in prs.slides:
        target = next(
            (sh for sh in slide.shapes
             if (sh.has_text_frame and sh.text_frame.text.strip().lower() == "funda_breadth")
             or sh.name.lower().startswith("funda_breadth")),
            None,
        )
        if target:
            target._element.getparent().remove(target._element)
            slide.shapes.add_picture(
                str(img_path), Cm(left_cm), Cm(top_cm), Cm(width_cm), Cm(height_cm)
            )
            break
    return prs


# Import SMI functions from the dedicated module.  The SMI module resides
# in ``technical_analysis/equity/smi.py`` and provides helper functions
# analogous to the SPX, CSI, Nikkei, TASI, Sensex and DAX functions.  These
# allow technical analysis of the Swiss Market Index (SMI).  If the module is
# not present (e.g. during development), we define no‑op stand‑ins so that
# the application continues to run without error.  The fallback for the
# range computation uses the SPX range bounds to avoid crashing when the
# SMI module is missing.
try:
    from technical_analysis.equity.smi import (
        make_smi_figure,
        insert_smi_technical_chart_with_callout,
        insert_smi_technical_chart,
        insert_smi_technical_score_number,
        insert_smi_momentum_score_number,
        insert_smi_subtitle,
        insert_smi_average_gauge,
        insert_smi_technical_assessment,
        insert_smi_source,
        _get_smi_technical_score,
        _get_smi_momentum_score,
        _compute_range_bounds as _compute_range_bounds_smi,
    )
except Exception:
    # Define no‑op stand‑ins if the SMI module is unavailable
    def make_smi_figure(*args, **kwargs):  # type: ignore
        return go.Figure()

    def insert_smi_technical_chart_with_callout(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_smi_technical_chart(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_smi_technical_score_number(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_smi_momentum_score_number(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_smi_subtitle(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_smi_average_gauge(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_smi_technical_assessment(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_smi_source(prs, *args, **kwargs):  # type: ignore
        return prs

    def _get_smi_technical_score(*args, **kwargs):  # type: ignore
        return None

    def _get_smi_momentum_score(*args, **kwargs):  # type: ignore
        return None

    # Fallback: if the SMI module is unavailable, fall back to the SPX range computation
    def _compute_range_bounds_smi(*args, **kwargs):  # type: ignore
        return _compute_range_bounds_spx(*args, **kwargs)

# Import IBOV functions from the dedicated module.  The IBOV module resides
# in ``technical_analysis/equity/ibov.py`` and provides helper functions
# analogous to the SPX, CSI, Nikkei, TASI, Sensex, DAX and SMI modules.  These
# allow technical analysis of the Bovespa (IBOV).  If the module is not
# available, define fallbacks to avoid errors and use the SPX range computation
# as a last resort.
try:
    from technical_analysis.equity.ibov import (
        make_ibov_figure,
        insert_ibov_technical_chart_with_callout,
        insert_ibov_technical_chart,
        insert_ibov_technical_score_number,
        insert_ibov_momentum_score_number,
        insert_ibov_subtitle,
        insert_ibov_average_gauge,
        insert_ibov_technical_assessment,
        insert_ibov_source,
        _get_ibov_technical_score,
        _get_ibov_momentum_score,
        _compute_range_bounds as _compute_range_bounds_ibov,
    )
except Exception:
    # Define no‑op stand‑ins if the IBOV module is unavailable
    def make_ibov_figure(*args, **kwargs):  # type: ignore
        return go.Figure()

    def insert_ibov_technical_chart_with_callout(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_ibov_technical_chart(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_ibov_technical_score_number(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_ibov_momentum_score_number(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_ibov_subtitle(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_ibov_average_gauge(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_ibov_technical_assessment(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_ibov_source(prs, *args, **kwargs):  # type: ignore
        return prs

    def _get_ibov_technical_score(*args, **kwargs):  # type: ignore
        return None

    def _get_ibov_momentum_score(*args, **kwargs):  # type: ignore
        return None

    # Fallback: if the IBOV module is unavailable, fall back to the SPX range computation
    def _compute_range_bounds_ibov(*args, **kwargs):  # type: ignore
        return _compute_range_bounds_spx(*args, **kwargs)

# Import Mexbol functions from the dedicated module.  The Mexbol module resides
# in ``technical_analysis/equity/mexbol.py`` and provides helper functions
# analogous to the SPX, CSI, Nikkei, TASI, Sensex, DAX, SMI and IBOV modules.
# These allow technical analysis of the Mexbol index.  If the module is not
# available, define fallbacks to avoid errors and use the SPX range computation
# as a last resort.
try:
    from technical_analysis.equity.mexbol import (
        make_mexbol_figure,
        insert_mexbol_technical_chart_with_callout,
        insert_mexbol_technical_chart,
        insert_mexbol_technical_score_number,
        insert_mexbol_momentum_score_number,
        insert_mexbol_subtitle,
        insert_mexbol_average_gauge,
        insert_mexbol_technical_assessment,
        insert_mexbol_source,
        _get_mexbol_technical_score,
        _get_mexbol_momentum_score,
        _compute_range_bounds as _compute_range_bounds_mexbol,
    )
except Exception:
    # Define no‑op stand‑ins if the Mexbol module is unavailable
    def make_mexbol_figure(*args, **kwargs):  # type: ignore
        return go.Figure()

    def insert_mexbol_technical_chart_with_callout(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_mexbol_technical_chart(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_mexbol_technical_score_number(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_mexbol_momentum_score_number(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_mexbol_subtitle(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_mexbol_average_gauge(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_mexbol_technical_assessment(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_mexbol_source(prs, *args, **kwargs):  # type: ignore
        return prs

    def _get_mexbol_technical_score(*args, **kwargs):  # type: ignore
        return None

    def _get_mexbol_momentum_score(*args, **kwargs):  # type: ignore
        return None

    # Fallback: if the Mexbol module is unavailable, fall back to the SPX range computation
    def _compute_range_bounds_mexbol(*args, **kwargs):  # type: ignore
        return _compute_range_bounds_spx(*args, **kwargs)

# Import Gold functions from the dedicated module.  The Gold module resides
# in ``technical_analysis/commodity/gold.py`` and provides helper functions
# analogous to the SPX, CSI, Nikkei, TASI, Sensex, DAX, SMI and IBOV modules.
# To support running locally (e.g. when the technical_analysis package is
# unavailable), a second import attempt is made from a top‑level ``gold``
# module.  Fallbacks ensure the application remains functional even if the
# Gold module cannot be imported.
try:
    from technical_analysis.commodity.gold import (
        make_gold_figure,
        insert_gold_technical_chart_with_callout,
        insert_gold_technical_chart,
        insert_gold_technical_score_number,
        insert_gold_momentum_score_number,
        insert_gold_subtitle,
        insert_gold_average_gauge,
        insert_gold_technical_assessment,
        insert_gold_source,
        _get_gold_technical_score,
        _get_gold_momentum_score,
        _compute_range_bounds as _compute_range_bounds_gold,
    )
except Exception:
    try:
        from gold import (
            make_gold_figure,
            insert_gold_technical_chart_with_callout,
            insert_gold_technical_chart,
            insert_gold_technical_score_number,
            insert_gold_momentum_score_number,
            insert_gold_subtitle,
            insert_gold_average_gauge,
            insert_gold_technical_assessment,
            insert_gold_source,
            _get_gold_technical_score,
            _get_gold_momentum_score,
            _compute_range_bounds as _compute_range_bounds_gold,
        )
    except Exception:
        # Define no‑op stand‑ins if the Gold module is unavailable
        def make_gold_figure(*args, **kwargs):  # type: ignore
            return go.Figure()

        def insert_gold_technical_chart_with_callout(prs, *args, **kwargs):  # type: ignore
            return prs

        def insert_gold_technical_chart(prs, *args, **kwargs):  # type: ignore
            return prs

        def insert_gold_technical_score_number(prs, *args, **kwargs):  # type: ignore
            return prs

        def insert_gold_momentum_score_number(prs, *args, **kwargs):  # type: ignore
            return prs

        def insert_gold_subtitle(prs, *args, **kwargs):  # type: ignore
            return prs

        def insert_gold_average_gauge(prs, *args, **kwargs):  # type: ignore
            return prs

        def insert_gold_technical_assessment(prs, *args, **kwargs):  # type: ignore
            return prs

        def insert_gold_source(prs, *args, **kwargs):  # type: ignore
            return prs

        def _get_gold_technical_score(*args, **kwargs):  # type: ignore
            return None

        def _get_gold_momentum_score(*args, **kwargs):  # type: ignore
            return None

        # Fallback: if the Gold module is unavailable, fall back to the SPX range computation
        def _compute_range_bounds_gold(*args, **kwargs):  # type: ignore
            return _compute_range_bounds_spx(*args, **kwargs)

# Import Silver functions from the dedicated module.  Similar to Gold, these
# helpers reside in ``technical_analysis/commodity/silver.py``.  If that
# package is unavailable, a second attempt is made to import a top‑level
# ``silver`` module.  No‑op fallbacks are defined if both imports fail.
try:
    from technical_analysis.commodity.silver import (
        make_silver_figure,
        insert_silver_technical_chart_with_callout,
        insert_silver_technical_chart,
        insert_silver_technical_score_number,
        insert_silver_momentum_score_number,
        insert_silver_subtitle,
        insert_silver_average_gauge,
        insert_silver_technical_assessment,
        insert_silver_source,
        _get_silver_technical_score,
        _get_silver_momentum_score,
        _compute_range_bounds as _compute_range_bounds_silver,
    )
except Exception:
    try:
        from silver import (
            make_silver_figure,
            insert_silver_technical_chart_with_callout,
            insert_silver_technical_chart,
            insert_silver_technical_score_number,
            insert_silver_momentum_score_number,
            insert_silver_subtitle,
            insert_silver_average_gauge,
            insert_silver_technical_assessment,
            insert_silver_source,
            _get_silver_technical_score,
            _get_silver_momentum_score,
            _compute_range_bounds as _compute_range_bounds_silver,
        )
    except Exception:
        def make_silver_figure(*args, **kwargs):  # type: ignore
            return go.Figure()
        def insert_silver_technical_chart_with_callout(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_silver_technical_chart(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_silver_technical_score_number(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_silver_momentum_score_number(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_silver_subtitle(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_silver_average_gauge(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_silver_technical_assessment(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_silver_source(prs, *args, **kwargs):  # type: ignore
            return prs
        def _get_silver_technical_score(*args, **kwargs):  # type: ignore
            return None
        def _get_silver_momentum_score(*args, **kwargs):  # type: ignore
            return None
        def _compute_range_bounds_silver(*args, **kwargs):  # type: ignore
            return _compute_range_bounds_spx(*args, **kwargs)

# Import Platinum functions from the dedicated module.  Similar to Gold and Silver,
# these helpers reside in ``technical_analysis/commodity/platinum.py``.  If that
# package is unavailable, a second attempt is made to import a top‑level
# ``platinum`` module.  No‑op fallbacks are defined if both imports fail.
try:
    from technical_analysis.commodity.platinum import (
        make_platinum_figure,
        insert_platinum_technical_chart_with_callout,
        insert_platinum_technical_chart,
        insert_platinum_technical_score_number,
        insert_platinum_momentum_score_number,
        insert_platinum_subtitle,
        insert_platinum_average_gauge,
        insert_platinum_technical_assessment,
        insert_platinum_source,
        _get_platinum_technical_score,
        _get_platinum_momentum_score,
        _compute_range_bounds as _compute_range_bounds_platinum,
    )
except Exception:
    try:
        from platinum import (
            make_platinum_figure,
            insert_platinum_technical_chart_with_callout,
            insert_platinum_technical_chart,
            insert_platinum_technical_score_number,
            insert_platinum_momentum_score_number,
            insert_platinum_subtitle,
            insert_platinum_average_gauge,
            insert_platinum_technical_assessment,
            insert_platinum_source,
            _get_platinum_technical_score,
            _get_platinum_momentum_score,
            _compute_range_bounds as _compute_range_bounds_platinum,
        )
    except Exception:
        def make_platinum_figure(*args, **kwargs):  # type: ignore
            return go.Figure()
        def insert_platinum_technical_chart_with_callout(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_platinum_technical_chart(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_platinum_technical_score_number(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_platinum_momentum_score_number(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_platinum_subtitle(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_platinum_average_gauge(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_platinum_technical_assessment(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_platinum_source(prs, *args, **kwargs):  # type: ignore
            return prs
        def _get_platinum_technical_score(*args, **kwargs):  # type: ignore
            return None
        def _get_platinum_momentum_score(*args, **kwargs):  # type: ignore
            return None
        def _compute_range_bounds_platinum(*args, **kwargs):  # type: ignore
            return _compute_range_bounds_spx(*args, **kwargs)
except Exception:
    # Define no‑op stand‑ins if the SMI module is unavailable
    def make_smi_figure(*args, **kwargs):  # type: ignore
        return go.Figure()

    def insert_smi_technical_chart_with_callout(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_smi_technical_chart(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_smi_technical_score_number(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_smi_momentum_score_number(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_smi_subtitle(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_smi_average_gauge(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_smi_technical_assessment(prs, *args, **kwargs):  # type: ignore
        return prs

    def insert_smi_source(prs, *args, **kwargs):  # type: ignore
        return prs

    def _get_smi_technical_score(*args, **kwargs):  # type: ignore
        return None

    def _get_smi_momentum_score(*args, **kwargs):  # type: ignore
        return None

    # Fallback: if the SMI module is unavailable, fall back to the SPX range computation
    def _compute_range_bounds_smi(*args, **kwargs):  # type: ignore
        return _compute_range_bounds_spx(*args, **kwargs)

# Import Oil functions from the dedicated module.  Similar to Gold, Silver and Platinum,
# these helpers reside in ``technical_analysis/commodity/oil.py``.  If that
# package is unavailable, a second attempt is made to import a top‑level
# ``oil`` module.  No‑op fallbacks are defined if both imports fail.
try:
    from technical_analysis.commodity.oil import (
        make_oil_figure,
        insert_oil_technical_chart_with_callout,
        insert_oil_technical_chart,
        insert_oil_technical_score_number,
        insert_oil_momentum_score_number,
        insert_oil_subtitle,
        insert_oil_average_gauge,
        insert_oil_technical_assessment,
        insert_oil_source,
        _get_oil_technical_score,
        _get_oil_momentum_score,
        _compute_range_bounds as _compute_range_bounds_oil,
    )
except Exception:
    try:
        from oil import (
            make_oil_figure,
            insert_oil_technical_chart_with_callout,
            insert_oil_technical_chart,
            insert_oil_technical_score_number,
            insert_oil_momentum_score_number,
            insert_oil_subtitle,
            insert_oil_average_gauge,
            insert_oil_technical_assessment,
            insert_oil_source,
            _get_oil_technical_score,
            _get_oil_momentum_score,
            _compute_range_bounds as _compute_range_bounds_oil,
        )
    except Exception:
        def make_oil_figure(*args, **kwargs):  # type: ignore
            return go.Figure()
        def insert_oil_technical_chart_with_callout(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_oil_technical_chart(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_oil_technical_score_number(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_oil_momentum_score_number(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_oil_subtitle(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_oil_average_gauge(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_oil_technical_assessment(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_oil_source(prs, *args, **kwargs):  # type: ignore
            return prs
        def _get_oil_technical_score(*args, **kwargs):  # type: ignore
            return None
        def _get_oil_momentum_score(*args, **kwargs):  # type: ignore
            return None
        def _compute_range_bounds_oil(*args, **kwargs):  # type: ignore
            return _compute_range_bounds_spx(*args, **kwargs)

# Import Copper functions from the dedicated module.  Similar to other commodities,
# helpers reside in ``technical_analysis/commodity/copper.py``.  If that package
# is unavailable, a second attempt is made to import a top‑level ``copper``
# module.  Fallback functions ensure the application remains functional when
# copper analysis is not available.
try:
    from technical_analysis.commodity.copper import (
        make_copper_figure,
        insert_copper_technical_chart_with_callout,
        insert_copper_technical_chart,
        insert_copper_technical_score_number,
        insert_copper_momentum_score_number,
        insert_copper_subtitle,
        insert_copper_average_gauge,
        insert_copper_technical_assessment,
        insert_copper_source,
        _get_copper_technical_score,
        _get_copper_momentum_score,
        _compute_range_bounds as _compute_range_bounds_copper,
    )
except Exception:
    try:
        from copper import (
            make_copper_figure,
            insert_copper_technical_chart_with_callout,
            insert_copper_technical_chart,
            insert_copper_technical_score_number,
            insert_copper_momentum_score_number,
            insert_copper_subtitle,
            insert_copper_average_gauge,
            insert_copper_technical_assessment,
            insert_copper_source,
            _get_copper_technical_score,
            _get_copper_momentum_score,
            _compute_range_bounds as _compute_range_bounds_copper,
        )
    except Exception:
        def make_copper_figure(*args, **kwargs):  # type: ignore
            return go.Figure()
        def insert_copper_technical_chart_with_callout(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_copper_technical_chart(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_copper_technical_score_number(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_copper_momentum_score_number(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_copper_subtitle(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_copper_average_gauge(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_copper_technical_assessment(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_copper_source(prs, *args, **kwargs):  # type: ignore
            return prs
        def _get_copper_technical_score(*args, **kwargs):  # type: ignore
            return None
        def _get_copper_momentum_score(*args, **kwargs):  # type: ignore
            return None
        def _compute_range_bounds_copper(*args, **kwargs):  # type: ignore
            return _compute_range_bounds_spx(*args, **kwargs)

# Import Palladium functions from the dedicated module.  Similar to other commodities,
# these helpers reside in ``technical_analysis/commodity/palladium.py``.  If that
# package is unavailable, a second attempt is made to import a top‑level
# ``palladium`` module.  No‑op fallbacks are defined if both imports fail.
try:
    # Preferred: import from the standard technical_analysis package
    from technical_analysis.commodity.palladium import (
        make_palladium_figure,
        insert_palladium_technical_chart_with_callout,
        insert_palladium_technical_chart,
        insert_palladium_technical_score_number,
        insert_palladium_momentum_score_number,
        insert_palladium_subtitle,
        insert_palladium_average_gauge,
        insert_palladium_technical_assessment,
        insert_palladium_source,
        _get_palladium_technical_score,
        _get_palladium_momentum_score,
        _compute_range_bounds as _compute_range_bounds_palladium,
    )
except Exception:
    try:
        # Secondary: import from our extended implementation if available
        from palladium_full import (
            make_palladium_figure,
            insert_palladium_technical_chart_with_callout,
            insert_palladium_technical_chart,
            insert_palladium_technical_score_number,
            insert_palladium_momentum_score_number,
            insert_palladium_subtitle,
            insert_palladium_average_gauge,
            insert_palladium_technical_assessment,
            insert_palladium_source,
            _get_palladium_technical_score,
            _get_palladium_momentum_score,
            _compute_range_bounds as _compute_range_bounds_palladium,
        )
    except Exception:
        try:
            # Fallback: import from the lean palladium module
            from palladium import (
                make_palladium_figure,
                insert_palladium_technical_chart_with_callout,
                insert_palladium_technical_chart,
                insert_palladium_technical_score_number,
                insert_palladium_momentum_score_number,
                insert_palladium_subtitle,
                insert_palladium_average_gauge,
                insert_palladium_technical_assessment,
                insert_palladium_source,
                _get_palladium_technical_score,
                _get_palladium_momentum_score,
                _compute_range_bounds as _compute_range_bounds_palladium,
            )
        except Exception:
            # No implementation found – define harmless stand‑ins
            def make_palladium_figure(*args, **kwargs):  # type: ignore
                return go.Figure()
            def insert_palladium_technical_chart_with_callout(prs, *args, **kwargs):  # type: ignore
                return prs
            def insert_palladium_technical_chart(prs, *args, **kwargs):  # type: ignore
                return prs
            def insert_palladium_technical_score_number(prs, *args, **kwargs):  # type: ignore
                return prs
            def insert_palladium_momentum_score_number(prs, *args, **kwargs):  # type: ignore
                return prs
            def insert_palladium_subtitle(prs, *args, **kwargs):  # type: ignore
                return prs
            def insert_palladium_average_gauge(prs, *args, **kwargs):  # type: ignore
                return prs
            def insert_palladium_technical_assessment(prs, *args, **kwargs):  # type: ignore
                return prs
            def insert_palladium_source(prs, *args, **kwargs):  # type: ignore
                return prs
            def _get_palladium_technical_score(*args, **kwargs):  # type: ignore
                return None
            def _get_palladium_momentum_score(*args, **kwargs):  # type: ignore
                return None
            def _compute_range_bounds_palladium(*args, **kwargs):  # type: ignore
                return _compute_range_bounds_spx(*args, **kwargs)

# Import Bitcoin functions from the dedicated module.  The Bitcoin module resides
# in ``technical_analysis/crypto/bitcoin.py`` and provides helper functions
# analogous to those for commodities.  If that package cannot be imported,
# a fallback attempt is made to import from a top‑level ``bitcoin`` module.
# When both imports fail, define no‑op stand‑ins to keep the app running.
try:
    from technical_analysis.crypto.bitcoin import (
        make_bitcoin_figure,
        insert_bitcoin_technical_chart_with_callout,
        insert_bitcoin_technical_chart,
        insert_bitcoin_technical_score_number,
        insert_bitcoin_momentum_score_number,
        insert_bitcoin_subtitle,
        insert_bitcoin_average_gauge,
        insert_bitcoin_technical_assessment,
        insert_bitcoin_source,
        _get_bitcoin_technical_score,
        _get_bitcoin_momentum_score,
        _compute_range_bounds as _compute_range_bounds_bitcoin,
    )
except Exception:
    try:
        from bitcoin import (
            make_bitcoin_figure,
            insert_bitcoin_technical_chart_with_callout,
            insert_bitcoin_technical_chart,
            insert_bitcoin_technical_score_number,
            insert_bitcoin_momentum_score_number,
            insert_bitcoin_subtitle,
            insert_bitcoin_average_gauge,
            insert_bitcoin_technical_assessment,
            insert_bitcoin_source,
            _get_bitcoin_technical_score,
            _get_bitcoin_momentum_score,
            _compute_range_bounds as _compute_range_bounds_bitcoin,
        )
    except Exception:
        def make_bitcoin_figure(*args, **kwargs):  # type: ignore
            return go.Figure()
        def insert_bitcoin_technical_chart_with_callout(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_bitcoin_technical_chart(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_bitcoin_technical_score_number(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_bitcoin_momentum_score_number(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_bitcoin_subtitle(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_bitcoin_average_gauge(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_bitcoin_technical_assessment(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_bitcoin_source(prs, *args, **kwargs):  # type: ignore
            return prs
        def _get_bitcoin_technical_score(*args, **kwargs):  # type: ignore
            return None
        def _get_bitcoin_momentum_score(*args, **kwargs):  # type: ignore
            return None
        def _compute_range_bounds_bitcoin(*args, **kwargs):  # type: ignore
            return _compute_range_bounds_spx(*args, **kwargs)

# Import Ethereum functions from the dedicated module.  The Ethereum module resides
# in ``technical_analysis/crypto/ethereum.py`` and provides helper functions
# analogous to those for other crypto assets.  If unavailable, fall back to a
# top‑level ``ethereum`` module.  Define no‑op stand‑ins as a last resort.
try:
    from technical_analysis.crypto.ethereum import (
        make_ethereum_figure,
        insert_ethereum_technical_chart_with_callout,
        insert_ethereum_technical_chart,
        insert_ethereum_technical_score_number,
        insert_ethereum_momentum_score_number,
        insert_ethereum_subtitle,
        insert_ethereum_average_gauge,
        insert_ethereum_technical_assessment,
        insert_ethereum_source,
        _get_ethereum_technical_score,
        _get_ethereum_momentum_score,
        _compute_range_bounds as _compute_range_bounds_ethereum,
    )
except Exception:
    try:
        from ethereum import (
            make_ethereum_figure,
            insert_ethereum_technical_chart_with_callout,
            insert_ethereum_technical_chart,
            insert_ethereum_technical_score_number,
            insert_ethereum_momentum_score_number,
            insert_ethereum_subtitle,
            insert_ethereum_average_gauge,
            insert_ethereum_technical_assessment,
            insert_ethereum_source,
            _get_ethereum_technical_score,
            _get_ethereum_momentum_score,
            _compute_range_bounds as _compute_range_bounds_ethereum,
        )
    except Exception:
        def make_ethereum_figure(*args, **kwargs):  # type: ignore
            return go.Figure()
        def insert_ethereum_technical_chart_with_callout(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_ethereum_technical_chart(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_ethereum_technical_score_number(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_ethereum_momentum_score_number(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_ethereum_subtitle(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_ethereum_average_gauge(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_ethereum_technical_assessment(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_ethereum_source(prs, *args, **kwargs):  # type: ignore
            return prs
        def _get_ethereum_technical_score(*args, **kwargs):  # type: ignore
            return None
        def _get_ethereum_momentum_score(*args, **kwargs):  # type: ignore
            return None
        def _compute_range_bounds_ethereum(*args, **kwargs):  # type: ignore
            return _compute_range_bounds_spx(*args, **kwargs)

# Import Ripple functions from the dedicated module.  The Ripple module resides
# in ``technical_analysis/crypto/ripple.py`` and provides helper functions
# analogous to the other crypto assets.  If unavailable, fall back to a
# top‑level ``ripple`` module or define no‑op stand‑ins.
try:
    from technical_analysis.crypto.ripple import (
        make_ripple_figure,
        insert_ripple_technical_chart_with_callout,
        insert_ripple_technical_chart,
        insert_ripple_technical_score_number,
        insert_ripple_momentum_score_number,
        insert_ripple_subtitle,
        insert_ripple_average_gauge,
        insert_ripple_technical_assessment,
        insert_ripple_source,
        _get_ripple_technical_score,
        _get_ripple_momentum_score,
        _compute_range_bounds as _compute_range_bounds_ripple,
    )
except Exception:
    try:
        from ripple import (
            make_ripple_figure,
            insert_ripple_technical_chart_with_callout,
            insert_ripple_technical_chart,
            insert_ripple_technical_score_number,
            insert_ripple_momentum_score_number,
            insert_ripple_subtitle,
            insert_ripple_average_gauge,
            insert_ripple_technical_assessment,
            insert_ripple_source,
            _get_ripple_technical_score,
            _get_ripple_momentum_score,
            _compute_range_bounds as _compute_range_bounds_ripple,
        )
    except Exception:
        def make_ripple_figure(*args, **kwargs):  # type: ignore
            return go.Figure()
        def insert_ripple_technical_chart_with_callout(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_ripple_technical_chart(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_ripple_technical_score_number(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_ripple_momentum_score_number(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_ripple_subtitle(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_ripple_average_gauge(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_ripple_technical_assessment(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_ripple_source(prs, *args, **kwargs):  # type: ignore
            return prs
        def _get_ripple_technical_score(*args, **kwargs):  # type: ignore
            return None
        def _get_ripple_momentum_score(*args, **kwargs):  # type: ignore
            return None
        def _compute_range_bounds_ripple(*args, **kwargs):  # type: ignore
            return _compute_range_bounds_spx(*args, **kwargs)

# Import Solana functions from the dedicated module.  The Solana module resides
# in ``technical_analysis/crypto/solana.py`` and provides helper functions
# analogous to the other crypto assets.  If unavailable, fall back to a
# top‑level ``solana`` module or define no‑op stand‑ins.
try:
    from technical_analysis.crypto.solana import (
        make_solana_figure,
        insert_solana_technical_chart_with_callout,
        insert_solana_technical_chart,
        insert_solana_technical_score_number,
        insert_solana_momentum_score_number,
        insert_solana_subtitle,
        insert_solana_average_gauge,
        insert_solana_technical_assessment,
        insert_solana_source,
        _get_solana_technical_score,
        _get_solana_momentum_score,
        _compute_range_bounds as _compute_range_bounds_solana,
    )
except Exception:
    try:
        from solana import (
            make_solana_figure,
            insert_solana_technical_chart_with_callout,
            insert_solana_technical_chart,
            insert_solana_technical_score_number,
            insert_solana_momentum_score_number,
            insert_solana_subtitle,
            insert_solana_average_gauge,
            insert_solana_technical_assessment,
            insert_solana_source,
            _get_solana_technical_score,
            _get_solana_momentum_score,
            _compute_range_bounds as _compute_range_bounds_solana,
        )
    except Exception:
        def make_solana_figure(*args, **kwargs):  # type: ignore
            return go.Figure()
        def insert_solana_technical_chart_with_callout(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_solana_technical_chart(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_solana_technical_score_number(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_solana_momentum_score_number(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_solana_subtitle(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_solana_average_gauge(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_solana_technical_assessment(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_solana_source(prs, *args, **kwargs):  # type: ignore
            return prs
        def _get_solana_technical_score(*args, **kwargs):  # type: ignore
            return None
        def _get_solana_momentum_score(*args, **kwargs):  # type: ignore
            return None
        def _compute_range_bounds_solana(*args, **kwargs):  # type: ignore
            return _compute_range_bounds_spx(*args, **kwargs)

# Import Binance functions from the dedicated module.  The Binance module resides
# in ``technical_analysis/crypto/binance.py`` and provides helper functions
# analogous to the other crypto assets.  If unavailable, fall back to a
# top‑level ``binance`` module or define no‑op stand‑ins.
try:
    from technical_analysis.crypto.binance import (
        make_binance_figure,
        insert_binance_technical_chart_with_callout,
        insert_binance_technical_chart,
        insert_binance_technical_score_number,
        insert_binance_momentum_score_number,
        insert_binance_subtitle,
        insert_binance_average_gauge,
        insert_binance_technical_assessment,
        insert_binance_source,
        _get_binance_technical_score,
        _get_binance_momentum_score,
        _compute_range_bounds as _compute_range_bounds_binance,
    )
except Exception:
    try:
        from binance import (
            make_binance_figure,
            insert_binance_technical_chart_with_callout,
            insert_binance_technical_chart,
            insert_binance_technical_score_number,
            insert_binance_momentum_score_number,
            insert_binance_subtitle,
            insert_binance_average_gauge,
            insert_binance_technical_assessment,
            insert_binance_source,
            _get_binance_technical_score,
            _get_binance_momentum_score,
            _compute_range_bounds as _compute_range_bounds_binance,
        )
    except Exception:
        def make_binance_figure(*args, **kwargs):  # type: ignore
            return go.Figure()
        def insert_binance_technical_chart_with_callout(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_binance_technical_chart(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_binance_technical_score_number(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_binance_momentum_score_number(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_binance_subtitle(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_binance_average_gauge(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_binance_technical_assessment(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_binance_source(prs, *args, **kwargs):  # type: ignore
            return prs
        def _get_binance_technical_score(*args, **kwargs):  # type: ignore
            return None
        def _get_binance_momentum_score(*args, **kwargs):  # type: ignore
            return None
        def _compute_range_bounds_binance(*args, **kwargs):  # type: ignore
            return _compute_range_bounds_spx(*args, **kwargs)

# Import Ethereum functions from the dedicated module.  The Ethereum module resides
# in ``technical_analysis/crypto/ethereum.py`` and provides helper functions
# analogous to the Bitcoin module.  If it cannot be imported, a fallback to a
# top‑level ``ethereum`` module is attempted.  When both imports fail, no‑op
# stand‑ins are defined.
try:
    from technical_analysis.crypto.ethereum import (
        make_ethereum_figure,
        insert_ethereum_technical_chart_with_callout,
        insert_ethereum_technical_chart,
        insert_ethereum_technical_score_number,
        insert_ethereum_momentum_score_number,
        insert_ethereum_subtitle,
        insert_ethereum_average_gauge,
        insert_ethereum_technical_assessment,
        insert_ethereum_source,
        _get_ethereum_technical_score,
        _get_ethereum_momentum_score,
        _compute_range_bounds as _compute_range_bounds_ethereum,
    )
except Exception:
    try:
        from ethereum import (
            make_ethereum_figure,
            insert_ethereum_technical_chart_with_callout,
            insert_ethereum_technical_chart,
            insert_ethereum_technical_score_number,
            insert_ethereum_momentum_score_number,
            insert_ethereum_subtitle,
            insert_ethereum_average_gauge,
            insert_ethereum_technical_assessment,
            insert_ethereum_source,
            _get_ethereum_technical_score,
            _get_ethereum_momentum_score,
            _compute_range_bounds as _compute_range_bounds_ethereum,
        )
    except Exception:
        def make_ethereum_figure(*args, **kwargs):  # type: ignore
            return go.Figure()
        def insert_ethereum_technical_chart_with_callout(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_ethereum_technical_chart(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_ethereum_technical_score_number(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_ethereum_momentum_score_number(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_ethereum_subtitle(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_ethereum_average_gauge(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_ethereum_technical_assessment(prs, *args, **kwargs):  # type: ignore
            return prs
        def insert_ethereum_source(prs, *args, **kwargs):  # type: ignore
            return prs
        def _get_ethereum_technical_score(*args, **kwargs):  # type: ignore
            return None
        def _get_ethereum_momentum_score(*args, **kwargs):  # type: ignore
            return None
        def _compute_range_bounds_ethereum(*args, **kwargs):  # type: ignore
            return _compute_range_bounds_spx(*args, **kwargs)

# Import CSI functions from the dedicated module.  The CSI module resides
# in ``technical_analysis/equity/csi.py`` and provides helper functions
# analogous to the SPX functions.  These allow technical analysis of the
# Shenzhen CSI 300 index.  If the module is not present, Streamlit
# will fall back gracefully when CSI analysis is not requested.
try:
    from technical_analysis.equity.csi import (
        make_csi_figure,
        insert_csi_technical_chart_with_callout,
        insert_csi_technical_chart,
        insert_csi_technical_score_number,
        insert_csi_momentum_score_number,
        insert_csi_subtitle,
        insert_csi_average_gauge,
        insert_csi_technical_assessment,
        insert_csi_source,
        _get_csi_technical_score,
        _get_csi_momentum_score,
        _compute_range_bounds as _compute_range_bounds_csi,
    )
except Exception:
    # Define no-op stand‑ins if the CSI module is unavailable
    def make_csi_figure(*args, **kwargs):
        return go.Figure()
    def insert_csi_technical_chart_with_callout(prs, *args, **kwargs):
        return prs
    def insert_csi_technical_chart(prs, *args, **kwargs):
        return prs
    def insert_csi_technical_score_number(prs, *args, **kwargs):
        return prs
    def insert_csi_momentum_score_number(prs, *args, **kwargs):
        return prs
    def insert_csi_subtitle(prs, *args, **kwargs):
        return prs
    def insert_csi_average_gauge(prs, *args, **kwargs):
        return prs
    def insert_csi_technical_assessment(prs, *args, **kwargs):
        return prs
    def insert_csi_source(prs, *args, **kwargs):
        return prs
    def _get_csi_technical_score(*args, **kwargs):
        return None
    def _get_csi_momentum_score(*args, **kwargs):
        return None

    # Fallback: if the CSI module is unavailable, fall back to the SPX range computation
    def _compute_range_bounds_csi(*args, **kwargs):  # type: ignore
        return _compute_range_bounds_spx(*args, **kwargs)

# Import Nikkei functions from the dedicated module.  The Nikkei module
# resides in ``technical_analysis/equity/nikkei.py`` and provides helper
# functions analogous to the SPX and CSI functions.  These allow
# technical analysis of the Nikkei 225 index.  If the module is not
# present, Streamlit will fall back gracefully when Nikkei analysis is
# not requested.
try:
    from technical_analysis.equity.nikkei import (
        make_nikkei_figure,
        insert_nikkei_technical_chart_with_callout,
        insert_nikkei_technical_chart,
        insert_nikkei_technical_score_number,
        insert_nikkei_momentum_score_number,
        insert_nikkei_subtitle,
        insert_nikkei_average_gauge,
        insert_nikkei_technical_assessment,
        insert_nikkei_source,
        _get_nikkei_technical_score,
        _get_nikkei_momentum_score,
        _compute_range_bounds as _compute_range_bounds_nikkei,
    )
except Exception:
    # Define no-op stand‑ins if the Nikkei module is unavailable
    def make_nikkei_figure(*args, **kwargs):
        return go.Figure()

    def insert_nikkei_technical_chart_with_callout(prs, *args, **kwargs):
        return prs

    def insert_nikkei_technical_chart(prs, *args, **kwargs):
        return prs

    def insert_nikkei_technical_score_number(prs, *args, **kwargs):
        return prs

    def insert_nikkei_momentum_score_number(prs, *args, **kwargs):
        return prs

    def insert_nikkei_subtitle(prs, *args, **kwargs):
        return prs

    def insert_nikkei_average_gauge(prs, *args, **kwargs):
        return prs

    def insert_nikkei_technical_assessment(prs, *args, **kwargs):
        return prs

    def insert_nikkei_source(prs, *args, **kwargs):
        return prs

    def _get_nikkei_technical_score(*args, **kwargs):
        return None

    def _get_nikkei_momentum_score(*args, **kwargs):
        return None

    # Fallback: if the Nikkei module is unavailable, fall back to the SPX range computation
    def _compute_range_bounds_nikkei(*args, **kwargs):  # type: ignore
        return _compute_range_bounds_spx(*args, **kwargs)

# Import TASI functions from the dedicated module.  The TASI module
# resides in ``technical_analysis/equity/tasi.py`` and provides helper
# functions analogous to the SPX, CSI and Nikkei functions.  These allow
# technical analysis of the TASI (Saudi) index.  If the module is not
# present, Streamlit will fall back gracefully when TASI analysis is
# not requested.
try:
    from technical_analysis.equity.tasi import (
        make_tasi_figure,
        insert_tasi_technical_chart_with_callout,
        insert_tasi_technical_chart,
        insert_tasi_technical_score_number,
        insert_tasi_momentum_score_number,
        insert_tasi_subtitle,
        insert_tasi_average_gauge,
        insert_tasi_technical_assessment,
        insert_tasi_source,
        _get_tasi_technical_score,
        _get_tasi_momentum_score,
        _compute_range_bounds as _compute_range_bounds_tasi,
    )
except Exception:
    # Define no-op stand‑ins if the TASI module is unavailable
    def make_tasi_figure(*args, **kwargs):
        return go.Figure()
    def insert_tasi_technical_chart_with_callout(prs, *args, **kwargs):
        return prs
    def insert_tasi_technical_chart(prs, *args, **kwargs):
        return prs
    def insert_tasi_technical_score_number(prs, *args, **kwargs):
        return prs
    def insert_tasi_momentum_score_number(prs, *args, **kwargs):
        return prs
    def insert_tasi_subtitle(prs, *args, **kwargs):
        return prs
    def insert_tasi_average_gauge(prs, *args, **kwargs):
        return prs
    def insert_tasi_technical_assessment(prs, *args, **kwargs):
        return prs
    def insert_tasi_source(prs, *args, **kwargs):
        return prs
    def _get_tasi_technical_score(*args, **kwargs):
        return None
    def _get_tasi_momentum_score(*args, **kwargs):
        return None
    # Fallback: use the SPX range computation as a generic fallback
    def _compute_range_bounds_tasi(*args, **kwargs):  # type: ignore
        return _compute_range_bounds_spx(*args, **kwargs)

# Import Sensex functions from the dedicated module.  The Sensex module resides
# in ``technical_analysis/equity/sensex.py`` and provides helper functions
# analogous to the SPX, CSI, Nikkei and TASI functions.  These allow
# technical analysis of the BSE Sensex 30 index.  If the module is not present,
# Streamlit will fall back gracefully when Sensex analysis is not requested.
try:
    from technical_analysis.equity.sensex import (
        make_sensex_figure,
        insert_sensex_technical_chart_with_callout,
        insert_sensex_technical_chart,
        insert_sensex_technical_score_number,
        insert_sensex_momentum_score_number,
        insert_sensex_subtitle,
        insert_sensex_average_gauge,
        insert_sensex_technical_assessment,
        insert_sensex_source,
        _get_sensex_technical_score,
        _get_sensex_momentum_score,
        _compute_range_bounds as _compute_range_bounds_sensex,
    )
except Exception:
    # Define no-op stand-ins if the Sensex module is unavailable
    def make_sensex_figure(*args, **kwargs):
        return go.Figure()
    def insert_sensex_technical_chart_with_callout(prs, *args, **kwargs):
        return prs
    def insert_sensex_technical_chart(prs, *args, **kwargs):
        return prs
    def insert_sensex_technical_score_number(prs, *args, **kwargs):
        return prs
    def insert_sensex_momentum_score_number(prs, *args, **kwargs):
        return prs
    def insert_sensex_subtitle(prs, *args, **kwargs):
        return prs
    def insert_sensex_average_gauge(prs, *args, **kwargs):
        return prs
    def insert_sensex_technical_assessment(prs, *args, **kwargs):
        return prs
    def insert_sensex_source(prs, *args, **kwargs):
        return prs
    def _get_sensex_technical_score(*args, **kwargs):
        return None
    def _get_sensex_momentum_score(*args, **kwargs):
        return None
    # Fallback: use the SPX range computation as a generic fallback
    def _compute_range_bounds_sensex(*args, **kwargs):  # type: ignore
        return _compute_range_bounds_spx(*args, **kwargs)

# Import DAX functions from the dedicated module.  The DAX module resides
# in ``technical_analysis/equity/dax.py`` and provides helper functions
# analogous to the SPX, CSI, Nikkei, TASI and Sensex functions.  These allow
# technical analysis of the German DAX index.  If the module is not present,
# Streamlit will fall back gracefully when DAX analysis is not requested.
try:
    from technical_analysis.equity.dax import (
        make_dax_figure,
        insert_dax_technical_chart_with_callout,
        insert_dax_technical_chart,
        insert_dax_technical_score_number,
        insert_dax_momentum_score_number,
        insert_dax_subtitle,
        insert_dax_average_gauge,
        insert_dax_technical_assessment,
        insert_dax_source,
        _get_dax_technical_score,
        _get_dax_momentum_score,
        _compute_range_bounds as _compute_range_bounds_dax,
    )
except Exception:
    # Define no‑op stand-ins if the DAX module is unavailable
    def make_dax_figure(*args, **kwargs):
        return go.Figure()
    def insert_dax_technical_chart_with_callout(prs, *args, **kwargs):
        return prs
    def insert_dax_technical_chart(prs, *args, **kwargs):
        return prs
    def insert_dax_technical_score_number(prs, *args, **kwargs):
        return prs
    def insert_dax_momentum_score_number(prs, *args, **kwargs):
        return prs
    def insert_dax_subtitle(prs, *args, **kwargs):
        return prs
    def insert_dax_average_gauge(prs, *args, **kwargs):
        return prs
    def insert_dax_technical_assessment(prs, *args, **kwargs):
        return prs
    def insert_dax_source(prs, *args, **kwargs):
        return prs
    def _get_dax_technical_score(*args, **kwargs):
        return None
    def _get_dax_momentum_score(*args, **kwargs):
        return None
    # Fallback: use the SPX range computation as a generic fallback
    def _compute_range_bounds_dax(*args, **kwargs):  # type: ignore
        return _compute_range_bounds_spx(*args, **kwargs)

# Import helper to adjust price data according to price mode.  The utils
# module resides at the project root (e.g. ``ic/utils.py``) so that it can
# be shared across technical analysis and performance modules.
from utils import adjust_prices_for_mode

# Import performance dashboard helpers (unchanged)
from performance.equity_perf import (
    create_weekly_performance_chart,
    create_historical_performance_table,
    insert_equity_performance_bar_slide,
    insert_equity_performance_histo_slide,
)

# Import FX performance functions
try:
    from performance.fx_perf import (
        create_weekly_performance_chart as create_weekly_fx_performance_chart,
        create_historical_performance_table as create_historical_fx_performance_table,
        insert_fx_performance_bar_slide,
        insert_fx_performance_histo_slide,
    )
except Exception:
    # If FX module not available, define no-op placeholders
    def create_weekly_fx_performance_chart(*args, **kwargs):
        return (b"", None)
    def create_historical_fx_performance_table(*args, **kwargs):
        return (b"", None)
    def insert_fx_performance_bar_slide(prs, image_bytes, *args, **kwargs):
        return prs
    def insert_fx_performance_histo_slide(prs, image_bytes, *args, **kwargs):
        return prs

# Import Crypto performance functions
try:
    from performance.crypto_perf import (
        create_weekly_performance_chart as create_weekly_crypto_performance_chart,
        create_historical_performance_table as create_historical_crypto_performance_table,
        insert_crypto_performance_bar_slide,
        insert_crypto_performance_histo_slide,
    )
except Exception:
    # If Crypto module not available, define no-op placeholders
    def create_weekly_crypto_performance_chart(*args, **kwargs):
        return (b"", None)
    def create_historical_crypto_performance_table(*args, **kwargs):
        return (b"", None)
    def insert_crypto_performance_bar_slide(prs, image_bytes, *args, **kwargs):
        return prs
    def insert_crypto_performance_histo_slide(prs, image_bytes, *args, **kwargs):
        return prs

# Import Credit performance functions
try:
    from performance.credit_perf import (
        create_weekly_performance_chart as create_weekly_credit_performance_chart,
        create_historical_performance_table as create_historical_credit_performance_table,
        insert_credit_performance_bar_slide,
        insert_credit_performance_histo_slide,
    )
except Exception:
    # If Credit module not available, define no-op placeholders
    def create_weekly_credit_performance_chart(*args, **kwargs):  # type: ignore
        return (b"", None)
    def create_historical_credit_performance_table(*args, **kwargs):  # type: ignore
        return (b"", None)
    def insert_credit_performance_bar_slide(prs, image_bytes, *args, **kwargs):  # type: ignore
        return prs
    def insert_credit_performance_histo_slide(prs, image_bytes, *args, **kwargs):  # type: ignore
        return prs

# Import Commodity performance functions
try:
    from performance.commodity_perf import (
        create_weekly_performance_chart as create_weekly_commodity_performance_chart,
        create_historical_performance_table as create_historical_commodity_performance_table,
        insert_commodity_performance_bar_slide,
        insert_commodity_performance_histo_slide,
    )
except Exception:
    def create_weekly_commodity_performance_chart(*args, **kwargs):  # type: ignore
        return (b"", None)
    def create_historical_commodity_performance_table(*args, **kwargs):  # type: ignore
        return (b"", None)
    def insert_commodity_performance_bar_slide(prs, image_bytes, *args, **kwargs):  # type: ignore
        return prs
    def insert_commodity_performance_histo_slide(prs, image_bytes, *args, **kwargs):  # type: ignore
        return prs
    
# Import Rates performance functions
try:
    from performance.rates_perf import (
        create_weekly_performance_chart as create_weekly_rates_performance_chart,
        create_historical_performance_table as create_historical_rates_performance_table,
        insert_rates_performance_bar_slide,
        insert_rates_performance_histo_slide,
    )
except Exception:
    # If Rates module is not available, define no-op placeholders
    def create_weekly_rates_performance_chart(*args, **kwargs):  # type: ignore
        return (b"", None)
    def create_historical_rates_performance_table(*args, **kwargs):  # type: ignore
        return (b"", None)
    def insert_rates_performance_bar_slide(prs, image_bytes, *args, **kwargs):  # type: ignore
        return prs
    def insert_rates_performance_histo_slide(prs, image_bytes, *args, **kwargs):  # type: ignore
        return prs

###############################################################################
# Synthetic data helpers (fallback when no Excel is loaded)
###############################################################################

def _create_synthetic_spx_series() -> pd.DataFrame:
    """Create a synthetic SPX price series for demonstration purposes."""
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=730)
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    np.random.seed(42)
    returns = np.random.normal(loc=0, scale=0.01, size=len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"Date": dates, "Price": prices})


def _add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add moving averages to a DataFrame with a Price column."""
    out = df.copy()
    for w in (50, 100, 200):
        out[f"MA_{w}"] = out["Price"].rolling(w, min_periods=1).mean()
    return out


def _build_fallback_figure(
    df_full: pd.DataFrame, anchor_date: pd.Timestamp | None = None
) -> go.Figure:
    """
    Build a Plotly figure using synthetic data when no Excel file is loaded.
    """
    if df_full.empty:
        return go.Figure()

    today = df_full["Date"].max().normalize()
    # Determine the lookback window for the fallback chart based on the
    # currently selected analysis timeframe.  When running under
    # Streamlit the ``ta_timeframe_days`` key will be present in
    # ``st.session_state``; otherwise it falls back to one year
    # (365 days).  This ensures the synthetic fallback chart aligns
    # with the timeframe used for real data.
    try:
        lookback_days = int(st.session_state.get("ta_timeframe_days", 365))  # type: ignore
    except Exception:
        lookback_days = 365
    start = today - pd.Timedelta(days=lookback_days)
    df = df_full[df_full["Date"].between(start, today)].reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Price"],
            mode="lines",
            name="S&P 500 Price",
            line=dict(color="#153D64", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df.get("MA_50", df["Price"]),
            mode="lines",
            name="50-day MA",
            line=dict(color="#008000", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df.get("MA_100", df["Price"]),
            mode="lines",
            name="100-day MA",
            line=dict(color="#FFA500", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df.get("MA_200", df["Price"]),
            mode="lines",
            name="200-day MA",
            line=dict(color="#FF0000", width=1.5),
        )
    )

    hi, lo = df["Price"].max(), df["Price"].min()
    span = hi - lo
    for lvl in [hi, hi - 0.236 * span, hi - 0.382 * span, hi - 0.5 * span, hi - 0.618 * span, lo]:
        fig.add_hline(
            y=lvl, line=dict(color="grey", dash="dash", width=1), opacity=0.6
        )

    if anchor_date is not None:
        subset = df_full[df_full["Date"].between(anchor_date, today)].copy()
        if not subset.empty:
            X = subset["Date"].map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
            y_vals = subset["Price"].to_numpy()
            model = LinearRegression().fit(X, y_vals)
            trend = model.predict(X)
            resid = y_vals - trend
            upper = trend + resid.max()
            lower = trend + resid.min()
            uptrend = model.coef_[0] > 0
            lineclr = "green" if uptrend else "red"
            fillclr = "rgba(0,150,0,0.25)" if uptrend else "rgba(200,0,0,0.25)"
            fig.add_trace(
                go.Scatter(
                    x=subset["Date"],
                    y=upper,
                    mode="lines",
                    line=dict(color=lineclr, dash="dash"),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=subset["Date"],
                    y=lower,
                    mode="lines",
                    line=dict(color=lineclr, dash="dash"),
                    fill="tonexty",
                    fillcolor=fillclr,
                    showlegend=False,
                )
            )

    fig.update_layout(
        margin=dict(l=30, r=30, t=60, b=40),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        xaxis_title=None,
        yaxis_title=None,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )
    return fig


###############################################################################
# Streamlit configuration
###############################################################################

st.set_page_config(page_title="IC Technical", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select page", [
        "Upload", 
        "YTD Update", 
        "Technical Analysis", 
        "Market Breadth",
        "Generate Presentation",
    ]
)


def show_upload_page():
    """Handle file uploads for Excel and PowerPoint templates."""
    st.sidebar.header("Upload files")
    excel_file = st.sidebar.file_uploader(
        "Upload consolidated Excel file", type=["xlsx", "xlsm", "xls"], key="excel_upload"
    )
    if excel_file is not None:
        st.session_state["excel_file"] = excel_file
    pptx_file = st.sidebar.file_uploader(
        "Upload PowerPoint template", type=["pptx", "pptm"], key="ppt_upload"
    )
    if pptx_file is not None:
        st.session_state["pptx_file"] = pptx_file
    st.sidebar.success("Files uploaded. Navigate to other pages to continue.")

    # Allow the user to choose between using the last recorded price (which may
    # be an intraday or current price) and the last close price (i.e. the
    # previous trading day's close).  The choice is stored in session state
    # and will affect how data is loaded and displayed elsewhere in the app.
    # Persist the selected price mode across pages.  Use the previously selected
    # value from session state (if any) to determine the default index.  If no
    # value has been stored yet, default to "Last Price".
    current_mode = st.session_state.get("price_mode", "Last Price")
    options = ["Last Price", "Last Close"]
    default_index = options.index(current_mode) if current_mode in options else 0
    price_mode = st.sidebar.radio(
        "Price mode",
        options=options,
        index=default_index,
        help=(
            "Select 'Last Close' to use the previous day's closing prices for all markets. "
            "Select 'Last Price' to use the most recent price in the data (which may be intraday)."
        ),
        key="price_mode_select",
    )
    st.session_state["price_mode"] = price_mode


def show_ytd_update_page():
    """Display YTD update charts and configuration."""
    st.sidebar.header("YTD Update")
    if "excel_file" not in st.session_state:
        st.sidebar.error("Please upload an Excel file on the Upload page first.")
        st.stop()

    # Lazy import heavy modules
    from ytd_perf.loader_update import load_data
    from utils import adjust_prices_for_mode
    from ytd_perf.equity_ytd import get_equity_ytd_series, create_equity_chart
    from ytd_perf.commodity_ytd import get_commodity_ytd_series, create_commodity_chart
    from ytd_perf.crypto_ytd import get_crypto_ytd_series, create_crypto_chart

    prices_df, params_df = load_data(st.session_state["excel_file"])
    # Determine whether to use the last price or the last close using
    # the centralised adjust_prices_for_mode helper.  This returns an
    # adjusted DataFrame and the effective date used for YTD calculations.
    used_date = None
    if not prices_df.empty:
        price_mode = st.session_state.get("price_mode", "Last Price")
        prices_df, used_date = adjust_prices_for_mode(prices_df, price_mode)
    # Display a caption indicating which date's prices are being used
    if used_date is not None:
        price_mode = st.session_state.get("price_mode", "Last Price")
        if price_mode == "Last Close":
            st.sidebar.caption(f"Prices as of {used_date.strftime('%d/%m/%Y')} close")
        else:
            st.sidebar.caption(f"Prices as of {used_date.strftime('%d/%m/%Y')}")

    # Equities configuration
    st.sidebar.subheader("Equities")
    eq_params = params_df[params_df["Asset Class"] == "Equity"]
    eq_name_to_ticker = {row["Name"]: row["Tickers"] for _, row in eq_params.iterrows()}
    eq_names_available = eq_params["Name"].tolist()
    default_eq = [
        name
        for name in [
            "Dax",
            "Ibov",
            "S&P 500",
            "Sensex",
            "SMI",
            "CSI 300",
            "Nikkei 225",
            "TASI",
            "Mexbol",
        ]
        if name in eq_names_available
    ]
    selected_eq_names = st.sidebar.multiselect(
        "Select equity indices",
        options=eq_names_available,
        default=st.session_state.get("selected_eq_names", default_eq),
        key="eq_indices",
    )
    st.session_state["selected_eq_names"] = selected_eq_names
    eq_tickers = [eq_name_to_ticker[name] for name in selected_eq_names]
    eq_subtitle = st.sidebar.text_input(
        "Equity subtitle", value=st.session_state.get("eq_subtitle", ""), key="eq_subtitle_input"
    )
    st.session_state["eq_subtitle"] = eq_subtitle

    # Commodities configuration
    st.sidebar.subheader("Commodities")
    co_params = params_df[params_df["Asset Class"] == "Commodity"]
    co_name_to_ticker = {row["Name"]: row["Tickers"] for _, row in co_params.iterrows()}
    co_names_available = co_params["Name"].tolist()
    default_co = [
        name
        for name in ["Gold", "Silver", "Oil (WTI)", "Platinum", "Copper", "Uranium","Palladium"]
        if name in co_names_available
    ]
    selected_co_names = st.sidebar.multiselect(
        "Select commodity indices",
        options=co_names_available,
        default=st.session_state.get("selected_co_names", default_co),
        key="co_indices",
    )
    st.session_state["selected_co_names"] = selected_co_names
    co_tickers = [co_name_to_ticker[name] for name in selected_co_names]
    co_subtitle = st.sidebar.text_input(
        "Commodity subtitle", value=st.session_state.get("co_subtitle", ""), key="co_subtitle_input"
    )
    st.session_state["co_subtitle"] = co_subtitle

    # Crypto configuration
    st.sidebar.subheader("Cryptocurrencies")
    cr_params = params_df[params_df["Asset Class"] == "Crypto"]
    cr_name_to_ticker = {row["Name"]: row["Tickers"] for _, row in cr_params.iterrows()}
    cr_names_available = cr_params["Name"].tolist()
    default_cr = [
        name
        for name in ["Ripple", "Bitcoin", "Binance", "Ethereum", "Solana","Ton"]
        if name in cr_names_available
    ]
    selected_cr_names = st.sidebar.multiselect(
        "Select crypto indices",
        options=cr_names_available,
        default=st.session_state.get("selected_cr_names", default_cr),
        key="cr_indices",
    )
    st.session_state["selected_cr_names"] = selected_cr_names
    cr_tickers = [cr_name_to_ticker[name] for name in selected_cr_names]
    cr_subtitle = st.sidebar.text_input(
        "Crypto subtitle", value=st.session_state.get("cr_subtitle", ""), key="cr_subtitle_input"
    )
    st.session_state["cr_subtitle"] = cr_subtitle

    # Persist selections
    st.session_state["selected_eq_tickers"] = eq_tickers
    st.session_state["selected_co_tickers"] = co_tickers
    st.session_state["selected_cr_tickers"] = cr_tickers

    st.header("YTD Performance Charts")
    with st.expander("Equity Chart", expanded=True):
        # Pass the selected price mode to compute YTD using either
        # intraday (Last Price) or previous close (Last Close).  This
        # ensures the chart reflects the user's choice in the sidebar.
        price_mode = st.session_state.get("price_mode", "Last Price")
        df_eq = get_equity_ytd_series(
            st.session_state["excel_file"], tickers=eq_tickers, price_mode=price_mode
        )
        st.pyplot(create_equity_chart(df_eq))
    with st.expander("Commodity Chart", expanded=False):
        price_mode = st.session_state.get("price_mode", "Last Price")
        df_co = get_commodity_ytd_series(
            st.session_state["excel_file"], tickers=co_tickers, price_mode=price_mode
        )
        st.pyplot(create_commodity_chart(df_co))
    with st.expander("Crypto Chart", expanded=False):
        # Pass price mode to ensure crypto YTD uses the same intraday/close setting
        price_mode = st.session_state.get("price_mode", "Last Price")
        df_cr = get_crypto_ytd_series(
            st.session_state["excel_file"], tickers=cr_tickers, price_mode=price_mode
        )
        st.pyplot(create_crypto_chart(df_cr))

    st.sidebar.success("Configure YTD charts, then go to 'Generate Presentation'.")


def show_technical_analysis_page():
    """Display the technical analysis interface for Equity (SPX) and other asset classes."""
    st.sidebar.header("Technical Analysis")
    asset_class = st.sidebar.radio(
        "Asset class", ["Equity", "Commodity", "Crypto"], index=0
    )

    # -------------------------------------------------------------------
    # Analysis timeframe selection
    # -------------------------------------------------------------------
    # Allow the user to choose the lookback horizon for all technical charts.
    # Providing both "6 months" and "1 year" lets users toggle between a
    # shorter six‑month window (≈180 days) and a full year (365 days).
    timeframe_options: dict[str, int] = {"6 months": 180, "1 year": 365}
    # Determine the default based on any previously stored selection; fall
    # back to six months if none is present.
    default_tf_label = st.session_state.get("ta_timeframe_label", "6 months")
    tf_labels = list(timeframe_options.keys())
    if default_tf_label not in tf_labels:
        default_tf_idx = 0
    else:
        default_tf_idx = tf_labels.index(default_tf_label)
    selected_tf_label = st.sidebar.selectbox(
        "Analysis timeframe",
        options=tf_labels,
        index=default_tf_idx,
        key="ta_timeframe_select",
    )
    # Persist the selection and derive the numeric days value
    st.session_state["ta_timeframe_label"] = selected_tf_label
    st.session_state["ta_timeframe_days"] = timeframe_options[selected_tf_label]

    # Propagate the chosen timeframe into technical analysis modules that
    # support configurable lookback windows.  The Mexbol and Palladium
    # modules define a ``PLOT_LOOKBACK_DAYS`` constant which can be
    # overridden at runtime.  We attempt to set this attribute here.
    try:
        import technical_analysis.equity.mexbol as _mex_module  # type: ignore
        _mex_module.PLOT_LOOKBACK_DAYS = st.session_state["ta_timeframe_days"]
    except Exception:
        pass

    try:
        import technical_analysis.equity.csi as _csi_module  # same package as your CSI code
        if hasattr(_csi_module, "PLOT_LOOKBACK_DAYS"):
            _csi_module.PLOT_LOOKBACK_DAYS = st.session_state["ta_timeframe_days"]
    except Exception:
        pass
    
    try:
        import technical_analysis.equity.dax as _dax_module  # same package as your DAX code
        if hasattr(_dax_module, "PLOT_LOOKBACK_DAYS"):
            _dax_module.PLOT_LOOKBACK_DAYS = st.session_state["ta_timeframe_days"]
    except Exception:
        pass

    try:
        import technical_analysis.equity.ibov as _ibov_module  # same package as your IBOV code
        if hasattr(_ibov_module, "PLOT_LOOKBACK_DAYS"):
            _ibov_module.PLOT_LOOKBACK_DAYS = st.session_state["ta_timeframe_days"]
    except Exception:
        pass

    try:
        import technical_analysis.equity.nikkei as _nikkei_module  # same package as your NIKKEI code
        if hasattr(_nikkei_module, "PLOT_LOOKBACK_DAYS"):
            _nikkei_module.PLOT_LOOKBACK_DAYS = st.session_state["ta_timeframe_days"]
    except Exception:
        pass

    try:
        import technical_analysis.equity.sensex as _sensex_module  # same package as your SENSEX code
        if hasattr(_sensex_module, "PLOT_LOOKBACK_DAYS"):
            _sensex_module.PLOT_LOOKBACK_DAYS = st.session_state["ta_timeframe_days"]
    except Exception:
        pass

    try:
        import technical_analysis.equity.smi as _smi_module  # same package as your SMI code
        if hasattr(_smi_module, "PLOT_LOOKBACK_DAYS"):
            _smi_module.PLOT_LOOKBACK_DAYS = st.session_state["ta_timeframe_days"]
    except Exception:
        pass

    try:
        import technical_analysis.equity.spx as _spx_module  # same package as your SPX code
        if hasattr(_spx_module, "PLOT_LOOKBACK_DAYS"):
            _spx_module.PLOT_LOOKBACK_DAYS = st.session_state["ta_timeframe_days"]
    except Exception:
        pass

    # Also attempt to update the lean palladium module (if used)
    try:
        import palladium as _palladium_alt  # type: ignore
        _palladium_alt.PLOT_LOOKBACK_DAYS = st.session_state["ta_timeframe_days"]
    except Exception:
        pass

    # Provide a clear channel button to reset the regression channel for both indices
    if st.sidebar.button("Clear channel", key="ta_clear_global"):
        # Remove stored anchors for all indices if present
        for key in [
            "spx_anchor",
            "csi_anchor",
            "nikkei_anchor",
            "tasi_anchor",
            "sensex_anchor",
            "dax_anchor",
            "smi_anchor",
            "ibov_anchor",
            # also clear anchors for commodity and crypto assets
            "gold_anchor",
            "silver_anchor",
            "platinum_anchor",
            "oil_anchor",
            "copper_anchor",
            "bitcoin_anchor",
            "ethereum_anchor",
            "ripple_anchor",
            "solana_anchor",
            "binance_anchor",
        ]:
            if key in st.session_state:
                st.session_state.pop(key)
        st.rerun()

    excel_available = "excel_file" in st.session_state

    if asset_class == "Equity":
        # Allow the user to select which equity index they wish to analyse.  We
        # provide two options: S&P 500 and CSI 300.  The selection is stored
        # in session state to persist across reruns.
        # Provide index options.  Add Nikkei 225 alongside SPX and CSI.
        # Include SMI (Swiss Market Index) alongside existing indices
        # Include IBOV (Brazil Bovespa) alongside existing indices
        # Add Mexbol to the list of available equity indices
        index_options = ["S&P 500", "CSI 300", "Nikkei 225", "TASI", "Sensex", "Dax", "SMI", "Ibov", "Mexbol"]
        default_index = st.session_state.get("ta_equity_index", "S&P 500")
        selected_index = st.sidebar.selectbox(
            "Select equity index for technical analysis",
            options=index_options,
            index=index_options.index(default_index) if default_index in index_options else 0,
            key="ta_equity_index_select",
        )
        # Persist the selected index
        st.session_state["ta_equity_index"] = selected_index

        # Determine ticker and names based on the selected index
        # Determine ticker and label keys based on the selected index
        if selected_index == "S&P 500":
            ticker = "SPX Index"
            ticker_key = "spx"
            chart_title = "S&P 500 Technical Chart"
        elif selected_index == "CSI 300":
            ticker = "SHSZ300 Index"
            ticker_key = "csi"
            chart_title = "CSI 300 Technical Chart"
        elif selected_index == "Nikkei 225":
            ticker = "NKY Index"
            ticker_key = "nikkei"
            chart_title = "Nikkei 225 Technical Chart"
        elif selected_index == "TASI":
            ticker = "SASEIDX Index"
            ticker_key = "tasi"
            chart_title = "TASI Technical Chart"
        elif selected_index == "Sensex":
            ticker = "SENSEX Index"
            ticker_key = "sensex"
            chart_title = "Sensex Technical Chart"
        elif selected_index == "Dax":
            ticker = "DAX Index"
            ticker_key = "dax"
            chart_title = "DAX Technical Chart"
        elif selected_index == "SMI":
            ticker = "SMI Index"
            ticker_key = "smi"
            chart_title = "SMI Technical Chart"
        elif selected_index == "Ibov":
            ticker = "IBOV Index"
            ticker_key = "ibov"
            chart_title = "Ibov Technical Chart"
        elif selected_index == "Mexbol":
            # Mexbol index configuration
            ticker = "MEXBOL Index"
            ticker_key = "mexbol"
            chart_title = "Mexbol Technical Chart"
        else:
            # Default fallback (should not occur)
            ticker = "SPX Index"
            ticker_key = "spx"
            chart_title = "S&P 500 Technical Chart"

        # Load data for interactive chart (real or synthetic)
        if excel_available:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                tmp.write(st.session_state["excel_file"].getbuffer())
                tmp.flush()
                temp_path = Path(tmp.name)
            df_prices = pd.read_excel(temp_path, sheet_name="data_prices")
            df_prices = df_prices.drop(index=0)
            df_prices = df_prices[df_prices[df_prices.columns[0]] != "DATES"]
            df_prices["Date"] = pd.to_datetime(
                df_prices[df_prices.columns[0]], errors="coerce"
            )
            df_prices["Price"] = pd.to_numeric(df_prices[ticker], errors="coerce")
            df_prices = df_prices.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(
                drop=True
            )
            # Adjust the prices according to the selected price mode using the helper.
            price_mode = st.session_state.get("price_mode", "Last Price")
            df_prices, used_date = adjust_prices_for_mode(df_prices, price_mode)
            df_full = df_prices.copy()
            # Store the used date for later caption display (per index)
            st.session_state[f"{ticker_key}_used_date"] = used_date
        else:
            # Use synthetic series only for SPX; for CSI default to SPX synthetic as fallback
            df_prices = _create_synthetic_spx_series()
            df_full = df_prices.copy()

        min_date = df_prices["Date"].min().date()
        max_date = df_prices["Date"].max().date()

        # Chart with controls in expander
        with st.expander(chart_title, expanded=True):
            # Display a caption indicating which date's prices are being used
            used_date = st.session_state.get(f"{ticker_key}_used_date")
            price_mode = st.session_state.get("price_mode", "Last Price")
            if used_date is not None:
                if price_mode == "Last Close":
                    st.caption(f"Prices as of {used_date.strftime('%d/%m/%Y')} close")
                else:
                    st.caption(f"Prices as of {used_date.strftime('%d/%m/%Y')}")
            # -------------------------------------------------------------------
            # Display technical and momentum scores first
            # -------------------------------------------------------------------
            st.subheader("Technical and momentum scores")
            tech_score = None
            mom_score = None
            if excel_available:
                try:
                    # Use the temporary file path for reading scores so that
                    # pandas can access the Excel multiple times reliably.
                    if selected_index == "S&P 500":
                        tech_score = _get_spx_technical_score(temp_path)
                    elif selected_index == "CSI 300":
                        tech_score = _get_csi_technical_score(temp_path)
                    elif selected_index == "Nikkei 225":
                        tech_score = _get_nikkei_technical_score(temp_path)
                    elif selected_index == "TASI":
                        tech_score = _get_tasi_technical_score(temp_path)
                    elif selected_index == "Sensex":
                        tech_score = _get_sensex_technical_score(temp_path)
                    elif selected_index == "Dax":
                        tech_score = _get_dax_technical_score(temp_path)
                    elif selected_index == "SMI":
                        tech_score = _get_smi_technical_score(temp_path)
                    elif selected_index == "Ibov":
                        tech_score = _get_ibov_technical_score(temp_path)
                    elif selected_index == "Mexbol":
                        tech_score = _get_mexbol_technical_score(temp_path)
                    else:
                        tech_score = None
                except Exception:
                    tech_score = None
                try:
                    if selected_index == "S&P 500":
                        mom_score = _get_spx_momentum_score(temp_path)
                    elif selected_index == "CSI 300":
                        mom_score = _get_csi_momentum_score(temp_path)
                    elif selected_index == "Nikkei 225":
                        mom_score = _get_nikkei_momentum_score(temp_path)
                    elif selected_index == "TASI":
                        mom_score = _get_tasi_momentum_score(temp_path)
                    elif selected_index == "Sensex":
                        mom_score = _get_sensex_momentum_score(temp_path)
                    elif selected_index == "Dax":
                        mom_score = _get_dax_momentum_score(temp_path)
                    elif selected_index == "SMI":
                        mom_score = _get_smi_momentum_score(temp_path)
                    elif selected_index == "Ibov":
                        mom_score = _get_ibov_momentum_score(temp_path)
                    elif selected_index == "Mexbol":
                        mom_score = _get_mexbol_momentum_score(temp_path)
                    else:
                        mom_score = None
                except Exception:
                    mom_score = None

            # Prepare DMAS and table if both scores are available
            dmas = None
            if tech_score is not None and mom_score is not None:
                dmas = round((float(tech_score) + float(mom_score)) / 2.0, 1)
                df_scores = pd.DataFrame(
                    {
                        "Technical Score": [tech_score],
                        "Momentum Score": [mom_score],
                        "Average (DMAS)": [dmas],
                    }
                )
                st.table(df_scores)
                # Provide an input for last week's average DMAS to be used in the gauge
                # Only applicable to SPX and CSI indices.  The value is stored in
                # session state and used when generating the presentation.
                if selected_index == "S&P 500":
                    # Provide a number input with sensible defaults and bounds
                    spx_last_week_input = st.number_input(
                        "Last week's average (DMAS)",
                        min_value=0.0,
                        max_value=100.0,
                        value=st.session_state.get("spx_last_week_avg", 50.0),
                        key="spx_last_week_avg_input",
                    )
                    st.session_state["spx_last_week_avg"] = spx_last_week_input
                elif selected_index == "CSI 300":
                    csi_last_week_input = st.number_input(
                        "Last week's average (DMAS)",
                        min_value=0.0,
                        max_value=100.0,
                        value=st.session_state.get("csi_last_week_avg", 50.0),
                        key="csi_last_week_avg_input",
                    )
                    st.session_state["csi_last_week_avg"] = csi_last_week_input
                elif selected_index == "Nikkei 225":
                    nikkei_last_week_input = st.number_input(
                        "Last week's average (DMAS)",
                        min_value=0.0,
                        max_value=100.0,
                        value=st.session_state.get("nikkei_last_week_avg", 50.0),
                        key="nikkei_last_week_avg_input",
                    )
                    st.session_state["nikkei_last_week_avg"] = nikkei_last_week_input
                elif selected_index == "TASI":
                    tasi_last_week_input = st.number_input(
                        "Last week's average (DMAS)",
                        min_value=0.0,
                        max_value=100.0,
                        value=st.session_state.get("tasi_last_week_avg", 50.0),
                        key="tasi_last_week_avg_input",
                    )
                    st.session_state["tasi_last_week_avg"] = tasi_last_week_input
                elif selected_index == "Sensex":
                    sensex_last_week_input = st.number_input(
                        "Last week's average (DMAS)",
                        min_value=0.0,
                        max_value=100.0,
                        value=st.session_state.get("sensex_last_week_avg", 50.0),
                        key="sensex_last_week_avg_input",
                    )
                    st.session_state["sensex_last_week_avg"] = sensex_last_week_input
                elif selected_index == "Dax":
                    dax_last_week_input = st.number_input(
                        "Last week's average (DMAS)",
                        min_value=0.0,
                        max_value=100.0,
                        value=st.session_state.get("dax_last_week_avg", 50.0),
                        key="dax_last_week_avg_input",
                    )
                    st.session_state["dax_last_week_avg"] = dax_last_week_input
                elif selected_index == "SMI":
                    smi_last_week_input = st.number_input(
                        "Last week's average (DMAS)",
                        min_value=0.0,
                        max_value=100.0,
                        value=st.session_state.get("smi_last_week_avg", 50.0),
                        key="smi_last_week_avg_input",
                    )
                    st.session_state["smi_last_week_avg"] = smi_last_week_input
                elif selected_index == "Ibov":
                    ibov_last_week_input = st.number_input(
                        "Last week's average (DMAS)",
                        min_value=0.0,
                        max_value=100.0,
                        value=st.session_state.get("ibov_last_week_avg", 50.0),
                        key="ibov_last_week_avg_input",
                    )
                    st.session_state["ibov_last_week_avg"] = ibov_last_week_input
                elif selected_index == "Mexbol":
                    # Capture last week's average DMAS for the Mexbol index
                    mexbol_last_week_input = st.number_input(
                        "Last week's average (DMAS)",
                        min_value=0.0,
                        max_value=100.0,
                        value=st.session_state.get("mexbol_last_week_avg", 50.0),
                        key="mexbol_last_week_avg_input",
                    )
                    st.session_state["mexbol_last_week_avg"] = mexbol_last_week_input
            else:
                st.info(
                    "Technical or momentum score not available in the uploaded Excel. "
                    "Please ensure sheets 'data_technical_score' and 'data_trend_rating' exist."
                )
            # -------------------------------------------------------------------
            # Show recent trading range (high/low) beneath the score table
            # -------------------------------------------------------------------
            try:
                # Compute trading range for the last 90 days based on implied volatility or realised volatility.
                current_price = df_full["Price"].iloc[-1] if not df_full.empty else None
                if current_price is not None and not np.isnan(current_price):
                    # Attempt to use implied volatility for the S&P 500 (VIX)
                    use_implied = False
                    vol_val = None
                    if selected_index == "S&P 500":
                        try:
                            df_vol = pd.read_excel(temp_path, sheet_name="data_prices")
                            df_vol = df_vol.drop(index=0)
                            df_vol = df_vol[df_vol[df_vol.columns[0]] != "DATES"]
                            df_vol["Date"] = pd.to_datetime(df_vol[df_vol.columns[0]], errors="coerce")
                            if "VIX Index" in df_vol.columns:
                                df_vol["Price"] = pd.to_numeric(df_vol["VIX Index"], errors="coerce")
                                df_vol = df_vol.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)[["Date", "Price"]]
                                pm = st.session_state.get("price_mode", "Last Price")
                                if adjust_prices_for_mode is not None:
                                    try:
                                        df_vol, _ = adjust_prices_for_mode(df_vol, pm)
                                    except Exception:
                                        pass
                                if not df_vol.empty:
                                    vol_val = float(df_vol["Price"].iloc[-1])
                                    use_implied = True
                        except Exception:
                            use_implied = False
                    elif selected_index == "SMI":
                        # Attempt to use implied volatility for SMI (VSMI1M)
                        try:
                            df_vol = pd.read_excel(temp_path, sheet_name="data_prices")
                            df_vol = df_vol.drop(index=0)
                            df_vol = df_vol[df_vol[df_vol.columns[0]] != "DATES"]
                            df_vol["Date"] = pd.to_datetime(df_vol[df_vol.columns[0]], errors="coerce")
                            if "VSMI1M Index" in df_vol.columns:
                                df_vol["Price"] = pd.to_numeric(df_vol["VSMI1M Index"], errors="coerce")
                                df_vol = df_vol.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)[["Date", "Price"]]
                                pm = st.session_state.get("price_mode", "Last Price")
                                if adjust_prices_for_mode is not None:
                                    try:
                                        df_vol, _ = adjust_prices_for_mode(df_vol, pm)
                                    except Exception:
                                        pass
                                if not df_vol.empty:
                                    vol_val = float(df_vol["Price"].iloc[-1])
                                    use_implied = True
                        except Exception:
                            use_implied = False
                    # Compute expected move from implied volatility if available
                    if use_implied and vol_val is not None:
                        expected_move = (current_price * (vol_val / 100.0)) / np.sqrt(52.0)
                        lower_bound = current_price - expected_move
                        upper_bound = current_price + expected_move
                        # Enforce minimum ±1 % band around current price
                        min_span = 0.02 * current_price
                        if (upper_bound - lower_bound) < min_span:
                            half = min_span / 2.0
                            lower_bound = current_price - half
                            upper_bound = current_price + half
                    else:
                        # Use realised-volatility based bounds based on the selected index
                        if selected_index == "S&P 500":
                            upper_bound, lower_bound = _compute_range_bounds_spx(df_full, lookback_days=90)
                        elif selected_index == "CSI 300":
                            upper_bound, lower_bound = _compute_range_bounds_csi(df_full, lookback_days=90)
                        elif selected_index == "Nikkei 225":
                            upper_bound, lower_bound = _compute_range_bounds_nikkei(df_full, lookback_days=90)
                        elif selected_index == "TASI":
                            upper_bound, lower_bound = _compute_range_bounds_tasi(df_full, lookback_days=90)
                        elif selected_index == "Sensex":
                            upper_bound, lower_bound = _compute_range_bounds_sensex(df_full, lookback_days=90)
                        elif selected_index == "Dax":
                            upper_bound, lower_bound = _compute_range_bounds_dax(df_full, lookback_days=90)
                        elif selected_index == "SMI":
                            upper_bound, lower_bound = _compute_range_bounds_smi(df_full, lookback_days=90)
                        elif selected_index == "Ibov":
                            upper_bound, lower_bound = _compute_range_bounds_ibov(df_full, lookback_days=90)
                        elif selected_index == "Mexbol":
                            upper_bound, lower_bound = _compute_range_bounds_mexbol(df_full, lookback_days=90)
                        else:
                            upper_bound, lower_bound = _compute_range_bounds_spx(df_full, lookback_days=90)
                    low_pct = (lower_bound - current_price) / current_price * 100.0
                    high_pct = (upper_bound - current_price) / current_price * 100.0
                    st.write(
                        f"Trading range (90d): Low {lower_bound:,.0f} ({low_pct:+.1f}%), "
                        f"High {upper_bound:,.0f} ({high_pct:+.1f}%)"
                    )
                else:
                    # No current price: use realised-volatility-based bounds
                    if selected_index == "S&P 500":
                        upper_bound, lower_bound = _compute_range_bounds_spx(df_full, lookback_days=90)
                    elif selected_index == "CSI 300":
                        upper_bound, lower_bound = _compute_range_bounds_csi(df_full, lookback_days=90)
                    elif selected_index == "Nikkei 225":
                        upper_bound, lower_bound = _compute_range_bounds_nikkei(df_full, lookback_days=90)
                    elif selected_index == "TASI":
                        upper_bound, lower_bound = _compute_range_bounds_tasi(df_full, lookback_days=90)
                    elif selected_index == "Sensex":
                        upper_bound, lower_bound = _compute_range_bounds_sensex(df_full, lookback_days=90)
                    elif selected_index == "Dax":
                        upper_bound, lower_bound = _compute_range_bounds_dax(df_full, lookback_days=90)
                    elif selected_index == "Ibov":
                        upper_bound, lower_bound = _compute_range_bounds_ibov(df_full, lookback_days=90)
                    elif selected_index == "Mexbol":
                        upper_bound, lower_bound = _compute_range_bounds_mexbol(df_full, lookback_days=90)
                    elif selected_index == "SMI":
                        upper_bound, lower_bound = _compute_range_bounds_smi(df_full, lookback_days=90)
                    elif selected_index == "Ibov":
                        upper_bound, lower_bound = _compute_range_bounds_ibov(df_full, lookback_days=90)
                    elif selected_index == "Mexbol":
                        upper_bound, lower_bound = _compute_range_bounds_mexbol(df_full, lookback_days=90)
                    else:
                        upper_bound, lower_bound = _compute_range_bounds_spx(df_full, lookback_days=90)
                    st.write(
                        f"Trading range (90d): Low {lower_bound:,.0f} – High {upper_bound:,.0f}"
                    )
            except Exception:
                pass

            # -------------------------------------------------------------------
            # Regression channel controls second
            # -------------------------------------------------------------------
            enable_channel = st.checkbox(
                "Enable regression channel",
                value=bool(st.session_state.get(f"{ticker_key}_anchor")),
                key=f"{ticker_key}_enable_channel",
            )

            anchor_ts = None
            if enable_channel:
                # When the regression channel is enabled, default the anchor to
                # the start of the selected analysis timeframe unless a
                # previous anchor has been stored in the session.  This
                # replaces the fixed 180‑day default with the user‑chosen
                # timeframe (e.g. 180 or 365 days).
                default_anchor = st.session_state.get(
                    f"{ticker_key}_anchor",
                    (max_date - pd.Timedelta(days=st.session_state.get("ta_timeframe_days", 180))),
                )
                anchor_input = st.date_input(
                    "Select anchor date",
                    value=default_anchor,
                    min_value=min_date,
                    max_value=max_date,
                    key=f"{ticker_key}_anchor_date_input",
                )
                anchor_ts = pd.to_datetime(anchor_input)
                st.session_state[f"{ticker_key}_anchor"] = anchor_ts
            else:
                if f"{ticker_key}_anchor" in st.session_state:
                    st.session_state.pop(f"{ticker_key}_anchor")
                anchor_ts = None

            # -------------------------------------------------------------------
            # Assessment selection third
            # -------------------------------------------------------------------
            if tech_score is not None and mom_score is not None and dmas is not None:
                options = [
                    "Strongly Bearish",
                    "Bearish",
                    "Slightly Bearish",
                    "Neutral",
                    "Slightly Bullish",
                    "Bullish",
                    "Strongly Bullish",
                ]
                def _default_index_from_dmas(val: float) -> int:
                    if val >= 80:
                        return options.index("Strongly Bullish")
                    elif val >= 70:
                        return options.index("Bullish")
                    elif val >= 60:
                        return options.index("Slightly Bullish")
                    elif val >= 40:
                        return options.index("Neutral")
                    elif val >= 30:
                        return options.index("Slightly Bearish")
                    elif val >= 20:
                        return options.index("Bearish")
                    else:
                        return options.index("Strongly Bearish")

                default_idx = _default_index_from_dmas(dmas)
                user_view = st.selectbox(
                    "Select your assessment",
                    options,
                    index=default_idx,
                    key=f"{ticker_key}_view_select",
                )
                st.session_state[f"{ticker_key}_selected_view"] = user_view
                st.caption(
                    "Your selection will override the automatically computed view in the presentation."
                )

            # -------------------------------------------------------------------
            # Subtitle input fourth
            # -------------------------------------------------------------------
            subtitle_value = st.text_input(
                f"{ticker_key.upper()} subtitle" if selected_index == "S&P 500" else f"{ticker_key.upper()} subtitle",
                value=st.session_state.get(f"{ticker_key}_subtitle", ""),
                key=f"{ticker_key}_subtitle_input",
            )
            st.session_state[f"{ticker_key}_subtitle"] = subtitle_value

            # -------------------------------------------------------------------
            # Finally, build and show the interactive chart
            # -------------------------------------------------------------------
            if excel_available:
                pmode = st.session_state.get("price_mode", "Last Price")
                if selected_index == "S&P 500":
                    fig = make_spx_figure(temp_path, anchor_date=anchor_ts, price_mode=pmode)
                elif selected_index == "CSI 300":
                    fig = make_csi_figure(temp_path, anchor_date=anchor_ts, price_mode=pmode)
                elif selected_index == "Nikkei 225":
                    fig = make_nikkei_figure(temp_path, anchor_date=anchor_ts, price_mode=pmode)
                elif selected_index == "TASI":
                    fig = make_tasi_figure(temp_path, anchor_date=anchor_ts, price_mode=pmode)
                elif selected_index == "Sensex":
                    fig = make_sensex_figure(temp_path, anchor_date=anchor_ts, price_mode=pmode)
                elif selected_index == "Dax":
                    fig = make_dax_figure(temp_path, anchor_date=anchor_ts, price_mode=pmode)
                elif selected_index == "SMI":
                    fig = make_smi_figure(temp_path, anchor_date=anchor_ts, price_mode=pmode)
                elif selected_index == "Ibov":
                    fig = make_ibov_figure(temp_path, anchor_date=anchor_ts, price_mode=pmode)
                elif selected_index == "Mexbol":
                    fig = make_mexbol_figure(temp_path, anchor_date=anchor_ts, price_mode=pmode)
                else:
                    # default fallback: use SPX figure
                    fig = make_spx_figure(temp_path, anchor_date=anchor_ts, price_mode=pmode)
            else:
                df_ma = _add_moving_averages(df_full)
                fig = _build_fallback_figure(df_ma, anchor_date=anchor_ts)

            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Use the controls above to enable and configure the regression channel. "
                "Green shading indicates an uptrend; red shading indicates a downtrend."
            )

    elif asset_class == "Commodity":
        # Delegate to the commodity technical analysis handler
        show_commodity_technical_analysis()
    elif asset_class == "Crypto":
        # Delegate to the crypto technical analysis handler
        show_crypto_technical_analysis()
    else:
        # Fallback for unsupported asset classes
        with st.expander(f"{asset_class} technical charts", expanded=False):
            st.info(f"{asset_class} technical analysis not implemented yet.")


def show_commodity_technical_analysis() -> None:
    """Render the technical analysis interface for commodity assets such as Gold.

    This function mirrors the equity technical analysis interface but is
    customised for commodity tickers.  Currently only Gold is supported.
    It handles data loading, score retrieval, DMAS computation, trading
    range estimation using an implied volatility index (XAUUSDV1M),
    regression channel controls, assessment selection, subtitle input and
    interactive chart rendering.  State is persisted in
    ``st.session_state`` to allow regeneration of the chart with the
    regression channel anchored at a user‑selected date.
    """
    # Identify whether an Excel file has been uploaded
    excel_available = "excel_file" in st.session_state

    # Commodity selection (Gold and Silver)
    # Include Gold, Silver, Platinum, Oil and Copper in the commodity options
    # Include Palladium in the list of supported commodities
    index_options = ["Gold", "Silver", "Platinum", "Palladium", "Oil", "Copper"]
    default_index = st.session_state.get("ta_commodity_index", "Gold")
    selected_index = st.sidebar.selectbox(
        "Select commodity for technical analysis",
        options=index_options,
        index=index_options.index(default_index) if default_index in index_options else 0,
        key="ta_commodity_index_select",
    )
    # Persist the selected commodity
    st.session_state["ta_commodity_index"] = selected_index

    # Determine ticker and keys based on selection
    if selected_index == "Gold":
        ticker = "GCA Comdty"
        ticker_key = "gold"
        chart_title = "Gold Technical Chart"
    elif selected_index == "Silver":
        ticker = "SIA Comdty"
        ticker_key = "silver"
        chart_title = "Silver Technical Chart"
    elif selected_index == "Platinum":
        ticker = "XPT Comdty"
        ticker_key = "platinum"
        chart_title = "Platinum Technical Chart"
    elif selected_index == "Palladium":
        ticker = "XPD Curncy"
        ticker_key = "palladium"
        chart_title = "Palladium Technical Chart"
    elif selected_index == "Oil":
        ticker = "CL1 Comdty"
        ticker_key = "oil"
        chart_title = "Oil Technical Chart"
    elif selected_index == "Copper":
        ticker = "LP1 Comdty"
        ticker_key = "copper"
        chart_title = "Copper Technical Chart"
    else:
        # Default back to Gold if an unknown commodity is selected
        ticker = "GCA Comdty"
        ticker_key = "gold"
        chart_title = f"{selected_index} Technical Chart"

    # Load price data (either from Excel or fallback synthetic)
    if excel_available:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(st.session_state["excel_file"].getbuffer())
            tmp.flush()
            temp_path = Path(tmp.name)
        # Read the data_prices sheet and tidy
        df_prices = pd.read_excel(temp_path, sheet_name="data_prices")
        df_prices = df_prices.drop(index=0)
        df_prices = df_prices[df_prices[df_prices.columns[0]] != "DATES"]
        df_prices["Date"] = pd.to_datetime(df_prices[df_prices.columns[0]], errors="coerce")
        df_prices["Price"] = pd.to_numeric(df_prices[ticker], errors="coerce")
        df_prices = df_prices.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)
        # Adjust for price mode using the helper if available
        price_mode = st.session_state.get("price_mode", "Last Price")
        if adjust_prices_for_mode is not None and price_mode:
            try:
                df_prices, used_date = adjust_prices_for_mode(df_prices, price_mode)
            except Exception:
                used_date = None
        else:
            used_date = None
        df_full = df_prices.copy()
        st.session_state[f"{ticker_key}_used_date"] = used_date
    else:
        # Use a synthetic SPX series as a fallback when no Excel is provided
        df_prices = _create_synthetic_spx_series()
        df_full = df_prices.copy()
        used_date = None
    # Determine min and max dates for regression channel controls
    min_date = df_prices["Date"].min().date()
    max_date = df_prices["Date"].max().date()

    # Chart and controls
    with st.expander(chart_title, expanded=True):
        # Caption for date used
        used_date = st.session_state.get(f"{ticker_key}_used_date")
        price_mode = st.session_state.get("price_mode", "Last Price")
        if used_date is not None:
            if price_mode == "Last Close":
                st.caption(f"Prices as of {used_date.strftime('%d/%m/%Y')} close")
            else:
                st.caption(f"Prices as of {used_date.strftime('%d/%m/%Y')}")
        # -----------------------------------------------------------------
        # Technical and momentum scores
        # -----------------------------------------------------------------
        st.subheader("Technical and momentum scores")
        tech_score: Optional[float] = None
        mom_score: Optional[float] = None
        if excel_available:
            try:
                if selected_index == "Gold":
                    tech_score = _get_gold_technical_score(temp_path)
                elif selected_index == "Silver":
                    tech_score = _get_silver_technical_score(temp_path)
                elif selected_index == "Platinum":
                    tech_score = _get_platinum_technical_score(temp_path)
                elif selected_index == "Palladium":
                    tech_score = _get_palladium_technical_score(temp_path)
                elif selected_index == "Oil":
                    tech_score = _get_oil_technical_score(temp_path)
                elif selected_index == "Copper":
                    tech_score = _get_copper_technical_score(temp_path)
            except Exception:
                tech_score = None
            try:
                if selected_index == "Gold":
                    mom_score = _get_gold_momentum_score(temp_path)
                elif selected_index == "Silver":
                    mom_score = _get_silver_momentum_score(temp_path)
                elif selected_index == "Platinum":
                    mom_score = _get_platinum_momentum_score(temp_path)
                elif selected_index == "Palladium":
                    mom_score = _get_palladium_momentum_score(temp_path)
                elif selected_index == "Oil":
                    mom_score = _get_oil_momentum_score(temp_path)
                elif selected_index == "Copper":
                    mom_score = _get_copper_momentum_score(temp_path)
            except Exception:
                mom_score = None
        # Compute DMAS if scores are available
        dmas: Optional[float] = None
        if tech_score is not None and mom_score is not None:
            dmas = round((float(tech_score) + float(mom_score)) / 2.0, 1)
            df_scores = pd.DataFrame(
                {
                    "Technical Score": [tech_score],
                    "Momentum Score": [mom_score],
                    "Average (DMAS)": [dmas],
                }
            )
            st.table(df_scores)
            # Allow user to input last week's DMAS for the gauge. Use a key based on the commodity
            gauge_key = f"{ticker_key}_last_week_avg"
            gauge_input_key = f"{ticker_key}_last_week_avg_input"
            last_week_default = st.session_state.get(gauge_key, 50.0)
            last_week_input = st.number_input(
                "Last week's average (DMAS)",
                min_value=0.0,
                max_value=100.0,
                value=last_week_default,
                key=gauge_input_key,
            )
            st.session_state[gauge_key] = last_week_input
        else:
            st.info(
                "Technical or momentum score not available in the uploaded Excel. "
                "Please ensure sheets 'data_technical_score' and 'data_trend_rating' exist."
            )
        # -----------------------------------------------------------------
        # Trading range (90d) estimation
        # -----------------------------------------------------------------
        try:
            current_price = df_full["Price"].iloc[-1] if not df_full.empty else None
            if current_price is not None and not np.isnan(current_price):
                use_implied = False
                vol_val: Optional[float] = None
                # Attempt to use implied volatility via XAUUSDV1M BGN Curncy (Gold) or XAGUSDV1M BGN Curncy (Silver)
                vol_col_name = None
                if selected_index == "Gold":
                    vol_col_name = "XAUUSDV1M BGN Curncy"
                elif selected_index == "Silver":
                    vol_col_name = "XAGUSDV1M BGN Curncy"
                elif selected_index == "Platinum":
                    vol_col_name = "XPTUSDV1M BGN Curncy"
                elif selected_index == "Palladium":
                    vol_col_name = "XPDUSDV1M BGN Curncy"
                elif selected_index == "Oil":
                    # Oil implied volatility index column
                    vol_col_name = "WTI US 1M 50D VOL BVOL Equity"
                else:
                    # Copper implied volatility index column
                    vol_col_name = "LPR1 Index"
                if vol_col_name is not None:
                    try:
                        df_vol = pd.read_excel(temp_path, sheet_name="data_prices")
                        df_vol = df_vol.drop(index=0)
                        df_vol = df_vol[df_vol[df_vol.columns[0]] != "DATES"]
                        df_vol["Date"] = pd.to_datetime(df_vol[df_vol.columns[0]], errors="coerce")
                        if vol_col_name in df_vol.columns:
                            df_vol["Price"] = pd.to_numeric(df_vol[vol_col_name], errors="coerce")
                            df_vol = df_vol.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)[["Date", "Price"]]
                            pm = st.session_state.get("price_mode", "Last Price")
                            if adjust_prices_for_mode is not None:
                                try:
                                    df_vol, _ = adjust_prices_for_mode(df_vol, pm)
                                except Exception:
                                    pass
                            if not df_vol.empty:
                                vol_val = float(df_vol["Price"].iloc[-1])
                                use_implied = True
                    except Exception:
                        use_implied = False
                # If implied vol available, compute expected move
                if use_implied and vol_val is not None:
                    expected_move = (current_price * (vol_val / 100.0)) / np.sqrt(52.0)
                    lower_bound = current_price - expected_move
                    upper_bound = current_price + expected_move
                    min_span = 0.02 * current_price
                    if (upper_bound - lower_bound) < min_span:
                        half = min_span / 2.0
                        lower_bound = current_price - half
                        upper_bound = current_price + half
                else:
                    # Fallback to realised volatility when implied vol is unavailable
                    if selected_index == "Gold":
                        upper_bound, lower_bound = _compute_range_bounds_gold(df_full, lookback_days=90)
                    elif selected_index == "Silver":
                        upper_bound, lower_bound = _compute_range_bounds_silver(df_full, lookback_days=90)
                    elif selected_index == "Platinum":
                        upper_bound, lower_bound = _compute_range_bounds_platinum(df_full, lookback_days=90)
                    elif selected_index == "Palladium":
                        upper_bound, lower_bound = _compute_range_bounds_palladium(df_full, lookback_days=90)
                    elif selected_index == "Oil":
                        upper_bound, lower_bound = _compute_range_bounds_oil(df_full, lookback_days=90)
                    else:
                        # Copper (or any other commodity default)
                        upper_bound, lower_bound = _compute_range_bounds_copper(df_full, lookback_days=90)
                low_pct = (lower_bound - current_price) / current_price * 100.0
                high_pct = (upper_bound - current_price) / current_price * 100.0
                st.write(
                    f"Trading range (90d): Low {lower_bound:,.0f} ({low_pct:+.1f}%), "
                    f"High {upper_bound:,.0f} ({high_pct:+.1f}%)"
                )
            else:
                if selected_index == "Gold":
                    upper_bound, lower_bound = _compute_range_bounds_gold(df_full, lookback_days=90)
                elif selected_index == "Silver":
                    upper_bound, lower_bound = _compute_range_bounds_silver(df_full, lookback_days=90)
                elif selected_index == "Platinum":
                    upper_bound, lower_bound = _compute_range_bounds_platinum(df_full, lookback_days=90)
                elif selected_index == "Palladium":
                    upper_bound, lower_bound = _compute_range_bounds_palladium(df_full, lookback_days=90)
                elif selected_index == "Oil":
                    upper_bound, lower_bound = _compute_range_bounds_oil(df_full, lookback_days=90)
                else:
                    # Copper (or other commodity)
                    upper_bound, lower_bound = _compute_range_bounds_copper(df_full, lookback_days=90)
                st.write(
                    f"Trading range (90d): Low {lower_bound:,.0f} – High {upper_bound:,.0f}"
                )
        except Exception:
            pass
        # -----------------------------------------------------------------
        # Regression channel controls
        # -----------------------------------------------------------------
        enable_channel = st.checkbox(
            "Enable regression channel",
            value=bool(st.session_state.get(f"{ticker_key}_anchor")),
            key=f"{ticker_key}_enable_channel",
        )
        anchor_ts: Optional[pd.Timestamp] = None
        if enable_channel:
            # Default the anchor to the beginning of the selected
            # timeframe when no previous anchor is stored.  Uses
            # ``ta_timeframe_days`` instead of a fixed 180‑day window.
            default_anchor = st.session_state.get(
                f"{ticker_key}_anchor",
                (max_date - pd.Timedelta(days=st.session_state.get("ta_timeframe_days", 180))),
            )
            anchor_input = st.date_input(
                "Select anchor date",
                value=default_anchor,
                min_value=min_date,
                max_value=max_date,
                key=f"{ticker_key}_anchor_date_input",
            )
            anchor_ts = pd.to_datetime(anchor_input)
            st.session_state[f"{ticker_key}_anchor"] = anchor_ts
        else:
            if f"{ticker_key}_anchor" in st.session_state:
                st.session_state.pop(f"{ticker_key}_anchor")
            anchor_ts = None
        # -----------------------------------------------------------------
        # Assessment selection
        # -----------------------------------------------------------------
        if tech_score is not None and mom_score is not None and dmas is not None:
            options = [
                "Strongly Bearish",
                "Bearish",
                "Slightly Bearish",
                "Neutral",
                "Slightly Bullish",
                "Bullish",
                "Strongly Bullish",
            ]
            def _default_index_from_dmas(val: float) -> int:
                if val >= 80:
                    return options.index("Strongly Bullish")
                elif val >= 70:
                    return options.index("Bullish")
                elif val >= 60:
                    return options.index("Slightly Bullish")
                elif val >= 40:
                    return options.index("Neutral")
                elif val >= 30:
                    return options.index("Slightly Bearish")
                elif val >= 20:
                    return options.index("Bearish")
                else:
                    return options.index("Strongly Bearish")
            default_idx = _default_index_from_dmas(dmas)
            user_view = st.selectbox(
                "Select your assessment",
                options,
                index=default_idx,
                key=f"{ticker_key}_view_select",
            )
            st.session_state[f"{ticker_key}_selected_view"] = user_view
            st.caption(
                "Your selection will override the automatically computed view in the presentation."
            )
        # -----------------------------------------------------------------
        # Subtitle input
        # -----------------------------------------------------------------
        subtitle_value = st.text_input(
            f"{ticker_key.upper()} subtitle",
            value=st.session_state.get(f"{ticker_key}_subtitle", ""),
            key=f"{ticker_key}_subtitle_input",
        )
        st.session_state[f"{ticker_key}_subtitle"] = subtitle_value
        # -----------------------------------------------------------------
        # Interactive chart
        # -----------------------------------------------------------------
        if excel_available:
            pmode = st.session_state.get("price_mode", "Last Price")
            if selected_index == "Gold":
                fig = make_gold_figure(temp_path, anchor_date=anchor_ts, price_mode=pmode)
            elif selected_index == "Silver":
                fig = make_silver_figure(temp_path, anchor_date=anchor_ts, price_mode=pmode)
            elif selected_index == "Platinum":
                fig = make_platinum_figure(temp_path, anchor_date=anchor_ts, price_mode=pmode)
            elif selected_index == "Palladium":
                fig = make_palladium_figure(temp_path, anchor_date=anchor_ts, price_mode=pmode)
            elif selected_index == "Oil":
                fig = make_oil_figure(temp_path, anchor_date=anchor_ts, price_mode=pmode)
            elif selected_index == "Copper":
                fig = make_copper_figure(temp_path, anchor_date=anchor_ts, price_mode=pmode)
            else:
                # Fallback: show an empty figure if unknown commodity
                fig = go.Figure()
        else:
            # Fallback: compute simple MA and regression channel on synthetic data
            from technical_analysis.equity.spx import _add_moving_averages, _build_fallback_figure  # type: ignore
            df_ma = _add_moving_averages(df_full)
            fig = _build_fallback_figure(df_ma, anchor_date=anchor_ts)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Use the controls above to enable and configure the regression channel. "
            "Green shading indicates an uptrend; red shading indicates a downtrend."
        )


def show_crypto_technical_analysis() -> None:
    """Render the technical analysis interface for crypto assets such as Bitcoin.

    This function closely mirrors the commodity technical analysis interface
    but is tailored for crypto tickers.  At present only Bitcoin is
    supported.  It handles data loading from the uploaded Excel, score
    retrieval, DMAS computation, trading range estimation using the
    Bitcoin volatility index (BVXS Index) with a realised volatility
    fallback, regression channel controls, assessment selection, subtitle
    input and interactive chart rendering.  State is persisted in
    ``st.session_state`` so that the regression channel can be anchored
    at a user‑selected date across reruns.
    """
    # Identify whether an Excel file has been uploaded
    excel_available = "excel_file" in st.session_state

    # Allow selection of supported crypto assets
    index_options = ["Bitcoin", "Ethereum", "Ripple", "Solana", "Binance"]
    default_index = st.session_state.get("ta_crypto_index", "Bitcoin") if st.session_state.get("ta_crypto_index") in index_options else "Bitcoin"
    selected_index = st.sidebar.selectbox(
        "Select crypto for technical analysis",
        options=index_options,
        index=index_options.index(default_index) if default_index in index_options else 0,
        key="ta_crypto_index_select",
    )
    # Persist selection
    st.session_state["ta_crypto_index"] = selected_index

    # Determine ticker, key, chart title, volatility column and helper functions based on selected crypto
    if selected_index == "Bitcoin":
        ticker = "XBTUSD Curncy"
        ticker_key = "bitcoin"
        chart_title = "Bitcoin Technical Chart"
        vol_col_name = "BVXS Index"
        get_tech_score = _get_bitcoin_technical_score
        get_mom_score = _get_bitcoin_momentum_score
        compute_range_fallback = _compute_range_bounds_bitcoin
        make_figure_func = make_bitcoin_figure
    elif selected_index == "Ethereum":
        ticker = "XETUSD Curncy"
        ticker_key = "ethereum"
        chart_title = "Ethereum Technical Chart"
        # Placeholder for implied volatility column; not yet available
        vol_col_name = "XETUSDV1M BGN Curncy"
        get_tech_score = _get_ethereum_technical_score
        get_mom_score = _get_ethereum_momentum_score
        compute_range_fallback = _compute_range_bounds_ethereum
        make_figure_func = make_ethereum_figure
    elif selected_index == "Ripple":
        ticker = "XRPUSD Curncy"
        ticker_key = "ripple"
        chart_title = "Ripple Technical Chart"
        # Placeholder for implied volatility column; not yet available
        vol_col_name = "XRPUSDV1M BGN Curncy"
        get_tech_score = _get_ripple_technical_score
        get_mom_score = _get_ripple_momentum_score
        compute_range_fallback = _compute_range_bounds_ripple
        make_figure_func = make_ripple_figure
    elif selected_index == "Solana":
        ticker = "XSOUSD Curncy"
        ticker_key = "solana"
        chart_title = "Solana Technical Chart"
        # Placeholder for implied volatility column; not yet available
        vol_col_name = "XSOUSDV1M BGN Curncy"
        get_tech_score = _get_solana_technical_score
        get_mom_score = _get_solana_momentum_score
        compute_range_fallback = _compute_range_bounds_solana
        make_figure_func = make_solana_figure
    elif selected_index == "Binance":
        ticker = "XBIUSD Curncy"
        ticker_key = "binance"
        chart_title = "Binance Technical Chart"
        # Placeholder for implied volatility column; not yet available
        vol_col_name = "XBIUSDV1M BGN Curncy"
        get_tech_score = _get_binance_technical_score
        get_mom_score = _get_binance_momentum_score
        compute_range_fallback = _compute_range_bounds_binance
        make_figure_func = make_binance_figure
    else:
        # Default to Bitcoin if unknown selection
        ticker = "XBTUSD Curncy"
        ticker_key = "bitcoin"
        chart_title = f"{selected_index} Technical Chart"
        vol_col_name = "BVXS Index"
        get_tech_score = _get_bitcoin_technical_score
        get_mom_score = _get_bitcoin_momentum_score
        compute_range_fallback = _compute_range_bounds_bitcoin
        make_figure_func = make_bitcoin_figure

    # Load price data
    if excel_available:
        # Save uploaded Excel to a temporary path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(st.session_state["excel_file"].getbuffer())
            tmp.flush()
            temp_path = Path(tmp.name)
        # Read data_prices and clean
        df_prices = pd.read_excel(temp_path, sheet_name="data_prices")
        df_prices = df_prices.drop(index=0)
        df_prices = df_prices[df_prices[df_prices.columns[0]] != "DATES"]
        df_prices["Date"] = pd.to_datetime(df_prices[df_prices.columns[0]], errors="coerce")
        df_prices["Price"] = pd.to_numeric(df_prices[ticker], errors="coerce")
        df_prices = df_prices.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)
        # Adjust prices for selected price mode
        price_mode = st.session_state.get("price_mode", "Last Price")
        used_date = None
        if adjust_prices_for_mode is not None and price_mode:
            try:
                df_prices, used_date = adjust_prices_for_mode(df_prices, price_mode)
            except Exception:
                used_date = None
        st.session_state[f"{ticker_key}_used_date"] = used_date
        df_full = df_prices.copy()
    else:
        # Fallback: synthetic SPX series (not ideal but ensures a chart)
        df_prices = _create_synthetic_spx_series()
        df_full = df_prices.copy()
        used_date = None

    # Determine date range for channel controls
    min_date = df_prices["Date"].min().date()
    max_date = df_prices["Date"].max().date()

    # Chart and controls
    with st.expander(chart_title, expanded=True):
        # Caption with used date
        used_date = st.session_state.get(f"{ticker_key}_used_date")
        price_mode = st.session_state.get("price_mode", "Last Price")
        if used_date is not None:
            if price_mode == "Last Close":
                st.caption(f"Prices as of {used_date.strftime('%d/%m/%Y')} close")
            else:
                st.caption(f"Prices as of {used_date.strftime('%d/%m/%Y')}")
        # -----------------------------------------------------------------
        # Technical and momentum scores
        # -----------------------------------------------------------------
        st.subheader("Technical and momentum scores")
        tech_score: Optional[float] = None
        mom_score: Optional[float] = None
        if excel_available:
            try:
                tech_score = get_tech_score(temp_path)
            except Exception:
                tech_score = None
            try:
                mom_score = get_mom_score(temp_path)
            except Exception:
                mom_score = None
        # Compute DMAS if available
        dmas: Optional[float] = None
        if tech_score is not None and mom_score is not None:
            dmas = round((float(tech_score) + float(mom_score)) / 2.0, 1)
            df_scores = pd.DataFrame(
                {
                    "Technical Score": [tech_score],
                    "Momentum Score": [mom_score],
                    "Average (DMAS)": [dmas],
                }
            )
            st.table(df_scores)
            # Last week's DMAS input
            gauge_key = f"{ticker_key}_last_week_avg"
            gauge_input_key = f"{ticker_key}_last_week_avg_input"
            last_week_default = st.session_state.get(gauge_key, 50.0)
            last_week_input = st.number_input(
                "Last week's average (DMAS)",
                min_value=0.0,
                max_value=100.0,
                value=last_week_default,
                key=gauge_input_key,
            )
            st.session_state[gauge_key] = last_week_input
        else:
            st.info(
                "Technical or momentum score not available in the uploaded Excel. "
                "Please ensure sheets 'data_technical_score' and 'data_trend_rating' exist."
            )
        # -----------------------------------------------------------------
        # Trading range estimation (90d)
        # -----------------------------------------------------------------
        try:
            current_price = df_full["Price"].iloc[-1] if not df_full.empty else None
            if current_price is not None and not np.isnan(current_price):
                use_implied = False
                vol_val: Optional[float] = None
                # Already determined vol_col_name based on selected crypto
                # Attempt to load implied volatility from the Excel file
                if excel_available:
                    try:
                        df_vol = pd.read_excel(temp_path, sheet_name="data_prices")
                        df_vol = df_vol.drop(index=0)
                        df_vol = df_vol[df_vol[df_vol.columns[0]] != "DATES"]
                        df_vol["Date"] = pd.to_datetime(df_vol[df_vol.columns[0]], errors="coerce")
                        if vol_col_name in df_vol.columns:
                            df_vol["Price"] = pd.to_numeric(df_vol[vol_col_name], errors="coerce")
                            df_vol = df_vol.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)[[
                                "Date",
                                "Price",
                            ]]
                            pm = st.session_state.get("price_mode", "Last Price")
                            if adjust_prices_for_mode is not None:
                                try:
                                    df_vol, _ = adjust_prices_for_mode(df_vol, pm)
                                except Exception:
                                    pass
                            if not df_vol.empty:
                                vol_val = float(df_vol["Price"].iloc[-1])
                                use_implied = True
                    except Exception:
                        use_implied = False
                if use_implied and vol_val is not None:
                    expected_move = (current_price * (vol_val / 100.0)) / np.sqrt(52.0)
                    lower_bound = current_price - expected_move
                    upper_bound = current_price + expected_move
                    min_span = 0.02 * current_price
                    if (upper_bound - lower_bound) < min_span:
                        half = min_span / 2.0
                        lower_bound = current_price - half
                        upper_bound = current_price + half
                else:
                    # Fallback to realised volatility
                    # Use crypto-specific realised volatility fallback
                    upper_bound, lower_bound = compute_range_fallback(df_full, lookback_days=90)
                low_pct = (lower_bound - current_price) / current_price * 100.0
                high_pct = (upper_bound - current_price) / current_price * 100.0
                st.write(
                    f"Trading range (90d): Low {lower_bound:,.0f} ({low_pct:+.1f}%), "
                    f"High {upper_bound:,.0f} ({high_pct:+.1f}%)"
                )
            else:
                # Use crypto-specific realised volatility fallback
                upper_bound, lower_bound = compute_range_fallback(df_full, lookback_days=90)
                st.write(
                    f"Trading range (90d): Low {lower_bound:,.0f} – High {upper_bound:,.0f}"
                )
        except Exception:
            pass
        # -----------------------------------------------------------------
        # Regression channel controls
        # -----------------------------------------------------------------
        enable_channel = st.checkbox(
            "Enable regression channel",
            value=bool(st.session_state.get(f"{ticker_key}_anchor")),
            key=f"{ticker_key}_enable_channel",
        )
        anchor_ts: Optional[pd.Timestamp] = None
        if enable_channel:
            # Default anchor uses the selected timeframe rather than a
            # fixed 180‑day window when none is stored in session.
            default_anchor = st.session_state.get(
                f"{ticker_key}_anchor",
                (max_date - pd.Timedelta(days=st.session_state.get("ta_timeframe_days", 180))),
            )
            anchor_input = st.date_input(
                "Select anchor date",
                value=default_anchor,
                min_value=min_date,
                max_value=max_date,
                key=f"{ticker_key}_anchor_date_input",
            )
            anchor_ts = pd.to_datetime(anchor_input)
            st.session_state[f"{ticker_key}_anchor"] = anchor_ts
        else:
            if f"{ticker_key}_anchor" in st.session_state:
                st.session_state.pop(f"{ticker_key}_anchor")
            anchor_ts = None
        # -----------------------------------------------------------------
        # Assessment selection
        # -----------------------------------------------------------------
        if tech_score is not None and mom_score is not None and dmas is not None:
            options = [
                "Strongly Bearish",
                "Bearish",
                "Slightly Bearish",
                "Neutral",
                "Slightly Bullish",
                "Bullish",
                "Strongly Bullish",
            ]
            def _default_index_from_dmas(val: float) -> int:
                if val >= 80:
                    return options.index("Strongly Bullish")
                elif val >= 70:
                    return options.index("Bullish")
                elif val >= 60:
                    return options.index("Slightly Bullish")
                elif val >= 40:
                    return options.index("Neutral")
                elif val >= 30:
                    return options.index("Slightly Bearish")
                elif val >= 20:
                    return options.index("Bearish")
                else:
                    return options.index("Strongly Bearish")
            default_idx = _default_index_from_dmas(dmas)
            user_view = st.selectbox(
                "Select your assessment",
                options,
                index=default_idx,
                key=f"{ticker_key}_view_select",
            )
            st.session_state[f"{ticker_key}_selected_view"] = user_view
            st.caption(
                "Your selection will override the automatically computed view in the presentation."
            )
        # -----------------------------------------------------------------
        # Subtitle input
        # -----------------------------------------------------------------
        subtitle_value = st.text_input(
            f"{ticker_key.upper()} subtitle",
            value=st.session_state.get(f"{ticker_key}_subtitle", ""),
            key=f"{ticker_key}_subtitle_input",
        )
        st.session_state[f"{ticker_key}_subtitle"] = subtitle_value
        # -----------------------------------------------------------------
        # Interactive chart
        # -----------------------------------------------------------------
        if excel_available:
            pmode = st.session_state.get("price_mode", "Last Price")
            fig = make_figure_func(temp_path, anchor_date=anchor_ts, price_mode=pmode)
        else:
            # Build fallback figure on synthetic data
            from technical_analysis.equity.spx import _add_moving_averages, _build_fallback_figure  # type: ignore
            df_ma = _add_moving_averages(df_full)
            fig = _build_fallback_figure(df_ma, anchor_date=anchor_ts)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Use the controls above to enable and configure the regression channel. "
            "Green shading indicates an uptrend; red shading indicates a downtrend."
        )


def show_market_breadth_page() -> None:
    st.header("Market Breadth")

    if "excel_file" not in st.session_state:
        st.error("Upload an Excel file first (Upload page).")
        return

    import tempfile, pathlib
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(st.session_state["excel_file"].getbuffer())
        xl_path = pathlib.Path(tmp.name)

    df = _load_breadth_page_data(xl_path)
    if df.empty:
        st.warning("No breadth data found in **bql_formula** (columns AB–AE).")
        return

    st.dataframe(
        _style_breadth_page(df),
        use_container_width=True,
        height=min(600, 50 + 25 * len(df)),
    )

    with st.expander("Debug – first parsed rows"):
        st.write(_debug_breadth_rows(xl_path))


def show_generate_presentation_page():
    """Generate a customised PowerPoint presentation based on user selections."""
    st.sidebar.header("Generate Presentation")
    if "excel_file" not in st.session_state or "pptx_file" not in st.session_state:
        st.sidebar.error(
            "Please upload both an Excel file and a PowerPoint template in the Upload page."
        )
        st.stop()

    # Lazy import functions for inserting charts into PPT
    from ytd_perf.equity_ytd import insert_equity_chart
    from ytd_perf.commodity_ytd import insert_commodity_chart
    from ytd_perf.crypto_ytd import insert_crypto_chart

    st.sidebar.write("### Summary of selections")
    st.sidebar.write("Equities:", st.session_state.get("selected_eq_names", []))
    st.sidebar.write("Commodities:", st.session_state.get("selected_co_names", []))
    st.sidebar.write("Cryptos:", st.session_state.get("selected_cr_names", []))
    # Display FX pairs being analysed (fixed list) for user awareness
    st.sidebar.write(
        "FX:",
        [
            "DXY",
            "EUR/USD",
            "EUR/CHF",
            "EUR/GBP",
            "EUR/JPY",
            "EUR/AUD",
            "EUR/CAD",
            "EUR/BRL",
            "EUR/RUB",
            "EUR/ZAR",
            "EUR/MXN",
        ],
    )
    # Display rates tickers being analysed (fixed list)
    st.sidebar.write(
        "Rates:",
        [
            "US - 2Y",
            "US - 10Y",
            "US - 30Y",
            "EUR - 2Y",
            "EUR - 10Y",
            "EUR - 30Y",
            "CN - 2Y",
            "CN - 10Y",
            "CN - 30Y",
            "JP - 2Y",
            "JP - 10Y",
            "JP - 30Y",
        ],
    )
    # Display credit indices being analysed (fixed list) for user awareness
    st.sidebar.write(
        "Credit:",
        [
            "USD - IG",
            "USD - HY",
            "EUR - IG",
            "EUR - HY",
            "Asia (ex JP) - IG",
            "Asia - HY",
            "EM - IG",
            "EM - HY",
        ],
    )

    if st.sidebar.button("Generate updated PPTX", key="gen_ppt_button"):
        # Write the uploaded PPTX to a temporary file so that python-pptx
        # can read it reliably.  Also write the uploaded Excel file to a
        # temporary XLSX path so that multiple reads do not exhaust the
        # underlying file-like object.  The Excel path is reused for
        # inserting charts and scores throughout the presentation.
        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as tmp_input:
            tmp_input.write(st.session_state["pptx_file"].getbuffer())
            tmp_input.flush()
            prs = Presentation(tmp_input.name)

        # Persist the Excel to a temporary path to avoid file pointer
        # exhaustion when pandas reads multiple sheets.  Without this,
        # repeated reads from the UploadedFile can yield empty DataFrames.
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_xls:
            tmp_xls.write(st.session_state["excel_file"].getbuffer())
            tmp_xls.flush()
            excel_path_for_ppt = Path(tmp_xls.name)

        # Insert YTD charts
        prs = insert_equity_chart(
            prs,
            excel_path_for_ppt,
            subtitle=st.session_state.get("eq_subtitle", ""),
            tickers=st.session_state.get("selected_eq_tickers", []),
            price_mode=st.session_state.get("price_mode", "Last Price"),
        )
        prs = insert_commodity_chart(
            prs,
            excel_path_for_ppt,
            subtitle=st.session_state.get("co_subtitle", ""),
            tickers=st.session_state.get("selected_co_tickers", []),
            price_mode=st.session_state.get("price_mode", "Last Price"),
        )
        prs = insert_crypto_chart(
            prs,
            excel_path_for_ppt,
            subtitle=st.session_state.get("cr_subtitle", ""),
            tickers=st.session_state.get("selected_cr_tickers", []),
            price_mode=st.session_state.get("price_mode", "Last Price"),
        )


        # Determine which equity index was selected for technical analysis (not used here since we insert all indices)
        selected_index = st.session_state.get("ta_equity_index", "S&P 500")

        # Retrieve anchors for SPX, CSI, Nikkei, TASI, Sensex and DAX slides
        spx_anchor_dt = st.session_state.get("spx_anchor")
        csi_anchor_dt = st.session_state.get("csi_anchor")
        nikkei_anchor_dt = st.session_state.get("nikkei_anchor")
        tasi_anchor_dt = st.session_state.get("tasi_anchor")
        sensex_anchor_dt = st.session_state.get("sensex_anchor")
        dax_anchor_dt = st.session_state.get("dax_anchor")
        smi_anchor_dt = st.session_state.get("smi_anchor")
        ibov_anchor_dt = st.session_state.get("ibov_anchor")
        mexbol_anchor_dt = st.session_state.get("mexbol_anchor")
        # Anchor for Gold regression channel (commodity)
        gold_anchor_dt = st.session_state.get("gold_anchor")
        # Anchor for Silver regression channel (commodity)
        silver_anchor_dt = st.session_state.get("silver_anchor")
        # Anchor for Platinum regression channel (commodity)
        platinum_anchor_dt = st.session_state.get("platinum_anchor")
        # Anchor for Palladium regression channel (commodity)
        palladium_anchor_dt = st.session_state.get("palladium_anchor")
        # Anchor for Oil regression channel (commodity)
        oil_anchor_dt = st.session_state.get("oil_anchor")
        # Anchor for Copper regression channel (commodity)
        copper_anchor_dt = st.session_state.get("copper_anchor")
        # Anchor for Bitcoin regression channel (crypto)
        bitcoin_anchor_dt = st.session_state.get("bitcoin_anchor")
        # Anchor for Ethereum regression channel (crypto)
        ethereum_anchor_dt = st.session_state.get("ethereum_anchor")
        # Anchor for Ripple regression channel (crypto)
        ripple_anchor_dt = st.session_state.get("ripple_anchor")
        # Anchor for Solana regression channel (crypto)
        solana_anchor_dt = st.session_state.get("solana_anchor")
        # Anchor for Binance regression channel (crypto)
        binance_anchor_dt = st.session_state.get("binance_anchor")

        # Common price mode
        pmode = st.session_state.get("price_mode", "Last Price")

        # ------------------------------------------------------------------
        # Insert SPX technical analysis slide (always)
        # ------------------------------------------------------------------
        prs = insert_spx_technical_chart_with_callout(
            prs,
            excel_path_for_ppt,
            spx_anchor_dt,
            price_mode=pmode,
        )
        # Insert SPX technical score number
        prs = insert_spx_technical_score_number(
            prs,
            excel_path_for_ppt,
        )
        # Insert SPX momentum score number
        prs = insert_spx_momentum_score_number(
            prs,
            excel_path_for_ppt,
        )
        # Insert SPX subtitle from user input
        prs = insert_spx_subtitle(
            prs,
            st.session_state.get("spx_subtitle", ""),
        )
        # Insert SPX average gauge (last week's average is 0–100)
        spx_last_week_avg = st.session_state.get("spx_last_week_avg", 50.0)
        prs = insert_spx_average_gauge(
            prs,
            excel_path_for_ppt,
            spx_last_week_avg,
        )
        # Insert the technical assessment text into the 'spx_view' textbox.
        manual_view_spx = st.session_state.get("spx_selected_view")
        prs = insert_spx_technical_assessment(
            prs,
            excel_path_for_ppt,
            manual_desc=manual_view_spx,
        )
        # Compute used date for SPX source footnote
        try:
            import pandas as pd
            df_prices = pd.read_excel(excel_path_for_ppt, sheet_name="data_prices")
            df_prices = df_prices.drop(index=0)
            df_prices = df_prices[df_prices[df_prices.columns[0]] != "DATES"]
            df_prices["Date"] = pd.to_datetime(df_prices[df_prices.columns[0]], errors="coerce")
            df_prices["Price"] = pd.to_numeric(df_prices["SPX Index"], errors="coerce")
            df_prices = df_prices.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)[
                ["Date", "Price"]
            ]
            df_adj, used_date_spx = adjust_prices_for_mode(df_prices, pmode)
        except Exception:
            used_date_spx = None
        prs = insert_spx_source(
            prs,
            used_date_spx,
            pmode,
        )

        # ------------------------------------------------------------------
        # Insert CSI technical analysis slide (always)
        # ------------------------------------------------------------------
        prs = insert_csi_technical_chart_with_callout(
            prs,
            excel_path_for_ppt,
            csi_anchor_dt,
            price_mode=pmode,
        )
        # Insert CSI technical score number
        prs = insert_csi_technical_score_number(
            prs,
            excel_path_for_ppt,
        )
        # Insert CSI momentum score number
        prs = insert_csi_momentum_score_number(
            prs,
            excel_path_for_ppt,
        )
        # Insert CSI subtitle from user input
        prs = insert_csi_subtitle(
            prs,
            st.session_state.get("csi_subtitle", ""),
        )
        # Insert CSI average gauge (last week's average is 0–100)
        csi_last_week_avg = st.session_state.get("csi_last_week_avg", 50.0)
        prs = insert_csi_average_gauge(
            prs,
            excel_path_for_ppt,
            csi_last_week_avg,
        )
        # Insert the technical assessment text into the 'csi_view' textbox.
        manual_view_csi = st.session_state.get("csi_selected_view")
        prs = insert_csi_technical_assessment(
            prs,
            excel_path_for_ppt,
            manual_desc=manual_view_csi,
        )
        # Compute used date for CSI source footnote
        try:
            import pandas as pd
            df_prices_csi = pd.read_excel(excel_path_for_ppt, sheet_name="data_prices")
            df_prices_csi = df_prices_csi.drop(index=0)
            df_prices_csi = df_prices_csi[df_prices_csi[df_prices_csi.columns[0]] != "DATES"]
            df_prices_csi["Date"] = pd.to_datetime(df_prices_csi[df_prices_csi.columns[0]], errors="coerce")
            df_prices_csi["Price"] = pd.to_numeric(df_prices_csi["SHSZ300 Index"], errors="coerce")
            df_prices_csi = df_prices_csi.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)[
                ["Date", "Price"]
            ]
            df_adj_csi, used_date_csi = adjust_prices_for_mode(df_prices_csi, pmode)
        except Exception:
            used_date_csi = None
        prs = insert_csi_source(
            prs,
            used_date_csi,
            pmode,
        )

        # ------------------------------------------------------------------
        # Insert Nikkei technical analysis slide (always)
        # ------------------------------------------------------------------
        prs = insert_nikkei_technical_chart_with_callout(
            prs,
            excel_path_for_ppt,
            nikkei_anchor_dt,
            price_mode=pmode,
        )
        # Insert Nikkei technical score number
        prs = insert_nikkei_technical_score_number(
            prs,
            excel_path_for_ppt,
        )
        # Insert Nikkei momentum score number
        prs = insert_nikkei_momentum_score_number(
            prs,
            excel_path_for_ppt,
        )
        # Insert Nikkei subtitle from user input
        prs = insert_nikkei_subtitle(
            prs,
            st.session_state.get("nikkei_subtitle", ""),
        )
        # Insert Nikkei average gauge (last week's average is 0–100)
        nikkei_last_week_avg = st.session_state.get("nikkei_last_week_avg", 50.0)
        prs = insert_nikkei_average_gauge(
            prs,
            excel_path_for_ppt,
            nikkei_last_week_avg,
        )
        # Insert the technical assessment text into the 'nikkei_view' textbox
        manual_view_nikkei = st.session_state.get("nikkei_selected_view")
        prs = insert_nikkei_technical_assessment(
            prs,
            excel_path_for_ppt,
            manual_desc=manual_view_nikkei,
        )
        # Compute used date for Nikkei source footnote
        try:
            import pandas as pd
            df_prices_nikkei = pd.read_excel(excel_path_for_ppt, sheet_name="data_prices")
            df_prices_nikkei = df_prices_nikkei.drop(index=0)
            df_prices_nikkei = df_prices_nikkei[df_prices_nikkei[df_prices_nikkei.columns[0]] != "DATES"]
            df_prices_nikkei["Date"] = pd.to_datetime(df_prices_nikkei[df_prices_nikkei.columns[0]], errors="coerce")
            df_prices_nikkei["Price"] = pd.to_numeric(df_prices_nikkei["NKY Index"], errors="coerce")
            df_prices_nikkei = df_prices_nikkei.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)[
                ["Date", "Price"]
            ]
            df_adj_nikkei, used_date_nikkei = adjust_prices_for_mode(df_prices_nikkei, pmode)
        except Exception:
            used_date_nikkei = None
        prs = insert_nikkei_source(
            prs,
            used_date_nikkei,
            pmode,
        )

        # ------------------------------------------------------------------
        # Insert TASI technical analysis slide (always)
        # ------------------------------------------------------------------
        prs = insert_tasi_technical_chart_with_callout(
            prs,
            excel_path_for_ppt,
            tasi_anchor_dt,
            price_mode=pmode,
        )
        # Insert TASI technical score number
        prs = insert_tasi_technical_score_number(
            prs,
            excel_path_for_ppt,
        )
        # Insert TASI momentum score number
        prs = insert_tasi_momentum_score_number(
            prs,
            excel_path_for_ppt,
        )
        # Insert TASI subtitle from user input
        prs = insert_tasi_subtitle(
            prs,
            st.session_state.get("tasi_subtitle", ""),
        )
        # Insert TASI average gauge (last week's average is 0–100)
        tasi_last_week_avg = st.session_state.get("tasi_last_week_avg", 50.0)
        prs = insert_tasi_average_gauge(
            prs,
            excel_path_for_ppt,
            tasi_last_week_avg,
        )
        # Insert the technical assessment text into the 'tasi_view' textbox
        manual_view_tasi = st.session_state.get("tasi_selected_view")
        prs = insert_tasi_technical_assessment(
            prs,
            excel_path_for_ppt,
            manual_desc=manual_view_tasi,
        )
        # Compute used date for TASI source footnote
        try:
            import pandas as pd
            df_prices_tasi = pd.read_excel(excel_path_for_ppt, sheet_name="data_prices")
            df_prices_tasi = df_prices_tasi.drop(index=0)
            df_prices_tasi = df_prices_tasi[df_prices_tasi[df_prices_tasi.columns[0]] != "DATES"]
            df_prices_tasi["Date"] = pd.to_datetime(df_prices_tasi[df_prices_tasi.columns[0]], errors="coerce")
            # Use the SASEIDX Index column for TASI prices
            df_prices_tasi["Price"] = pd.to_numeric(df_prices_tasi["SASEIDX Index"], errors="coerce")
            df_prices_tasi = df_prices_tasi.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)[
                ["Date", "Price"]
            ]
            df_adj_tasi, used_date_tasi = adjust_prices_for_mode(df_prices_tasi, pmode)
        except Exception:
            used_date_tasi = None
        prs = insert_tasi_source(
            prs,
            used_date_tasi,
            pmode,
        )

        # ------------------------------------------------------------------
        # Insert Sensex technical analysis slide
        # ------------------------------------------------------------------
        # Sensex technical analysis uses realised volatility and a separate implied vol index (INVIXN)
        prs = insert_sensex_technical_chart_with_callout(
            prs,
            excel_path_for_ppt,
            sensex_anchor_dt,
            price_mode=pmode,
        )
        # Insert Sensex technical score number
        prs = insert_sensex_technical_score_number(
            prs,
            excel_path_for_ppt,
        )
        # Insert Sensex momentum score number
        prs = insert_sensex_momentum_score_number(
            prs,
            excel_path_for_ppt,
        )
        # Insert Sensex subtitle from user input
        prs = insert_sensex_subtitle(
            prs,
            st.session_state.get("sensex_subtitle", ""),
        )
        # Insert Sensex average gauge (last week's average is 0–100)
        sensex_last_week_avg = st.session_state.get("sensex_last_week_avg", 50.0)
        prs = insert_sensex_average_gauge(
            prs,
            excel_path_for_ppt,
            sensex_last_week_avg,
        )
        # Insert the technical assessment text into the 'sensex_view' textbox.
        manual_view_sensex = st.session_state.get("sensex_selected_view")
        prs = insert_sensex_technical_assessment(
            prs,
            excel_path_for_ppt,
            manual_desc=manual_view_sensex,
        )
        # Compute used date for Sensex source footnote
        try:
            import pandas as pd
            df_prices_sensex = pd.read_excel(excel_path_for_ppt, sheet_name="data_prices")
            df_prices_sensex = df_prices_sensex.drop(index=0)
            df_prices_sensex = df_prices_sensex[df_prices_sensex[df_prices_sensex.columns[0]] != "DATES"]
            df_prices_sensex["Date"] = pd.to_datetime(df_prices_sensex[df_prices_sensex.columns[0]], errors="coerce")
            # Use the SENSEX Index column for Sensex prices
            df_prices_sensex["Price"] = pd.to_numeric(df_prices_sensex["SENSEX Index"], errors="coerce")
            df_prices_sensex = df_prices_sensex.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)[
                ["Date", "Price"]
            ]
            df_adj_sensex, used_date_sensex = adjust_prices_for_mode(df_prices_sensex, pmode)
        except Exception:
            used_date_sensex = None
        prs = insert_sensex_source(
            prs,
            used_date_sensex,
            pmode,
        )

        # ------------------------------------------------------------------
        # Insert DAX technical analysis slide (always)
        # ------------------------------------------------------------------
        prs = insert_dax_technical_chart_with_callout(
            prs,
            excel_path_for_ppt,
            dax_anchor_dt,
            price_mode=pmode,
        )
        # Insert DAX technical score number
        prs = insert_dax_technical_score_number(
            prs,
            excel_path_for_ppt,
        )
        # Insert DAX momentum score number
        prs = insert_dax_momentum_score_number(
            prs,
            excel_path_for_ppt,
        )
        # Insert DAX subtitle from user input
        prs = insert_dax_subtitle(
            prs,
            st.session_state.get("dax_subtitle", ""),
        )
        # Insert DAX average gauge (last week's average is 0–100)
        dax_last_week_avg = st.session_state.get("dax_last_week_avg", 50.0)
        prs = insert_dax_average_gauge(
            prs,
            excel_path_for_ppt,
            dax_last_week_avg,
        )
        # Insert the technical assessment text into the 'dax_view' textbox.
        manual_view_dax = st.session_state.get("dax_selected_view")
        prs = insert_dax_technical_assessment(
            prs,
            excel_path_for_ppt,
            manual_desc=manual_view_dax,
        )
        # Compute used date for DAX source footnote
        try:
            import pandas as pd
            df_prices_dax = pd.read_excel(excel_path_for_ppt, sheet_name="data_prices")
            df_prices_dax = df_prices_dax.drop(index=0)
            df_prices_dax = df_prices_dax[df_prices_dax[df_prices_dax.columns[0]] != "DATES"]
            df_prices_dax["Date"] = pd.to_datetime(df_prices_dax[df_prices_dax.columns[0]], errors="coerce")
            # Use the DAX Index column for DAX prices
            df_prices_dax["Price"] = pd.to_numeric(df_prices_dax["DAX Index"], errors="coerce")
            df_prices_dax = df_prices_dax.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)[
                ["Date", "Price"]
            ]
            df_adj_dax, used_date_dax = adjust_prices_for_mode(df_prices_dax, pmode)
        except Exception:
            used_date_dax = None
        prs = insert_dax_source(
            prs,
            used_date_dax,
            pmode,
        )

        # ------------------------------------------------------------------
        # Insert SMI technical analysis slide (always)
        # ------------------------------------------------------------------
        prs = insert_smi_technical_chart_with_callout(
            prs,
            excel_path_for_ppt,
            smi_anchor_dt,
            price_mode=pmode,
        )
        # Insert SMI technical score number
        prs = insert_smi_technical_score_number(
            prs,
            excel_path_for_ppt,
        )
        # Insert SMI momentum score number
        prs = insert_smi_momentum_score_number(
            prs,
            excel_path_for_ppt,
        )
        # Insert SMI subtitle from user input
        prs = insert_smi_subtitle(
            prs,
            st.session_state.get("smi_subtitle", ""),
        )
        # Insert SMI average gauge (last week's average is 0–100)
        smi_last_week_avg = st.session_state.get("smi_last_week_avg", 50.0)
        prs = insert_smi_average_gauge(
            prs,
            excel_path_for_ppt,
            smi_last_week_avg,
        )
        # Insert the technical assessment text into the 'smi_view' textbox.
        manual_view_smi = st.session_state.get("smi_selected_view")
        prs = insert_smi_technical_assessment(
            prs,
            excel_path_for_ppt,
            manual_desc=manual_view_smi,
        )
        # Compute used date for SMI source footnote
        try:
            import pandas as pd
            df_prices_smi = pd.read_excel(excel_path_for_ppt, sheet_name="data_prices")
            df_prices_smi = df_prices_smi.drop(index=0)
            df_prices_smi = df_prices_smi[df_prices_smi[df_prices_smi.columns[0]] != "DATES"]
            df_prices_smi["Date"] = pd.to_datetime(df_prices_smi[df_prices_smi.columns[0]], errors="coerce")
            # Use the SMI Index column for SMI prices
            df_prices_smi["Price"] = pd.to_numeric(df_prices_smi["SMI Index"], errors="coerce")
            df_prices_smi = df_prices_smi.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)[
                ["Date", "Price"]
            ]
            df_adj_smi, used_date_smi = adjust_prices_for_mode(df_prices_smi, pmode)
        except Exception:
            used_date_smi = None
        prs = insert_smi_source(
            prs,
            used_date_smi,
            pmode,
        )

        # ------------------------------------------------------------------
        # Insert IBOV technical analysis slide (always)
        # ------------------------------------------------------------------
        prs = insert_ibov_technical_chart_with_callout(
            prs,
            excel_path_for_ppt,
            ibov_anchor_dt,
            price_mode=pmode,
        )
        # Insert IBOV technical score number
        prs = insert_ibov_technical_score_number(
            prs,
            excel_path_for_ppt,
        )
        # Insert IBOV momentum score number
        prs = insert_ibov_momentum_score_number(
            prs,
            excel_path_for_ppt,
        )
        # Insert IBOV subtitle from user input
        prs = insert_ibov_subtitle(
            prs,
            st.session_state.get("ibov_subtitle", ""),
        )
        # Insert IBOV average gauge (last week's average is 0–100)
        ibov_last_week_avg = st.session_state.get("ibov_last_week_avg", 50.0)
        prs = insert_ibov_average_gauge(
            prs,
            excel_path_for_ppt,
            ibov_last_week_avg,
        )
        # Insert the technical assessment text into the 'ibov_view' textbox.
        manual_view_ibov = st.session_state.get("ibov_selected_view")
        prs = insert_ibov_technical_assessment(
            prs,
            excel_path_for_ppt,
            manual_desc=manual_view_ibov,
        )
        # Compute used date for IBOV source footnote
        try:
            import pandas as pd
            df_prices_ibov = pd.read_excel(excel_path_for_ppt, sheet_name="data_prices")
            df_prices_ibov = df_prices_ibov.drop(index=0)
            df_prices_ibov = df_prices_ibov[df_prices_ibov[df_prices_ibov.columns[0]] != "DATES"]
            df_prices_ibov["Date"] = pd.to_datetime(df_prices_ibov[df_prices_ibov.columns[0]], errors="coerce")
            # Use the IBOV Index column for IBOV prices
            df_prices_ibov["Price"] = pd.to_numeric(df_prices_ibov["IBOV Index"], errors="coerce")
            df_prices_ibov = df_prices_ibov.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)[
                ["Date", "Price"]
            ]
            df_adj_ibov, used_date_ibov = adjust_prices_for_mode(df_prices_ibov, pmode)
        except Exception:
            used_date_ibov = None
        prs = insert_ibov_source(
            prs,
            used_date_ibov,
            pmode,
        )

        # ------------------------------------------------------------------
        # Insert Mexbol technical analysis slide (always)
        # ------------------------------------------------------------------
        prs = insert_mexbol_technical_chart_with_callout(
            prs,
            excel_path_for_ppt,
            mexbol_anchor_dt,
            price_mode=pmode,
        )
        # Insert Mexbol technical score number
        prs = insert_mexbol_technical_score_number(
            prs,
            excel_path_for_ppt,
        )
        # Insert Mexbol momentum score number
        prs = insert_mexbol_momentum_score_number(
            prs,
            excel_path_for_ppt,
        )
        # Insert Mexbol subtitle from user input
        prs = insert_mexbol_subtitle(
            prs,
            st.session_state.get("mexbol_subtitle", ""),
        )
        # Insert Mexbol average gauge (last week's average is 0–100)
        mexbol_last_week_avg = st.session_state.get("mexbol_last_week_avg", 50.0)
        prs = insert_mexbol_average_gauge(
            prs,
            excel_path_for_ppt,
            mexbol_last_week_avg,
        )
        # Insert the technical assessment text into the 'mexbol_view' textbox.
        manual_view_mexbol = st.session_state.get("mexbol_selected_view")
        prs = insert_mexbol_technical_assessment(
            prs,
            excel_path_for_ppt,
            manual_desc=manual_view_mexbol,
        )
        # Compute used date for Mexbol source footnote
        try:
            import pandas as pd
            df_prices_mexbol = pd.read_excel(excel_path_for_ppt, sheet_name="data_prices")
            df_prices_mexbol = df_prices_mexbol.drop(index=0)
            df_prices_mexbol = df_prices_mexbol[df_prices_mexbol[df_prices_mexbol.columns[0]] != "DATES"]
            df_prices_mexbol["Date"] = pd.to_datetime(df_prices_mexbol[df_prices_mexbol.columns[0]], errors="coerce")
            # Use the MEXBOL Index column for Mexbol prices
            df_prices_mexbol["Price"] = pd.to_numeric(df_prices_mexbol["MEXBOL Index"], errors="coerce")
            df_prices_mexbol = df_prices_mexbol.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)[
                ["Date", "Price"]
            ]
            df_adj_mexbol, used_date_mexbol = adjust_prices_for_mode(df_prices_mexbol, pmode)
        except Exception:
            used_date_mexbol = None
        prs = insert_mexbol_source(
            prs,
            used_date_mexbol,
            pmode,
        )

        # ------------------------------------------------------------------
        # Insert Gold technical analysis slide (commodity)
        # ------------------------------------------------------------------
        # Only attempt if Gold helper functions are available (imported above)
        try:
            # Insert the Gold chart with call-out and regression channel anchored at gold_anchor_dt
            prs = insert_gold_technical_chart_with_callout(
                prs,
                excel_path_for_ppt,
                gold_anchor_dt,
                price_mode=pmode,
            )
            # Insert Gold technical and momentum scores
            prs = insert_gold_technical_score_number(
                prs,
                excel_path_for_ppt,
            )
            prs = insert_gold_momentum_score_number(
                prs,
                excel_path_for_ppt,
            )
            # Insert Gold subtitle from user input
            prs = insert_gold_subtitle(
                prs,
                st.session_state.get("gold_subtitle", ""),
            )
            # Insert Gold average gauge (last week's average DMAS)
            gold_last_week_avg = st.session_state.get("gold_last_week_avg", 50.0)
            prs = insert_gold_average_gauge(
                prs,
                excel_path_for_ppt,
                gold_last_week_avg,
            )
            # Insert the technical assessment text into the 'gold_view' textbox
            manual_view_gold = st.session_state.get("gold_selected_view")
            prs = insert_gold_technical_assessment(
                prs,
                excel_path_for_ppt,
                manual_desc=manual_view_gold,
            )
            # Compute used date for Gold source footnote
            try:
                import pandas as pd
                df_prices_gold = pd.read_excel(excel_path_for_ppt, sheet_name="data_prices")
                df_prices_gold = df_prices_gold.drop(index=0)
                df_prices_gold = df_prices_gold[df_prices_gold[df_prices_gold.columns[0]] != "DATES"]
                df_prices_gold["Date"] = pd.to_datetime(df_prices_gold[df_prices_gold.columns[0]], errors="coerce")
                # Use the GCA Comdty column for Gold prices
                df_prices_gold["Price"] = pd.to_numeric(df_prices_gold["GCA Comdty"], errors="coerce")
                df_prices_gold = df_prices_gold.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)[
                    ["Date", "Price"]
                ]
                df_adj_gold, used_date_gold = adjust_prices_for_mode(df_prices_gold, pmode)
            except Exception:
                used_date_gold = None
            prs = insert_gold_source(
                prs,
                used_date_gold,
                pmode,
            )
        except Exception:
            # If any part of the Gold insertion fails, continue to Silver without error
            pass

        # ------------------------------------------------------------------
        # Insert Silver technical analysis slide (commodity)
        # ------------------------------------------------------------------
        try:
            prs = insert_silver_technical_chart_with_callout(
                prs,
                excel_path_for_ppt,
                silver_anchor_dt,
                price_mode=pmode,
            )
            prs = insert_silver_technical_score_number(
                prs,
                excel_path_for_ppt,
            )
            prs = insert_silver_momentum_score_number(
                prs,
                excel_path_for_ppt,
            )
            prs = insert_silver_subtitle(
                prs,
                st.session_state.get("silver_subtitle", ""),
            )
            silver_last_week_avg = st.session_state.get("silver_last_week_avg", 50.0)
            prs = insert_silver_average_gauge(
                prs,
                excel_path_for_ppt,
                silver_last_week_avg,
            )
            manual_view_silver = st.session_state.get("silver_selected_view")
            prs = insert_silver_technical_assessment(
                prs,
                excel_path_for_ppt,
                manual_desc=manual_view_silver,
            )
            try:
                import pandas as pd
                df_prices_silver = pd.read_excel(excel_path_for_ppt, sheet_name="data_prices")
                df_prices_silver = df_prices_silver.drop(index=0)
                df_prices_silver = df_prices_silver[df_prices_silver[df_prices_silver.columns[0]] != "DATES"]
                df_prices_silver["Date"] = pd.to_datetime(df_prices_silver[df_prices_silver.columns[0]], errors="coerce")
                df_prices_silver["Price"] = pd.to_numeric(df_prices_silver["SIA Comdty"], errors="coerce")
                df_prices_silver = df_prices_silver.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)[
                    ["Date", "Price"]
                ]
                df_adj_silver, used_date_silver = adjust_prices_for_mode(df_prices_silver, pmode)
            except Exception:
                used_date_silver = None
            prs = insert_silver_source(
                prs,
                used_date_silver,
                pmode,
            )
        except Exception:
            # If Gold or Silver module is unavailable or insertion fails, continue without error
            pass

        # ------------------------------------------------------------------
        # Insert Platinum technical analysis slide (commodity)
        # ------------------------------------------------------------------
        try:
            # Insert the Platinum chart with call-out and regression channel anchored at platinum_anchor_dt
            prs = insert_platinum_technical_chart_with_callout(
                prs,
                excel_path_for_ppt,
                platinum_anchor_dt,
                price_mode=pmode,
            )
            # Insert Platinum technical and momentum scores
            prs = insert_platinum_technical_score_number(
                prs,
                excel_path_for_ppt,
            )
            prs = insert_platinum_momentum_score_number(
                prs,
                excel_path_for_ppt,
            )
            # Insert Platinum subtitle from user input
            prs = insert_platinum_subtitle(
                prs,
                st.session_state.get("platinum_subtitle", ""),
            )
            # Insert Platinum average gauge (last week's average DMAS)
            platinum_last_week_avg = st.session_state.get("platinum_last_week_avg", 50.0)
            prs = insert_platinum_average_gauge(
                prs,
                excel_path_for_ppt,
                platinum_last_week_avg,
            )
            # Insert the technical assessment text into the 'platinum_view' textbox
            manual_view_platinum = st.session_state.get("platinum_selected_view")
            prs = insert_platinum_technical_assessment(
                prs,
                excel_path_for_ppt,
                manual_desc=manual_view_platinum,
            )
            # Compute used date for Platinum source footnote
            try:
                import pandas as pd
                df_prices_platinum = pd.read_excel(excel_path_for_ppt, sheet_name="data_prices")
                df_prices_platinum = df_prices_platinum.drop(index=0)
                df_prices_platinum = df_prices_platinum[
                    df_prices_platinum[df_prices_platinum.columns[0]] != "DATES"
                ]
                df_prices_platinum["Date"] = pd.to_datetime(
                    df_prices_platinum[df_prices_platinum.columns[0]], errors="coerce"
                )
                # Use the XPT Comdty column for Platinum prices
                df_prices_platinum["Price"] = pd.to_numeric(
                    df_prices_platinum["XPT Comdty"], errors="coerce"
                )
                df_prices_platinum = df_prices_platinum.dropna(subset=["Date", "Price"]).sort_values(
                    "Date"
                ).reset_index(drop=True)[
                    ["Date", "Price"]
                ]
                df_adj_platinum, used_date_platinum = adjust_prices_for_mode(
                    df_prices_platinum, pmode
                )
            except Exception:
                used_date_platinum = None
            prs = insert_platinum_source(
                prs,
                used_date_platinum,
                pmode,
            )
        except Exception:
            # If Platinum module is unavailable or insertion fails, continue without error
            pass

        # ------------------------------------------------------------------
        # Insert Palladium technical analysis slide (commodity)
        # ------------------------------------------------------------------
        try:
            # Insert the Palladium chart with call-out and regression channel anchored at palladium_anchor_dt
            prs = insert_palladium_technical_chart_with_callout(
                prs,
                excel_path_for_ppt,
                palladium_anchor_dt,
                price_mode=pmode,
            )
            # Insert Palladium technical and momentum scores
            prs = insert_palladium_technical_score_number(
                prs,
                excel_path_for_ppt,
            )
            prs = insert_palladium_momentum_score_number(
                prs,
                excel_path_for_ppt,
            )
            # Insert Palladium subtitle from user input
            prs = insert_palladium_subtitle(
                prs,
                st.session_state.get("palladium_subtitle", ""),
            )
            # Insert Palladium average gauge (last week's average DMAS)
            palladium_last_week_avg = st.session_state.get("palladium_last_week_avg", 50.0)
            prs = insert_palladium_average_gauge(
                prs,
                excel_path_for_ppt,
                palladium_last_week_avg,
            )
            # Insert the technical assessment text into the 'palladium_view' textbox
            manual_view_palladium = st.session_state.get("palladium_selected_view")
            prs = insert_palladium_technical_assessment(
                prs,
                excel_path_for_ppt,
                manual_desc=manual_view_palladium,
            )
            # Compute used date for Palladium source footnote
            try:
                import pandas as pd
                df_prices_palladium = pd.read_excel(excel_path_for_ppt, sheet_name="data_prices")
                df_prices_palladium = df_prices_palladium.drop(index=0)
                df_prices_palladium = df_prices_palladium[
                    df_prices_palladium[df_prices_palladium.columns[0]] != "DATES"
                ]
                df_prices_palladium["Date"] = pd.to_datetime(
                    df_prices_palladium[df_prices_palladium.columns[0]], errors="coerce"
                )
                # Use the XPD Curncy column for Palladium prices
                df_prices_palladium["Price"] = pd.to_numeric(
                    df_prices_palladium["XPD Curncy"], errors="coerce"
                )
                df_prices_palladium = df_prices_palladium.dropna(subset=["Date", "Price"]).sort_values(
                    "Date"
                ).reset_index(drop=True)[
                    ["Date", "Price"]
                ]
                df_adj_palladium, used_date_palladium = adjust_prices_for_mode(
                    df_prices_palladium, pmode
                )
            except Exception:
                used_date_palladium = None
            prs = insert_palladium_source(
                prs,
                used_date_palladium,
                pmode,
            )
        except Exception:
            # If Palladium module is unavailable or insertion fails, continue without error
            pass

        # ------------------------------------------------------------------
        # Insert Oil technical analysis slide (commodity)
        # ------------------------------------------------------------------
        try:
            # Insert the Oil chart with call-out and regression channel anchored at oil_anchor_dt
            prs = insert_oil_technical_chart_with_callout(
                prs,
                excel_path_for_ppt,
                oil_anchor_dt,
                price_mode=pmode,
            )
            # Insert Oil technical and momentum scores
            prs = insert_oil_technical_score_number(
                prs,
                excel_path_for_ppt,
            )
            prs = insert_oil_momentum_score_number(
                prs,
                excel_path_for_ppt,
            )
            # Insert Oil subtitle from user input
            prs = insert_oil_subtitle(
                prs,
                st.session_state.get("oil_subtitle", ""),
            )
            # Insert Oil average gauge (last week's average DMAS)
            oil_last_week_avg = st.session_state.get("oil_last_week_avg", 50.0)
            prs = insert_oil_average_gauge(
                prs,
                excel_path_for_ppt,
                oil_last_week_avg,
            )
            # Insert the technical assessment text into the 'oil_view' textbox
            manual_view_oil = st.session_state.get("oil_selected_view")
            prs = insert_oil_technical_assessment(
                prs,
                excel_path_for_ppt,
                manual_desc=manual_view_oil,
            )
            # Compute used date for Oil source footnote
            try:
                import pandas as pd
                df_prices_oil = pd.read_excel(excel_path_for_ppt, sheet_name="data_prices")
                df_prices_oil = df_prices_oil.drop(index=0)
                df_prices_oil = df_prices_oil[
                    df_prices_oil[df_prices_oil.columns[0]] != "DATES"
                ]
                df_prices_oil["Date"] = pd.to_datetime(
                    df_prices_oil[df_prices_oil.columns[0]], errors="coerce"
                )
                # Use the CL1 Comdty column for Oil prices
                df_prices_oil["Price"] = pd.to_numeric(
                    df_prices_oil["CL1 Comdty"], errors="coerce"
                )
                df_prices_oil = df_prices_oil.dropna(subset=["Date", "Price"]).sort_values(
                    "Date"
                ).reset_index(drop=True)[
                    ["Date", "Price"]
                ]
                df_adj_oil, used_date_oil = adjust_prices_for_mode(
                    df_prices_oil, pmode
                )
            except Exception:
                used_date_oil = None
            prs = insert_oil_source(
                prs,
                used_date_oil,
                pmode,
            )
        except Exception:
            # If Oil module is unavailable or insertion fails, continue without error
            pass

        # ------------------------------------------------------------------
        # Insert Copper technical analysis slide (commodity)
        # ------------------------------------------------------------------
        try:
            # Insert the Copper chart with call-out and regression channel anchored at copper_anchor_dt
            prs = insert_copper_technical_chart_with_callout(
                prs,
                excel_path_for_ppt,
                copper_anchor_dt,
                price_mode=pmode,
            )
            # Insert Copper technical and momentum scores
            prs = insert_copper_technical_score_number(
                prs,
                excel_path_for_ppt,
            )
            prs = insert_copper_momentum_score_number(
                prs,
                excel_path_for_ppt,
            )
            # Insert Copper subtitle from user input
            prs = insert_copper_subtitle(
                prs,
                st.session_state.get("copper_subtitle", ""),
            )
            # Insert Copper average gauge (last week's average DMAS)
            copper_last_week_avg = st.session_state.get("copper_last_week_avg", 50.0)
            prs = insert_copper_average_gauge(
                prs,
                excel_path_for_ppt,
                copper_last_week_avg,
            )
            # Insert the technical assessment text into the 'copper_view' textbox
            manual_view_copper = st.session_state.get("copper_selected_view")
            prs = insert_copper_technical_assessment(
                prs,
                excel_path_for_ppt,
                manual_desc=manual_view_copper,
            )
            # Compute used date for Copper source footnote
            try:
                import pandas as pd
                df_prices_copper = pd.read_excel(excel_path_for_ppt, sheet_name="data_prices")
                df_prices_copper = df_prices_copper.drop(index=0)
                df_prices_copper = df_prices_copper[
                    df_prices_copper[df_prices_copper.columns[0]] != "DATES"
                ]
                df_prices_copper["Date"] = pd.to_datetime(
                    df_prices_copper[df_prices_copper.columns[0]], errors="coerce"
                )
                # Use the LP1 Comdty column for Copper prices
                df_prices_copper["Price"] = pd.to_numeric(
                    df_prices_copper["LP1 Comdty"], errors="coerce"
                )
                df_prices_copper = df_prices_copper.dropna(subset=["Date", "Price"]).sort_values(
                    "Date"
                ).reset_index(drop=True)[
                    ["Date", "Price"]
                ]
                df_adj_copper, used_date_copper = adjust_prices_for_mode(
                    df_prices_copper, pmode
                )
            except Exception:
                used_date_copper = None
            prs = insert_copper_source(
                prs,
                used_date_copper,
                pmode,
            )
        except Exception:
            # If Copper module is unavailable or insertion fails, continue without error
            pass

        # ------------------------------------------------------------------
        # Insert Bitcoin technical analysis slide (crypto)
        # ------------------------------------------------------------------
        try:
            # Insert the Bitcoin chart with call-out and regression channel anchored at bitcoin_anchor_dt
            prs = insert_bitcoin_technical_chart_with_callout(
                prs,
                excel_path_for_ppt,
                bitcoin_anchor_dt,
                price_mode=pmode,
            )
            # Insert Bitcoin technical and momentum scores
            prs = insert_bitcoin_technical_score_number(
                prs,
                excel_path_for_ppt,
            )
            prs = insert_bitcoin_momentum_score_number(
                prs,
                excel_path_for_ppt,
            )
            # Insert Bitcoin subtitle from user input
            prs = insert_bitcoin_subtitle(
                prs,
                st.session_state.get("bitcoin_subtitle", ""),
            )
            # Insert Bitcoin average gauge (last week's average DMAS)
            bitcoin_last_week_avg = st.session_state.get("bitcoin_last_week_avg", 50.0)
            prs = insert_bitcoin_average_gauge(
                prs,
                excel_path_for_ppt,
                bitcoin_last_week_avg,
            )
            # Insert the technical assessment text into the 'bitcoin_view' textbox
            manual_view_bitcoin = st.session_state.get("bitcoin_selected_view")
            prs = insert_bitcoin_technical_assessment(
                prs,
                excel_path_for_ppt,
                manual_desc=manual_view_bitcoin,
            )
            # Compute used date for Bitcoin source footnote
            try:
                import pandas as pd
                df_prices_bitcoin = pd.read_excel(excel_path_for_ppt, sheet_name="data_prices")
                df_prices_bitcoin = df_prices_bitcoin.drop(index=0)
                df_prices_bitcoin = df_prices_bitcoin[
                    df_prices_bitcoin[df_prices_bitcoin.columns[0]] != "DATES"
                ]
                df_prices_bitcoin["Date"] = pd.to_datetime(
                    df_prices_bitcoin[df_prices_bitcoin.columns[0]], errors="coerce"
                )
                # Use the XBTUSD Curncy column for Bitcoin prices
                df_prices_bitcoin["Price"] = pd.to_numeric(
                    df_prices_bitcoin["XBTUSD Curncy"], errors="coerce"
                )
                df_prices_bitcoin = df_prices_bitcoin.dropna(subset=["Date", "Price"]).sort_values(
                    "Date"
                ).reset_index(drop=True)[
                    ["Date", "Price"]
                ]
                df_adj_bitcoin, used_date_bitcoin = adjust_prices_for_mode(
                    df_prices_bitcoin, pmode
                )
            except Exception:
                used_date_bitcoin = None
            prs = insert_bitcoin_source(
                prs,
                used_date_bitcoin,
                pmode,
            )
        except Exception:
            # If the Bitcoin module is unavailable or insertion fails, continue without error
            pass

        # ------------------------------------------------------------------
        # Insert Ethereum technical analysis slide (crypto)
        # ------------------------------------------------------------------
        try:
            # Insert the Ethereum chart with call-out and regression channel anchored at ethereum_anchor_dt
            prs = insert_ethereum_technical_chart_with_callout(
                prs,
                excel_path_for_ppt,
                ethereum_anchor_dt,
                price_mode=pmode,
            )
            # Insert Ethereum technical and momentum scores
            prs = insert_ethereum_technical_score_number(
                prs,
                excel_path_for_ppt,
            )
            prs = insert_ethereum_momentum_score_number(
                prs,
                excel_path_for_ppt,
            )
            # Insert Ethereum subtitle from user input
            prs = insert_ethereum_subtitle(
                prs,
                st.session_state.get("ethereum_subtitle", ""),
            )
            # Insert Ethereum average gauge (last week's average DMAS)
            ethereum_last_week_avg = st.session_state.get("ethereum_last_week_avg", 50.0)
            prs = insert_ethereum_average_gauge(
                prs,
                excel_path_for_ppt,
                ethereum_last_week_avg,
            )
            # Insert the technical assessment text into the 'ethereum_view' textbox
            manual_view_ethereum = st.session_state.get("ethereum_selected_view")
            prs = insert_ethereum_technical_assessment(
                prs,
                excel_path_for_ppt,
                manual_desc=manual_view_ethereum,
            )
            # Compute used date for Ethereum source footnote
            try:
                import pandas as pd
                df_prices_ethereum = pd.read_excel(excel_path_for_ppt, sheet_name="data_prices")
                df_prices_ethereum = df_prices_ethereum.drop(index=0)
                df_prices_ethereum = df_prices_ethereum[
                    df_prices_ethereum[df_prices_ethereum.columns[0]] != "DATES"
                ]
                df_prices_ethereum["Date"] = pd.to_datetime(
                    df_prices_ethereum[df_prices_ethereum.columns[0]], errors="coerce"
                )
                # Use the XETUSD Curncy column for Ethereum prices
                df_prices_ethereum["Price"] = pd.to_numeric(
                    df_prices_ethereum["XETUSD Curncy"], errors="coerce"
                )
                df_prices_ethereum = df_prices_ethereum.dropna(subset=["Date", "Price"]).sort_values(
                    "Date"
                ).reset_index(drop=True)[
                    ["Date", "Price"]
                ]
                df_adj_ethereum, used_date_ethereum = adjust_prices_for_mode(
                    df_prices_ethereum, pmode
                )
            except Exception:
                used_date_ethereum = None
            prs = insert_ethereum_source(
                prs,
                used_date_ethereum,
                pmode,
            )
        except Exception:
            # If the Ethereum module is unavailable or insertion fails, continue without error
            pass

        # ------------------------------------------------------------------
        # Insert Ethereum technical analysis slide (crypto)
        # ------------------------------------------------------------------
        try:
            # Insert the Ethereum chart with call-out and regression channel anchored at ethereum_anchor_dt
            prs = insert_ethereum_technical_chart_with_callout(
                prs,
                excel_path_for_ppt,
                ethereum_anchor_dt,
                price_mode=pmode,
            )
            # Insert Ethereum technical and momentum scores
            prs = insert_ethereum_technical_score_number(
                prs,
                excel_path_for_ppt,
            )
            prs = insert_ethereum_momentum_score_number(
                prs,
                excel_path_for_ppt,
            )
            # Insert Ethereum subtitle from user input
            prs = insert_ethereum_subtitle(
                prs,
                st.session_state.get("ethereum_subtitle", ""),
            )
            # Insert Ethereum average gauge (last week's average DMAS)
            ethereum_last_week_avg = st.session_state.get("ethereum_last_week_avg", 50.0)
            prs = insert_ethereum_average_gauge(
                prs,
                excel_path_for_ppt,
                ethereum_last_week_avg,
            )
            # Insert the technical assessment text into the 'ethereum_view' textbox
            manual_view_ethereum = st.session_state.get("ethereum_selected_view")
            prs = insert_ethereum_technical_assessment(
                prs,
                excel_path_for_ppt,
                manual_desc=manual_view_ethereum,
            )
            # Compute used date for Ethereum source footnote
            try:
                import pandas as pd
                df_prices_ethereum = pd.read_excel(excel_path_for_ppt, sheet_name="data_prices")
                df_prices_ethereum = df_prices_ethereum.drop(index=0)
                df_prices_ethereum = df_prices_ethereum[
                    df_prices_ethereum[df_prices_ethereum.columns[0]] != "DATES"
                ]
                df_prices_ethereum["Date"] = pd.to_datetime(
                    df_prices_ethereum[df_prices_ethereum.columns[0]], errors="coerce"
                )
                # Use the XETUSD Curncy column for Ethereum prices
                df_prices_ethereum["Price"] = pd.to_numeric(
                    df_prices_ethereum["XETUSD Curncy"], errors="coerce"
                )
                df_prices_ethereum = df_prices_ethereum.dropna(subset=["Date", "Price"]).sort_values(
                    "Date"
                ).reset_index(drop=True)[
                    ["Date", "Price"]
                ]
                df_adj_ethereum, used_date_ethereum = adjust_prices_for_mode(
                    df_prices_ethereum, pmode
                )
            except Exception:
                used_date_ethereum = None
            prs = insert_ethereum_source(
                prs,
                used_date_ethereum,
                pmode,
            )
        except Exception:
            # If the Ethereum module is unavailable or insertion fails, continue without error
            pass

        # ------------------------------------------------------------------
        # Insert Ripple technical analysis slide (crypto)
        # ------------------------------------------------------------------
        try:
            # Insert the Ripple chart with call-out and regression channel anchored at ripple_anchor_dt
            prs = insert_ripple_technical_chart_with_callout(
                prs,
                excel_path_for_ppt,
                ripple_anchor_dt,
                price_mode=pmode,
            )
            # Insert Ripple technical and momentum scores
            prs = insert_ripple_technical_score_number(
                prs,
                excel_path_for_ppt,
            )
            prs = insert_ripple_momentum_score_number(
                prs,
                excel_path_for_ppt,
            )
            # Insert Ripple subtitle from user input
            prs = insert_ripple_subtitle(
                prs,
                st.session_state.get("ripple_subtitle", ""),
            )
            # Insert Ripple average gauge (last week's average DMAS)
            ripple_last_week_avg = st.session_state.get("ripple_last_week_avg", 50.0)
            prs = insert_ripple_average_gauge(
                prs,
                excel_path_for_ppt,
                ripple_last_week_avg,
            )
            # Insert the technical assessment text into the 'ripple_view' textbox
            manual_view_ripple = st.session_state.get("ripple_selected_view")
            prs = insert_ripple_technical_assessment(
                prs,
                excel_path_for_ppt,
                manual_desc=manual_view_ripple,
            )
            # Compute used date for Ripple source footnote
            try:
                import pandas as pd
                df_prices_ripple = pd.read_excel(excel_path_for_ppt, sheet_name="data_prices")
                df_prices_ripple = df_prices_ripple.drop(index=0)
                df_prices_ripple = df_prices_ripple[
                    df_prices_ripple[df_prices_ripple.columns[0]] != "DATES"
                ]
                df_prices_ripple["Date"] = pd.to_datetime(
                    df_prices_ripple[df_prices_ripple.columns[0]], errors="coerce"
                )
                # Use the XRPUSD Curncy column for Ripple prices
                df_prices_ripple["Price"] = pd.to_numeric(
                    df_prices_ripple["XRPUSD Curncy"], errors="coerce"
                )
                df_prices_ripple = df_prices_ripple.dropna(subset=["Date", "Price"]).sort_values(
                    "Date"
                ).reset_index(drop=True)[
                    ["Date", "Price"]
                ]
                df_adj_ripple, used_date_ripple = adjust_prices_for_mode(
                    df_prices_ripple, pmode
                )
            except Exception:
                used_date_ripple = None
            prs = insert_ripple_source(
                prs,
                used_date_ripple,
                pmode,
            )
        except Exception:
            # If the Ripple module is unavailable or insertion fails, continue without error
            pass

        # ------------------------------------------------------------------
        # Insert Solana technical analysis slide (crypto)
        # ------------------------------------------------------------------
        try:
            # Insert the Solana chart with call-out and regression channel anchored at solana_anchor_dt
            prs = insert_solana_technical_chart_with_callout(
                prs,
                excel_path_for_ppt,
                solana_anchor_dt,
                price_mode=pmode,
            )
            # Insert Solana technical and momentum scores
            prs = insert_solana_technical_score_number(
                prs,
                excel_path_for_ppt,
            )
            prs = insert_solana_momentum_score_number(
                prs,
                excel_path_for_ppt,
            )
            # Insert Solana subtitle from user input
            prs = insert_solana_subtitle(
                prs,
                st.session_state.get("solana_subtitle", ""),
            )
            # Insert Solana average gauge (last week's average DMAS)
            solana_last_week_avg = st.session_state.get("solana_last_week_avg", 50.0)
            prs = insert_solana_average_gauge(
                prs,
                excel_path_for_ppt,
                solana_last_week_avg,
            )
            # Insert the technical assessment text into the 'solana_view' textbox
            manual_view_solana = st.session_state.get("solana_selected_view")
            prs = insert_solana_technical_assessment(
                prs,
                excel_path_for_ppt,
                manual_desc=manual_view_solana,
            )
            # Compute used date for Solana source footnote
            try:
                import pandas as pd
                df_prices_solana = pd.read_excel(excel_path_for_ppt, sheet_name="data_prices")
                df_prices_solana = df_prices_solana.drop(index=0)
                df_prices_solana = df_prices_solana[df_prices_solana[df_prices_solana.columns[0]] != "DATES"]
                df_prices_solana["Date"] = pd.to_datetime(
                    df_prices_solana[df_prices_solana.columns[0]], errors="coerce"
                )
                # Use the XSOUSD Curncy column for Solana prices
                df_prices_solana["Price"] = pd.to_numeric(
                    df_prices_solana["XSOUSD Curncy"], errors="coerce"
                )
                df_prices_solana = df_prices_solana.dropna(subset=["Date", "Price"]).sort_values(
                    "Date"
                ).reset_index(drop=True)[["Date", "Price"]]
                df_adj_solana, used_date_solana = adjust_prices_for_mode(
                    df_prices_solana, pmode
                )
            except Exception:
                used_date_solana = None
            prs = insert_solana_source(
                prs,
                used_date_solana,
                pmode,
            )
        except Exception:
            # If the Solana module is unavailable or insertion fails, continue without error
            pass

        # ------------------------------------------------------------------
        # Insert Binance technical analysis slide (crypto)
        # ------------------------------------------------------------------
        try:
            # Insert the Binance chart with call-out and regression channel anchored at binance_anchor_dt
            prs = insert_binance_technical_chart_with_callout(
                prs,
                excel_path_for_ppt,
                binance_anchor_dt,
                price_mode=pmode,
            )
            # Insert Binance technical and momentum scores
            prs = insert_binance_technical_score_number(
                prs,
                excel_path_for_ppt,
            )
            prs = insert_binance_momentum_score_number(
                prs,
                excel_path_for_ppt,
            )
            # Insert Binance subtitle from user input
            prs = insert_binance_subtitle(
                prs,
                st.session_state.get("binance_subtitle", ""),
            )
            # Insert Binance average gauge (last week's average DMAS)
            binance_last_week_avg = st.session_state.get("binance_last_week_avg", 50.0)
            prs = insert_binance_average_gauge(
                prs,
                excel_path_for_ppt,
                binance_last_week_avg,
            )
            # Insert the technical assessment text into the 'binance_view' textbox
            manual_view_binance = st.session_state.get("binance_selected_view")
            prs = insert_binance_technical_assessment(
                prs,
                excel_path_for_ppt,
                manual_desc=manual_view_binance,
            )
            # Compute used date for Binance source footnote
            try:
                import pandas as pd
                df_prices_binance = pd.read_excel(excel_path_for_ppt, sheet_name="data_prices")
                df_prices_binance = df_prices_binance.drop(index=0)
                df_prices_binance = df_prices_binance[df_prices_binance[df_prices_binance.columns[0]] != "DATES"]
                df_prices_binance["Date"] = pd.to_datetime(
                    df_prices_binance[df_prices_binance.columns[0]], errors="coerce"
                )
                # Use the XBIUSD Curncy column for Binance prices
                df_prices_binance["Price"] = pd.to_numeric(
                    df_prices_binance["XBIUSD Curncy"], errors="coerce"
                )
                df_prices_binance = df_prices_binance.dropna(subset=["Date", "Price"]).sort_values(
                    "Date"
                ).reset_index(drop=True)[["Date", "Price"]]
                df_adj_binance, used_date_binance = adjust_prices_for_mode(
                    df_prices_binance, pmode
                )
            except Exception:
                used_date_binance = None
            prs = insert_binance_source(
                prs,
                used_date_binance,
                pmode,
            )
        except Exception:
            # If the Binance module is unavailable or insertion fails, continue without error
            pass

        # When CSI 300 is the selected index, the technical analysis slides
        # for CSI have already been inserted in the branch above.  Avoid
        # inserting CSI slides again here.  Likewise, when SPX is selected,
        # CSI slides are not inserted at all.  This prevents duplicate
        # insertion of CSI slides that could override SPX content or leave
        # placeholders empty.

        # ------------------------------------------------------------------
        # Insert Equity performance charts
        # ------------------------------------------------------------------
        try:
            # Generate the weekly performance bar chart with price-mode adjustment
            bar_bytes, perf_used_date = create_weekly_performance_chart(
                excel_path_for_ppt,
                price_mode=st.session_state.get("price_mode", "Last Price"),
            )
            prs = insert_equity_performance_bar_slide(
                prs,
                bar_bytes,
                used_date=perf_used_date,
                price_mode=st.session_state.get("price_mode", "Last Price"),
                left_cm=1.63,
                top_cm=4.73,
                width_cm=22.48,
                height_cm=10.61,
            )
            # Generate the historical performance heatmap with price-mode adjustment
            histo_bytes, histo_used_date = create_historical_performance_table(
                excel_path_for_ppt,
                price_mode=st.session_state.get("price_mode", "Last Price"),
            )
            prs = insert_equity_performance_histo_slide(
                prs,
                histo_bytes,
                used_date=histo_used_date,
                price_mode=st.session_state.get("price_mode", "Last Price"),
                left_cm=2.16,
                top_cm=4.70,
                width_cm=19.43,
                height_cm=10.61,
            )

            # ------------------------------------------------------------------
            # Insert FX performance charts
            # ------------------------------------------------------------------
            # Generate the weekly FX performance bar chart with price-mode adjustment
            fx_bar_bytes, fx_used_date = create_weekly_fx_performance_chart(
                excel_path_for_ppt,
                price_mode=st.session_state.get("price_mode", "Last Price"),
            )
            prs = insert_fx_performance_bar_slide(
                prs,
                fx_bar_bytes,
                used_date=fx_used_date,
                price_mode=st.session_state.get("price_mode", "Last Price"),
                left_cm=1.63,
                top_cm=4.73,
                width_cm=22.48,
                height_cm=10.61,
            )

            # Generate the FX historical performance heatmap with price-mode adjustment
            fx_histo_bytes, fx_used_date2 = create_historical_fx_performance_table(
                excel_path_for_ppt,
                price_mode=st.session_state.get("price_mode", "Last Price"),
            )
            prs = insert_fx_performance_histo_slide(
                prs,
                fx_histo_bytes,
                used_date=fx_used_date2,
                price_mode=st.session_state.get("price_mode", "Last Price"),
                left_cm=2.16,
                top_cm=4.70,
                width_cm=19.43,
                height_cm=10.61,
            )

            # ------------------------------------------------------------------
            # Insert cryptocurrency performance charts
            # ------------------------------------------------------------------
            # Generate the weekly crypto performance bar chart with price-mode adjustment
            crypto_bar_bytes, crypto_used_date = create_weekly_crypto_performance_chart(
                excel_path_for_ppt,
                price_mode=st.session_state.get("price_mode", "Last Price"),
            )
            prs = insert_crypto_performance_bar_slide(
                prs,
                crypto_bar_bytes,
                used_date=crypto_used_date,
                price_mode=st.session_state.get("price_mode", "Last Price"),
                left_cm=1.63,
                top_cm=4.73,
                width_cm=22.48,
                height_cm=10.61,
            )

            # Generate the cryptocurrency historical performance heatmap with price-mode adjustment
            crypto_histo_bytes, crypto_used_date2 = create_historical_crypto_performance_table(
                excel_path_for_ppt,
                price_mode=st.session_state.get("price_mode", "Last Price"),
            )
            prs = insert_crypto_performance_histo_slide(
                prs,
                crypto_histo_bytes,
                used_date=crypto_used_date2,
                price_mode=st.session_state.get("price_mode", "Last Price"),
                left_cm=2.16,
                top_cm=4.70,
                width_cm=19.43,
                height_cm=10.61,
            )

            # ------------------------------------------------------------------
            # Insert Rates performance charts
            # ------------------------------------------------------------------
            # Generate the weekly rates performance bar chart with price-mode adjustment
            rates_bar_bytes, rates_used_date = create_weekly_rates_performance_chart(
                excel_path_for_ppt,
                price_mode=st.session_state.get("price_mode", "Last Price"),
            )
            prs = insert_rates_performance_bar_slide(
                prs,
                rates_bar_bytes,
                used_date=rates_used_date,
                price_mode=st.session_state.get("price_mode", "Last Price"),
                left_cm=1.63,
                top_cm=4.73,
                width_cm=22.48,
                height_cm=10.61,
            )

            # Generate the rates historical performance heatmap with price-mode adjustment
            rates_histo_bytes, rates_used_date2 = create_historical_rates_performance_table(
                excel_path_for_ppt,
                price_mode=st.session_state.get("price_mode", "Last Price"),
            )
            prs = insert_rates_performance_histo_slide(
                prs,
                rates_histo_bytes,
                used_date=rates_used_date2,
                price_mode=st.session_state.get("price_mode", "Last Price"),
                left_cm=2.16,
                top_cm=4.70,
                width_cm=19.43,
                height_cm=10.61,
            )

            # ------------------------------------------------------------------
            # Insert Credit performance charts
            # ------------------------------------------------------------------
            # Generate the weekly credit performance bar chart with price-mode adjustment
            credit_bar_bytes, credit_used_date = create_weekly_credit_performance_chart(
                excel_path_for_ppt,
                price_mode=st.session_state.get("price_mode", "Last Price"),
            )
            prs = insert_credit_performance_bar_slide(
                prs,
                credit_bar_bytes,
                used_date=credit_used_date,
                price_mode=st.session_state.get("price_mode", "Last Price"),
                left_cm=1.63,
                top_cm=4.73,
                width_cm=22.48,
                height_cm=10.61,
            )

            # Generate the credit historical performance heatmap with price-mode adjustment
            credit_histo_bytes, credit_used_date2 = create_historical_credit_performance_table(
                excel_path_for_ppt,
                price_mode=st.session_state.get("price_mode", "Last Price"),
            )
            prs = insert_credit_performance_histo_slide(
                prs,
                credit_histo_bytes,
                used_date=credit_used_date2,
                price_mode=st.session_state.get("price_mode", "Last Price"),
                left_cm=2.16,
                top_cm=4.70,
                width_cm=19.43,
                height_cm=10.61,
            )

            # ------------------------------------------------------------------
            # Insert Commodity performance charts
            # ------------------------------------------------------------------
            # Generate the weekly commodity performance bar chart with price-mode adjustment
            commo_bar_bytes, commo_used_date = create_weekly_commodity_performance_chart(
                excel_path_for_ppt,
                price_mode=st.session_state.get("price_mode", "Last Price"),
            )
            prs = insert_commodity_performance_bar_slide(
                prs,
                commo_bar_bytes,
                used_date=commo_used_date,
                price_mode=st.session_state.get("price_mode", "Last Price"),
                left_cm=1.63,
                top_cm=4.73,
                width_cm=22.48,
                height_cm=10.61,
            )

            # Generate the commodity historical performance heatmap with price-mode adjustment
            commo_histo_bytes, commo_used_date2 = create_historical_commodity_performance_table(
                excel_path_for_ppt,
                price_mode=st.session_state.get("price_mode", "Last Price"),
            )
            prs = insert_commodity_performance_histo_slide(
                prs,
                commo_histo_bytes,
                used_date=commo_used_date2,
                price_mode=st.session_state.get("price_mode", "Last Price"),
                left_cm=2.16,
                top_cm=4.70,
                width_cm=19.43,
                height_cm=10.61,
            )
        except Exception:
            # If anything fails, continue without the performance slides
            pass

        # ------------------------------------------------------------------
        # Insert market breadth (funda breath) table
        # ------------------------------------------------------------------
        
        from pptx.util import Cm
        def _paste_breadth_test_picture(prs, img_path):
            for slide in prs.slides:
                ph = next(
                    (s for s in slide.shapes
                    if (s.has_text_frame and s.text_frame.text.strip().lower() == "funda_breadth")
                    or s.name.lower().startswith("funda_breadth")),
                    None,
                )
                if ph:
                    pic = slide.shapes.add_picture(
                        str(img_path),
                        Cm(2.58), Cm(5.38),      # position
                        Cm(19.55), Cm(4.25)      # size
                    )
                    pic.alt_text = "tcignore breadth image"
                    break
            return prs


        try:
            prs = insert_funda_breath_table(prs, excel_path_for_ppt)
        except Exception:
            # Ignore errors to avoid breaking presentation generation
            pass

        out_stream = BytesIO()
        try:
            img_path = _copy_breadth_test_range(Path(excel_path_for_ppt))
            prs = _paste_breadth_test_picture(prs, img_path)
        except Exception as e:
            print("Breadth‑test step failed:", e)
        prs.save(out_stream)
        out_stream.seek(0)
        updated_bytes = out_stream.getvalue()

        # Always generate a macro‑free PowerPoint (.pptx).  Converting a
        # macro‑enabled template (.pptm) to .pptx removes any embedded VBA
        # projects and prevents runtime errors when opening the file.  The
        # MIME type for .pptx files is used for all downloads.
        fname = "updated_presentation.pptx"
        mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

        st.sidebar.success("Updated presentation created successfully.")
        st.sidebar.download_button(
            "Download updated presentation",
            data=updated_bytes,
            file_name=fname,
            mime=mime,
            key="download_ppt_button",
        )
    st.write("Click the button in the sidebar to generate your updated presentation.")


# -----------------------------------------------------------------------------
# Main navigation dispatch
# -----------------------------------------------------------------------------
if page == "Upload":
    show_upload_page()
elif page == "YTD Update":
    show_ytd_update_page()
elif page == "Technical Analysis":
    show_technical_analysis_page()
elif page == "Market Breadth":
    show_market_breadth_page()
elif page == "Generate Presentation":
    show_generate_presentation_page()