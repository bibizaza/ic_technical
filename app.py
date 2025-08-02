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
    start = today - pd.Timedelta(days=365)
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
    "Select page", ["Upload", "YTD Update", "Technical Analysis", "Generate Presentation"]
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
            "Shenzen CSI 300",
            "Nikkei 225",
            "TASI",
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
        for name in ["Gold", "Silver", "Oil (WTI)", "Platinum", "Copper", "Uranium"]
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
        for name in ["Ripple", "Bitcoin", "Binance", "Ethereum", "Solana"]
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

    # Provide a clear channel button to reset the regression channel for both indices
    if st.sidebar.button("Clear channel", key="ta_clear_global"):
        # Remove stored anchors for all indices if present
        for key in ["spx_anchor", "csi_anchor", "nikkei_anchor", "tasi_anchor", "sensex_anchor", "dax_anchor", "smi_anchor", "ibov_anchor"]:
            if key in st.session_state:
                st.session_state.pop(key)
        st.experimental_rerun()

    excel_available = "excel_file" in st.session_state

    if asset_class == "Equity":
        # Allow the user to select which equity index they wish to analyse.  We
        # provide two options: S&P 500 and CSI 300.  The selection is stored
        # in session state to persist across reruns.
        # Provide index options.  Add Nikkei 225 alongside SPX and CSI.
        # Include SMI (Swiss Market Index) alongside existing indices
        # Include IBOV (Brazil Bovespa) alongside existing indices
        index_options = ["S&P 500", "CSI 300", "Nikkei 225", "TASI", "Sensex", "Dax", "SMI", "Ibov"]
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
                    elif selected_index == "SMI":
                        upper_bound, lower_bound = _compute_range_bounds_smi(df_full, lookback_days=90)
                    elif selected_index == "Ibov":
                        upper_bound, lower_bound = _compute_range_bounds_ibov(df_full, lookback_days=90)
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
                default_anchor = st.session_state.get(
                    f"{ticker_key}_anchor",
                    (max_date - pd.Timedelta(days=180)),
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
    else:
        # Fallback for unsupported asset classes (e.g. Crypto)
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
    index_options = ["Gold", "Silver", "Platinum", "Oil", "Copper"]
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
            default_anchor = st.session_state.get(
                f"{ticker_key}_anchor",
                (max_date - pd.Timedelta(days=180)),
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
        # Anchor for Gold regression channel (commodity)
        gold_anchor_dt = st.session_state.get("gold_anchor")
        # Anchor for Silver regression channel (commodity)
        silver_anchor_dt = st.session_state.get("silver_anchor")
        # Anchor for Platinum regression channel (commodity)
        platinum_anchor_dt = st.session_state.get("platinum_anchor")
        # Anchor for Oil regression channel (commodity)
        oil_anchor_dt = st.session_state.get("oil_anchor")
        # Anchor for Copper regression channel (commodity)
        copper_anchor_dt = st.session_state.get("copper_anchor")

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

        out_stream = BytesIO()
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
elif page == "Generate Presentation":
    show_generate_presentation_page()