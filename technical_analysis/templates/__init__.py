"""
Technical Analysis Templates Module.

This module provides HTML templates for generating technical analysis charts
using Chart.js and Playwright for high-resolution PNG export.

Available templates:
- TECHNICAL_ANALYSIS_V2_HTML_TEMPLATE: Full HTML template for SPX v2 charts
- FULL_SLIDE_HTML_TEMPLATE: Complete slide template for high-quality PNG export

Available functions:
- render_full_slide: Render complete slide as high-quality PNG
- render_test_slide: Render test slide with sample data
"""

from technical_analysis.templates.technical_analysis_v2 import (
    TECHNICAL_ANALYSIS_V2_HTML_TEMPLATE,
    TECH_V2_CSS,
    TECH_V2_HTML_BODY,
    TECH_V2_JAVASCRIPT,
    build_technical_analysis_v2_template,
    # Full slide wrapper functions
    build_full_slide_template,
    get_category_for_ticker,
    get_display_name_for_ticker,
)

from technical_analysis.templates.full_slide_template import (
    FULL_SLIDE_HTML_TEMPLATE,
    FULL_SLIDE_CSS,
    FULL_SLIDE_HTML_BODY,
    get_category,
    get_logo_base64,
)

from technical_analysis.templates.full_slide_renderer import (
    render_full_slide,
    render_test_slide,
    prepare_chart_data,
    build_full_slide_html,
    get_display_name,
)

__all__ = [
    # V2 template
    'TECHNICAL_ANALYSIS_V2_HTML_TEMPLATE',
    'TECH_V2_CSS',
    'TECH_V2_HTML_BODY',
    'TECH_V2_JAVASCRIPT',
    'build_technical_analysis_v2_template',
    # Full slide wrapper (simplified)
    'build_full_slide_template',
    'get_category_for_ticker',
    'get_display_name_for_ticker',
    # Full slide template (legacy - can be removed)
    'FULL_SLIDE_HTML_TEMPLATE',
    'FULL_SLIDE_CSS',
    'FULL_SLIDE_HTML_BODY',
    'get_category',
    'get_logo_base64',
    # Full slide renderer (legacy - can be removed)
    'render_full_slide',
    'render_test_slide',
    'prepare_chart_data',
    'build_full_slide_html',
    'get_display_name',
]
