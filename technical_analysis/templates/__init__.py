"""
Technical Analysis Templates Module.

This module provides HTML templates for generating technical analysis charts
using Chart.js and Playwright for high-resolution PNG export.

Available templates:
- TECHNICAL_ANALYSIS_V2_HTML_TEMPLATE: Full HTML template for SPX v2 charts
"""

from technical_analysis.templates.technical_analysis_v2 import (
    TECHNICAL_ANALYSIS_V2_HTML_TEMPLATE,
    TECH_V2_CSS,
    TECH_V2_HTML_BODY,
    TECH_V2_JAVASCRIPT,
    build_technical_analysis_v2_template,
)

__all__ = [
    'TECHNICAL_ANALYSIS_V2_HTML_TEMPLATE',
    'TECH_V2_CSS',
    'TECH_V2_HTML_BODY',
    'TECH_V2_JAVASCRIPT',
    'build_technical_analysis_v2_template',
]
