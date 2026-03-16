"""
Full Slide HTML Template for High-Quality PNG Export.

This module provides a complete slide template that includes:
- Navy banner with category text
- Herculis logo
- Gold accent bar
- Title and subtitle
- Chart area (embeds existing chart HTML)
- Source footer

The template renders at 4x scale (3840x2400px) for crisp output,
bypassing PowerPoint compression entirely.

Slide Specifications:
- Width: 25.40 cm (960px at 1x, 3840px at 4x)
- Height: 15.88 cm (600px at 1x, 2400px at 4x)
- Aspect ratio: 16:9
"""

from typing import Optional
import base64
import os

# =============================================================================
# FULL SLIDE CSS
# =============================================================================

FULL_SLIDE_CSS = '''
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;1,400;1,500&family=Inter:wght@400;500;600;700&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    width: {{ 960 * scale }}px;
    height: {{ 600 * scale }}px;
    background: white;
    position: relative;
    font-family: 'Inter', sans-serif;
    overflow: hidden;
}

/* Navy Banner */
.banner {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: {{ 52 * scale }}px;
    background: linear-gradient(135deg, #1a365d 0%, #1e3a5f 50%, #1a365d 100%);
}

.banner-text {
    position: absolute;
    left: {{ 15 * scale }}px;
    top: 50%;
    transform: translateY(-50%);
    font-family: 'Playfair Display', Georgia, serif;
    font-style: italic;
    font-size: {{ 26 * scale }}px;
    color: white;
    letter-spacing: 0.5px;
}

/* Logo */
.logo {
    position: absolute;
    right: {{ 10 * scale }}px;
    top: {{ 2 * scale }}px;
    width: {{ 120 * scale }}px;
    height: auto;
}

/* Text-based logo fallback */
.logo-text {
    position: absolute;
    right: {{ 15 * scale }}px;
    top: 50%;
    transform: translateY(-50%);
    font-family: 'Playfair Display', Georgia, serif;
    font-style: italic;
    font-size: {{ 18 * scale }}px;
    color: #c9a227;
    letter-spacing: 1px;
}

/* Gold accent bar */
.gold-bar {
    position: absolute;
    left: {{ 43 * scale }}px;
    top: {{ 93 * scale }}px;
    width: {{ 4 * scale }}px;
    height: {{ 77 * scale }}px;
    background: #c9a227;
    border-radius: {{ 2 * scale }}px;
}

/* Title */
.title {
    position: absolute;
    left: {{ 56 * scale }}px;
    top: {{ 90 * scale }}px;
    font-family: 'Playfair Display', Georgia, serif;
    font-style: italic;
    font-size: {{ 30 * scale }}px;
    color: #c9a227;
}

/* Subtitle */
.subtitle {
    position: absolute;
    left: {{ 56 * scale }}px;
    top: {{ 136 * scale }}px;
    font-family: 'Inter', sans-serif;
    font-size: {{ 13 * scale }}px;
    font-weight: 400;
    color: #1a365d;
    max-width: {{ 850 * scale }}px;
    line-height: 1.4;
}

/* Chart container */
.chart-container {
    position: absolute;
    left: {{ 43 * scale }}px;
    top: {{ 181 * scale }}px;
    width: {{ 895 * scale }}px;
    height: {{ 394 * scale }}px;
    overflow: hidden;
}

/* Source footer */
.source {
    position: absolute;
    left: {{ 43 * scale }}px;
    top: {{ 575 * scale }}px;
    font-family: 'Inter', sans-serif;
    font-size: {{ 9 * scale }}px;
    color: #94a3b8;
}
'''

# =============================================================================
# FULL SLIDE HTML BODY
# =============================================================================

FULL_SLIDE_HTML_BODY = '''
<body>
    <!-- Navy Banner -->
    <div class="banner">
        <span class="banner-text">{{ category }}</span>
        {% if logo_base64 %}
        <img class="logo" src="data:image/png;base64,{{ logo_base64 }}" alt="Herculis">
        {% else %}
        <span class="logo-text">HERCULIS</span>
        {% endif %}
    </div>

    <!-- Gold accent bar -->
    <div class="gold-bar"></div>

    <!-- Title -->
    <div class="title">{{ instrument }}: {{ view }}</div>

    <!-- Subtitle -->
    <div class="subtitle">{{ subtitle }}</div>

    <!-- Chart (embedded from existing template) -->
    <div class="chart-container">
        {{ chart_html | safe }}
    </div>

    <!-- Source -->
    <div class="source">Source: Bloomberg, Herculis Group. Data as of {{ date }}</div>
</body>
'''

# =============================================================================
# CHART-ONLY CSS (embedded within chart container)
# =============================================================================

CHART_ONLY_CSS = '''
@import url('https://fonts.cdnfonts.com/css/calibri-light');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

.chart-wrapper {
    font-family: 'Calibri', Calibri, 'Segoe UI', Arial, sans-serif;
    background: radial-gradient(
        ellipse 140% 140% at 20% 50%,
        #FFFFFF 0%,
        #FFFFFF 40%,
        #F4F7FA 55%,
        #E8EEF4 70%,
        #D8E2EC 85%,
        #C8D4E2 100%
    );
    width: 100%;
    height: 100%;
    padding: 0;
    border-radius: 8px;
}

.main-container {
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 100%;
}

.price-row {
    display: flex;
    height: 300px;
}

.price-chart-area {
    flex: 1;
    position: relative;
    padding: 8px;
    padding-left: 5px;
    padding-right: 10px;
    margin-left: 0;
    background: linear-gradient(90deg,
        rgba(255,255,255,0.95) 0%,
        rgba(255,255,255,0.85) 70%,
        rgba(248,250,252,0.75) 90%,
        rgba(240,244,248,0.6) 100%
    );
    border: 1px solid #D1D9E6;
    border-right: none;
    border-radius: 8px 0 0 0;
    box-shadow: inset -20px 0 30px -15px rgba(27, 58, 90, 0.08);
}

.price-chart-container {
    position: relative;
    width: 100%;
    height: 100%;
}

.dmas-panel {
    width: 200px;
    min-width: 200px;
    background: linear-gradient(180deg, #1B3A5A 0%, #152D45 100%);
    padding: 12px 15px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    border: 1px solid #1B3A5A;
    border-left: none;
    border-radius: 0 8px 0 0;
    box-shadow: -8px 0 20px -5px rgba(27, 58, 90, 0.25);
}

.panel-title {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: rgba(255, 255, 255, 0.6);
    text-align: center;
    margin-bottom: 2px;
}

.dmas-value {
    font-size: 36px;
    font-weight: 700;
    color: #FFFFFF;
    text-align: center;
    line-height: 1;
}

.dmas-bar {
    height: 6px;
    background: rgba(255, 255, 255, 0.15);
    border-radius: 3px;
    position: relative;
    margin: 6px 0;
}

.dmas-marker {
    position: absolute;
    width: 4px;
    height: 12px;
    background: #c9a227;
    border-radius: 2px;
    top: -3px;
    transform: translateX(-50%);
}

.dmas-change {
    font-size: 11px;
    text-align: center;
    margin-bottom: 6px;
}

.sub-score-section {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 6px;
    padding: 8px;
}

.sub-score-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}

.sub-score-label {
    font-size: 9px;
    color: rgba(255, 255, 255, 0.7);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.sub-score-value {
    font-size: 16px;
    font-weight: 700;
    color: #FFFFFF;
}

.score-trend {
    font-size: 10px;
    margin-right: 4px;
}

.sub-score-bar {
    height: 4px;
    background: rgba(255, 255, 255, 0.15);
    border-radius: 2px;
    margin: 4px 0;
}

.sub-score-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s ease;
}

.sub-score-status {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
}

.panel-footnote {
    font-size: 7px;
    color: rgba(255, 255, 255, 0.4);
    text-align: center;
    line-height: 1.3;
    margin-top: auto;
    padding-top: 6px;
}

.rsi-row {
    display: flex;
    height: 90px;
    flex-shrink: 0;
}

.rsi-chart-area {
    flex: 1;
    position: relative;
    padding: 0px 10px 8px 5px;
    background: linear-gradient(90deg,
        rgba(255,255,255,0.95) 0%,
        rgba(255,255,255,0.85) 70%,
        rgba(248,250,252,0.75) 90%,
        rgba(240,244,248,0.6) 100%
    );
    border: 1px solid #D1D9E6;
    border-top: none;
    border-right: none;
    border-radius: 0 0 0 8px;
    box-shadow: inset -20px 0 30px -15px rgba(27, 58, 90, 0.08);
}

.rsi-chart-container {
    position: relative;
    width: 100%;
    height: 100%;
}

.rsi-panel {
    width: 200px;
    min-width: 200px;
    background: linear-gradient(180deg, #152D45 0%, #0F2132 100%);
    padding: 8px 15px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    border: 1px solid #1B3A5A;
    border-left: none;
    border-top: none;
    border-radius: 0 0 8px 0;
}

.chart-legend {
    position: absolute;
    top: 5px;
    left: 5px;
    right: 10px;
    display: flex;
    justify-content: center;
    gap: 20px;
    font-size: 10px;
    z-index: 10;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
}

.legend-line {
    width: 20px;
    height: 3px;
    border-radius: 1.5px;
}

.legend-text {
    color: #040C38;
    font-weight: bold;
}
'''

# =============================================================================
# CATEGORY MAPPING
# =============================================================================

INSTRUMENT_CATEGORIES = {
    # Equity
    'spx': 'Equity',
    's&p 500': 'Equity',
    'dax': 'Equity',
    'smi': 'Equity',
    'nikkei': 'Equity',
    'nikkei 225': 'Equity',
    'sensex': 'Equity',
    'csi 300': 'Equity',
    'csi': 'Equity',
    'ibov': 'Equity',
    'ibovespa': 'Equity',
    'mexbol': 'Equity',
    'tasi': 'Equity',

    # Commodities
    'gold': 'Commodities',
    'silver': 'Commodities',
    'oil': 'Commodities',
    'brent': 'Commodities',
    'copper': 'Commodities',
    'platinum': 'Commodities',
    'palladium': 'Commodities',

    # Crypto
    'bitcoin': 'Crypto',
    'btc': 'Crypto',
    'ethereum': 'Crypto',
    'eth': 'Crypto',
    'solana': 'Crypto',
    'sol': 'Crypto',
    'ripple': 'Crypto',
    'xrp': 'Crypto',
    'binance': 'Crypto',
    'bnb': 'Crypto',
}


def get_category(instrument: str) -> str:
    """Get the category for an instrument."""
    return INSTRUMENT_CATEGORIES.get(instrument.lower(), 'Markets')


# =============================================================================
# TEMPLATE ASSEMBLY
# =============================================================================

def build_full_slide_template() -> str:
    """
    Assemble the full slide HTML template.

    Returns
    -------
    str
        Complete HTML template ready for Jinja2 rendering.
    """
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation"></script>
    <style>
{FULL_SLIDE_CSS}
    </style>
</head>
{FULL_SLIDE_HTML_BODY}
</html>
'''


def get_logo_base64(logo_path: Optional[str] = None) -> Optional[str]:
    """
    Load logo image and return as base64 string.

    Parameters
    ----------
    logo_path : str, optional
        Path to the logo image file.

    Returns
    -------
    str or None
        Base64-encoded image data, or None if file not found.
    """
    if logo_path is None:
        # Try default locations
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'herculis_logo.png'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'logo.png'),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                logo_path = path
                break

    if logo_path and os.path.exists(logo_path):
        with open(logo_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    return None


# Pre-built template
FULL_SLIDE_HTML_TEMPLATE = build_full_slide_template()
