"""
Full Slide Renderer for High-Quality PNG Export.

This module renders complete technical analysis slides as high-quality PNG images,
bypassing PowerPoint compression entirely.

Usage:
    from technical_analysis.templates.full_slide_renderer import render_full_slide

    # Render a Gold slide
    png_bytes = render_full_slide(
        instrument='gold',
        view='Bullish',
        subtitle='Impressive setup points to potential for sustained uptrend...',
        excel_path='data.xlsx',
        output_path='exports/gold_technical.png',  # Optional: save to file
        scale=4,  # 4x = 3840x2400px output
    )

Slide Specifications:
- Base: 960 × 600 px (16:9)
- At 4x scale: 3840 × 2400 px
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from io import BytesIO
from typing import Optional, Tuple, Dict, Any

import pandas as pd
from jinja2 import Environment, BaseLoader

from playwright.sync_api import sync_playwright

from technical_analysis.templates.full_slide_template import (
    FULL_SLIDE_CSS,
    FULL_SLIDE_HTML_BODY,
    CHART_ONLY_CSS,
    get_category,
    get_logo_base64,
)
from technical_analysis.templates.technical_analysis_v2 import (
    TECH_V2_HTML_BODY,
    TECH_V2_JAVASCRIPT,
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Base slide dimensions (1x scale)
SLIDE_BASE_WIDTH = 960
SLIDE_BASE_HEIGHT = 600

# Default scale factor for high-quality output
DEFAULT_SCALE = 4

# Chart dimensions within the slide (at 1x) - exact from PPTX
# 23.67cm = 3578px at 4x → 894.5 at 1x (rounded to 895)
# 10.50cm = 1588px at 4x → 397 at 1x
CHART_WIDTH = 895
CHART_HEIGHT = 397


# =============================================================================
# INSTRUMENT DISPLAY NAMES
# =============================================================================

INSTRUMENT_DISPLAY_NAMES = {
    # Equity
    'spx': 'S&P 500',
    's&p 500': 'S&P 500',
    'dax': 'DAX',
    'smi': 'SMI',
    'nikkei': 'Nikkei 225',
    'nikkei 225': 'Nikkei 225',
    'sensex': 'Sensex',
    'csi': 'CSI 300',
    'csi 300': 'CSI 300',
    'ibov': 'Ibovespa',
    'ibovespa': 'Ibovespa',
    'mexbol': 'MEXBOL',
    'tasi': 'TASI',

    # Commodities
    'gold': 'Gold',
    'silver': 'Silver',
    'oil': 'Brent Oil',
    'brent': 'Brent Oil',
    'copper': 'Copper',
    'platinum': 'Platinum',
    'palladium': 'Palladium',

    # Crypto
    'bitcoin': 'Bitcoin',
    'btc': 'Bitcoin',
    'ethereum': 'Ethereum',
    'eth': 'Ethereum',
    'solana': 'Solana',
    'sol': 'Solana',
    'ripple': 'Ripple',
    'xrp': 'Ripple',
    'binance': 'Binance Coin',
    'bnb': 'Binance Coin',
}


def get_display_name(instrument: str) -> str:
    """Get the display name for an instrument."""
    return INSTRUMENT_DISPLAY_NAMES.get(instrument.lower(), instrument.title())


# =============================================================================
# CHART DATA PREPARATION
# =============================================================================

def prepare_chart_data(
    df: pd.DataFrame,
    lookback_days: int = 90,
    dmas_score: int = 50,
    technical_score: int = 50,
    momentum_score: int = 50,
    rsi_current: int = 50,
    higher_range: float = 0,
    lower_range: float = 0,
    last_price: float = 0,
) -> Dict[str, Any]:
    """
    Prepare chart data for template rendering.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with 'date' and 'close' columns.
    lookback_days : int
        Number of days to display on chart.
    dmas_score : int
        DMAS composite score (0-100).
    technical_score : int
        Technical score (0-100).
    momentum_score : int
        Momentum score (0-100).
    rsi_current : int
        Current RSI value (0-100).
    higher_range : float
        Upper trading range bound.
    lower_range : float
        Lower trading range bound.
    last_price : float
        Most recent price.

    Returns
    -------
    dict
        Data dictionary for template rendering.
    """
    # Get recent data
    df_recent = df.tail(lookback_days).copy()

    # Calculate moving averages
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['ma100'] = df['close'].rolling(window=100).mean()
    df['ma200'] = df['close'].rolling(window=200).mean()

    df_recent = df.tail(lookback_days).copy()

    # Prepare labels and data
    price_labels = df_recent['date'].dt.strftime('%Y-%m-%d').tolist()
    price_data = df_recent['close'].tolist()
    ma50_data = df_recent['ma50'].tolist()
    ma100_data = df_recent['ma100'].tolist()
    ma200_data = df_recent['ma200'].tolist()

    # Calculate RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_recent = rsi.tail(lookback_days)
    rsi_data = rsi_recent.fillna(50).tolist()
    rsi_labels = df_recent['date'].dt.strftime('%Y-%m-%d').tolist()

    # Calculate Fibonacci levels (from 90-day high/low)
    high_90 = df_recent['close'].max()
    low_90 = df_recent['close'].min()
    fib_range = high_90 - low_90
    fib_levels = [
        {'level': 0.0, 'price': low_90},
        {'level': 0.236, 'price': low_90 + fib_range * 0.236},
        {'level': 0.382, 'price': low_90 + fib_range * 0.382},
        {'level': 0.5, 'price': low_90 + fib_range * 0.5},
        {'level': 0.618, 'price': low_90 + fib_range * 0.618},
        {'level': 0.786, 'price': low_90 + fib_range * 0.786},
        {'level': 1.0, 'price': high_90},
    ]

    # Y-axis range (with 5% padding)
    price_min = min(price_data)
    price_max = max(price_data)
    padding = (price_max - price_min) * 0.05
    price_y_min = price_min - padding
    price_y_max = price_max + padding

    # Format last price
    if last_price == 0:
        last_price = price_data[-1] if price_data else 0
    last_price_formatted = f"{last_price:,.2f}"

    # Calculate range percentages
    if last_price > 0:
        higher_range_pct = f"+{((higher_range / last_price - 1) * 100):.1f}%" if higher_range > 0 else ""
        lower_range_pct = f"{((lower_range / last_price - 1) * 100):.1f}%" if lower_range > 0 else ""
    else:
        higher_range_pct = ""
        lower_range_pct = ""

    # Score colors and statuses
    def get_score_color(score):
        if score >= 70:
            return '#10B981'  # Green
        elif score >= 30:
            return '#F59E0B'  # Amber
        else:
            return '#EF4444'  # Red

    def get_score_status(score):
        if score >= 70:
            return 'BULLISH'
        elif score >= 30:
            return 'NEUTRAL'
        else:
            return 'BEARISH'

    def get_rsi_interpretation(rsi_val):
        if rsi_val >= 70:
            return 'OVERBOUGHT'
        elif rsi_val <= 30:
            return 'OVERSOLD'
        else:
            return 'NEUTRAL'

    return {
        'price_labels': price_labels,
        'price_data': price_data,
        'ma50_data': ma50_data,
        'ma100_data': ma100_data,
        'ma200_data': ma200_data,
        'show_ma50': True,
        'show_ma100': True,
        'show_ma200': True,
        'fib_levels': fib_levels,
        'price_y_min': price_y_min,
        'price_y_max': price_y_max,
        'higher_range': higher_range if higher_range > 0 else price_y_max,
        'lower_range': lower_range if lower_range > 0 else price_y_min,
        'higher_range_pct': higher_range_pct,
        'lower_range_pct': lower_range_pct,
        'rsi_labels': rsi_labels,
        'rsi_data': rsi_data,
        'last_price': last_price_formatted,
        'dmas_score': dmas_score,
        'dmas_change_text': '',
        'dmas_change_color': '#94a3b8',
        'technical_score': technical_score,
        'technical_color': get_score_color(technical_score),
        'technical_status': get_score_status(technical_score),
        'technical_trend': '',
        'technical_trend_color': '#94a3b8',
        'momentum_score': momentum_score,
        'momentum_color': get_score_color(momentum_score),
        'momentum_status': get_score_status(momentum_score),
        'momentum_trend': '',
        'momentum_trend_color': '#94a3b8',
        'rsi_current': rsi_current,
        'rsi_color': get_score_color(50),  # RSI uses different interpretation
        'rsi_interpretation': get_rsi_interpretation(rsi_current),
        'rsi_trend': '',
        'rsi_trend_color': '#94a3b8',
    }


# =============================================================================
# FULL SLIDE HTML BUILDER
# =============================================================================

def build_full_slide_html(
    instrument: str,
    view: str,
    subtitle: str,
    chart_data: Dict[str, Any],
    date_str: str,
    scale: int = DEFAULT_SCALE,
    logo_base64: Optional[str] = None,
) -> str:
    """
    Build the complete HTML for a full slide.

    Parameters
    ----------
    instrument : str
        Instrument identifier (e.g., 'gold', 'spx').
    view : str
        Market view (e.g., 'Bullish', 'Bearish', 'Neutral').
    subtitle : str
        Subtitle text generated by Claude Haiku.
    chart_data : dict
        Chart data from prepare_chart_data().
    date_str : str
        Date string for footer (e.g., '27/01/2026').
    scale : int
        Scale factor (default 4 for high-res).
    logo_base64 : str, optional
        Base64-encoded logo image.

    Returns
    -------
    str
        Complete HTML ready for Playwright rendering.
    """
    category = get_category(instrument)
    display_name = get_display_name(instrument)

    # Build the full HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;1,400;1,500&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation"></script>
    <style>
@import url('https://fonts.cdnfonts.com/css/calibri-light');

* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

body {{
    width: {SLIDE_BASE_WIDTH * scale}px;
    height: {SLIDE_BASE_HEIGHT * scale}px;
    background: white;
    position: relative;
    font-family: 'Calibri', sans-serif;
    overflow: hidden;
}}

/* Navy Banner - exact from PPTX */
.banner {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: {52 * scale}px;
    background: linear-gradient(135deg, #1a365d 0%, #1e3a5f 100%);
}}

/* Gold horizontal line under banner - exact from PPTX */
.gold-line {{
    position: absolute;
    top: {52 * scale}px;
    left: 0;
    width: 100%;
    height: {1 * scale}px;
    background: #c9a227;
}}

.banner-text {{
    position: absolute;
    left: {15 * scale}px;
    top: {26 * scale}px;
    transform: translateY(-50%);
    font-family: 'Calibri', sans-serif;
    font-size: {26 * scale}px;
    font-weight: bold;
    font-style: italic;
    color: white;
}}

/* Text-based logo fallback */
.logo-text {{
    position: absolute;
    right: {15 * scale}px;
    top: {26 * scale}px;
    transform: translateY(-50%);
    font-family: 'Calibri', sans-serif;
    font-size: {18 * scale}px;
    font-weight: bold;
    font-style: italic;
    color: #c9a227;
}}

/* Logo image - exact from PPTX */
.logo {{
    position: absolute;
    left: {826 * scale}px;
    top: {1 * scale}px;
    width: {124 * scale}px;
    height: {113 * scale}px;
}}

/* Gold vertical accent bar - exact from PPTX */
.gold-bar {{
    position: absolute;
    left: {43 * scale}px;
    top: {93 * scale}px;
    width: {1 * scale}px;
    height: {78 * scale}px;
    background: #c9a227;
    border-radius: {1 * scale}px;
}}

/* Title - exact from PPTX: 24pt Calibri bold, #00B0F0 (NOT italic) */
.title {{
    position: absolute;
    left: {56 * scale}px;
    top: {90 * scale}px;
    font-family: 'Calibri', sans-serif;
    font-size: {24 * scale}px;
    font-weight: bold;
    font-style: normal;
    color: #00B0F0;
}}

/* Subtitle - exact from PPTX: 16pt Calibri bold, #040C38 */
.subtitle {{
    position: absolute;
    left: {56 * scale}px;
    top: {136 * scale}px;
    font-family: 'Calibri', sans-serif;
    font-size: {16 * scale}px;
    font-weight: bold;
    color: #040C38;
    max-width: {850 * scale}px;
    line-height: 1.3;
}}

/* Chart container - exact from PPTX */
.chart-container {{
    position: absolute;
    left: {43 * scale}px;
    top: {182 * scale}px;
    width: {CHART_WIDTH * scale}px;
    height: {CHART_HEIGHT * scale}px;
    overflow: hidden;
}}

/* Source footer - exact from PPTX */
.source {{
    position: absolute;
    left: {43 * scale}px;
    top: {575 * scale}px;
    font-family: 'Calibri', sans-serif;
    font-size: {8 * scale}px;
    font-weight: bold;
    color: #94a3b8;
}}

/* Chart styles */
.chart-wrapper {{
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
    border-radius: {8 * scale}px;
}}

.main-container {{
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 100%;
}}

.price-row {{
    display: flex;
    height: {240 * scale}px;
}}

.price-chart-area {{
    flex: 1;
    position: relative;
    padding: {8 * scale}px;
    padding-left: {5 * scale}px;
    padding-right: {10 * scale}px;
    margin-left: 0;
    background: linear-gradient(90deg,
        rgba(255,255,255,0.95) 0%,
        rgba(255,255,255,0.85) 70%,
        rgba(248,250,252,0.75) 90%,
        rgba(240,244,248,0.6) 100%
    );
    border: {1 * scale}px solid #D1D9E6;
    border-right: none;
    border-radius: {8 * scale}px 0 0 0;
    box-shadow: inset -{20 * scale}px 0 {30 * scale}px -{15 * scale}px rgba(27, 58, 90, 0.08);
}}

.price-chart-container {{
    position: relative;
    width: 100%;
    height: 100%;
}}

.dmas-panel {{
    width: {160 * scale}px;
    min-width: {160 * scale}px;
    background: linear-gradient(180deg, #1B3A5A 0%, #152D45 100%);
    padding: {10 * scale}px {12 * scale}px;
    display: flex;
    flex-direction: column;
    gap: {5 * scale}px;
    border: {1 * scale}px solid #1B3A5A;
    border-left: none;
    border-radius: 0 {8 * scale}px 0 0;
    box-shadow: -{8 * scale}px 0 {20 * scale}px -{5 * scale}px rgba(27, 58, 90, 0.25);
}}

.panel-title {{
    font-size: {7 * scale}px;
    text-transform: uppercase;
    letter-spacing: {1.5 * scale}px;
    color: rgba(255, 255, 255, 0.6);
    text-align: center;
    margin-bottom: {2 * scale}px;
}}

.dmas-value {{
    font-size: {28 * scale}px;
    font-weight: 700;
    color: #FFFFFF;
    text-align: center;
    line-height: 1;
}}

.dmas-bar {{
    height: {5 * scale}px;
    background: rgba(255, 255, 255, 0.15);
    border-radius: {2.5 * scale}px;
    position: relative;
    margin: {5 * scale}px 0;
}}

.dmas-marker {{
    position: absolute;
    width: {3 * scale}px;
    height: {10 * scale}px;
    background: #c9a227;
    border-radius: {1.5 * scale}px;
    top: -{2.5 * scale}px;
    transform: translateX(-50%);
}}

.dmas-change {{
    font-size: {9 * scale}px;
    text-align: center;
    margin-bottom: {5 * scale}px;
    color: #94a3b8;
}}

.sub-score-section {{
    background: rgba(0, 0, 0, 0.2);
    border-radius: {5 * scale}px;
    padding: {6 * scale}px;
}}

.sub-score-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: {3 * scale}px;
}}

.sub-score-label {{
    font-size: {7 * scale}px;
    color: rgba(255, 255, 255, 0.7);
    text-transform: uppercase;
    letter-spacing: {0.5 * scale}px;
}}

.sub-score-value {{
    font-size: {13 * scale}px;
    font-weight: 700;
    color: #FFFFFF;
}}

.score-trend {{
    font-size: {8 * scale}px;
    margin-right: {3 * scale}px;
}}

.sub-score-bar {{
    height: {3 * scale}px;
    background: rgba(255, 255, 255, 0.15);
    border-radius: {1.5 * scale}px;
    margin: {3 * scale}px 0;
}}

.sub-score-fill {{
    height: 100%;
    border-radius: {1.5 * scale}px;
    transition: width 0.3s ease;
}}

.sub-score-status {{
    display: flex;
    align-items: center;
    gap: {3 * scale}px;
    font-size: {6 * scale}px;
    text-transform: uppercase;
    letter-spacing: {0.5 * scale}px;
}}

.status-dot {{
    width: {5 * scale}px;
    height: {5 * scale}px;
    border-radius: 50%;
}}

.panel-footnote {{
    font-size: {5 * scale}px;
    color: rgba(255, 255, 255, 0.4);
    text-align: center;
    line-height: 1.3;
    margin-top: auto;
    padding-top: {5 * scale}px;
}}

.rsi-row {{
    display: flex;
    height: {72 * scale}px;
    flex-shrink: 0;
}}

.rsi-chart-area {{
    flex: 1;
    position: relative;
    padding: 0 {10 * scale}px {8 * scale}px {5 * scale}px;
    background: linear-gradient(90deg,
        rgba(255,255,255,0.95) 0%,
        rgba(255,255,255,0.85) 70%,
        rgba(248,250,252,0.75) 90%,
        rgba(240,244,248,0.6) 100%
    );
    border: {1 * scale}px solid #D1D9E6;
    border-top: none;
    border-right: none;
    border-radius: 0 0 0 {8 * scale}px;
    box-shadow: inset -{20 * scale}px 0 {30 * scale}px -{15 * scale}px rgba(27, 58, 90, 0.08);
}}

.rsi-chart-container {{
    position: relative;
    width: 100%;
    height: 100%;
}}

.rsi-panel {{
    width: {160 * scale}px;
    min-width: {160 * scale}px;
    background: linear-gradient(180deg, #152D45 0%, #0F2132 100%);
    padding: {6 * scale}px {12 * scale}px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    border: {1 * scale}px solid #1B3A5A;
    border-left: none;
    border-top: none;
    border-radius: 0 0 {8 * scale}px 0;
}}

.chart-legend {{
    position: absolute;
    top: {4 * scale}px;
    left: {5 * scale}px;
    right: {10 * scale}px;
    display: flex;
    justify-content: center;
    gap: {16 * scale}px;
    font-size: {8 * scale}px;
    z-index: 10;
}}

.legend-item {{
    display: flex;
    align-items: center;
    gap: {4 * scale}px;
}}

.legend-line {{
    width: {16 * scale}px;
    height: {2.5 * scale}px;
    border-radius: {1.25 * scale}px;
}}

.legend-text {{
    color: #040C38;
    font-weight: bold;
}}
    </style>
</head>
<body>
    <!-- Navy Banner -->
    <div class="banner">
        <span class="banner-text">{category}</span>
    </div>

    <!-- Gold horizontal line under banner -->
    <div class="gold-line"></div>

    <!-- Logo (image or text fallback) -->
    {"<img class='logo' src='data:image/png;base64," + logo_base64 + "' alt='Herculis'>" if logo_base64 else "<span class='logo-text'>HERCULIS</span>"}

    <!-- Gold vertical accent bar -->
    <div class="gold-bar"></div>

    <!-- Title -->
    <div class="title">{display_name}: {view}</div>

    <!-- Subtitle -->
    <div class="subtitle">{subtitle}</div>

    <!-- Chart -->
    <div class="chart-container">
        <div class="chart-wrapper">
            <div class="main-container">
                <!-- Price Row -->
                <div class="price-row">
                    <div class="price-chart-area">
                        <div class="chart-legend">
                            <div class="legend-item">
                                <div class="legend-line" style="background: #1B3A5A;"></div>
                                <span class="legend-text">Price ({chart_data['last_price']})</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-line" style="background: #10B981;"></div>
                                <span class="legend-text">50-day MA</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-line" style="background: #F59E0B;"></div>
                                <span class="legend-text">100-day MA</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-line" style="background: #EF4444;"></div>
                                <span class="legend-text">200-day MA</span>
                            </div>
                        </div>
                        <div class="price-chart-container">
                            <canvas id="priceChart"></canvas>
                        </div>
                    </div>
                    <div class="dmas-panel">
                        <div class="panel-title">DMAS Score*</div>
                        <div class="dmas-value">{chart_data['dmas_score']}</div>
                        <div class="dmas-bar">
                            <div class="dmas-marker" style="left: {chart_data['dmas_score']}%;"></div>
                        </div>
                        <div class="dmas-change">{chart_data['dmas_change_text']}</div>

                        <div class="sub-score-section">
                            <div class="sub-score-row">
                                <span class="sub-score-label">Technical</span>
                                <span class="sub-score-value">
                                    <span class="score-trend" style="color: {chart_data['technical_trend_color']};">{chart_data['technical_trend']}</span>
                                    {chart_data['technical_score']}
                                </span>
                            </div>
                            <div class="sub-score-bar">
                                <div class="sub-score-fill" style="width: {chart_data['technical_score']}%; background: {chart_data['technical_color']};"></div>
                            </div>
                            <div class="sub-score-status">
                                <div class="status-dot" style="background: {chart_data['technical_color']};"></div>
                                <span style="color: {chart_data['technical_color']};">{chart_data['technical_status']}</span>
                            </div>
                        </div>

                        <div class="sub-score-section">
                            <div class="sub-score-row">
                                <span class="sub-score-label">Momentum</span>
                                <span class="sub-score-value">
                                    <span class="score-trend" style="color: {chart_data['momentum_trend_color']};">{chart_data['momentum_trend']}</span>
                                    {chart_data['momentum_score']}
                                </span>
                            </div>
                            <div class="sub-score-bar">
                                <div class="sub-score-fill" style="width: {chart_data['momentum_score']}%; background: {chart_data['momentum_color']};"></div>
                            </div>
                            <div class="sub-score-status">
                                <div class="status-dot" style="background: {chart_data['momentum_color']};"></div>
                                <span style="color: {chart_data['momentum_color']};">{chart_data['momentum_status']}</span>
                            </div>
                        </div>

                        <div class="panel-footnote">* DMAS score is a proprietary scoring of Herculis and is calculated as the average of the technical and momentum scores</div>
                    </div>
                </div>

                <!-- RSI Row -->
                <div class="rsi-row">
                    <div class="rsi-chart-area">
                        <div class="rsi-chart-container">
                            <canvas id="rsiChart"></canvas>
                        </div>
                    </div>
                    <div class="rsi-panel">
                        <div class="sub-score-section">
                            <div class="sub-score-row">
                                <span class="sub-score-label">RSI (14)</span>
                                <span class="sub-score-value">
                                    <span class="score-trend" style="color: {chart_data['rsi_trend_color']};">{chart_data['rsi_trend']}</span>
                                    {chart_data['rsi_current']}
                                </span>
                            </div>
                            <div class="sub-score-bar">
                                <div class="sub-score-fill" style="width: {chart_data['rsi_current']}%; background: {chart_data['rsi_color']};"></div>
                            </div>
                            <div class="sub-score-status">
                                <div class="status-dot" style="background: {chart_data['rsi_color']};"></div>
                                <span style="color: {chart_data['rsi_color']};">{chart_data['rsi_interpretation']}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Source footer -->
    <div class="source">Source: Bloomberg, Herculis Group. Data as of {date_str}</div>

    <script>
const scale = {scale};
const priceLabels = {json.dumps(chart_data['price_labels'])};
const priceData = {json.dumps(chart_data['price_data'])};
const ma50Data = {json.dumps(chart_data['ma50_data'])};
const ma100Data = {json.dumps(chart_data['ma100_data'])};
const ma200Data = {json.dumps(chart_data['ma200_data'])};
const showMa50 = true;
const showMa100 = true;
const showMa200 = true;
const fibLevels = {json.dumps(chart_data['fib_levels'])};
const priceYMin = {chart_data['price_y_min']};
const priceYMax = {chart_data['price_y_max']};
const higherRange = {chart_data['higher_range']};
const lowerRange = {chart_data['lower_range']};
const higherRangePct = "{chart_data['higher_range_pct']}";
const lowerRangePct = "{chart_data['lower_range_pct']}";
const rsiLabels = {json.dumps(chart_data['rsi_labels'])};
const rsiData = {json.dumps(chart_data['rsi_data'])};

// Fibonacci annotations
const fibAnnotations = {{}};
fibLevels.forEach((fib, index) => {{
    fibAnnotations['fib' + index] = {{
        type: 'line',
        yMin: fib.price,
        yMax: fib.price,
        borderColor: 'rgba(148, 163, 184, 0.3)',
        borderWidth: 1,
        borderDash: [4, 4],
    }};
}});

// Price Chart
const priceCtx = document.getElementById('priceChart').getContext('2d');
const priceChart = new Chart(priceCtx, {{
    type: 'line',
    data: {{
        labels: priceLabels,
        datasets: [
            {{
                label: 'Price',
                data: priceData,
                borderColor: '#1B3A5A',
                borderWidth: 2 * scale,
                pointRadius: 0,
                tension: 0.1,
                fill: false,
            }},
            showMa50 ? {{
                label: '50 MA',
                data: ma50Data,
                borderColor: '#10B981',
                borderWidth: 1.5 * scale,
                pointRadius: 0,
                tension: 0.1,
                fill: false,
            }} : null,
            showMa100 ? {{
                label: '100 MA',
                data: ma100Data,
                borderColor: '#F59E0B',
                borderWidth: 1.5 * scale,
                pointRadius: 0,
                tension: 0.1,
                fill: false,
            }} : null,
            showMa200 ? {{
                label: '200 MA',
                data: ma200Data,
                borderColor: '#EF4444',
                borderWidth: 1.5 * scale,
                pointRadius: 0,
                tension: 0.1,
                fill: false,
            }} : null,
        ].filter(d => d !== null)
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        animation: {{
            duration: 0,
            onComplete: function() {{
                document.body.setAttribute('data-chart-ready', 'true');
            }}
        }},
        layout: {{
            padding: {{
                top: 20 * scale,
                right: 10 * scale,
                bottom: 5 * scale,
                left: 5 * scale,
            }}
        }},
        plugins: {{
            legend: {{ display: false }},
            tooltip: {{ enabled: false }},
            annotation: {{
                annotations: fibAnnotations
            }}
        }},
        scales: {{
            x: {{
                display: true,
                grid: {{ display: false }},
                ticks: {{
                    font: {{ size: 7 * scale, family: 'Calibri', weight: 'bold' }},
                    color: '#040C38',
                    padding: 4 * scale,
                    maxRotation: 0,
                    autoSkip: true,
                    maxTicksLimit: 12,
                }},
            }},
            y: {{
                display: true,
                position: 'left',
                min: priceYMin,
                max: priceYMax,
                grid: {{ display: false }},
                ticks: {{
                    font: {{ size: 7 * scale, family: 'Calibri', weight: 'bold' }},
                    color: '#040C38',
                    padding: 4 * scale,
                    callback: function(value) {{
                        return value.toLocaleString();
                    }}
                }},
            }}
        }}
    }}
}});

// RSI Chart
const rsiCtx = document.getElementById('rsiChart').getContext('2d');
const rsiChart = new Chart(rsiCtx, {{
    type: 'line',
    data: {{
        labels: rsiLabels,
        datasets: [{{
            label: 'RSI',
            data: rsiData,
            borderColor: '#8B5CF6',
            borderWidth: 2 * scale,
            pointRadius: 0,
            tension: 0.1,
            fill: false,
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        animation: {{ duration: 0 }},
        layout: {{
            padding: {{
                top: 10 * scale,
                right: 10 * scale,
                bottom: 5 * scale,
                left: 5 * scale,
            }}
        }},
        plugins: {{
            legend: {{ display: false }},
            tooltip: {{ enabled: false }},
            annotation: {{
                annotations: {{
                    oversoldZone: {{
                        type: 'box',
                        yMin: 0,
                        yMax: 30,
                        backgroundColor: 'rgba(16, 185, 129, 0.08)',
                        borderWidth: 0,
                    }},
                    overboughtZone: {{
                        type: 'box',
                        yMin: 70,
                        yMax: 100,
                        backgroundColor: 'rgba(239, 68, 68, 0.08)',
                        borderWidth: 0,
                    }},
                    line30: {{
                        type: 'line',
                        yMin: 30,
                        yMax: 30,
                        borderColor: 'rgba(16, 185, 129, 0.4)',
                        borderWidth: 1 * scale,
                        borderDash: [4 * scale, 4 * scale],
                    }},
                    line70: {{
                        type: 'line',
                        yMin: 70,
                        yMax: 70,
                        borderColor: 'rgba(239, 68, 68, 0.4)',
                        borderWidth: 1 * scale,
                        borderDash: [4 * scale, 4 * scale],
                    }}
                }}
            }}
        }},
        scales: {{
            x: {{
                display: true,
                grid: {{ display: false }},
                ticks: {{
                    font: {{ size: 7 * scale, family: 'Calibri', weight: 'bold' }},
                    color: '#040C38',
                    padding: 4 * scale,
                    maxRotation: 0,
                    autoSkip: true,
                    maxTicksLimit: 12,
                }},
            }},
            y: {{
                display: true,
                position: 'left',
                min: 0,
                max: 100,
                grid: {{ display: false }},
                ticks: {{
                    font: {{ size: 7 * scale, family: 'Calibri', weight: 'bold' }},
                    color: '#040C38',
                    padding: 4 * scale,
                    autoSkip: false,
                    callback: function(value) {{
                        if (value === 30 || value === 70) {{
                            return value;
                        }}
                        return null;
                    }},
                    stepSize: 10,
                }},
            }}
        }}
    }}
}});
    </script>
</body>
</html>'''

    return html


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_full_slide(
    instrument: str,
    view: str,
    subtitle: str,
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    scale: int = DEFAULT_SCALE,
    dmas_score: int = 50,
    technical_score: int = 50,
    momentum_score: int = 50,
    rsi_current: int = 50,
    higher_range: float = 0,
    lower_range: float = 0,
    logo_path: Optional[str] = None,
    date_str: Optional[str] = None,
) -> bytes:
    """
    Render a complete technical analysis slide as high-quality PNG.

    Parameters
    ----------
    instrument : str
        Instrument identifier (e.g., 'gold', 'spx', 'bitcoin').
    view : str
        Market view (e.g., 'Bullish', 'Bearish', 'Neutral').
    subtitle : str
        Subtitle text for the slide.
    df : pd.DataFrame
        Price data with 'date' and 'close' columns.
    output_path : str, optional
        Path to save PNG file. If None, returns bytes only.
    scale : int
        Scale factor for output (default 4 = 3840x2400px).
    dmas_score : int
        DMAS composite score (0-100).
    technical_score : int
        Technical score (0-100).
    momentum_score : int
        Momentum score (0-100).
    rsi_current : int
        Current RSI value (0-100).
    higher_range : float
        Upper trading range bound.
    lower_range : float
        Lower trading range bound.
    logo_path : str, optional
        Path to Herculis logo image.
    date_str : str, optional
        Date string for footer. If None, uses today's date.

    Returns
    -------
    bytes
        PNG image data.
    """
    # Prepare date string
    if date_str is None:
        date_str = datetime.now().strftime('%d/%m/%Y')

    # Get logo as base64
    logo_base64 = get_logo_base64(logo_path)

    # Prepare chart data
    chart_data = prepare_chart_data(
        df=df,
        lookback_days=90,
        dmas_score=dmas_score,
        technical_score=technical_score,
        momentum_score=momentum_score,
        rsi_current=rsi_current,
        higher_range=higher_range,
        lower_range=lower_range,
    )

    # Build HTML
    html_content = build_full_slide_html(
        instrument=instrument,
        view=view,
        subtitle=subtitle,
        chart_data=chart_data,
        date_str=date_str,
        scale=scale,
        logo_base64=logo_base64,
    )

    # Debug: Save HTML for inspection
    try:
        debug_html_path = os.path.join(tempfile.gettempdir(), "full_slide_debug.html")
        with open(debug_html_path, "w") as f:
            f.write(html_content)
        print(f"[Full Slide] Debug HTML saved to: {debug_html_path}")
    except Exception as e:
        print(f"[Full Slide] Could not save debug HTML: {e}")

    # Render with Playwright
    png_bytes = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                viewport={
                    'width': SLIDE_BASE_WIDTH * scale,
                    'height': SLIDE_BASE_HEIGHT * scale
                },
                device_scale_factor=1  # Already scaled in HTML
            )

            page.set_content(html_content, wait_until='networkidle')

            # Wait for Chart.js to render
            try:
                page.wait_for_function("typeof Chart !== 'undefined'", timeout=10000)
                page.wait_for_selector('body[data-chart-ready="true"]', timeout=15000)
                print("[Full Slide] Chart.js rendering complete")
            except Exception as wait_err:
                print(f"[Full Slide] Wait timeout, using fallback: {wait_err}")
                page.wait_for_timeout(5000)

            # Take screenshot
            png_bytes = page.screenshot()
            print(f"[Full Slide] Screenshot taken: {len(png_bytes)} bytes")
            browser.close()

    except Exception as e:
        print(f"[Full Slide] Playwright error: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Save to file if path provided
    if output_path and png_bytes:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(png_bytes)
        print(f"[Full Slide] Saved to: {output_path}")

    return png_bytes


# =============================================================================
# CONVENIENCE FUNCTION FOR TESTING
# =============================================================================

def render_test_slide(
    instrument: str = 'gold',
    output_path: str = 'exports/test_slide.png',
) -> bytes:
    """
    Render a test slide with sample data.

    Parameters
    ----------
    instrument : str
        Instrument to render (default 'gold').
    output_path : str
        Output file path.

    Returns
    -------
    bytes
        PNG image data.
    """
    import numpy as np

    # Generate sample price data
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=300, freq='D')
    base_price = 2000 if instrument == 'gold' else 5000
    prices = base_price * (1 + np.cumsum(np.random.randn(300) * 0.01))

    df = pd.DataFrame({
        'date': dates,
        'close': prices,
    })

    return render_full_slide(
        instrument=instrument,
        view='Bullish',
        subtitle='Impressive setup points to potential for sustained uptrend. Key resistance levels have been cleared with strong volume confirmation.',
        df=df,
        output_path=output_path,
        scale=4,
        dmas_score=72,
        technical_score=75,
        momentum_score=69,
        rsi_current=62,
        higher_range=base_price * 1.05,
        lower_range=base_price * 0.95,
    )
