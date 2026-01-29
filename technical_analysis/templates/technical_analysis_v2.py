"""
Technical Analysis V2 Template Components.

This module separates the CSS, HTML, and JavaScript for the Technical Analysis V2
chart into distinct components for better maintainability.

The template uses:
- Chart.js for interactive charts
- Jinja2 for variable interpolation
- Playwright for PNG export

Components:
- TECH_V2_CSS: All styling for the chart layout
- TECH_V2_HTML_BODY: The HTML structure with Jinja2 placeholders
- TECH_V2_JAVASCRIPT: Chart.js configuration and rendering logic
"""

# =============================================================================
# CSS STYLES
# =============================================================================

TECH_V2_CSS = '''
@import url('https://fonts.cdnfonts.com/css/calibri-light');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
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
    width: {{ width }}px;
    height: {{ height }}px;
    padding: 0;
}

.main-container {
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 100%;
}

.price-row {
    display: flex;
    height: 300px;  /* Reduced to accommodate taller RSI row */
}

.price-chart-area {
    flex: 1;
    position: relative;
    padding: 8px;
    padding-left: 5px;    /* Reduced from 20px - extend chart left */
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
    width: 200px;         /* Increased from ~160px for better proportions */
    min-width: 200px;
    background: linear-gradient(180deg, #1B3A5A 0%, #152D45 100%);
    padding: 12px 15px;   /* More horizontal padding */
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
    letter-spacing: 1px;
    color: rgba(255,255,255,0.5);
    border-bottom: 1px solid rgba(255,255,255,0.15);
    padding-bottom: 6px;
    text-align: center;
}

.dmas-score-card {
    background: rgba(255,255,255,0.1);
    border-radius: 5px;
    padding: 10px 15px;
    margin-bottom: 8px;
    text-align: center;
}

.dmas-label {
    font-size: 8px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.6);
    margin-bottom: 3px;
}

.dmas-asterisk {
    color: rgba(255,255,255,0.8);
    font-size: 10px;
    vertical-align: super;
}

.dmas-value {
    font-size: 36px;      /* Adjusted for wider panel */
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.1;
}

.dmas-progress {
    width: 100%;
    height: 8px;
    /* Full gradient scale: red -> orange -> yellow -> green */
    background: linear-gradient(90deg, #EF4444 0%, #F59E0B 25%, #EAB308 50%, #84CC16 75%, #22C55E 100%);
    border-radius: 4px;
    margin: 8px 0;
    position: relative;
    overflow: visible;
}

.dmas-marker {
    position: absolute;
    width: 4px;
    height: 12px;
    background: #FFFFFF;
    border-radius: 2px;
    top: -2px;
    transform: translateX(-50%);
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
}

.dmas-change {
    font-size: 10px;
    color: {{ dmas_change_color }};
    margin-top: 4px;
}

.sub-score-card {
    background: rgba(255,255,255,0.08);
    border-radius: 5px;
    padding: 8px 12px;
    margin-bottom: 6px;
}

.sub-score-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}

.sub-score-title {
    font-size: 10px;
    color: rgba(255,255,255,0.7);
}

.sub-score-value {
    font-size: 18px;
    font-weight: 700;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 3px;
}

.score-trend {
    font-size: 12px;
    font-weight: 600;
}

.sub-score-progress {
    height: 5px;
    background: rgba(255,255,255,0.15);
    border-radius: 2.5px;
    overflow: hidden;
    margin-bottom: 4px;
}

.sub-score-fill {
    height: 100%;
    border-radius: 2.5px;
}

.sub-score-status {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 9px;
}

.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
}

.rsi-row {
    display: flex;
    height: 120px;  /* Increased for better RSI visibility */
}

.rsi-chart-area {
    flex: 1;
    position: relative;
    padding: 10px;
    padding-left: 5px;    /* Same as price chart - critical for Y-axis alignment */
    padding-right: 10px;
    padding-top: 5px;
    margin-left: 0;
    background: linear-gradient(90deg,
        rgba(255,255,255,0.95) 0%,
        rgba(255,255,255,0.85) 70%,
        rgba(248,250,252,0.75) 90%,
        rgba(240,244,248,0.6) 100%
    );
    border: 1px solid #D1D9E6;
    border-top: none;
    border-right: none;
    border-radius: 0 0 0 8px;  /* Restored bottom-left radius */
    box-shadow: inset -20px 0 30px -15px rgba(27, 58, 90, 0.08);
}

.rsi-chart-container {
    position: relative;
    width: 100%;
    height: 100%;
}

.rsi-title {
    position: absolute;
    top: 5px;
    left: 15px;
    font-size: 11px;
    font-weight: 600;
    color: #1B3A5A;
    z-index: 10;
}

.rsi-panel {
    width: 200px;         /* Same as DMAS panel */
    min-width: 200px;
    background: linear-gradient(180deg, #1B3A5A 0%, #152D45 100%);
    padding: 12px 15px;
    display: flex;
    flex-direction: column;
    justify-content: center;  /* Center the card vertically */
    border: 1px solid #1B3A5A;
    border-top: none;
    border-left: none;
    border-radius: 0 0 8px 0;  /* Restored bottom-right radius */
    box-shadow: -8px 0 20px -5px rgba(27, 58, 90, 0.25);
}

.rsi-panel .sub-score-card {
    margin-bottom: 0;  /* Override default margin for RSI panel */
}

.panel-footnote {
    font-size: 6px;
    color: rgba(255,255,255,0.4);
    text-align: left;
    margin-top: 6px;
    padding-top: 4px;
    line-height: 1.3;
}

.chart-legend {
    position: absolute;
    top: 5px;
    left: 5px;            /* Align with chart left edge */
    right: 10px;          /* Align with chart right edge */
    display: flex;
    justify-content: center;  /* CENTER the legend */
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
# HTML BODY STRUCTURE
# =============================================================================

TECH_V2_HTML_BODY = '''
<div class="main-container">
    <div class="price-row">
        <div class="price-chart-area">
            <div class="chart-legend">
                <div class="legend-item">
                    <div class="legend-line" style="background: #1B3A5A;"></div>
                    <span class="legend-text">Price ({{ last_price }})</span>
                </div>
                {% if show_ma50 %}
                <div class="legend-item">
                    <div class="legend-line" style="background: #10B981;"></div>
                    <span class="legend-text">50-day MA</span>
                </div>
                {% endif %}
                {% if show_ma100 %}
                <div class="legend-item">
                    <div class="legend-line" style="background: #F59E0B;"></div>
                    <span class="legend-text">100-day MA</span>
                </div>
                {% endif %}
                {% if show_ma200 %}
                <div class="legend-item">
                    <div class="legend-line" style="background: #EF4444;"></div>
                    <span class="legend-text">200-day MA</span>
                </div>
                {% endif %}
            </div>
            <div class="price-chart-container">
                <canvas id="priceChart"></canvas>
            </div>
        </div>

        <div class="dmas-panel">
            <div class="panel-title">Dynamic Market Analysis</div>
            <div class="dmas-score-card">
                <div class="dmas-label">DMAS Score<span class="dmas-asterisk">*</span></div>
                <div class="dmas-value">{{ dmas_score }}</div>
                <div class="dmas-progress">
                    <div class="dmas-marker" style="left: {{ dmas_score }}%;"></div>
                </div>
                <div class="dmas-change">{{ dmas_change_text }}</div>
            </div>
            <div class="sub-score-card">
                <div class="sub-score-header">
                    <span class="sub-score-title">Technical</span>
                    <span class="sub-score-value">
                        <span class="score-trend" style="color: {{ technical_trend_color }};">{{ technical_trend }}</span>
                        {{ technical_score }}
                    </span>
                </div>
                <div class="sub-score-progress">
                    <div class="sub-score-fill" style="width: {{ technical_score }}%; background: {{ technical_color }};"></div>
                </div>
                <div class="sub-score-status">
                    <div class="status-dot" style="background: {{ technical_color }};"></div>
                    <span style="color: {{ technical_color }};">{{ technical_status }}</span>
                </div>
            </div>
            <div class="sub-score-card">
                <div class="sub-score-header">
                    <span class="sub-score-title">Momentum</span>
                    <span class="sub-score-value">
                        <span class="score-trend" style="color: {{ momentum_trend_color }};">{{ momentum_trend }}</span>
                        {{ momentum_score }}
                    </span>
                </div>
                <div class="sub-score-progress">
                    <div class="sub-score-fill" style="width: {{ momentum_score }}%; background: {{ momentum_color }};"></div>
                </div>
                <div class="sub-score-status">
                    <div class="status-dot" style="background: {{ momentum_color }};"></div>
                    <span style="color: {{ momentum_color }};">{{ momentum_status }}</span>
                </div>
            </div>
        </div>
    </div>

    <div class="rsi-row">
        <div class="rsi-chart-area">
            <div class="rsi-title">RSI (14)</div>
            <div class="rsi-chart-container">
                <canvas id="rsiChart"></canvas>
            </div>
        </div>
        <div class="rsi-panel">
            <div class="sub-score-card">
                <div class="sub-score-header">
                    <span class="sub-score-title">RSI (14)</span>
                    <span class="sub-score-value">
                        <span class="score-trend" style="color: {{ rsi_trend_color }};">{{ rsi_trend }}</span>
                        {{ rsi_current }}
                    </span>
                </div>
                <div class="sub-score-progress">
                    <div class="sub-score-fill" style="width: {{ rsi_current }}%; background: {{ rsi_color }};"></div>
                </div>
                <div class="sub-score-status">
                    <div class="status-dot" style="background: {{ rsi_color }};"></div>
                    <span style="color: {{ rsi_color }};">{{ rsi_interpretation }}</span>
                </div>
            </div>
            <div class="panel-footnote">* DMAS score is a proprietary scoring of Herculis and is calculated as the average of the technical and momentum scores</div>
        </div>
    </div>
</div>
'''

# =============================================================================
# JAVASCRIPT FOR CHART.JS
# =============================================================================

TECH_V2_JAVASCRIPT = '''
const scale = {{ scale }};
const priceLabels = {{ price_labels | tojson }};
const priceData = {{ price_data | tojson }};
const ma50Data = {{ ma50_data | tojson }};
const ma100Data = {{ ma100_data | tojson }};
const ma200Data = {{ ma200_data | tojson }};
const showMa50 = {{ show_ma50 | tojson }};
const showMa100 = {{ show_ma100 | tojson }};
const showMa200 = {{ show_ma200 | tojson }};
const fibLevels = {{ fib_levels | tojson }};
const priceYMin = {{ price_y_min }};
const priceYMax = {{ price_y_max }};
const higherRange = {{ higher_range }};
const lowerRange = {{ lower_range }};
const higherRangePct = "{{ higher_range_pct }}";
const lowerRangePct = "{{ lower_range_pct }}";
const rsiLabels = {{ rsi_labels | tojson }};
const rsiData = {{ rsi_data | tojson }};

const fibAnnotations = {};
fibLevels.forEach((level, idx) => {
    fibAnnotations['fib' + idx] = {
        type: 'line',
        yMin: level,
        yMax: level,
        borderColor: 'rgba(150, 150, 150, 0.25)',  // Subtle grey, background reference
        borderWidth: 1,
        borderDash: [3, 3],  // Shorter dashes
    };
});

fibAnnotations['higherRange'] = {
    type: 'line',
    yMin: higherRange,
    yMax: higherRange,
    xMin: priceLabels.length - 1,
    borderColor: '#1B3A5A',
    borderWidth: 1.5 * scale / 3,
    borderDash: [5, 3],
};
fibAnnotations['lowerRange'] = {
    type: 'line',
    yMin: lowerRange,
    yMax: lowerRange,
    xMin: priceLabels.length - 1,
    borderColor: '#1B3A5A',
    borderWidth: 1.5 * scale / 3,
    borderDash: [5, 3],
};

const priceCtx = document.getElementById('priceChart').getContext('2d');
const priceChart = new Chart(priceCtx, {
    type: 'line',
    data: {
        labels: priceLabels,
        datasets: [
            {
                label: 'Price',
                data: priceData,
                borderColor: '#1B3A5A',
                borderWidth: 3,  // Bold price line for visibility
                pointRadius: 0,
                tension: 0.1,
                fill: false,
                order: 1,
            },
            {
                label: '50-day MA',
                data: showMa50 ? ma50Data : [],  // Hide if too far from price
                borderColor: '#10B981',
                borderWidth: 2.5 * scale / 3,
                pointRadius: 0,
                tension: 0.1,
                fill: false,
                order: 2,
                hidden: !showMa50,
            },
            {
                label: '100-day MA',
                data: showMa100 ? ma100Data : [],  // Hide if too far from price
                borderColor: '#F59E0B',
                borderWidth: 2.5 * scale / 3,
                pointRadius: 0,
                tension: 0.1,
                fill: false,
                order: 3,
                hidden: !showMa100,
            },
            {
                label: '200-day MA',
                data: showMa200 ? ma200Data : [],  // Hide if too far from price
                borderColor: '#EF4444',
                borderWidth: 2.5 * scale / 3,
                pointRadius: 0,
                tension: 0.1,
                fill: false,
                order: 4,
                hidden: !showMa200,
            },
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 0,
            onComplete: function() {
                document.body.setAttribute('data-chart-ready', 'true');
            }
        },
        layout: {
            padding: {
                top: 25,
                right: 90,
                bottom: 5,
                left: 5,      /* Reduced from 50 - extend chart left */
            }
        },
        plugins: {
            legend: { display: false },
            tooltip: { enabled: false },
            annotation: {
                annotations: fibAnnotations
            }
        },
        scales: {
            x: {
                display: true,
                grid: { display: false },
                ticks: {
                    font: { size: 9 * scale, family: 'Calibri', weight: 'bold' },
                    color: '#040C38',
                    padding: 5,
                    maxRotation: 0,
                    autoSkip: true,
                    maxTicksLimit: 12,
                },
                afterFit: function(axis) {
                    axis.paddingLeft = 0;
                    axis.paddingRight = 0;
                }
            },
            y: {
                display: true,
                position: 'left',
                min: priceYMin,
                max: priceYMax,
                grid: {
                    display: false,
                },
                ticks: {
                    font: { size: 9 * scale, family: 'Calibri', weight: 'bold' },
                    color: '#040C38',
                    padding: 5,
                    callback: function(value) {
                        return value.toLocaleString();
                    }
                },
                afterFit: function(axis) {
                    axis.width = 50;  // Force same width for both charts
                }
            }
        }
    },
    plugins: [{
        id: 'regressionChannel',
        beforeDatasetsDraw: function(chart) {
            const ctx = chart.ctx;
            const yScale = chart.scales.y;
            const xScale = chart.scales.x;
            const chartArea = chart.chartArea;

            // 1-month regression channel (21 trading days)
            const lookback = 21;
            const n = Math.min(lookback, priceData.length);
            const startIdx = priceData.length - n;
            const recentPrices = priceData.slice(startIdx);

            if (n < 3) return;  // Not enough data

            // === 1. Calculate OLS Linear Regression ===
            const xValues = Array.from({ length: n }, (_, i) => i);

            const sumX = xValues.reduce((a, b) => a + b, 0);
            const sumY = recentPrices.reduce((a, b) => a + b, 0);
            const sumXY = xValues.reduce((acc, x, i) => acc + x * recentPrices[i], 0);
            const sumX2 = xValues.reduce((acc, x) => acc + x * x, 0);

            const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
            const intercept = (sumY - slope * sumX) / n;

            // Regression line values
            const regressionValues = xValues.map(x => intercept + slope * x);

            // === 2. Find Max Distance Above & Below Regression ===
            let maxAbove = 0;
            let maxBelow = 0;

            recentPrices.forEach((price, i) => {
                const regValue = regressionValues[i];
                const diff = price - regValue;

                if (diff > maxAbove) maxAbove = diff;
                if (diff < maxBelow) maxBelow = diff;  // maxBelow will be negative
            });

            // === 3. Create Parallel Bands ===
            const upperBand = regressionValues.map(v => v + maxAbove);
            const lowerBand = regressionValues.map(v => v + maxBelow);

            // === 4. Determine Color Based on Slope ===
            const isPositive = slope > 0;

            const fillColor = isPositive
                ? 'rgba(16, 185, 129, 0.15)'   // Green fill (subtle)
                : 'rgba(239, 68, 68, 0.15)';   // Red fill (subtle)

            const lineColor = isPositive
                ? 'rgba(16, 185, 129, 0.6)'    // Green, bolder (60% opacity)
                : 'rgba(239, 68, 68, 0.6)';    // Red, bolder (60% opacity)

            // === 5. Helper functions for coordinate conversion ===
            const getX = (i) => xScale.getPixelForValue(startIdx + i);
            const getY = (price) => yScale.getPixelForValue(price);

            // === 6. Draw Filled Channel ===
            ctx.save();

            // Clip to chart area
            ctx.beginPath();
            ctx.rect(chartArea.left, chartArea.top, chartArea.right - chartArea.left, chartArea.bottom - chartArea.top);
            ctx.clip();

            ctx.beginPath();

            // Upper band (left to right)
            ctx.moveTo(getX(0), getY(upperBand[0]));
            for (let i = 1; i < n; i++) {
                ctx.lineTo(getX(i), getY(upperBand[i]));
            }

            // Lower band (right to left)
            for (let i = n - 1; i >= 0; i--) {
                ctx.lineTo(getX(i), getY(lowerBand[i]));
            }

            ctx.closePath();
            ctx.fillStyle = fillColor;
            ctx.fill();

            // === 7. Draw Dashed Band Lines ===
            ctx.setLineDash([6, 4]);
            ctx.strokeStyle = lineColor;
            ctx.lineWidth = 2;  // Bolder for hero element

            // Upper band line
            ctx.beginPath();
            ctx.moveTo(getX(0), getY(upperBand[0]));
            for (let i = 1; i < n; i++) {
                ctx.lineTo(getX(i), getY(upperBand[i]));
            }
            ctx.stroke();

            // Lower band line
            ctx.beginPath();
            ctx.moveTo(getX(0), getY(lowerBand[0]));
            for (let i = 1; i < n; i++) {
                ctx.lineTo(getX(i), getY(lowerBand[i]));
            }
            ctx.stroke();

            ctx.restore();  // Reset clipping and line dash
        }
    },
    {
        id: 'tradingRangeLabels',
        afterDraw: function(chart) {
            const ctx = chart.ctx;
            const yScale = chart.scales.y;
            const xScale = chart.scales.x;
            const chartArea = chart.chartArea;
            const fontSize = 9 * scale;
            const smallFontSize = 7 * scale;
            const dotRadius = 3 * scale;

            // Get x position for range start (at last data point)
            const dotX = chartArea.right;
            const lineEndX = chartArea.right + 80 * scale;  // End of dashed line

            // Higher range
            const higherY = yScale.getPixelForValue(higherRange);

            // Draw labels ABOVE the line for Higher Range (right-aligned)
            ctx.font = `${smallFontSize}px Calibri`;
            ctx.textAlign = 'right';
            ctx.fillStyle = '#64748B';
            ctx.fillText('Higher Range', lineEndX, higherY - 18 * scale);
            ctx.font = `600 ${fontSize}px Calibri`;
            ctx.fillStyle = '#10B981';
            ctx.fillText(higherRange.toLocaleString() + ' (' + higherRangePct + ')', lineEndX, higherY - 6 * scale);

            // Draw dot at higher range
            ctx.beginPath();
            ctx.arc(dotX, higherY, dotRadius, 0, Math.PI * 2);
            ctx.fillStyle = '#1B3A5A';
            ctx.fill();

            // Draw dashed line from dot to right (BELOW text)
            ctx.save();
            ctx.setLineDash([4, 3]);
            ctx.strokeStyle = '#1B3A5A';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(dotX + dotRadius, higherY);
            ctx.lineTo(lineEndX, higherY);
            ctx.stroke();
            ctx.restore();

            // Lower range
            const lowerY = yScale.getPixelForValue(lowerRange);

            // Draw dot at lower range
            ctx.beginPath();
            ctx.arc(dotX, lowerY, dotRadius, 0, Math.PI * 2);
            ctx.fillStyle = '#1B3A5A';
            ctx.fill();

            // Draw dashed line from dot to right (ABOVE text)
            ctx.save();
            ctx.setLineDash([4, 3]);
            ctx.strokeStyle = '#1B3A5A';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(dotX + dotRadius, lowerY);
            ctx.lineTo(lineEndX, lowerY);
            ctx.stroke();
            ctx.restore();

            // Draw labels BELOW the line for Lower Range (right-aligned)
            ctx.font = `600 ${fontSize}px Calibri`;
            ctx.textAlign = 'right';
            ctx.fillStyle = '#EF4444';
            ctx.fillText(lowerRange.toLocaleString() + ' (' + lowerRangePct + ')', lineEndX, lowerY + 14 * scale);
            ctx.font = `${smallFontSize}px Calibri`;
            ctx.fillStyle = '#64748B';
            ctx.fillText('Lower Range', lineEndX, lowerY + 26 * scale);
        }
    }]
});

const rsiCtx = document.getElementById('rsiChart').getContext('2d');
const rsiChart = new Chart(rsiCtx, {
    type: 'line',
    data: {
        labels: rsiLabels,
        datasets: [{
            label: 'RSI',
            data: rsiData,
            borderColor: '#8B5CF6',
            borderWidth: 2.5 * scale / 3,
            pointRadius: 0,
            tension: 0.1,
            fill: false,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        layout: {
            padding: {
                top: 20,
                right: 90,
                bottom: 5,
                left: 5,      /* Same as price chart - keep aligned */
            }
        },
        plugins: {
            legend: { display: false },
            tooltip: { enabled: false },
            annotation: {
                annotations: {
                    oversoldZone: {
                        type: 'box',
                        yMin: 0,
                        yMax: 30,
                        backgroundColor: 'rgba(16, 185, 129, 0.08)',
                        borderWidth: 0,
                    },
                    overboughtZone: {
                        type: 'box',
                        yMin: 70,
                        yMax: 100,
                        backgroundColor: 'rgba(239, 68, 68, 0.08)',
                        borderWidth: 0,
                    },
                    line30: {
                        type: 'line',
                        yMin: 30,
                        yMax: 30,
                        borderColor: 'rgba(16, 185, 129, 0.4)',
                        borderWidth: 1,
                        borderDash: [4, 4],
                    },
                    line70: {
                        type: 'line',
                        yMin: 70,
                        yMax: 70,
                        borderColor: 'rgba(239, 68, 68, 0.4)',
                        borderWidth: 1,
                        borderDash: [4, 4],
                    }
                }
            }
        },
        scales: {
            x: {
                display: true,
                grid: { display: false },
                ticks: {
                    font: { size: 9 * scale, family: 'Calibri', weight: 'bold' },
                    color: '#040C38',
                    padding: 5,
                    maxRotation: 0,
                    autoSkip: true,
                    maxTicksLimit: 12,
                },
                afterFit: function(axis) {
                    axis.paddingLeft = 0;
                    axis.paddingRight = 0;
                }
            },
            y: {
                display: true,
                position: 'left',
                min: 0,
                max: 100,
                grid: { display: false },
                ticks: {
                    font: { size: 9 * scale, family: 'Calibri', weight: 'bold' },
                    color: '#040C38',
                    padding: 5,
                    autoSkip: false,
                    // Show only RSI threshold levels: 30 and 70
                    callback: function(value) {
                        if (value === 30 || value === 70) {
                            return value;
                        }
                        return null;  // Hide 0, 100, and other tick labels
                    },
                    stepSize: 10,  // Generate ticks at 10-unit intervals
                },
                afterFit: function(axis) {
                    axis.width = 50;  // Force same width as price chart
                }
            }
        }
    }
});
'''

# =============================================================================
# TEMPLATE ASSEMBLY FUNCTION
# =============================================================================

def build_technical_analysis_v2_template() -> str:
    """
    Assemble the full HTML template from CSS, HTML body, and JavaScript components.

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
{TECH_V2_CSS}
    </style>
</head>
<body>
{TECH_V2_HTML_BODY}
    <script>
{TECH_V2_JAVASCRIPT}
    </script>
</body>
</html>
'''


# Pre-built template for backward compatibility
TECHNICAL_ANALYSIS_V2_HTML_TEMPLATE = build_technical_analysis_v2_template()


# =============================================================================
# FULL SLIDE WRAPPER (for high-quality PNG export)
# =============================================================================

FULL_SLIDE_CSS = '''
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;1,400;1,500&family=Inter:wght@400;500;600;700&display=swap');

.slide-container {
    width: {{ slide_width }}px;
    height: {{ slide_height }}px;
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
.slide-title {
    position: absolute;
    left: {{ 56 * scale }}px;
    top: {{ 90 * scale }}px;
    font-family: 'Playfair Display', Georgia, serif;
    font-style: italic;
    font-size: {{ 30 * scale }}px;
    color: #c9a227;
}

/* Subtitle */
.slide-subtitle {
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
    width: {{ chart_width }}px;
    height: {{ chart_height }}px;
    overflow: hidden;
}

/* Source footer */
.slide-source {
    position: absolute;
    left: {{ 43 * scale }}px;
    top: {{ 575 * scale }}px;
    font-family: 'Inter', sans-serif;
    font-size: {{ 9 * scale }}px;
    color: #94a3b8;
}
'''


def build_full_slide_template(
    category: str,
    instrument: str,
    view: str,
    subtitle: str,
    date_str: str,
    scale: int = 4,
) -> str:
    """
    Build HTML template for full slide export.

    This wraps the existing chart template with slide elements:
    - Navy banner with category
    - Herculis logo (text fallback)
    - Gold accent bar
    - Title and subtitle
    - Chart area (placeholder for existing chart)
    - Source footer

    Parameters
    ----------
    category : str
        Asset category (Equity, Commodities, Crypto).
    instrument : str
        Instrument display name (e.g., "Gold", "S&P 500").
    view : str
        Market view (e.g., "Bullish", "Bearish").
    subtitle : str
        Subtitle text.
    date_str : str
        Date string for source footer (e.g., "27/01/2026").
    scale : int
        Scale factor for output (default 4 = 3840x2400px).

    Returns
    -------
    str
        HTML template with Jinja2 placeholders for chart data.
    """
    # Slide dimensions at scale
    slide_width = 960 * scale
    slide_height = 600 * scale
    chart_width = 895 * scale
    chart_height = 394 * scale

    # Build the full slide CSS with scale values
    full_css = FULL_SLIDE_CSS.replace('{{ slide_width }}', str(slide_width))
    full_css = full_css.replace('{{ slide_height }}', str(slide_height))
    full_css = full_css.replace('{{ chart_width }}', str(chart_width))
    full_css = full_css.replace('{{ chart_height }}', str(chart_height))
    full_css = full_css.replace('{{ scale }}', str(scale))
    # Handle multiplication expressions
    for i in [52, 15, 26, 18, 43, 93, 4, 77, 2, 56, 90, 30, 136, 13, 850, 181, 575, 9]:
        full_css = full_css.replace(f'{{{{ {i} * scale }}}}', str(i * scale))

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation"></script>
    <style>
{full_css}
{TECH_V2_CSS}
    </style>
</head>
<body>
    <div class="slide-container">
        <!-- Navy Banner -->
        <div class="banner">
            <span class="banner-text">{category}</span>
            <span class="logo-text">HERCULIS</span>
        </div>

        <!-- Gold accent bar -->
        <div class="gold-bar"></div>

        <!-- Title -->
        <div class="slide-title">{instrument}: {view}</div>

        <!-- Subtitle -->
        <div class="slide-subtitle">{subtitle}</div>

        <!-- Chart (existing template embedded) -->
        <div class="chart-container">
{TECH_V2_HTML_BODY}
        </div>

        <!-- Source footer -->
        <div class="slide-source">Source: Bloomberg, Herculis Group. Data as of {date_str}</div>
    </div>

    <script>
{TECH_V2_JAVASCRIPT}
    </script>
</body>
</html>
'''


# Category mapping for instruments (supports multiple ticker formats)
INSTRUMENT_CATEGORIES = {
    # Equity - Index tickers
    'spx': 'Equity', 's&p 500': 'Equity', 'spx index': 'Equity',
    'dax': 'Equity', 'dax index': 'Equity',
    'smi': 'Equity', 'smi index': 'Equity',
    'nikkei': 'Equity', 'nky index': 'Equity', 'nikkei 225': 'Equity',
    'sensex': 'Equity', 'sensex index': 'Equity',
    'csi': 'Equity', 'shsz300 index': 'Equity', 'csi 300': 'Equity',
    'ibov': 'Equity', 'ibov index': 'Equity', 'ibovespa': 'Equity',
    'mexbol': 'Equity', 'mexbol index': 'Equity',
    'tasi': 'Equity', 'saseidx index': 'Equity',
    # Commodities - Bloomberg tickers (gca, sia, xpt, xpd, cl1, lp1)
    'gold': 'Commodities', 'gca': 'Commodities', 'gca comdty': 'Commodities',
    'silver': 'Commodities', 'sia': 'Commodities', 'sia comdty': 'Commodities',
    'oil': 'Commodities', 'cl1': 'Commodities', 'cl1 comdty': 'Commodities',
    'copper': 'Commodities', 'lp1': 'Commodities', 'lp1 comdty': 'Commodities',
    'platinum': 'Commodities', 'xpt': 'Commodities', 'xpt comdty': 'Commodities',
    'palladium': 'Commodities', 'xpd': 'Commodities', 'xpd curncy': 'Commodities',
    # Crypto - Bloomberg tickers
    'bitcoin': 'Crypto', 'btc': 'Crypto', 'xbtusd curncy': 'Crypto',
    'ethereum': 'Crypto', 'eth': 'Crypto', 'xetusd curncy': 'Crypto',
    'solana': 'Crypto', 'sol': 'Crypto', 'xsousd curncy': 'Crypto',
    'ripple': 'Crypto', 'xrp': 'Crypto', 'xrpusd curncy': 'Crypto',
    'binance': 'Crypto', 'bnb': 'Crypto', 'xbiusd curncy': 'Crypto',
}


def get_category_for_ticker(ticker: str) -> str:
    """Get category for a Bloomberg ticker."""
    return INSTRUMENT_CATEGORIES.get(ticker.lower(), 'Markets')


def get_display_name_for_ticker(ticker: str) -> str:
    """Get display name for a Bloomberg ticker."""
    DISPLAY_NAMES = {
        # Equity
        'spx index': 'S&P 500',
        'dax index': 'DAX',
        'smi index': 'SMI',
        'nky index': 'Nikkei 225',
        'sensex index': 'Sensex',
        'shsz300 index': 'CSI 300',
        'ibov index': 'Ibovespa',
        'mexbol index': 'MEXBOL',
        'saseidx index': 'TASI',
        # Commodities (Bloomberg ticker formats)
        'gca comdty': 'Gold',
        'sia comdty': 'Silver',
        'cl1 comdty': 'Brent Oil',
        'lp1 comdty': 'Copper',
        'xpt comdty': 'Platinum',
        'xpd curncy': 'Palladium',
        # Crypto (Bloomberg ticker formats)
        'xbtusd curncy': 'Bitcoin',
        'xetusd curncy': 'Ethereum',
        'xsousd curncy': 'Solana',
        'xrpusd curncy': 'Ripple',
        'xbiusd curncy': 'Binance Coin',
    }
    return DISPLAY_NAMES.get(ticker.lower(), ticker.split()[0].title())
