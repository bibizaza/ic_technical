"""HTML template for Weekly Performance chart."""

# Placeholder template for YTD charts when insufficient data
YTD_INSUFFICIENT_DATA_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        @import url('https://fonts.cdnfonts.com/css/calibri-light');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
            background: #FFFFFF;
            width: {{ width }}px;
            height: {{ height }}px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .placeholder-container {
            text-align: center;
            color: #64748B;
        }

        .placeholder-title {
            font-size: {{ 18 * scale }}px;
            font-weight: 600;
            color: #1B3A5A;
            margin-bottom: {{ 15 * scale }}px;
        }

        .placeholder-message {
            font-size: {{ 14 * scale }}px;
            color: #64748B;
        }
    </style>
</head>
<body>
    <div class="placeholder-container">
        <div class="placeholder-title">{{ chart_title }}</div>
        <div class="placeholder-message">Insufficient data for YTD chart</div>
    </div>
</body>
</html>
'''

WEEKLY_PERFORMANCE_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    
    <style>
        @import url('https://fonts.cdnfonts.com/css/calibri-light');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
            background: #FFFFFF;
            width: {{ width }}px;
            height: {{ height }}px;
            padding: {{ 10 * scale }}px;
        }

        .chart-container {
            display: flex;
            flex-direction: column;
            gap: {{ 4 * scale }}px;
        }

        .row {
            display: flex;
            align-items: center;
            height: {{ 32 * scale }}px;
            padding: 0 {{ 10 * scale }}px;
            border-radius: {{ 4 * scale }}px;
            border-left: {{ 3 * scale }}px solid transparent;
            box-sizing: border-box;
        }

        .row:nth-child(odd) {
            background: #F8FAFC;
        }

        /* Top performer - gold accent */
        .row.top-performer {
            background: linear-gradient(90deg, #FEF9E7 0%, #FFFEF5 100%);
            border-left: {{ 3 * scale }}px solid #C9A227;
        }

        .row.top-performer .market-name {
            color: #92710C;
            font-weight: 600;
        }

        /* Worst performer - red accent */
        .row.worst-performer {
            background: linear-gradient(90deg, #FEF2F2 0%, #FFFAFA 100%);
            border-left: {{ 3 * scale }}px solid #EF4444;
        }

        .row.worst-performer .market-name {
            color: #B91C1C;
        }

        /* Market info */
        .market-info {
            width: {{ 110 * scale }}px;
            min-width: {{ 110 * scale }}px;
            max-width: {{ 110 * scale }}px;
            flex-shrink: 0;
            display: flex;
            align-items: center;
            gap: {{ 8 * scale }}px;
        }

        .flag {
            font-size: {{ 22 * scale }}px;
            line-height: 1;
        }

        /* Flag images */
        .market-info img,
        .market-info .flag-img {
            width: {{ 22 * scale }}px !important;
            height: {{ 14 * scale }}px !important;
            vertical-align: middle;
            margin-right: {{ 4 * scale }}px;
            flex-shrink: 0;
        }

        .market-name {
            flex: 1;
            font-size: {{ 10 * scale }}px;
            font-weight: 500;
            color: #334155;
        }

        /* Price column */
        .price {
            width: {{ 85 * scale }}px;
            flex-shrink: 0;
            text-align: left;
            font-size: {{ 9 * scale }}px;
            font-weight: 400;
            color: #94A3B8;
            padding-left: {{ 4 * scale }}px;
        }

        /* Bar container */
        .bar-container {
            flex: 1;
            display: flex;
            align-items: center;
            padding: 0 {{ 15 * scale }}px;
            position: relative;
        }

        .bar-track {
            width: 100%;
            height: {{ 5 * scale }}px;
            background: #F1F5F9;
            border-radius: {{ 3 * scale }}px;
            position: relative;
        }

        /* Center line (zero axis) */
        .center-line {
            position: absolute;
            left: 50%;
            top: {{ -4 * scale }}px;
            bottom: {{ -4 * scale }}px;
            width: {{ 2 * scale }}px;
            background: #CBD5E1;
            border-radius: {{ 1 * scale }}px;
            z-index: 10;
        }

        /* Bars */
        .bar {
            position: absolute;
            height: {{ 12 * scale }}px;
            top: 50%;
            transform: translateY(-50%);
            z-index: 2;
        }

        .bar.positive {
            left: calc(50% + 1px);
            background: linear-gradient(90deg, #4ADE80, #16A34A);
            box-shadow: 0 {{ 2 * scale }}px {{ 4 * scale }}px rgba(34, 197, 94, 0.25);
            border-radius: 0 {{ 6 * scale }}px {{ 6 * scale }}px 0;
        }

        .bar.negative {
            right: calc(50% + 1px);
            background: linear-gradient(270deg, #F87171, #DC2626);
            box-shadow: 0 {{ 2 * scale }}px {{ 4 * scale }}px rgba(239, 68, 68, 0.25);
            border-radius: {{ 6 * scale }}px 0 0 {{ 6 * scale }}px;
        }

        /* Performance value */
        .performance {
            width: {{ 55 * scale }}px;
            text-align: right;
            font-size: {{ 11 * scale }}px;
            font-weight: 700;
        }

        .performance.positive { color: #16A34A; }
        .performance.negative { color: #DC2626; }
        .performance.zero { color: #64748B; }

        /* Scale */
        .scale {
            display: flex;
            justify-content: center;
            padding: {{ 8 * scale }}px 0 0 0;
            margin-left: {{ 195 * scale }}px;  /* market-info (110) + price (85) */
            margin-right: {{ 55 * scale }}px;
        }

        .scale-inner {
            display: flex;
            justify-content: space-between;
            width: 100%;
            padding: 0 {{ 15 * scale }}px;
            font-size: {{ 7 * scale }}px;
            color: #94A3B8;
        }
    </style>
</head>
<body>
    <div class="chart-container">
        {% for row in rows %}
        <div class="row {{ row.highlight_class }}">
            <div class="market-info">
                {{ row.flag_html | safe }}
                <span class="market-name">{{ row.name }}</span>
            </div>
            <div class="price">{{ row.formatted_price|default('') }}</div>
            <div class="bar-container">
                <div class="bar-track">
                    <div class="center-line"></div>
                    {% if row.value != 0 %}
                    <div class="bar {{ row.bar_class }}" style="width: {{ row.bar_width }}%;"></div>
                    {% endif %}
                </div>
            </div>
            <div class="performance {{ row.value_class }}">{{ row.formatted_value }}</div>
        </div>
        {% endfor %}
    </div>

    <div class="scale">
        <div class="scale-inner">
            <span>{{ scale_min }}</span>
            <span>{{ scale_mid_low }}</span>
            <span>0</span>
            <span>{{ scale_mid_high }}</span>
            <span>{{ scale_max }}</span>
        </div>
    </div>
</body>
</html>
'''


# =============================================================================
# HISTORICAL PERFORMANCE HEATMAP TEMPLATE
# =============================================================================

HISTORICAL_PERFORMANCE_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    
    <style>
        @import url('https://fonts.cdnfonts.com/css/calibri-light');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Calibri', Calibri, 'Segoe UI', Arial, sans-serif;
            background: #FFFFFF;
            width: {{ width }}px;
            height: {{ height }}px;
            padding: {{ 10 * scale }}px;
        }

        .table-container {
            display: flex;
            flex-direction: column;
            gap: {{ 3 * scale }}px;
        }

        /* Header row */
        .header-row {
            display: flex;
            align-items: center;
            padding: {{ 5 * scale }}px 0;
        }

        .header-row .market-col {
            width: {{ 140 * scale }}px;
        }

        .header-row .period-col {
            flex: 1;
            text-align: center;
            font-size: {{ 12 * scale }}px;
            font-weight: 600;
            color: #1B3A5A;
        }

        .header-row .period-col.ytd {
            flex: 1.3;
            font-size: {{ 13 * scale }}px;
            font-weight: 700;
            color: #1B3A5A;
            margin-right: {{ 16 * scale }}px;
        }

        /* Data rows */
        .data-row {
            display: flex;
            align-items: center;
            gap: {{ 4 * scale }}px;
        }

        .market-col {
            width: {{ 140 * scale }}px;
            display: flex;
            align-items: center;
            gap: {{ 8 * scale }}px;
            padding-right: {{ 10 * scale }}px;
        }

        .flag {
            font-size: {{ 28 * scale }}px;
            line-height: 1;
        }

        /* Flag images */
        .market-col img,
        .market-col .flag-img {
            width: {{ 28 * scale }}px !important;
            height: {{ 18 * scale }}px !important;
            vertical-align: middle;
            margin-right: {{ 5 * scale }}px;
            flex-shrink: 0;
        }

        .market-name {
            font-size: {{ 12 * scale }}px;
            font-weight: 400;
            color: #1B3A5A;
        }

        /* Value cells */
        .value-cell {
            flex: 1;
            height: {{ 32 * scale }}px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: {{ 5 * scale }}px;
            font-size: {{ 10 * scale }}px;
            font-weight: 600;
            color: #FFFFFF;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }

        /* YTD column - emphasized */
        .value-cell.ytd {
            flex: 1.3;
            height: {{ 34 * scale }}px;
            font-size: {{ 12 * scale }}px;
            font-weight: 700;
            margin-right: {{ 13 * scale }}px;
            border-radius: {{ 6 * scale }}px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15), 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Color scale - Positive (green) - 5 levels */
        .positive-1 { background: linear-gradient(135deg, #BBF7D0, #86EFAC); color: #166534; text-shadow: none; }
        .positive-2 { background: linear-gradient(135deg, #86EFAC, #4ADE80); color: #166534; text-shadow: none; }
        .positive-3 { background: linear-gradient(135deg, #4ADE80, #22C55E); color: #FFFFFF; }
        .positive-4 { background: linear-gradient(135deg, #22C55E, #16A34A); color: #FFFFFF; }
        .positive-5 { background: linear-gradient(135deg, #16A34A, #15803D); color: #FFFFFF; }

        /* Color scale - Negative (red) - 5 levels */
        .negative-1 { background: linear-gradient(135deg, #FECACA, #FCA5A5); color: #991B1B; text-shadow: none; }
        .negative-2 { background: linear-gradient(135deg, #FCA5A5, #F87171); color: #991B1B; text-shadow: none; }
        .negative-3 { background: linear-gradient(135deg, #F87171, #EF4444); color: #FFFFFF; }
        .negative-4 { background: linear-gradient(135deg, #EF4444, #DC2626); color: #FFFFFF; }
        .negative-5 { background: linear-gradient(135deg, #DC2626, #B91C1C); color: #FFFFFF; }

        /* Neutral (near zero) */
        .neutral { background: #F1F5F9; color: #64748B; text-shadow: none; }
    </style>
</head>
<body>
    <div class="table-container">
        <!-- Header -->
        <div class="header-row">
            <div class="market-col"></div>
            <div class="period-col ytd">YTD</div>
            <div class="period-col">1M</div>
            <div class="period-col">3M</div>
            <div class="period-col">6M</div>
            <div class="period-col">12M</div>
        </div>

        {% for row in rows %}
        <div class="data-row">
            <div class="market-col">
                {{ row.flag_html | safe }}
                <span class="market-name">{{ row.name }}</span>
            </div>
            <div class="value-cell ytd {{ row.ytd_class }}">{{ row.ytd_formatted }}</div>
            <div class="value-cell {{ row.m1_class }}">{{ row.m1_formatted }}</div>
            <div class="value-cell {{ row.m3_class }}">{{ row.m3_formatted }}</div>
            <div class="value-cell {{ row.m6_class }}">{{ row.m6_formatted }}</div>
            <div class="value-cell {{ row.m12_class }}">{{ row.m12_formatted }}</div>
        </div>
        {% endfor %}
    </div>
</body>
</html>
'''


# =============================================================================
# GOVERNMENT BONDS RATES TEMPLATE
# =============================================================================

BONDS_RATES_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    
    <style>
        @import url('https://fonts.cdnfonts.com/css/calibri-light');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Calibri', Calibri, 'Segoe UI', Arial, sans-serif;
            background: #FFFFFF;
            width: {{ width }}px;
            height: {{ height }}px;
            padding: {{ 10 * scale }}px;
        }

        .chart-container {
            display: flex;
            flex-direction: column;
            gap: {{ 1 * scale }}px;
        }

        /* Country group header */
        .country-header {
            display: flex;
            align-items: center;
            gap: {{ 6 * scale }}px;
            padding: {{ 4 * scale }}px {{ 10 * scale }}px;
            margin-top: {{ 3 * scale }}px;
            background: #F8FAFC;
            border-left: {{ 3 * scale }}px solid #1B3A5A;
            border-radius: 0 {{ 4 * scale }}px {{ 4 * scale }}px 0;
        }

        .country-header:first-child {
            margin-top: 0;
        }

        .country-header .flag {
            font-size: {{ 22 * scale }}px;
            line-height: 1;
        }

        /* Flag images in country headers */
        .country-header img,
        .country-header .flag-img {
            width: {{ 22 * scale }}px !important;
            height: {{ 14 * scale }}px !important;
            vertical-align: middle;
            margin-right: {{ 4 * scale }}px;
            flex-shrink: 0;
        }

        .country-header .country-name {
            font-size: {{ 10 * scale }}px;
            font-weight: 700;
            color: #1B3A5A;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Data rows */
        .row {
            display: flex;
            align-items: center;
            height: {{ 20 * scale }}px;
            padding: 0 {{ 10 * scale }}px 0 {{ 24 * scale }}px;
        }

        .row:nth-child(odd) {
            background: rgba(248, 250, 252, 0.5);
        }

        /* Tenor label */
        .tenor {
            width: {{ 45 * scale }}px;
            font-size: {{ 9 * scale }}px;
            font-weight: 500;
            color: #64748B;
        }

        /* Yield column */
        .yield {
            width: {{ 55 * scale }}px;
            flex-shrink: 0;
            text-align: left;
            font-size: {{ 9 * scale }}px;
            font-weight: 400;
            color: #94A3B8;
            padding-left: {{ 4 * scale }}px;
        }

        /* Bar container */
        .bar-container {
            flex: 1;
            display: flex;
            align-items: center;
            padding: 0 {{ 12 * scale }}px;
            position: relative;
        }

        .bar-track {
            width: 100%;
            height: {{ 5 * scale }}px;
            background: #F1F5F9;
            border-radius: {{ 3 * scale }}px;
            position: relative;
        }

        /* Center line (zero axis) */
        .center-line {
            position: absolute;
            left: 50%;
            top: {{ -3 * scale }}px;
            bottom: {{ -3 * scale }}px;
            width: {{ 2 * scale }}px;
            background: #CBD5E1;
            border-radius: {{ 1 * scale }}px;
            z-index: 10;
        }

        /* Bars - INVERTED COLORS */
        .bar {
            position: absolute;
            height: {{ 8 * scale }}px;
            top: 50%;
            transform: translateY(-50%);
            z-index: 2;
        }

        /* Rates UP = Bad = Red (extends right) */
        .bar.rates-up {
            left: calc(50% + 1px);
            background: linear-gradient(90deg, #F87171, #DC2626);
            box-shadow: 0 {{ 2 * scale }}px {{ 4 * scale }}px rgba(239, 68, 68, 0.25);
            border-radius: 0 {{ 5 * scale }}px {{ 5 * scale }}px 0;
        }

        /* Rates DOWN = Good = Green (extends left) */
        .bar.rates-down {
            right: calc(50% + 1px);
            background: linear-gradient(270deg, #4ADE80, #16A34A);
            box-shadow: 0 {{ 2 * scale }}px {{ 4 * scale }}px rgba(34, 197, 94, 0.25);
            border-radius: {{ 5 * scale }}px 0 0 {{ 5 * scale }}px;
        }

        /* Performance value */
        .performance {
            width: {{ 55 * scale }}px;
            text-align: right;
            font-size: {{ 10 * scale }}px;
            font-weight: 700;
        }

        .performance.rates-up { color: #DC2626; }
        .performance.rates-down { color: #16A34A; }

        /* Scale */
        .scale {
            display: flex;
            justify-content: center;
            padding: {{ 8 * scale }}px 0 0 0;
            margin-left: {{ 128 * scale }}px;
            margin-right: {{ 55 * scale }}px;
        }

        .scale-inner {
            display: flex;
            justify-content: space-between;
            width: 100%;
            padding: 0 {{ 12 * scale }}px;
            font-size: {{ 7 * scale }}px;
            color: #94A3B8;
        }

        /* Legend - hidden to save vertical space */
        .legend {
            display: none;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: {{ 5 * scale }}px;
        }

        .legend-bar {
            width: {{ 16 * scale }}px;
            height: {{ 5 * scale }}px;
            border-radius: {{ 3 * scale }}px;
        }

        .legend-bar.green {
            background: linear-gradient(90deg, #4ADE80, #16A34A);
        }

        .legend-bar.red {
            background: linear-gradient(90deg, #F87171, #DC2626);
        }
    </style>
</head>
<body>
    <div class="chart-container">
        {% for country in countries %}
        <!-- {{ country.name }} -->
        <div class="country-header">
            {{ country.flag_html | safe }}
            <span class="country-name">{{ country.name }}</span>
        </div>
        {% for tenor in country.tenors %}
        <div class="row">
            <div class="tenor">{{ tenor.label }}</div>
            <div class="yield">{{ tenor.formatted_yield }}</div>
            <div class="bar-container">
                <div class="bar-track">
                    <div class="center-line"></div>
                    {% if tenor.change != 0 %}
                    <div class="bar {{ tenor.bar_class }}" style="width: {{ tenor.bar_width }}%;"></div>
                    {% endif %}
                </div>
            </div>
            <div class="performance {{ tenor.value_class }}">{{ tenor.formatted_change }}</div>
        </div>
        {% endfor %}
        {% endfor %}
    </div>

    <div class="scale">
        <div class="scale-inner">
            <span>{{ scale_min }}</span>
            <span>{{ scale_mid_low }}</span>
            <span>0</span>
            <span>{{ scale_mid_high }}</span>
            <span>{{ scale_max }}</span>
        </div>
    </div>

    <div class="legend">
        <div class="legend-item">
            <div class="legend-bar green"></div>
            Rates down (bullish)
        </div>
        <div class="legend-item">
            <div class="legend-bar red"></div>
            Rates up (bearish)
        </div>
    </div>
</body>
</html>
'''


# =============================================================================
# GOVERNMENT BONDS HISTORICAL PERFORMANCE HEATMAP TEMPLATE
# =============================================================================

BONDS_HISTORICAL_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    
    <style>
        @import url('https://fonts.cdnfonts.com/css/calibri-light');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Calibri', Calibri, 'Segoe UI', Arial, sans-serif;
            background: #FFFFFF;
            width: {{ width }}px;
            height: {{ height }}px;
            padding: {{ 8 * scale }}px;
        }

        .table-container {
            display: flex;
            flex-direction: column;
            gap: {{ 1 * scale }}px;
        }

        /* Header row */
        .header-row {
            display: flex;
            align-items: center;
            padding: {{ 3 * scale }}px 0;
        }

        .header-row .market-col {
            width: {{ 100 * scale }}px;
        }

        .header-row .period-col {
            flex: 1;
            text-align: center;
            font-size: {{ 7 * scale }}px;
            font-weight: 600;
            color: #1B3A5A;
        }

        .header-row .period-col.ytd {
            flex: 1.3;
            font-size: {{ 8 * scale }}px;
            font-weight: 700;
            color: #1B3A5A;
            margin-right: {{ 10 * scale }}px;
        }

        /* Country group header */
        .country-header {
            display: flex;
            align-items: center;
            gap: {{ 5 * scale }}px;
            padding: {{ 3 * scale }}px {{ 6 * scale }}px;
            margin-top: {{ 2 * scale }}px;
            background: #F8FAFC;
            border-left: {{ 2 * scale }}px solid #1B3A5A;
            border-radius: 0 {{ 3 * scale }}px {{ 3 * scale }}px 0;
        }

        .country-header:first-of-type {
            margin-top: 0;
        }

        .country-header .flag {
            font-size: {{ 22 * scale }}px;
            line-height: 1;
        }

        /* Flag images */
        .country-header img,
        .country-header .flag-img {
            width: {{ 22 * scale }}px !important;
            height: {{ 14 * scale }}px !important;
            vertical-align: middle;
            margin-right: {{ 4 * scale }}px;
            flex-shrink: 0;
        }

        .country-header .country-name {
            font-size: {{ 7 * scale }}px;
            font-weight: 700;
            color: #1B3A5A;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Data rows */
        .data-row {
            display: flex;
            align-items: center;
            gap: {{ 2 * scale }}px;
            padding-left: {{ 16 * scale }}px;
        }

        .tenor-col {
            width: {{ 84 * scale }}px;
            display: flex;
            align-items: center;
            padding-right: {{ 6 * scale }}px;
        }

        .tenor-label {
            font-size: {{ 7 * scale }}px;
            font-weight: 500;
            color: #64748B;
        }

        /* Value cells */
        .value-cell {
            flex: 1;
            height: {{ 18 * scale }}px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: {{ 3 * scale }}px;
            font-size: {{ 6 * scale }}px;
            font-weight: 600;
            color: #FFFFFF;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }

        /* YTD column - emphasized */
        .value-cell.ytd {
            flex: 1.3;
            height: {{ 20 * scale }}px;
            font-size: {{ 7 * scale }}px;
            font-weight: 700;
            margin-right: {{ 10 * scale }}px;
            border-radius: {{ 4 * scale }}px;
            box-shadow: 0 {{ 3 * scale }}px {{ 8 * scale }}px rgba(0,0,0,0.15), 0 {{ 1 * scale }}px {{ 3 * scale }}px rgba(0,0,0,0.1);
        }

        /* Color scale - INVERTED for rates */
        /* Rates DOWN = Good = Green (5 levels) */
        .green-1 { background: linear-gradient(135deg, #BBF7D0, #86EFAC); color: #166534; text-shadow: none; }
        .green-2 { background: linear-gradient(135deg, #86EFAC, #4ADE80); color: #166534; text-shadow: none; }
        .green-3 { background: linear-gradient(135deg, #4ADE80, #22C55E); color: #FFFFFF; }
        .green-4 { background: linear-gradient(135deg, #22C55E, #16A34A); color: #FFFFFF; }
        .green-5 { background: linear-gradient(135deg, #16A34A, #15803D); color: #FFFFFF; }

        /* Rates UP = Bad = Red (5 levels) */
        .red-1 { background: linear-gradient(135deg, #FECACA, #FCA5A5); color: #991B1B; text-shadow: none; }
        .red-2 { background: linear-gradient(135deg, #FCA5A5, #F87171); color: #991B1B; text-shadow: none; }
        .red-3 { background: linear-gradient(135deg, #F87171, #EF4444); color: #FFFFFF; }
        .red-4 { background: linear-gradient(135deg, #EF4444, #DC2626); color: #FFFFFF; }
        .red-5 { background: linear-gradient(135deg, #DC2626, #B91C1C); color: #FFFFFF; }
    </style>
</head>
<body>
    <div class="table-container">
        <!-- Header -->
        <div class="header-row">
            <div class="market-col"></div>
            <div class="period-col ytd">YTD</div>
            <div class="period-col">1M</div>
            <div class="period-col">3M</div>
            <div class="period-col">6M</div>
            <div class="period-col">12M</div>
        </div>

        {% for country in countries %}
        <!-- {{ country.name }} -->
        <div class="country-header">
            {{ country.flag_html | safe }}
            <span class="country-name">{{ country.name }}</span>
        </div>
        {% for tenor in country.tenors %}
        <div class="data-row">
            <div class="tenor-col">
                <span class="tenor-label">{{ tenor.label }}</span>
            </div>
            <div class="value-cell ytd {{ tenor.ytd_class }}">{{ tenor.ytd_formatted }}</div>
            <div class="value-cell {{ tenor.m1_class }}">{{ tenor.m1_formatted }}</div>
            <div class="value-cell {{ tenor.m3_class }}">{{ tenor.m3_formatted }}</div>
            <div class="value-cell {{ tenor.m6_class }}">{{ tenor.m6_formatted }}</div>
            <div class="value-cell {{ tenor.m12_class }}">{{ tenor.m12_formatted }}</div>
        </div>
        {% endfor %}
        {% endfor %}
    </div>
</body>
</html>
'''

CORP_BONDS_WEEKLY_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    
    <style>
        @import url('https://fonts.cdnfonts.com/css/calibri-light');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Calibri', Calibri, 'Segoe UI', Arial, sans-serif;
            background: #FFFFFF;
            width: {{ width }}px;
            height: {{ height }}px;
            padding: {{ 10 * scale }}px;
        }

        .chart-container {
            display: flex;
            flex-direction: column;
            gap: {{ 3 * scale }}px;
        }

        /* Data rows */
        .row {
            display: flex;
            align-items: center;
            height: {{ 28 * scale }}px;
            padding: 0 {{ 10 * scale }}px;
            border-radius: {{ 5 * scale }}px;
        }

        .row:nth-child(even) {
            background: #F8FAFC;
        }

        /* Top performer highlight */
        .row.top-performer {
            background: linear-gradient(90deg, rgba(201, 162, 39, 0.1), transparent);
            border-left: {{ 3 * scale }}px solid #C9A227;
        }

        /* Worst performer highlight */
        .row.worst-performer {
            background: linear-gradient(90deg, rgba(239, 68, 68, 0.1), transparent);
            border-left: {{ 3 * scale }}px solid #EF4444;
        }

        /* Market label */
        .market-col {
            width: {{ 130 * scale }}px;
            display: flex;
            align-items: center;
            gap: {{ 6 * scale }}px;
        }

        .flag {
            font-size: {{ 22 * scale }}px;
            line-height: 1;
        }

        /* Flag images */
        .market-col img,
        .market-col .flag-img {
            width: {{ 22 * scale }}px !important;
            height: {{ 14 * scale }}px !important;
            vertical-align: middle;
            margin-right: {{ 4 * scale }}px;
            flex-shrink: 0;
        }

        .credit-badge {
            font-size: {{ 6 * scale }}px;
            font-weight: 700;
            padding: {{ 2 * scale }}px {{ 4 * scale }}px;
            border-radius: {{ 2 * scale }}px;
        }

        .credit-badge.ig {
            background: #DBEAFE;
            color: #1E40AF;
        }

        .credit-badge.hy {
            background: #FEF3C7;
            color: #B45309;
        }

        .market-name {
            font-size: {{ 9 * scale }}px;
            font-weight: 500;
            color: #334155;
        }

        /* Bar container */
        .bar-container {
            flex: 1;
            display: flex;
            align-items: center;
            padding: 0 {{ 12 * scale }}px;
            position: relative;
        }

        .bar-track {
            width: 100%;
            height: {{ 7 * scale }}px;
            background: #F1F5F9;
            border-radius: {{ 3 * scale }}px;
            position: relative;
        }

        /* Center line */
        .center-line {
            position: absolute;
            left: 50%;
            top: {{ -3 * scale }}px;
            bottom: {{ -3 * scale }}px;
            width: {{ 2 * scale }}px;
            background: #CBD5E1;
            border-radius: {{ 1 * scale }}px;
            z-index: 10;
        }

        /* Bars */
        .bar {
            position: absolute;
            height: {{ 14 * scale }}px;
            top: 50%;
            transform: translateY(-50%);
            z-index: 2;
        }

        .bar.positive {
            left: calc(50% + 1px);
            background: linear-gradient(90deg, #4ADE80, #16A34A);
            box-shadow: 0 {{ 2 * scale }}px {{ 4 * scale }}px rgba(34, 197, 94, 0.25);
            border-radius: 0 {{ 5 * scale }}px {{ 5 * scale }}px 0;
        }

        .bar.negative {
            right: calc(50% + 1px);
            background: linear-gradient(270deg, #F87171, #DC2626);
            box-shadow: 0 {{ 2 * scale }}px {{ 4 * scale }}px rgba(239, 68, 68, 0.25);
            border-radius: {{ 5 * scale }}px 0 0 {{ 5 * scale }}px;
        }

        /* Performance value */
        .performance {
            width: {{ 55 * scale }}px;
            text-align: right;
            font-size: {{ 10 * scale }}px;
            font-weight: 700;
        }

        .performance.positive { color: #16A34A; }
        .performance.negative { color: #DC2626; }

        /* Scale */
        .scale {
            display: flex;
            justify-content: center;
            padding: {{ 8 * scale }}px 0 0 0;
            margin-left: {{ 140 * scale }}px;
            margin-right: {{ 55 * scale }}px;
        }

        .scale-inner {
            display: flex;
            justify-content: space-between;
            width: 100%;
            padding: 0 {{ 12 * scale }}px;
            font-size: {{ 7 * scale }}px;
            color: #94A3B8;
        }
    </style>
</head>
<body>
    <div class="chart-container">
        {% for row in rows %}
        <div class="row {{ row.highlight_class }}">
            <div class="market-col">
                {{ row.flag_html | safe }}
                <span class="credit-badge {{ row.credit_class }}">{{ row.credit_type }}</span>
                <span class="market-name">{{ row.name }}</span>
            </div>
            <div class="bar-container">
                <div class="bar-track">
                    <div class="center-line"></div>
                    {% if row.value != 0 %}
                    <div class="bar {{ row.bar_class }}" style="width: {{ row.bar_width }}%;"></div>
                    {% endif %}
                </div>
            </div>
            <div class="performance {{ row.value_class }}">{{ row.formatted_value }}</div>
        </div>
        {% endfor %}
    </div>

    <div class="scale">
        <div class="scale-inner">
            <span>{{ scale_min }}</span>
            <span>{{ scale_mid_low }}</span>
            <span>0</span>
            <span>{{ scale_mid_high }}</span>
            <span>{{ scale_max }}</span>
        </div>
    </div>
</body>
</html>
'''


# =============================================================================
# CORPORATE BONDS HISTORICAL PERFORMANCE HEATMAP TEMPLATE
# =============================================================================

CORP_BONDS_HISTORICAL_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    
    <style>
        @import url('https://fonts.cdnfonts.com/css/calibri-light');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Calibri', Calibri, 'Segoe UI', Arial, sans-serif;
            background: #FFFFFF;
            width: {{ width }}px;
            height: {{ height }}px;
            padding: {{ 10 * scale }}px;
        }

        .table-container {
            display: flex;
            flex-direction: column;
            gap: {{ 2 * scale }}px;
        }

        /* Header row */
        .header-row {
            display: flex;
            align-items: center;
            padding: {{ 3 * scale }}px 0;
        }

        .header-row .market-col {
            width: {{ 130 * scale }}px;
        }

        .header-row .period-col {
            flex: 1;
            text-align: center;
            font-size: {{ 9 * scale }}px;
            font-weight: 600;
            color: #1B3A5A;
        }

        .header-row .period-col.ytd {
            flex: 1.3;
            font-size: {{ 10 * scale }}px;
            font-weight: 700;
            color: #1B3A5A;
            margin-right: {{ 12 * scale }}px;
        }

        /* Data rows */
        .data-row {
            display: flex;
            align-items: center;
            gap: {{ 3 * scale }}px;
        }

        .market-col {
            width: {{ 130 * scale }}px;
            display: flex;
            align-items: center;
            gap: {{ 5 * scale }}px;
            padding-right: {{ 8 * scale }}px;
        }

        .flag {
            font-size: {{ 22 * scale }}px;
            line-height: 1;
        }

        /* Flag images */
        .market-col img,
        .market-col .flag-img {
            width: {{ 22 * scale }}px !important;
            height: {{ 14 * scale }}px !important;
            vertical-align: middle;
            margin-right: {{ 4 * scale }}px;
            flex-shrink: 0;
        }

        .credit-badge {
            font-size: {{ 5 * scale }}px;
            font-weight: 700;
            padding: {{ 1 * scale }}px {{ 3 * scale }}px;
            border-radius: {{ 2 * scale }}px;
        }

        .credit-badge.ig {
            background: #DBEAFE;
            color: #1E40AF;
        }

        .credit-badge.hy {
            background: #FEF3C7;
            color: #B45309;
        }

        .market-name {
            font-size: {{ 8 * scale }}px;
            font-weight: 500;
            color: #334155;
        }

        /* Value cells */
        .value-cell {
            flex: 1;
            height: {{ 22 * scale }}px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: {{ 4 * scale }}px;
            font-size: {{ 8 * scale }}px;
            font-weight: 600;
            color: #FFFFFF;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }

        /* YTD column - emphasized */
        .value-cell.ytd {
            flex: 1.3;
            height: {{ 24 * scale }}px;
            font-size: {{ 9 * scale }}px;
            font-weight: 700;
            margin-right: {{ 12 * scale }}px;
            border-radius: {{ 6 * scale }}px;
            box-shadow: 0 {{ 3 * scale }}px {{ 10 * scale }}px rgba(0,0,0,0.15), 0 {{ 1 * scale }}px {{ 3 * scale }}px rgba(0,0,0,0.1);
        }

        /* Color scale - Positive (green) - 5 levels */
        .positive-1 { background: linear-gradient(135deg, #BBF7D0, #86EFAC); color: #166534; text-shadow: none; }
        .positive-2 { background: linear-gradient(135deg, #86EFAC, #4ADE80); color: #166534; text-shadow: none; }
        .positive-3 { background: linear-gradient(135deg, #4ADE80, #22C55E); color: #FFFFFF; }
        .positive-4 { background: linear-gradient(135deg, #22C55E, #16A34A); color: #FFFFFF; }
        .positive-5 { background: linear-gradient(135deg, #16A34A, #15803D); color: #FFFFFF; }

        /* Color scale - Negative (red) - 5 levels */
        .negative-1 { background: linear-gradient(135deg, #FECACA, #FCA5A5); color: #991B1B; text-shadow: none; }
        .negative-2 { background: linear-gradient(135deg, #FCA5A5, #F87171); color: #991B1B; text-shadow: none; }
        .negative-3 { background: linear-gradient(135deg, #F87171, #EF4444); color: #FFFFFF; }
        .negative-4 { background: linear-gradient(135deg, #EF4444, #DC2626); color: #FFFFFF; }
        .negative-5 { background: linear-gradient(135deg, #DC2626, #B91C1C); color: #FFFFFF; }
    </style>
</head>
<body>
    <div class="table-container">
        <!-- Header -->
        <div class="header-row">
            <div class="market-col"></div>
            <div class="period-col ytd">YTD</div>
            <div class="period-col">1M</div>
            <div class="period-col">3M</div>
            <div class="period-col">6M</div>
            <div class="period-col">12M</div>
        </div>

        {% for row in rows %}
        <div class="data-row">
            <div class="market-col">
                {{ row.flag_html | safe }}
                <span class="credit-badge {{ row.credit_class }}">{{ row.credit_type }}</span>
                <span class="market-name">{{ row.name }}</span>
            </div>
            <div class="value-cell ytd {{ row.ytd_class }}">{{ row.ytd_formatted }}</div>
            <div class="value-cell {{ row.m1_class }}">{{ row.m1_formatted }}</div>
            <div class="value-cell {{ row.m3_class }}">{{ row.m3_formatted }}</div>
            <div class="value-cell {{ row.m6_class }}">{{ row.m6_formatted }}</div>
            <div class="value-cell {{ row.m12_class }}">{{ row.m12_formatted }}</div>
        </div>
        {% endfor %}
    </div>
</body>
</html>
'''

# Commodities Weekly Performance HTML template
COMMODITIES_WEEKLY_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        @import url('https://fonts.cdnfonts.com/css/calibri-light');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Calibri', Calibri, 'Segoe UI', Arial, sans-serif;
            background: #FFFFFF;
            width: {{ width }}px;
            height: {{ height }}px;
            padding: {{ 10 * scale }}px;
        }

        .chart-container {
            display: flex;
            flex-direction: column;
            gap: {{ 1 * scale }}px;
        }

        /* Category group header - matches country-header */
        .category-header {
            display: flex;
            align-items: center;
            gap: {{ 6 * scale }}px;
            padding: {{ 4 * scale }}px {{ 10 * scale }}px;
            margin-top: {{ 3 * scale }}px;
            background: #F8FAFC;
            border-left: {{ 3 * scale }}px solid #1B3A5A;
            border-radius: 0 {{ 4 * scale }}px {{ 4 * scale }}px 0;
        }

        .category-header:first-child {
            margin-top: 0;
        }

        .category-header .icon {
            font-size: {{ 14 * scale }}px;
        }

        .category-header .category-name {
            font-size: {{ 10 * scale }}px;
            font-weight: 700;
            color: #1B3A5A;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Data rows - matches tenor rows */
        .row {
            display: flex;
            align-items: center;
            height: {{ 20 * scale }}px;
            padding: 0 {{ 10 * scale }}px 0 {{ 24 * scale }}px;
        }

        .row:nth-child(odd) {
            background: rgba(248, 250, 252, 0.5);
        }

        /* Commodity label - matches tenor */
        .commodity-info {
            display: flex;
            align-items: center;
            gap: {{ 6 * scale }}px;
            width: {{ 110 * scale }}px;
        }

        .commodity-info .icon {
            font-size: {{ 12 * scale }}px;
        }

        .commodity-info .name {
            font-size: {{ 9 * scale }}px;
            font-weight: 500;
            color: #64748B;
        }

        /* Price column */
        .price {
            width: {{ 85 * scale }}px;
            flex-shrink: 0;
            text-align: left;
            font-size: {{ 9 * scale }}px;
            font-weight: 400;
            color: #94A3B8;
            padding-left: {{ 4 * scale }}px;
        }

        /* Bar container */
        .bar-container {
            flex: 1;
            display: flex;
            align-items: center;
            padding: 0 {{ 12 * scale }}px;
            position: relative;
        }

        .bar-track {
            width: 100%;
            height: {{ 5 * scale }}px;
            background: #F1F5F9;
            border-radius: {{ 3 * scale }}px;
            position: relative;
        }

        /* Center line (zero axis) */
        .center-line {
            position: absolute;
            left: 50%;
            top: {{ -3 * scale }}px;
            bottom: {{ -3 * scale }}px;
            width: {{ 2 * scale }}px;
            background: #CBD5E1;
            border-radius: {{ 1 * scale }}px;
            z-index: 10;
        }

        /* Bars */
        .bar {
            position: absolute;
            height: {{ 8 * scale }}px;
            top: 50%;
            transform: translateY(-50%);
            z-index: 2;
        }

        .bar.positive {
            left: calc(50% + 1px);
            background: linear-gradient(90deg, #4ADE80, #16A34A);
            box-shadow: 0 {{ 2 * scale }}px {{ 4 * scale }}px rgba(34, 197, 94, 0.25);
            border-radius: 0 {{ 5 * scale }}px {{ 5 * scale }}px 0;
        }

        .bar.negative {
            right: calc(50% + 1px);
            background: linear-gradient(270deg, #F87171, #DC2626);
            box-shadow: 0 {{ 2 * scale }}px {{ 4 * scale }}px rgba(239, 68, 68, 0.25);
            border-radius: {{ 5 * scale }}px 0 0 {{ 5 * scale }}px;
        }

        /* Performance value */
        .performance {
            width: {{ 55 * scale }}px;
            text-align: right;
            font-size: {{ 10 * scale }}px;
            font-weight: 700;
        }

        .performance.positive { color: #16A34A; }
        .performance.negative { color: #DC2626; }

        /* Scale */
        .scale {
            display: flex;
            justify-content: center;
            padding: {{ 8 * scale }}px 0 0 0;
            margin-left: {{ 223 * scale }}px;
            margin-right: {{ 55 * scale }}px;
        }

        .scale-inner {
            display: flex;
            justify-content: space-between;
            width: 100%;
            padding: 0 {{ 12 * scale }}px;
            font-size: {{ 7 * scale }}px;
            color: #94A3B8;
        }
    </style>
</head>
<body>
    <div class="chart-container">
        {% for category in categories %}
        <!-- {{ category.name }} -->
        <div class="category-header">
            <span class="icon">{{ category.icon }}</span>
            <span class="category-name">{{ category.name }}</span>
        </div>
        {% for item in category.commodities %}
        <div class="row">
            <div class="commodity-info">
                <span class="icon">{{ item.icon }}</span>
                <span class="name">{{ item.name }}</span>
            </div>
            <div class="price">{{ item.formatted_price|default('') }}</div>
            <div class="bar-container">
                <div class="bar-track">
                    <div class="center-line"></div>
                    {% if item.value != 0 %}
                    <div class="bar {{ item.bar_class }}" style="width: {{ item.bar_width }}%;"></div>
                    {% endif %}
                </div>
            </div>
            <div class="performance {{ item.value_class }}">{{ item.formatted_value }}</div>
        </div>
        {% endfor %}
        {% endfor %}
    </div>

    <div class="scale">
        <div class="scale-inner">
            <span>{{ scale_min }}</span>
            <span>{{ scale_mid_low }}</span>
            <span>0</span>
            <span>{{ scale_mid_high }}</span>
            <span>{{ scale_max }}</span>
        </div>
    </div>
</body>
</html>
'''


# =============================================================================
# COMMODITIES HISTORICAL PERFORMANCE HEATMAP TEMPLATE
# =============================================================================

COMMODITIES_HISTORICAL_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        @import url('https://fonts.cdnfonts.com/css/calibri-light');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Calibri', Calibri, 'Segoe UI', Arial, sans-serif;
            background: #FFFFFF;
            width: {{ width }}px;
            height: {{ height }}px;
            padding: {{ 8 * scale }}px;
        }

        .table-container {
            display: flex;
            flex-direction: column;
            gap: {{ 1 * scale }}px;
        }

        /* Header row */
        .header-row {
            display: flex;
            align-items: center;
            padding: {{ 4 * scale }}px 0;
        }

        .header-row .market-col {
            width: {{ 110 * scale }}px;
        }

        .header-row .period-col {
            flex: 1;
            text-align: center;
            font-size: {{ 8 * scale }}px;
            font-weight: 600;
            color: #1B3A5A;
        }

        .header-row .period-col.ytd {
            flex: 1.3;
            font-size: {{ 9 * scale }}px;
            font-weight: 700;
            color: #1B3A5A;
            margin-right: {{ 8 * scale }}px;
        }

        /* Category group header */
        .category-header {
            display: flex;
            align-items: center;
            gap: {{ 5 * scale }}px;
            padding: {{ 3 * scale }}px {{ 8 * scale }}px;
            margin-top: {{ 3 * scale }}px;
            background: #F8FAFC;
            border-left: {{ 2 * scale }}px solid #1B3A5A;
            border-radius: 0 {{ 3 * scale }}px {{ 3 * scale }}px 0;
        }

        .category-header:first-of-type {
            margin-top: 0;
        }

        .category-header .category-icon {
            font-size: {{ 9 * scale }}px;
        }

        .category-header .category-name {
            font-size: {{ 7 * scale }}px;
            font-weight: 700;
            color: #1B3A5A;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Data rows */
        .data-row {
            display: flex;
            align-items: center;
            gap: {{ 2 * scale }}px;
            padding-left: {{ 12 * scale }}px;
        }

        .commodity-col {
            width: {{ 98 * scale }}px;
            display: flex;
            align-items: center;
            gap: {{ 4 * scale }}px;
            padding-right: {{ 4 * scale }}px;
        }

        .commodity-icon {
            font-size: {{ 9 * scale }}px;
        }

        .commodity-name {
            font-size: {{ 7 * scale }}px;
            font-weight: 500;
            color: #334155;
        }

        /* Value cells */
        .value-cell {
            flex: 1;
            height: {{ 18 * scale }}px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: {{ 3 * scale }}px;
            font-size: {{ 6 * scale }}px;
            font-weight: 600;
            color: #FFFFFF;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }

        /* YTD column - emphasized */
        .value-cell.ytd {
            flex: 1.3;
            height: {{ 22 * scale }}px;
            font-size: {{ 8 * scale }}px;
            font-weight: 700;
            margin-right: {{ 8 * scale }}px;
            border-radius: {{ 4 * scale }}px;
            box-shadow: 0 {{ 2 * scale }}px {{ 6 * scale }}px rgba(0,0,0,0.15), 0 {{ 1 * scale }}px {{ 2 * scale }}px rgba(0,0,0,0.1);
        }

        /* Color scale - Positive (green) - 5 levels */
        .positive-1 { background: linear-gradient(135deg, #BBF7D0, #86EFAC); color: #166534; text-shadow: none; }
        .positive-2 { background: linear-gradient(135deg, #86EFAC, #4ADE80); color: #166534; text-shadow: none; }
        .positive-3 { background: linear-gradient(135deg, #4ADE80, #22C55E); color: #FFFFFF; }
        .positive-4 { background: linear-gradient(135deg, #22C55E, #16A34A); color: #FFFFFF; }
        .positive-5 { background: linear-gradient(135deg, #16A34A, #15803D); color: #FFFFFF; }

        /* Color scale - Negative (red) - 5 levels */
        .negative-1 { background: linear-gradient(135deg, #FECACA, #FCA5A5); color: #991B1B; text-shadow: none; }
        .negative-2 { background: linear-gradient(135deg, #FCA5A5, #F87171); color: #991B1B; text-shadow: none; }
        .negative-3 { background: linear-gradient(135deg, #F87171, #EF4444); color: #FFFFFF; }
        .negative-4 { background: linear-gradient(135deg, #EF4444, #DC2626); color: #FFFFFF; }
        .negative-5 { background: linear-gradient(135deg, #DC2626, #B91C1C); color: #FFFFFF; }
    </style>
</head>
<body>
    <div class="table-container">
        <!-- Header -->
        <div class="header-row">
            <div class="market-col"></div>
            <div class="period-col ytd">YTD</div>
            <div class="period-col">1M</div>
            <div class="period-col">3M</div>
            <div class="period-col">6M</div>
            <div class="period-col">12M</div>
        </div>

        {% for category in categories %}
        <!-- {{ category.name }} -->
        <div class="category-header">
            <span class="category-icon">{{ category.icon }}</span>
            <span class="category-name">{{ category.name }}</span>
        </div>
        {% for item in category.commodities %}
        <div class="data-row">
            <div class="commodity-col">
                <span class="commodity-icon">{{ item.icon }}</span>
                <span class="commodity-name">{{ item.name }}</span>
            </div>
            <div class="value-cell ytd {{ item.ytd_class }}">{{ item.ytd_formatted }}</div>
            <div class="value-cell {{ item.m1_class }}">{{ item.m1_formatted }}</div>
            <div class="value-cell {{ item.m3_class }}">{{ item.m3_formatted }}</div>
            <div class="value-cell {{ item.m6_class }}">{{ item.m6_formatted }}</div>
            <div class="value-cell {{ item.m12_class }}">{{ item.m12_formatted }}</div>
        </div>
        {% endfor %}
        {% endfor %}
    </div>
</body>
</html>
'''


# =============================================================================
# CURRENCY WEEKLY PERFORMANCE TEMPLATE
# =============================================================================

CURRENCY_WEEKLY_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    
    <style>
        @import url('https://fonts.cdnfonts.com/css/calibri-light');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Calibri', Calibri, 'Segoe UI', Arial, sans-serif;
            background: #FFFFFF;
            width: {{ width }}px;
            height: {{ height }}px;
            padding: {{ 10 * scale }}px;
        }

        .chart-container {
            display: flex;
            flex-direction: column;
            gap: {{ 6 * scale }}px;
        }

        /* Data rows */
        .row {
            display: flex;
            align-items: center;
            height: {{ 38 * scale }}px;
            padding: 0 {{ 12 * scale }}px;
            border-radius: {{ 6 * scale }}px;
        }

        .row:nth-child(even) {
            background: #F8FAFC;
        }

        /* Top performer highlight */
        .row.top-performer {
            background: linear-gradient(90deg, rgba(201, 162, 39, 0.1), transparent);
            border-left: {{ 3 * scale }}px solid #C9A227;
        }

        /* Worst performer highlight */
        .row.worst-performer {
            background: linear-gradient(90deg, rgba(239, 68, 68, 0.1), transparent);
            border-left: {{ 3 * scale }}px solid #EF4444;
        }

        /* Currency label */
        .currency-col {
            width: {{ 120 * scale }}px;
            display: flex;
            align-items: center;
            gap: {{ 10 * scale }}px;
        }

        .flag {
            font-size: {{ 22 * scale }}px;
            line-height: 1;
        }

        /* Flag images */
        .currency-col img,
        .currency-col .flag-img {
            width: {{ 22 * scale }}px !important;
            height: {{ 14 * scale }}px !important;
            vertical-align: middle;
            margin-right: {{ 4 * scale }}px;
            flex-shrink: 0;
        }

        .currency-name {
            font-size: {{ 11 * scale }}px;
            font-weight: 600;
            color: #334155;
        }

        /* Rate column */
        .rate {
            width: {{ 70 * scale }}px;
            flex-shrink: 0;
            text-align: left;
            font-size: {{ 9 * scale }}px;
            font-weight: 400;
            color: #94A3B8;
            padding-left: {{ 4 * scale }}px;
        }

        /* Bar container */
        .bar-container {
            flex: 1;
            display: flex;
            align-items: center;
            padding: 0 {{ 15 * scale }}px;
            position: relative;
        }

        .bar-track {
            width: 100%;
            height: {{ 6 * scale }}px;
            background: #F1F5F9;
            border-radius: {{ 3 * scale }}px;
            position: relative;
        }

        /* Center line */
        .center-line {
            position: absolute;
            left: 50%;
            top: {{ -4 * scale }}px;
            bottom: {{ -4 * scale }}px;
            width: {{ 2 * scale }}px;
            background: #CBD5E1;
            border-radius: {{ 1 * scale }}px;
            z-index: 10;
        }

        /* Bars */
        .bar {
            position: absolute;
            height: {{ 14 * scale }}px;
            top: 50%;
            transform: translateY(-50%);
            z-index: 2;
        }

        .bar.positive {
            left: calc(50% + 1px);
            background: linear-gradient(90deg, #4ADE80, #16A34A);
            box-shadow: 0 {{ 2 * scale }}px {{ 4 * scale }}px rgba(34, 197, 94, 0.25);
            border-radius: 0 {{ 7 * scale }}px {{ 7 * scale }}px 0;
        }

        .bar.negative {
            right: calc(50% + 1px);
            background: linear-gradient(270deg, #F87171, #DC2626);
            box-shadow: 0 {{ 2 * scale }}px {{ 4 * scale }}px rgba(239, 68, 68, 0.25);
            border-radius: {{ 7 * scale }}px 0 0 {{ 7 * scale }}px;
        }

        /* Performance value */
        .performance {
            width: {{ 60 * scale }}px;
            text-align: right;
            font-size: {{ 11 * scale }}px;
            font-weight: 700;
        }

        .performance.positive { color: #16A34A; }
        .performance.negative { color: #DC2626; }

        /* Scale */
        .scale {
            display: flex;
            justify-content: center;
            padding: {{ 10 * scale }}px 0 0 0;
            margin-left: {{ 206 * scale }}px;
            margin-right: {{ 60 * scale }}px;
        }

        .scale-inner {
            display: flex;
            justify-content: space-between;
            width: 100%;
            padding: 0 {{ 15 * scale }}px;
            font-size: {{ 8 * scale }}px;
            color: #94A3B8;
        }
    </style>
</head>
<body>
    <div class="chart-container">
        {% for row in rows %}
        <div class="row {{ row.highlight_class }}">
            <div class="currency-col">
                {{ row.flag_html | safe }}
                <span class="currency-name">{{ row.name }}</span>
            </div>
            <div class="rate">{{ row.formatted_rate|default('') }}</div>
            <div class="bar-container">
                <div class="bar-track">
                    <div class="center-line"></div>
                    {% if row.value != 0 %}
                    <div class="bar {{ row.bar_class }}" style="width: {{ row.bar_width }}%;"></div>
                    {% endif %}
                </div>
            </div>
            <div class="performance {{ row.value_class }}">{{ row.formatted_value }}</div>
        </div>
        {% endfor %}
    </div>

    <div class="scale">
        <div class="scale-inner">
            <span>{{ scale_min }}</span>
            <span>{{ scale_mid_low }}</span>
            <span>0</span>
            <span>{{ scale_mid_high }}</span>
            <span>{{ scale_max }}</span>
        </div>
    </div>
</body>
</html>
'''


# =============================================================================
# CURRENCY HISTORICAL PERFORMANCE HEATMAP TEMPLATE
# =============================================================================

CURRENCY_HISTORICAL_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    
    <style>
        @import url('https://fonts.cdnfonts.com/css/calibri-light');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Calibri', Calibri, 'Segoe UI', Arial, sans-serif;
            background: #FFFFFF;
            width: {{ width }}px;
            height: {{ height }}px;
            padding: {{ 10 * scale }}px;
        }

        .table-container {
            display: flex;
            flex-direction: column;
            gap: {{ 4 * scale }}px;
        }

        /* Header row */
        .header-row {
            display: flex;
            align-items: center;
            padding: {{ 6 * scale }}px 0;
        }

        .header-row .currency-col {
            width: {{ 120 * scale }}px;
        }

        .header-row .period-col {
            flex: 1;
            text-align: center;
            font-size: {{ 9 * scale }}px;
            font-weight: 600;
            color: #1B3A5A;
        }

        .header-row .period-col.ytd {
            flex: 1.3;
            font-size: {{ 10 * scale }}px;
            font-weight: 700;
            color: #1B3A5A;
            margin-right: {{ 12 * scale }}px;
        }

        /* Data rows */
        .data-row {
            display: flex;
            align-items: center;
            gap: {{ 3 * scale }}px;
        }

        .currency-col {
            width: {{ 120 * scale }}px;
            display: flex;
            align-items: center;
            gap: {{ 8 * scale }}px;
            padding-right: {{ 8 * scale }}px;
        }

        .flag {
            font-size: {{ 22 * scale }}px;
            line-height: 1;
        }

        /* Flag images */
        .currency-col img,
        .currency-col .flag-img {
            width: {{ 22 * scale }}px !important;
            height: {{ 14 * scale }}px !important;
            vertical-align: middle;
            margin-right: {{ 4 * scale }}px;
            flex-shrink: 0;
        }

        .currency-name {
            font-size: {{ 10 * scale }}px;
            font-weight: 600;
            color: #334155;
        }

        /* Value cells */
        .value-cell {
            flex: 1;
            height: {{ 32 * scale }}px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: {{ 5 * scale }}px;
            font-size: {{ 10 * scale }}px;
            font-weight: 600;
            color: #FFFFFF;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }

        /* YTD column - emphasized */
        .value-cell.ytd {
            flex: 1.3;
            height: {{ 36 * scale }}px;
            font-size: {{ 12 * scale }}px;
            font-weight: 700;
            margin-right: {{ 12 * scale }}px;
            border-radius: {{ 6 * scale }}px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15), 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Color scale - Positive (green) - 5 levels */
        .positive-1 { background: linear-gradient(135deg, #BBF7D0, #86EFAC); color: #166534; text-shadow: none; }
        .positive-2 { background: linear-gradient(135deg, #86EFAC, #4ADE80); color: #166534; text-shadow: none; }
        .positive-3 { background: linear-gradient(135deg, #4ADE80, #22C55E); color: #FFFFFF; }
        .positive-4 { background: linear-gradient(135deg, #22C55E, #16A34A); color: #FFFFFF; }
        .positive-5 { background: linear-gradient(135deg, #16A34A, #15803D); color: #FFFFFF; }

        /* Color scale - Negative (red) - 5 levels */
        .negative-1 { background: linear-gradient(135deg, #FECACA, #FCA5A5); color: #991B1B; text-shadow: none; }
        .negative-2 { background: linear-gradient(135deg, #FCA5A5, #F87171); color: #991B1B; text-shadow: none; }
        .negative-3 { background: linear-gradient(135deg, #F87171, #EF4444); color: #FFFFFF; }
        .negative-4 { background: linear-gradient(135deg, #EF4444, #DC2626); color: #FFFFFF; }
        .negative-5 { background: linear-gradient(135deg, #DC2626, #B91C1C); color: #FFFFFF; }

        /* Neutral (near zero) */
        .neutral { background: #F1F5F9; color: #64748B; text-shadow: none; }
    </style>
</head>
<body>
    <div class="table-container">
        <!-- Header -->
        <div class="header-row">
            <div class="currency-col"></div>
            <div class="period-col ytd">YTD</div>
            <div class="period-col">1M</div>
            <div class="period-col">3M</div>
            <div class="period-col">6M</div>
            <div class="period-col">12M</div>
        </div>

        {% for row in rows %}
        <div class="data-row">
            <div class="currency-col">
                {{ row.flag_html | safe }}
                <span class="currency-name">{{ row.name }}</span>
            </div>
            <div class="value-cell ytd {{ row.ytd_class }}">{{ row.ytd_formatted }}</div>
            <div class="value-cell {{ row.m1_class }}">{{ row.m1_formatted }}</div>
            <div class="value-cell {{ row.m3_class }}">{{ row.m3_formatted }}</div>
            <div class="value-cell {{ row.m6_class }}">{{ row.m6_formatted }}</div>
            <div class="value-cell {{ row.m12_class }}">{{ row.m12_formatted }}</div>
        </div>
        {% endfor %}
    </div>
</body>
</html>
'''


# =============================================================================
# CRYPTO WEEKLY PERFORMANCE TEMPLATE
# =============================================================================

CRYPTO_WEEKLY_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        @import url('https://fonts.cdnfonts.com/css/calibri-light');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Calibri', Calibri, 'Segoe UI', Arial, sans-serif;
            background: #FFFFFF;
            width: {{ width }}px;
            height: {{ height }}px;
            padding: {{ 10 * scale }}px;
        }

        .chart-container {
            display: flex;
            flex-direction: column;
            gap: {{ 3 * scale }}px;
        }

        /* Data rows */
        .row {
            display: flex;
            align-items: center;
            height: {{ 28 * scale }}px;
            padding: 0 {{ 10 * scale }}px;
            border-radius: {{ 5 * scale }}px;
        }

        .row:nth-child(even) {
            background: #F8FAFC;
        }

        /* Top performer highlight */
        .row.top-performer {
            background: linear-gradient(90deg, rgba(201, 162, 39, 0.1), transparent);
            border-left: {{ 3 * scale }}px solid #C9A227;
        }

        /* Worst performer highlight */
        .row.worst-performer {
            background: linear-gradient(90deg, rgba(239, 68, 68, 0.1), transparent);
            border-left: {{ 3 * scale }}px solid #EF4444;
        }

        /* Crypto label */
        .crypto-col {
            width: {{ 160 * scale }}px;
            display: flex;
            align-items: center;
            gap: {{ 6 * scale }}px;
        }

        .crypto-name {
            font-size: {{ 10 * scale }}px;
            font-weight: 600;
            color: #334155;
        }

        /* Crypto logo */
        .flag-img {
            width: {{ 22 * scale }}px;
            height: {{ 22 * scale }}px;
            border-radius: 50%;
            vertical-align: middle;
            flex-shrink: 0;
            object-fit: cover;
        }

        /* Price column */
        .price {
            width: {{ 85 * scale }}px;
            flex-shrink: 0;
            text-align: left;
            font-size: {{ 9 * scale }}px;
            font-weight: 400;
            color: #94A3B8;
            padding-left: {{ 4 * scale }}px;
        }

        /* Bar container */
        .bar-container {
            flex: 1;
            display: flex;
            align-items: center;
            padding: 0 {{ 12 * scale }}px;
            position: relative;
        }

        .bar-track {
            width: 100%;
            height: {{ 5 * scale }}px;
            background: #F1F5F9;
            border-radius: {{ 3 * scale }}px;
            position: relative;
        }

        /* Center line */
        .center-line {
            position: absolute;
            left: 50%;
            top: {{ -3 * scale }}px;
            bottom: {{ -3 * scale }}px;
            width: {{ 2 * scale }}px;
            background: #CBD5E1;
            border-radius: {{ 1 * scale }}px;
            z-index: 10;
        }

        /* Bars */
        .bar {
            position: absolute;
            height: {{ 10 * scale }}px;
            top: 50%;
            transform: translateY(-50%);
            z-index: 2;
        }

        .bar.positive {
            left: calc(50% + 1px);
            background: linear-gradient(90deg, #4ADE80, #16A34A);
            box-shadow: 0 {{ 2 * scale }}px {{ 4 * scale }}px rgba(34, 197, 94, 0.25);
            border-radius: 0 {{ 5 * scale }}px {{ 5 * scale }}px 0;
        }

        .bar.negative {
            right: calc(50% + 1px);
            background: linear-gradient(270deg, #F87171, #DC2626);
            box-shadow: 0 {{ 2 * scale }}px {{ 4 * scale }}px rgba(239, 68, 68, 0.25);
            border-radius: {{ 5 * scale }}px 0 0 {{ 5 * scale }}px;
        }

        /* Performance value */
        .performance {
            width: {{ 60 * scale }}px;
            text-align: right;
            font-size: {{ 10 * scale }}px;
            font-weight: 700;
        }

        .performance.positive { color: #16A34A; }
        .performance.negative { color: #DC2626; }

        /* Scale */
        .scale {
            display: flex;
            justify-content: center;
            padding: {{ 8 * scale }}px 0 0 0;
            margin-left: {{ 261 * scale }}px;
            margin-right: {{ 60 * scale }}px;
        }

        .scale-inner {
            display: flex;
            justify-content: space-between;
            width: 100%;
            padding: 0 {{ 12 * scale }}px;
            font-size: {{ 7 * scale }}px;
            color: #94A3B8;
        }
    </style>
</head>
<body>
    <div class="chart-container">
        {% for row in rows %}
        <div class="row {{ row.highlight_class }}">
            <div class="crypto-col">
                {% if row.flag_html %}{{ row.flag_html | safe }}{% endif %}
                <span class="crypto-name">{{ row.name }}</span>
            </div>
            <div class="price">{{ row.formatted_price|default('') }}</div>
            <div class="bar-container">
                <div class="bar-track">
                    <div class="center-line"></div>
                    {% if row.value != 0 %}
                    <div class="bar {{ row.bar_class }}" style="width: {{ row.bar_width }}%;"></div>
                    {% endif %}
                </div>
            </div>
            <div class="performance {{ row.value_class }}">{{ row.formatted_value }}</div>
        </div>
        {% endfor %}
    </div>

    <div class="scale">
        <div class="scale-inner">
            <span>{{ scale_min }}</span>
            <span>{{ scale_mid_low }}</span>
            <span>0</span>
            <span>{{ scale_mid_high }}</span>
            <span>{{ scale_max }}</span>
        </div>
    </div>
</body>
</html>
'''


# =============================================================================
# CRYPTO HISTORICAL PERFORMANCE HEATMAP TEMPLATE
# =============================================================================

CRYPTO_HISTORICAL_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        @import url('https://fonts.cdnfonts.com/css/calibri-light');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Calibri', Calibri, 'Segoe UI', Arial, sans-serif;
            background: #FFFFFF;
            width: {{ width }}px;
            height: {{ height }}px;
            padding: {{ 10 * scale }}px;
        }

        .table-container {
            display: flex;
            flex-direction: column;
            gap: {{ 2 * scale }}px;
        }

        /* Header row */
        .header-row {
            display: flex;
            align-items: center;
            padding: {{ 4 * scale }}px 0;
        }

        .header-row .market-col {
            width: {{ 160 * scale }}px;
        }

        .header-row .period-col {
            flex: 1;
            text-align: center;
            font-size: {{ 9 * scale }}px;
            font-weight: 600;
            color: #1B3A5A;
        }

        .header-row .period-col.ytd {
            flex: 1.3;
            font-size: {{ 10 * scale }}px;
            font-weight: 700;
            color: #1B3A5A;
            margin-right: {{ 10 * scale }}px;
        }

        /* Data rows */
        .data-row {
            display: flex;
            align-items: center;
            gap: {{ 2 * scale }}px;
        }

        .crypto-col {
            width: {{ 160 * scale }}px;
            display: flex;
            align-items: center;
            gap: {{ 6 * scale }}px;
            padding-right: {{ 10 * scale }}px;
        }

        .crypto-name {
            font-size: {{ 9 * scale }}px;
            font-weight: 600;
            color: #334155;
        }

        /* Crypto logo */
        .flag-img {
            width: {{ 20 * scale }}px;
            height: {{ 20 * scale }}px;
            border-radius: 50%;
            vertical-align: middle;
            flex-shrink: 0;
            object-fit: cover;
        }

        /* Value cells */
        .value-cell {
            flex: 1;
            height: {{ 24 * scale }}px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: {{ 4 * scale }}px;
            font-size: {{ 7 * scale }}px;
            font-weight: 600;
            color: #FFFFFF;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }

        /* YTD column - emphasized */
        .value-cell.ytd {
            flex: 1.3;
            height: {{ 28 * scale }}px;
            font-size: {{ 9 * scale }}px;
            font-weight: 700;
            margin-right: {{ 10 * scale }}px;
            border-radius: {{ 5 * scale }}px;
            box-shadow: 0 {{ 2 * scale }}px {{ 8 * scale }}px rgba(0,0,0,0.15), 0 {{ 1 * scale }}px {{ 3 * scale }}px rgba(0,0,0,0.1);
        }

        /* Color scale - Positive (green) - 5 levels */
        .positive-1 { background: linear-gradient(135deg, #BBF7D0, #86EFAC); color: #166534; text-shadow: none; }
        .positive-2 { background: linear-gradient(135deg, #86EFAC, #4ADE80); color: #166534; text-shadow: none; }
        .positive-3 { background: linear-gradient(135deg, #4ADE80, #22C55E); color: #FFFFFF; }
        .positive-4 { background: linear-gradient(135deg, #22C55E, #16A34A); color: #FFFFFF; }
        .positive-5 { background: linear-gradient(135deg, #16A34A, #15803D); color: #FFFFFF; }

        /* Color scale - Negative (red) - 5 levels */
        .negative-1 { background: linear-gradient(135deg, #FECACA, #FCA5A5); color: #991B1B; text-shadow: none; }
        .negative-2 { background: linear-gradient(135deg, #FCA5A5, #F87171); color: #991B1B; text-shadow: none; }
        .negative-3 { background: linear-gradient(135deg, #F87171, #EF4444); color: #FFFFFF; }
        .negative-4 { background: linear-gradient(135deg, #EF4444, #DC2626); color: #FFFFFF; }
        .negative-5 { background: linear-gradient(135deg, #DC2626, #B91C1C); color: #FFFFFF; }
    </style>
</head>
<body>
    <div class="table-container">
        <!-- Header -->
        <div class="header-row">
            <div class="market-col"></div>
            <div class="period-col ytd">YTD</div>
            <div class="period-col">1M</div>
            <div class="period-col">3M</div>
            <div class="period-col">6M</div>
            <div class="period-col">12M</div>
        </div>

        {% for row in rows %}
        <div class="data-row">
            <div class="crypto-col">
                {% if row.flag_html %}{{ row.flag_html | safe }}{% endif %}
                <span class="crypto-name">{{ row.name }}</span>
            </div>
            <div class="value-cell ytd {{ row.ytd_class }}">{{ row.ytd_formatted }}</div>
            <div class="value-cell {{ row.m1_class }}">{{ row.m1_formatted }}</div>
            <div class="value-cell {{ row.m3_class }}">{{ row.m3_formatted }}</div>
            <div class="value-cell {{ row.m6_class }}">{{ row.m6_formatted }}</div>
            <div class="value-cell {{ row.m12_class }}">{{ row.m12_formatted }}</div>
        </div>
        {% endfor %}
    </div>
</body>
</html>
'''


# =============================================================================
# EQUITY YTD EVOLUTION CHART TEMPLATE (Chart.js Line Chart)
# =============================================================================

EQUITY_YTD_EVOLUTION_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.cdnfonts.com/css/calibri-light');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Calibri', Calibri, 'Segoe UI', Arial, sans-serif;
            background: #FFFFFF;
            width: {{ width }}px;
            height: {{ height }}px;
            padding: {{ 15 * scale }}px;
        }

        .chart-wrapper {
            position: relative;
            width: 100%;
            height: 100%;
        }

        .chart-title {
            text-align: center;
            font-size: {{ 14 * scale }}px;
            font-weight: 600;
            color: #1B3A5A;
            margin-bottom: {{ 10 * scale }}px;
        }

        .chart-container {
            position: relative;
            width: 100%;
            height: calc(100% - {{ 30 * scale }}px);
        }
    </style>
</head>
<body>
    <div class="chart-wrapper">
        <div class="chart-title">{{ chart_title }}</div>
        <div class="chart-container">
            <canvas id="ytdChart"></canvas>
        </div>
    </div>

    <script>
        const labels = {{ labels | tojson }};
        const datasets = {{ datasets | tojson }};
        const scale = {{ scale }};
        const yMin = {{ y_min }};
        const yMax = {{ y_max }};

        // ============================================
        // LABEL COLLISION AVOIDANCE ALGORITHM
        // ============================================
        function resolveOverlaps(labels, minGap) {
            // Sort by y position (top to bottom)
            labels.sort((a, b) => a.y - b.y);

            // Push down overlapping labels
            for (let i = 1; i < labels.length; i++) {
                const prev = labels[i - 1];
                const curr = labels[i];
                const gap = curr.y - prev.y;

                if (gap < minGap) {
                    curr.y = prev.y + minGap;
                    curr.adjusted = true;
                }
            }

            return labels;
        }

        // ============================================
        // CHART CONFIGURATION
        // ============================================
        const ctx = document.getElementById('ytdChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets.map(d => ({
                    ...d,
                    borderWidth: 2.5 * scale / 3,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    tension: 0.3,
                    fill: false,
                }))
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
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        enabled: true,
                        backgroundColor: 'rgba(27, 58, 90, 0.9)',
                        titleFont: { family: 'Calibri', size: 10 * scale },
                        bodyFont: { family: 'Calibri', size: 9 * scale },
                        padding: 8 * scale,
                        cornerRadius: 4 * scale,
                        callbacks: {
                            label: function(context) {
                                const value = context.parsed.y;
                                const sign = value >= 0 ? '+' : '';
                                return `${context.dataset.label}: ${sign}${value.toFixed(1)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            display: false,
                        },
                        ticks: {
                            font: { family: 'Calibri', size: 9 * scale, weight: '500' },
                            color: '#64748B',
                            autoSkip: false,
                            maxRotation: 0,
                            callback: function(value, index) {
                                // Only show the first occurrence of each month
                                const label = labels[index];
                                if (index === 0) return label;
                                const prevLabel = labels[index - 1];
                                return label !== prevLabel ? label : '';
                            }
                        }
                    },
                    y: {
                        min: yMin,
                        max: yMax,
                        grid: {
                            display: true,
                            color: function(context) {
                                if (context.tick.value === 0) {
                                    return 'rgba(0, 0, 0, 0.3)';
                                }
                                return 'transparent';
                            },
                            lineWidth: function(context) {
                                if (context.tick.value === 0) {
                                    return 2;
                                }
                                return 0;
                            },
                            drawBorder: false,
                        },
                        ticks: {
                            font: { family: 'Calibri', size: 9 * scale, weight: '500' },
                            color: '#64748B',
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                layout: {
                    padding: {
                        right: 110 * scale
                    }
                }
            },
            plugins: [{
                id: 'endLabelsNoOverlap',
                afterDraw: function(chart) {
                    const ctx = chart.ctx;
                    const minGap = 16 * scale;

                    // Collect label positions
                    let labelData = [];

                    chart.data.datasets.forEach((dataset, i) => {
                        const meta = chart.getDatasetMeta(i);
                        const lastPoint = meta.data[meta.data.length - 1];

                        if (lastPoint) {
                            const value = dataset.data[dataset.data.length - 1];
                            const sign = value >= 0 ? '+' : '';
                            const label = `${dataset.label}: ${sign}${value.toFixed(1)}%`;

                            labelData.push({
                                x: lastPoint.x,
                                y: lastPoint.y,
                                originalY: lastPoint.y,
                                label: label,
                                color: dataset.borderColor,
                                adjusted: false
                            });
                        }
                    });

                    // Resolve overlaps
                    labelData = resolveOverlaps(labelData, minGap);

                    // Draw labels
                    const fontSize = 9 * scale;
                    labelData.forEach(item => {
                        ctx.save();
                        ctx.font = `600 ${fontSize}px Calibri`;
                        const textWidth = ctx.measureText(item.label).width;
                        const labelX = item.x + 10 * scale;
                        const labelY = item.y;
                        const pillHeight = 14 * scale;
                        const pillPadding = 4 * scale;

                        // Draw connector line if adjusted
                        if (item.adjusted) {
                            ctx.beginPath();
                            ctx.strokeStyle = item.color + '60';
                            ctx.lineWidth = 1;
                            ctx.setLineDash([2 * scale, 2 * scale]);
                            ctx.moveTo(item.x + 2, item.originalY);
                            ctx.lineTo(labelX - 2, labelY);
                            ctx.stroke();
                            ctx.setLineDash([]);
                        }

                        // Draw background pill
                        ctx.fillStyle = item.color + '15';
                        ctx.beginPath();
                        ctx.roundRect(
                            labelX - pillPadding,
                            labelY - pillHeight/2,
                            textWidth + pillPadding * 2.5,
                            pillHeight,
                            3 * scale
                        );
                        ctx.fill();

                        // Draw border
                        ctx.strokeStyle = item.color + '30';
                        ctx.lineWidth = 1;
                        ctx.stroke();

                        // Draw text
                        ctx.fillStyle = item.color;
                        ctx.textBaseline = 'middle';
                        ctx.fillText(item.label, labelX, labelY);
                        ctx.restore();
                    });
                }
            }]
        });
    </script>
</body>
</html>
'''


# =============================================================================
# COMMODITY YTD EVOLUTION LINE CHART TEMPLATE (Chart.js)
# =============================================================================

COMMODITY_YTD_EVOLUTION_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.cdnfonts.com/css/calibri-light');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Calibri', Calibri, 'Segoe UI', Arial, sans-serif;
            background: #FFFFFF;
            width: {{ width }}px;
            height: {{ height }}px;
            padding: {{ 15 * scale }}px;
        }

        .chart-wrapper {
            position: relative;
            width: 100%;
            height: 100%;
        }

        .chart-title {
            text-align: center;
            font-size: {{ 14 * scale }}px;
            font-weight: 600;
            color: #1B3A5A;
            margin-bottom: {{ 10 * scale }}px;
        }

        .chart-container {
            position: relative;
            width: 100%;
            height: calc(100% - {{ 30 * scale }}px);
        }
    </style>
</head>
<body>
    <div class="chart-wrapper">
        <div class="chart-title">{{ chart_title }}</div>
        <div class="chart-container">
            <canvas id="ytdChart"></canvas>
        </div>
    </div>

    <script>
        const labels = {{ labels | tojson }};
        const datasets = {{ datasets | tojson }};
        const scale = {{ scale }};
        const yMin = {{ y_min }};
        const yMax = {{ y_max }};

        // Label collision avoidance
        function resolveOverlaps(labels, minGap) {
            labels.sort((a, b) => a.y - b.y);
            for (let i = 1; i < labels.length; i++) {
                const prev = labels[i - 1];
                const curr = labels[i];
                const gap = curr.y - prev.y;
                if (gap < minGap) {
                    curr.y = prev.y + minGap;
                    curr.adjusted = true;
                }
            }
            return labels;
        }

        const ctx = document.getElementById('ytdChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets.map(d => ({
                    label: d.label,
                    data: d.data,
                    borderColor: d.borderColor,
                    backgroundColor: 'transparent',
                    borderWidth: 2.5 * scale / 3,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    tension: 0.3,
                    fill: false,
                }))
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
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        enabled: true,
                        backgroundColor: 'rgba(27, 58, 90, 0.9)',
                        titleFont: { family: 'Calibri', size: 10 * scale },
                        bodyFont: { family: 'Calibri', size: 9 * scale },
                        padding: 8 * scale,
                        cornerRadius: 4 * scale,
                        callbacks: {
                            label: function(context) {
                                const value = context.parsed.y;
                                const sign = value >= 0 ? '+' : '';
                                return `${context.dataset.label}: ${sign}${value.toFixed(1)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: {
                            font: { family: 'Calibri', size: 9 * scale, weight: '500' },
                            color: '#64748B',
                            autoSkip: false,
                            maxRotation: 0,
                            callback: function(value, index) {
                                const label = labels[index];
                                if (index === 0) return label;
                                const prevLabel = labels[index - 1];
                                return label !== prevLabel ? label : '';
                            }
                        }
                    },
                    y: {
                        min: yMin,
                        max: yMax,
                        grid: {
                            display: true,
                            color: function(context) {
                                if (context.tick.value === 0) {
                                    return 'rgba(0, 0, 0, 0.3)';
                                }
                                return 'transparent';
                            },
                            lineWidth: function(context) {
                                if (context.tick.value === 0) {
                                    return 2;
                                }
                                return 0;
                            },
                            drawBorder: false,
                        },
                        ticks: {
                            font: { family: 'Calibri', size: 9 * scale, weight: '500' },
                            color: '#64748B',
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                layout: {
                    padding: {
                        right: 110 * scale
                    }
                }
            },
            plugins: [{
                id: 'endLabelsNoOverlap',
                afterDraw: function(chart) {
                    const ctx = chart.ctx;
                    const minGap = 16 * scale;

                    let labelData = [];

                    chart.data.datasets.forEach((dataset, i) => {
                        const meta = chart.getDatasetMeta(i);
                        const lastPoint = meta.data[meta.data.length - 1];

                        if (lastPoint) {
                            const value = dataset.data[dataset.data.length - 1];
                            const sign = value >= 0 ? '+' : '';
                            const label = `${dataset.label}: ${sign}${value.toFixed(1)}%`;

                            labelData.push({
                                x: lastPoint.x,
                                y: lastPoint.y,
                                originalY: lastPoint.y,
                                label: label,
                                color: dataset.borderColor,
                                adjusted: false
                            });
                        }
                    });

                    labelData = resolveOverlaps(labelData, minGap);

                    const fontSize = 9 * scale;
                    labelData.forEach(item => {
                        ctx.save();
                        ctx.font = `600 ${fontSize}px Calibri`;
                        const textWidth = ctx.measureText(item.label).width;
                        const labelX = item.x + 10 * scale;
                        const labelY = item.y;
                        const pillHeight = 14 * scale;
                        const pillPadding = 4 * scale;

                        if (item.adjusted) {
                            ctx.beginPath();
                            ctx.strokeStyle = item.color + '60';
                            ctx.lineWidth = 1;
                            ctx.setLineDash([2 * scale, 2 * scale]);
                            ctx.moveTo(item.x + 2, item.originalY);
                            ctx.lineTo(labelX - 2, labelY);
                            ctx.stroke();
                            ctx.setLineDash([]);
                        }

                        ctx.fillStyle = item.color + '20';
                        ctx.beginPath();
                        ctx.roundRect(
                            labelX - pillPadding,
                            labelY - pillHeight/2,
                            textWidth + pillPadding * 2.5,
                            pillHeight,
                            3 * scale
                        );
                        ctx.fill();

                        ctx.strokeStyle = item.color + '40';
                        ctx.lineWidth = 1;
                        ctx.stroke();

                        ctx.fillStyle = item.color;
                        ctx.textBaseline = 'middle';
                        ctx.fillText(item.label, labelX, labelY);
                        ctx.restore();
                    });
                }
            }]
        });
    </script>
</body>
</html>
'''


# =============================================================================
# CRYPTO YTD EVOLUTION LINE CHART TEMPLATE (Chart.js)
# =============================================================================

CRYPTO_YTD_EVOLUTION_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.cdnfonts.com/css/calibri-light');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Calibri', Calibri, 'Segoe UI', Arial, sans-serif;
            background: #FFFFFF;
            width: {{ width }}px;
            height: {{ height }}px;
            padding: {{ 15 * scale }}px;
        }

        .chart-wrapper {
            position: relative;
            width: 100%;
            height: 100%;
        }

        .chart-title {
            text-align: center;
            font-size: {{ 14 * scale }}px;
            font-weight: 600;
            color: #1B3A5A;
            margin-bottom: {{ 10 * scale }}px;
        }

        .chart-container {
            position: relative;
            width: 100%;
            height: calc(100% - {{ 30 * scale }}px);
        }
    </style>
</head>
<body>
    <div class="chart-wrapper">
        <div class="chart-title">{{ chart_title }}</div>
        <div class="chart-container">
            <canvas id="ytdChart"></canvas>
        </div>
    </div>

    <script>
        const labels = {{ labels | tojson }};
        const datasets = {{ datasets | tojson }};
        const scale = {{ scale }};
        const yMin = {{ y_min }};
        const yMax = {{ y_max }};

        // Label collision avoidance
        function resolveOverlaps(labels, minGap) {
            labels.sort((a, b) => a.y - b.y);
            for (let i = 1; i < labels.length; i++) {
                const prev = labels[i - 1];
                const curr = labels[i];
                const gap = curr.y - prev.y;
                if (gap < minGap) {
                    curr.y = prev.y + minGap;
                    curr.adjusted = true;
                }
            }
            return labels;
        }

        const ctx = document.getElementById('ytdChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets.map(d => ({
                    label: d.label,
                    data: d.data,
                    borderColor: d.borderColor,
                    backgroundColor: 'transparent',
                    borderWidth: 2.5 * scale / 3,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    tension: 0.3,
                    fill: false,
                }))
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
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        enabled: true,
                        backgroundColor: 'rgba(27, 58, 90, 0.9)',
                        titleFont: { family: 'Calibri', size: 10 * scale },
                        bodyFont: { family: 'Calibri', size: 9 * scale },
                        padding: 8 * scale,
                        cornerRadius: 4 * scale,
                        callbacks: {
                            label: function(context) {
                                const value = context.parsed.y;
                                const sign = value >= 0 ? '+' : '';
                                return `${context.dataset.label}: ${sign}${value.toFixed(1)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: {
                            font: { family: 'Calibri', size: 9 * scale, weight: '500' },
                            color: '#64748B',
                            autoSkip: false,
                            maxRotation: 0,
                            callback: function(value, index) {
                                const label = labels[index];
                                if (index === 0) return label;
                                const prevLabel = labels[index - 1];
                                return label !== prevLabel ? label : '';
                            }
                        }
                    },
                    y: {
                        min: yMin,
                        max: yMax,
                        grid: {
                            display: true,
                            color: function(context) {
                                if (context.tick.value === 0) {
                                    return 'rgba(0, 0, 0, 0.3)';
                                }
                                return 'transparent';
                            },
                            lineWidth: function(context) {
                                if (context.tick.value === 0) {
                                    return 2;
                                }
                                return 0;
                            },
                            drawBorder: false,
                        },
                        ticks: {
                            font: { family: 'Calibri', size: 9 * scale, weight: '500' },
                            color: '#64748B',
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                layout: {
                    padding: {
                        right: 110 * scale
                    }
                }
            },
            plugins: [{
                id: 'endLabelsNoOverlap',
                afterDraw: function(chart) {
                    const ctx = chart.ctx;
                    const minGap = 16 * scale;

                    let labelData = [];

                    chart.data.datasets.forEach((dataset, i) => {
                        const meta = chart.getDatasetMeta(i);
                        const lastPoint = meta.data[meta.data.length - 1];

                        if (lastPoint) {
                            const value = dataset.data[dataset.data.length - 1];
                            const sign = value >= 0 ? '+' : '';
                            const label = `${dataset.label}: ${sign}${value.toFixed(1)}%`;

                            labelData.push({
                                x: lastPoint.x,
                                y: lastPoint.y,
                                originalY: lastPoint.y,
                                label: label,
                                color: dataset.borderColor,
                                adjusted: false
                            });
                        }
                    });

                    labelData = resolveOverlaps(labelData, minGap);

                    const fontSize = 9 * scale;
                    labelData.forEach(item => {
                        ctx.save();
                        ctx.font = `600 ${fontSize}px Calibri`;
                        const textWidth = ctx.measureText(item.label).width;
                        const labelX = item.x + 10 * scale;
                        const labelY = item.y;
                        const pillHeight = 14 * scale;
                        const pillPadding = 4 * scale;

                        if (item.adjusted) {
                            ctx.beginPath();
                            ctx.strokeStyle = item.color + '60';
                            ctx.lineWidth = 1;
                            ctx.setLineDash([2 * scale, 2 * scale]);
                            ctx.moveTo(item.x + 2, item.originalY);
                            ctx.lineTo(labelX - 2, labelY);
                            ctx.stroke();
                            ctx.setLineDash([]);
                        }

                        ctx.fillStyle = item.color + '20';
                        ctx.beginPath();
                        ctx.roundRect(
                            labelX - pillPadding,
                            labelY - pillHeight/2,
                            textWidth + pillPadding * 2.5,
                            pillHeight,
                            3 * scale
                        );
                        ctx.fill();

                        ctx.strokeStyle = item.color + '40';
                        ctx.lineWidth = 1;
                        ctx.stroke();

                        ctx.fillStyle = item.color;
                        ctx.textBaseline = 'middle';
                        ctx.fillText(item.label, labelX, labelY);
                        ctx.restore();
                    });
                }
            }]
        });
    </script>
</body>
</html>
'''


# =============================================================================
# FX IMPACT ANALYSIS TEMPLATE (EUR)
# =============================================================================

FX_IMPACT_ANALYSIS_EUR_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        @import url('https://fonts.cdnfonts.com/css/calibri-light');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Calibri', Calibri, 'Segoe UI', Arial, sans-serif;
            background: #FFFFFF;
            width: {{ width }}px;
            height: {{ height }}px;
            padding: {{ 15 * scale }}px;
        }

        .container {
            display: flex;
            flex-direction: column;
            height: 100%;
        }

        /* Table section */
        .table-container {
            display: flex;
            flex-direction: column;
            gap: {{ 2 * scale }}px;
        }

        /* Table header */
        .table-header {
            display: flex;
            align-items: center;
            padding: {{ 8 * scale }}px {{ 10 * scale }}px;
            background: #F8FAFC;
            border-radius: {{ 6 * scale }}px;
        }

        .col-index {
            width: {{ 130 * scale }}px;
            font-size: {{ 9 * scale }}px;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .col-local {
            width: {{ 70 * scale }}px;
            font-size: {{ 9 * scale }}px;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            text-align: center;
        }

        .col-eur {
            width: {{ 70 * scale }}px;
            font-size: {{ 9 * scale }}px;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            text-align: center;
        }

        .col-fx-effect {
            width: {{ 80 * scale }}px;
            font-size: {{ 9 * scale }}px;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            text-align: center;
        }

        .col-bar {
            flex: 1;
            font-size: {{ 9 * scale }}px;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            text-align: center;
        }

        /* Data rows */
        .data-row {
            display: flex;
            align-items: center;
            padding: {{ 6 * scale }}px {{ 10 * scale }}px;
            border-radius: {{ 6 * scale }}px;
            border: {{ 1 * scale }}px solid transparent;
        }

        .data-row:nth-child(even) {
            background: rgba(248, 250, 252, 0.5);
        }

        /* FX Tailwind highlight (best) */
        .data-row.fx-tailwind {
            background: linear-gradient(90deg, rgba(34, 197, 94, 0.08), transparent);
            border-left: {{ 3 * scale }}px solid #22C55E;
        }

        /* FX Headwind highlight (worst) */
        .data-row.fx-headwind {
            background: linear-gradient(90deg, rgba(239, 68, 68, 0.08), transparent);
            border-left: {{ 3 * scale }}px solid #EF4444;
        }

        .data-row .col-index {
            display: flex;
            align-items: center;
            gap: {{ 8 * scale }}px;
            font-weight: 500;
            color: #334155;
        }

        /* Flag images */
        .col-index img,
        .col-index .flag-img {
            width: {{ 22 * scale }}px !important;
            height: {{ 14 * scale }}px !important;
            vertical-align: middle;
            flex-shrink: 0;
        }

        .index-name {
            font-size: {{ 10 * scale }}px;
            font-weight: 500;
            color: #334155;
        }

        .data-row .col-local,
        .data-row .col-eur {
            font-size: {{ 10 * scale }}px;
            font-weight: 500;
            color: #64748B;
        }

        .data-row .col-fx-effect {
            font-size: {{ 10 * scale }}px;
            font-weight: 700;
        }

        .col-fx-effect.positive { color: #16A34A; }
        .col-fx-effect.negative { color: #DC2626; }

        /* Bar container */
        .bar-container {
            flex: 1;
            display: flex;
            align-items: center;
            padding: 0 {{ 10 * scale }}px;
            position: relative;
        }

        .bar-track {
            width: 100%;
            height: {{ 6 * scale }}px;
            background: #F1F5F9;
            border-radius: {{ 3 * scale }}px;
            position: relative;
        }

        /* Center line */
        .center-line {
            position: absolute;
            left: 50%;
            top: {{ -3 * scale }}px;
            bottom: {{ -3 * scale }}px;
            width: {{ 2 * scale }}px;
            background: #CBD5E1;
            border-radius: {{ 1 * scale }}px;
            z-index: 10;
        }

        /* Bars */
        .bar {
            position: absolute;
            height: {{ 10 * scale }}px;
            top: 50%;
            transform: translateY(-50%);
            z-index: 2;
        }

        .bar.positive {
            left: calc(50% + 1px);
            background: linear-gradient(90deg, #4ADE80, #16A34A);
            box-shadow: 0 {{ 2 * scale }}px {{ 4 * scale }}px rgba(34, 197, 94, 0.25);
            border-radius: 0 {{ 5 * scale }}px {{ 5 * scale }}px 0;
        }

        .bar.negative {
            right: calc(50% + 1px);
            background: linear-gradient(270deg, #F87171, #DC2626);
            box-shadow: 0 {{ 2 * scale }}px {{ 4 * scale }}px rgba(239, 68, 68, 0.25);
            border-radius: {{ 5 * scale }}px 0 0 {{ 5 * scale }}px;
        }

        /* Key Insight box */
        .insight-box {
            margin-top: {{ 6 * scale }}px;
            padding: {{ 12 * scale }}px {{ 15 * scale }}px;
            background: linear-gradient(90deg, rgba(201, 162, 39, 0.08), rgba(201, 162, 39, 0.02));
            border-left: {{ 4 * scale }}px solid #C9A227;
            border-radius: 0 {{ 8 * scale }}px {{ 8 * scale }}px 0;
        }

        .insight-header {
            display: flex;
            align-items: center;
            gap: {{ 6 * scale }}px;
            margin-bottom: {{ 6 * scale }}px;
        }

        .insight-icon {
            font-size: {{ 12 * scale }}px;
        }

        .insight-title {
            font-size: {{ 11 * scale }}px;
            font-weight: 700;
            color: #1B3A5A;
        }

        .insight-text {
            font-size: {{ 9 * scale }}px;
            font-weight: 400;
            color: #475569;
            line-height: 1.5;
        }

        .insight-text .highlight-positive {
            color: #16A34A;
            font-weight: 600;
        }

        .insight-text .highlight-negative {
            color: #DC2626;
            font-weight: 600;
        }

        /* Scale */
        .scale {
            display: flex;
            justify-content: center;
            padding: {{ 1 * scale }}px 0 0 0;
            margin-left: {{ 350 * scale }}px;
        }

        .scale-inner {
            display: flex;
            justify-content: space-between;
            width: 100%;
            padding: 0 {{ 10 * scale }}px;
            font-size: {{ 7 * scale }}px;
            color: #94A3B8;
        }

        /* Legend */
        .legend-row {
            display: flex;
            justify-content: center;
            gap: {{ 20 * scale }}px;
            margin-top: {{ 6 * scale }}px;
            margin-left: {{ 350 * scale }}px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: {{ 6 * scale }}px;
            font-size: {{ 8 * scale }}px;
            color: #64748B;
        }

        .legend-dot {
            width: {{ 8 * scale }}px;
            height: {{ 8 * scale }}px;
            border-radius: 50%;
        }

        .legend-dot.green {
            background: linear-gradient(135deg, #10B981, #34D399);
            box-shadow: 0 {{ 1 * scale }}px {{ 3 * scale }}px rgba(16, 185, 129, 0.4);
        }

        .legend-dot.red {
            background: linear-gradient(135deg, #EF4444, #F87171);
            box-shadow: 0 {{ 1 * scale }}px {{ 3 * scale }}px rgba(239, 68, 68, 0.4);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="table-container">
            <div class="table-header">
                <div class="col-index">Index</div>
                <div class="col-local">Local</div>
                <div class="col-eur">EUR</div>
                <div class="col-fx-effect">FX Effect</div>
                <div class="col-bar">Impact</div>
            </div>

            {% for row in rows %}
            <div class="data-row {{ row.highlight_class }}">
                <div class="col-index">
                    {{ row.flag_html | safe }}
                    <span class="index-name">{{ row.name }}</span>
                </div>
                <div class="col-local">{{ row.local_formatted }}</div>
                <div class="col-eur">{{ row.eur_formatted }}</div>
                <div class="col-fx-effect {{ row.fx_class }}">{{ row.fx_formatted }}</div>
                <div class="bar-container">
                    <div class="bar-track">
                        <div class="center-line"></div>
                        {% if row.fx_effect != 0 %}
                        <div class="bar {{ row.bar_class }}" style="width: {{ row.bar_width }}%;"></div>
                        {% endif %}
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>

        <div class="scale">
            <div class="scale-inner">
                <span>{{ scale_min }}</span>
                <span>{{ scale_mid_low }}</span>
                <span>0</span>
                <span>{{ scale_mid_high }}</span>
                <span>{{ scale_max }}</span>
            </div>
        </div>

        <div class="legend-row">
            <div class="legend-item">
                <span class="legend-dot green"></span>
                <span>FX tailwind (currency strengthened vs EUR)</span>
            </div>
            <div class="legend-item">
                <span class="legend-dot red"></span>
                <span>FX headwind (currency weakened vs EUR)</span>
            </div>
        </div>

        <div class="insight-box">
            <div class="insight-header">
                <span class="insight-icon">💡</span>
                <span class="insight-title">Key Insight</span>
            </div>
            <div class="insight-text">{{ insight_text | safe }}</div>
        </div>
    </div>
</body>
</html>
'''


# =============================================================================
# FX IMPACT ANALYSIS TEMPLATE (CHF)
# =============================================================================

FX_IMPACT_ANALYSIS_CHF_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        @import url('https://fonts.cdnfonts.com/css/calibri-light');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Calibri', Calibri, 'Segoe UI', Arial, sans-serif;
            background: #FFFFFF;
            width: {{ width }}px;
            height: {{ height }}px;
            padding: {{ 15 * scale }}px;
        }

        .container {
            display: flex;
            flex-direction: column;
            height: 100%;
        }

        /* Table section */
        .table-container {
            display: flex;
            flex-direction: column;
            gap: {{ 2 * scale }}px;
        }

        /* Table header */
        .table-header {
            display: flex;
            align-items: center;
            padding: {{ 8 * scale }}px {{ 10 * scale }}px;
            background: #F8FAFC;
            border-radius: {{ 6 * scale }}px;
        }

        .col-index {
            width: {{ 130 * scale }}px;
            font-size: {{ 9 * scale }}px;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .col-local {
            width: {{ 70 * scale }}px;
            font-size: {{ 9 * scale }}px;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            text-align: center;
        }

        .col-chf {
            width: {{ 70 * scale }}px;
            font-size: {{ 9 * scale }}px;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            text-align: center;
        }

        .col-fx-effect {
            width: {{ 80 * scale }}px;
            font-size: {{ 9 * scale }}px;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            text-align: center;
        }

        .col-bar {
            flex: 1;
            font-size: {{ 9 * scale }}px;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            text-align: center;
        }

        /* Data rows */
        .data-row {
            display: flex;
            align-items: center;
            padding: {{ 6 * scale }}px {{ 10 * scale }}px;
            border-radius: {{ 6 * scale }}px;
            border: {{ 1 * scale }}px solid transparent;
        }

        .data-row:nth-child(even) {
            background: rgba(248, 250, 252, 0.5);
        }

        /* FX Tailwind highlight (best) */
        .data-row.fx-tailwind {
            background: linear-gradient(90deg, rgba(34, 197, 94, 0.08), transparent);
            border-left: {{ 3 * scale }}px solid #22C55E;
        }

        /* FX Headwind highlight (worst) */
        .data-row.fx-headwind {
            background: linear-gradient(90deg, rgba(239, 68, 68, 0.08), transparent);
            border-left: {{ 3 * scale }}px solid #EF4444;
        }

        .data-row .col-index {
            display: flex;
            align-items: center;
            gap: {{ 8 * scale }}px;
            font-weight: 500;
            color: #334155;
        }

        /* Flag images */
        .col-index img,
        .col-index .flag-img {
            width: {{ 22 * scale }}px !important;
            height: {{ 14 * scale }}px !important;
            vertical-align: middle;
            flex-shrink: 0;
        }

        .index-name {
            font-size: {{ 10 * scale }}px;
            font-weight: 500;
            color: #334155;
        }

        .data-row .col-local,
        .data-row .col-chf {
            font-size: {{ 10 * scale }}px;
            font-weight: 500;
            color: #64748B;
        }

        .data-row .col-fx-effect {
            font-size: {{ 10 * scale }}px;
            font-weight: 700;
        }

        .col-fx-effect.positive { color: #16A34A; }
        .col-fx-effect.negative { color: #DC2626; }

        /* Bar container */
        .bar-container {
            flex: 1;
            display: flex;
            align-items: center;
            padding: 0 {{ 10 * scale }}px;
            position: relative;
        }

        .bar-track {
            width: 100%;
            height: {{ 6 * scale }}px;
            background: #F1F5F9;
            border-radius: {{ 3 * scale }}px;
            position: relative;
        }

        /* Center line */
        .center-line {
            position: absolute;
            left: 50%;
            top: {{ -3 * scale }}px;
            bottom: {{ -3 * scale }}px;
            width: {{ 2 * scale }}px;
            background: #CBD5E1;
            border-radius: {{ 1 * scale }}px;
            z-index: 10;
        }

        /* Bars */
        .bar {
            position: absolute;
            height: {{ 10 * scale }}px;
            top: 50%;
            transform: translateY(-50%);
            z-index: 2;
        }

        .bar.positive {
            left: calc(50% + 1px);
            background: linear-gradient(90deg, #4ADE80, #16A34A);
            box-shadow: 0 {{ 2 * scale }}px {{ 4 * scale }}px rgba(34, 197, 94, 0.25);
            border-radius: 0 {{ 5 * scale }}px {{ 5 * scale }}px 0;
        }

        .bar.negative {
            right: calc(50% + 1px);
            background: linear-gradient(270deg, #F87171, #DC2626);
            box-shadow: 0 {{ 2 * scale }}px {{ 4 * scale }}px rgba(239, 68, 68, 0.25);
            border-radius: {{ 5 * scale }}px 0 0 {{ 5 * scale }}px;
        }

        /* Key Insight box */
        .insight-box {
            margin-top: {{ 6 * scale }}px;
            padding: {{ 12 * scale }}px {{ 15 * scale }}px;
            background: linear-gradient(90deg, rgba(201, 162, 39, 0.08), rgba(201, 162, 39, 0.02));
            border-left: {{ 4 * scale }}px solid #C9A227;
            border-radius: 0 {{ 8 * scale }}px {{ 8 * scale }}px 0;
        }

        .insight-header {
            display: flex;
            align-items: center;
            gap: {{ 6 * scale }}px;
            margin-bottom: {{ 6 * scale }}px;
        }

        .insight-icon {
            font-size: {{ 12 * scale }}px;
        }

        .insight-title {
            font-size: {{ 11 * scale }}px;
            font-weight: 700;
            color: #1B3A5A;
        }

        .insight-text {
            font-size: {{ 9 * scale }}px;
            font-weight: 400;
            color: #475569;
            line-height: 1.5;
        }

        .insight-text .highlight-positive {
            color: #16A34A;
            font-weight: 600;
        }

        .insight-text .highlight-negative {
            color: #DC2626;
            font-weight: 600;
        }

        /* Scale */
        .scale {
            display: flex;
            justify-content: center;
            padding: {{ 1 * scale }}px 0 0 0;
            margin-left: {{ 350 * scale }}px;
        }

        .scale-inner {
            display: flex;
            justify-content: space-between;
            width: 100%;
            padding: 0 {{ 10 * scale }}px;
            font-size: {{ 7 * scale }}px;
            color: #94A3B8;
        }

        /* Legend */
        .legend-row {
            display: flex;
            justify-content: center;
            gap: {{ 20 * scale }}px;
            margin-top: {{ 6 * scale }}px;
            margin-left: {{ 350 * scale }}px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: {{ 6 * scale }}px;
            font-size: {{ 8 * scale }}px;
            color: #64748B;
        }

        .legend-dot {
            width: {{ 8 * scale }}px;
            height: {{ 8 * scale }}px;
            border-radius: 50%;
        }

        .legend-dot.green {
            background: linear-gradient(135deg, #10B981, #34D399);
            box-shadow: 0 {{ 1 * scale }}px {{ 3 * scale }}px rgba(16, 185, 129, 0.4);
        }

        .legend-dot.red {
            background: linear-gradient(135deg, #EF4444, #F87171);
            box-shadow: 0 {{ 1 * scale }}px {{ 3 * scale }}px rgba(239, 68, 68, 0.4);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="table-container">
            <div class="table-header">
                <div class="col-index">Index</div>
                <div class="col-local">Local</div>
                <div class="col-chf">CHF</div>
                <div class="col-fx-effect">FX Effect</div>
                <div class="col-bar">Impact</div>
            </div>

            {% for row in rows %}
            <div class="data-row {{ row.highlight_class }}">
                <div class="col-index">
                    {{ row.flag_html | safe }}
                    <span class="index-name">{{ row.name }}</span>
                </div>
                <div class="col-local">{{ row.local_formatted }}</div>
                <div class="col-chf">{{ row.chf_formatted }}</div>
                <div class="col-fx-effect {{ row.fx_class }}">{{ row.fx_formatted }}</div>
                <div class="bar-container">
                    <div class="bar-track">
                        <div class="center-line"></div>
                        {% if row.fx_effect != 0 %}
                        <div class="bar {{ row.bar_class }}" style="width: {{ row.bar_width }}%;"></div>
                        {% endif %}
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>

        <div class="scale">
            <div class="scale-inner">
                <span>{{ scale_min }}</span>
                <span>{{ scale_mid_low }}</span>
                <span>0</span>
                <span>{{ scale_mid_high }}</span>
                <span>{{ scale_max }}</span>
            </div>
        </div>

        <div class="legend-row">
            <div class="legend-item">
                <span class="legend-dot green"></span>
                <span>FX tailwind (currency strengthened vs CHF)</span>
            </div>
            <div class="legend-item">
                <span class="legend-dot red"></span>
                <span>FX headwind (currency weakened vs CHF)</span>
            </div>
        </div>

        <div class="insight-box">
            <div class="insight-header">
                <span class="insight-icon">💡</span>
                <span class="insight-title">Key Insight</span>
            </div>
            <div class="insight-text">{{ insight_text | safe }}</div>
        </div>
    </div>
</body>
</html>
'''


# =============================================================================
# TECHNICAL ANALYSIS CHART V2 - Chart.js + Playwright
# =============================================================================
# Template has been refactored to technical_analysis/templates/technical_analysis_v2.py
# Re-export for backward compatibility

from technical_analysis.templates import TECHNICAL_ANALYSIS_V2_HTML_TEMPLATE

