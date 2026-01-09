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
            display: flex;
            align-items: center;
            gap: {{ 8 * scale }}px;
        }

        .flag {
            font-size: {{ 14 * scale }}px;
            line-height: 1;
        }

        .market-name {
            font-size: {{ 10 * scale }}px;
            font-weight: 500;
            color: #334155;
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
            margin-left: {{ 110 * scale }}px;
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
            gap: {{ 2 * scale }}px;
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
            width: {{ 110 * scale }}px;
            display: flex;
            align-items: center;
            gap: {{ 6 * scale }}px;
            padding-right: {{ 8 * scale }}px;
        }

        .flag {
            font-size: {{ 10 * scale }}px;
            line-height: 1;
        }

        .market-name {
            font-size: {{ 8 * scale }}px;
            font-weight: 500;
            color: #334155;
        }

        /* Value cells */
        .value-cell {
            flex: 1;
            height: {{ 24 * scale }}px;
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
            height: {{ 26 * scale }}px;
            font-size: {{ 9 * scale }}px;
            font-weight: 700;
            margin-right: {{ 10 * scale }}px;
            border-radius: {{ 5 * scale }}px;
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
            font-size: {{ 14 * scale }}px;
            line-height: 1;
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
            margin-left: {{ 69 * scale }}px;
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
            font-size: {{ 10 * scale }}px;
            line-height: 1;
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
            font-size: {{ 10 * scale }}px;
            line-height: 1;
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
            font-size: {{ 10 * scale }}px;
            line-height: 1;
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
            margin-left: {{ 134 * scale }}px;
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
            font-size: {{ 14 * scale }}px;
            line-height: 1;
        }

        .currency-name {
            font-size: {{ 11 * scale }}px;
            font-weight: 600;
            color: #334155;
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
            margin-left: {{ 132 * scale }}px;
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
            font-size: {{ 14 * scale }}px;
            line-height: 1;
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
        }

        .crypto-name {
            font-size: {{ 10 * scale }}px;
            font-weight: 600;
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
            margin-left: {{ 172 * scale }}px;
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
                <span class="crypto-name">{{ row.name }}</span>
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
            padding-right: {{ 10 * scale }}px;
        }

        .crypto-name {
            font-size: {{ 9 * scale }}px;
            font-weight: 600;
            color: #334155;
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
# TECHNICAL ANALYSIS CHART V2 - Chart.js + Playwright
# =============================================================================

TECHNICAL_ANALYSIS_V2_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation"></script>
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
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-right: none;
            border-radius: 8px 0 0 0;
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

        .dmas-value {
            font-size: 36px;      /* Adjusted for wider panel */
            font-weight: 700;
            color: #FFFFFF;
            line-height: 1.1;
        }

        .dmas-progress {
            width: 100%;
            height: 8px;
            /* Full gradient scale: red → orange → yellow → green */
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
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-top: none;
            border-right: none;
            border-radius: 0 0 0 8px;
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
            padding: 8px 15px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 4px;
            border: 1px solid #1B3A5A;
            border-top: none;
            border-left: none;
            border-radius: 0 0 8px 0;
            list-style: none;
        }

        .rsi-panel * {
            list-style: none;
        }

        .rsi-current-title {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: rgba(255,255,255,0.5);
            text-align: center;
        }

        .rsi-current-value {
            font-size: 24px;
            font-weight: 700;
            text-align: center;
            line-height: 1.1;
            list-style: none;
        }

        .rsi-current-value::before,
        .rsi-current-value::after {
            content: none;
            display: none;
        }

        .rsi-gauge {
            height: 6px;
            background: linear-gradient(90deg, #22C55E 0%, #22C55E 30%, #EAB308 50%, #EF4444 70%, #EF4444 100%);
            border-radius: 3px;
            position: relative;
            margin: 4px 0;
        }

        .rsi-gauge-marker {
            position: absolute;
            width: 4px;
            height: 12px;
            background: #FFFFFF;
            border-radius: 2px;
            top: -2px;
            transform: translateX(-50%);
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }

        .rsi-interpretation {
            font-size: 9px;
            text-align: center;
            color: rgba(255,255,255,0.8);
        }

        .rsi-context {
            font-size: 8px;
            text-align: center;
            color: rgba(255,255,255,0.5);
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
            color: #64748B;
        }
    </style>
</head>
<body>
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
                    <div class="dmas-label">DMAS Score</div>
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
                <div class="rsi-current-title">RSI</div>
                <div class="rsi-current-value" style="color: {{ rsi_color }};">{{ rsi_current }}</div>
                <div class="rsi-interpretation" style="color: {{ rsi_color }};">{{ rsi_interpretation }}</div>
                <div class="rsi-context" style="color: {{ rsi_color }}; opacity: 0.7;">{{ rsi_context }}</div>
            </div>
        </div>
    </div>

    <script>
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
                        borderWidth: 1.5 * scale / 3,
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
                        borderWidth: 1.5 * scale / 3,
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
                        borderWidth: 1.5 * scale / 3,
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
                            font: { size: 9 * scale, family: 'Calibri' },
                            color: '#64748B',
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
                            font: { size: 9 * scale, family: 'Calibri' },
                            color: '#64748B',
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

                    // Draw labels ABOVE the line for Higher Range
                    ctx.font = `${smallFontSize}px Calibri`;
                    ctx.textAlign = 'left';
                    ctx.fillStyle = '#64748B';
                    ctx.fillText('Higher Range', chartArea.right + 10 * scale, higherY - 18 * scale);
                    ctx.font = `600 ${fontSize}px Calibri`;
                    ctx.fillStyle = '#10B981';
                    ctx.fillText(higherRange.toLocaleString() + ' (' + higherRangePct + ')', chartArea.right + 10 * scale, higherY - 6 * scale);

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

                    // Draw labels BELOW the line for Lower Range
                    ctx.font = `600 ${fontSize}px Calibri`;
                    ctx.textAlign = 'left';
                    ctx.fillStyle = '#EF4444';
                    ctx.fillText(lowerRange.toLocaleString() + ' (' + lowerRangePct + ')', chartArea.right + 10 * scale, lowerY + 14 * scale);
                    ctx.font = `${smallFontSize}px Calibri`;
                    ctx.fillStyle = '#64748B';
                    ctx.fillText('Lower Range', chartArea.right + 10 * scale, lowerY + 26 * scale);
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
                    borderWidth: 2 * scale / 3,
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
                            font: { size: 9 * scale, family: 'Calibri' },
                            color: '#64748B',
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
                            font: { size: 9 * scale, family: 'Calibri' },
                            color: '#64748B',
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
    </script>
</body>
</html>
'''

