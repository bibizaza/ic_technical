"""HTML template for Weekly Performance chart."""

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
                <span class="flag">{{ row.flag }}</span>
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
            gap: {{ 3 * scale }}px;
        }

        /* Header row */
        .header-row {
            display: flex;
            align-items: center;
            padding: {{ 6 * scale }}px 0;
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
            font-size: {{ 12 * scale }}px;
        }

        .market-name {
            font-size: {{ 9 * scale }}px;
            font-weight: 500;
            color: #334155;
        }

        /* Value cells */
        .value-cell {
            flex: 1;
            height: {{ 30 * scale }}px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: {{ 5 * scale }}px;
            font-size: {{ 9 * scale }}px;
            font-weight: 600;
            color: #FFFFFF;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }

        /* YTD column - emphasized */
        .value-cell.ytd {
            flex: 1.3;
            height: {{ 34 * scale }}px;
            font-size: {{ 11 * scale }}px;
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
                <span class="flag">{{ row.flag }}</span>
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
            <span class="flag">{{ country.flag }}</span>
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
            font-size: {{ 9 * scale }}px;
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
            <span class="flag">{{ country.flag }}</span>
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
            gap: {{ 6 * scale }}px;
        }

        /* Data rows */
        .row {
            display: flex;
            align-items: center;
            height: {{ 38 * scale }}px;
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
            font-size: {{ 12 * scale }}px;
        }

        .credit-badge {
            font-size: {{ 7 * scale }}px;
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
                <span class="flag">{{ row.flag }}</span>
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
            gap: {{ 3 * scale }}px;
        }

        /* Header row */
        .header-row {
            display: flex;
            align-items: center;
            padding: {{ 4 * scale }}px 0;
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
            font-size: {{ 12 * scale }}px;
        }

        .credit-badge {
            font-size: {{ 6 * scale }}px;
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
            font-size: {{ 9 * scale }}px;
            font-weight: 500;
            color: #334155;
        }

        /* Value cells */
        .value-cell {
            flex: 1;
            height: {{ 26 * scale }}px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: {{ 5 * scale }}px;
            font-size: {{ 9 * scale }}px;
            font-weight: 600;
            color: #FFFFFF;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }

        /* YTD column - emphasized */
        .value-cell.ytd {
            flex: 1.3;
            height: {{ 30 * scale }}px;
            font-size: {{ 11 * scale }}px;
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
                <span class="flag">{{ row.flag }}</span>
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
        {% for item in category.items %}
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
