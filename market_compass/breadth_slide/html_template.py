"""HTML template for Composite Breadth Score table.

Dimensions and styling match fundamental_slide exactly:
  - SCALE_FACTOR = 4, base 750×360px
  - Cell height: 23*scale px, font: 13*scale px
  - Header background: #1B3A5A
  - Rank column: gold (#C9A227 header, #FEF9E7 cell)
  - Composite column: mini ring gauge (SVG, viewBox 0 0 32 32, r=12)
  - Trend / Conviction / Sentiment: mini-bar + numeric value
"""

BREADTH_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
            background: transparent;
            width: {{ width }}px;
            min-height: {{ height }}px;
            padding: 0;
            margin: 0;
        }

        /* ========== TABLE ========== */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: {{ 13 * scale }}px;
        }

        /* ========== HEADER ========== */
        th {
            background: #1B3A5A;
            color: #FFFFFF;
            font-weight: 600;
            padding: {{ 10 * scale }}px {{ 8 * scale }}px;
            text-align: center;
            border: none;
            height: {{ 18 * scale }}px;
        }

        th:first-child {
            text-align: left;
            padding-left: {{ 12 * scale }}px;
            width: {{ 108 * scale }}px;
        }

        th.rank-col {
            background: #C9A227;
            color: #1B3A5A;
            width: {{ 60 * scale }}px;
            font-weight: 700;
        }

        th.composite-col {
            width: {{ 72 * scale }}px;
        }

        /* ========== DATA ROWS ========== */
        td {
            padding: 0;
            border-bottom: {{ 1 * scale }}px solid #E8E8E8;
            background: #FFFFFF;
        }

        tr:nth-child(even) td {
            background: #F8F9FA;
        }

        /* ========== CELL CONTENT WRAPPER ========== */
        .cell-content {
            display: flex;
            align-items: center;
            justify-content: center;
            height: {{ 23 * scale }}px;
            padding: 0 {{ 10 * scale }}px;
        }

        /* ========== INDEX COLUMN ========== */
        .cell-content.index-name {
            justify-content: flex-start;
            gap: {{ 8 * scale }}px;
            padding-left: {{ 12 * scale }}px;
            white-space: nowrap;
        }

        .cell-content.index-name .flag {
            font-size: {{ 18 * scale }}px;
            line-height: 1;
        }

        .cell-content.index-name .name {
            font-weight: 700;
            color: #040C38;
            font-size: {{ 13 * scale }}px;
        }

        /* ========== RANK COLUMN ========== */
        .cell-content.rank {
            justify-content: center;
            font-weight: 700;
            color: #92710C;
            background: #FEF9E7;
        }

        tr:nth-child(even) .cell-content.rank {
            background: #FCF3CD;
        }

        /* ========== COMPOSITE PILL ========== */
        .pill {
            display: inline-block;
            padding: {{ 2 * scale }}px {{ 10 * scale }}px;
            border-radius: {{ 4 * scale }}px;
            font-weight: 700;
            font-size: {{ 13 * scale }}px;
        }

        .pill.green { background: #DCFCE7; color: #15803D; }
        .pill.amber { background: #FEF9C3; color: #A16207; }
        .pill.red   { background: #FEE2E2; color: #B91C1C; }

        /* ========== BAR + VALUE CELLS ========== */
        .cell-content.bar-val {
            gap: {{ 6 * scale }}px;
        }

        .mini-bar {
            width: {{ 50 * scale }}px;
            height: {{ 7 * scale }}px;
            background: #E5E7EB;
            border-radius: {{ 4 * scale }}px;
            overflow: hidden;
            flex-shrink: 0;
        }

        .mini-fill {
            height: 100%;
            border-radius: {{ 4 * scale }}px;
        }

        .mini-fill.green { background: linear-gradient(90deg, #22C55E, #16A34A); }
        .mini-fill.amber { background: linear-gradient(90deg, #F59E0B, #D97706); }
        .mini-fill.red   { background: linear-gradient(90deg, #EF4444, #DC2626); }

        .bar-value {
            font-weight: 600;
            font-size: {{ 13 * scale }}px;
            min-width: {{ 26 * scale }}px;
            text-align: right;
        }

        .bar-value.green { color: #16A34A; }
        .bar-value.amber { color: #D97706; }
        .bar-value.red   { color: #DC2626; }
    </style>
</head>
<body>
    <table style="margin-top: {{ 20 * scale }}px;">
        <thead>
            <tr>
                <th>Index</th>
                <th class="rank-col">Rank</th>
                <th class="composite-col">Composite</th>
                <th>Trend</th>
                <th>Conviction</th>
                <th>Sentiment</th>
            </tr>
        </thead>
        <tbody>
            {% for row in rows %}
            <tr>
                <td>
                    <div class="cell-content index-name">
                        <span class="flag">{{ row.flag }}</span>
                        <span class="name">{{ row.name }}</span>
                    </div>
                </td>
                <td>
                    <div class="cell-content rank">{{ row.rank }}</div>
                </td>
                <td>
                    <div class="cell-content">
                        <span class="pill {{ row.composite_class }}">{{ row.composite }}</span>
                    </div>
                </td>
                <td>
                    <div class="cell-content bar-val">
                        <div class="mini-bar"><div class="mini-fill {{ row.trend_class }}" style="width: {{ row.trend }}%;"></div></div>
                        <span class="bar-value {{ row.trend_class }}">{{ row.trend_int }}</span>
                    </div>
                </td>
                <td>
                    <div class="cell-content bar-val">
                        <div class="mini-bar"><div class="mini-fill {{ row.conviction_class }}" style="width: {{ row.conviction }}%;"></div></div>
                        <span class="bar-value {{ row.conviction_class }}">{{ row.conviction_int }}</span>
                    </div>
                </td>
                <td>
                    <div class="cell-content bar-val">
                        <div class="mini-bar"><div class="mini-fill {{ row.sentiment_class }}" style="width: {{ row.sentiment }}%;"></div></div>
                        <span class="bar-value {{ row.sentiment_class }}">{{ row.sentiment_int }}</span>
                    </div>
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>
'''
