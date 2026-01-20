"""HTML template for Fundamental Rank table."""

FUNDAMENTAL_HTML_TEMPLATE = '''
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

        /* Table styling - same as Breadth */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: {{ 13 * scale }}px;
        }

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

        /* Rank column header - GOLD */
        th.rank-col {
            background: #C9A227;
            color: #1B3A5A;
            width: {{ 60 * scale }}px;
            font-weight: 700;
        }

        /* Remove padding from td - let inner divs handle it */
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

        /* Market column - left align */
        .cell-content.market {
            justify-content: flex-start;
            gap: {{ 10 * scale }}px;
            padding-left: {{ 12 * scale }}px;
            white-space: nowrap;
        }

        .cell-content.market img,
        .cell-content.market .flag {
            flex-shrink: 0;
        }

        /* Rank column - centered, gold background */
        .cell-content.rank {
            justify-content: center;
            font-weight: 700;
            color: #92710C;
            background: #FEF9E7;
        }

        tr:nth-child(even) .cell-content.rank {
            background: #FCF3CD;
        }

        /* ========== DOT STYLING ========== */
        .cell-content.dot {
            gap: {{ 5 * scale }}px;
        }

        .dot-circle {
            width: {{ 8 * scale }}px;
            height: {{ 8 * scale }}px;
            border-radius: 50%;
            flex-shrink: 0;
        }

        .dot-value {
            font-weight: 600;
            font-size: {{ 13 * scale }}px;
        }

        /* Color zones: 1-3 = good (green), 4-6 = neutral (yellow), 7-9 = bad (red) */
        .dot-circle.good { background: #22C55E; }
        .dot-circle.neutral { background: #EAB308; }
        .dot-circle.bad { background: #EF4444; }

        .dot-value.good { color: #16A34A; }
        .dot-value.neutral { color: #CA8A04; }
        .dot-value.bad { color: #DC2626; }
    </style>
</head>
<body>
    <table>
        <thead>
            <tr>
                <th>Market</th>
                <th class="rank-col">Rank</th>
                <th>Valuation</th>
                <th>Growth</th>
                <th>Profitability</th>
                <th>Quality</th>
                <th>Leverage</th>
                <th>Dividend</th>
            </tr>
        </thead>
        <tbody>
            {% for row in rows %}
            <tr>
                <td>
                    <div class="cell-content market">
                        {{ row.flag_html | safe }}
                        <span>{{ row.index_name }}</span>
                    </div>
                </td>
                <td>
                    <div class="cell-content rank">
                        {{ row.rank }}
                    </div>
                </td>
                <td>
                    <div class="cell-content dot">
                        <span class="dot-circle {{ row.rv_class }}"></span>
                        <span class="dot-value {{ row.rv_class }}">{{ row.rv }}</span>
                    </div>
                </td>
                <td>
                    <div class="cell-content dot">
                        <span class="dot-circle {{ row.growth_class }}"></span>
                        <span class="dot-value {{ row.growth_class }}">{{ row.growth }}</span>
                    </div>
                </td>
                <td>
                    <div class="cell-content dot">
                        <span class="dot-circle {{ row.profitability_class }}"></span>
                        <span class="dot-value {{ row.profitability_class }}">{{ row.profitability }}</span>
                    </div>
                </td>
                <td>
                    <div class="cell-content dot">
                        <span class="dot-circle {{ row.quality_class }}"></span>
                        <span class="dot-value {{ row.quality_class }}">{{ row.quality }}</span>
                    </div>
                </td>
                <td>
                    <div class="cell-content dot">
                        <span class="dot-circle {{ row.leverage_class }}"></span>
                        <span class="dot-value {{ row.leverage_class }}">{{ row.leverage }}</span>
                    </div>
                </td>
                <td>
                    <div class="cell-content dot">
                        <span class="dot-circle {{ row.dividend_class }}"></span>
                        <span class="dot-value {{ row.dividend_class }}">{{ row.dividend }}</span>
                    </div>
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>
'''
