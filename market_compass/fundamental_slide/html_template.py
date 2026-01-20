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
            height: {{ height }}px;
            padding: 0;
            margin: 0;
        }

        table {
            width: 100%;
            height: calc(100% - {{ 20 * scale }}px);
            border-collapse: collapse;
            font-size: {{ 9 * scale }}px;
        }

        th {
            background: #1B3A5A;
            color: #FFFFFF;
            font-weight: 600;
            padding: {{ 7 * scale }}px {{ 6.5 * scale }}px;
            text-align: center;
            border: none;
        }

        th:first-child {
            text-align: left;
            padding-left: {{ 8 * scale }}px;
            width: {{ 80 * scale }}px;
        }

        /* Rank column header - GOLD */
        th.rank-col {
            background: #C9A227;
            color: #1B3A5A;
            width: {{ 45 * scale }}px;
            font-weight: 700;
        }

        td {
            padding: {{ 6.5 * scale }}px {{ 6.5 * scale }}px;
            text-align: center;
            border-bottom: {{ 1 * scale }}px solid #E8E8E8;
            background: #FFFFFF;
        }

        td:first-child {
            text-align: left;
            padding-left: {{ 8 * scale }}px;
            font-weight: 500;
        }

        .index-cell {
            display: flex;
            align-items: center;
            gap: {{ 6 * scale }}px;
            white-space: nowrap;
        }

        tr:nth-child(even) td {
            background: #F8F9FA;
        }

        /* Rank cell - GOLD tint */
        .rank-cell {
            background: #FEF9E7 !important;
            font-weight: 700;
            color: #92710C;
        }

        tr:nth-child(even) .rank-cell {
            background: #FCF3CD !important;
        }

        /* ========== DOT STYLING ========== */
        .dot-cell {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: {{ 3 * scale }}px;
        }

        .dot {
            width: {{ 6 * scale }}px;
            height: {{ 6 * scale }}px;
            border-radius: 50%;
            flex-shrink: 0;
        }

        .dot-value {
            font-weight: 600;
            font-size: {{ 9 * scale }}px;
        }

        /* Color zones: 1-3 = good (green), 4-6 = neutral (yellow), 7-9 = bad (red) */
        .dot.good { background: #22C55E; }
        .dot.neutral { background: #EAB308; }
        .dot.bad { background: #EF4444; }

        .dot-value.good { color: #16A34A; }
        .dot-value.neutral { color: #CA8A04; }
        .dot-value.bad { color: #DC2626; }

        /* Footnote */
        .footnote {
            font-size: {{ 7 * scale }}px;
            color: #666666;
            padding: {{ 4 * scale }}px {{ 8 * scale }}px;
            font-style: italic;
        }
    </style>
</head>
<body>
    <table>
        <thead>
            <tr>
                <th>Market</th>
                <th class="rank-col">Rank</th>
                <th>RV</th>
                <th>Grw</th>
                <th>Prof</th>
                <th>Qual</th>
                <th>Lev</th>
                <th>Div</th>
            </tr>
        </thead>
        <tbody>
            {% for row in rows %}
            <tr>
                <td class="index-cell">{{ row.flag_html | safe }} {{ row.index_name }}</td>
                <td class="rank-cell">{{ row.rank }}</td>
                <td><div class="dot-cell"><span class="dot {{ row.rv_class }}"></span><span class="dot-value {{ row.rv_class }}">{{ row.rv }}</span></div></td>
                <td><div class="dot-cell"><span class="dot {{ row.growth_class }}"></span><span class="dot-value {{ row.growth_class }}">{{ row.growth }}</span></div></td>
                <td><div class="dot-cell"><span class="dot {{ row.profitability_class }}"></span><span class="dot-value {{ row.profitability_class }}">{{ row.profitability }}</span></div></td>
                <td><div class="dot-cell"><span class="dot {{ row.quality_class }}"></span><span class="dot-value {{ row.quality_class }}">{{ row.quality }}</span></div></td>
                <td><div class="dot-cell"><span class="dot {{ row.leverage_class }}"></span><span class="dot-value {{ row.leverage_class }}">{{ row.leverage }}</span></div></td>
                <td><div class="dot-cell"><span class="dot {{ row.dividend_class }}"></span><span class="dot-value {{ row.dividend_class }}">{{ row.dividend }}</span></div></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    <div class="footnote">* RV=Relative Valuation, Grw=Growth, Prof=Profitability, Qual=Quality, Lev=Leverage, Div=Dividend</div>
</body>
</html>
'''
