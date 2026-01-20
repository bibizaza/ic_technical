"""HTML template for Breadth Rank table."""

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

        /* Table styling */
        table {
            width: 100%;
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
            height: {{ 12 * scale }}px;
        }

        th:first-child {
            text-align: left;
            padding-left: {{ 8 * scale }}px;
            width: {{ 75 * scale }}px;
        }

        /* Rank column header - GOLD */
        th.rank-col {
            background: #C9A227;
            color: #1B3A5A;
            width: {{ 40 * scale }}px;
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
            height: {{ 12.25 * scale }}px;
            padding: 0 {{ 6 * scale }}px;
        }

        /* Market/Index column - left align */
        .cell-content.market {
            justify-content: flex-start;
            gap: {{ 6 * scale }}px;
            padding-left: {{ 8 * scale }}px;
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

        /* ========== PROGRESS BAR STYLING ========== */
        .cell-content.pct {
            gap: {{ 4 * scale }}px;
        }

        .pct-gauge {
            width: {{ 40 * scale }}px;
            height: {{ 5 * scale }}px;
            background: #E5E7EB;
            border-radius: {{ 3 * scale }}px;
            overflow: hidden;
            flex-shrink: 0;
        }

        .pct-fill {
            height: 100%;
            border-radius: {{ 3 * scale }}px;
        }

        .pct-value {
            font-weight: 600;
            font-size: {{ 9 * scale }}px;
            min-width: {{ 28 * scale }}px;
            text-align: right;
        }

        /* Progress bar colors */
        .pct-fill.high { background: linear-gradient(90deg, #22C55E, #16A34A); }
        .pct-fill.med-high { background: linear-gradient(90deg, #84CC16, #65A30D); }
        .pct-fill.med { background: linear-gradient(90deg, #EAB308, #CA8A04); }
        .pct-fill.med-low { background: linear-gradient(90deg, #F97316, #EA580C); }
        .pct-fill.low { background: linear-gradient(90deg, #EF4444, #DC2626); }

        /* Text color to match bar */
        .pct-value.high { color: #16A34A; }
        .pct-value.med-high { color: #65A30D; }
        .pct-value.med { color: #CA8A04; }
        .pct-value.med-low { color: #EA580C; }
        .pct-value.low { color: #DC2626; }
    </style>
</head>
<body>
    <table>
        <thead>
            <tr>
                <th>Index</th>
                <th class="rank-col">Rank</th>
                <th>Above Both MAs</th>
                <th>Above 20D MA</th>
                <th>Above 50D MA</th>
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
                    <div class="cell-content pct">
                        <div class="pct-gauge"><div class="pct-fill {{ row.pct_both_class }}" style="width: {{ row.pct_both }}%;"></div></div>
                        <span class="pct-value {{ row.pct_both_class }}">{{ row.pct_both }}%</span>
                    </div>
                </td>
                <td>
                    <div class="cell-content pct">
                        <div class="pct-gauge"><div class="pct-fill {{ row.pct_20d_class }}" style="width: {{ row.pct_20d }}%;"></div></div>
                        <span class="pct-value {{ row.pct_20d_class }}">{{ row.pct_20d }}%</span>
                    </div>
                </td>
                <td>
                    <div class="cell-content pct">
                        <div class="pct-gauge"><div class="pct-fill {{ row.pct_50d_class }}" style="width: {{ row.pct_50d }}%;"></div></div>
                        <span class="pct-value {{ row.pct_50d_class }}">{{ row.pct_50d }}%</span>
                    </div>
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>
'''
