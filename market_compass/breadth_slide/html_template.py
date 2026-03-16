"""HTML template for Composite Breadth Score table."""

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
            background: #FFFFFF;
            width: {{ width }}px;
            min-height: {{ height }}px;
            padding: {{ 14 * scale }}px {{ 16 * scale }}px;
            margin: 0;
        }

        /* ========== TITLE ========== */
        .title {
            color: #00B0F0;
            font-size: {{ 20 * scale }}px;
            font-weight: 700;
            margin-bottom: {{ 6 * scale }}px;
        }

        /* ========== SUBTITLE PARAGRAPH ========== */
        .subtitle {
            color: #555555;
            font-size: {{ 8 * scale }}px;
            line-height: 1.5;
            margin-bottom: {{ 12 * scale }}px;
        }

        .subtitle strong {
            color: #040C38;
        }

        /* ========== TABLE ========== */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: {{ 12 * scale }}px;
        }

        /* ========== HEADER ========== */
        th {
            background: #040C38;
            color: #FFFFFF;
            font-weight: 700;
            font-size: {{ 10 * scale }}px;
            letter-spacing: {{ 0.5 * scale }}px;
            text-transform: uppercase;
            padding: {{ 9 * scale }}px {{ 8 * scale }}px;
            text-align: center;
            border: none;
        }

        th:nth-child(2) {
            text-align: left;
            padding-left: {{ 12 * scale }}px;
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
        .cell {
            display: flex;
            align-items: center;
            justify-content: center;
            height: {{ 28 * scale }}px;
            padding: 0 {{ 8 * scale }}px;
        }

        /* ========== RANK COLUMN ========== */
        .cell.rank {
            font-weight: 700;
            font-size: {{ 15 * scale }}px;
            color: #C5A044;
        }

        /* ========== INDEX COLUMN ========== */
        .cell.index-name {
            justify-content: flex-start;
            gap: {{ 8 * scale }}px;
            padding-left: {{ 12 * scale }}px;
            white-space: nowrap;
        }

        .cell.index-name .flag {
            font-size: {{ 18 * scale }}px;
            line-height: 1;
        }

        .cell.index-name .name {
            font-weight: 700;
            color: #040C38;
            font-size: {{ 12 * scale }}px;
        }

        /* ========== COMPOSITE PILL ========== */
        .pill {
            display: inline-block;
            padding: {{ 3 * scale }}px {{ 14 * scale }}px;
            border-radius: {{ 12 * scale }}px;
            font-weight: 700;
            font-size: {{ 12 * scale }}px;
            color: #FFFFFF;
            min-width: {{ 44 * scale }}px;
            text-align: center;
        }

        .pill.green { background: #22C55E; }
        .pill.amber { background: #F59E0B; }
        .pill.red   { background: #EF4444; }

        /* ========== BAR + VALUE CELLS ========== */
        .cell.bar-val {
            gap: {{ 6 * scale }}px;
        }

        .mini-bar {
            width: {{ 55 * scale }}px;
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
            font-size: {{ 12 * scale }}px;
            min-width: {{ 28 * scale }}px;
            text-align: right;
        }

        .bar-value.green { color: #16A34A; }
        .bar-value.amber { color: #D97706; }
        .bar-value.red   { color: #DC2626; }

        /* ========== FOOTER ========== */
        .footer {
            margin-top: {{ 8 * scale }}px;
            font-size: {{ 7 * scale }}px;
            color: #999999;
            font-style: italic;
            text-align: right;
        }
    </style>
</head>
<body>
    <div class="title">Composite Breadth Score</div>
    <p class="subtitle">
        Breadth measures how broad-based a market move really is. We score each index across three dimensions:
        <strong>Trend</strong> &mdash; the percentage of members trading above their 50-day and 100-day moving averages.
        <strong>Momentum</strong> &mdash; MACD-based participation, including the balance of fresh buy vs sell signals.
        <strong>Balance</strong> &mdash; the inverse of Bollinger Band extremes: a healthy market has few members at either
        overbought or oversold levels. Higher balance = more sustainable trend.
        Rank 1 = strongest breadth.
    </p>

    <table>
        <thead>
            <tr>
                <th style="width: {{ 42 * scale }}px;">Rank</th>
                <th style="width: {{ 120 * scale }}px;">Index</th>
                <th style="width: {{ 80 * scale }}px;">Composite</th>
                <th>Trend</th>
                <th>Momentum</th>
                <th>Balance</th>
            </tr>
        </thead>
        <tbody>
            {% for row in rows %}
            <tr>
                <td>
                    <div class="cell rank">{{ row.rank }}</div>
                </td>
                <td>
                    <div class="cell index-name">
                        <span class="flag">{{ row.flag }}</span>
                        <span class="name">{{ row.name }}</span>
                    </div>
                </td>
                <td>
                    <div class="cell">
                        <span class="pill {{ row.composite_class }}">{{ row.composite }}</span>
                    </div>
                </td>
                <td>
                    <div class="cell bar-val">
                        <div class="mini-bar"><div class="mini-fill {{ row.trend_class }}" style="width: {{ row.trend }}%;"></div></div>
                        <span class="bar-value {{ row.trend_class }}">{{ row.trend_int }}</span>
                    </div>
                </td>
                <td>
                    <div class="cell bar-val">
                        <div class="mini-bar"><div class="mini-fill {{ row.momentum_class }}" style="width: {{ row.momentum }}%;"></div></div>
                        <span class="bar-value {{ row.momentum_class }}">{{ row.momentum_int }}</span>
                    </div>
                </td>
                <td>
                    <div class="cell bar-val">
                        <div class="mini-bar"><div class="mini-fill {{ row.balance_class }}" style="width: {{ row.balance }}%;"></div></div>
                        <span class="bar-value {{ row.balance_class }}">{{ row.balance_int }}</span>
                    </div>
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    <p class="footer">Source: Bloomberg, Herculis Group</p>
</body>
</html>
'''
