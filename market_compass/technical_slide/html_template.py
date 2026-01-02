"""HTML template for Technical Analysis tables only."""

TABLES_HTML_TEMPLATE = '''
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
            width: 1180px;
            height: 480px;
            padding: 0;
        }

        .tables-container {
            display: flex;
            gap: 20px;
            width: 100%;
            height: 100%;
        }

        .left-column {
            width: 570px;
            flex-shrink: 0;
        }

        .right-column {
            width: 570px;
            flex-shrink: 0;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }

        th {
            background: #1B3A5A;
            color: #FFFFFF;
            font-weight: 600;
            padding: 7px 6px;
            text-align: center;
            border: none;
        }

        th:first-child {
            text-align: left;
            padding-left: 10px;
            width: 80px;
        }

        th:nth-child(5) {
            width: 85px;
        }

        td {
            padding: 5px 6px;
            text-align: center;
            border-bottom: 1px solid #E8E8E8;
        }

        td:first-child {
            text-align: left;
            padding-left: 10px;
            font-weight: 500;
            width: 80px;
        }

        tr:nth-child(odd) td { background: #FFFFFF; }
        tr:nth-child(even) td { background: #F8F9FA; }

        .positive { color: #16A34A; font-weight: 600; }
        .negative { color: #DC2626; font-weight: 600; }
        .neutral { color: #1A1A2E; }
        .rsi-overbought { color: #DC2626; font-weight: 600; }
        .rsi-oversold { color: #16A34A; font-weight: 600; }

        .dmas-cell { padding: 4px 6px; }

        .dmas-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 5px;
        }

        .dmas-gauge {
            width: 45px;
            height: 5px;
            background: #E5E7EB;
            border-radius: 3px;
            overflow: hidden;
            flex-shrink: 0;
        }

        .dmas-fill {
            height: 100%;
            border-radius: 3px;
        }

        .dmas-value {
            font-weight: 700;
            font-size: 11px;
            min-width: 20px;
            text-align: right;
        }

        .dmas-fill.bullish { background: linear-gradient(90deg, #22C55E, #16A34A); }
        .dmas-fill.constructive { background: linear-gradient(90deg, #84CC16, #65A30D); }
        .dmas-fill.neutral { background: linear-gradient(90deg, #EAB308, #CA8A04); }
        .dmas-fill.cautious { background: linear-gradient(90deg, #F97316, #EA580C); }
        .dmas-fill.bearish { background: linear-gradient(90deg, #EF4444, #DC2626); }

        .outlook {
            padding: 3px 5px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 9px;
            display: inline-block;
            min-width: 65px;
        }

        .outlook-bullish { background: #DCFCE7; color: #166534; }
        .outlook-constructive { background: #ECFCCB; color: #3F6212; }
        .outlook-neutral { background: #FEF9C3; color: #854D0E; }
        .outlook-cautious { background: #FFEDD5; color: #C2410C; }
        .outlook-bearish { background: #FEE2E2; color: #DC2626; }
    </style>
</head>
<body>
    <div class="tables-container">
        <!-- LEFT: EQUITY -->
        <div class="left-column">
            <table>
                <thead>
                    <tr>
                        <th>Equity</th>
                        <th>Mkt Cap ($)</th>
                        <th>RSI</th>
                        <th>vs 50d</th>
                        <th>DMAS</th>
                        <th>Outlook</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in equity_rows %}
                    <tr>
                        <td>{{ row.name }}</td>
                        <td>{{ row.market_cap }}</td>
                        <td class="{{ row.rsi_class }}">{{ row.rsi }}</td>
                        <td class="{{ row.ma_class }}">{{ row.vs_50d_ma_fmt }}</td>
                        <td class="dmas-cell">
                            <div class="dmas-container">
                                <div class="dmas-gauge"><div class="dmas-fill {{ row.outlook_lower }}" style="width: {{ row.dmas }}%;"></div></div>
                                <span class="dmas-value {{ row.dmas_class }}">{{ row.dmas }}</span>
                            </div>
                        </td>
                        <td><span class="outlook outlook-{{ row.outlook_lower }}">{{ row.outlook }}</span></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- RIGHT: COMMODITY + CRYPTO -->
        <div class="right-column">
            <table>
                <thead>
                    <tr>
                        <th>Commodity</th>
                        <th>Mkt Cap ($)</th>
                        <th>RSI</th>
                        <th>vs 50d</th>
                        <th>DMAS</th>
                        <th>Outlook</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in commodity_rows %}
                    <tr>
                        <td>{{ row.name }}</td>
                        <td>{{ row.market_cap }}</td>
                        <td class="{{ row.rsi_class }}">{{ row.rsi }}</td>
                        <td class="{{ row.ma_class }}">{{ row.vs_50d_ma_fmt }}</td>
                        <td class="dmas-cell">
                            <div class="dmas-container">
                                <div class="dmas-gauge"><div class="dmas-fill {{ row.outlook_lower }}" style="width: {{ row.dmas }}%;"></div></div>
                                <span class="dmas-value {{ row.dmas_class }}">{{ row.dmas }}</span>
                            </div>
                        </td>
                        <td><span class="outlook outlook-{{ row.outlook_lower }}">{{ row.outlook }}</span></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>

            <table>
                <thead>
                    <tr>
                        <th>Crypto</th>
                        <th>Mkt Cap ($)</th>
                        <th>RSI</th>
                        <th>vs 50d</th>
                        <th>DMAS</th>
                        <th>Outlook</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in crypto_rows %}
                    <tr>
                        <td>{{ row.name }}</td>
                        <td>{{ row.market_cap }}</td>
                        <td class="{{ row.rsi_class }}">{{ row.rsi }}</td>
                        <td class="{{ row.ma_class }}">{{ row.vs_50d_ma_fmt }}</td>
                        <td class="dmas-cell">
                            <div class="dmas-container">
                                <div class="dmas-gauge"><div class="dmas-fill {{ row.outlook_lower }}" style="width: {{ row.dmas }}%;"></div></div>
                                <span class="dmas-value {{ row.dmas_class }}">{{ row.dmas }}</span>
                            </div>
                        </td>
                        <td><span class="outlook outlook-{{ row.outlook_lower }}">{{ row.outlook }}</span></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
'''
