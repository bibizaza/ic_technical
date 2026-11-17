# Frozen US rules by period and in stress windows (information only, no gate)

Rules unchanged: BH = S&P 500 total return; R0, R1 = all-to-cash Delphi rule; V2 = R1 x volatility size; K1 = V2 with 70% SPY + 3x10% C5 sector ETFs. K1 ends 2026-08-31 (C5 picks stop); the others end 2026-11-13. Code: `stress_periods.py`.

Matched end 2026-08-31 (total return / Sharpe / worst fall): 2019+: BH 245%/0.80/-34%, R1 193%/0.91/-22%, V2 174%/0.95/-16%, K1 178%/1.04/-11%. 2021+: BH 121%/0.74/-25%, R1 109%/0.82/-22%, V2 109%/0.91/-16%, K1 109%/0.99/-11%. 2023+: BH 110%/1.15/-19%, R1 89%/1.17/-10%, V2 83%/1.13/-10%, K1 74%/1.09/-11%.

| Window | Rule | Total | Sharpe | Worst fall |
|---|---|---:|---:|---:|
| 2009-2013 | BH | +202.3% | 1.35 | -18.7% |
| 2009-2013 | R0 | +58.0% | 0.83 | -15.8% |
| 2009-2013 | R1 | +112.8% | 1.05 | -18.7% |
| 2009-2013 | V2 | +89.3% | 1.07 | -17.8% |
| 2009-2013 | K1 | +86.3% | 1.09 | -16.1% |
| 2014-2018 | BH | +49.7% | 0.64 | -19.4% |
| 2014-2018 | R0 | +22.0% | 0.42 | -10.9% |
| 2014-2018 | R1 | +32.3% | 0.55 | -10.8% |
| 2014-2018 | V2 | +30.7% | 0.55 | -10.5% |
| 2014-2018 | K1 | +22.1% | 0.43 | -10.2% |
| 2019-2023 | BH | +106.8% | 0.71 | -33.8% |
| 2019-2023 | R0 | +76.0% | 0.86 | -12.7% |
| 2019-2023 | R1 | +78.8% | 0.76 | -21.7% |
| 2019-2023 | V2 | +73.4% | 0.83 | -15.9% |
| 2019-2023 | K1 | +85.0% | 0.99 | -11.3% |
| 2021-2026 | BH | +118.3% | 0.70 | -24.5% |
| 2021-2026 | R0 | +96.2% | 0.84 | -11.5% |
| 2021-2026 | R1 | +106.2% | 0.76 | -21.7% |
| 2021-2026 | V2 | +105.9% | 0.84 | -15.9% |
| 2021-2026 | K1 | +108.7% | 0.99 | -11.3% |
| 2023-2026 | BH | +107.3% | 1.05 | -18.8% |
| 2023-2026 | R0 | +64.8% | 0.83 | -8.4% |
| 2023-2026 | R1 | +85.8% | 1.04 | -9.9% |
| 2023-2026 | V2 | +80.2% | 1.01 | -9.9% |
| 2023-2026 | K1 | +73.5% | 1.09 | -10.5% |
| COVID fall 19 Feb-23 Mar 2020 | BH | -33.8% |  | -33.8% |
| COVID fall 19 Feb-23 Mar 2020 | R0 | -7.5% |  | -12.7% |
| COVID fall 19 Feb-23 Mar 2020 | R1 | -7.5% |  | -12.7% |
| COVID fall 19 Feb-23 Mar 2020 | V2 | -7.5% |  | -11.5% |
| COVID fall 19 Feb-23 Mar 2020 | K1 | -6.6% |  | -10.8% |
| COVID rebound 24 Mar-31 Aug 2020 | BH | +57.8% | 3.51 | -7.1% |
| COVID rebound 24 Mar-31 Aug 2020 | R0 | +12.4% | 1.91 | -7.1% |
| COVID rebound 24 Mar-31 Aug 2020 | R1 | +12.4% | 1.91 | -7.1% |
| COVID rebound 24 Mar-31 Aug 2020 | V2 | +9.3% | 2.00 | -5.5% |
| COVID rebound 24 Mar-31 Aug 2020 | K1 | +10.2% | 2.27 | -5.0% |
| 2022 full year | BH | -18.2% | -0.77 | -24.5% |
| 2022 full year | R0 | -7.5% | -2.16 | -9.3% |
| 2022 full year | R1 | -13.8% | -0.84 | -21.7% |
| 2022 full year | V2 | -10.0% | -0.92 | -15.9% |
| 2022 full year | K1 | -5.7% | -0.63 | -11.3% |
| 2022 fall 3 Jan-12 Oct | BH | -24.5% | -1.42 | -24.5% |
| 2022 fall 3 Jan-12 Oct | R0 | -8.8% | -2.67 | -9.3% |
| 2022 fall 3 Jan-12 Oct | R1 | -15.0% | -0.99 | -21.7% |
| 2022 fall 3 Jan-12 Oct | V2 | -11.2% | -1.10 | -15.9% |
| 2022 fall 3 Jan-12 Oct | K1 | -7.0% | -0.77 | -11.3% |
| 2023 rebound | BH | +26.2% | 1.49 | -9.9% |
| 2023 rebound | R0 | +12.1% | 0.72 | -7.6% |
| 2023 rebound | R1 | +15.1% | 0.91 | -9.9% |
| 2023 rebound | V2 | +15.3% | 0.94 | -9.9% |
| 2023 rebound | K1 | +15.5% | 0.96 | -10.5% |
| Liberation Day fall 19 Feb-8 Apr 2025 | BH | -18.8% |  | -18.8% |
| Liberation Day fall 19 Feb-8 Apr 2025 | R0 | -4.6% |  | -5.9% |
| Liberation Day fall 19 Feb-8 Apr 2025 | R1 | -4.6% |  | -5.9% |
| Liberation Day fall 19 Feb-8 Apr 2025 | V2 | -4.6% |  | -5.9% |
| Liberation Day fall 19 Feb-8 Apr 2025 | K1 | -4.2% |  | -5.1% |
| Liberation Day rebound 9 Apr-30 Jun 2025 | BH | +24.9% |  | -5.4% |
| Liberation Day rebound 9 Apr-30 Jun 2025 | R0 | +4.6% |  | -1.3% |
| Liberation Day rebound 9 Apr-30 Jun 2025 | R1 | +4.6% |  | -1.3% |
| Liberation Day rebound 9 Apr-30 Jun 2025 | V2 | +4.3% |  | -1.2% |
| Liberation Day rebound 9 Apr-30 Jun 2025 | K1 | +3.4% |  | -1.2% |
| 2025 full year | BH | +17.8% | 0.76 | -18.8% |
| 2025 full year | R0 | +15.9% | 1.11 | -5.9% |
| 2025 full year | R1 | +15.9% | 1.11 | -5.9% |
| 2025 full year | V2 | +15.0% | 1.08 | -5.9% |
| 2025 full year | K1 | +11.9% | 0.91 | -5.1% |

Sharpe is blank for windows under 60 days.

## Sector picks (C5 -> ETF, 1999-2026)

Share of months picked: XLP 68%, XLU 65%, XLE 61%, XLY 38%, XLV 20%, IYZ 19%, XLF 15%, XLK 11%, XLC 2%, XLB 0%, XLI 0%. Two or more of XLP/XLU/XLE in 68% of months.
Reason (macro_health `tools/regime_market/sectors.py:102` `trend_pred`): each sector forecast = intercept ("normal level") + growth slope + inflation slope, fitted on beta-adjusted returns. Low-beta sectors have a positive average beta-adjusted return, so the intercept keeps them on top.

## Why the Navigator looks much better

The Navigator headline (+6.3% a year, 2006-2026) is a long-short, beta-adjusted spread on 12 Ken French industries: top 3 minus bottom 3, 100% each side. On raw returns it is +4.2% (IC p 0.18). Long-only on real ETFs (`macro_health/results/regime_etf_test.md`, 1999-2026): 80% SPY + 20% favoured alpha +0.6% a year; SPY + 50% favoured - 50% avoided alpha +2.6% (t 2.21). K1 holds only 30% in favoured sectors and never underweights the avoided ones, so it keeps a small part of the signal.
