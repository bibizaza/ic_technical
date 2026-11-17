# Breadth thrust re-entry B

Verdict: **PASS**

Primary: R0+B vs R0, S&P 500, Tuesday base, 1965-04-08 → 2026-11-13. Thrust days: 23. Episodes: 10. S size in the product: live.
Breadth: UNICORN_NYSE_ADV/UNICORN_NYSE_DEC to 2020-02-10, then BBG_NYSE_ADV/BBG_NYSE_DEC. Overlap days: 3919; median abs difference of the advance share 0.0444.

| Gate | Evidence | Result |
|---|---|---|
| G1 FULL Sharpe R0+B > R0 | 0.5077 > 0.4376 | PASS |
| G1 H1 Sharpe R0+B > R0 | 0.3495 > 0.2619 | PASS |
| G1 H2 Sharpe R0+B > R0 | 0.6179 > 0.5616 | PASS |
| G1 weekday-mean Sharpe | 0.4921 > 0.4246 (MON 0.522/0.461, TUE 0.508/0.438, WED 0.482/0.416, THU 0.487/0.412, FRI 0.462/0.396) | PASS |
| G2 FULL maxDD >= R0 - 0.02 | -0.3295 vs -0.3295 | PASS |
| G2 maxDD 2000-03..2003-03 | -0.1307 vs -0.1307 | PASS |
| G2 maxDD 2007-10..2009-06 | -0.0967 vs -0.1129 | PASS |
| G2 maxDD 2022-01..2022-12 | -0.0935 vs -0.0935 | PASS |
| G3 Sharpe without 2020 and 2025 | 0.5129 > 0.4414 | PASS |
| G4 bootstrap not-better < 0.10 | 0.0000 | PASS |
| G5 placebo: real gain > 90th pct | +0.0702 vs +0.0257 (placebo share >= real 0.001) | PASS |
| G6 more than half of episodes add value | 9 of 10 | PASS |
| G7 product: V2+F+S+B Sharpe >= V2+F+S and maxDD >= -0.02 | 0.8090 vs 0.7819; -0.2554 vs -0.2554 | PASS |
| G8 episodes from 2003 (rule public): total P&L vs R0 > 0 | +30.68% over 4 episodes (added to the code after the first run; spec 6t had it) | PASS |

| Period | Rule | Sharpe | CAGR | Max DD |
|---|---|---:|---:|---:|
| FULL | R0 | 0.438 | 8.98% | -33.0% |
| FULL | R0+B | 0.508 | 9.90% | -33.0% |
| H1 to 1990 | R0 | 0.262 | 9.78% | -33.0% |
| H1 to 1990 | R0+B | 0.349 | 10.84% | -33.0% |
| H2 from 1991 | R0 | 0.562 | 8.41% | -24.0% |
| H2 from 1991 | R0+B | 0.618 | 9.22% | -24.0% |
| P2021 | R0 | 0.844 | 12.22% | -11.5% |
| P2021 | R0+B | 0.877 | 12.75% | -10.3% |

| Thrust (entry) | End | Reason | Stop level L | P&L vs R0 |
|---|---|---|---:|---:|
| 1967-01-16 | 1967-02-09 | base invested | 76.60 | +2.83% |
| 1971-12-08 | 1972-01-06 | base invested | 90.16 | +6.24% |
| 1975-01-08 | 1975-02-06 | base invested | 65.01 | +10.84% |
| 1982-08-23 | 1982-09-09 | base invested | 102.42 | +5.85% |
| 1991-02-05 | 1991-02-07 | base invested | 306.01 | +0.00% |
| 1992-01-06 | 1992-01-09 | base invested | 375.22 | +0.16% |
| 2009-03-23 | 2009-06-04 | base invested | 676.53 | +16.26% |
| 2009-07-23 | 2009-08-06 | base invested | 855.16 | +2.47% |
| 2019-01-09 | 2019-03-07 | base invested | 2351.10 | +6.69% |
| 2023-01-12 | 2023-02-09 | base invested | 3583.07 | +2.82% |

## Product by window (total return / Sharpe / worst fall)

| Window | Buy & hold | V2+F+S | V2+F+S+B |
|---|---:|---:|---:|
| FULL 1999-02 to 2026-08 | +863.0% / 0.42 / -55.3% | +1102.8% / 0.68 / -25.5% | +1252.8% / 0.71 / -25.5% |
| First half 1999-2012 | +43.2% / 0.12 / -55.3% | +122.9% / 0.37 / -25.5% | +134.1% / 0.39 / -25.5% |
| Second half 2013-2026 | +572.7% / 0.80 / -33.8% | +439.5% / 0.99 / -15.9% | +477.8% / 1.03 / -15.9% |
| 2009-2013 | +202.3% / 1.35 / -18.7% | +92.3% / 1.09 / -17.8% | +102.0% / 1.15 / -17.8% |
| 2014-2018 | +49.7% / 0.64 / -19.4% | +34.4% / 0.60 / -10.5% | +34.4% / 0.60 / -10.5% |
| 2019-2023 | +106.8% / 0.71 / -33.8% | +87.2% / 0.95 / -15.9% | +100.4% / 1.05 / -15.9% |
| 2021-2026 | +118.3% / 0.70 / -24.5% | +114.7% / 0.90 / -15.9% | +120.4% / 0.93 / -15.9% |
| 2023-2026 | +107.3% / 1.05 / -18.8% | +87.9% / 1.10 / -9.9% | +92.9% / 1.14 / -9.9% |
| Dot-com bear 2000-03 to 2002-10 | -47.4% / -1.15 / -47.4% | -11.8% / -0.97 / -16.6% | -11.8% / -0.97 / -16.6% |
| 2008 crisis 2007-10 to 2009-03 | -54.9% / -1.33 / -55.3% | -5.3% / -0.98 / -8.0% | -5.3% / -0.98 / -8.0% |
| 2009 rebound Mar-Dec | +68.3% / 2.77 / -6.9% | +18.7% / 1.82 / -5.0% | +24.6% / 2.13 / -5.0% |
| COVID fall 19 Feb-23 Mar 2020 | -33.8% / nan / -33.8% | -7.5% / nan / -11.5% | -7.5% / nan / -11.5% |
| COVID rebound 24 Mar-31 Aug 2020 | +57.8% / 3.51 / -7.1% | +17.2% / 3.04 / -5.5% | +17.2% / 3.04 / -5.5% |
| 2022 full year | -18.2% / -0.77 / -24.5% | -10.0% / -0.92 / -15.9% | -10.0% / -0.92 / -15.9% |
| 2023 rebound | +26.2% / 1.49 / -9.9% | +15.3% / 0.94 / -9.9% | +18.3% / 1.13 / -9.9% |
| Liberation Day fall 19 Feb-8 Apr 2025 | -18.8% / nan / -18.8% | -4.6% / nan / -5.9% | -4.6% / nan / -5.9% |
| Liberation Day rebound 9 Apr-30 Jun 2025 | +24.9% / nan / -5.4% | +8.8% / nan / -2.2% | +8.8% / nan / -2.2% |
| 2025 full year | +17.8% / 0.76 / -18.8% | +19.9% / 1.44 / -5.9% | +19.9% / 1.44 / -5.9% |

B episodes on the R1 base: 2009-03-23→2009-06-04 (base invested), 2019-01-09→2019-03-07 (base invested), 2023-01-12→2023-02-09 (base invested)

## Report only (written after the verdict)

Zweig thrust days: 1971-12-03, 1974-10-10, 1975-01-03, 1982-08-20, 1984-08-03, 2004-05-25, 2009-03-18, 2011-10-14, 2013-10-18, 2013-10-22, 2015-10-08, 2019-01-07, 2020-05-27, 2023-03-31, 2023-11-03, 2024-08-19, 2025-04-24

## Report only: single breadth source, 1965-04 → 2020-02-10 (spec 6t data note; no G7)

Episodes: 9. Gates met: 11 of 11.

| Gate | Evidence | Result |
|---|---|---|
| G1 FULL Sharpe R0+B > R0 | 0.4613 > 0.3848 | PASS |
| G1 H1 Sharpe R0+B > R0 | 0.3495 > 0.2619 | PASS |
| G1 H2 Sharpe R0+B > R0 | 0.5590 > 0.4941 | PASS |
| G1 weekday-mean Sharpe | 0.4745 > 0.4023 (MON 0.497/0.433, TUE 0.461/0.385, WED 0.502/0.433, THU 0.471/0.391, FRI 0.442/0.370) | PASS |
| G2 FULL maxDD >= R0 - 0.02 | -0.3295 vs -0.3295 | PASS |
| G2 maxDD 2000-03..2003-03 | -0.1307 vs -0.1307 | PASS |
| G2 maxDD 2007-10..2009-06 | -0.0967 vs -0.1129 | PASS |
| G3 Sharpe without 2020 and 2025 | 0.4613 > 0.3848 | PASS |
| G4 bootstrap not-better < 0.10 | 0.0000 | PASS |
| G5 placebo: real gain > 90th pct | +0.0765 vs +0.0273 (placebo share >= real 0.000) | PASS |
| G6 more than half of episodes add value | 8 of 9 | PASS |
