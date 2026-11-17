# Shock re-entry S

**Europe (the test): PASS** · US (information only, S was found on US data): fail

## Europe — STOXX 600, R0 base, one run

Sample 1994-02-03 → 2026-11-16. Cash: ECB Euribor 3M monthly (from 1994-01); FRED unreachable. Equity = SXXP price change + 3%/252. Episodes: 8.

| Gate | Evidence | Result |
|---|---|---|
| FULL Sharpe R0+S > R0 | 0.5198 > 0.4380 | PASS |
| (report) H1 Sharpe R0+S vs R0 | 0.9069 vs 0.8787 | report |
| (report) H2 Sharpe R0+S vs R0 | 0.2557 vs 0.1408 | report |
| Weekday-mean Sharpe | 0.5378 > 0.4679 (MON 0.561/0.467, TUE 0.520/0.438, WED 0.574/0.527, THU 0.531/0.465, FRI 0.504/0.443) | PASS |
| FULL maxDD >= R0 - 0.02 | -0.3252 vs -0.3452 | PASS |
| Placebo: real gain > 90th pct | +0.0818 vs +0.0359 (placebo share >= real 0.011) | PASS |
| More than half of episodes add value | 7 of 8 | PASS |

| Period | Rule | Sharpe | CAGR | Max DD |
|---|---|---:|---:|---:|
| FULL | R0 | 0.438 | 6.69% | -34.5% |
| FULL | R0+S | 0.520 | 7.93% | -32.5% |
| H1 to 2006 | R0 | 0.879 | 14.17% | -21.8% |
| H1 to 2006 | R0+S | 0.907 | 15.19% | -18.7% |
| H2 from 2007 | R0 | 0.141 | 2.10% | -34.5% |
| H2 from 2007 | R0+S | 0.256 | 3.47% | -32.5% |
| P2021 | R0 | 0.491 | 6.92% | -16.4% |
| P2021 | R0+S | 0.535 | 7.47% | -16.4% |

| 1-year high | Low | Entry | End | Reason | Fall | P&L vs R0 |
|---|---|---|---|---|---:|---:|
| 1998-07-20 | 1998-10-08 | 1998-10-14 | 1999-01-07 | base invested | -32.7% | +26.11% |
| 2001-07-02 | 2001-09-21 | 2002-06-11 | 2002-07-11 | stop | -29.3% | -10.80% |
| 2011-07-07 | 2011-09-22 | 2012-06-07 | 2012-07-05 | base invested | -22.1% | +6.54% |
| 2015-11-30 | 2016-02-11 | 2016-08-10 | 2016-09-08 | base invested | -21.2% | +1.35% |
| 2015-12-01 | 2016-02-11 | 2016-11-22 | 2016-12-08 | base invested | -21.0% | +2.17% |
| 2020-02-19 | 2020-03-18 | 2020-03-25 | 2020-07-09 | base invested | -35.5% | +15.12% |
| 2022-01-05 | 2022-03-08 | 2022-03-22 | 2022-04-07 | base invested | -16.0% | +0.57% |
| 2025-03-03 | 2025-04-09 | 2025-04-24 | 2025-05-08 | base invested | -16.6% | +2.53% |

Price-only returns (no dividend proxy): Sharpe R0+S 0.345 vs R0 0.261.

## US — S&P 500, R0 base, 1955-2026 (information only)

| Gate | Evidence | Result |
|---|---|---|
| FULL Sharpe R0+S > R0 | 0.5453 > 0.5142 | PASS |
| H1 Sharpe R0+S > R0 | 0.4379 > 0.4636 | FAIL |
| H2 Sharpe R0+S > R0 | 0.6443 > 0.5616 | PASS |
| Weekday-mean Sharpe | 0.5304 > 0.4941 (MON 0.545/0.520, TUE 0.545/0.514, WED 0.540/0.497, THU 0.524/0.482, FRI 0.499/0.457) | PASS |
| FULL maxDD >= R0 - 0.02 | -0.3295 vs -0.3295 | PASS |
| maxDD 2000-03..2003-03 | -0.1307 vs -0.1307 | PASS |
| maxDD 2007-10..2009-06 | -0.1129 vs -0.1129 | PASS |
| maxDD 2022-01..2022-12 | -0.0935 vs -0.0935 | PASS |
| Sharpe without 2020 and 2025 | 0.5237 > 0.5192 | PASS |
| Bootstrap not-better < 0.10 | 0.1185 | FAIL |
| Placebo: real gain > 90th pct | +0.0311 vs +0.0195 (placebo share >= real 0.036) | PASS |
| More than half of episodes add value | 6 of 7 | PASS |

| Period | Rule | Sharpe | CAGR | Max DD |
|---|---|---:|---:|---:|
| FULL | R0 | 0.514 | 9.46% | -33.0% |
| FULL | R0+S | 0.545 | 9.99% | -33.0% |
| H1 to 1990 | R0 | 0.464 | 10.51% | -33.0% |
| H1 to 1990 | R0+S | 0.438 | 10.34% | -33.0% |
| H2 from 1991 | R0 | 0.562 | 8.41% | -24.0% |
| H2 from 1991 | R0+S | 0.644 | 9.63% | -24.0% |
| P2021 | R0 | 0.844 | 12.22% | -11.5% |
| P2021 | R0+S | 0.949 | 13.66% | -11.5% |

| 1-year high | Low | Entry | End | Reason | Fall | P&L vs R0 |
|---|---|---|---|---|---:|---:|
| 1987-08-25 | 1987-10-19 | 1987-10-22 | 1987-12-04 | stop | -33.2% | -8.15% |
| 1987-10-05 | 1987-12-04 | 1988-08-29 | 1988-10-06 | base invested | -31.7% | +3.18% |
| 1990-07-16 | 1990-10-11 | 1990-12-04 | 1991-02-07 | base invested | -19.9% | +8.03% |
| 1998-07-17 | 1998-08-31 | 1998-09-23 | 1998-11-05 | base invested | -19.3% | +6.97% |
| 2010-04-23 | 2010-07-02 | 2010-08-02 | 2010-08-05 | base invested | -16.0% | +0.61% |
| 2020-02-19 | 2020-03-23 | 2020-03-25 | 2020-06-04 | base invested | -33.9% | +19.24% |
| 2025-02-19 | 2025-04-08 | 2025-04-24 | 2025-06-05 | base invested | -18.9% | +7.76% |

## US product by period — Sharpe (1999-02 → 2026-08)

| Window | Buy & hold | V2+F | V2+F+S | K1+F macro-only | K1+F+S macro-only |
|---|---:|---:|---:|---:|---:|
| FULL 1999-02 to 2026-08 | 0.42 | 0.67 | 0.68 | 0.68 | 0.69 |
| First half 1999-2012 | 0.12 | 0.37 | 0.37 | 0.40 | 0.40 |
| Second half 2013-2026 | 0.82 | 0.99 | 1.00 | 0.97 | 0.99 |
| 2009-2013 | 1.35 | 1.09 | 1.09 | 1.09 | 1.09 |
| 2014-2018 | 0.64 | 0.60 | 0.60 | 0.38 | 0.38 |
| 2019-2023 | 0.71 | 0.87 | 0.91 | 0.99 | 1.04 |
| 2021-2026 | 0.74 | 0.96 | 0.96 | 1.06 | 1.07 |
| 2023-2026 | 1.15 | 1.21 | 1.22 | 1.25 | 1.25 |
| Dot-com bear 2000-03 to 2002-10 | -1.15 | -0.97 | -0.97 | -0.79 | -0.79 |
| 2008 crisis 2007-10 to 2009-03 | -1.33 | -0.98 | -0.98 | -0.97 | -0.97 |
| 2009 rebound Mar-Dec | 2.77 | 1.82 | 1.82 | 2.06 | 2.06 |
| COVID rebound 24 Mar-31 Aug 2020 | 3.51 | 2.30 | 2.71 | 2.57 | 3.10 |
| 2022 full year | -0.77 | -0.92 | -0.92 | -0.70 | -0.70 |
| 2023 rebound | 1.49 | 0.94 | 0.94 | 1.02 | 1.02 |
| 2025 full year | 0.76 | 1.39 | 1.42 | 1.51 | 1.53 |

## US product by period — Total return (1999-02 → 2026-08)

| Window | Buy & hold | V2+F | V2+F+S | K1+F macro-only | K1+F+S macro-only |
|---|---:|---:|---:|---:|---:|
| FULL 1999-02 to 2026-08 | +876.7% | +1043.3% | +1075.6% | +1046.0% | +1083.2% |
| First half 1999-2012 | +43.2% | +122.9% | +122.9% | +135.3% | +135.3% |
| Second half 2013-2026 | +582.3% | +412.8% | +427.4% | +387.1% | +402.9% |
| 2009-2013 | +202.3% | +92.3% | +92.3% | +90.9% | +90.9% |
| 2014-2018 | +49.7% | +34.4% | +34.4% | +19.6% | +19.6% |
| 2019-2023 | +106.8% | +77.1% | +81.7% | +88.8% | +94.6% |
| 2021-2026 | +121.4% | +115.7% | +116.2% | +128.1% | +128.5% |
| 2023-2026 | +110.3% | +88.7% | +89.2% | +92.4% | +92.7% |
| Dot-com bear 2000-03 to 2002-10 | -47.4% | -11.8% | -11.8% | -8.0% | -8.0% |
| 2008 crisis 2007-10 to 2009-03 | -54.9% | -5.3% | -5.3% | -4.7% | -4.7% |
| 2009 rebound Mar-Dec | +68.3% | +18.7% | +18.7% | +20.1% | +20.1% |
| COVID fall 19 Feb-23 Mar 2020 | -33.8% | -7.5% | -7.5% | -6.8% | -6.8% |
| COVID rebound 24 Mar-31 Aug 2020 | +57.8% | +10.9% | +13.8% | +11.7% | +15.2% |
| 2022 full year | -18.2% | -10.0% | -10.0% | -6.5% | -6.5% |
| 2023 rebound | +26.2% | +15.3% | +15.3% | +16.7% | +16.7% |
| Liberation Day fall 19 Feb-8 Apr 2025 | -18.8% | -4.6% | -4.6% | -5.0% | -5.0% |
| Liberation Day rebound 9 Apr-30 Jun 2025 | +24.9% | +7.9% | +8.2% | +8.5% | +8.8% |
| 2025 full year | +17.8% | +18.9% | +19.2% | +20.0% | +20.2% |

## US product by period — Worst fall (1999-02 → 2026-08)

| Window | Buy & hold | V2+F | V2+F+S | K1+F macro-only | K1+F+S macro-only |
|---|---:|---:|---:|---:|---:|
| FULL 1999-02 to 2026-08 | -55.3% | -25.5% | -25.5% | -23.5% | -23.5% |
| First half 1999-2012 | -55.3% | -25.5% | -25.5% | -23.5% | -23.5% |
| Second half 2013-2026 | -33.8% | -15.9% | -15.9% | -12.0% | -12.0% |
| 2009-2013 | -18.7% | -17.8% | -17.8% | -18.4% | -18.4% |
| 2014-2018 | -19.4% | -10.5% | -10.5% | -10.7% | -10.7% |
| 2019-2023 | -33.8% | -15.9% | -15.9% | -12.0% | -12.0% |
| 2021-2026 | -24.5% | -15.9% | -15.9% | -12.0% | -12.0% |
| 2023-2026 | -18.8% | -9.9% | -9.9% | -10.2% | -10.2% |
| Dot-com bear 2000-03 to 2002-10 | -47.4% | -16.6% | -16.6% | -16.3% | -16.3% |
| 2008 crisis 2007-10 to 2009-03 | -55.3% | -8.0% | -8.0% | -7.4% | -7.4% |
| 2009 rebound Mar-Dec | -6.9% | -5.0% | -5.0% | -4.7% | -4.7% |
| COVID fall 19 Feb-23 Mar 2020 | -33.8% | -11.5% | -11.5% | -10.9% | -10.9% |
| COVID rebound 24 Mar-31 Aug 2020 | -7.1% | -5.5% | -5.5% | -5.0% | -5.0% |
| 2022 full year | -24.5% | -15.9% | -15.9% | -12.0% | -12.0% |
| 2023 rebound | -9.9% | -9.9% | -9.9% | -10.2% | -10.2% |
| Liberation Day fall 19 Feb-8 Apr 2025 | -18.8% | -5.9% | -5.9% | -6.4% | -6.4% |
| Liberation Day rebound 9 Apr-30 Jun 2025 | -5.4% | -1.3% | -1.3% | -1.3% | -1.3% |
| 2025 full year | -18.8% | -5.9% | -5.9% | -6.4% | -6.4% |

US product S episodes (R1 base): 1998-09-23→1998-11-05 (base invested), 2020-03-25→2020-06-04 (base invested), 2025-04-24→2025-06-05 (base invested)

## Method

As spec 6s. A trigger while the base weight in force is already 1 is not an episode (S has nothing to do); after an episode the scan restarts the day after it ends. R0 base = run_fast.states (reproduces trading.sleeve_weights). Europe DMAS = gauge.dmas_on on SXXP closes, every trading day from 1988 (cached out/dmas_sxxp_daily.csv). Placebo: entries on random days with base weight 0, L = lowest of the 21 closes ending there, same stop, base-invested end and 126-day limit.
