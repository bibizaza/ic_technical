# "Panic is over" floor P (credit + VIX) and fallback F (VIX only)

Adoption order (pre-registered): P if P passes; F only if P fails and F passes. P FAIL, F FAIL.

## P — US high-yield OAS (lagged 1 day) + VIX < 30

Verdict: **FAIL**

Sample: 2000-09-21 → 2026-11-13. Episodes (R0, Tuesday): 7.

| Gate | Evidence | Result |
|---|---|---|
| G1 FULL Sharpe P > R0 | 0.5684 > 0.6088 | FAIL |
| G1 H1 Sharpe P > R0 | 0.2176 > 0.2958 | FAIL |
| G1 H2 Sharpe P > R0 | 0.7322 > 0.7551 | FAIL |
| G1 weekday-mean Sharpe | 0.5392 > 0.5740 (MON 0.554/0.577, TUE 0.568/0.609, WED 0.487/0.521, THU 0.544/0.579, FRI 0.542/0.583) | FAIL |
| G2 FULL maxDD >= R0 - 0.02 | -0.2043 vs -0.1755 | FAIL |
| G2 maxDD 2000-03..2003-03 | -0.1127 vs -0.1065 | PASS |
| G2 maxDD 2007-10..2009-06 | -0.1439 vs -0.1129 | FAIL |
| G2 maxDD 2022-01..2022-12 | -0.1518 vs -0.0935 | FAIL |
| G3 Sharpe without 2020 and 2025 | 0.5778 > 0.6280 | FAIL |
| G4 bootstrap not-better < 0.10 | 0.9465 | FAIL |
| G5 real gain > placebo 90th pct | -0.0404 vs +0.0521 (placebo share >= real 0.973) | FAIL |
| G6 more than half of episodes add value | 3 of 7 | FAIL |
| G7 V2+rule Sharpe >= V2 and maxDD >= V2 - 0.02 | 0.6733 vs 0.6999; -0.2182 vs -0.2127 | FAIL |

| Period | Rule | Sharpe | CAGR | Max DD |
|---|---|---:|---:|---:|
| FULL | R0 | 0.609 | 7.96% | -17.6% |
| FULL | R0+rule | 0.568 | 7.53% | -20.4% |
| FULL | V2 | 0.700 | 9.33% | -21.3% |
| FULL | V2+rule | 0.673 | 9.03% | -21.8% |
| H1 | R0 | 0.296 | 4.95% | -17.6% |
| H1 | R0+rule | 0.218 | 4.20% | -20.4% |
| H1 | V2 | 0.434 | 6.46% | -21.3% |
| H1 | V2+rule | 0.358 | 5.70% | -21.8% |
| H2 | R0 | 0.755 | 9.65% | -15.8% |
| H2 | R0+rule | 0.732 | 9.40% | -16.8% |
| H2 | V2 | 0.825 | 10.94% | -17.8% |
| H2 | V2+rule | 0.821 | 10.91% | -17.8% |
| P2021 | R0 | 0.844 | 12.22% | -11.5% |
| P2021 | R0+rule | 0.743 | 11.13% | -16.8% |
| P2021 | V2 | 0.844 | 13.14% | -15.9% |
| P2021 | V2+rule | 0.806 | 12.70% | -16.1% |

Rebound capture (share of the gap between the base and buy & hold that the rule closes):

| Window | Base | Base+rule | Buy & hold | Captured |
|---|---:|---:|---:|---:|
| COVID rebound 24 Mar-31 Aug 2020 (R0) | +12.4% | +14.1% | +57.8% | 4% |
| COVID rebound 24 Mar-31 Aug 2020 (V2) | +9.3% | +10.9% | +57.8% | 3% |
| Liberation Day rebound 9 Apr-30 Jun 2025 (R0) | +4.6% | +5.2% | +24.9% | 3% |
| Liberation Day rebound 9 Apr-30 Jun 2025 (V2) | +4.3% | +4.9% | +24.9% | 3% |

Episodes (R0 base, Tuesday; stress = LF98OAS):

| Spell start | Entry | End | Reason | Stress | Peak | VIX | P&L vs R0 |
|---|---|---|---|---:|---:|---:|---:|
| 2000-09-19 | 2001-02-13 | 2001-02-27 | price stop | 6.72 | 9.04 | 21.4 | -3.02% |
| 2007-11-20 | 2008-06-10 | 2008-07-15 | price stop | 6.15 | 8.31 | 23.2 | -3.44% |
| 2015-12-15 | 2016-03-22 | 2016-04-05 | base invested | 6.24 | 8.39 | 14.2 | +0.70% |
| 2018-10-23 | 2019-02-19 | 2019-03-05 | base invested | 4.02 | 5.37 | 14.9 | -0.29% |
| 2020-03-03 | 2020-05-26 | 2020-06-02 | base invested | 6.80 | 11.00 | 28.0 | +1.39% |
| 2022-01-25 | 2022-08-09 | 2022-09-27 | price stop | 4.26 | 5.83 | 21.8 | -6.02% |
| 2025-03-04 | 2025-05-13 | 2025-06-03 | base invested | 3.05 | 4.53 | 18.2 | +0.53% |

V2 product episodes (R1 spells): 2001-02-13→2001-02-27 (price stop), 2008-06-10→2008-07-15 (price stop), 2016-03-22→2016-04-05 (base invested), 2019-02-19→2019-03-05 (base invested), 2020-05-26→2020-06-02 (base invested), 2022-08-09→2022-09-06 (price stop), 2025-05-13→2025-06-03 (base invested)

## F — VIX half of spell peak + VIX < 30 (fallback, 1990+)

Verdict: **FAIL**

Sample: 1990-02-08 → 2026-11-13. Episodes (R0, Tuesday): 7.

| Gate | Evidence | Result |
|---|---|---|
| G1 FULL Sharpe F > R0 | 0.5572 > 0.5311 | PASS |
| G1 H1 Sharpe F > R0 | 0.3394 > 0.3277 | PASS |
| G1 H2 Sharpe F > R0 | 0.7973 > 0.7551 | PASS |
| G1 weekday-mean Sharpe | 0.5571 > 0.5270 (MON 0.556/0.523, TUE 0.557/0.531, WED 0.540/0.517, THU 0.562/0.533, FRI 0.570/0.532) | PASS |
| G2 FULL maxDD >= R0 - 0.02 | -0.2397 vs -0.2397 | PASS |
| G2 maxDD 2000-03..2003-03 | -0.1307 vs -0.1307 | PASS |
| G2 maxDD 2007-10..2009-06 | -0.0987 vs -0.1129 | PASS |
| G2 maxDD 2022-01..2022-12 | -0.0935 vs -0.0935 | PASS |
| G3 Sharpe without 2020 and 2025 | 0.5561 > 0.5411 | PASS |
| G4 bootstrap not-better < 0.10 | 0.0050 | PASS |
| G5 real gain > placebo 90th pct | +0.0262 vs +0.0361 (placebo share >= real 0.230) | FAIL |
| G6 more than half of episodes add value | 7 of 7 | PASS |
| G7 V2+rule Sharpe >= V2 and maxDD >= V2 - 0.02 | 0.7564 vs 0.7316; -0.2554 vs -0.2554 | PASS |

| Period | Rule | Sharpe | CAGR | Max DD |
|---|---|---:|---:|---:|
| FULL | R0 | 0.531 | 8.14% | -24.0% |
| FULL | R0+rule | 0.557 | 8.46% | -24.0% |
| FULL | V2 | 0.732 | 10.58% | -25.5% |
| FULL | V2+rule | 0.756 | 10.90% | -25.5% |
| H1 | R0 | 0.328 | 6.88% | -24.0% |
| H1 | R0+rule | 0.339 | 7.01% | -24.0% |
| H1 | V2 | 0.630 | 10.21% | -25.5% |
| H1 | V2+rule | 0.638 | 10.32% | -25.5% |
| H2 | R0 | 0.755 | 9.65% | -15.8% |
| H2 | R0+rule | 0.797 | 10.19% | -15.8% |
| H2 | V2 | 0.825 | 10.94% | -17.8% |
| H2 | V2+rule | 0.865 | 11.49% | -17.8% |
| P2021 | R0 | 0.844 | 12.22% | -11.5% |
| P2021 | R0+rule | 0.895 | 12.86% | -11.5% |
| P2021 | V2 | 0.844 | 13.14% | -15.9% |
| P2021 | V2+rule | 0.891 | 13.80% | -15.9% |

Rebound capture (share of the gap between the base and buy & hold that the rule closes):

| Window | Base | Base+rule | Buy & hold | Captured |
|---|---:|---:|---:|---:|
| COVID rebound 24 Mar-31 Aug 2020 (R0) | +12.4% | +14.1% | +57.8% | 4% |
| COVID rebound 24 Mar-31 Aug 2020 (V2) | +9.3% | +10.9% | +57.8% | 3% |
| Liberation Day rebound 9 Apr-30 Jun 2025 (R0) | +4.6% | +8.2% | +24.9% | 18% |
| Liberation Day rebound 9 Apr-30 Jun 2025 (V2) | +4.3% | +7.9% | +24.9% | 17% |

Episodes (R0 base, Tuesday; stress = VIX):

| Spell start | Entry | End | Reason | Stress | Peak | VIX | P&L vs R0 |
|---|---|---|---|---:|---:|---:|---:|
| 1994-03-01 | 1994-06-14 | 1994-09-06 | base invested | 11.60 | 23.87 | 11.6 | +0.91% |
| 2007-11-20 | 2009-05-19 | 2009-06-02 | base invested | 28.80 | 80.86 | 28.8 | +1.55% |
| 2015-05-05 | 2015-10-06 | 2015-11-03 | base invested | 19.40 | 40.74 | 19.4 | +2.69% |
| 2015-12-15 | 2016-03-29 | 2016-04-05 | base invested | 13.82 | 28.14 | 13.8 | +0.03% |
| 2018-10-23 | 2019-02-05 | 2019-03-05 | base invested | 15.57 | 36.07 | 15.6 | +0.68% |
| 2020-03-03 | 2020-05-26 | 2020-06-02 | base invested | 28.01 | 82.69 | 28.0 | +1.39% |
| 2025-03-04 | 2025-04-29 | 2025-06-03 | base invested | 24.17 | 52.33 | 24.2 | +3.38% |

V2 product episodes (R1 spells): 2009-05-19→2009-06-02 (base invested), 2015-10-06→2015-11-03 (base invested), 2016-03-29→2016-04-05 (base invested), 2019-02-05→2019-03-05 (base invested), 2020-05-26→2020-06-02 (base invested), 2025-04-29→2025-06-03 (base invested)

## Report only (written after the verdicts)

VIX / VIX3M at P entries from 2009-09: 2016-03-22 0.80, 2019-02-19 0.90, 2020-05-26 0.91, 2022-08-09 0.88, 2025-05-13 0.91

| Rule | Ratio | VIX cap | Floor | Sharpe rule | Sharpe R0 | Episodes | Max DD rule |
|---|---|---|---|---:|---:|---:|---:|
| P | 0.67 | 25 | 0.5 | 0.609 | 0.609 | 0 | -17.6% |
| P | 0.67 | 25 | 1.0 | 0.609 | 0.609 | 0 | -17.6% |
| P | 0.67 | 30 | 0.5 | 0.619 | 0.609 | 2 | -16.2% |
| P | 0.67 | 30 | 1.0 | 0.628 | 0.609 | 2 | -15.8% |
| P | 0.67 | 35 | 0.5 | 0.629 | 0.609 | 2 | -16.9% |
| P | 0.67 | 35 | 1.0 | 0.644 | 0.609 | 2 | -16.3% |
| P | 0.75 | 25 | 0.5 | 0.563 | 0.609 | 6 | -20.4% |
| P | 0.75 | 25 | 1.0 | 0.512 | 0.609 | 6 | -23.3% |
| P | 0.75 | 30 | 0.5 | 0.568 | 0.609 | 7 | -20.4% |
| P | 0.75 | 30 | 1.0 | 0.522 | 0.609 | 7 | -23.3% |
| P | 0.75 | 35 | 0.5 | 0.574 | 0.609 | 7 | -20.4% |
| P | 0.75 | 35 | 1.0 | 0.531 | 0.609 | 7 | -23.3% |
| P | 0.8 | 25 | 0.5 | 0.585 | 0.609 | 8 | -21.9% |
| P | 0.8 | 25 | 1.0 | 0.551 | 0.609 | 8 | -26.1% |
| P | 0.8 | 30 | 0.5 | 0.594 | 0.609 | 10 | -21.9% |
| P | 0.8 | 30 | 1.0 | 0.567 | 0.609 | 10 | -26.1% |
| P | 0.8 | 35 | 0.5 | 0.597 | 0.609 | 10 | -21.9% |
| P | 0.8 | 35 | 1.0 | 0.569 | 0.609 | 10 | -26.1% |
| F | 0.4 | 25 | 0.5 | 0.538 | 0.531 | 2 | -24.0% |
| F | 0.4 | 25 | 1.0 | 0.544 | 0.531 | 2 | -24.0% |
| F | 0.4 | 30 | 0.5 | 0.545 | 0.531 | 4 | -24.0% |
| F | 0.4 | 30 | 1.0 | 0.557 | 0.531 | 4 | -24.0% |
| F | 0.4 | 35 | 0.5 | 0.557 | 0.531 | 4 | -24.0% |
| F | 0.4 | 35 | 1.0 | 0.580 | 0.531 | 4 | -24.0% |
| F | 0.5 | 25 | 0.5 | 0.550 | 0.531 | 5 | -24.0% |
| F | 0.5 | 25 | 1.0 | 0.567 | 0.531 | 5 | -24.0% |
| F | 0.5 | 30 | 0.5 | 0.557 | 0.531 | 7 | -24.0% |
| F | 0.5 | 30 | 1.0 | 0.580 | 0.531 | 7 | -24.0% |
| F | 0.5 | 35 | 0.5 | 0.559 | 0.531 | 7 | -24.0% |
| F | 0.5 | 35 | 1.0 | 0.581 | 0.531 | 7 | -24.0% |
| F | 0.6 | 25 | 0.5 | 0.544 | 0.531 | 16 | -30.6% |
| F | 0.6 | 25 | 1.0 | 0.544 | 0.531 | 16 | -37.0% |
| F | 0.6 | 30 | 0.5 | 0.533 | 0.531 | 19 | -31.6% |
| F | 0.6 | 30 | 1.0 | 0.520 | 0.531 | 19 | -38.9% |
| F | 0.6 | 35 | 0.5 | 0.543 | 0.531 | 19 | -31.6% |
| F | 0.6 | 35 | 1.0 | 0.538 | 0.531 | 19 | -38.9% |

## Deviations

First run (same day) had a coding error: F was evaluated with the P ratio 0.75 instead of the pre-registered 0.5. Its F gate table was wrong and discarded; P was unaffected. The sensitivity table of that run already showed the correct F row (ratio 0.5, cap 30, floor 0.5) before this corrected gate run. No rule, threshold or gate changed.

## Method

As spec 6y and its implementation notes. Base R0 = run_c.base_r0; spells and P state read on decision dates; floor earns entry+2 → end+1. Placebo: 1,000 draws, default_rng(7), spells without replacement, uniform date inside the spell, price stop and base-invested end only. G7 uses V2 (run_k1.v2_weights) with R1 cash spells, compared over the P sample. Sensitivity rows use the Tuesday R0 base only. F sensitivity ratios 0.4/0.5/0.6 (P: 0.67/0.75/0.80 as specified). Inputs read-only.
