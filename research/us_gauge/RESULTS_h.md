# H: HTEI re-entry (pre-registered, spec 6h, one run)

**Verdict: FAIL**

Sample 2010-02-08 → 2026-11-13. HTEI US score < 35 while the keys are in cash. Episode exits as B. Size = live vol target. Product = spec 11b (daily keys, 5-pt sizes, trade next day if gap >= 10 pts). 10 bp costs. HTEI history: backtest to 2026-04-12, live after, so mostly in-sample. Code: `run_h.py`.

| Gate | Evidence | Result |
|---|---|---|
| G1 Sharpe product+H > product | 0.838 vs 0.885 | FAIL |
| G2 worst fall not worse by > 2 pts | -17.5% vs -17.7% | PASS |
| G3 placebo p <= 0.10 | p = 0.567 (placebo median 0.846) | FAIL |
| G4 more than half of episodes beat cash | 14 of 28 | FAIL |
| G5 both halves not worse | 2010-2017: 1.058 vs 0.924; 2018-2026: 0.660 vs 0.855 | FAIL |

| Since 2010 | Product | Product + H |
|---|---:|---:|
| Sharpe | 0.89 | 0.84 |
| Return a year | 11.5% | 11.3% |
| Worst fall | -17.7% | -17.5% |
| Days with a different weight | | 293 |

## Episodes

| Entry | End | Reason | HTEI at entry | Days | S&P 500 | T-bill |
|---|---|---|---:|---:|---:|---:|
| 2011-11-21 | 2011-12-05 | base invested | 32.8 | 9 | +6.0% | +0.0% |
| 2014-10-15 | 2014-11-05 | base invested | 28.7 | 15 | +9.2% | +0.0% |
| 2015-08-20 | 2015-08-21 | stop | 32.9 | 1 | -3.9% | +0.0% |
| 2015-08-24 | 2015-08-25 | stop | 22.4 | 1 | +3.9% | +0.0% |
| 2015-08-26 | 2015-11-04 | base invested | 25.8 | 49 | +6.1% | -0.0% |
| 2016-01-14 | 2016-01-15 | stop | 26.3 | 1 | +0.1% | +0.0% |
| 2016-01-19 | 2016-01-20 | stop | 24.1 | 1 | +0.5% | +0.0% |
| 2016-01-21 | 2016-02-08 | stop | 23.2 | 12 | -2.8% | +0.0% |
| 2016-02-09 | 2016-02-10 | stop | 24.8 | 1 | -1.2% | +0.0% |
| 2016-02-11 | 2016-04-05 | base invested | 24.3 | 36 | +11.2% | +0.0% |
| 2018-12-06 | 2018-12-14 | stop | 27.0 | 6 | -3.3% | +0.1% |
| 2018-12-17 | 2018-12-19 | stop | 26.0 | 2 | -3.1% | +0.0% |
| 2018-12-20 | 2018-12-21 | stop | 24.4 | 1 | -2.7% | +0.0% |
| 2018-12-24 | 2019-03-05 | base invested | 23.5 | 47 | +12.7% | +0.4% |
| 2020-03-03 | 2020-03-09 | stop | 25.1 | 4 | -7.9% | +0.0% |
| 2020-03-10 | 2020-03-11 | stop | 23.0 | 1 | -9.5% | +0.0% |
| 2020-03-12 | 2020-03-16 | stop | 20.6 | 2 | -6.7% | +0.0% |
| 2020-03-17 | 2020-03-20 | stop | 19.5 | 3 | -6.7% | -0.0% |
| 2020-03-23 | 2020-06-03 | base invested | 19.1 | 50 | +27.7% | -0.0% |
| 2022-02-22 | 2022-02-23 | stop | 31.6 | 1 | +1.5% | +0.0% |
| 2022-02-24 | 2022-03-03 | base invested | 30.6 | 5 | -1.2% | +0.0% |
| 2023-03-15 | 2023-04-05 | base invested | 33.8 | 15 | +3.8% | +0.2% |
| 2025-03-05 | 2025-03-06 | stop | 34.3 | 1 | +0.6% | +0.0% |
| 2025-03-07 | 2025-03-10 | stop | 32.8 | 1 | -0.8% | +0.0% |
| 2025-03-11 | 2025-03-13 | stop | 29.8 | 2 | +0.7% | +0.0% |
| 2025-03-14 | 2025-04-03 | stop | 32.9 | 14 | -10.5% | +0.2% |
| 2025-04-04 | 2025-04-07 | stop | 24.2 | 1 | -1.6% | +0.0% |
| 2025-04-08 | 2025-06-04 | base invested | 24.5 | 39 | +9.1% | +0.6% |

Cash days with HTEI score available: 637. Of those, HTEI < 35: 126.
