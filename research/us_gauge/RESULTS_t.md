# T: fast stress de-risking batch (pre-registered, spec 6T, one run)

**T1: FAIL. T2: FAIL. T3: FAIL.** Consequence (declared before the run): PASS → proposed to the user; FAIL → rejected list, no change.

Base = product spec 11d, inputs to 2026-11-13. 10 bp, cash = T-bill. T2 window starts 2004-09-28 (first day both canaries are defined). Code: `run_t.py`.

## T1 tranched month lock: **FAIL**

| Gate | Evidence | Result |
|---|---|---|
| G1 Sharpe better, full and both halves | full 0.718 vs 0.713; 1999-02 → 2012-12 0.479 vs 0.517; after 2012-12 0.964 vs 0.913 | FAIL |
| G2 worst fall not deeper by > 2 pts | -22.8% vs -20.1% | FAIL |
| G3 bootstrap not-better share < 25% | 45.9% | FAIL |

| 1999-02-01 → 2026-11-13 | Product | Product + T1 |
|---|---:|---:|
| Sharpe | 0.71 | 0.72 |
| Return a year | 9.9% | 9.9% |
| Worst fall | -20.1% | -22.8% |
| Trades a year | 9.4 | 11.9 |

## T2 bond / EM canary cap: **FAIL**

| Gate | Evidence | Result |
|---|---|---|
| G1 Sharpe better, full and both halves | full 0.831 vs 0.871; 2004-09 → 2015-06 0.745 vs 0.831; after 2015-06 0.933 vs 0.911 | FAIL |
| G2 worst fall not deeper by > 2 pts | -16.3% vs -17.7% | PASS |
| G3 bootstrap not-better share < 25% | 73.2% | FAIL |
| G4 placebo p <= 0.033 | p = 0.033 (placebo median Sharpe 0.653) | PASS |

| 2004-09-28 → 2026-11-13 | Product | Product + T2 |
|---|---:|---:|
| Sharpe | 0.87 | 0.83 |
| Return a year | 11.8% | 9.5% |
| Worst fall | -17.7% | -16.3% |
| Trades a year | 9.1 | 29.6 |
| Share of days capped (below 100%) | | 47% |

## T3 credit-confirmed stress cap: **FAIL**

| Gate | Evidence | Result |
|---|---|---|
| G1 Sharpe better, full and both halves | full 0.710 vs 0.713; 1999-02 → 2012-12 0.535 vs 0.517; after 2012-12 0.891 vs 0.913 | FAIL |
| G2 worst fall not deeper by > 2 pts | -19.7% vs -20.1% | PASS |
| G3 bootstrap not-better share < 25% | 50.6% | FAIL |
| G4 placebo p <= 0.033 | p = 0.162 (placebo median Sharpe 0.686) | FAIL |

| 1999-02-01 → 2026-11-13 | Product | Product + T3 |
|---|---:|---:|
| Sharpe | 0.71 | 0.71 |
| Return a year | 9.9% | 9.8% |
| Worst fall | -20.1% | -19.7% |
| Trades a year | 9.4 | 10.6 |
| Share of days capped (below 100%) | | 10% |

## Information only: stress episodes (total return / worst fall inside the window)

| Episode | S&P | Product | T1 | T2 | T3 |
|---|---:|---:|---:|---:|---:|
| Dot-com bear | -47.4% / -47.4% | -8.1% / -10.9% | -10.8% / -13.3% | -8.1% / -10.9% | -7.6% / -10.5% |
| 2008 crisis | -54.9% / -55.3% | -3.7% / -6.6% | -4.3% / -6.6% | -3.7% / -6.6% | -2.4% / -5.3% |
| 2011 debt ceiling | -18.5% / -18.7% | -15.1% / -15.3% | -15.1% / -15.3% | -16.1% / -16.3% | -15.1% / -15.3% |
| 2015–16 | -12.9% / -13.0% | -5.3% / -9.3% | +0.4% / -2.9% | +1.0% / -3.2% | -3.8% / -7.8% |
| Q4 2018 | -18.7% / -19.4% | -6.1% / -8.7% | -6.2% / -8.7% | -2.6% / -3.1% | -6.5% / -8.7% |
| COVID fall | -33.5% / -33.8% | -8.5% / -11.5% | -8.5% / -11.5% | -6.0% / -8.7% | -8.7% / -11.3% |
| 2022 | -18.2% / -24.5% | -13.9% / -17.7% | -11.7% / -15.9% | -4.4% / -5.3% | -14.0% / -17.8% |
| Liberation Day fall | -18.6% / -18.8% | -5.4% / -6.0% | -8.0% / -8.5% | -5.4% / -6.0% | -5.4% / -6.0% |
| Nov 2026 to date | -0.2% / -4.4% | -0.8% / -4.4% | -0.8% / -4.4% | +0.0% / -2.2% | -0.8% / -4.4% |

Caps on the last input day (2026-11-13): T2 = 0%, T3 = 100%.
