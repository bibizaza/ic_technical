# E1 — equal weight vs cap weight

Verdict: FAIL

First decision: 2003-11-04. Common return sample: 2003-11-06 to 2026-11-16 (5,793 trading days).
H1 ends 2014-12-31; H2 starts 2015-01-01. P2021 is information only. Sub-periods slice the full run.

## Pre-registered gate

| Condition | Evidence | Result |
|---|---|---|
| FULL Sharpe E1 > SPY | 0.514376 > 0.551807 | FAIL |
| H1 Sharpe E1 > SPY | 0.458285 > 0.433990 | PASS |
| H2 Sharpe E1 > SPY | 0.579506 > 0.677195 | FAIL |
| Bootstrap not-better share < 0.25 | 0.8990 < 0.25 | FAIL |
| FULL switches/year <= 2.35 | 2.088037 <= 2.35 | PASS |

## Metrics

| Period | Rule | Sharpe | CAGR | Max DD | Switches/year |
|---|---|---:|---:|---:|---:|
| FULL | SPY | 0.551807 | 10.8320% | -55.1891% | 0.000000 |
| H1 | SPY | 0.433990 | 8.3145% | -55.1891% | 0.000000 |
| H2 | SPY | 0.677195 | 13.2519% | -33.7174% | 0.000000 |
| P2021 | SPY | 0.662306 | 13.8510% | -24.4963% | 0.000000 |
| FULL | RSP | 0.512596 | 10.4525% | -59.9228% | 0.000000 |
| H1 | RSP | 0.492327 | 10.1899% | -59.9228% | 0.000000 |
| H2 | RSP | 0.540474 | 10.7000% | -39.0413% | 0.000000 |
| P2021 | RSP | 0.556575 | 11.4548% | -21.3893% | 0.000000 |
| FULL | E1 | 0.514376 | 10.2451% | -57.3434% | 2.088037 |
| H1 | E1 | 0.458285 | 9.1026% | -57.3434% | 1.526185 |
| H2 | E1 | 0.579506 | 11.3300% | -33.7174% | 2.616209 |
| P2021 | E1 | 0.536888 | 11.3782% | -25.6508% | 2.562712 |

## Method

SPY.US_TR and RSP.US_TR; cash = FRED_DTB3/100/252. Inputs loaded with refresh=False.
At each first monthly Tuesday decision, hold RSP if the 126-trading-day change in ln(RSP/SPY) is positive, otherwise SPY. Execution t+1 close; returns from t+2.
Cost: 10 bp times the sum of absolute changes in both ETF weights (20 bp per full switch). Initial allocation has no switch cost; switch costs are charged at t+2, following the existing gauge convention. Holiday Tuesdays use the preceding trading date and its calendar month.
Sharpe uses excess returns and ddof=1. Paired moving-block bootstrap: 63 days, 2,000 draws, default_rng(7). No variants.

## Deviations

No rule, threshold, window or gate changes. Inherited cost and calendar conventions are stated above.
