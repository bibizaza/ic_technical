# K1 — full US combination

Verdict: PASS

Sample: 1999-02-01 through 2026-08-31 (6938 trading days). Halves split 2013-01-01; P2021 is information only.

C5 reproduction: n=247, IC=+0.062200957, long-short=+6.278627%.

## Gate table

| Gate | Evidence | Result |
|---|---|---|
| FULL Sharpe K1 > V2-SPY | 0.670785 > 0.636660 | PASS |
| H1 Sharpe K1 > V2-SPY | 0.414366 > 0.353954 | PASS |
| H2 Sharpe K1 > V2-SPY | 0.947618 > 0.933061 | PASS |
| FULL Sharpe K1 > BH | 0.670785 > 0.422907 | PASS |
| Bootstrap not-better share < 0.25 | 0.1615 | PASS |
| Equity turnover/year <= 2.35 | 1.657923 | PASS |

## Metrics

| Period | Rule | Sharpe | CAGR | Max DD | Vol | Equity turnover/yr | Sector-reset turnover/yr |
|---|---|---:|---:|---:|---:|---:|---:|
| FULL | K1 | 0.670785 | 8.8618% | -24.4774% | 10.5869% | 1.657923 | 0.688681 |
| FULL | V2-SPY | 0.636660 | 8.8314% | -25.4371% | 11.2188% | 1.657923 | 0.000000 |
| FULL | BH | 0.422907 | 8.5999% | -55.1891% | 19.2145% | 0.000000 | 0.000000 |
| H1 | K1 | 0.414366 | 6.3770% | -24.4774% | 10.8648% | 1.316591 | 0.694900 |
| H1 | V2-SPY | 0.353954 | 5.8084% | -25.4371% | 11.3621% | 1.316591 | 0.000000 |
| H1 | BH | 0.120138 | 2.6025% | -55.1891% | 21.3292% | 0.000000 | 0.000000 |
| H2 | K1 | 0.947618 | 11.4540% | -11.3304% | 10.2955% | 2.005812 | 0.682342 |
| H2 | V2-SPY | 0.933061 | 12.0014% | -15.8149% | 11.0695% | 2.005812 | 0.000000 |
| H2 | BH | 0.823634 | 15.0733% | -33.7174% | 16.7836% | 0.000000 | 0.000000 |
| P2021 | K1 | 0.986833 | 13.9402% | -11.3304% | 10.6486% | 1.428877 | 0.726179 |
| P2021 | V2-SPY | 0.912969 | 13.9038% | -15.8149% | 11.5887% | 1.428877 | 0.000000 |
| P2021 | BH | 0.737633 | 15.0951% | -24.4963% | 16.7259% | 0.000000 | 0.000000 |

## Latest reading

As of sample end 2026-08-31: effective R1 state = Invested; v = 1.000000; effective equity weight w = 0.932791. v is the current volatility target; w includes the scheduled trading thresholds and lag.
Last holding month: 2026-08; ETFs: XLE, XLP, XLU.

## Method

V2-all uses the unchanged run_vol functions and identical construction; V2-SPY shares the exact same daily w. S2 picks come directly from unchanged run_s2.tilt_returns, including its C5 reproduction, mapping, distinct-ETF selection and availability rule. Daily returns use offline EODHD adjusted total-return levels; sizing retains the preregistered S&P input. Cash retains DTB3 then BIL.
Each monthly reset executes at the first trading-day close and earns from the following day. Holdings drift daily and reset to 70/10/10/10. January 1999 warms up the sleeve; all windows slice the continuous simulation. Costs are 0.001 times absolute equity-weight change plus 0.001 times w times internal reset turnover. Reported sector turnover is w times internal reset turnover, in total-portfolio units. Annual turnover divides by trading days/252.
Sharpe uses daily excess returns, ddof=1 and sqrt(252); volatility uses total returns. Paired bootstrap uses 63-day blocks, 2,000 draws, default_rng(7). No variants; US candidate testing is closed.

## Deviations

Data rule added in spec commit 3c369ba before any K1 metrics: carry the last observed ETF price on XLI 1999-01-26 and IYZ 2000-06-28; return is zero on the missing day and the full price move is booked on the next observed day. Applied only to K1 SPY/sector price inputs; any unregistered post-inception price gap stops the run, and missing required returns, cash or equity weights still stop. No other rule, threshold, window or gate changes. Integration fix: extend the already loaded market_store src package search path with macro_health/src so unchanged C5 imports resolve; no external files are modified. Inherited cost accounting debits equity changes and monthly resets on the first return day with the new weights (the day after execution close). January warm-up absorbs initial sleeve allocation before the evaluation window. Latest reading is dated to the common sample end, rather than mixing later timing data with stale sector picks. All market-store loads use refresh=False; external inputs are read-only, with network and external writes guarded. No commits or pushes.
