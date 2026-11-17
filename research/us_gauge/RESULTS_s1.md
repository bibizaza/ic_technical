# S1 — C5 sector tilt

## Published-result reproduction

| Measure | Published | Reproduced |
|---|---:|---:|
| Months | 247 | 247 |
| Beta-adjusted 1-month IC | +0.062 | +0.062200957 |
| Long-short annual return | +6.3% | +6.278627% |

Reused macro_health/tools/regime_market: cached panel.parquet (the published sectors.main/data.load path) and raw/ff12_vw.csv; no downloads or external writes.
C5 uses c5_trends(final=False), expanding industry regressions with lag=1 and minimum 60 observations. Beta uses 60 months, minimum 36. Long-short uses top/bottom 3 and 10 bp times turnover. Test decision window: 2006-01-31 through 2026-09-30, exactly evaluate.WINDOWS; outcomes are next-month returns.

Verdict: PASS

## Pre-registered gate

| Condition | Evidence | Result |
|---|---|---|
| FULL Sharpe S1 > market | 0.601195 > 0.552539 | PASS |
| 1973–2005 Sharpe S1 > market | 0.528517 > 0.479312 | PASS |
| 2006–2026 Sharpe S1 > market | 0.704303 > 0.656405 | PASS |
| Bootstrap not-better share < 0.25 | 0.0225 < 0.25 | PASS |

## Metrics

| Period | Rule | Months | Sharpe | CAGR | Max DD |
|---|---|---:|---:|---:|---:|
| FULL | market | 606 | 0.552539 | 12.1055% | -50.3117% |
| FULL | S1 | 606 | 0.601195 | 12.6951% | -47.1948% |
| 1973–2005 | market | 359 | 0.479312 | 12.7328% | -44.9947% |
| 1973–2005 | S1 | 359 | 0.528517 | 13.3710% | -40.8065% |
| 2006–2026 | market | 247 | 0.656405 | 11.2000% | -50.3117% |
| 2006–2026 | S1 | 247 | 0.704303 | 11.7199% | -47.1948% |

## Method

Common decision-date sample: 1976-02-29 through 2026-07-31 (606 months); each row earns the following month’s return, as in the published sector test. Sub-periods slice the full run.
S1 holds 70% Ken French market and 10% in each of the top three C5 industries. Benchmark holds 100% market. Cash is Ken French RF for the outcome month. All returns are total returns.
Monthly cost is 10 bp times the sum of absolute target-weight changes, matching the published sector turnover convention. Initial allocation is charged to both portfolios at the first available prediction (1976-02). No restart at window boundaries.
Sharpe uses excess returns, ddof=1 and sqrt(12); CAGR annualizes at 12 months. Paired moving-block bootstrap: 3 months, 2,000 draws, default_rng(7).
This is not independent evidence of C5. Ken French industries do not map one-to-one to SPDR sector ETFs; implementation mapping is deferred.

## Deviations

No rule, threshold, window or gate changes; no variants. The nominal 1973–2005 window has 359 usable months from 1976-02 because C5 needs feature and regression warm-up; the test window has 247 usable months through 2026-07, matching the published sample. The cached panel is reused exactly as in the published runner, alongside cached raw industry returns. The local tools package search path is extended to import macro_health’s namespace package; existing-directory mkdir probes are allowed but directory creation, external writes and network access are blocked.
