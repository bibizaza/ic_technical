# M1 — macro-only sector rank

Verdict: M1a FAIL · M1b PASS · overall FAIL

Checks: C5 reproduces (IC +0.062, LS +6.3%); intercept + macro = C5 (max diff 6.9e-18).

## Gate

| Gate | Evidence | Result |
|---|---|---|
| M1a dev IC p < 0.10 | IC +0.0500, p 0.0020 | PASS |
| M1a dev alpha p < 0.10 | alpha +4.23%/yr, p 0.0170 | PASS |
| M1a test IC p < 0.10 | IC +0.0059, p 0.3959 | FAIL |
| M1a test alpha p < 0.10 | alpha +5.62%/yr, p 0.0271 | PASS |
| M1b FULL Sharpe > SPY | 0.5373 > 0.4980 | PASS |
| M1b H1 Sharpe > SPY | 0.1577 > 0.1112 | PASS |
| M1b H2 Sharpe > SPY | 0.9798 > 0.9433 | PASS |
| M1b bootstrap not-better < 0.25 | 0.1100 | PASS |

## Signal test, Ken French 12 (M1a vs published C5)

| Window | Score | IC | IC p | Long-short/yr | Alpha/yr | Alpha p |
|---|---|---:|---:|---:|---:|---:|
| dev | C5 | +0.052 | 0.005 | +4.5% | +2.4% | 0.077 |
| dev | M1 | +0.050 | 0.002 | +4.4% | +4.2% | 0.017 |
| test | C5 | +0.062 | 0.003 | +6.3% | +6.3% | 0.007 |
| test | M1 | +0.006 | 0.396 | +6.2% | +5.6% | 0.027 |

## SPDR ETFs, monthly (M1b; S2 and FIXED XLP/XLU/XLE for information)

| Period | Rule | Months | Sharpe | CAGR | Max DD |
|---|---|---:|---:|---:|---:|
| FULL | SPY | 332 | 0.498 | 8.69% | -50.8% |
| FULL | M1b | 332 | 0.537 | 9.29% | -49.4% |
| FULL | S2 | 332 | 0.545 | 9.18% | -46.8% |
| FULL | FIXED | 332 | 0.538 | 8.78% | -46.6% |
| H1 | SPY | 168 | 0.111 | 2.83% | -50.8% |
| H1 | M1b | 168 | 0.158 | 3.58% | -49.4% |
| H1 | S2 | 168 | 0.193 | 4.15% | -46.8% |
| H1 | FIXED | 168 | 0.198 | 4.16% | -46.6% |
| H2 | SPY | 164 | 0.943 | 15.04% | -23.9% |
| H2 | M1b | 164 | 0.980 | 15.47% | -18.3% |
| H2 | S2 | 164 | 0.965 | 14.60% | -19.2% |
| H2 | FIXED | 164 | 0.914 | 13.73% | -21.7% |
| P2021 | SPY | 68 | 0.799 | 15.02% | -23.9% |
| P2021 | M1b | 68 | 0.962 | 17.63% | -16.8% |
| P2021 | S2 | 68 | 0.904 | 16.08% | -17.2% |
| P2021 | FIXED | 68 | 0.903 | 15.31% | -16.6% |

## Share of months picked

| ETF | M1 | C5 (S2) |
|---|---:|---:|
| IYZ | 20% | 19% |
| XLB | 3% | 0% |
| XLC | 11% | 2% |
| XLE | 50% | 61% |
| XLF | 25% | 15% |
| XLI | 4% | 0% |
| XLK | 40% | 11% |
| XLP | 38% | 68% |
| XLU | 28% | 65% |
| XLV | 27% | 20% |
| XLY | 54% | 38% |

## K1 with M1 picks (information only; daily, 1999-02 → 2026-08-31)

| Period | Sharpe K1-M1 | Sharpe K1 | Sharpe BH | CAGR K1-M1 | CAGR K1 | Max DD K1-M1 | Max DD K1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FULL | 0.648 | 0.671 | 0.423 | 8.85% | 8.86% | -23.5% | -24.5% |
| H1 | 0.392 | 0.414 | 0.120 | 6.21% | 6.38% | -23.5% | -24.5% |
| H2 | 0.918 | 0.948 | 0.824 | 11.60% | 11.45% | -12.0% | -11.3% |
| P2021 | 1.010 | 0.987 | 0.738 | 14.98% | 13.94% | -12.0% | -11.3% |

## Method

M1 = C5 regression unchanged (expanding OLS per industry on g12, i12; lag 1; cached vintage), score = slope terms only. M1a uses macro_health sectors.evaluate unchanged. M1b reuses run_s2.tilt_returns unchanged. FIXED ranks NoDur > Utils > Enrgy every month C5 exists, same machinery. K1-M1 reuses run_k1.v2_weights and combine unchanged. Inputs read-only. Test no. 13 on this history.
