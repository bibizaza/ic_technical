# S2 — C5 on sector ETFs

Verdict: PASS

C5 reproduction: n=247, IC=+0.062200957, long-short=+6.278627%; published precision reproduced.

Return months: 1999-01-31 through 2026-08-31 (332 months). Halves split 2013-01-01; P2021 is information only.

## Pre-registered gate

| Gate | Evidence | Result |
|---|---|---|
| FULL Sharpe S2 > SPY | 0.544732 > 0.498039 | PASS |
| H1 Sharpe S2 > SPY | 0.193259 > 0.111208 | PASS |
| H2 Sharpe S2 > SPY | 0.964857 > 0.943259 | PASS |
| Bootstrap not-better share < 0.25 | 0.0895 | PASS |

## Metrics

| Period | Rule | Months | Sharpe | CAGR | Max DD |
|---|---|---:|---:|---:|---:|
| FULL | SPY | 332 | 0.498039 | 8.6913% | -50.7976% |
| FULL | S2 | 332 | 0.544732 | 9.1841% | -46.7794% |
| H1 | SPY | 168 | 0.111208 | 2.8293% | -50.7976% |
| H1 | S2 | 168 | 0.193259 | 4.1474% | -46.7794% |
| H2 | SPY | 164 | 0.943259 | 15.0429% | -23.9280% |
| H2 | S2 | 164 | 0.964857 | 14.5962% | -19.1531% |
| P2021 | SPY | 68 | 0.798698 | 15.0207% | -23.9280% |
| P2021 | S2 | 68 | 0.904239 | 16.0770% | -17.1840% |

## ETF holding frequency

| ETF | Months held | Share of months |
|---|---:|---:|
| SPY | 332 | 100.00% |
| IYZ | 64 | 19.28% |
| XLB | 1 | 0.30% |
| XLC | 8 | 2.41% |
| XLE | 204 | 61.45% |
| XLF | 50 | 15.06% |
| XLI | 0 | 0.00% |
| XLK | 35 | 10.54% |
| XLP | 225 | 67.77% |
| XLU | 217 | 65.36% |
| XLV | 65 | 19.58% |
| XLY | 127 | 38.25% |

## Method

Reused run_s1.reproduce exactly: cached macro_health panel, C5 final=False, expanding industry regressions, lag=1 and the original vintage. Predictions at month-end t are held in t+1. Windows and telecom mapping are indexed by holding/return month.
Cached bbg_tr_<TICKER>.csv month-end levels are normalized to calendar month-end, then converted to returns without filling missing months. Start requires SPY and nine distinct mapped ETFs with returns. Pick the first three distinct available ETFs down the C5 ranking; skip Other and missing returns. Durbl/Shops both map to XLY; Telcm maps to IYZ through May 2018 and XLC from June 2018. XLC is skipped until it has a computable monthly return.
S2 holds 70% SPY and 10% in each selected ETF. Monthly cost is 10 bp times absolute target-weight turnover, matching S1. Both portfolios pay 10 bp initial allocation cost. Cash for Sharpe is outcome-month Ken French RF. Subperiods slice the full run without restart.
Sharpe uses ddof=1 and sqrt(12). Paired bootstrap: 3-month blocks, 2,000 draws, default_rng(7). This is not independent C5 evidence.

## Deviations

No rule, threshold, window or gate changes; no variants. The final common return month is 2026-08-31, limited by cached C5 predictions and Ken French RF as well as ETF returns (ETF cache ends 2026-09-30). All inputs were read-only; no downloads or external writes.
