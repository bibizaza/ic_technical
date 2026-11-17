# US sleeve tilts — pre-registration (2026-11-17, before any run)

Two tests that **stay invested** and change what the US equity sleeve holds. Goal (user, 2026-11-17):
best return per unit of risk (Sharpe of returns in excess of cash). Same discipline as the US gauge
(`2026-11-16-us-quant-gauge-design.md`): rules fixed here; no change after results; no variants.
Cash for excess returns: `FRED_DTB3`/100/252 per day (daily tests) or Ken French RF (monthly test).

## E1 — equal weight vs cap weight (daily data)

**Mechanism:** leadership between the mega-caps and the average stock comes in long waves
(concentration builds for years, then reverses). Relative momentum follows the wave.

- **Data:** `SPY.US_TR` (cap weight), `RSP.US_TR` (equal weight), total return, from 2003-04-30.
- **Signal:** at the first decision date (Tuesday rule, as in the gauge) of each month,
  `rel = ln(RSP_TR/SPY_TR)` today minus 126 trading days ago. **Hold RSP if rel > 0, else SPY.**
- **Mechanics:** decide at close t, trade at close t+1, earn from t+2; cost 10 bp × (|Δw_RSP| + |Δw_SPY|)
  (= 20 bp per full switch).
- **Window:** first monthly decision with 126 days of history (Nov 2003) → end. Halves split at
  2015-01-01. P2021 (2021-01-01 → end) for information.
- **Gate:** Sharpe(E1) > Sharpe(SPY buy & hold) in FULL, H1 and H2; paired block bootstrap
  (63-day blocks, 2,000 draws, `default_rng(7)`) not-better share < 25%; switches/year ≤ 2.35.
- Also reported: RSP buy & hold.

## S1 — macro sector tilt with C5 (monthly data, Ken French 12 industries)

**Mechanism:** industries respond differently to the direction of growth and inflation; C5 (12-month
change in the macro_health growth and inflation composites) ranks them. C5 passed a pre-registered
long-short test (`macro_health/results/regime_sectors_test.md`) in 1973–2005 and 2006–2026.
**This is not independent evidence:** it asks whether a *long-only* tilt keeps part of that gain.

- **Predictions:** reuse `macro_health/tools/regime_market` exactly as in the passed test (C5 features,
  expanding regression per industry, the same lag and data vintage). **Check first:** reproduce the
  published C5 beta-adjusted 1-month IC (+0.062) and long-short (+6.3% a year) for 2006–2026; if it does
  not reproduce, stop.
- **Portfolio S1:** each month, **70% market + 30% equal weight in the top 3 C5 industries**
  (market = Ken French Mkt). Rebalanced monthly; cost 10 bp × turnover.
- **Benchmark:** 100% market.
- **Windows:** 1973–2005 and 2006–2026 (the original windows), and FULL.
- **Gate:** Sharpe(S1) > Sharpe(market) in FULL and in both windows; paired block bootstrap on monthly
  returns (3-month blocks, 2,000 draws, `default_rng(7)`) not-better share < 25%.
- **Implementation note (not tested here):** Delphi would use SPDR sector ETFs; Ken French industries
  do not map one-to-one. A mapping is a later step, only if S1 passes.
