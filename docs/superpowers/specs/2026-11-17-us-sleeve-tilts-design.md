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

---

# Added 2026-11-17 (before any run): volatility sizing (V1, V2) and C5 on sector ETFs (S2)

Same discipline: fixed here, no change after results, no variants. Cash = `FRED_DTB3`/100/252
before 2007-05-31 and `BIL.US_TR` after (corrected cash leg). S&P daily total return = `BBG_SPX`
change + `SHILLER_SPX_DIV`/price/252 (as in the gauge). Decide at close t, trade at close t+1,
earn from t+2. Cost 10 bp × |Δ weight|. **Turnover** = sum of |Δ weight| per year.

## Volatility sizing

**Mechanism (Moreira–Muir):** volatility clusters, but expected return does not rise with it, so
return per unit of risk is lower when volatility is high. Cut size gradually as volatility rises.

- `σ(t)` = standard deviation of the last 21 daily S&P total returns ending at t, × √252.
- **Target weight** `v(t) = min(1, 0.15 / σ(t))`.
- **When to trade:** at the first decision date (Tuesday rule) of each month, move to the target if
  |target − held| ≥ 0.10; at any other weekly decision date, only if |target − held| ≥ 0.25.

**V1 — volatility sizing alone** (equity weight = v; the rest in cash).
- Window: first monthly decision in 1955-01 → end. Halves: → 1989-12-31 and 1990-01-01 →.
  Also shown: 1993-04 → end with the gauge halves (split 2010), and P2021 (information).
- **Gate:** Sharpe(V1) > Sharpe(buy & hold) in FULL and both halves; bootstrap (63-day blocks,
  2,000 draws, `default_rng(7)`) not-better share < 25%; turnover ≤ 2.35 a year.

**V2 — R1 with volatility sizing** (equity weight = R1 sleeve weight × v, both as defined; when R1 is
in Cash the weight is 0 ("all") or 0.5 × v ("half")). The trading thresholds apply to the combined
weight; R1's own state changes always trade at their scheduled dates.
- Window: as R1 (1993-04 → end, halves split 2010-01-01), P2021 for information.
- **Gate:** Sharpe(V2) > Sharpe(R1) in FULL, H1 and H2; bootstrap vs R1 < 25%; turnover ≤ 2.35 a year.

## S2 — C5 on SPDR sector ETFs (monthly)

**Question:** does the S1 gain survive with real, tradable sector ETFs?

- **Predictions:** C5 exactly as reproduced in S1 (`research/us_gauge/run_s1.py`), made at month-end t,
  held during month t+1.
- **Mapping (Ken French → ETF):** NoDur→XLP, Durbl→XLY, Manuf→XLI, Enrgy→XLE, Chems→XLB, BusEq→XLK,
  Telcm→IYZ until 2018-05, XLC from 2018-06, Utils→XLU, Shops→XLY, Hlth→XLV, Money→XLF, Other→none.
- **Pick:** go down the C5 ranking (best first); take the first **3 distinct** ETFs that have a return
  for the month; skip Other and ETFs with no data yet.
- **Portfolio S2:** 70% SPY + 10% in each of the 3 ETFs, rebalanced monthly; cost 10 bp × turnover.
- **Benchmark:** 100% SPY. Data: `macro_health/data/regime_market/raw/bbg_tr_*.csv` (monthly total
  return; SPY from 1998-01, most sectors from 1998-12). Cash: Ken French RF.
- **Window:** first month where SPY and at least 9 mapped ETFs have returns (1999-01) → last month
  with data. Halves split at 2013-01-01. P2021 for information.
- **Gate:** Sharpe(S2) > Sharpe(SPY) in FULL and both halves; bootstrap (3-month blocks, 2,000
  draws, `default_rng(7)`) not-better share < 25%.
