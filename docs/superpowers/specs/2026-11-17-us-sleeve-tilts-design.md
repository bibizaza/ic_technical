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

---

# Added 2026-11-17 (before any run): the full US combo (K1) — last US test

**Question:** do the three layers that passed add up when used together?
Timing = R1 (trend + earnings). Size = volatility sizing (as in V2). Content = C5 sector tilt (as in S2).
After K1, no more US candidates are tested (US closed; Europe next).

**K1, daily:**
- **Equity weight** `w(t)` = the V2-all daily weight, exactly as in `run_vol.py` (R1 state × v, same
  trading thresholds, same t+2 convention). The rest is in cash (DTB3 → BIL.US_TR, as corrected).
- **Inside the equity part:** 70% SPY + 10% in each of the 3 S2 ETFs for that month (same C5
  predictions, mapping and pick rule as `run_s2.py`). The month-t+1 picks start at the close of the
  first trading day of month t+1 (earn from the next day). Within a month the 4 holdings drift
  (buy and hold); they are reset to 70/10/10/10 at each monthly reset.
- **Returns:** daily total return from `SPY.US_TR` and `<ETF>.US_TR` (EODHD adjusted).
- **Costs:** 10 bp × |Δ equity weight| + 10 bp × equity weight × (sum of |Δ weights| inside the
  equity part at each monthly reset).
- **Window:** 1999-02-01 → the last day of the last month with a C5 pick (2026-08-31). Halves split
  2013-01-01 (as S2). P2021 for information.

**Compared with (same window, same daily data):**
- **V2-SPY:** the same equity weight `w(t)`, but 100% SPY inside the equity part (timing + size only).
- **BH:** 100% SPY.

**Gate:**
1. Sharpe(K1) > Sharpe(V2-SPY) in FULL, H1 and H2;
2. Sharpe(K1) > Sharpe(BH) in FULL;
3. paired block bootstrap K1 vs V2-SPY (63-day blocks, 2,000 draws, `default_rng(7)`) not-better
   share < 25%;
4. equity-weight turnover ≤ 2.35 a year.
Also reported: max drawdown, CAGR, and the sector-reset turnover.

**K1 data rule (added 2026-11-17 after the first K1 run stopped on input validation, before any K1
metric was computed):** the sector ETF series miss 2 S&P trading days in 1999–2026 (XLI 1999-01-26,
IYZ 2000-06-28). On a missing day the ETF price is carried forward from the last observation
(return 0 that day; the price move is booked on the next day with a price). No other change.
