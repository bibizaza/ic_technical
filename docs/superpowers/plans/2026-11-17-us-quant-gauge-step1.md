# US Quant Gauge — Step 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the pre-registered step-1 test: does "trend + earnings revisions" (R1) beat DMAS alone (R0) on the S&P 500 under the real Delphi trading rule?

**Architecture:** A small research package `research/us_gauge/` in this repo. Data loading, signals, the Delphi trading rule, and evaluation each get their own module with unit tests. One runner script writes CSVs and a `RESULTS.md` with the gate verdict. The trend key reuses the production functions in `pipeline/momentum.py`.

**Tech Stack:** Python 3.12 (`/Users/larazanella/miniforge3/bin/python3`), pandas, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-us-quant-gauge-design.md` (approved 2026-11-17). Read it before any task; section numbers below refer to it.

## Global Constraints

- Data only from `~/Library/CloudStorage/Dropbox/Tools_In_Construction/data/market_store/` via `market_store.load(name, refresh=False)`. **Never call Bloomberg.** Free data (FRED, EODHD) may be fetched once and must be saved with `market_store.save()` and listed in `data/CATALOG.md`.
- Never print or log the EODHD API key.
- Signal at date t uses only data dated ≤ t; the trade is executed at the close of the next trading day.
- DMAS uses the 4-factor form (no MXWO): call `pipeline.momentum._compute_mars_momentum(close, date, benchmark_close=None)`; DMAS = `int((technical_float + momentum_float) / 2)`.
- Costs: 10 bp × |change in equity weight|, charged on the execution day.
- Bootstrap: 63-day blocks, 2,000 draws, `numpy.random.default_rng(7)`.
- Evaluation: first decision date in April 1993 to the last stored date. Halves: 1993-04 → 2009-12-31 and 2010-01-01 → end.
- Thresholds are fixed by the spec. **Do not change any threshold, window or rule after seeing a result.**
- Commit after each task and push (`git push`).
- Tests put the repo root on `sys.path` the same way `tests/test_har_range.py` does; tests that need Dropbox data use `pytest.mark.skipif` when the path is missing.

## Review Focus

1. **A Tuesday that is a holiday:** the decision uses the last S&P trading day on or before that Tuesday (never a later day). Tested in Task 4.
2. **A fiscal-year roll exactly at the edge of the 63-day window** (roll on t−63 or on t): the roll on t counts as "between"; a roll on t−63 does not. Tested in Task 2.
3. **Missing EPS** (FY1 starts 1990-03-12, FY2 on 1990-01-31; FY2 blanked in 2005): REV uses whichever revision is defined; REV is NaN when neither is. Tested in Task 2.
4. **The end of the data in mid-month:** the last partial month is kept, with no crash and no look-ahead. Tested in Task 4.
5. **Cash data switch on 2007-05-30** (DTB3 → BIL), and dividends after 2026-03 (last value carried forward): no gap, no NaN in daily returns. Tested in Task 1.

---

### Task 1: Inputs on one daily calendar

**Files:**
- Create: `research/__init__.py` (empty), `research/us_gauge/__init__.py` (empty), `research/us_gauge/data.py`
- Test: `tests/test_us_gauge_data.py`
- Modify (Dropbox, not git): `market_store/` gets `FRED_DTB3.csv` and `BIL.US.csv`; add both lines to `data/CATALOG.md` (US table) and a log line.

**Interfaces:**
- Produces: `load_inputs() -> pd.DataFrame` indexed by S&P trading days (the `BBG_SPX` index) with columns
  `px` (BBG_SPX close), `eq_ret` (daily total return = px pct change + dividend yield/252),
  `cash_ret` (daily cash return), `fy1`, `fy2` (FY2 set to NaN where FY2 == FY1 exactly, then both
  aligned with "last value on or before").
- Produces: `cash_daily(dtb3: pd.Series, bil: pd.Series, index: pd.DatetimeIndex) -> pd.Series`.

- [ ] **Step 1: Store the free cash data once.** DTB3 from `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3` (no key); save as `FRED_DTB3` (percent, annual, drop "."). BIL with `market_store.load("BIL.US")` (EODHD; saves itself). Check: `load("FRED_DTB3", refresh=False)` starts in 1954; `load("BIL.US", refresh=False)` starts 2007-05-30 or 2007-05-31.
- [ ] **Step 2: Write failing tests** in `tests/test_us_gauge_data.py`:
  - `test_cash_daily_switches_to_bil`: synthetic DTB3 = 5.0 every day, BIL rising 0.01% a day from 2007-05-30; assert `cash_daily` before 2007-05-30 == `0.05/252` and from 2007-05-31 on == BIL pct change; no NaN.
  - `test_fy2_copy_is_blanked`: input where fy2 == fy1 on 3 days; assert those 3 rows are NaN in the output `fy2`.
  - `test_load_inputs_real` (skip if the Dropbox path is missing): index starts 1950-01-03; no NaN in `eq_ret` or `cash_ret` after 1954-01-04; `eq_ret` after 2026-03-31 still includes a dividend part (the last Shiller value carried forward).
- [ ] **Step 3: Run** `/Users/larazanella/miniforge3/bin/python3 -m pytest tests/test_us_gauge_data.py -v` → FAIL (module missing).
- [ ] **Step 4: Implement** `research/us_gauge/data.py`. Dividend yield per day = Shiller monthly D (forward-filled to days) / px / 252. Cash: DTB3/100/252 (forward-filled) up to and including 2007-05-30, BIL daily pct change after.
- [ ] **Step 5: Run the tests** → PASS.
- [ ] **Step 6: Commit and push** `research/ tests/test_us_gauge_data.py` with message `us_gauge: inputs on one daily calendar`.

### Task 2: Earnings-revision key

**Files:**
- Create: `research/us_gauge/revisions.py`
- Test: `tests/test_us_gauge_revisions.py`

**Interfaces:**
- Consumes: `fy1`, `fy2` columns from Task 1.
- Produces:
  - `detect_rolls(fy1: pd.Series, fy2: pd.Series) -> pd.DatetimeIndex` — spec 6.1 step 4 (|Δ| > 3% in FY1 or FY2, new FY1 within 5% of previous FY2, first such day per calendar year).
  - `revisions(fy1, fy2, rolls, window=63) -> pd.Series` — REV per spec 6.1 steps 1–3. "Between t−63 and t" means a roll r with t−63 < r ≤ t (row positions).
  - `rev_zscore(rev: pd.Series, window=1260, min_obs=756) -> pd.Series` — mean and SD over the 1,260 rows **before** t (t excluded), at least 756 non-NaN; else NaN.

- [ ] **Step 1: Write failing tests:**
  - `test_one_roll_per_year_real` (skip if data missing): on real data, exactly one roll per year 1991–2026, all in February or March, and 2009 is detected.
  - `test_rev_uses_fy2_across_roll`: synthetic 200 rows; fy1 = 100 then jumps to 110 at row 150 (roll); fy2 = 110 before the roll, 120 after. At row 160, rev_FY1 = ln(110/110) = 0 (compared with old FY2 at row 97), rev_FY2 undefined → REV = 0.
  - `test_roll_on_window_edge`: roll at row t counts (FY2 comparator used); roll at row t−63 does not.
  - `test_rev_nan_when_both_missing`: fy1 and fy2 NaN at t−63 → REV NaN.
  - `test_zscore_excludes_today`: rev = 0 for 1,260 rows with small noise, then a spike of 10 at row 1,260; z at that row is large (> 5), so the spike was not in its own window.
- [ ] **Step 2: Run** `... -m pytest tests/test_us_gauge_revisions.py -v` → FAIL.
- [ ] **Step 3: Implement** `research/us_gauge/revisions.py`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit and push** with message `us_gauge: earnings-revision key`.

### Task 3: Trend key, gauge score and state

**Files:**
- Create: `research/us_gauge/gauge.py`
- Test: `tests/test_us_gauge_gauge.py`

**Interfaces:**
- Consumes: `px` from Task 1; `rev_zscore` output from Task 2.
- Produces:
  - `dmas_on(px: pd.Series, dates: pd.DatetimeIndex) -> pd.Series` (int) — for each date, `int((_compute_technical_score(px[:d], d) + _compute_mars_momentum(px[:d], d, benchmark_close=None)) / 2)`; if momentum is None, use technical (same fallback as production).
  - `gauge_frame(dmas: pd.Series, rev_z: pd.Series) -> pd.DataFrame` with columns `dmas`, `rev_z`, `T_in` (dmas ≥ 45), `R_in` (rev_z ≥ 0; NaN rev_z → False), `T_n` ((dmas − 45)/15), `G` (max(T_n, rev_z), with NaN rev_z ignored), `cash_R1` (not T_in and not R_in), `cash_R0` (not T_in), `deep` (dmas < 30), `band` (labels `"G>=1"`, `"0.5-1"`, `"0-0.5"`, `"-0.5-0"`, `"G<-0.5"`).

- [ ] **Step 1: Write failing tests:**
  - `test_dmas_matches_production_formula`: on the real `BBG_SPX` (skip if missing), `dmas_on` at 2026-11-10 equals `int((t+m)/2)` computed by calling the two production functions directly.
  - `test_dmas_no_lookahead`: `dmas_on(px, [d])` equals `dmas_on(px[:d], [d])` for 3 dates.
  - `test_gauge_rules`: rows (dmas, rev_z) → (cash_R1, cash_R0, G): (50, −1) → (False, False, 0.333…); (40, 0.2) → (False, True, 0.2); (40, −0.1) → (True, True, −0.1); (25, −2) → (True, True, −1.333…) and deep True.
  - `test_band_edges`: G = 1.0 → "G>=1"; 0.5 → "0.5-1"; 0.0 → "0-0.5"; −0.5 → "-0.5-0"; −0.51 → "G<-0.5".
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `research/us_gauge/gauge.py` (import the two private functions from `pipeline.momentum`).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit and push** with message `us_gauge: trend key, gauge score and state`.

### Task 4: Delphi trading rule

**Files:**
- Create: `research/us_gauge/trading.py`
- Test: `tests/test_us_gauge_trading.py`

**Interfaces:**
- Produces:
  - `decision_dates(trading_days: pd.DatetimeIndex, start: str) -> pd.DatetimeIndex` — for every calendar Tuesday from `start`, the last trading day on or before it (deduplicated).
  - `sleeve_weights(trading_days, dates, cash_signal: pd.Series, deep: pd.Series, cash_weight: float) -> pd.Series` — daily equity weight. Monthly: the first decision date of each calendar month sets the weight to `cash_weight` if `cash_signal` else 1.0. Other decision dates: if the current weight is 1.0 and `cash_signal` and `deep`, set it to `cash_weight`; never raise it. Each change takes effect from the **next** trading day's return (decided at close t, traded at close t+1, so it earns from t+2's return). Before the first decision date: NaN.
  - `net_returns(w: pd.Series, eq_ret: pd.Series, cash_ret: pd.Series, cost_bp=10.0) -> pd.Series` — `w*eq_ret + (1−w)*cash_ret − cost_bp/1e4 * |Δw|`, the cost charged on the day the new weight starts.

- [ ] **Step 1: Write failing tests:**
  - `test_tuesday_holiday_uses_monday`: trading days without 2024-12-24 (a Tuesday removed); decision date for that week is 2024-12-23.
  - `test_monthly_sets_state_and_early_exit_only_when_deep`: 3 months of synthetic dates; cash_signal True mid-month with deep False → weight stays 1.0 until the next month's first decision date; with deep True → weight goes to `cash_weight` after the next close.
  - `test_no_early_reentry`: weight at `cash_weight`, cash_signal turns False mid-month → weight unchanged until the next month's first decision date.
  - `test_execution_lag`: a decision at day t changes `w` first on row t+2 (decided at t, traded at close t+1).
  - `test_half_version`: `cash_weight=0.5` gives weights in {0.5, 1.0} only.
  - `test_cost_charged_once`: w goes 1.0 → 0.0 once; total cost = 10 bp.
  - `test_partial_last_month`: data ending on a Wednesday mid-month → no error; last weights defined.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `research/us_gauge/trading.py`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit and push** with message `us_gauge: Delphi monthly trading rule with early exit`.

### Task 5: Evaluation

**Files:**
- Create: `research/us_gauge/evaluate.py`
- Test: `tests/test_us_gauge_evaluate.py`

**Interfaces:**
- Produces:
  - `metrics(r: pd.Series, cash: pd.Series) -> dict` with keys `sharpe` (mean(r − cash)/sd(r − cash)·√252), `cagr`, `maxdd` (negative), `calmar` (cagr/|maxdd|), `vol`.
  - `bootstrap_not_better(r1, r0, cash, block=63, n=2000, seed=7) -> float` — paired: the same block starts for both; returns the share of draws where Sharpe(r1) ≤ Sharpe(r0).
  - `switches_per_year(w: pd.Series) -> float` — count of days where w changes / years.
  - `switch_risk_table(gf: pd.DataFrame, state_col: str) -> pd.DataFrame` — per `band`: number of decision weeks, share whose state changes within 4 weeks and within 13 weeks, and grade (`low` < 25%, `medium` 25–50%, `high` > 50% on the 4-week share).
  - `worst_drawdowns(r: pd.Series, k=5) -> pd.DataFrame` (peak, trough, recovery dates, depth).

- [ ] **Step 1: Write failing tests:**
  - `test_metrics_known_series`: constant daily excess 0.0004 with alternating ±0.01 noise → sharpe matches a hand-computed value to 1e-9; maxdd of [+10%, −50%, +0%] path = −0.5.
  - `test_bootstrap_identical_is_one`: r1 == r0 → returns 1.0 (ties count as "not better").
  - `test_bootstrap_dominant_is_zero`: r1 = r0 + 0.001 every day → returns 0.0.
  - `test_bootstrap_seed_reproducible`: two calls give the same number.
  - `test_switch_risk_grades`: synthetic states with known flip shares 0.10 / 0.40 / 0.70 → grades low / medium / high.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `research/us_gauge/evaluate.py`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit and push** with message `us_gauge: evaluation metrics, bootstrap, switch-risk table`.

### Task 6: Runner, leakage checks, results

**Files:**
- Create: `research/us_gauge/run_step1.py`, `research/us_gauge/RESULTS.md` (written by the script)
- Create (by the script): `research/us_gauge/out/step1_metrics.csv`, `step1_signals.csv`, `step1_switch_risk.csv`, `step1_drawdowns.csv`, `step1_sensitivity.csv`
- Test: `tests/test_us_gauge_leakage.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `build_signals(inputs: pd.DataFrame, rev_window=63, z_threshold=0.0) -> pd.DataFrame` (the gauge frame on decision dates, from 1990 so the z-score can warm up) and `main()`.

- [ ] **Step 1: Write the failing leakage test** `test_signals_truncation_invariant`: for 5 dates spread over 1995–2025, `build_signals(inputs)` at date d equals `build_signals(inputs[:d])` at d (all columns). Skip if data missing.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** `build_signals`. **Step 4: Run** → PASS.
- [ ] **Step 5: Implement `main()`**:
  1. Rules BH, R0, R1 × versions "all" (cash_weight 0.0) and "half" (0.5); BH has no version.
  2. Metrics for FULL, H1, H2 (H1/H2 are slices of the one FULL run, not separate runs) → `step1_metrics.csv` (rule, version, period, sharpe, cagr, maxdd, calmar, vol, switches_per_year, share_in_cash).
  3. Bootstrap R1 vs R0 on FULL for both versions.
  4. Gate (spec 6.4) on the "all" version: (1) Sharpe R1 > R0 in FULL, H1, H2; (2) maxdd(R1) ≥ maxdd(R0) − 0.02 in FULL; (3) bootstrap < 0.25; (4) switches_per_year(R1) ≤ switches_per_year(R0) in FULL. Verdict PASS only if all 4 hold.
  5. Sensitivity (information only, labelled so in the CSV and in RESULTS.md): R1 with z_threshold −0.5, and with rev_window 126.
  6. Switch-risk table for R1's gauge, worst 5 drawdowns per rule.
  7. Write `RESULTS.md`: the gate table with each condition and PASS/FAIL, the verdict, the all/half comparison, the latest gauge reading (state, G, band, grade, DMAS, REV_z) at the last decision date.
- [ ] **Step 6: Run** from the repo root: `/Users/larazanella/miniforge3/bin/python3 -m research.us_gauge.run_step1` → the 5 CSVs exist in `research/us_gauge/out/`; `RESULTS.md` has a line starting `Verdict:`.
- [ ] **Step 7: Full test run** `/Users/larazanella/miniforge3/bin/python3 -m pytest tests/ -v` → all pass (including the existing `tests/test_har_range.py`).
- [ ] **Step 8: Commit and push** `research/us_gauge tests/test_us_gauge_leakage.py` with message `us_gauge: step-1 runner and results`.
