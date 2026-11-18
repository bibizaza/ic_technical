# US quant gauge — design

Date: 2026-11-16. Status: draft for approval.

## 0. Correction log

- **2026-11-17 — cash leg data error.** Steps 1, 1b, 1c and 2 used `BIL.US` (EODHD unadjusted close)
  as the cash return from 2007-05-31. BIL pays its yield as monthly dividends, so its price is flat:
  cash earned ~0% instead of the T-bill rate (e.g. ~5% in 2024). The pre-registered intent was BIL's
  total return. **Worse:** BIL's 2017-11-30 share split shows as a +100% one-day "cash return" in the
  price series, so every excess-return series had a −100% day: all Sharpe values and bootstraps so far
  are distorted (e.g. buy & hold 0.29). **Fix:** use `BIL.US_TR` (adjusted close, split- and
  dividend-adjusted). No rule, threshold, window or gate changes.
  All four steps are re-run and their verdicts re-read with the corrected cash leg; the old results
  stay in git history.

## 1. Intent

Replace the pure technical DMAS rating with a **quantitative "Invested / Cash" gauge** per market,
starting with the **US (S&P 500)**.

- The gauge **moves real money**. It drives the US equity sleeve of **Delphi** (about 13% of the
  portfolio in May 2026: `500 FP`, `D5BM GY`). Delphi rebalances monthly. The IC meets weekly.
- What the IC cares about is **Invested or Cash** and **how close we are to the switch** — not
  Bullish vs Constructive.
- Each market has its own drivers. In developed markets, earnings and earnings revisions drive
  returns. The design adds drivers **one at a time**, and each one must prove itself, because more
  inputs and shorter histories raise the risk of overfitting.

## 2. Goals, non-goals, anti-goals

**Goals**
- A US gauge with a clear state (Invested / Cash), a score, the distance to the switch, and a
  switch-risk grade (low / medium / high).
- Validated on the **real Delphi trading rule** (monthly rebalance + early exit), not on a weekly
  paper rule.
- Each added input passes a gate written **before** its test is run.

**Non-goals (for now)**
- Europe, EM, Japan (next, same method, once the US is done).
- Building the nutshell slide. The slide comes after a gauge passes.
- Changing production DMAS or the IC deck.
- Rebuilding HTEI before 2010 (only 5 of its 7 inputs exist before 2010; positioning starts 2009).
- Shorting or leverage. The user never shorts.

**Anti-goals (a failure even if it "works")**
- Thresholds or windows chosen after looking at results.
- A gauge the IC cannot explain in one sentence.
- A gauge that flips often enough to make Delphi trade every month.

## 3. Constraints

- **Data:** only from the stores listed in
  `~/Library/CloudStorage/Dropbox/Tools_In_Construction/data/CATALOG.md`. Never call Bloomberg.
  Missing Bloomberg data: ask the user for a manual export (exact formulas + inbox path).
  Free data (FRED, CFTC) may be downloaded, and must be stored in `market_store/` and added to the catalog.
- **Heavy runs go to Codex** (codex:codex-rescue agent) to save Claude tokens.
- **Python:** `/Users/larazanella/miniforge3/bin/python3`.

## 4. Approaches considered

| | Approach | Trade-off | Decision |
|---|---|---|---|
| A | **Two keys:** trend (DMAS) + earnings revisions. Cash only when both say out. | 2 inputs, low overfit risk. Revisions are often late at tops. | **Step 1** |
| B | **Pillar composite:** trend, revisions, macro, credit, each z-scored, simple average. | Closer to "own drivers", but many choices and data only from 1990. | Reached **gradually**: pillars are added one step at a time |
| C | Machine learning on the pillars | Already lost to DMAS in the 2026-11 walk-forward test (Sharpe 0.46/0.43 vs 0.57) | Dropped |

**Picked by the user (2026-11-16): step-by-step.** Start with A. Add pillars one at a time (B), each behind a gate.
HTEI is shown next to the gauge as context; it is not an input (history from 2010 only).

## 5. Data

| Item | Store name (`market_store.load(name, refresh=False)`) | From |
|---|---|---|
| S&P 500 price | `BBG_SPX` | 1950-01-03 |
| Forward EPS, current FY | `BBG_SPX_EPS_FY1` | 1990-03-12 |
| Forward EPS, next FY | `BBG_SPX_EPS_FY2` | 1990-01-31 |
| Forward EPS, blended 12M | `BBG_SPX_EPS_BF12` | 1990-01-31 (stored, not used in step 1) |
| S&P dividends (monthly, 12m total) | `SHILLER_SPX_DIV` | 1871 → 2026-03 |
| 3-month T-bill | **to add:** FRED `DTB3` → `FRED_DTB3` | 1954 |
| BIL ETF (cash leg after 2007-05-30) | **to add:** EODHD `BIL.US` | 2007 |
| CFTC Traders in Financial Futures, E-mini S&P 500 | **to add (step 2):** `CFTC_TFF_ES_*` | 2006-06 |

**EPS quirk:** FY1 and FY2 roll to the new fiscal year in **February–March**, after results are
reported, not on 1 January. On the roll day the new FY1 ≈ the old FY2. From 2005-02-15 to
2005-04-08 the stored FY2 is a copy of FY1 (Bloomberg glitch); see 6.1 step 4.

## 6. Step 1 — trend + earnings revisions (pre-registered)

### 6.1 Signals (computed on each Tuesday close; act at the next close)

**Key T (trend):** DMAS rebuilt from `BBG_SPX` with `pipeline/momentum.py` (`compute_scores`),
in its **4-factor form for the whole history** (no MXWO; momentum weights 50 / 22.2 / 22.2 / 5.6).
The live gauge computes T the same way, so the tested T and the live T are identical. *(Claude's
call: the gauge replaces DMAS on the slide, so it does not need the production 5-factor version;
one definition over the whole history matters more. Production DMAS agreed with this version about
97% of weeks in 2008–26.)* **T is "in" when DMAS ≥ 45.**

**Calendar:** the EPS series are dated on every weekday. Align them to S&P trading days (last value
on or before each S&P date).

**Key R (earnings revisions):**
1. `rev_FY1(t) = ln( FY1(t) / C1 )`, where `C1 = FY1(t−63)` if no roll happened between t−63 and t,
   else `FY2(t−63)` (the same fiscal year, before the roll). 63 = trading days ≈ 3 months.
2. `rev_FY2(t) = ln( FY2(t) / FY2(t−63) )` only if no roll happened in between; else undefined.
3. `REV(t)` = mean of the defined values among rev_FY1 and rev_FY2.
4. A roll is a day where FY1 **or** FY2 changes by more than 3% (up or down) in one day **and** the new
   FY1 is within 5% of the previous day's FY2. Only the **first** such day in each calendar year counts.
   (Checked on the data: exactly one roll in every year 1991–2026, all in February–March. 2009 is caught
   only through the FY2 jump, because 2008 and 2009 estimates were almost equal.)
   **Data cleaning first:** on days where FY2 = FY1 exactly, FY2 is set to missing (Bloomberg glitch
   2005-02-15 → 2005-04-08, where FY2 was a copy of FY1).
5. `REV_z(t)` = z-score of REV(t) against REV over the trailing **1,260 trading days** (5 years),
   past values only, minimum 756 days (3 years). Before that: undefined.
6. **R is "in" when REV_z ≥ 0.**
   (A z-score is used because analysts cut estimates through most years. A raw "REV ≥ 0" rule would say
   "out" most of the time.)

**Gauge state:** **Cash only when T is out AND R is out.** Otherwise Invested.

**Gauge score and distance to the switch (pre-registered, for display; not part of the gate):**
- Put both keys on one scale where 0 = the key's switch point: `T_n = (DMAS − 45) / 15`
  (15 DMAS points ≈ one old rating band), `R_n = REV_z`.
- **Gauge score `G = max(T_n, R_n)`.** G ≥ 0 ⇔ Invested (at least one key is in). G is the distance
  to the switch: when Invested, how far the stronger key is above its switch point; when in Cash,
  how far the key closest to flipping back is below it.
- **Bands on G:** ≥ 1 · 0.5 to 1 · 0 to 0.5 · −0.5 to 0 · < −0.5.
- **Switch-risk grade per band**, from the share of weeks that changed state within 4 weeks over the
  evaluation period: below 25% = low, 25–50% = medium, above 50% = high.

### 6.2 Delphi trading rule

- **Monthly rebalance:** at the first Tuesday of each month, the sleeve takes the gauge state.
- **Early exit (during the month):** at any other Tuesday, if the sleeve is invested, the gauge says
  Cash, **and DMAS < 30** (deep: on the S&P since 1955, only 19% of weeks under 30 flipped back
  within 4 weeks), move to cash at the next close.
- **No early re-entry.** Coming back in waits for the next monthly rebalance. *(Claude's call: the
  user only asked for an early exit. Revisit if the results show long waits after bottoms.)*
- **Two exposure versions, both reported:** "all" (Cash = 0% equity) and "half" (Cash = 50% equity).
- **Equity return:** S&P price + dividends (`SHILLER_SPX_DIV` / price / 252 per day; after 2026-03
  the last value is carried forward).
- **Cash return:** BIL from 2007-05-30, DTB3 before.
- **Costs:** 10 bp × |change in equity weight|.

### 6.3 Rules compared (all with the same trading rule and both exposure versions)

| Code | Rule |
|---|---|
| BH | Buy and hold S&P total return |
| R0 | DMAS alone (Cash when DMAS < 45) — **the baseline to beat** |
| R1 | Two keys (6.1) — **the candidate** |

No other variants are tested in step 1. Sensitivity runs (REV_z threshold −0.5, window 126 days)
are reported **for information only** and are never used to pick the rule.

### 6.4 Test period and pass gate

- **Evaluation:** first Tuesday of April 1993 (EPS start + 3-year z-score warm-up) to the last stored date.
- **Halves:** 1993-04 → 2009-12 and 2010-01 → end. (2000–02 and 2008 in the first; 2020 and 2022 in the second.)
- **Sharpe** = mean daily return over cash / SD × √252. **Calmar** = CAGR / |max drawdown|.
- **R1 passes step 1 only if, for the "all" version:**
  1. Sharpe(R1) > Sharpe(R0) in FULL **and** in both halves;
  2. max drawdown(R1) is not worse than R0 by more than 2 percentage points in FULL;
  3. paired block bootstrap (63-day blocks, 2,000 draws, `numpy.random.default_rng(7)`):
     the share of draws where Sharpe(R1) ≤ Sharpe(R0) is **below 25%**;
  4. **turnover:** switches per year of the sleeve (R1) are not higher than R0's in FULL
     (this puts the "no monthly trading" anti-goal into the gate).
- The "half" version is reported next to it. **The user decides all vs half** from the two results.
- **Leakage checks (must pass before any result is read):** every signal at date t uses data ≤ t;
  the trade happens at t+1; the z-score window never includes t+1 or later.

### 6.5 Also reported (to build the slide later)

- Number of switches per year, and time spent in Cash, for R0 and R1.
- The **switch-risk table**: for each gauge band, the share of weeks that flipped state within 4 and 13 weeks.
- Worst 5 drawdowns of each rule, with dates.

## 6b. Step 1b — re-entry rule and earnings alone (pre-registered 2026-11-17, before any run)

**Why:** step 1 (R1) failed only the drawdown gate (−35% vs −24%): in mid-2000 and mid-2002 the trend
had broken but revisions were still positive (analysts cut late), so R1 stayed invested. Revisions
looked useful for coming back in after a fall. This idea comes **from** the step-1 result, so the US
test alone cannot confirm it: **R2 is adopted only if it passes on the US and then on Europe
(STOXX 600) with the same rules, unchanged.**

Same data, signals (DMAS, REV_z), trading mechanics (decide at close t, trade at close t+1, weight
earns from t+2, 10 bp), period, halves, metrics, bootstrap and versions (all / half) as step 1.

### R2 — trend exits, earnings re-entry (state machine on weekly decision dates)

State: `pos` ∈ {Invested, Cash}, `mode` ∈ {trend, earnings}, `armed` ∈ {True, False}.
"Monthly" = the first decision date of a calendar month. REV_z NaN never arms and counts as "not ≥ 0".

1. **Start** (first monthly date): Invested if DMAS ≥ 45, else Cash. mode = trend.
   armed = (Cash and REV_z < 0).
2. **Invested, mode trend** (= R0): monthly date and DMAS < 45 → Cash; other date and DMAS < 30 → Cash.
3. **Invested, mode earnings** (= R1 logic): if DMAS ≥ 45 → mode = trend (no trade).
   Else, with "out" = (DMAS < 45 and REV_z < 0): monthly date and out → Cash; other date and out and
   DMAS < 30 → Cash.
4. **On every move to Cash:** mode = trend; armed = (REV_z < 0).
5. **Cash:** first, if REV_z < 0 → armed = True. Then:
   - monthly date and DMAS ≥ 45 → Invested, mode trend (normal re-entry);
   - else, on **any** decision date, armed and REV_z ≥ 0 → Invested, mode earnings, armed = False
     (**early re-entry**: revisions were negative during this cash spell and have now recovered).

Effect: before a first exit, earnings cannot keep the sleeve invested (this is the step-1 failure);
after an exit, earnings can bring it back early, once they have been cut and recover.

### R3 — earnings alone (information + candidate)

Cash at a monthly date when REV_z < 0 (NaN = invested). No early exit, no early re-entry.
Tested because R1 suggested revisions carry most of the value. Gate vs R0 as below; also reported
vs BH in FULL, H1, H2.

### Gate (unchanged from 6.4, "all" version, each candidate vs R0)

Sharpe > R0 in FULL, H1, H2; FULL maxDD ≥ R0 maxDD − 0.02; bootstrap not-better share < 0.25;
FULL switches/year ≤ R0. **Multiple testing:** three candidates have now been tried on the US
(R1, R2, R3). A US PASS is a candidate for the Europe check, not a final result.

### Europe check (after the US run, only for candidates that pass)

STOXX 600 (`SXXP Index`) price + BEst EPS FY1, FY2, BF12 (manual Bloomberg export, same settings as
the S&P file) and a EUR cash rate (free, FRED 3-month euro interbank, monthly). Same rules, no change.

## 6c. Step 1c — R1 + VIX, Sharpe goal (pre-registered 2026-11-17, before any run)

**Goal set by the user (2026-11-17): the best return per unit of risk → Sharpe is the main measure.**
From this step on, maximum drawdown is reported but is **not** a gate condition. This applies
going forward only: R1, R2 and R3 keep their earlier verdicts.

**Why VIX:** R1 lost most in 2000–02, when the trend was broken but revisions were still positive.
VIX was often high in those years. **Honesty note:** before fixing the level, Claude looked at VIX
alone in 1999–2003 (21-day average ≥ 25 about 32–51% of the time per year; 17% of the time over
1993–2026). No rule result was seen. 25 is a common round "stress" level.

**Data:** `FRED_VIXCLS` (daily VIX close from 1990-01-02), aligned to S&P days (last value on or before).

### R4 — trend + earnings + VIX

- `V_high(t)` = 21-trading-day average of VIX close, ending at t, is ≥ 25.
- **Cash when T is out AND (R is out OR V_high).** Otherwise Invested.
  (= R1, plus: a broken trend with high VIX goes to cash even when revisions are positive.)
- Same Delphi mechanics as R1: monthly state; early exit at other dates when the gauge says Cash and
  DMAS < 30; no early re-entry; t+1 execution, t+2 return; 10 bp; all / half.

### Gate (Sharpe goal, "all" version)

1. Sharpe(R4) > Sharpe(R1) in FULL, H1 and H2 (R1 = the best rule so far);
2. Sharpe(R4) > Sharpe(BH) in FULL;
3. paired block bootstrap R4 vs R1 (63-day blocks, 2,000 draws, `default_rng(7)`): share of draws with
   Sharpe(R4) ≤ Sharpe(R1) below 25%;
4. switches per year of R4 ≤ R0's (2.35): no more trading than DMAS alone.

**Also reported (information only, not gated):** every rule (BH, R0, R1, R2, R3, R4) on the period
**2021-01-01 → end** (a sub-period of the same run), and max drawdown everywhere. Four candidates have
now been tried on the US; any PASS still needs the unchanged Europe check.

## 7. Step 2 — CoT positioning (shape only; pre-registered in full before it runs)

- **Data:** CFTC *Traders in Financial Futures*, E-mini S&P 500, **leveraged funds** net position as
  a % of open interest, weekly, from 2006-06. Free; store in `market_store/`.
- **Direction: follow, not contrarian.** HTEI found hedge-fund positioning in ES has a *positive*
  link to later returns (t = +2.77), so it was rejected there as a contrarian input.
- **Timing:** positions are as of Tuesday and published on Friday. A Tuesday decision uses the
  positions of the previous Tuesday.
- **Signal:** z-score vs trailing 156 weeks (3 years). "In" when z ≥ 0.
- **How it joins:** to be fixed in its own pre-registration (for example a third key with a 2-of-3
  vote). It must beat the step-1 winner with the same kind of gate.
- **Caveat:** the test runs from about 2009, so there are few bear markets. Power is low.

## 7b. Step 2 — CoT positioning, full pre-registration (2026-11-17, before any run)

Nothing CoT-related has been looked at on this data except HTEI's earlier finding (hedge-fund ES
positioning has a *positive* link to later returns, t = +2.77). Gate uses the Sharpe goal (6c).

**Data** (`market_store`, see CATALOG): `CFTC_ES_LEV_LONG`, `CFTC_ES_LEV_SHORT`, `CFTC_ES_OI`
(E-mini S&P 500, leveraged funds, by report date), `CFTC_ES_RELEASE_LAG_DAYS`.

**Point in time:** a report is usable from its release date = report date + lag days. A decision at t
uses the **latest report whose release date ≤ t**.

**Signal C:**
1. `net(r) = (LEV_LONG − LEV_SHORT) / OI` for each report r.
2. `C_z` = z-score of net against the previous **156 reports** (3 years; current report excluded),
   minimum 104 reports. Before that: undefined.
3. **C is "in" when C_z ≥ 0** (follow the hedge funds). Undefined → not decided (see period).

**R5 — 2-of-3 vote:** keys T (DMAS ≥ 45), R (REV_z ≥ 0), C (C_z ≥ 0).
**Cash when at least 2 of the 3 keys are out.** Same Delphi mechanics as R1 (monthly state, early exit
when Cash and DMAS < 30, no early re-entry, t+1 execution, t+2 return, 10 bp, all / half).

**Window (CoT history is short):** first decision date in **July 2008** (first month where C_z is
defined for every decision) → end. Halves: 2008-07 → 2017-06 and 2017-07 → end. R1, R0 and BH are
re-measured on the **same window** (slices of their full runs). P2021 (2021-01-01 → end) shown for
information.

**Gate ("all" version):**
1. Sharpe(R5) > Sharpe(R1) on the window and in both halves;
2. Sharpe(R5) > Sharpe(BH) on the window;
3. paired block bootstrap R5 vs R1 on the window (63-day blocks, 2,000 draws, `default_rng(7)`):
   not-better share < 25%;
4. switches per year R5 ≤ R0 on the window.

**Caveat:** ~18 years, few bear markets (2008 tail, 2011, 2015–16, 2018, 2020, 2022): low power.
Five candidates have now been tried on the US; any PASS still needs a Europe check (EUR-relevant CoT:
Euro Stoxx 50 futures, if available).

## 8. Later steps

Credit spreads (IG/HY OAS, from 1989/1994) and macro (recession model, regime; macro_health) as
further pillars, one at a time, with the same gate. Then Europe with the same method.

## 9. Left to the builder

- Code layout. Suggested: `research/us_gauge/` in this repo, one script per step, results as CSV
  next to it, plus a short `RESULTS.md`.
- How to store the new free data (`FRED_DTB3`, `BIL.US`, CFTC), as long as it goes into
  `market_store/` and `CATALOG.md`.
- Output tables and charts.

## 10. Claude's calls (please check)

1. REV defined on FY1 and FY2 with the roll mapping, 3-month window (6.1).
2. REV z-score window of 5 years, threshold 0.
3. "Deep" for the early exit = DMAS < 30.
4. No early re-entry.
5. Pass gate numbers: 2 pp drawdown tolerance, bootstrap below 25%, turnover not above DMAS alone.
6. Evaluation from April 1993, halves split at 2010.
7. Trend key T uses the 4-factor DMAS (no MXWO) for the whole history and live.
8. Gauge score G = max((DMAS − 45)/15, REV_z), with fixed bands and switch-risk cut-offs (25% / 50%).

---

## 6x. Capitulation floor "C" (pre-registered 2026-11-17, before any run; US test #13)

Designed by Fable 5.2 (blind: no returns computed, no events counted), at the user's request to
"position for a rebound" after an extreme, fast fall. Origin: `RESULTS_stress.md` (2020, 2025
rebounds missed). The US test cannot confirm it; adoption needs US PASS and then Europe PASS, same
code, one run each. No variant, threshold or gate changes. (Numbering note: M1 in the sleeve-tilts
spec was also called #13; C is the 14th candidate on this history, the fallback the 15th.)

Definitions (daily closes P = `BBG_SPX`, price index):
 r_t = ln(P_t/P_{t-1}); s_t = std(r, ddof=1) over the 252 days ending t-21;
 Z_t = ln(P_t/P_{t-21}) / (s_t * sqrt(21)).
State: Idle / Armed / On, evaluated at every trading-day close, independent of the base rule state.
 Idle -> Armed: Z_t <= -3. L = P_t.
 Armed: L = min(L, P_t). If P_t > P_{t-5} and P_t > L -> On, e = t, L frozen.
        If 21 trading days pass since the last day with Z <= -3 -> Idle.
 On: target weight = max(base target weight, 0.50). Floor is not multiplied by V2 sizing.
     End -> Idle when P_t < L (stop) or t = e + 63 trading days. Re-arming needs Z_t <= -3 on a later day.
Mechanics: decide at close t, trade at close t+1, earn from t+2, 10 bp per unit weight change.
 Override changes are forced trades (bypass V2 10%/25% bands). Base rule state machine runs unchanged.

Primary test: R0+C vs R0, S&P 500 1955-2026, T-bill cash. H1 to 1990-12-31, H2 from 1991-01-01.
Secondary (report only): R1+C vs R1 and V2+C vs V2, 1993-2026. Also report at 25 bp cost.
Gates (all must hold, primary test):
 G1 Sharpe > base in FULL, H1, H2, and on the mean over base-rule decision weekdays Mon-Fri.
 G2 maxDD >= base maxDD - 0.02 in FULL, and within 2000-03-01..2003-03-31 and 2007-10-01..2009-06-30.
 G3 FULL Sharpe > base after removing 2020-02-19..2020-08-31 and 2025-02-19..2025-06-30.
 G4 Paired moving-block bootstrap (63 days, 2,000 draws, default_rng(7)): share(C not better) < 0.10.
 G5 Placebo: 1,000 draws (default_rng(7)); each real episode's entry moved to a uniformly random trading day
    with base weight < 0.50, L = lowest close of the prior 21 days, same stop and 63-day limit.
    Real Sharpe gain > 90th percentile of placebo gains.
 G6 More than half of On episodes have positive cumulative return difference vs base.
Report (no gate): per-episode table (arm, entry, end, reason, P&L vs base); Z -2.5/-3.5 and floor 25%/100%
 shown only after the gate verdict is written.
Fallback (test #14): identical with floor 0.25, run only if G1, G3-G6 pass and G2 fails.
Europe (one run): STOXX 600, EUR cash, R0+C vs R0 from first available date; R1+C from 2005-04 report only.
 PASS = FULL Sharpe > base, maxDD >= base - 0.02, bootstrap share < 0.25, placebo > 90th percentile.
Multiple testing: 13th US candidate (Bonferroni p ~0.004 not attainable); a US PASS is "qualifies for Europe",
 never "significant". Design was informed by knowledge of the 1987, 2008 and 2020 paths.

**Implementation notes (fixed before any run, Claude):**
- Equity returns are total return (`eq_ret` = price change + Shiller dividend/252); the signal uses
  the price index only. Cash = DTB3 then BIL total return (as corrected).
- R0 base = existing `sleeve_weights` with cash = DMAS < 45, deep = DMAS < 30, all-to-cash.
  Primary decisions: Tuesday rule (`decision_dates`) from 1955-01-01; returns from the first
  effective base weight. G1 weekday mean: the same base on Mon, Tue, Wed, Thu, Fri decision days
  (last trading day on or before that weekday), C unchanged; mean of the 5 Sharpes, C vs base.
- Combined weight on each return day = max(base weight, floor), where the floor is 0.50 on the
  days from e+2 through end+1 (end = the close where the stop or time limit hits).
- Armed: on each day, L is updated first, then entry is checked, then the 21-day lapse. The arming
  day cannot be an entry (P_t = L). In Idle, arming is checked from the day after an episode ends.
- Lapse counter: trading days since the last day with Z <= -3; lapse when it reaches 21.
- G2 window drawdowns restart wealth at the window start. G3 drops the listed days, then computes
  Sharpe. Placebo L = minimum of the 21 closes ending at the placebo entry day; placebo windows
  that overlap are merged (max floor). G6 episode P&L = compounded combined return / compounded
  base return - 1, over the episode's effective days.

---

## 6y. "Panic is over" floor P (pre-registered 2026-11-17, before any run and before opening LF98OAS/LP01OAS/V2X)

Designed by Fable 5.2 (blind: nothing run, nothing counted, new Bloomberg series not opened) at the
user's request ("set it up"), after C failed. Story for clients: "we come back when panic ends,
measured by credit and fear gauges, not by guessing headlines".

Hypothesis: re-enter when stress recedes, not when price is low (C failed: RESULTS_c.md).
Inputs: OAS = LF98OAS Index daily close, lagged 1 trading day; VIX = FRED_VIXCLS same-day close; P = S&P close.
Decision dates: the base rule's weekly dates (first of month sets base state, as now).
Cash spell: consecutive decision dates with base state = Cash; d0 = first of them.
 OASpk_t = max OAS from (d0 - 21 trading days) to t.  L_t = lowest P close from d0 to t.
Entry (decision date t, base Cash, no entry or stop yet in this spell):
 OAS_t <= 0.75 * OASpk_t AND VIX_t < 30  -> On; freeze L = L_t.
On: sleeve weight = max(base weight, 0.50); the 0.50 is not multiplied by V2 sizing; forced trade
 (bypasses V2 bands).
End: (a) base state becomes Invested -> Off, base/V2 weight applies;
     (b) stop on a decision date: OAS_t > OASpk at entry, or P_t < L -> Off, and no re-entry in this spell.
Mechanics: decide close t, trade close t+1, earn from t+2, 10 bp. Base state machine unchanged.

Fallback F (run and reported separately, 1990+): entry VIX_t <= 0.5 * VIXpk_t AND VIX_t < 30,
 VIXpk defined like OASpk; stop: VIX_t > VIXpk at entry, or P_t < L. Everything else identical.
 Adoption order: P if P passes; F only if P fails and F passes.

Primary test: R0+P vs R0, S&P 500, from first daily LF98OAS date to 2026 (no monthly splicing).
 H1 to 2009-12-31, H2 from 2010-01-01. Also at 25 bp (report).
Gates (all):
 G1 Sharpe > base in FULL, H1, H2, and on the Mon-Fri decision-weekday mean.
 G2 maxDD >= base - 0.02 in FULL and within 2000-03-01..2003-03-31, 2007-10-01..2009-06-30,
    2022-01-01..2022-12-31.
 G3 FULL Sharpe > base after removing 2020-02-19..2020-08-31 and 2025-02-19..2025-06-30.
 G4 Paired moving-block bootstrap (63 days, 2,000 draws, default_rng(7)): share(not better) < 0.10.
 G5 Placebo, 1,000 draws (default_rng(7)): each real entry moved to a uniformly random base-Cash decision
    date (max one per spell, same weight, same price stop, same end). Real Sharpe gain > 90th percentile.
 G6 More than half of On episodes have positive cumulative return difference vs base.
 G7 Product check: V2+P Sharpe >= V2 and maxDD >= V2 - 0.02, 1994-2026.
Report (no gate): episode table (spell start, entry, end, reason, OAS, VIX, P&L vs base); share of the
 2020 and 2025 rebounds captured; VIX vs VIX3M at entries from 2009-09; thresholds 0.67/0.80 and VIX 25/35
 and floor 1.00 shown only after the verdict.
Europe (one run, after a US pass): STOXX 600, EUR cash, R0 base, OAS = LP01OAS, VIX = V2X, thresholds
 0.75 and 30 unchanged, from the first date with both series.
 PASS = FULL and weekday-mean Sharpe > base, maxDD >= base - 0.02, placebo > 90th percentile.
Honesty: design informed by memory of 2002, 2008-09, 2020, 2022, 2025 stress paths. Prior odds: US pass
 about 1 in 3, US + Europe about 1 in 6. Expected capture: second half of V rebounds only.

**Implementation notes (fixed before any data is opened, Claude):**
- R0 base = `run_c.base_r0` (DMAS < 45 cash, DMAS < 30 early exit, all-to-cash), Tuesday rule for
  the primary; G1 weekday mean uses Mon..Fri bases exactly as in C. DMAS from `out/dmas_daily_1955.csv`.
- "OAS lagged 1 trading day": OAS_t = the last LF98OAS close on or before the S&P trading day before t.
  Days with no OAS value carry the last value. The sample starts at the first decision date with
  OAS available for the whole lookback (d0 - 21 days).
- Spells and states are read on decision dates only. A spell that is already open at sample start
  begins at the sample start. The floor earns from entry+2 through end+1, as in C.
- Placebo "same end": the placebo On ends when the base turns Invested or at the price stop
  (P_t < L with L = lowest close from d0 to the placebo entry). No OAS stop in the placebo. Spells
  with a real entry and spells without one are both eligible; one placebo entry per real entry,
  spells drawn without replacement, then a uniform decision date inside the spell.
- G7: V2 = `run_k1.v2_weights` (R1 states). Spells for V2+P = R1 Cash spells on its Tuesday dates;
  weight = max(V2 weight, 0.50) while On.
- Fallback F uses the same code with VIX in place of OAS (same-day, no lag), sample from 1990-01.

---

## 6z. Faster re-entry "X" (pre-registered 2026-11-17, before any run)

Idea 1 of `research/us_gauge/IDEAS_fable_2026-11-17.md`, requested by the user. Mechanism: the base rule
leaves early on a deep break (DMAS < 30 on any Tuesday) but comes back only on the first decision of
a month. Rebounds are front-loaded, so this calendar delay costs money. The F placebo (random
half re-entries inside cash spells did about as well as F) points the same way.

Rule X (changes only the base state machine `trading.sleeve_weights`):
- First decision of a month: unchanged (Cash if the rule's cash signal holds, else Invested).
- Other decision dates: unchanged early exit (Invested and cash signal and DMAS < 30 -> Cash), and
  **new: Cash and DMAS >= 60 -> Invested** (60 = 45 + 15, the mirror of the exit at 30 = 45 - 15).
- Applies to R0 (cash signal DMAS < 45) and to R1 (cash signal DMAS < 45 and REV_z < 0).
- Weight on re-entry: 1 (base). V2 sizing then applies as usual (state changes are forced trades).
- Mechanics unchanged: decide close t, trade t+1, earn t+2, 10 bp.

Primary test: R0+X vs R0, S&P 500, Tuesday decisions from 1955-01-01 to 2026-11-13, T-bill/BIL cash.
H1 to 1990-12-31, H2 from 1991-01-01. Also at 25 bp (report).
Gates (all):
 G1 Sharpe > base in FULL, H1, H2, and on the mean over Mon..Fri decision weekdays.
 G2 maxDD >= base - 0.02 in FULL and within 2000-03-01..2003-03-31, 2007-10-01..2009-06-30, 2022-01-01..2022-12-31.
 G3 FULL Sharpe > base after removing 2020-02-19..2020-08-31 and 2025-02-19..2025-06-30.
 G4 Paired moving-block bootstrap (63 days, 2,000 draws, default_rng(7)): share(not better) < 0.10.
 G5 Placebo, 1,000 draws (default_rng(7)): each real early re-entry is moved to a uniformly random
    non-first-of-month decision date in the same cash spell, from the spell start up to (not
    including) the base rule's own re-entry date; everything after follows the X state machine.
    Real Sharpe gain > 90th percentile of placebo gains.
 G6 More than half of early re-entries have positive cumulative return difference vs base over the
    days until the two rules hold the same state again.
Product (report, and a gate G7): R1+X vs R1 and V2(R1+X) vs V2, 1993-04 to 2026-11. G7: V2 with X
 Sharpe >= V2 and maxDD >= V2 - 0.02.
Report (no gate): early re-entry table; K1 / K1 + F (C5 and macro-only) with X, by period and in the
 stress windows of `RESULTS_combo_f.md`; F re-judged on top of X (does V2+X+F beat V2+X?).
 Thresholds 55 and 65 shown only after the verdict.
Europe (one run, after a US pass): STOXX 600 price index + dividend proxy as available, EUR cash, R0+X vs
 R0 from the first DMAS date. PASS = FULL and weekday-mean Sharpe > base, maxDD >= base - 0.02.

---

## 6s. Shock re-entry "S" (pre-registered 2026-11-17, before any run)

Origin: Fable's V-shape event study (`research/us_gauge/VSHAPE_fable_2026-11-17.md`). On 41 S&P falls
since 1955, falls of 15%+ that hit their low within 63 trading days of a 1-year high ("shock from a
high") rebounded (5 V / 1 failed, +11.7% in 3 months), while fast falls inside a broken trend failed.
**S was found by looking at the US data, so the US run is NOT evidence. Only the Europe run (STOXX
600, never studied) decides.** The user asked for both runs.

Definitions (daily closes P of the price index):
 h_t = date of the highest close in the 252 trading days ending t; H = P(h_t).
 L_t = lowest close from h_t to t; l_t its date.
 Trigger on close t when all hold: L_t <= 0.85 * H; l_t is at most 63 trading days after h_t;
 P_t >= 1.10 * L_t; and S is not already On.
 Re-trigger limit: after an episode for a given h ends, a new episode for the same h needs a lower L
 (a new low within 63 days of h). A new h (new 1-year high) resets this.
On: the equity state is Invested whatever the base rule says.
 - R0 / R1 (binary base): weight = max(base weight, 1.0).
 - V2 product: weight = max(V2 weight, v(t)), with v(t) = min(1, 0.15/sigma21) at the trigger close
   (S overrides the timing, not the size). S trades are forced trades (bypass V2 bands). F, if
   present, keeps its own floor: weight = max(V2, F floor, S weight).
End (checked every close): base state Invested (base takes over), close below L (stop), or 126
 trading days after the trigger.
Mechanics: decide close t, trade close t+1, earn t+2; 10 bp per unit weight change.

US run (information only, not evidence): R0+S vs R0, S&P 500, Tuesday base, 1955-01 → 2026-11, the
same gates as 6x/6z (G1 FULL/H1 to 1990/H2/weekday mean; G2 FULL and 2000-03..2003-03,
2007-10..2009-06, 2022 drawdowns within 2 pp; G3 without 2020-02-19..08-31 and 2025-02-19..06-30;
G4 bootstrap < 0.10; G5 placebo: each episode moved to a random day with base weight 0, L = lowest
close of the prior 21 days, same stop and 126-day limit, 1,000 draws, default_rng(7), real gain >
90th pct; G6 more than half of episodes positive vs base). Product report: V2+F+S vs V2+F, and
K1+F+S macro-only, by period and stress window.

**Europe run (the test, one run):** STOXX 600 (`BBG_SXXP`), R0 base built exactly as in the US
(DMAS from `pipeline/momentum.py` on SXXP closes, cash below 45, early exit below 30, Tuesday rule),
decisions from 1988-01-01. Equity return = SXXP price change + 3.0%/252 a day (dividend proxy; SXXP
is a price index). Cash = EUR 3-month interbank rate (FRED IR3TIB01DEM156N Germany to 1998-12,
IR3TIB01EZM156N euro area from 1999-01; monthly, % a year /100/252, carried daily). If FRED is not
reachable, the same rates from the ECB/Bundesbank sources; the switch is recorded.
Europe PASS = all of: FULL Sharpe R0+S > R0; weekday-mean Sharpe > R0; FULL maxDD >= R0 - 0.02;
placebo real gain > 90th pct; more than half of episodes positive. Also reported: price-only
returns (no dividend proxy), halves split 2007-01-01, episode table.

## 6sl. S-live: shock re-entry sized by the live volatility target (pre-registered 2026-11-17, before any run)

Origin: Fable's V-gap study (`research/us_gauge/VGAP_fable_2026-11-17.md`, part I). In the product,
S sets its weight once, at v(t) of the trigger close (16% in March 2020), and keeps it frozen for
the whole episode. Europe validated S at full weight (R0 base), not at that size. S-live keeps
every S rule of 6s (trigger, episodes, stop, 126-day end, re-trigger limit) and only changes the
weight while S is On:
 - S-frozen (today): S weight = v(trigger close), constant over the episode.
 - S-live (candidate): S weight on earn day t+2 = v(t), the same daily target the V2 sizing uses,
   v(t) = min(1, 0.15/sigma21(t)) (`run_vol.target_weight`). No bands; every change pays 10 bp.
 - S-full (report only, a risk-budget choice for the user): S weight = 1.
 Product weight = max(base weight, F floor if present, S weight). Base and episodes are the same
 in all three; only the S weight differs.
The US numbers (3 episodes: 1998, 2020, 2025) were seen in the V-gap study: the US run is
information only.

**Europe run (the test, one run).** STOXX 600 (`BBG_SXXP`), same inputs as the 6s Europe run:
equity = SXXP price change + 3.0%/252; cash = ECB Euribor 3M monthly (stored as `ECB_EURIBOR3M`),
sample from 1994-02-01 (the 6s run sample, so results are comparable); DMAS from
`out/dmas_sxxp_daily.csv`. Base = R0 timing on SXXP (Tuesday, cash below 45, early exit below 30)
x volatility size, exactly as V2 does it (`run_fast.v2_from_states` on the R0 states, bands 0.10 at
month start / 0.25 otherwise). No F (not built for Europe). S episodes from `run_s.shock_episodes`
on SXXP closes with the binary R0 weight (the 6s episodes).
Europe PASS = all of:
 1. FULL Sharpe base+S-live > base+S-frozen;
 2. weekday-mean Sharpe (Monday..Friday decision days) base+S-live > base+S-frozen;
 3. FULL maxDD base+S-live >= base+S-frozen - 0.02.
Also reported (no gate): base without S, base+S-full, halves split 2007-01-01, price-only
returns, per-episode P&L of live vs frozen, and the US product by period and stress window
(V2+F+S-frozen, V2+F+S-live, V2+F+S-full, K1+F+S-live macro-only).

## 6t. Breadth thrust re-entry "B" (pre-registered 2026-11-17, before any run)

Draft by Fable 5.2, 2026-11-17. To be copied into the gauge spec and committed **before** any run.
Code: `research/us_gauge/run_b.py` (guarded: needs `--one-run`). Tests: `tests/test_us_gauge_breadth.py`
(synthetic data and the thrust-date check only; no return of the rule has been computed).

### Origin and what has already been seen

- Source of the rule: Walter Deemer, "Breakaway Momentum 101" (walterdeemer.com/bam.htm, read in full).
  His table lists 25 signals since 1945. He states the concept dates from the 1970s and that the
  signals after 1973 were noted in real time. The page has been public since 2002.
- Seen so far: the thrust **dates** on the free breadth series (they reproduce Deemer's table,
  16 of 16 between 1965 and 2019, each within 2 trading days). I also know the S&P 500 history.
  Not seen: any return, Sharpe, episode count or P&L of rule B.
- The rule is 40+ years old and widely watched. Expect decay (see "Hurdle").

### Data

- NYSE advancing issues A_t and declining issues D_t, daily, all NYSE issues.
  - 1965-03-01 → 2020-02-10: `UNICORN_NYSE_ADV`, `UNICORN_NYSE_DEC` (Unicorn Research Corporation free
    files, taken from the Internet Archive copy of unicorn.us.com/advdec, snapshot 2026-01-10).
  - From 2020-02-11: Bloomberg, downloaded by hand, stored as `BBG_NYSE_ADV`, `BBG_NYSE_DEC`.
    Download from 2015-01-01 so that five years overlap. The run reports the overlap difference.
    Bloomberg values are used only after 2020-02-10.
  - Days with A <= 0 or D <= 0 are dropped (6 days in the free file, all zero rows).
- S&P 500, cash and DMAS: as in 6s (`load_inputs`, `out/dmas_daily_1955.csv`).
- Europe: no STOXX 600 breadth series exists in the store, so there is no Europe holdout for B.
  The US gate set below is the test. This is weaker than S; say so in any write-up.

### Definitions (zero free choices)

- Ratio: R_t = (sum of A over the 10 most recent valid breadth days ending t) /
  (sum of D over the same 10 days).
- **Thrust day:** R_t > 1.97 and R on the previous valid day <= 1.97 (an upward crossing).
- Trigger at close t when: t is a thrust day; the base rule's equity weight in force on t is below 1
  (base in Cash); B is not already On; and at least 63 closes exist.
- L = lowest S&P 500 close of the 63 closes ending t.
- **On:**
  - binary base (R0, R1): weight = max(base weight, 1.0);
  - product (V2+F+S): weight = max(product weight, v(t)), v(t) = min(1, 0.15 / sigma21(t)), updated
    every day (decided t, in force t+2), forced trades (no V2 bands).
- **End**, checked at every close after t, first that holds: base weight in force is 1 (base takes
  over); close below L (stop); 126 trading days after t.
- Re-trigger: thrust days during an episode are ignored. A new episode needs a new thrust day after
  the end.
- Mechanics: decide close t, trade close t+1, earn from t+2; 10 bp per unit weight change. The base
  rule keeps its Tuesday / first-of-month state machine unchanged. B is evaluated every trading day,
  as S is in `run_s.py`.
- Not used: Deemer's later 20-day rule (ratio > 1.72), which he added in 2019 after seeing 2016;
  Zweig's thrust (reported only, after the verdict).

### Test (one run)

Primary: R0+B vs R0, S&P 500, Tuesday base, decisions from 1965-04-01 to the last breadth day.
H1 to 1990-12-31, H2 from 1991-01-01. All gates must hold.

- G1 Sharpe R0+B > R0 in FULL, H1, H2, and on the mean of the five decision weekdays.
- G2 maxDD >= R0 - 0.02 in FULL and inside 2000-03-01..2003-03-31, 2007-10-01..2009-06-30, 2022.
- G3 FULL Sharpe > R0 without 2020-02-19..2020-08-31 and 2025-02-19..2025-06-30.
- G4 paired moving-block bootstrap (63 days, 2,000 draws, seed 7): share(not better) < 0.10.
- G5 placebo, 1,000 draws, default_rng(7): each episode moved to a uniformly random day with base
  weight 0; L = lowest of the 63 closes ending there; same stop, base-invested end and 126-day limit.
  Real Sharpe gain > 90th percentile.
- G6 more than half of the episodes have a positive return difference vs R0.
- G7 product: V2+F+S+B has Sharpe >= V2+F+S and maxDD >= V2+F+S - 0.02, 1993-06 onward. S is sized
  live (spec 6sl passed; `S_SIZE = 'live'` in `run_b.py`).
- G8 (kept): the episodes entered from 2003-01-01, after the
  rule was public, have a positive total return difference vs R0.

Report, no gate: metrics by period, episode table, product by stress window, overlap check of the two
breadth sources, Zweig thrust dates. Sensitivities (threshold 1.90 / 2.05, window 63 → 21) only after
the verdict is written.

### Hurdle after publication

- McLean and Pontiff (2016, read): across 97 published stock predictors, returns are 26% lower out of
  sample and 58% lower after publication. Their sample is cross-sectional stock anomalies, not market
  timing, so it is a guide, not a measurement.
- Applied here: whatever the pre-2003 episodes show, plan on about 40% of it afterwards. That is why
  G5 (placebo) and G8 (post-publication episodes) carry the weight, and why a pass on G1 alone would
  mean little.
- Expected triggers: Deemer's table has 13 ten-day signals from 1975 to 2023, about one every
  3.5 years. Only those that fall while the base rule is in Cash become episodes. If there are fewer
  than 10 episodes, the result is reported as "too few to conclude", whatever the gates say.
- My prior that B passes every gate: about 1 in 4.

### Backup proxy (not part of the test)

Kenneth French daily 49-industry portfolios (from 1926): A = industries with a positive return,
D = industries with a negative return. Industries move together far more than single stocks, so the
1.97 threshold does not carry over and a new threshold would be a free choice. Use it only to describe
1926-1965, never as the test series.

### Run condition
B runs once, only when `BBG_NYSE_ADV` / `BBG_NYSE_DEC` reach the end of the sample (user download). No partial run on the 1965-2020 free data.

### 6t pre-run data note (2026-11-17, before any return of B was computed)
The Bloomberg file (NYAADV / NYADCL Index, from 2004-07-16) counts a narrower universe than the
Unicorn files: about 60% of the advancing and declining counts; daily correlation 0.97; 10-day ratio
correlation 0.96, median absolute difference 0.06. On the 2004-07..2020-02 overlap, the 1.97 crossing
fires 12 times on Bloomberg vs 6 on Unicorn (Unicorn matches Deemer's published table). After the
splice date, Bloomberg gives 2 thrust days (2020-06-05, 2023-01-12). Decision: the test runs as
registered (Unicorn to 2020-02-10, Bloomberg after). Added report, no gate: the same gates on the
single-source sample 1965-04 → 2020-02-10.

## 11. Product decision (user, 2026-11-17)

**Core (herculis.ai quant page and Delphi core) = no sector tilt.** Equity part = 100% S&P 500
(SPY). Equity weight = max(V2, F floor, S-live weight, B weight), where V2 = R1 timing x volatility
size (spec 6c / run_k1.v2_weights), F = VIX re-entry floor 50% (6y), S-live = shock re-entry at the
live V2 size (6s, 6sl), B = Deemer breadth thrust at the live V2 size (6t). Cash = T-bill / BIL.
Reference result (RESULTS_sens.md, 100% SPY column, 1999-02 → 2026-08, 10 bp): total +1261%,
Sharpe 0.72, worst fall -25.4% (S&P 500: +877%, 0.42, -55.3%).

**Satellite (Delphi only, later) = the sector tilt as a separate overlay.** It is kept only if it
shows real added value on its own, measured against SPY, under a test pre-registered before it
runs. Reason: in the core, the 70/30 tilt left the 27-year return unchanged (+1266% vs +1261%),
cut the worst fall slightly (-23.5% vs -25.4%) and lost 14 points in 2014-2018; that is not enough
to put it in the core.

### 11a. Trading rule (user decision, 2026-11-18)

- Every trading day t, compute the desired weight = max(R1 state x v, F floor, S-live, B).
- If it differs from the held weight by 10 points or more, or if we go to / leave cash: trade at the close of t+1, earn from t+2.
- We do not wait for the committee. The Wednesday committee reviews the trades and the gauges.
- The keys (DMAS, revisions) keep the tested Tuesday rule (month lock, early exit only if DMAS < 30).
- Info (RESULTS_core.md, same period 1999-02 → 2026-11-13): act next day Sharpe 0.70, worst fall −25.9%; wait for Wednesday 0.70 / −26.7%; keys read daily with month lock 0.70 / −19.9% (but 2013–2026 0.92 vs 1.01); keys read daily without lock 0.53 / −34.7% (rejected: too many false switches).
- The 10-point threshold is a choice taken from the old month-start band. Not optimised.

### 11b. Everything daily, sizes in 5-point steps (user decision, 2026-11-18)

- Replaces 11a. The keys are read every trading day (daily DMAS + daily revision z). The month lock stays: the first trading day of the month sets the state; mid-month exit only if DMAS < 30. Without the lock the Sharpe falls to 0.53 (info row), so the lock is kept.
- Desired weight rounded to the nearest 5 points. Trade next day if the gap is >= 10 points, or to/from cash.
- RESULTS_core.md row 6 (1999-02 → 2026-11-13): Sharpe 0.69, +1108%, worst fall −20.1%. Not rounded: 0.70 / −19.9%. Keys on Tuesdays: 0.70 / −25.9% (stronger 2013–2026: 1.01 vs 0.90).
- Slide: no back-test (it goes to herculis.ai). Chart shows the 200-day average. Re-entry rules shown with their current reading and trigger.

### 6h. H: HTEI re-entry — pre-registration (2026-11-18, before any run)

**Idea.** The gauge is trend-following and misses V-shaped rebounds. HTEI is contrarian (high fear = BUY). Test HTEI's BUY zone as a 4th re-entry layer, used only when the keys put us in cash.

**Data.** `Tools_In_Construction/HTEI/daily_signals/signals_history.csv`, column `us_score`, read-only. Row date = close date of the inputs. 4,203 rows are a backtest, 155 rows are live (from 2026-04-13). The HTEI components were chosen on 2010–2026 data, so this test is mostly IN-SAMPLE for HTEI. A PASS means "promising, keep watching live", not proof.

**Rule (fixed before the run).**
- Trigger day t: `us_score < 35` at close t (level, not crossing).
- An episode starts at a trigger day only if the base keys weight in force is 0 (cash), using the product keys (daily, month lock).
- Episode mechanics = B (run_b.thrust_episodes): L = lowest close of the 63 closes ending t; end at the first of: base invested, close below L, 126 trading days. Triggers inside an episode are ignored; after the end, the next trigger day starts a new episode.
- Size = live volatility target v (as S-live and B). Trade t+1 close, earn from t+2.
- Product + H: desired = max(product desired, H), rounded to 5 points, trade next day if gap >= 10 pts or to/from cash (spec 11b). 10 bp costs.

**Sample.** First HTEI date (2010-02-08) → last common date. Halves: to 2017-12-31 and from 2018-01-01.

**Gates (all must pass).**
- G1. Sharpe(product + H) > Sharpe(product).
- G2. Worst fall(product + H) >= worst fall(product) − 2 points.
- G3. Placebo: 1,000 draws, `default_rng(7)`, each episode moved to a random day with base weight 0 (run_b.placebo_spans: same L, stop, base-invested end and time limit), same sizing and execution. p = share of placebo Sharpe >= real Sharpe. Pass if p <= 0.10.
- G4. More than half of the episodes beat cash (S&P return over the episode days > T-bill return).
- G5. Both halves: Sharpe(product + H) >= Sharpe(product).

**One run.** No change of threshold (35), exits or sample after the run. Results go to `research/us_gauge/RESULTS_h.md`.

### 6L. L: re-entry layers at full weight — pre-registration (2026-11-18, before any run)

**Idea (VGAP2_fable_2026-11-18.md, idea 1).** Inside a shock (S) or breadth (B) episode the price stop already bounds the loss. The live volatility size counts the same risk twice and keeps the size small right after a crash (18% at the COVID low). Test S at 100% (S-full) against S at the live size (S-live).

**Fresh sample.** Six indices never opened in this project: `N225.INDX`, `TWII.INDX`, `GSPTSE.INDX`, `HSI.INDX`, `KS11.INDX`, `AXJO.INDX` (EODHD, in the market store, read-only). Price returns only, cash 0%, 10 bp per unit traded. Data cleaning fixed now: drop NaN and non-positive closes; daily moves above 25% are listed in the report, not removed.

**Base in each market.** Plain trend rule: `gauge.dmas_on` on Tuesday decision days (`run_p.decision_days`, W-TUE), cash if DMAS < 45, month lock, mid-month exit if DMAS < 30 (`run_fast.states`), times the volatility size (`run_fast.v2_from_states`). Decisions start 252 trading days after the first close. Shock episodes: `run_s.shock_episodes` on the plain-rule weight (unchanged mechanics: fall >= 15% with the low within 63 days of the 1-year high, +10% off the low, exit on base invested, close below the low, or 126 days).
- S-live: max(base, live v) during an episode (as spec 6sl). S-full: max(base, 1) during an episode.

**Clusters.** Entries sorted by date across markets; a new cluster starts when an entry is more than 14 calendar days after the first entry of the current cluster.

**Gates (all must pass).**
- G1. Mean of the six Sharpe differences (S-full − S-live) > 0, and positive in at least 4 of 6 markets.
- G2. Worst fall of S-full no more than 3 points deeper than S-live in every market.
- G3. More than half of the clusters have a positive mean index return over their episodes (earn days); also true without the 2020 and 2025 clusters.
- G4. Worst single episode index return (earn days, i.e. at full weight) no worse than −15%.
- G5. Placebo: 1,000 draws, `default_rng(7)`; in each market the same number of episodes moved to random base-cash days (`run_s.placebo_spans`), at full weight. Pooled gain = mean over markets of Sharpe(base + episodes at 100%) − Sharpe(base). p = share of draws with pooled gain >= real pooled gain. Pass if p <= 0.10.
- Fewer than 10 clusters → "too few to conclude", no change.
- If G1, G3 and G5 pass but G2 or G4 fail → adopt the pre-declared fallback: S and B at max(live v, 50%).

**Product consequence.** PASS → S and B at 100% during episodes in the core. B has no fresh sample (NYSE breadth only); it follows S by symmetry, and the report must say so. The US product effect is reported as information (in-sample).

**One run.** No change of markets, thresholds, gates or cleaning after the run. Results to `research/us_gauge/RESULTS_l.md`.
