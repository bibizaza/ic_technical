# US quant gauge — design

Date: 2026-11-16. Status: draft for approval.

## 0. Correction log

- **2026-11-17 — cash leg data error.** Steps 1, 1b, 1c and 2 used `BIL.US` (EODHD unadjusted close)
  as the cash return from 2007-05-31. BIL pays its yield as monthly dividends, so its price is flat:
  cash earned ~0% instead of the T-bill rate (e.g. ~5% in 2024). The pre-registered intent was BIL's
  total return. **Fix:** use `BIL.US_TR` (adjusted close). No rule, threshold, window or gate changes.
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
