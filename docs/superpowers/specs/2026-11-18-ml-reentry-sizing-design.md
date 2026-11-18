# ML sizing of the shock re-entry ("M") — design and pre-registration

Draft by Fable 5.2, 2026-11-18. Design only: **nothing was fitted, no label or return after any event was
computed.** Only event entry dates were counted (scratch `mlcount/count.py`, `count_gated.py`, read-only).
To be committed before any feature or label file is built. Parent spec: `2026-11-16-us-quant-gauge-design.md` (11d core).

## 0. Summary

- **One question.** On the day a shock re-entry (S) fires: how likely is it that the low breaks again (a 1987/1929-type retest)? The answer picks one of three sizes that already exist in the spec: live vol size, max(live, 50%) (today's 11d), or 100%.
- **Events.** S-type price events, same rule in 21 markets: 261 events, about 78 independent clusters, 1965–2026. The US has 9.
- **Model.** A 6-feature logistic regression is the gate model. A small monotone gradient-boosting model is a challenger only.
- **Test.** One walk-forward run, out-of-sample 2000–2026: 159 events, about 45 clusters. Permutation p ≤ 0.01 on the sizing gain.
- **Honest odds.** Full pass about 1 in 7. Expected US Sharpe gain if adopted +0.01 to +0.03. The real value is tail control: being allowed to hold 100% when the odds are good, and less when they are not.

## 1. Question and decision

**Model output:** p = probability that an S episode ends on its stop (a close below the crash low L within 126 trading days of entry). One number per event, computed once at the trigger close t, frozen for the episode.

**Decision it drives:** the S weight during that episode only (spec 6s mechanics unchanged: trigger, stop below L, 126-day limit, keys take over).

| Bucket | Condition (π = stop share in the training events) | S weight |
|---|---|---|
| Low retest risk | p ≤ 0.6 π | 100% |
| Normal | between | max(live vol size, 50%) — spec 11d, unchanged |
| High retest risk | p ≥ 1.5 π | live vol size (no floor) |

The multipliers 0.6 and 1.5 are fixed now. They are not tuned.

**Why S and only S.**
- S is the only re-entry layer defined on prices alone, so the same event can be built in every market. B needs NYSE breadth and F needs VIX: US only, 10 and 7 episodes, no stop ever observed. Nothing to learn from.
- S is where the open risk question sits. The user moved S from 100% (11c) to a 50% floor (11d) because of 1929/1930 and 1987. That was a hedge without a signal. This study asks whether a signal exists.
- S has observed failures in every market (18 stops in 75 episodes in the 8 markets already studied), so the label has two classes.
- Not in scope: the size when the keys flip back (no stop, no event definition, a different question); the cash/hold decision of the keys; any change to triggers, stops or timing. The rules decide WHEN. The model only chooses among three sizes.

## 2. Events

### 2.1 Definition (daily closes P of one index; zero free choices beyond spec 6s)

- h_t = date of the highest close in the 252 trading days ending t; H = P(h_t). L_t = lowest close from h_t to t; l_t its date.
- **Trigger at close t** when all hold: L ≤ 0.85·H; l − h ≤ 63 trading days; P_t ≥ 1.10·L; for the same h, L is lower than the L of any earlier event (a new h resets this).
- After a trigger, no new event in this market until the event closes: the first close below L, or t + 126 trading days.
- **No base gating.** The event does not ask whether a trend rule is in cash. Reason: the US base uses the earnings key, other markets cannot; a price-only event is the same object everywhere. The product episodes (trend rule in cash, spec 6s state machine) are used in gate G4; they are a separate, smaller list (2.3), not a filter of this one.
- Each event stores: market, h, l, t, H, L, P_t, and `base_cash` (plain trend rule weight in force at t < 1; information).

### 2.2 Panel and cleaning (fixed now)

21 indices: `GSPC`, `BBG_SXXP`, N225, TWII, GSPTSE, HSI, KS11, AXJO, AEX, ATX, BFX, BSESN, BVSP, FCHI, GDAXI, IBEX, MXX, NZ50, OMXS30, SSMI, STOXX50E (`.INDX`, market store, `refresh=False`).
- Drop NaN, non-positive closes and duplicate dates. BVSP is used from 1995-01-02 only (gap 1992–95, hyperinflation; 9 daily moves above 25% before).
- Daily moves above 25% elsewhere (HSI 1, MXX 1, OMXS30 1) are listed in the report, not removed.
- Sample: trigger dates from 1965-01-01. An event needs 756 closes before h (for the run-up feature). US 1928–1964 is left out of training (no FX, no rates, already used in 6LU, and judged a different market by the user); the model's score on those events is a report-only line.
- Events whose 126-day window is not complete at the data end are "open": no label, kept for the live log.

### 2.3 Counts (entry dates only; counted 2026-11-18)

| | Events | Clusters (45 d) | Clusters (21 d) | Calendar years |
|---|---:|---:|---:|---:|
| All, 1965–2026 | 261 | 78 | 115 | 47 |
| Training only, before 2000 | 102 | 33 | 50 | 23 |
| Out-of-sample, 2000–2026 | 159 | 45 | 65 | 24 |
| of which 2016–2026 | 71 | 18 | 23 | 9 |
| 13 never-opened markets, 2000–2026 ("clean set") | 94 | 35 | 43 | 20 |
| Product episodes (`run_s.shock_episodes` on the plain Tuesday trend rule, as 6L), all / 2000+ | 180 / 115 | 69 / 38 | 80 / 45 | 40 / 20 |
| US, all / 2000+ | 9 / 4 | 9 / 4 | | |

Per market: TWII 27, N225 23, HSI 22, MXX 18, KS11 16, BVSP 15, OMXS30 13, GDAXI 12, ATX 11, FCHI 11, IBEX 11, STOXX50E 11, AEX 10, SSMI 10, SXXP 10, BFX 9, GSPC 9, AXJO 8, BSESN 7, GSPTSE 7, NZ50 1. By region: Europe 108, emerging 83, Asia-Pacific developed 54, North America 16.
US trigger dates: 1980-05-22, 1987-10-21, 1988-08-29, 1990-12-04, 1998-09-23, 2010-08-02, 2020-03-25, 2023-04-24, 2025-04-24. The count after the 756-day history filter will be a few lower; the builder reports it before labels exist.
Wider families were counted and rejected: no speed limit gives 800 events but only 121 clusters (many re-triggers inside one bear), and they are not the product's decision. A 10% fall gives 469 events of a different kind.

### 2.4 Overlap and clustering: the real sample size

- **Cluster** = events (all markets) chained by entry date: an event joins the current cluster if its entry is within 45 calendar days of the previous entry in that cluster. Largest clusters: 21 events (2020), 15, 12, 11, 11. 38 clusters are single events.
- **The cluster is the unit everywhere**: training weight of an event = 1 / (events in its cluster); all test statistics are cluster means; bootstrap and permutation move whole clusters.
- Effective sample: about 78 clusters in total and 45 out-of-sample. With a stop rate near 25% (seen in the 8 studied markets) that is roughly **11 failed clusters in the test**. Every number in this design must be read against that.
- Europe is 108 events but STOXX 600, EURO STOXX 50, DAX, CAC, AEX, … are close to one market. Clusters handle this; no index is dropped.

## 3. Label

- **Primary (classification):** y = 1 if any close in t+1 … t+126 is below L ("stopped"); else 0. This is exactly the product's loss case: a stop costs the distance to L, 9% or more, by construction.
- **Economic (for the gates, not a model target):** R = P(x+1)/P(t+1) − 1, where x is the first stop day or t+126. Trade in at the close of t+1 and out at the close of x+1 (spec 11a timing). Price index, cash 0%, in every market. Costs: 10 bp per unit of weight traded, charged on the weight difference between the model size and the 11d size.
- **Why "stopped" and not the return:** it is the user's question (retest or not); a two-class target with a 25% base rate is far easier to learn from 78 clusters than a fat-tailed return; and the size map needs only a risk ranking. The return R is used to value the sizes, never to fit them.
- Horizon 126 trading days = the S time limit. The label ignores the base rule (no "keys take over" end) for the reason in 2.1.

## 4. Features (15, all known at the trigger close)

Timing rule: local closes up to t. Every other series (other markets, FX, US rates, credit) uses the last value dated on or before **t − 1 business day** (time zones, and FRED publishes daily data the next day). The trade is at the close of t+1, so this is conservative. No feature uses revised macro data.

| # | Feature | Definition | Why | Source, history | Markets |
|---|---|---|---|---|---|
| 1 | `depth` | L/H − 1 | size of the shock | closes | all |
| 2 | `speed_hl` | l − h, trading days (1–63) | crash vs slide | closes | all |
| 3 | `days_le` | t − l, trading days | a +10% bounce in 2 days is short covering; stopped entries came faster (STOPS memo, in-sample) | closes | all |
| 4 | `crash_share` | worst one-day log return in (h, l] / log(L/H) | one-day crash (1987) vs spread fall | closes | all |
| 5 | `vol_rel` | σ21(t) / median σ21 of the 1,260 days ending h | panic relative to this market's normal | closes | all |
| 6 | `runup_2y` | log(H / P(h − 504)) | falls after a bubble top keep failing (1929, Japan 1990, 2000) | closes | all |
| 7 | `attempt` | earlier events in this market in the 504 days before t (cap 3) | first bounce in a long bear vs a fresh shock | closes | all |
| 8 | `glob_breadth` | share of the other panel markets ≥ 15% below their 252-day high, at t−1 | global shock vs local shock | panel closes | all (≥ 3 other markets live; from 1984 ≥ 7) |
| 9 | `rel_depth` | local drawdown minus the median drawdown of the other markets, at t−1 | is this market the centre of the problem | panel closes | all |
| 10 | `fx_shock` | log change of local currency per USD from h to t−1 (+ = local currency fell). US: minus the mean change of JPY, CHF, GBP, CAD vs USD | currency crisis vs equity-only shock; the CHF/SMI point | `FRED_DEX*`, 1971 (KRW, HKD 1981; TWD 1983; MXN 1993; BRL 1995; EUR 1999) | see below |
| 11 | `vol_level` | median σ21 of the 1,260 days ending h | structural volatility: proxy for retail-driven and emerging markets | closes | all |
| 12 | `region` | North America / Europe / Asia-Pacific developed (JP, AU, NZ, HK) / emerging (TW, KR, IN, BR, MX); 3 dummies | market structure, without one dummy per market (21 dummies on 78 clusters would overfit) | fixed list | all |
| 13 | `d_us10y` | change of the US 10-year yield from h to t−1, points | rate shock (1987, 1994, 2022) vs flight to quality (2020) | `FRED_DGS10`, 1962 | all (global) |
| 14 | `curve_min` | lowest (10y − 3m T-bill) of the 252 days ending t−1 | inverted curve in the last year = late cycle; real-time, never revised | `FRED_DGS10`, `FRED_DTB3` | all (global) |
| 15 | `d_credit` | change of the Baa − Aaa spread from the month before h to t−1 | credit stress = solvency problem, not only a price shock | daily `FRED_DBAA − FRED_DAAA` from 1986; before, monthly `FRED_BAA − FRED_AAA`, a month's value usable from the first business day of the next month | all (global) |

**FX detail.** Euro members before 1999 (DE, FR, NL, BE, AT, SXXP, STOXX50E): `FRED_DEXDNUS` (Danish krone, tied to the mark) as the proxy. Spain before 1999 and any event before the FX series starts: missing. From 1999: `FRED_DEXUSEU` (inverted). HKD is pegged: the feature is near zero, which is correct information.

**Missing values.** Logistic: training-fold median. Boosting: native missing handling. No event is dropped for a missing feature. Expected share missing: below 5% (FX only).

**Left out on purpose (say it on the slide if asked).**
- VIX, VXO, VIX term structure, AAII, put/call, margin debt, CFTC, HY OAS, NYSE breadth, EPS revisions, CAPE, HTEI: **US only, and the US has 9 events.** A US-only feature cannot be learnt from 9 events and cannot be pooled. They stay on the dashboard as context. Realised volatility (5, 11) stands in for VIX in every market.
- OECD recession flags: dated after the fact (and the files end in 2022) — look-ahead. Not usable.
- OECD leading indicators: revised and re-normalised; FRED holds only today's vintage. Not in the gate model. Local 3-month and 10-year rates: start too late in key markets (Japan 3m 2002, Korea 10y 2000, Switzerland 3m 1999).
- Volume and intraday range: zero or flat in most non-US files before 1999–2004.
- A true retail-share series: none exists free with history. Features 11 and 12 are the proxy.
- **Report-only extended model (never a gate):** boosting with the 15 features plus local 10y change h→t (monthly, lagged one month), local curve, and the 6-month change of the OECD leading indicator lagged 3 months, on the events where they exist. It is flagged "revised data" and can at most motivate a later pre-registration.

## 5. Models (no tuning anywhere)

**Gate model — logistic regression ("M-L").**
- 6 features chosen now on economic priors: `runup_2y`, `attempt`, `days_le`, `vol_rel`, `fx_shock`, `d_credit`. Rule of thumb: about 20 failed clusters in training supports 5–7 coefficients, not 15.
- Winsorise at the training-fold 1st/99th percentile, standardise on the training fold, L2 penalty C = 0.5 (`sklearn LogisticRegression`), cluster weights, intercept free.
- Expected signs (reported, not forced): run-up +, attempt +, days_le −, fx_shock +, d_credit +; `vol_rel` open.

**Challenger — gradient boosting ("M-G"), xgboost 3.2 (installed).**
- All 15 features. `max_depth=2, n_estimators=200, learning_rate=0.03, min_child_weight=3, subsample=0.7, colsample_bytree=0.7, reg_lambda=5`, log-loss, cluster weights, mean of 25 seeds (0–24).
- Monotone constraints where the economics is clear: `runup_2y` +1, `d_credit` +1, `curve_min` −1. All others free.
- M-G can replace M-L **only if M-L passes all gates**, M-G passes them too, and M-G's out-of-sample log-loss is lower (cluster bootstrap, one-sided p < 0.10). If M-L fails, M-G is report-only whatever it shows. This keeps one primary test.

**Benchmarks in every table:** training base rate (no features); and three one-feature rules (`days_le`, `vol_rel`, `runup_2y` alone). A model that does not beat its best single feature is not worth its complexity; say so in the report.

## 6. Validation

### 6.1 Walk-forward (the one run)

| Block (entry date) | Training events | Test events | Test clusters (45 d) |
|---|---|---:|---:|
| 2000–2004 | entry ≤ 1999-03-31 | 28 | 10 |
| 2005–2009 | entry ≤ 2004-03-31 | 29 | 7 |
| 2010–2015 | entry ≤ 2009-03-31 | 31 | 10 |
| 2016–2026 | entry ≤ 2015-03-31 | 71 | 18 |

- **Purge and embargo:** a training event must have its 126-day label window closed plus 63 trading days before the block starts. The 9-month cut-off above is that rule (126 + 63 trading days) written as a date.
- A cluster is never split between training and test (the cut-off and the 45-day chaining guarantee it; the code asserts it).
- π and all scaling come from the training events of the block. The 159 out-of-sample predictions are pooled.
- No inner folds are needed: there is no hyperparameter to choose.

### 6.2 What has already been seen — stated plainly

- **US, all dates:** fully seen (about 150 looks). 1987, 1998, 2020 and 2025 are why this study exists. US results here are **never evidence**.
- **SXXP, N225, TWII, GSPTSE, HSI, KS11, AXJO:** S episodes and their outcomes to 2025 are in `RESULTS_l.md` and `STOPS_s`. The feature list was written by someone who has read them (e.g. "stopped entries came faster").
- **13 markets never opened** (AEX, ATX, BFX, BSESN, BVSP, FCHI, GDAXI, IBEX, MXX, NZ50, OMXS30, SSMI, STOXX50E): no price file was read before the count on 2026-11-18, and only entry dates were counted. **This is the clean set.** It shares the big clusters (2008, 2020) with the seen markets, so it is clean for design choices, not independent in outcome; the walk-forward takes care of the second point.
- A single "train to 2015, test 2016–2026" split is not used as the gate: it has 18 clusters, one of them (2020) holds 21 of 71 events. It is one block of the walk-forward and one stability row.
- **The only fully clean US test is the live record.** One US S event every 5 to 10 years: it cannot confirm the model in our planning horizon. Adoption therefore rests on pooled evidence, as S itself did (Europe). This follows the user's rule: other markets may teach the model; they do not veto US evidence. Here there is no US evidence to veto.

### 6.3 Statistics (all on the pooled out-of-sample events, cluster level)

- Δw_i = model S weight − 11d S weight, with v = live vol size at t: +(1 − max(v, 0.5)) in the low bucket, 0 in the normal bucket, −max(0, 0.5 − v) in the high bucket.
- **Sizing skill** K = mean over clusters of (Δw_c − mean Δw of its block) × R_c, net of costs. Subtracting the block mean removes the effect of simply holding more on average: K is positive only if the model holds more **in the right events**.
- AUC of p against y with cluster weights. Calibration slope: logistic regression of y on logit(p). Brier score against the base-rate forecast.
- **Permutation test:** 5,000 draws, `default_rng(7)`; cluster-level Δw (and p) are shuffled across clusters **within each block**; p-value = share of draws with K (or AUC) ≥ the real one.
- Also reported, no gate: leave-one-calendar-year-out predictions for all 261 events (this is the only way to score 1987 out of fold; it trains on later data, so it is information only); coefficients and their bootstrap sign stability; score of the US 1929–1954 events.

## 7. Gates and consequences (declared before the run)

| Gate | Requirement (M-L, out-of-sample 2000–2026) |
|---|---|
| G1 Ranking | AUC ≥ 0.60 and permutation p ≤ 0.01 |
| G2 Calibration | slope between 0.5 and 1.5; Brier score below the base-rate forecast |
| G3 Sizing skill | K > 0 with permutation p ≤ 0.01 |
| G4 Product, pooled | In each of the 21 markets: plain trend rule (DMAS 45/30, Tuesday, month lock, exactly `run_l.run_market`) × vol size + S (`run_s.shock_episodes`), each episode scored with the walk-forward model at its trigger day, S sized by the model vs S at max(live, 50%), 2000–2026, 10 bp, 5-pt rounding, 10-pt rule. Mean Sharpe difference > 0, positive in ≥ 13 of 21; worst fall no more than 2 points deeper in ≥ 18 of 21 and never more than 5 points deeper |
| G5 Stability | K > 0 in 2000–2009 and in 2010–2026; K > 0 without the 2020 cluster; K > 0 without 2008–2009; K > 0 in at least 3 of the 4 regions |
| G6 Clean set | On the 13 never-opened markets alone: AUC > 0.5 with permutation p ≤ 0.05, and K > 0 |
| G7 US non-inferiority (not evidence) | 11d core with model-sized S vs 11d core, 1999-02 → 2026: Sharpe ≥ core − 0.01, worst fall no more than 2 points deeper. US walk-forward events (2010, 2020, 2023, 2025): buckets shown, flagged in-sample for design |
| G8 Simplicity | M-L's K is above the K of each one-feature rule (else adopt nothing; report the one-feature finding for a later pre-registration) |

Fewer than 35 out-of-sample clusters with a label, or fewer than 8 failed clusters → "too few to conclude", no change.

**Consequences.**
- **All pass → adopt** the three-bucket size for S in the core (section 8), with the live kill rules of section 9.
- **G1, G2, G3, G6 pass; G4 or G5 fails on the worst-fall or stability leg → defensive half only:** the high-risk bucket goes to the live size; no step up to 100%. (Pre-declared fallback, like 6L.)
- **G1 passes at p ≤ 0.10 but not 0.01, or G3 fails with G1 and G2 passed → report only:** "retest risk" is shown on the dashboard as context; sizes stay 11d; the live log starts; no adoption before 20 live pooled events.
- **G1 fails at p > 0.10 → stop.** The file "ML sizing of re-entries" is closed with these data. Recorded in the rejected list.
- In every case: **one run.** No second run with other features, labels, thresholds, blocks, buckets or markets. A new idea needs a new pre-registration and new data.

## 8. Product integration (if adopted)

- At the close of an S trigger day t the model gives p and a bucket. The bucket is frozen until the episode ends.
- S weight each day: low → 1.00; normal → max(v(t), 0.50); high → v(t). v = min(1, 15%/σ21), live, as today.
- Desired weight = max(keys × v, F floor 50%, S weight, B 100%), rounded to 5 points; trade next day if the gap is ≥ 10 points or to/from cash (11b). S stop, 126-day limit and "keys take over" unchanged. B and F unchanged.
- The size is always between the live size and 100%. The model cannot open a position, close one, or go below what the vol target already allows.
- Any missing input or stale file on the trigger day → normal bucket (11d). Logged.
- Model frozen at adoption (fit on all closed events). Refit once a year in January with the same features and settings; the coefficients before and after are filed. No other change without a pre-registration.
- **Committee slide, one line under the S gauge:** "Shock re-entry ON since 24 Apr — size 100%. Retest risk 11% (normal 25%). Why: no run-up before the fall, credit spreads calm, currency stable." The three reasons are the three largest contributions to the score: coefficient × standardised value for M-L; `pred_contribs` (tree SHAP) for M-G. A fixed dictionary maps each feature and sign to a plain phrase. When S is off the line reads "armed: needs a 15% fall in 63 days, then +10%".
- Slide footnote, always: "Sized by a model trained on 260 crashes in 21 markets; the US has 9."

## 9. Risks, stop conditions, odds

- **Power.** About 45 test clusters and about 11 failures. To reach p ≤ 0.01 the AUC must be near 0.70. Markets rarely give that on the entry day. A true but modest signal (AUC 0.60) will fail the gate most of the time. This is accepted: a false adoption costs more than a missed one.
- **Entry-day blindness.** VGAP found that real and failed lows look alike until day 21. The model only has entry-day data. The most likely outcome is "no signal".
- **Pooling.** If the US answers to different drivers than Taiwan or Mexico, the pooled model is noise for the US. `region` and `vol_level` are the only defence; G5 checks regions.
- **Designer contamination.** Features were written knowing 8 markets. G6 on the clean set is the answer; it is weaker than a truly independent sample.
- **One cluster dominates.** 2020 is 21 events and a win everywhere. Cluster weights and "without 2020" in G5 handle it.
- **Small product effect.** In-sample, S at 100% vs the 50% floor is worth +0.02 Sharpe on 3 US episodes. Even a perfect model adds little to the headline. Do not sell it as a return engine.
- **Regime of one.** The next US shock may be of a kind the panel has never seen.

**Stop the build (before the run) if:** fewer than 200 labelled events or 60 clusters survive the history filter; a look-ahead test fails; more than 10% of events miss more than 2 features.
**Live kill rules (after adoption):** (a) if 2 of the first 4 episodes in the 100% bucket, counted over the US and STOXX 600, end on the stop → back to 11d, no new study; (b) every pooled live event in the 21 markets is logged with its p and bucket; after 20 closed live events, if K on the live log is negative → back to 11d.

**Odds (my estimate):** full pass and adoption about 15%. Defensive half only: a further 10%. Report-only signal: about 20%. Stop: about 55%. Expected US Sharpe gain if adopted: +0.01 to +0.03; expected cut in the cost of a failed re-entry: 3 to 5 points in the cases the model flags. This is lower than the earlier estimate (+0.03–0.05, 1 in 3): the cluster count showed 45 independent test cases, not "a few hundred".

## 10. Build plan

All code in `research/us_gauge/ml/`, tests in `tests/`. Market data `refresh=False`, inside `read_only_external()`. Steps 1–5 never read a price after t.

1. `ml/panel.py` — load and clean the 21 indices (2.2); report daily moves > 25%. Test `tests/test_us_gauge_ml_panel.py`: BVSP cut, duplicates, no network.
2. `ml/events.py` — event state machine (2.1), cluster ids (2.4), `base_cash` flag from the plain trend rule. Writes `ml/out/events.csv`. Test `test_us_gauge_ml_events.py`: synthetic paths (one V, one retest, a slow fall that must not trigger, re-trigger limit); counts match section 2.3 before the history filter (261 events / 78 clusters; 180 product episodes from `run_s.shock_episodes`).
3. `ml/features.py` — the 15 features with the t−1 rule and the monthly-lag rule. Writes `ml/out/features.csv`. Test `test_us_gauge_ml_leakage.py`: truncate every input at t (local) and t−1 (others) → identical features; shift a future price → no feature changes; monthly credit uses only finished months.
4. `ml/models.py` — M-L, M-G, benchmarks, bucket map, contributions. Test `test_us_gauge_ml_models.py`: fixed seeds reproduce; monotone constraints hold on a grid; on synthetic data with a planted signal the pipeline finds it, with shuffled labels it does not.
5. `ml/validate.py` — blocks, purge/embargo assertions, cluster weights, K, AUC, calibration, permutation. Test `test_us_gauge_ml_validate.py`: no cluster in both training and test; embargo dates; permutation p is uniform on noise (200 synthetic runs).
6. **Freeze.** Commit steps 1–5 and `features.csv`; write its SHA-256 and the commit hash into this spec. From here no feature, setting or gate may change.
7. `ml/labels.py` + `ml/run_m.py` (guarded: needs `--one-run`, refuses to run if the hash differs or `RESULTS_ml.md` exists) — labels, walk-forward, gates G1–G8, the 21-market product simulation and the US core check (reusing `run_l.run_market`, `core_slide_data`). Output `research/us_gauge/RESULTS_ml.md`. Test `test_us_gauge_ml_labels.py` on synthetic paths only.
8. If adopted: `ml/live.py` (score on a trigger day, slide line, live log `ml/out/live_log.csv`) and the spec entry "11e". If not: one paragraph in the rejected list.

## 11. Free data that would still help (not Bloomberg; none is needed to run)

| Data | Source | Why |
|---|---|---|
| Pre-euro FX vs USD: `DEXGEUS`, `DEXFRUS`, `DEXSPUS`, `DEXNEUS`, `DEXBEUS`, `DEXASUS`, `DEXITUS` (1971–1998, discontinued) | FRED | exact `fx_shock` for European events of 1987, 1990, 1992, 1998 instead of the krone proxy; fills Spain |
| Central-bank policy rates, daily/monthly, about 40 countries from 1946 | BIS data portal (free CSV) | a real-time, never-revised "local easing has started" feature in every market; the OECD rate files start too late |
| FTSE 100 / All-Share (not on EODHD), and longer histories for DAX, CAC, Hang Seng, Nikkei, Dow | Stooq (free daily CSV) | a 22nd market and more events before 2000, where training is thin (102 events) |
| Real-time vintages of the OECD leading indicators | ALFRED / OECD "MEI original release" — check that they exist per country | would allow a macro feature without look-ahead |

If any of these is added, it is added **before step 6** and the counts in 2.3 are re-stated. After the freeze, no new data.

## 12. Freeze (2026-11-18, before any label was computed)

- **SHA-256 of features.csv: `afc30914086be91c7b8e98518be8dcd45cb104628ead82e7f6dc0c8cbf8a0706`** (`research/us_gauge/ml/out/features.csv`).
- Frozen at commit **`1c13b5d7615fad8604deafb1cb86f783a7f29b04`** (steps 1–5, events.csv, features.csv). Also frozen: events.csv `62be226c740f49556748290abc5a162b5481c552505f6b89b47f138f7cf77bef`, product_episodes.csv `3dd3d9c9a263c73ee1437c3faef893ef658cf9acc0976868a513eb461ab8e76d`, product_features.csv `87fc25fd0b26daf05b48940557f802eb4a6156df32e86f56a404b29b2a87c2dd`.
- Counts. Before the history filter: 261 events, 78 clusters (45 d), 115 (21 d), as 2.3. After it (the study set): 242 events, 76 clusters; 239 with a complete window (74 clusters); out-of-sample 2000–2026: 159 events, 45 clusters (156 with a complete window); US 9 (none lost). Blocks as 6.1 (28/10, 29/7, 31/10, 71/18). Product episodes: 180 from 1965 (= from 1970), 115 from 2000; 184 on all dates. Stop-the-build checks: passed (239 ≥ 200, 74 ≥ 60, no event misses more than 2 features, leakage tests pass).
- Missing (study set): `fx_shock` 5.4%, `glob_breadth` 3.7%, nothing else.

**Implementation notes (fixed before the freeze; the most literal reading where the text is open).**
1. Data end: the 21 index files and FRED series are cut at 2026-11-11 (last date common to the 21 files). The run uses the same cut, so the event list and the 3 open events (TWII 2026-08-05, N225 2026-08-13, TWII 2026-11-11) stay as frozen. Open = 126-day window incomplete; also open if the exit day x+1 is past the data end.
2. The event state machine runs causally: whether the previous event of a market is still open at close k uses closes up to k only. No end date, reason or return is stored before step 7. Product-episode entries come from the same causal scan with the base gate; it equals `run_s.shock_episodes` (tested on synthetic paths; asserted on real data in the run).
3. US events before 1965 (8) are kept in events.csv (`us_early`); they count in `attempt` and in the re-trigger memory, not in the study. Cluster ids of the study are recomputed after the history filter (the 78 check uses the 1965+ list before the filter).
4. σ21 as `run_vol.sigma` (simple returns, ddof 1). `vol_level`, `vol_rel`: median of the σ21 values available in the 1,260 closes ending h (≥ 252 required; the filter gives ≥ 736). `attempt`: earlier events of the market from the state machine, entry within the 504 closes before t.
5. `glob_breadth`, `rel_depth`: drawdown = close / highest of the 252 closes ending it − 1; another market is live if it has 252 closes on or before t − 1 business day (pandas `BDay`). Breadth needs ≥ 3 live markets, else missing. `rel_depth`: local drawdown read at the same t − 1 date, minus the median of the live others.
6. `fx_shock`: value on or before h and on or before t − 1; missing if the series starts after h. Euro member with h before 1999 and t − 1 in 1999 (ATX 1999-05-31, AEX 1999-07-19, GDAXI 1999-07-20): krone change h → 1998-12-31 plus euro change 1999-01-04 → t − 1; IBEX in that case is missing. SXXP and STOXX50E use krone/euro. US: −mean of the log changes of JPY, CHF, GBP (1/DEXUSUK), CAD per USD. The monthly FRED_EX* series are not used.
7. `curve_min`: lowest of the last 252 observations of DGS10 − DTB3 (common dates) up to t − 1. `d_credit`: start = the calendar month before h's month: its monthly BAA − AAA value, or the mean of daily DBAA − DAAA in that month when daily data cover it (h from 1986-02); end = daily value at t − 1, or (monthly events) the last month usable at t − 1 (month M from the first business day of M + 1).
8. `region`: North America is the reference; 3 dummies, so M-G sees 17 columns.
9. M-L: training median → winsorise 1st/99th → standardise (unweighted training statistics) → `LogisticRegression(C=0.5)` with `sample_weight` = cluster weights as written (not normalised). M-G: the listed xgboost settings with the same weights, `n_jobs=1`; with weights summing to the number of clusters, `min_child_weight=3` allows few splits — kept as specified. One-feature benchmarks = the M-L pipeline on that feature alone. π = cluster-weighted stop share of the block's training events.
10. Costs: the extra weight is traded in (t+1) and out (x+1): cost = 2 × 10 bp × |Δw| per event. Δw_c, R_c and cost_c are cluster means of the events; K = mean over clusters of (Δw_c − block mean of Δw_c) × R_c − cost_c.
11. AUC: the gate level (≥ 0.60) is the event-level AUC with cluster weights. The permutation shuffles cluster-mean p within blocks, so its comparator is the AUC with cluster-mean p (like for like). K and AUC: 5,000 draws, `default_rng(7)` each. "Failed clusters" (too-few rule) = sum over clusters of the cluster's stop share.
12. G5: K recomputed on each subset; "without 2020" drops clusters with any entry in 2020; "without 2008–2009" drops clusters with any entry in those years; regions: K on each region's events. G6: AUC > 0.5, its permutation p ≤ 0.05 and K > 0 on the 13 clean markets' events.
13. G4: panel prices (BVSP from 1995, SXXP = `BBG_SXXP`); base = plain trend rule states as `run_l.run_market` × live vol size; desired = max(state × v, S weight), 5-pt rounding, `core_slide_data.act_on_change` (10 pts or to/from cash), 10 bp, cash 0%; episodes from `run_s.shock_episodes`; an episode with entry in a block is scored by that block's model, earlier ones get the 11d size; Sharpe and worst fall on 2000-01-01 → data end; "positive" = > 0.
14. G7: the core rebuilt as `core_slide_data` row 6 (inputs cut at the data end), S sized by the block model for episodes with entry in a block; features of these episodes computed on BBG_SPX closes with the same code.
15. Consequences, in this order: too few → no change; G1 permutation p > 0.10 → stop; all pass → adopt; G1, G2, G3, G6 pass, G7 and G8 pass, and the only failures are G4's worst-fall leg and/or G5 → defensive half; G1 at 0.01 < p ≤ 0.10, or G3 fails with G1 and G2 passed → report only; G8 failing turns adopt or defensive half into "adopt nothing, one-feature finding reported"; anything else → no change, recorded.
16. M-G replaces M-L only if both pass all gates and the cluster bootstrap (5,000 draws, `default_rng(7)`) gives P(M-G log-loss ≥ M-L log-loss) < 0.10. Report-only lines: LOYO with M-L; M-L coefficients with 1,000 cluster-bootstrap sign shares (`default_rng(7)`); US 1929–1954 events scored by M-L fitted on all labelled study events; the extended model (M-G + local 10-year change h → t and local 10y − 3m, monthly lagged one month; 6-month CLI change lagged 3 months; FRED OECD series by country, missing where none).
17. `run_m.py` needs `--one-run`; it refuses if `RESULTS_ml.md` exists, if the SHA-256 of features.csv differs from this section, or if the events or features recomputed from the store differ from the frozen files. `--dry-run` runs these checks and stops before any label.
