# Closing the V-shape rebound gap — research memo (Fable 5.2, 2026-11-17)

US data only. STOXX 600 was not read: it stays the holdout. Nothing here is a pre-registered test.
Code: `vgap_study.py` (run: `python -m research.us_gauge.vgap_study <folder with VXVCLS.csv, cboe_PUT.csv>`). Tables: `out/vgap_*.csv`.

## Bottom line

1. **Most of the gap cannot be closed by a rule.** Half of it is lost in the first week after the low. In that
   week, lows that later failed look the same as lows that became a V.
2. **The simple ideas do not work on 98 years of data.** Distance below the 200-day average, new N-day highs,
   fast momentum, VIX curve, a smarter volatility estimate, a put-write cash leg: none has an edge. Details below.
3. **One cheap fix exists inside the current product.** Rule S is implemented at a size frozen on the entry day
   (16% of the sleeve in March 2020). It was tested in Europe at full weight. Sizing it with the live volatility
   target, or at full weight, recovers part of the gap. Evidence: 3 US episodes since 1993, so it is a design
   correction, not a discovery.
4. **One untested information class remains: market breadth** (breadth thrusts). It needs a Bloomberg download.

## 1. The gap, measured

Product = V2+F+S timing and size weights (the sector layer does not time). K1 numbers from `out/s_combo.csv`.

| Rebound window | Buy & hold | K1+F+S | Captured |
|---|---:|---:|---:|
| 2009, 10 Mar-31 Dec | +68.3% | +20.1% | 29% |
| COVID, 24 Mar-31 Aug 2020 | +57.8% | +15.2% | 26% |
| Liberation Day, 9 Apr-30 Jun 2025 | +24.9% | +8.8% | 35% |

Where it is lost. The 10 V-shaped lows since 1993 where R1 was in cash at the low
(1998-08, 1998-10, 1999, 2002-10, 2009, 2016, 2018, 2020, 2022-10, 2025), mean values:

| Trading days after the low | Buy & hold | Product | Product weight | Share of the gap |
|---|---:|---:|---:|---:|
| 1-5 | +8.4% | +0.3% | 4% | 46% |
| 6-10 | +3.2% | +0.2% | 5% | 17% |
| 11-21 | +2.7% | +0.5% | 11% | 12% |
| 22-63 | +8.3% | +3.8% | 45% | 25% |
| 1-63 | +24.5% | +4.9% | 33% | |

- R1 came back 20 to 82 trading days after these lows (median about 48). The monthly decision and DMAS >= 45 set that.
- After re-entry the volatility size is still about 0.5, so days 22-63 are also half missed.
- All 27 events since 1993 are in `out/vgap_capture.csv`.

Why the first week is not available. Mean buy-and-hold return after the low, all 41 events 1955+:

| Class | n | 5 days | 10 days | 21 days | 63 days |
|---|---:|---:|---:|---:|---:|
| V | 17 | +7.3% | +9.6% | +12.1% | +20.9% |
| Failed (low undercut within a year) | 18 | +5.7% | +7.3% | +8.1% | +6.0% |

The two only separate after day 21. A rule that buys in week one buys both.

## 2. What the literature and practitioners say

| Idea | Claim | Sample and evidence | Published / crowded |
|---|---|---|---|
| Breadth thrust (Zweig 1986; Deemer; Whaley 2009) | A burst of advancing stocks within 10 days marks a durable low | About 20 Zweig signals since 1945, said to have no losing 12-month outcome. Vendors disagree on the list; some count invalid signals in 1974 and 2008. One S&P test since 2007: 53% winners. No audited out-of-sample study found | Yes, widely watched |
| Lowry 90% up-volume days | Two 90% up days after 90% down days end a panic | Vendor studies only; no count or failure table found | Yes |
| Below the 200-day average / oversold | Deep discount reverts | Tested here (section 3): no usable edge | Folk rule |
| VIX curve back to contango (VIX < VIX3M) | Forced hedging has ended | No rigorous study found; vendor heuristics. Tested here: no edge, fails in 2008 | Yes |
| Slow + fast trend (Goulding, Harvey, Mazzoleni, JFE 2023) | Blending 12-month and 1-month momentum handles turning points; "rebound" state = slow negative, fast positive | US 1969-2018 monthly; a 50/50 blend had the best Sharpe (0.51). Later notes warn the rebound subsample is small. Tested here: rebound months earn nothing extra | Yes, used by CTAs |
| Volatility targeting (Moreira-Muir 2017; Harvey et al. 2018) | Scaling by past vol raises equity Sharpe and cuts the left tail | Long samples, many assets. Cederburg et al.: no significant out-of-sample gain across 104 strategies | Yes, about $400 bn by one 2020 estimate |
| Put selling / variance premium (Bollerslev-Tauchen-Zhou 2009; CBOE PUT) | Insurance is overpriced, most after spikes | PUT index 1986-2018: Sharpe 0.65 vs 0.49 for the S&P, worst fall -33%. Tested here as a cash leg: hurts | Yes |

How the large systematic players actually do in V rebounds:
- **Trend followers (CTAs) do not win the rebound.** They win the fall (SG Trend +2.1% in Q1 2020 with the S&P -20%)
  and give back at the turn: trend followers -2.7% in May 2020 "as markets reverted"; SG Trend about -4% to -5% in April 2025,
  with "long exposures stopped out the prior week, so the rebound was largely missed". Faster models did better in 2020.
- **Vol-control funds are the product's twin.** They sell within one to three days and re-lever over weeks as realised
  volatility falls; that buying helped the second leg in April-May 2020. They also miss the first leg.
- **Who earns the first week:** liquidity providers, option sellers and discretionary buyers with spare cash. None of
  that is a timing rule on an index.

Client story that survives the evidence: "We do not try to catch the first week; no rule can, because a bear-market
bounce looks the same. We cut the fall, and we come back in the second leg when the shock type and volatility allow it."

## 3. Long-history tests (S&P 500 daily, 1928-2026; EODHD `GSPC.INDX`)

### 3a. Distance below the 200-day average — the user's example

Forward 63-day price return by bucket, all days 1928-2026 (unconditional: +2.0%, 64% positive).

| Close vs 200-day avg | Days | Episodes | Mean 63d | Positive | 5% worst case |
|---|---:|---:|---:|---:|---:|
| more than 5% above | 10,767 | 46 | +2.6% | 69% | -9% |
| 0 to 5% above | 5,995 | 83 | +2.0% | 66% | -11% |
| 0 to 5% below | 3,593 | 84 | +0.4% | 51% | -15% |
| 5 to 10% below | 2,245 | 52 | +0.7% | 53% | -21% |
| 10 to 15% below | 900 | 39 | +2.1% | 64% | -24% |
| 15 to 20% below | 483 | 24 | +4.4% | 70% | -19% |
| 20 to 30% below | 464 | 15 | -1.9% | 49% | -37% |
| more than 30% below | 187 | 6 | +12.3% | 49% | -29% |

- No bucket below the average beats being above it on a risk-adjusted basis. The left tail is two to four times fatter.
- The 15-20% bucket looks good, but it is 24 independent episodes; measured from the first day of each: +1.4%.
- First close 15% below the 200-day (20 episodes): +4.3% after 63 days, 65% positive. Bad cases: 1937 -24%,
  1974 -23%, Sept 2008 -21% (-39% at worst), July 2002 -13%. Good: 1933, 1987, 2020.
- By era the sign is not stable (1928-54: 20-30% below gives -4.8%).

### 3b. New N-day closing high after a 15%+ decline low (fast-trend re-entry)

One signal per new low. "Trap" = the low is broken within 126 days.

| N | Signals | Days after low | Gain already made | Mean 63d after | Trap share |
|---|---:|---:|---:|---:|---:|
| 10 | 78 | 6 | +7.8% | +0.9% | 62% |
| 21 | 61 | 14 | +10.8% | 0.0% | 51% |
| 42 | 46 | 26 | +13.8% | +0.7% | 35% |
| 63 | 41 | 44 | +16.5% | +1.6% | 27% |

Faster means more traps; slower means the rebound is spent. No N gives a forward return above the unconditional +2.0%.

### 3c. Fast momentum inside the cash state

- Slow/fast states, monthly 1929-2026 (1,174 months). Next-month return: Bull +0.8% (t 4.8), Correction +0.8% (t 2.7),
  Bear +0.2% (t 0.4), **Rebound +0.3% (t 0.5)**. A positive last month in a down-trend carries no information.
- Months that start with DMAS < 45, 1955+ (275 months). Next month: +0.2% if the last 21 days were up, +0.4% if not.
  Close above the 50-day average: -0.3% against +0.6%. Same in both halves. (Invested months: +0.8%.)
- This explains why X (DMAS >= 60) and F did not beat their placebos.

### 3d. Volatility estimate used for sizing

- In the 42 days after a 15%+ decline low (3,710 days), trailing 21-day volatility overstates the next 21 days by about
  14% (median ratio 0.86). Blends with 5-day and 1-year vol forecast better (QLIKE 0.24 against 0.31).
- But vol-sized buy-and-hold has the same Sharpe with every estimator: 21-day 0.48, 10-day 0.48, HAR-style 0.48,
  21-day/1-year blend 0.50 (1929-2026, price only). Sizing on VIX or min(21-day, VIX): 0.50 and 0.56 against 0.57 (1990+).
- Size 21 days after the 2020 low: 0.26 (21-day), 0.35 (blend), 0.36 (VIX). The gain is a few points of weight.

### 3e. VIX / VIX3M back below 1 (2007-12+, FRED VXVCLS)

54 crossings after 5+ inverted days: +1.9% after 63 days, against +2.6% unconditional. Fired in Feb-Mar, Jul-Aug and
Dec 2008 (-14% to -33%). Good in 2009-03, 2011-10, 2019-01, 2020-04, 2025-04. About 15 independent episodes.

### 3f. Put-write as the cash leg (CBOE PUT index, daily 2007-07+)

Holding 50% of the sleeve in PUT while R1 is in cash: Sharpe 0.72 against 0.82; worst fall -23% against -18%;
2008 crisis -22% against -5%. COVID rebound +21.9% against +10.9%. It buys the rebound by giving up the protection.

## 4. Candidates (two, not three)

### S-live — size the shock re-entry as volatility normalises (pre-register first)

- **Definition.** Rule S unchanged (trigger, stop, 126-day limit, end when base Invested). Only the size changes:
  on each day of an S episode the S weight is the V2 volatility target of that day, min(1, 15%/sigma 21d),
  decided at t, effective t+2, 10 bp, no trading bands. Today it is the target frozen on the entry day.
- **Why.** The frozen size has no rationale: V2 updates everywhere else, and trailing vol overstates the next month
  after a low. Europe validated S at full weight, not at 16%.
- **Effect in the US (information only, 3 episodes: 1998, 2020, 2025).**

| V2+F plus | COVID rebound | Liberation Day rebound | Sharpe 1993+ | Worst fall |
|---|---:|---:|---:|---:|
| no S | +10.9% | +7.9% | 0.755 | -25.5% |
| S frozen (today) | +13.8% | +8.2% | 0.767 | -25.5% |
| S live | +17.2% | +8.8% | 0.782 | -25.5% |
| S full weight | +31.0% | +12.4% | 0.805 | -25.5% |

- **Full weight** is a risk-budget decision for the user, not a statistical one: it means holding equities at
  40-60% volatility for some weeks. Report it next to S-live.
- **Triggers.** 3 on the R1 base since 1993; 7 on R0 since 1955. Fewer than 10: say so in the spec.
- **Falsification (Europe, one run).** Vol-sized R0 base plus S: S-live must have Sharpe >= S-frozen and worst fall
  no more than 2 points worse. If it fails, keep the frozen size.

### B — breadth thrust (needs data; pre-register before looking at it)

- **Definition, taken from the published rules.** Either (a) Zweig: the 10-day exponential average of NYSE
  advances / (advances + declines) rises from below 0.40 to above 0.615 within 10 trading days; or (b) Deemer:
  10-day sum of advances / 10-day sum of declines >= 1.97.
- **Action.** If the base rule is in Cash on the next Tuesday decision: equity weight = live V2 target. Ends when
  the base is Invested, on a close below the lowest close of the 63 days before entry, or after 126 trading days.
- **Why it could work.** It is the only idea here that uses information not in the index price: near-unanimous
  buying across thousands of stocks is what forced re-leveraging looks like.
- **Expected triggers.** About 20-30 thrusts since 1950 in the literature; perhaps 10-15 while in Cash.
  It is 40 years old and widely watched. My odds that it beats a random-date placebo: about 1 in 4.
- **Falsification.** US 1955+, R0 base, the S gates (placebo, weekday mean, bear-window drawdowns, without 2020/2025).
  A Europe check needs STOXX 600 advancers/decliners; if not available, the US split halves are the only check.

### Not worth a test (evidence above)

Distance below the 200-day average; N-day highs; fast momentum or 50-day cross inside the cash state; VIX curve;
a different volatility estimator; put-write cash leg; and the earlier C, P, X.

## 5. Bloomberg data to download (daily, PX_LAST, Max range)

- NYSE advancing and declining issues, and up and down volume. I am not sure of the tickers: search
  "NYSE advancing issues" in `SECF <GO>`. Needed for B (and Lowry 90% days). History should reach the 1950s-60s.
- `PUT Index` (CBOE PutWrite, from 1986) — only if the put-write idea is to be re-checked before 2007.
- `VIX3M Index` before 2009-09 is not needed: FRED VXVCLS is free from 2007-12.
- Sentiment (`AAIIBULL Index`, `AAIIBEAR Index`, weekly from 1987; `PCUSEQTR Index`): not requested. Published,
  noisy, and the VIX-based tests already failed.

## Sources

- Goulding, Harvey, Mazzoleni, "Momentum turning points", JFE 2023: https://econpapers.repec.org/RePEc:eee:jfinec:v:149:y:2023:i:3:p:378-406 ; summary https://www.cxoadvisory.com/momentum-investing/handling-reversals-in-price-trend-direction/
- Zweig breadth thrust record and caveats: https://www.thetrading.tools/concepts/zweig-breadth-thrust ; https://sentimentrader.com/blog/zweig-breadth-thrust-recovery ; https://stockcharts.com/articles/arthurhill/2023/11/the-zweig-breadth-thrust-trigg-609.html
- Deemer / Whaley thrusts: https://www.luxalgo.com/library/concept/breadth-thrusts/ ; https://www.sentimentrader.com/blog/another-thrust-narrowly-missed
- CTAs in April 2025: https://www.alternativeswatch.com/2025/06/02/trend-following-hedge-funds-endure-more-pain-societe-generale-indices/ ; https://www.toptradersunplugged.com/trend-following-week-in-review-april-11-2025/ ; https://advisoranalyst.com/?p=167192
- CTAs in spring 2020: https://hedgeweek.com/news/cta-hedge-funds-retreat-further-following-recent-may-losses ; https://www.hedgeweek.com/short-term-trend-following-hedge-funds-continue-thrive-other-ctas-slip-red/
- Vol-control flows: https://ECB.eu/pub/financial-stability/fsr/focus/2020/html/ecb.fsrbox202005_02~f6616db9be.en.html ; https://convextrade.com/glossary/vol-control-strategy ; https://www.bloomberg.com/news/articles/2020-01-17/volatility-targeting-funds-leverage-up-at-fastest-since-18-rout
- Volatility targeting: Harvey et al. https://people.duke.edu/~charvey/Research/Published_Papers/P135_The_impact_of.pdf ; Cederburg et al. summary https://alphaarchitect.com/?p=59058
- VIX term structure (practitioner): https://www.thetrading.tools/manuals/vix-term-structure
- PUT index: https://www.cboe.com/insights/posts/white-paper-shows-volatility-risk-premium-facilitated-higher-risk-adjusted-returns-for-put-index ; https://www.wikipedia.org/wiki/CBOE_S%26P_500_PutWrite_Index

Literature figures come from search summaries, not from the papers themselves; check before quoting them to a client.
