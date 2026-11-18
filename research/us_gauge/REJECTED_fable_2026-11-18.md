# Rejected tools under the "US-first" principle — review (Fable 5.2, 2026-11-18)

Review memo, not a test. Sources: spec `2026-11-16-us-quant-gauge-design.md` (6–11c), `2026-11-17-us-sleeve-tilts-design.md`,
every `RESULTS_*.md`, the four Fable memos, `STOPS_s`, and the branch log. One new calculation (section 3) is
**in-sample**: same 1999–2026 history as the core. Script (scratch, read-only):
`/private/tmp/claude-501/-Users-larazanella-Desktop-GitHub-Projects-ic-technical/abe5825f-ac19-4130-a032-71fdd52b333c/scratchpad/rejected/mt.py`.

## 0. Bottom line

1. **"US-first" rescues almost nothing.** Of 47 rejected or not-adopted items, 34 were rejected on **US data**
   (US gate, US placebo) or on product grounds, 8 were never run and 3 were superseded. Only one family failed on non-US data with
   good US numbers: **L (S and B at 100%)**, and the user has already adopted it (11c).
2. **The principle is fair, but it is not evidence.** It can say "a fail in Korea does not kill a US rule". It
   cannot turn a rule found on US data into a rule proven on US data. Other markets were our only fresh sample.
   Without them, each new US-only adoption has no holdout except the live record.
3. **Multiple testing is now the main risk.** About 27 decision-relevant candidates and about 150 looks at one
   US history. The core's edge over the S&P (+0.31 Sharpe, SE 0.13) is about what the best of 20–25 independent
   noise products would show. Plan on a forward Sharpe of 0.5–0.6, not 0.73.
4. **11c:** B at 100% is well supported on US data (it was pre-registered and tested at 100%). S at 100% is not
   supported by US data alone: the US edge is 2020 and 2025, the two events that motivated the rule. Keep 11c,
   but add three safeguards (section 5).
5. **Retests:** one is worth doing — the two legs of L on US data never used for them (S on 1928–1954,
   B on Deemer's 1945–1964 dates). Nothing else in the rejected list deserves a US retest.

## 1. Inventory

Class: **(a)** failed on non-US data, US evidence good · **(b)** failed on US data (gate, out-of-sample half,
or placebo) · **(c)** product choice · **n/t** never tested. Sharpe numbers are US. "R0" = DMAS alone,
"R1" = two keys, "V2" = R1 x volatility size.

### Timing keys and execution

| # | Tool | What it was | Why not adopted (US evidence) | Class |
|---|---|---|---|---|
| 1 | R2 | Trend exits, earnings re-entry | 1993–2026: Sharpe 0.563 vs R0 0.603; fails all 6 conditions; bootstrap 0.84 | b |
| 2 | R3 | Earnings revisions alone | 0.642 vs 0.603 full, but H2 0.746 < 0.755; worst fall −29.9% vs −24.0%; bootstrap 0.36 | b |
| 3 | R4 | R1 + VIX 21-day avg ≥ 25 | 0.624 vs R1 0.706; worse in both halves; bootstrap 0.92 | b |
| 4 | R5 | CoT hedge-fund key, 2-of-3 vote | 2008–26: 0.811 vs R1 0.831; fails H1, bootstrap 0.59, more switches | b |
| 5 | R1 "half" | Cash = 50% equity | 0.645 vs "all" 0.706, worst fall −36.9% | b / c |
| 6 | REV_z threshold −0.5 | Looser earnings key (info) | 0.618 vs 0.706; worst fall −45.6% | b |
| 7 | REV window 126 d | Slower revisions (info) | 0.639 vs 0.706; worst fall −37.8% | b |
| 8 | Keys daily, no month lock | Unlocked switch (info) | 1999–2026: 0.58 vs core 0.73; worst fall −34.7% | b |
| 9 | Keys on Tuesdays, act next day | Earlier rule (11a) | 0.75 / −25.9% vs core 0.73 / −20.1%; dropped for the worst fall | c |
| 10 | Tuesday keys, wait for Wednesday | Committee rule | 0.77 / −26.7% (in-sample best row); dropped by the user | c |
| 11 | Sizes not rounded | No 5-pt steps | 0.74 / −19.9% vs 0.73 / −20.1%; rounding kept for practice | c |
| 12 | Plain 200-day rule | Benchmark (Fable review) | 0.56 vs R1 0.71 (1993–2026) | b |
| 13 | Machine learning on pillars | Spec approach C | Walk-forward 0.46 / 0.43 vs DMAS 0.57 (before this project) | b |
| 14 | Pillar composite, credit and macro pillars | Spec approach B, section 8 | Never reached | n/t |

### Sizing

| # | Tool | What it was | Why not adopted | Class |
|---|---|---|---|---|
| 15 | V1 | Vol size on buy & hold | PASS (1955–2026: 0.496 vs 0.461; bootstrap 0.17). Lives inside V2 | absorbed |
| 16 | Other vol estimators | 10-day, HAR, blends, VIX, min(21d, VIX) | 1929–2026: Sharpe 0.48–0.50, same as 21-day. 1990+: 0.50–0.56 vs 0.57 | b |
| 17 | No vol size anywhere | Layers at 100%, keys at 100% | 0.71 vs 0.73; worst fall −28.2% vs −20.1% | b / c |

### What the sleeve holds

| # | Tool | What it was | Why not adopted | Class |
|---|---|---|---|---|
| 18 | E1 | RSP vs SPY on 6-month relative momentum | 2003–26: 0.514 vs SPY 0.552; fails H2; bootstrap 0.90 | b |
| 19 | S1 | C5 sector tilt, Ken French industries | PASS (0.601 vs 0.553), but not independent of C5's own test | see 21 |
| 20 | S2 | C5 on SPDR ETFs, 70/30 | PASS (0.545 vs 0.498). Not fresh: same ETFs already run on 2026-10-21 | see 21 |
| 21 | K1 | R1 x vol size x C5 sectors | PASS on gates (0.671 vs V2-SPY 0.637). Rejected: **US placebo** — a constant XLP/XLU/XLE tilt gives 0.679; all 84 static triples average 0.648. Core: 27-year return unchanged (+1266% vs +1261%), −14 pts in 2014–18. Moved to a Delphi satellite | b + c |
| 22 | M1 | C5 without its intercept (macro only) | M1a test-window IC +0.006, p 0.40 → FAIL. ETF leg passes (0.537 vs 0.498) | b |
| 23 | Sleeve split 80/20, 50/50 | Robustness (info) | Sharpe 0.73 for every split; 2014–18 Sharpe falls to 0.45 / 0.21 | c |

### Re-entry layers

| # | Tool | What it was | Why not adopted | Class |
|---|---|---|---|---|
| 24 | C | 50% floor after a −3 sigma month | 1955–2026: 0.468 vs R0 0.514; 16 of 39 episodes; worse than 99% of random entries. 8 threshold/floor variants all below R0 | b |
| 25 | P | 50% floor when HY spread is 25% off peak and VIX < 30 | 2000–26: 0.568 vs 0.609; 3 of 7 episodes; placebo share 0.97; 1 of 13 gates. 17 variants: best +0.035 with 2 episodes | b |
| 26 | X | Re-enter on any Tuesday if DMAS ≥ 60 | 1955–2026: 0.548 vs 0.514, 21 of 30. **Fails placebo** (p 0.45), 2000–03 and 2022 drawdowns, and the product (V2+X 0.728 < V2 0.732). 55 / 65: same | b |
| 27 | H | HTEI US score < 35 while in cash | 2010–26, mostly in-sample for HTEI: 0.838 vs 0.885; placebo p 0.57; 14 of 28; 2018–26 0.66 vs 0.86 | b |
| 28 | HTEI components (7) | Vol level, vol curve, sentiment, valuation, FCI, earnings, positioning | Reviewed in VGAP2, not run: levels, not turns; VIX curve repeats S and B | n/t |
| 29 | B, 5-day window | Deemer ratio on 5 days | 51 episodes, 0.418 vs R0 0.438; 3 of 12 gates; 2008 worst fall −56% | b |
| 30 | S-frozen | S at the entry-day vol size | Replaced by S-live (Europe test: 0.514 vs 0.477) | superseded |
| 31 | S and B at live vol size (11b) | 15% / 21-day vol during an episode | Replaced by 11c. US in-sample: 0.69 vs 0.73 | superseded |
| 32 | S and B at max(live, 50%) | Pre-declared fallback of 6L | Not triggered. US in-sample 0.70 | a (minor) |
| 33 | **L** | S at 100% vs live, six fresh indices | **FAIL 1 of 5 gates**: mean Sharpe gain −0.005, 3 of 6 markets, placebo p 0.40, worst episode −21.6% (HSI 1997). US in-sample: 0.73 vs 0.69. **Adopted anyway (11c)** | **a** |
| 34 | Zweig thrust | 10-day EMA breadth thrust | Dates reported only; original source not read | n/t |
| 35 | B thresholds 1.90 / 2.05 | Sensitivities named in 6t | Not in RESULTS_b | n/t |
| 36 | B20 | Deemer 20-day thrust (1.72) | Proposed in VGAP2; not run | n/t |
| 37 | T / D | New low with less fear | Proposed twice; not run. Fable odds 1 in 5 | n/t |

### Descriptive US studies (no rule, no gate; all 1928/1955–2026)

| # | Tool | Result | Class |
|---|---|---|---|
| 38 | Distance below the 200-day average (8 buckets, 3 first-cross rules) | No bucket beats being above; sign not stable by era | b |
| 39 | New N-day high after a 15% fall (N = 10, 21, 42, 63) | Forward 63-day return 0.0–1.6% vs +2.0% unconditional; 27–62% traps | b |
| 40 | Fast momentum / 50-day cross inside the cash state | +0.2% vs +0.4% next month; no information | b |
| 41 | Slow + fast trend states (Goulding–Harvey) | "Rebound" months: +0.3% (t 0.5); true only in 1955–89 | b |
| 42 | VIX / VIX3M back below 1 (2007+) | +1.9% vs +2.6% unconditional; failed three times in 2008 | b |
| 43 | Put-write as the cash leg (2007+) | 0.72 vs 0.82; 2008: −22% vs −5% | b |
| 44 | V-shape features: Fed cuts, Fed balance sheet, M2, Sahm, recession, EPS collapse, VIX ratio, VIX < 30, HY spread off peak, P/E, speed | None separates V from failed lows (41 events, about 30 features tried) | b |
| 45 | Fed text sentiment (Bloomberg index) | History from about 2009; cannot be validated | c (data) |
| 46 | Ideas 2–6 of IDEAS memo: recession filter on the exit, Treasuries as cash leg, US–Europe tilt, variance premium, put/call | Never run | n/t |
| 47 | Lower S threshold (+10%), same-day execution | Argued to be worth zero; not run | n/t |

Count: 47 items (about 115 variants with the sensitivity rows). (a) 2 · (b) 29 · (c) 5 · superseded/absorbed 3 · never tested 8.

### Adopted against a failed gate (the mirror list — say it on any slide)

| Layer | Its own gate | What supports it |
|---|---|---|
| R1 (keys) | FAIL on worst fall (−35.1% vs −24.0%); the gate was changed to Sharpe after the result | 0.706 vs 0.603; bootstrap 0.13. Europe check required by the review was **never run** |
| F (VIX floor 50%) | FAIL: placebo p 0.23 (random re-entry in the cash spell did as well) | 12 of 13 gates; 7 of 7 episodes; small size |
| S (shock) | US: fails H1 (1955–90: 0.438 vs 0.464) and bootstrap (0.12). Found on US data | **Europe** one-run PASS (0.520 vs 0.438, placebo p 0.011) — a non-US result |
| S, B at 100% | 6L FAIL | US in-sample; B's own test was at 100% |

## 2. What "US-first" changes in the classification

- **(a) is almost empty.** Every re-entry idea that died (C, P, X, H), every key (R2–R5), E1, M1 and the K1
  sector layer died on US data. The Europe check was written into the spec for R2, C, P, X, but none of them
  got that far. US-first gives no reason to reopen them.
- **L is the only real case**, and inside it the six markets do not all say the same thing. S at 100% vs live:
  Japan +0.008, Canada +0.013, Australia +0.012; Hong Kong −0.044, Korea −0.012, Taiwan −0.008. That split
  (developed vs Asian emerging) fits the user's argument, but **I saw it after the run**. It is a reading, not a test.
- **Careful with symmetry.** S's only clean out-of-sample pass is Europe. If non-US fails do not count, non-US
  passes should count less too. Honest position: Europe and Canada are close to the US in structure and both
  lean the user's way; say in advance which markets are "close", or the principle cannot be wrong.
- **What US-first quietly drops:** the Europe test of the keys (R1 + V2) that the 2026-11-17 review made a
  condition. The keys carry half of the edge and have the weakest statistics (section 3).

## 3. Multiple-testing cost

**How many tries.** Formal candidates on US history: R1–R5, E1, S1, V1, V2, S2, K1, M1a, M1b, C, P, F, X, S,
S-live, B, H, L = 22. Execution choices made after seeing results (Tuesday / daily / lock / Wednesday /
rounding) = 5. Report-only sensitivity rows ≈ 75. Descriptive looks (V-shape features, VGAP studies) ≈ 55.
**About 27 decision-relevant, about 150 looks in total**, all on S&P 1955/1965/1993–2026.

**How big is luck with our sample (in-sample calculation, 1999-02 → 2026-11-13, 63-day block bootstrap, 2,000 draws).**

| Step | Sharpe gain | Bootstrap SE | z | Correlation with previous |
|---|---:|---:|---:|---:|
| Keys vs buy & hold | +0.159 | 0.137 | 1.2 | 0.67 |
| + volatility size | +0.040 | 0.030 | 1.3 | 0.98 |
| + F | +0.049 | 0.024 | 2.1 | 0.99 |
| + S at 100% | +0.038 | 0.029 | 1.3 | 0.97 |
| + B at 100% | +0.054 | 0.032 | 1.7 | 0.98 |
| S, B at 100% vs live size (row 6) | +0.047 | 0.025 | 1.9 | 0.97 |
| **Core (row 6) vs buy & hold** | **+0.315** | **0.126** | **2.5** | 0.69 |

Expected best of N independent noise variants = SE x E[max of N normals] (1.35 for N = 7, 1.9 for 22, 2.65 for 150):

- **Whole product vs S&P:** 0.126 x 1.9 = **+0.24** from 22 pure-noise products; +0.33 from 150. Our variants are
  correlated, so the effective N is perhaps 10–25: +0.19 to +0.25. Observed: +0.31. A single-test z of 2.5
  does not pass a 22-test Bonferroni bar (z 2.8).
- **Re-entry family:** 7 rules tried (C, P, F, X, S, B, H), SE about 0.03 each. The best three of seven noise rules
  would add about +0.07. The three adopted add +0.14. So about half of the layer gain on the product period
  is what selection alone would give.
- **Each single layer:** +0.04 to +0.05 with SE 0.03. On the product period no layer is distinguishable from the
  best of a handful of noise overlays. The product-period table is a description, not evidence.

**Where real evidence sits (not in 1999–2026):** trend following (a century of out-of-sample literature);
B (pre-registered, 1965–2026, placebo p 0.001 — survives Bonferroni); S (Europe one run, p 0.011).
Weak: keys vs S&P (z 1.2, two bears), volatility size (literature: no reliable Sharpe gain), F (failed placebo),
S and B at 100% vs live (z 1.9 in-sample, failed the fresh sample).

**Planning number.** Haircut the in-sample edge by about half: forward Sharpe 0.5–0.6 against 0.42 for the
S&P. The same applies to the worst fall: −20% is the best case seen; plan on −25% to −30%.

**What US-first can justify:** using US-specific inputs (NYSE breadth, S&P revisions, VIX); not waiting for
Korea or Hong Kong before acting in the US; treating Europe and Canada as the relevant relatives.
**What it cannot justify:** counting US in-sample numbers as proof; adopting a rule that failed its own US
gate or placebo; running more US-only searches without a holdout. The untouched US data left is small:
S&P prices 1928–1954 and Deemer's breadth dates 1945–1964. After that, only the live record is new.
**Rule I would add to the spec:** the US file is closed again; any further US-only adoption needs (i) a pass
on unseen US data or (ii) 12 months of live paper record, and placebo p ≤ 0.01 instead of 0.10.

## 4. Retests worth doing (category (a) only)

Category (a) holds one family, L. Its two legs can each be checked on US data not yet used for them.

### Rank 1 — S at 100% on S&P 500, 1928–1954 ("L-US")

- **Why the US structure matters.** S bets that a fast fall from a high is forced selling in an intact trend, and
  that deep US liquidity brings buyers back. 1929–1954 has that market without the modern Fed backstop. A pass
  says the effect is older than the policy put. A fail is partly ambiguous by the user's own argument (different
  era), so fix the reading before the run.
- **Sample.** `GSPC.INDX` closes 1927-12 → 1954-12. Used so far only for descriptive studies (200-day buckets,
  N-day highs, momentum states). No S, R0 or DMAS result has been computed on it.
- **Rule (unchanged code).** Base R0 as in 6s (DMAS on closes, cash below 45, exit below 30, Tuesday, month lock),
  first decision 252 days after the first close. Episodes: `run_s.shock_episodes`. Weight = max(base, 1).
  Equity = price change + `SHILLER_SPX_DIV`/price/252. Cash: Ken French RF if already cached, else 0% — decide
  before the run. 10 bp.
- **Gates (one run).** G1 Sharpe R0+S > R0, full and on the five-weekday mean. G2 worst fall no more than 3
  points deeper. G3 more than half of the episodes positive. G4 worst single episode ≥ −15%. G5 placebo
  (`run_s.placebo_spans`, 1,000 draws, seed 7) p ≤ 0.10. Fewer than 5 episodes = "too few to conclude".
- **Consequence, declared now.** Pass or too few → 11c stays. Fail on G2 or G4 only → S at max(live, 50%), B stays
  at 100%. Fail on G1 and G5 → S back to the live size.
- **Honest odds.** Pass about 20%. Too few to conclude about 45%. Fail about 35% (1930 and 1933 look like
  failed shocks from memory; 1929 looks like a win).

### Rank 2 — B at 100% on Deemer's published dates, 1945–1964

- **Why.** B's breadth data start in 1965. Deemer's table lists his signals from 1945. The S&P path after those
  dates has not been used here. NYSE breadth is a US-only signal, so this is the one place where US-first
  is the natural frame.
- **Rule.** Thrust day = the date in Deemer's table (no ratio recomputed). Base R0 on `GSPC.INDX`. Episode,
  stop (63-day low), 126-day limit and 100% weight as 6t.
- **Gates.** Report only: episodes, share positive, worst episode, total P&L vs R0. A gate needs ≥ 5 episodes
  (thrust while R0 is in cash); I expect 2–5.
- **Caveat.** Dates before 1973 were back-filled by the author. Small selection risk.
- **Honest odds.** Informative result about 30%. Its main value: a first look at what a **failed** B episode
  costs, which the 1965–2026 sample does not contain (0 stops in 10).

### Rank 3 — none

Nothing else in the rejected list earns a US retest. X, C, P, H, R2–R5, E1 and M1 failed on US data. K1 failed a
US placebo. B20 (item 36) was never rejected; it is the only untested US-specific idea with a prior (about 1 in
4, lower than VGAP2's 1 in 3 because 1.72 was set in 2019 after 2016). I would hold it until ranks 1–2 are
done, count it as US test no. 23, and require placebo p ≤ 0.01.

## 5. Is 11c well supported on US data alone?

- **B at 100%: yes.** The pre-registered test (6t) ran B at 100% on the plain trend rule: 14 of 14 gates, placebo
  p 0.001, both halves, positive after publication. The live size was the deviation; 11c goes back to what was tested.
  Correction to the 11c text: 9 of 10 episodes add value (one is +0.00%), not 10 of 10. No episode was stopped,
  so the loss side has never been observed; the stop sits 9–18% below entry.
- **S at 100%: not on US data alone.** 7 episodes; the first half fails (1955–90: 0.438 vs 0.464); bootstrap 0.12;
  without 2020 and 2025 the gain is +0.005 (0.524 vs 0.519). Those two events are why the rule was written.
  The real support is Europe (7 of 8, at full weight) — a non-US market. At entry, 1987 (the loss) and 2020 (the
  best win) looked the same.
- **The headline hides the cost.** Worst fall −20.1% contains no failed S or B episode. On the product base
  there are 3 S and 3 B episodes since 1998, all winners.

**Safeguards I would add (no new fitted parameter):**
1. **Publish a stress worst fall** next to −20.1%: one failed episode at 100% inside a drawdown = about −30%.
2. **Pre-declared step-down.** If two of the next four S or B episodes end on the stop, both go to the 6L
   fallback, max(live size, 50%), without a new study.
3. **Run rank 1 and rank 2 before the slide goes out**, with the consequences written above.

## 6. Caveats

- Section 3 is one more look at the same history. The SEs are bootstrap estimates; the "effective N" is a judgment.
- The developed vs emerging split of the six L markets is post-hoc.
- Episode guesses for 1928–1954 are from memory of the price history, not from a run. Nothing was computed on it.
- I did not recompute any RESULTS file; the numbers in section 1 are copied from them.
