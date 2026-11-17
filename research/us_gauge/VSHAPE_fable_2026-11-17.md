# V-shaped rebounds vs failed rallies — event study (Fable 5.2, 2026-11-17)

Descriptive study, not a backtest. Script: `vshape_study.py` (inputs: market_store + FRED CSVs fetched to a scratch folder;
full feature table was written to scratch `events.csv`, 44 columns). I am no longer blind to US equity history.

## Verdict
There is no clean "secret recipe". With 41 events and about 30 features tried, any single separator is fragile.
Two things do stand out, and one popular idea does not:
1. **What kind of fall it was matters more than what the Fed did.** A fast, deep fall straight from a 1-year high
   (a shock in an uptrend) rebounded in a V in 5 of 7 cases. Falls that start from an already-broken trend mostly did not.
2. **Fed easing does not separate V from failed.** The Fed was cutting in 3 V lows and 8 failed lows. Balance-sheet
   growth above 10% in 13 weeks: 1 V (2020) against 3 failed (Oct-Nov 2008). 2020 is one case, not a pattern.
3. Credit "peaking with equities" looks perfect in hindsight (7 of 7 V) but only 6 V / 2 failed once made real-time.

## Method (fixed before looking at features)
- Zigzag on S&P 500 daily closes, 10% reversal. Decline = swing high to swing low. **Confirm** = first close 10% above
  the low; this is the earliest real-time date, and all "after" returns are measured from it.
- Kept: declines of 15% or more since 1955; 10-15% since 1990. 41 closed events.
- Class: **FAILED** = low undercut within 252 trading days (18). **V** = not undercut and at least half the loss
  regained 63 days after the low (17). **SLOW** = the rest (6).
- Note: 1987 counts as FAILED (the crash low was undercut by 0.4% on 4 Dec 1987); 2015-08 and 2018-02 are marginal
  retests too. A 10% zigzag splits long bears into legs (2000-02: four; 2008-09: five).
- Base rate from confirm, all 41 events: +2.3% after 3 months, +5.3% after 6 months.

## What separates them (3-month return after confirm; V / FAILED counts)

| Feature (known at confirm unless noted) | Yes: n, V/F, 3m | No: n, V/F, 3m | Comment |
|---|---|---|---|
| Decline 15%+ and low within 63 days of a 1-year high | 7, 5/1, +11.7% | 34, 12/17, +0.3% | 1980, 1987 (F), 1990 (slow), 1998, 2010, 2020, 2025. Welch p 0.02, before any multiple-test penalty |
| Decline 15%+ and 200-day average still rising at the low | 7, 6/1, +10.4% | 34, 11/17, +0.6% | Same idea, almost the same events (2011 in, 1990 out) |
| Low within 63 days of a 1-year high (any depth) | 13, 8/4, +7.0% | 28, 9/14, 0.0% | 10-15% dips add noise: 2000-04, 2015, 2018-02, 2022-03 failed |
| 200-day average rising at the low (any depth) | 16, 10/6, +4.4% | 25, 7/12, +0.9% | Weak alone |
| VIX peaked 10+ days before the price low ("less fear at the retest") | 6, 4/1, +9.5% | 35, 13/17, +1.0% | 1990, 2003, 2009, 2010, 2011; fail 2022-06. Final lows of longer declines |
| 2-year yield fell 50 bp+ from equity peak to confirm (1976+) | 12, 6/5, +7.5% | 29, 11/13, +0.1% | Market-implied Fed easing. Return gap, but the class split is poor: 2001 and 2008-03 fail |
| Fed funds cut 25 bp+ from peak to confirm | 13, 3/8, +2.5% | 28, 14/10, +2.2% | **No separation.** The Fed cuts in recessions |
| Fed balance sheet +10% in 13 weeks (2003+) | 4, 1/3, -2.9% | 37, 16/15, +2.8% | **No.** Oct-Nov 2008 had +77% to +143% and fell further |
| M2 12-month growth 7%+ | 14, 5/8, +1.2% | 27, 12/10, +2.8% | No |
| Unemployment rising (Sahm 0.5+) | 14, 3/7, +2.5% | 27, 14/11, +2.1% | Fewer Vs, but no difference in returns |
| NBER recession at the low (hindsight) | 16, 3/9, +2.5% | 25, 14/9, +2.1% | Same |
| Forward EPS 63-day change -4% or worse (1990+) | 7, 2/5, +2.9% | 34, 15/13, +2.1% | Collapsing EPS: failed in 2001 and 2008, but V in 2009 and 2020 |
| VIX at confirm 60% or less of its peak | 10, 5/5, -0.5% | 31, 12/13, +3.1% | No. Explains why F did not beat its placebo |
| VIX below 30 at confirm | 19, 10/8, +1.8% | 22, 7/10, +2.7% | No |
| HY OAS 10%+ off its peak at confirm (2000-08+) | 13, 8/5, +2.0% | 28, 9/13, +2.4% | No. Explains why P failed |
| HY OAS peaked within 5 days of the equity low (**hindsight**) | 7, 7/0, +13.0% | 34, 10/18, +0.1% | Real-time version (21 days after the low, OAS peak was at the low and is 10% off): 8, 6/2, +7.4% vs 14, 5/9, -0.7% |
| Forward P/E at the low 14 or less | 10, 5/4, +4.1% | 31, 12/14, +1.7% | No |
| Speed alone: decline lasted 40 days or less | 13, 8/5, +4.9% | 28, 9/13, +1.0% | Oct-Nov 2008 legs were fast too |

Reading: fast-and-deep **from a high** is the pattern (1980, 1998, 2020, 2025; exception 1987, which lost 6% then went
sideways). Fast falls **inside** a broken trend (Oct and Nov 2008, Sep 2001, Jul 2002) are the failed ones. This is
why rule C failed: it bought every statistically extreme fall, wherever it started.
Without 2020 and 2025 the first row still averages +8.4% over 5 events — but 5 events is 5 events.

## How early is it known
- "From a 1-year high, 15%+ down": known on the day of the low.
- The confirm day came 2 days after the low in 2020, 11 trading days in 2025, 16 in 1998. Waiting for +10% gives up
  that 10% by construction; the +11.7% is what remains over the next 3 months.
- 2-year yield and VIX-divergence: known at confirm. Credit peak timing: needs about 21 more days.

## Fed text sentiment
- Bloomberg Economics publishes a natural-language Fed sentiment index built from Fed-speaker headlines. I believe the
  ticker is `BENLPFED Index`, but I am not certain: on the terminal type "Fed sentiment" in the search bar, or look in
  `ECAN <GO>` / the Bloomberg Economics "Fedspeak" page, and check the first date with `GP`.
- History is the problem: to my knowledge it starts around 2009. That covers about 12 of the 41 events and not
  1998, 2000-02 or 2008. Too short to validate anything; fine as a dashboard/story item.
- Free alternatives: FOMC statements (1994+), minutes and speeches at federalreserve.gov, which can be scored with a
  hawk/dove word list; academic hawk-dove indexes exist but update irregularly. The SF Fed Daily News Sentiment Index
  (1980+) is economic news tone, not Fed tone.
- Better real-time measure of "the Fed is turning": the **2-year Treasury yield** (FRED DGS2, 1976+) and fed funds
  futures. It is what the text indexes try to predict, has 50 years of history, and is in the table above. It shows a
  return gap but does not sort V from failed.

## Candidate rules for the next pre-registered test (Claude to write)
1. **Shock re-entry S.** If the S&P 500 has fallen 15% or more from a 252-day closing high, the lowest close came
   within 63 trading days of that high, and the index now closes 10% above that low: go to full base weight (or a
   50% floor) regardless of DMAS. End: base rule Invested, or a close below the low, or 126 trading days.
   Price-only, so Europe (STOXX 600, 1986+) is a true one-run check with its own events.
2. **Retest with less fear T.** While in Cash: the index makes a new low for the decline, the VIX peak of the
   decline is 10+ trading days old (VIX at the new low is below that peak), and the index then closes 10% above the
   low: same action and exits as S. Catches 2003, 2009, 2010, 2011; Europe check with V2X (1999+).
   Optional report-only flag for both: 2-year yield down 50 bp+ since the equity peak.

Honest odds: S has 7 US events and I found it by looking; expect roughly 1 in 3 that Europe confirms it.

## Extra data to download
- Nothing is needed for S. For T: nothing (VIX, V2X stored).
- For the Fed story (report-only): `BENLPFED Index` (PX_LAST, daily, Max — verify the ticker), `USGG2YR Index` and
  `GDBR2 Index` (PX_LAST, daily, Max; FRED DGS2 is a free substitute for the US), `FARBAST Index` (Fed assets, weekly;
  FRED WALCL is the same series), `EBBSTOTA Index` (ECB assets, weekly — verify).

## Events

| Peak | Trough | Confirm (+10%) | Class | Depth | Days down | Days since 1y high | 200d slope at low | Regained 1m/3m/6m | Days to new high | 3m after confirm | 6m after confirm | Fed funds chg peak→confirm | VIX peak / at confirm | HY OAS at confirm (÷ peak) | Fwd EPS 63d | Fwd P/E low | NBER rec. at low | Sahm at confirm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1957-07-15 | 1957-10-22 | 1958-04-21 | SLOW | -21% | 70 | 70 | -1% | +15%/+23%/+38% | 227 | +8% | +20% | -1.56 |  /  |  () |  |  | 1 | 1.97 |
| 1961-12-12 | 1962-06-26 | 1962-07-11 | SLOW | -28% | 135 | 135 | -2% | +22%/+23%/+53% | 299 | -1% | +12% | +0.73 |  /  |  () |  |  | 0 | 0.00 |
| 1966-02-09 | 1966-10-07 | 1966-11-01 | SLOW | -22% | 167 | 167 | -4% | +36%/+46%/+75% | 143 | +7% | +17% | +0.81 |  /  |  () |  |  | 0 | 0.00 |
| 1968-11-29 | 1970-05-26 | 1970-05-29 | SLOW | -36% | 369 | 252 | -5% | +12%/+30%/+37% | 451 | +6% | +11% | +2.15 |  /  |  () |  |  | 1 | 0.77 |
| 1973-01-11 | 1973-08-22 | 1973-10-11 | FAILED | -16% | 154 | 154 | -1% | +34%/-9%/-30% | 1744 | -16% | -17% | +4.30 |  /  |  () |  |  | 0 | 0.00 |
| 1973-10-12 | 1974-02-11 | 1974-03-13 | FAILED | -19% | 83 | 251 | -5% | +44%/+0%/-53% | 1501 | -8% | -31% | -0.90 |  /  |  () |  |  | 1 | 0.20 |
| 1974-03-13 | 1974-10-03 | 1974-10-10 | SLOW | -38% | 142 | 246 | -10% | +31%/+23%/+50% | 334 | +4% | +21% | +1.44 |  /  |  () |  |  | 1 | 0.73 |
| 1976-09-21 | 1978-03-06 | 1978-04-24 | V | -19% | 366 | 244 | -3% | +13%/+62%/+80% | 366 | +2% | +2% | +1.67 |  /  |  () |  |  | 0 | 0.00 |
| 1980-02-13 | 1980-03-27 | 1980-05-22 | V | -17% | 30 | 30 | +3% | +37%/+89%/+151% | 74 | +15% | +29% | -2.47 |  /  |  () |  |  | 1 | 0.63 |
| 1980-11-28 | 1981-09-25 | 1981-11-02 | FAILED | -20% | 208 | 208 | -1% | +19%/+35%/-3% | 280 | -5% | -5% | -2.89 |  /  |  () |  |  | 1 | 0.07 |
| 1981-11-30 | 1982-03-08 | 1982-04-23 | FAILED | -15% | 67 | 240 | -4% | +42%/+15%/+81% | 148 | -6% | +17% | +2.73 |  /  |  () |  |  | 1 | 1.33 |
| 1987-08-25 | 1987-10-19 | 1987-10-21 | FAILED | -33% | 38 | 38 | +9% | +16%/+22%/+30% | 447 | -6% | -1% | +0.73 |  /  |  () |  |  | 0 | 0.00 |
| 1989-10-09 | 1990-01-30 | 1990-05-21 | FAILED | -10% | 78 | 78 | +5% | +27%/+25%/+90% | 82 | -8% | -11% | -0.88 | 27 / 17 |  () |  |  | 0 | 0.17 |
| 1990-07-16 | 1990-10-11 | 1990-12-04 | SLOW | -20% | 62 | 62 | -2% | +25%/+27%/+116% | 86 | +15% | +18% | -0.63 | 36 / 22 |  () | -2% | 10.8 | 1 | 0.53 |
| 1997-10-07 | 1997-10-27 | 1997-12-01 | V | -11% | 14 | 14 | +8% | +70%/+95%/+205% | 28 | +7% | +11% | +0.02 | 38 / 26 |  () | +0% | 18.0 | 0 | 0.00 |
| 1998-07-17 | 1998-08-31 | 1998-09-23 | V | -19% | 31 | 31 | +6% | +26%/+90%/+118% | 59 | +13% | +21% | -0.01 | 46 / 32 |  () | +0% | 19.3 | 0 | 0.07 |
| 1998-09-23 | 1998-10-08 | 1998-10-16 | V | -10% | 11 | 58 | +4% | +170%/+296%/+374% | 9 | +19% | +24% | -0.40 | 46 / 35 |  () | +0% | 19.3 | 0 | 0.10 |
| 1999-07-16 | 1999-10-15 | 1999-11-08 | V | -12% | 64 | 64 | +5% | +86%/+127%/+64% | 22 | +5% | +3% | +0.25 | 29 / 22 |  () | -1% | 22.2 | 0 | 0.00 |
| 2000-03-24 | 2000-04-14 | 2000-07-12 | FAILED | -11% | 15 | 15 | +2% | +64%/+90%/+10% | 1788 | -7% | -12% | +0.70 | 33 / 20 |  () | +5% | 22.6 | 0 | 0.00 |
| 2000-09-01 | 2001-04-04 | 2001-04-18 | FAILED | -27% | 147 | 147 | -4% | +39%/+28%/-11% | 1537 | -2% | -12% | -1.57 | 35 / 24 | 7.3 (0.81) | -4% | 18.8 | 1 | 0.20 |
| 2001-05-21 | 2001-09-21 | 2001-10-03 | FAILED | -26% | 82 | 252 | -6% | +36%/+50%/+48% | 1160 | +9% | +5% | -1.32 | 44 / 31 | 9.0 (1.00) | -5% | 17.3 | 1 | 0.77 |
| 2002-01-04 | 2002-07-23 | 2002-07-29 | FAILED | -32% | 137 | 240 | -4% | +40%/+27%/+22% | 582 | -0% | -4% | +0.05 | 45 / 31 | 8.7 (0.99) | -1% | 14.3 | 0 | 1.40 |
| 2002-08-22 | 2002-10-09 | 2002-10-15 | V | -19% | 33 | 192 | -7% | +68%/+81%/+51% | 160 | +4% | -0% | +0.01 | 43 / 34 | 10.3 (0.99) | -2% | 14.0 | 0 | 0.93 |
| 2002-11-27 | 2003-03-11 | 2003-03-21 | V | -15% | 69 | 246 | -8% | +47%/+133%/+161% | 43 | +11% | +16% | +0.01 | 35 / 29 | 7.3 (0.85) | +1% | 14.3 | 0 | 0.20 |
| 2007-10-09 | 2008-03-10 | 2008-05-01 | FAILED | -19% | 104 | 104 | -1% | +28%/+30%/-2% | 1272 | -10% | -34% | -2.54 | 32 / 19 | 6.5 (0.78) | -2% | 12.9 | 1 | 0.57 |
| 2008-05-19 | 2008-10-10 | 2008-10-13 | FAILED | -37% | 101 | 251 | -6% | +4%/-5%/-11% | 984 | -13% | -15% | -0.56 | 81 / 55 | 13.9 (1.00) | -5% | 9.5 | 1 | 1.17 |
| 2008-10-13 | 2008-10-27 | 2008-10-28 | FAILED | -15% | 10 | 249 | -7% | +5%/+16%/+16% | 6 | -10% | -7% | -0.58 | 81 / 67 | 15.1 (0.99) | -9% | 9.5 | 1 | 1.17 |
| 2008-11-04 | 2008-11-20 | 2008-11-24 | FAILED | -25% | 12 | 240 | -9% | +47%/+8%/+62% | 178 | -12% | +6% | +0.03 | 81 / 65 | 18.3 (0.99) | -14% | 9.0 | 1 | 1.33 |
| 2009-01-06 | 2009-03-09 | 2009-03-12 | V | -28% | 42 | 202 | -14% | +54%/+102%/+132% | 58 | +26% | +39% | +0.10 | 57 / 41 | 16.4 (0.97) | -17% | 10.3 | 1 | 2.33 |
| 2010-04-23 | 2010-07-02 | 2010-08-02 | V | -16% | 49 | 49 | +5% | +50%/+64%/+121% | 87 | +5% | +14% | -0.02 | 46 / 22 | 6.4 (0.89) | +4% | 11.4 | 0 | 0.17 |
| 2011-04-29 | 2011-10-03 | 2011-10-14 | V | -19% | 108 | 108 | +1% | +45%/+67%/+119% | 99 | +6% | +14% | -0.03 | 48 / 28 | 7.6 (0.87) | -0% | 10.2 | 0 | 0.00 |
| 2015-05-21 | 2015-08-25 | 2015-10-23 | FAILED | -12% | 66 | 66 | +2% | +25%/+83%/+32% | 220 | -8% | +1% | +0.00 | 41 / 14 | 5.7 (0.88) | -1% | 15.0 | 0 | 0.00 |
| 2015-11-03 | 2016-02-11 | 2016-03-11 | V | -13% | 68 | 183 | -1% | +68%/+84%/+127% | 80 | +4% | +5% | +0.26 | 28 / 16 | 6.4 (0.77) | -3% | 15.0 | 0 | 0.00 |
| 2018-01-26 | 2018-02-08 | 2018-07-25 | FAILED | -10% | 9 | 9 | +4% | +69%/+49%/+93% | 137 | -4% | -6% | +0.49 | 37 / 12 | 3.4 (0.92) | +3% | 16.3 | 0 | 0.00 |
| 2018-09-20 | 2018-12-24 | 2019-01-10 | V | -20% | 65 | 65 | -0% | +54%/+78%/+97% | 81 | +11% | +16% | +0.48 | 36 / 20 | 4.5 (0.83) | -1% | 13.7 | 0 | 0.00 |
| 2020-02-19 | 2020-03-23 | 2020-03-25 | V | -34% | 23 | 23 | +3% | +49%/+77%/+91% | 103 | +23% | +31% | -1.41 | 83 / 64 | 10.2 (0.93) | -7% | 13.6 | 1 | 0.03 |
| 2022-01-03 | 2022-03-08 | 2022-03-29 | FAILED | -13% | 44 | 44 | +4% | +50%/-2%/-30% | 469 | -18% | -20% | +0.25 | 36 / 19 | 3.3 (0.79) | +4% | 18.4 | 0 | 0.00 |
| 2022-03-29 | 2022-06-16 | 2022-07-28 | FAILED | -21% | 55 | 114 | -1% | +28%/+21%/+24% | 374 | -6% | -0% | +1.32 | 35 / 22 | 4.9 (0.84) | +1% | 15.5 | 0 | 0.00 |
| 2022-08-16 | 2022-10-12 | 2022-11-10 | V | -17% | 40 | 195 | -4% | +52%/+56%/+77% | 166 | +5% | +5% | +1.35 | 34 / 24 | 4.7 (0.83) | -2% | 15.3 | 0 | 0.00 |
| 2023-07-31 | 2023-10-27 | 2023-11-20 | V | -10% | 63 | 63 | +4% | +93%/+171%/+195% | 24 | +12% | +17% | +0.12 | 22 / 13 | 3.8 (0.87) | +2% | 17.3 | 0 | 0.13 |
| 2025-02-19 | 2025-04-08 | 2025-04-24 | V | -19% | 34 | 34 | +3% | +59%/+112%/+152% | 55 | +16% | +23% | +0.00 | 52 / 26 | 3.7 (0.81) | -0% | 18.0 | 0 | 0.27 |

Class "open" = fewer than 63 trading days after the low. Sahm = 3-month average unemployment minus its 12-month low,
lagged one month. Fed funds = 10-day average effective rate. OAS in %, daily from 2000-08-15.
