# Primary-source check of the V-gap ideas (Fable 5.2, 2026-11-17)

I downloaded the papers and read the text (PDF → text, then the abstract, data, results and the
passages on crashes and rebounds). "Read" = the number below was read in the paper or on the
author's own page. "Summary only" = I could not get the original; the claim comes from secondary
sources and is not verified.

Four originals could not be obtained: Zweig's book, Whaley's paper, Desmond's Lowry paper and
Liu-Tang-Zhou. They are marked below.

## Summary table

| # | Source | Status | Exact claim | Sample and method | In-sample vs after | Meaning for us |
|---|---|---|---|---|---|---|
| 1 | Deemer, "Breakaway Momentum 101" (author's page) | Read | A thrust = 10-day total NYSE advances > 1.97 × 10-day total declines; "cyclical bull markets are traditionally heralded" by it | NYSE breadth since 1945; 25 signals to Jan 2023 (one every 3.5 years); table of dates and peak ratios | Concept from the 1970s. He says the 12 signals after 1973 were noted in real time. The page gives **no** return statistics, only "declines tend to be brief and shallow". A 20-day rule (1.72) was added in 2019 after seeing 2016 | Clean, parameter-fixed definition → basis of rule B. No published performance number to haircut; our test is the evidence |
| 2 | Zweig, "Winning on Wall Street" (1986) | Summary only | 10-day exponential average of A/(A+D) from below 40% to above 61.5% within 10 days | Secondary sources quoting the book: 14 thrusts since 1945, average +24.6% in 11 months | Later "perfect record" claims are from vendors. A 2007+ S&P test found 53% winners | Not used in B (original not read). Dates reported only |
| 3 | Whaley, breadth thrust paper | Summary only (not found) | Thrust conditions on NYSE advances, declines and volume | One vendor counts 30 signals since 1950 on one condition | No out-of-sample study found | Not used |
| 4 | Desmond (Lowry), "Identifying Bear Market Bottoms…" (Dow Award 2002) | Summary only (paywalled) | 90% down-volume days cluster in panics; a 90% up day (or two 80% days) soon after marks the turn | Lowry up/down volume and points data | No failure table available | Needs up/down volume; not used in B |
| 5 | Moskowitz, Ooi, Pedersen, "Time series momentum" (JFE 2012) | Read | Past 12-month return predicts the next month in every one of 58 futures; diversified strategy Sharpe above 1; best in extreme markets (straddle-like) | 58 instruments, tests 1985-2009; 12-month look-back, 1-month hold, volatility-scaled | Persistence 1-12 months, partial reversal after. **"TSMOM suffers sharp losses when the crisis ends in March, April, and May of 2009. The ending of a crisis constitutes a sharp trend reversal"** | The authors themselves say trend rules lose at V turns. Our gap is the documented cost of the method, not a bug |
| 6 | Hurst, Ooi, Pedersen, "A Century of Evidence on Trend-Following" (2014 update) | Read | Trend following earned positive returns in every decade since 1880 | 67 markets, 1880-2013, 1/3/12-month signals, net of costs and 2/20 fees | Full sample net: 11.2% a year, 9.7% vol, Sharpe 0.77, zero correlation to equities. The 1880-1980 part is out of sample relative to the 1985+ papers | The trend leg (DMAS) rests on the best-supported effect we use |
| 7 | Moreira, Muir, "Volatility-Managed Portfolios" (NBER w22208; JF 2017) | Read | Scaling by last month's variance gives alpha: market factor 4.9% a year, appraisal ratio 0.33, Sharpe +25%, utility +65% | US factors 1926-2015, monthly | Weaker in 1956-1985. **"Our strategy cashed out almost completely [after autumn 2008] and returned to the market only as the spike in volatility receded."** They argue this is right because volatility fades faster than expected returns | V2 behaves exactly as designed. The paper accepts missing the first leg |
| 8 | Cederburg, O'Doherty, Wang, Yan (JFE 2020) | Read | Volatility management does not systematically beat the unmanaged portfolio in real time | 103 equity strategies | Managed wins 53, loses 50; only 8 significant, mostly momentum. Real-time combinations underperform; market-factor differences insignificant | Expect little Sharpe from V2 itself. Consistent with our +0.035 (p 0.17) |
| 9 | Liu, Tang, Zhou (JPM 2019) | Summary only | Moreira-Muir scaling has a look-ahead element; real-time gains are hard to get | — | — | Same direction as #8 |
| 10 | Harvey et al., "The Impact of Volatility Targeting" (JPM 2018) | Read | Helps risk assets, not bonds or commodities; cuts the left tail | 60+ assets, US equities daily from 1926 | US equity Sharpe 0.40 → 0.48-0.51; mechanism is the leverage effect (it adds a little momentum) | Supports V2 for drawdown control, modest Sharpe |
| 11 | Faber, "A Quantitative Approach to TAA" (2006; 2013 update) | Read | Hold when price > 10-month average, else cash | S&P 500 1901-2012, monthly | Invested about 70% of the time, under one round trip a year. Average return 11.22% vs 11.26% for buy and hold; compounded 10.18% vs 9.32%. **Underperforms in about half of all years** and "during a roaring bull market". Post-2005 (out of sample) "performed well" | Same profile as R0: the gain is lower volatility, not higher average return |
| 12 | Daniel, Moskowitz, "Momentum Crashes" (NBER w20439; JFE 2016) | Read | Momentum crashes come in "panic states", after market falls with high volatility, at the market rebound | US stocks 1927-2013, plus other markets | Mar-May 2009: past losers +163%, past winners +8%. Jul-Aug 1932: +232% vs +32%. Losers' up-market beta in bear markets 2.16 vs down-market 1.56 | **New for us:** rebounds are led by beaten-down, high-beta stocks. A defensive sector tilt (XLP/XLU/XLE) is the wrong sleeve in the first weeks of a rebound |
| 13 | Garg, Goulding, Harvey, Mazzoleni, "Momentum Turning Points" (2020 version; JFE 2023) | Read | Blending 12-month and 1-month signals handles turns better than either | US 1969-2018 monthly (and international) | Sharpe: market 0.38, slow 0.41, 50/50 blend 0.51, fast 0.34. States: Bull 48.3%, Bear 16.8%, Correction + Rebound 34.8%. They say Rebound months are followed by returns "similar to Bull phases, but with higher volatility" | My 1929-2026 check does **not** confirm the rebound claim outside 1955-89 (see Corrections) |
| 14 | "Breaking Bad Trends" (same authors, FAJ) | Summary only | More turning points → weaker trend returns | — | — | Not relied on |
| 15 | AQR, "Hold the Dip" (2025) | Read | Buying dips does not beat buy and hold | 196 dip rules on the S&P 500, 1965-2025 | Average Sharpe 0.04 below buy and hold; over 60% of rules underperform; 1989-2025: 0.27 below. Only 16 of 196 have a significant alpha | Confirms C's failure and the 200-day result on an independent design |
| 16 | Estrada, "Buy the Dip? Not So Fast" (2026) | Read (abstract and introduction) | Dip buying "fails to enhance risk-adjusted returns" against two benchmarks | — | — | Same |
| 17 | McLean, Pontiff, "Does Academic Research Destroy Stock Return Predictability?" (JF 2016) | Read | Predictor returns are 26% lower out of sample and 58% lower after publication; 32 points attributed to publication-informed trading | 97 cross-sectional stock predictors | Decay is larger for predictors with higher in-sample returns | Haircut for any published rule: plan on about 40% of the reported effect |

## What the papers change in our view

1. **The gap is in the literature.** Both the trend paper (#5) and the volatility paper (#7) state that
   their strategies sit out the first leg of a rebound. Nobody in these papers claims a fix.
2. **Dip-buying is tested and rejected** on designs independent of ours (#15, #16).
3. **A new lead from #12:** what you hold in the rebound matters. Past losers and high-beta stocks lead it.
   K1's sector sleeve leans defensive. A report-only check is cheap: K1 sleeve vs plain SPY in the 63 days
   after each re-entry. I have not run it.
4. **Breadth thrusts have no audited performance record.** Deemer's page gives dates, not returns. The
   "never failed" numbers are vendor claims. Our one-run test would be the first clean evidence we have.

## Hurdle for B after publication

McLean-Pontiff is about stock anomalies, so it is a guide only. Applied to B: assume 40% of any
pre-2003 effect survives. The draft pre-registration therefore leans on the placebo gate and adds a
post-2003 episode gate (G8). Fewer than 10 episodes = "too few to conclude".

## Corrections to VGAP_fable_2026-11-17.md

| Where | Was | Should be | Basis |
|---|---|---|---|
| Section 2, volatility targeting | Cederburg et al.: "104 strategies" | 103 strategies; managed wins 53, loses 50, 8 significant | Read (#8) |
| Section 2, breadth thrust | "About 20 Zweig signals since 1945, said to have no losing 12-month outcome" | Zweig's book, as quoted by secondary sources: 14 thrusts since 1945, average +24.6% in 11 months. "No losing outcome" is a vendor claim. Original not read | Summary only (#2) |
| Section 2, breadth thrust | "Deemer; Whaley 2009" with no numbers | Deemer: 25 signals since 1945 to Jan 2023, no return statistics published. Whaley's paper not found | Read (#1); #3 not found |
| Section 2, slow + fast trend | "Tested here: rebound months earn nothing extra" | The paper claims Rebound months are followed by Bull-like returns (1969-2018). My data: +0.8% a month in 1955-89 (Bull-like), but -0.2% in 1929-54 and +0.3% in 1990-2026. The claim does not hold outside the middle period | Read (#13) + `out/vgap_momentum_states.csv` |
| Section 2, volatility targeting | "about $400 bn by one 2020 estimate" | Unverified press figure; do not quote | Summary only |
| Section 2, put selling | PUT index Sharpe 0.65 vs 0.49, worst fall -33% | Unverified (vendor summary of Bondarenko); my own 2007+ test stands | Summary only |
| Section 4, candidate B | "Either Zweig or Deemer" | Deemer only (original read, dates reproduced). Zweig reported only | #1, #2 |
| Section 5, Bloomberg | "PX_LAST, Max range" for breadth | Free history 1965-2020 found; Bloomberg needed only from 2015 (overlap) to today | Data section of PREREG_B_draft.md |

No number from my own tests in the VGAP memo is affected.

## Links

1. http://www.walterdeemer.com/bam.htm
2. Zweig (secondary): https://marketinout.com/technical_analysis.php?id=30 ; https://www.moneyshow.com/articles/tradingidea-63982
3. Whaley (secondary): https://www.sentimentrader.com/blog/another-thrust-narrowly-missed
4. Desmond (secondary): https://www.scribd.com/document/369653221/2002DowAward-pdf
5. https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf
6. https://www.trendfollowing.com/whitepaper/Century_Evidence_Trend_Following.pdf
7. https://www.nber.org/system/files/working_papers/w22208/w22208.pdf
8. https://www.lehigh.edu/~xuy219/research/COWY.pdf
9. https://profiles.wustl.edu/en/publications/volatility-managed-portfolio-does-it-really-work/
10. https://people.duke.edu/~charvey/Research/Published_Papers/P135_The_impact_of.pdf
11. https://mebfaber.com/wp-content/uploads/2016/05/SSRN-id962461.pdf
12. https://www.nber.org/system/files/working_papers/w20439/w20439.pdf
13. https://benny.aeaweb.org/conference/2021/preliminary/paper/492Ds6fk
14. https://alphaarchitect.com/trend-following-research-breaking-bad-trends/
15. https://www.aqr.com/-/media/AQR/Documents/Alternative-Thinking/AQR-Alternative-Thinking---Hold-the-Dip.pdf?sc_lang=en
16. https://blog.iese.edu/jestrada/files/2026/06/BTD-USA.pdf
17. https://counterpointfunds.com/wp-content/uploads/2017/07/PredictabilityMcleanPontiff.pdf
