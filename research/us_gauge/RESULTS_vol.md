# Volatility sizing — V1 and V2

## V1 versus BH

Verdict: V1 PASS

Return sample: 1955-01-06 through 2026-11-13; halves split 1990-01-01.

| Gate | Evidence | Result |
|---|---|---|
| FULL Sharpe > BH | 0.496479 > 0.461302 | PASS |
| H1 Sharpe > BH | 0.441162 > 0.401901 | PASS |
| H2 Sharpe > BH | 0.544912 > 0.511404 | PASS |
| Bootstrap not-better share < 0.25 | 0.1685 | PASS |
| FULL turnover/year <= 2.35 | 0.669829 | PASS |

| Period | Rule | Sharpe | CAGR | Max DD | Turnover/year |
|---|---|---:|---:|---:|---:|
| FULL | V1 | 0.496479 | 10.2100% | -43.8699% | 0.669829 |
| FULL | BH | 0.461302 | 10.8658% | -55.2546% | 0.000000 |
| H1 | V1 | 0.441162 | 10.9651% | -37.5539% | 0.460351 |
| H1 | BH | 0.401901 | 10.9171% | -44.9645% | 0.000000 |
| H2 | V1 | 0.544912 | 9.4990% | -43.8699% | 0.868367 |
| H2 | BH | 0.511404 | 10.8171% | -55.2546% | 0.000000 |
| P2021 | V1 | 0.711957 | 12.7152% | -19.8373% | 0.638508 |
| P2021 | BH | 0.699303 | 14.2748% | -24.5330% | 0.000000 |
| G_FULL | V1 | 0.556206 | 9.4461% | -43.8699% | 0.918094 |
| G_FULL | BH | 0.518978 | 10.8392% | -55.2546% | 0.000000 |
| G_H1 | V1 | 0.329407 | 7.3224% | -43.8699% | 0.952552 |
| G_H1 | BH | 0.297232 | 7.7100% | -55.2546% | 0.000000 |
| G_H2 | V1 | 0.783878 | 11.5979% | -19.8373% | 0.883855 |
| G_H2 | BH | 0.771910 | 14.0386% | -33.7936% | 0.000000 |

## V2-all versus R1-all

Verdict: V2-all PASS

Return sample: 1993-04-08 through 2026-11-13; halves split 2010-01-01.

| Gate | Evidence | Result |
|---|---|---|
| FULL Sharpe > R1-all | 0.731606 > 0.705737 | PASS |
| H1 Sharpe > R1-all | 0.630426 > 0.598689 | PASS |
| H2 Sharpe > R1-all | 0.824883 > 0.802953 | PASS |
| Bootstrap not-better share < 0.25 | 0.1930 | PASS |
| FULL turnover/year <= 2.35 | 1.621209 | PASS |

| Period | Rule | Sharpe | CAGR | Max DD | Turnover/year |
|---|---|---:|---:|---:|---:|
| FULL | V2-all | 0.731606 | 10.5774% | -25.5374% | 1.621209 |
| FULL | R1-all | 0.705737 | 11.2611% | -35.0627% | 1.370375 |
| H1 | V2-all | 0.630426 | 10.2149% | -25.5374% | 1.411888 |
| H1 | R1-all | 0.598689 | 10.5534% | -35.0627% | 1.195446 |
| H2 | V2-all | 0.824883 | 10.9387% | -17.8328% | 1.829199 |
| H2 | R1-all | 0.802953 | 11.9688% | -21.7041% | 1.544190 |
| P2021 | V2-all | 0.844394 | 13.1435% | -15.8682% | 1.377499 |
| P2021 | R1-all | 0.763435 | 13.1676% | -21.7041% | 1.367707 |

## V2-half versus R1-half

Verdict: V2-half PASS

Return sample: 1993-04-08 through 2026-11-13; halves split 2010-01-01.

| Gate | Evidence | Result |
|---|---|---|
| FULL Sharpe > R1-half | 0.678809 > 0.644741 | PASS |
| H1 Sharpe > R1-half | 0.492763 > 0.455509 | PASS |
| H2 Sharpe > R1-half | 0.853867 > 0.830306 | PASS |
| Bootstrap not-better share < 0.25 | 0.2050 | PASS |
| FULL turnover/year <= 2.35 | 1.184468 | PASS |

| Period | Rule | Sharpe | CAGR | Max DD | Turnover/year |
|---|---|---:|---:|---:|---:|
| FULL | V2-half | 0.678809 | 10.3194% | -33.4228% | 1.184468 |
| FULL | R1-half | 0.644741 | 11.2911% | -36.8823% | 0.685187 |
| H1 | V2-half | 0.492763 | 8.9310% | -33.4228% | 1.087220 |
| H1 | R1-half | 0.455509 | 9.4434% | -36.8823% | 0.597723 |
| H2 | V2-half | 0.853867 | 11.7165% | -17.8328% | 1.281098 |
| H2 | R1-half | 0.830306 | 13.1580% | -22.3587% | 0.772095 |
| P2021 | V2-half | 0.840332 | 13.5536% | -16.3352% | 0.911473 |
| P2021 | R1-half | 0.766609 | 13.8585% | -22.3587% | 0.683853 |

## Method

Sigma uses the last 21 total returns through t, ddof=1, annualized by sqrt(252); v=min(1,0.15/sigma). Combined target changes trade at >=0.10 on the first monthly decision, >=0.25 on other weekly decisions. R1 state changes always execute. Tuesday/holiday calendar and R1 states reuse the gauge.
Decision t, execution close t+1, return exposure t+2. Cost is 10 bp times absolute weight change. Cash uses DTB3/100/252 through 2007-05-30 and BIL total returns thereafter. S&P total returns use price change plus SHILLER_SPX_DIV/price/252.
Turnover is sum of absolute daily weight changes divided by sample days/252; boundary changes are retained. All subperiods slice the continuous simulation. G_FULL/G_H1/G_H2 are the April 1993 gauge window and halves split 2010. P2021 is information only; V1 gates use the 1955 FULL/H1/H2 sample.
Sharpe uses excess returns. Bootstrap uses paired 63-day blocks, 2,000 draws, default_rng(7), separately for V1 and both V2 versions. Both V2 versions have separately reported gates. CSV includes dates, days, volatility and Calmar.

## Deviations

No rule, threshold, window or gate changes; no variants. load_inputs already includes BBG_SPX from 1950 and DTB3 from 1954-01-04, so it is unchanged. Inherited gauge conventions: initial allocation has no turnover/cost; costs are debited on t+2; holiday decisions use the actual trading date’s calendar month. No external writes or network access.
