# KR: Korea canary cap (pre-registered, spec 6KR, one run)

**KR: FAIL.** Consequence (declared before the run): PASS → proposed to the user; FAIL → rejected list, no change.

Base = product spec 11d, inputs to 2026-11-13. 10 bp, cash = T-bill. Window starts 2001-05-16 (first day EWY momentum is defined). Code: `run_kr.py`.

| Gate | Evidence | Result |
|---|---|---|
| G1 Sharpe better, full and both halves | full 0.745 vs 0.808; 2001-05 → 2013-12 0.717 vs 0.827; after 2013-12 0.779 vs 0.789 | FAIL |
| G2 worst fall not deeper by > 2 pts | -14.7% vs -17.7% | PASS |
| G3 bootstrap not-better share < 25% | 92.6% | FAIL |
| G4 placebo p <= 0.025 | p = 0.352 (placebo median Sharpe 0.730) | FAIL |

| 2001-05-16 → 2026-11-13 | Product | Product + KR |
|---|---:|---:|
| Sharpe | 0.81 | 0.75 |
| Return a year | 10.8% | 9.1% |
| Worst fall | -17.7% | -14.7% |
| Trades a year | 8.8 | 17.4 |
| Share of days capped at 50% | | 36% |

## Information only: same rule on EEM (not a test)

| 2004-04-13 → 2026-11-13 | Product | Product + EEM-only cap |
|---|---:|---:|
| Sharpe | 0.84 | 0.79 |
| Return a year | 11.4% | 9.8% |
| Worst fall | -17.7% | -16.2% |
| Trades a year | 8.9 | 19.4 |
| Share of days capped at 50% | | 35% |

Correlation of the KR and EEM-only cap states (daily, from 2004-04-13): +0.72.

## Information only: stress episodes (total return / worst fall inside the window)

| Episode | S&P | Product | KR | EEM-only |
|---|---:|---:|---:|---:|
| Dot-com bear | -47.4% / -47.4% | -8.1% / -10.9% | -8.1% / -10.9% | -8.1% / -10.9% |
| 2008 crisis | -54.9% / -55.3% | -3.7% / -6.6% | -3.7% / -6.6% | -3.7% / -6.6% |
| 2011 debt ceiling | -18.5% / -18.7% | -15.1% / -15.3% | -13.2% / -13.4% | -16.0% / -16.2% |
| 2015–16 | -12.9% / -13.0% | -5.3% / -9.3% | -3.5% / -7.5% | -1.5% / -5.6% |
| Q4 2018 | -18.7% / -19.4% | -6.1% / -8.7% | -2.8% / -5.0% | -3.9% / -5.3% |
| COVID fall | -33.5% / -33.8% | -8.5% / -11.5% | -3.8% / -6.5% | -6.0% / -8.7% |
| 2022 | -18.2% / -24.5% | -13.9% / -17.7% | -8.3% / -12.0% | -10.7% / -14.1% |
| Liberation Day fall | -18.6% / -18.8% | -5.4% / -6.0% | -3.9% / -4.6% | -5.4% / -6.0% |
| Nov 2026 to date | -0.2% / -4.4% | -0.8% / -4.4% | +0.0% / -2.2% | +0.0% / -2.2% |

Caps on the last input day (2026-11-13): KR = 50%, EEM-only = 50%.

## Post-run information (2026-11-24, user question: has the market changed since COVID?). Not a test; the verdict stays FAIL.

Code: `info_post_covid_caps.py` (run with `PYTHONPATH=.`). Same rules, shorter windows. Six years is short: the Sharpe error is about ±0.4, so gaps below that are noise.

| Window | Product | KR (Korea cap) | T2 (AGG + EEM cap) |
|---|---|---|---|
| 2020-01 → end | Sharpe 0.89, 13.7%/yr, worst −17.7%, 11 trades/yr | 0.90, 12.5%, −12.0%, 18 | 0.88, 11.2%, −9.3%, 32 |
| 2020-06 → end | 0.92, 14.0%, −17.7%, 11 | 0.88, 12.2%, −12.0%, 18 | 0.91, 11.2%, −9.1%, 32 |
| 2023-01 → end | 1.06, 17.4%, −9.4%, 6 | 0.90, 13.9%, −8.5%, 18 | 0.93, 13.4%, −9.1%, 32 |

Placebo on 2020-06 → end (300 draws): KR p 0.32, T2 p 0.097.
Reading: after COVID the caps keep about the same Sharpe, cut the worst fall a lot (T2 −9% vs −18%, mostly 2022), and cost 1.5–3% a year. Since 2023 they only cost.
