# M: macro filter on the trend exit (pre-registered, spec 6M, one run)

**M1: PASS. M2: FAIL.** Consequence (declared before the run): PASS → proposed to the user; FAIL → rejected list, no change.

S&P 500 1955-01-03 → 2026-11-11. Price skeleton as 6P/6Q, 10 bp, cash = T-bill. Candidate: cash only if DMAS < 45 AND the macro key is out. Monthly data usable from the 10th of the next month. Code: `run_macro.py`.

## M1 unemployment trend: **PASS**

| Gate | Evidence | Result |
|---|---|---|
| G1 Sharpe better, full and both halves | full 0.603 vs 0.546; 1955–1990 0.477 vs 0.477; 1991–end 0.720 vs 0.609 | PASS |
| G2 worst fall not deeper by > 2 pts | -25.3% vs -28.5% | PASS |
| G3 bootstrap not-better share < 25% | 18.7% | PASS |
| G4 placebo p <= 0.05 | p = 0.010 (placebo median Sharpe 0.503) | PASS |
| G5 real product (1999-02 → end) not worse | 0.716 vs 0.713 (worst fall -20.2% vs -20.1%) | PASS |

| 1955-01-05 → 2026-11-11 | Base | Base + M1 |
|---|---:|---:|
| Sharpe | 0.55 | 0.60 |
| Return a year | 9.5% | 10.9% |
| Worst fall | -28.5% | -25.3% |
| Trades a year | 6.1 | 7.5 |
| Share of days the key is "out" | | 37% |

## M2 credit trend (Baa − Aaa): **FAIL**

| Gate | Evidence | Result |
|---|---|---|
| G1 Sharpe better, full and both halves | full 0.538 vs 0.546; 1955–1990 0.372 vs 0.477; 1991–end 0.693 vs 0.609 | FAIL |
| G2 worst fall not deeper by > 2 pts | -25.3% vs -28.5% | PASS |
| G3 bootstrap not-better share < 25% | 53.8% | FAIL |
| G4 placebo p <= 0.05 | p = 0.248 (placebo median Sharpe 0.504) | FAIL |
| G5 real product (1999-02 → end) not worse | 0.652 vs 0.713 (worst fall -25.8% vs -20.1%) | FAIL |

| 1955-01-05 → 2026-11-11 | Base | Base + M2 |
|---|---:|---:|
| Sharpe | 0.55 | 0.54 |
| Return a year | 9.5% | 9.9% |
| Worst fall | -28.5% | -25.3% |
| Trades a year | 6.1 | 7.1 |
| Share of days the key is "out" | | 46% |

## Information only

Correlation of M1 and M2 states (daily): +0.21.

### Base trend exits (day the base goes to cash), by macro key on that day

| Key | State | Exits | Mean next 63 days | Mean next 252 days | Share of exits followed by a lower S&P after 252 days |
|---|---|---:|---:|---:|---:|
| M1 | out (confirms the exit) | 22 | +2.9% | +10.7% | 27% |
| M1 | in (would stay invested) | 58 | +2.7% | +10.6% | 21% |
| M2 | out (confirms the exit) | 35 | +2.1% | +13.6% | 14% |
| M2 | in (would stay invested) | 45 | +3.3% | +8.4% | 29% |

| Exit | M1 | M2 | Next 63 days | Next 252 days |
|---|---|---|---:|---:|
| 1956-05-25 | in | in | +8.5% | +8.7% |
| 1956-09-06 | out | out | -1.7% | -4.4% |
| 1957-08-21 | out | out | -11.3% | +8.4% |
| 1959-09-21 | in | in | +8.0% | +1.1% |
| 1960-01-22 | in | out | -2.6% | +8.8% |
| 1960-09-21 | out | out | +6.4% | +26.2% |
| 1962-04-16 | in | in | -15.3% | +5.5% |
| 1965-06-11 | in | in | +5.5% | +4.7% |
| 1966-03-03 | in | in | -3.2% | +2.1% |
| 1967-11-03 | out | out | +0.9% | +20.6% |
| 1968-02-01 | in | out | +7.1% | +9.6% |
| 1968-08-05 | in | out | +10.6% | +1.1% |
| 1969-02-26 | in | in | +5.7% | -6.0% |
| 1969-06-04 | in | in | -7.4% | -20.7% |
| 1971-10-22 | out | in | +9.3% | +16.4% |
| 1973-03-05 | in | in | -7.9% | -12.5% |
| 1973-11-08 | in | out | -13.0% | -26.8% |
| 1975-10-03 | out | out | +8.8% | +25.9% |
| 1976-11-10 | in | in | +2.9% | -1.7% |
| 1977-02-03 | out | in | -0.7% | -7.9% |
| 1978-11-03 | in | in | +3.3% | +12.3% |
| 1979-03-05 | in | out | +2.6% | +20.8% |
| 1979-10-25 | out | out | +15.1% | +36.4% |
| 1980-03-18 | out | out | +13.0% | +35.5% |
| 1981-06-03 | in | in | -4.7% | -9.4% |
| 1981-12-18 | out | out | -7.7% | +17.4% |
| 1984-01-05 | in | in | -5.6% | +2.0% |
| 1986-10-03 | in | out | +8.8% | +44.6% |
| 1987-10-20 | in | in | +3.4% | +22.2% |
| 1987-12-08 | in | in | +15.5% | +22.4% |
| 1988-08-17 | in | in | +3.8% | +37.1% |
| 1990-01-24 | out | out | +1.4% | +3.6% |
| 1990-08-08 | in | out | -7.0% | +19.5% |
| 1991-11-27 | out | in | +10.4% | +17.4% |
| 1992-04-03 | out | in | +3.8% | +13.2% |
| 1992-10-13 | out | in | +6.6% | +15.9% |
| 1993-08-04 | in | in | +5.2% | +5.8% |
| 1993-12-03 | in | in | +0.6% | +0.3% |
| 1994-03-03 | in | out | +0.1% | +7.8% |
| 1994-11-25 | in | in | +7.7% | +36.1% |
| 1996-07-25 | in | out | +12.7% | +51.9% |
| 1998-08-18 | in | out | +3.5% | +22.7% |
| 1999-06-03 | in | out | +2.7% | +12.8% |
| 1999-08-04 | in | in | +3.6% | +11.5% |
| 1999-09-27 | in | in | +13.9% | +13.5% |
| 2000-02-01 | in | in | +2.9% | -1.9% |
| 2000-04-18 | in | in | +3.1% | -12.0% |
| 2000-05-11 | in | out | +4.0% | -10.4% |
| 2000-09-20 | out | in | -9.8% | -29.7% |
| 2002-04-17 | out | out | -19.2% | -20.5% |
| 2004-08-10 | in | in | +8.4% | +16.1% |
| 2006-06-15 | in | in | +5.3% | +24.1% |
| 2006-07-18 | in | in | +11.2% | +27.8% |
| 2007-08-07 | out | in | +2.2% | -10.9% |
| 2007-11-12 | out | in | -4.5% | -36.1% |
| 2010-06-03 | out | in | -1.5% | +21.3% |
| 2010-08-24 | in | out | +14.4% | +12.6% |
| 2011-08-03 | in | in | -2.8% | +10.6% |
| 2011-12-21 | in | out | +12.9% | +17.1% |
| 2012-06-05 | in | out | +9.9% | +30.6% |
| 2014-02-05 | in | in | +7.8% | +20.1% |
| 2014-10-15 | in | out | +7.5% | +10.9% |
| 2015-02-04 | in | out | +2.4% | -4.2% |
| 2015-04-06 | in | out | -0.1% | +0.4% |
| 2015-06-03 | in | out | -9.0% | +1.7% |
| 2015-11-17 | in | out | -6.0% | +8.5% |
| 2015-12-11 | in | out | +0.7% | +14.6% |
| 2016-06-29 | in | in | +5.4% | +19.2% |
| 2016-11-03 | out | in | +10.3% | +26.4% |
| 2018-04-04 | in | in | +3.1% | +11.0% |
| 2018-10-15 | in | out | -4.4% | +10.8% |
| 2019-06-05 | in | out | +4.5% | +12.3% |
| 2020-03-03 | in | in | +3.1% | +29.5% |
| 2022-01-25 | in | in | -3.8% | -5.3% |
| 2022-12-09 | in | out | +0.0% | +20.0% |
| 2023-03-03 | in | out | +6.3% | +27.5% |
| 2023-10-27 | out | in | +20.1% | +43.6% |
| 2025-03-05 | in | in | +2.5% | +16.8% |
| 2026-03-24 | out | in | +12.5% | +17.0% |
| 2026-11-04 | in | in | -0.9% | -0.9% |

## Post-run checks (information only, 2026-11-23; no rule changed)

- **Where M1 gains:** time in cash falls from 34% to 15% of days. Biggest gains are bull years with a false trend exit: 2019 (+16 pts), 2014 (+13), 2010 (+13), 1999 (+11), 1968, 2023, 2006, 1987 (+10 each). Biggest losses are bear markets that start before unemployment rises: 1973 (−12), 1966 (−12), 2020 (−10), 1962 (−9), 2022 (−9). The top 3 years give 38% of the total gain, so it is not one year.
- **After the rule was published (Philosophical Economics, 2016-07 → 2026-11):** base Sharpe 0.81, worst fall −12.0%; base + M1 Sharpe 0.69, worst fall −19.4%. M1 is worse after publication (2020: the crash came before unemployment rose; 2022: the bear market came with falling unemployment).
- **Real product (1999-02 → end):** 0.716 vs 0.713, worst fall −20.2% vs −20.1%. The earnings key already does this job since 1990.
- **Today:** unemployment 4.1% (Oct 2026), below its 12-month average → M1 "in".
