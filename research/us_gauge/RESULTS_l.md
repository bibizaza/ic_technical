# L: re-entry layers at full weight (pre-registered, spec 6L, one run)

**Verdict: FAIL: keep S and B at the live size**

Six fresh indices, price only, cash 0%, 10 bp. Base = plain trend rule (DMAS 45/30, Tuesday, month lock) x vol size. S-live vs S-full = shock episodes at live vol size vs at 100%. B has no fresh sample; it follows S by symmetry. Code: `run_l.py`.

| Gate | Evidence | Result |
|---|---|---|
| G1 mean Sharpe gain > 0 and positive in >= 4 of 6 | mean -0.005; positive in 3 of 6 | FAIL |
| G2 worst fall no more than 3 pts deeper, every market | N225 -54.9% vs -57.7%; TWII -42.0% vs -42.0%; GSPTSE -27.6% vs -27.6%; HSI -47.4% vs -46.8%; KS11 -40.0% vs -41.5%; AXJO -38.5% vs -35.3% | FAIL |
| G3 > half of clusters positive, also without 2020 and 2025 | 25 of 48; without 2020/2025: 23 of 45 | PASS |
| G4 worst single episode >= -15% | -21.6% (HSI 1997-10-29) | FAIL |
| G5 placebo p <= 0.10 | real pooled gain -0.001; p = 0.403 (placebo median -0.004) | FAIL |

## By market

| Market | From | Episodes | Sharpe base | S-live | S-full | S-50% floor | Worst fall S-live | S-full |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| N225 | 1966-12-01 | 16 | 0.430 | 0.427 | 0.435 | 0.426 | -57.7% | -54.9% |
| TWII | 1970-11-13 | 12 | 0.727 | 0.702 | 0.694 | 0.701 | -42.0% | -42.0% |
| GSPTSE | 1980-06-26 | 4 | 0.602 | 0.607 | 0.620 | 0.614 | -27.6% | -27.6% |
| HSI | 1983-01-13 | 15 | 0.393 | 0.420 | 0.376 | 0.413 | -46.8% | -47.4% |
| KS11 | 1984-05-17 | 7 | 0.677 | 0.670 | 0.658 | 0.671 | -41.5% | -40.0% |
| AXJO | 1993-06-03 | 6 | 0.227 | 0.256 | 0.268 | 0.263 | -35.3% | -38.5% |

## Clusters

| # | First entry | Markets | Mean index return in episode |
|---:|---|---|---:|
| 0 | 1970-06-04 | N225 | +0.9% |
| 1 | 1971-09-28 | N225 | +2.6% |
| 2 | 1971-12-17 | TWII | -1.3% |
| 3 | 1974-03-21 | TWII | -11.1% |
| 4 | 1976-05-29 | TWII | -4.2% |
| 5 | 1979-09-05 | TWII | -1.2% |
| 6 | 1981-11-27 | GSPTSE | -10.4% |
| 7 | 1983-09-08 | HSI | -5.8% |
| 8 | 1986-11-12 | N225 | +6.9% |
| 9 | 1987-10-22 | N225 | -9.3% |
| 10 | 1987-11-13 | HSI, N225 | -2.0% |
| 11 | 1987-12-14 | GSPTSE, HSI | +17.1% |
| 12 | 1988-09-08 | GSPTSE | +2.0% |
| 13 | 1988-10-11 | HSI | +1.3% |
| 14 | 1989-06-12 | HSI | +20.0% |
| 15 | 1989-08-24 | KS11 | +1.9% |
| 16 | 1989-12-26 | TWII | +13.3% |
| 17 | 1990-04-27 | TWII, N225 | -7.8% |
| 18 | 1990-10-23 | HSI | +10.4% |
| 19 | 1991-06-17 | N225 | -0.4% |
| 20 | 1992-12-28 | HSI | +19.4% |
| 21 | 1993-12-16 | N225 | +12.5% |
| 22 | 1997-09-08 | HSI | -6.2% |
| 23 | 1997-10-29 | HSI, AXJO, TWII | -4.5% |
| 24 | 1999-04-05 | N225 | +2.3% |
| 25 | 1999-08-13 | TWII | +1.8% |
| 26 | 2000-03-29 | KS11 | -5.9% |
| 27 | 2000-06-02 | HSI | +1.6% |
| 28 | 2000-07-03 | N225 | -10.0% |
| 29 | 2001-10-12 | AXJO | +6.8% |
| 30 | 2002-06-27 | HSI | -15.3% |
| 31 | 2004-05-27 | KS11, HSI | -5.5% |
| 32 | 2006-08-17 | KS11 | +1.8% |
| 33 | 2007-09-27 | N225 | -9.9% |
| 34 | 2008-01-25 | AXJO, HSI | -10.6% |
| 35 | 2008-02-25 | TWII | +2.7% |
| 36 | 2010-10-07 | AXJO | +2.5% |
| 37 | 2011-06-09 | N225 | +6.6% |
| 38 | 2012-05-10 | KS11 | +3.4% |
| 39 | 2012-07-26 | HSI | +4.5% |
| 40 | 2012-08-17 | N225 | -4.5% |
| 41 | 2016-05-25 | TWII | +2.2% |
| 42 | 2019-02-13 | N225 | -0.5% |
| 43 | 2020-03-24 | GSPTSE, KS11, TWII, N225, AXJO | +22.0% |
| 44 | 2020-05-13 | HSI | +8.0% |
| 45 | 2022-08-16 | AXJO | -3.3% |
| 46 | 2024-08-08 | N225, KS11 | -2.8% |
| 47 | 2025-03-06 | TWII | -18.2% |

## Episodes

| Market | Entry | End | Reason | Days | Index return |
|---|---|---|---|---:|---:|
| N225 | 1970-06-04 | 1970-11-02 | time | 126 | +0.9% |
| N225 | 1971-09-28 | 1971-12-09 | base invested | 59 | +2.6% |
| TWII | 1971-12-17 | 1972-01-13 | base invested | 18 | -1.3% |
| TWII | 1974-03-21 | 1974-05-14 | stop | 44 | -11.1% |
| TWII | 1976-05-29 | 1976-06-04 | base invested | 4 | -4.2% |
| TWII | 1979-09-05 | 1979-09-06 | base invested | 1 | -1.2% |
| GSPTSE | 1981-11-27 | 1982-01-13 | stop | 33 | -10.4% |
| HSI | 1983-09-08 | 1983-12-08 | base invested | 62 | -5.8% |
| N225 | 1986-11-12 | 1986-12-04 | base invested | 17 | +6.9% |
| N225 | 1987-10-22 | 1987-11-10 | stop | 15 | -9.3% |
| HSI | 1987-11-13 | 1987-12-07 | stop | 16 | -14.1% |
| N225 | 1987-11-25 | 1988-03-03 | base invested | 72 | +10.0% |
| GSPTSE | 1987-12-14 | 1988-04-07 | base invested | 83 | +7.5% |
| HSI | 1987-12-16 | 1988-06-22 | time | 126 | +26.6% |
| GSPTSE | 1988-09-08 | 1988-11-03 | base invested | 40 | +2.0% |
| HSI | 1988-10-11 | 1988-11-03 | base invested | 16 | +1.3% |
| HSI | 1989-06-12 | 1989-10-05 | base invested | 80 | +20.0% |
| KS11 | 1989-08-24 | 1989-09-07 | base invested | 10 | +1.9% |
| TWII | 1989-12-26 | 1990-01-11 | base invested | 8 | +13.3% |
| TWII | 1990-04-27 | 1990-05-01 | stop | 2 | -7.7% |
| N225 | 1990-05-07 | 1990-08-07 | stop | 66 | -7.9% |
| HSI | 1990-10-23 | 1991-02-07 | base invested | 73 | +10.4% |
| N225 | 1991-06-17 | 1991-10-03 | base invested | 76 | -0.4% |
| HSI | 1992-12-28 | 1993-03-04 | base invested | 45 | +19.4% |
| N225 | 1993-12-16 | 1994-02-03 | base invested | 32 | +12.5% |
| HSI | 1997-09-08 | 1997-10-09 | base invested | 20 | -6.2% |
| HSI | 1997-10-29 | 1998-01-09 | stop | 49 | -21.6% |
| AXJO | 1997-11-04 | 1998-01-08 | base invested | 44 | +2.2% |
| TWII | 1997-11-05 | 1998-02-05 | base invested | 66 | +5.8% |
| N225 | 1999-04-05 | 1999-04-08 | base invested | 3 | +2.3% |
| TWII | 1999-08-13 | 1999-09-09 | base invested | 19 | +1.8% |
| KS11 | 2000-03-29 | 2000-04-06 | stop | 5 | -5.9% |
| HSI | 2000-06-02 | 2000-06-08 | base invested | 3 | +1.6% |
| N225 | 2000-07-03 | 2000-07-28 | stop | 18 | -10.0% |
| AXJO | 2001-10-12 | 2002-01-10 | base invested | 61 | +6.8% |
| HSI | 2002-06-27 | 2002-10-07 | stop | 70 | -15.3% |
| KS11 | 2004-05-27 | 2004-08-02 | stop | 47 | -11.0% |
| HSI | 2004-05-28 | 2004-07-08 | base invested | 27 | +0.0% |
| KS11 | 2006-08-17 | 2006-09-07 | base invested | 15 | +1.8% |
| N225 | 2007-09-27 | 2007-11-12 | stop | 31 | -9.9% |
| AXJO | 2008-01-25 | 2008-03-10 | stop | 30 | -10.2% |
| HSI | 2008-01-25 | 2008-03-17 | stop | 34 | -11.1% |
| TWII | 2008-02-25 | 2008-03-06 | base invested | 7 | +2.7% |
| AXJO | 2010-10-07 | 2010-11-04 | base invested | 20 | +2.5% |
| N225 | 2011-06-09 | 2011-07-07 | base invested | 20 | +6.6% |
| KS11 | 2012-05-10 | 2012-10-05 | base invested | 101 | +3.4% |
| HSI | 2012-07-26 | 2012-08-09 | base invested | 10 | +4.5% |
| N225 | 2012-08-17 | 2012-11-08 | base invested | 57 | -4.5% |
| TWII | 2016-05-25 | 2016-06-13 | base invested | 12 | +2.2% |
| N225 | 2019-02-13 | 2019-03-07 | base invested | 16 | -0.5% |
| GSPTSE | 2020-03-24 | 2020-07-09 | base invested | 74 | +19.6% |
| KS11 | 2020-03-24 | 2020-06-04 | base invested | 48 | +28.0% |
| TWII | 2020-03-25 | 2020-06-04 | base invested | 48 | +17.9% |
| N225 | 2020-03-25 | 2020-06-04 | base invested | 47 | +22.5% |
| AXJO | 2020-03-26 | 2020-07-09 | base invested | 72 | +22.2% |
| HSI | 2020-05-13 | 2020-07-09 | base invested | 39 | +8.0% |
| AXJO | 2022-08-16 | 2022-11-03 | base invested | 56 | -3.3% |
| N225 | 2024-08-08 | 2024-09-05 | base invested | 19 | +3.9% |
| KS11 | 2024-08-16 | 2024-11-13 | stop | 57 | -9.6% |
| TWII | 2025-03-06 | 2025-04-07 | stop | 20 | -18.2% |

## Data notes (daily moves above 25%, not removed)

- N225: none
- TWII: none
- GSPTSE: none
- HSI: 1987-10-26 -33%
- KS11: none
- AXJO: none

## US product, information only (in-sample, 1999-02 → last day)

| Variant | Sharpe | Return a year | Worst fall | COVID rebound | 2009 rebound |
|---|---:|---:|---:|---:|---:|
| S, B at live size (core) | 0.69 | 9.4% | -20.1% | +17% | +22% |
| S, B at max(live, 50%) | 0.70 | 9.6% | -20.1% | +21% | +24% |
| S, B at 100% | 0.73 | 10.4% | -20.1% | +31% | +34% |
