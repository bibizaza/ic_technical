# L-US: shock re-entry at 100%, S&P 500 1928–1954 (pre-registered, spec 6LU, one run)

**Verdict: TOO FEW TO CONCLUDE. Consequence (declared before the run): 11c stays.**

Sample 1929-01-10 → 1954-12-31 (Tuesday). Equity = GSPC price + Shiller dividends; cash 0%; 10 bp. Code: `run_lus.py`.

| Gate | Evidence | Result |
|---|---|---|
| G1 Sharpe R0+S > R0, Tuesday and weekday mean | Tue 0.472 vs 0.517; mean 0.513 vs 0.563 (MON 0.525/0.565, TUE 0.472/0.517, WED 0.535/0.592, THU 0.546/0.608, FRI 0.488/0.533) | FAIL |
| G2 worst fall no more than 3 pts deeper | -54.8% vs -46.0% | FAIL |
| G3 more than half of episodes positive | 2 of 4 | FAIL |
| G4 worst single episode >= -15% | -22.4% | FAIL |
| G5 placebo p <= 0.10 | p = 0.845 (placebo median 0.507) | FAIL |

| Tuesday | R0 | R0+S |
|---|---:|---:|
| Sharpe | 0.517 | 0.472 |
| Return a year | 7.22% | 6.71% |
| Worst fall | -46.0% | -54.8% |

## Episodes (Tuesday)

| 1-year high | Low | Entry | End | Reason | Fall | Days | Index return |
|---|---|---|---|---|---:|---:|---:|
| 1929-09-16 | 1929-10-29 | 1929-10-30 | 1929-11-11 | stop | -35.9% | 6 | -22.4% |
| 1929-09-16 | 1929-11-13 | 1929-11-15 | 1930-04-03 | base invested | -44.6% | 94 | +29.2% |
| 1929-09-19 | 1929-11-13 | 1930-09-18 | 1930-10-17 | stop | -44.3% | 20 | -14.7% |
| 1940-04-08 | 1940-06-10 | 1941-02-06 | 1941-08-07 | base invested | -28.6% | 126 | +4.3% |

## Information only: vol-sized base (R0 x vol size)

| | Sharpe | Worst fall |
|---|---:|---:|
| V | 0.674 | -44.8% |
| V+S live | 0.674 | -46.6% |
| V+S full | 0.584 | -46.0% |
