# Q: real money flows as a key, Fed Z.1 (pre-registered, spec 6Q, one run)

**Q1: FAIL. Q2: FAIL.** Consequence (declared before the run): PASS → proposed to the user as a key; FAIL → rejected list, no change.

S&P 500 1956-10-29 → 2026-11-11. Price skeleton as 6P (DMAS keys daily with month lock, vol size, S at max(live, 50%), 5-point sizes, trade next day if gap ≥ 10 points), 10 bp, cash = T-bill. Candidate: cash only if DMAS < 45 AND the flow key is out; the DMAS < 30 mid-month exit uses the same combined condition, as in the product (`run_fast.states`). Flow data: Fed Z.1, current vintage, usable 120 days after quarter end. Code: `run_q.py`.

## Q1 buyback yield (follow the companies): **FAIL**

| Gate | Evidence | Result |
|---|---|---|
| G1 Sharpe better, full and both halves | full 0.545 vs 0.527; 1955–1990 0.480 vs 0.432; 1991–end 0.605 vs 0.609 | FAIL |
| G2 worst fall not deeper by > 2 pts | -32.7% vs -28.5% | FAIL |
| G3 bootstrap not-better share < 25% | 37.8% | FAIL |
| G4 placebo p <= 0.05 | p = 0.203 (placebo median Sharpe 0.487) | FAIL |
| G5 real product (1999-02 → end) not worse | 0.587 vs 0.713 (worst fall -32.7% vs -20.1%) | FAIL |

| 1956-10-31 → 2026-11-11 | Base | Base + Q1 |
|---|---:|---:|
| Sharpe | 0.53 | 0.55 |
| Return a year | 9.3% | 10.2% |
| Worst fall | -28.5% | -32.7% |
| Trades a year | 6.1 | 7.6 |
| Share of days the flow key is "in" | | 48% |

## Q2 equity fund + ETF flows (fade the crowd): **FAIL**

| Gate | Evidence | Result |
|---|---|---|
| G1 Sharpe better, full and both halves | full 0.484 vs 0.527; 1955–1990 0.380 vs 0.432; 1991–end 0.570 vs 0.609 | FAIL |
| G2 worst fall not deeper by > 2 pts | -35.8% vs -28.5% | FAIL |
| G3 bootstrap not-better share < 25% | 73.8% | FAIL |
| G4 placebo p <= 0.05 | p = 0.483 (placebo median Sharpe 0.482) | FAIL |
| G5 real product (1999-02 → end) not worse | 0.477 vs 0.713 (worst fall -32.7% vs -20.1%) | FAIL |

| 1956-10-31 → 2026-11-11 | Base | Base + Q2 |
|---|---:|---:|
| Sharpe | 0.53 | 0.48 |
| Return a year | 9.3% | 9.5% |
| Worst fall | -28.5% | -35.8% |
| Trades a year | 6.1 | 8.1 |
| Share of days the flow key is "in" | | 55% |

## Information only

Correlation of Q1 and Q2 z-scores (quarterly): +0.10.

### Q1 z quintile → next S&P total return (full)

| Quintile | Days | Mean z | Next 1 year | Next 3 years |
|---|---:|---:|---:|---:|
| 1 low | 3,550.0 | -1.72 | +9.2% | +38.5% |
| 2 | 3,523.0 | -0.58 | +15.4% | +50.2% |
| 3 | 3,560.0 | +0.01 | +13.1% | +38.2% |
| 4 | 3,525.0 | +0.95 | +8.9% | +32.3% |
| 5 high | 3,468.0 | +2.22 | +13.1% | +36.5% |

### Q1 z quintile → next S&P total return (1955–1990)

| Quintile | Days | Mean z | Next 1 year | Next 3 years |
|---|---:|---:|---:|---:|
| 1 low | 1,767.0 | -1.81 | +7.0% | +23.2% |
| 2 | 1,724.0 | +0.01 | +11.2% | +37.9% |
| 3 | 1,697.0 | +0.92 | +9.8% | +38.1% |
| 4 | 1,705.0 | +1.41 | +9.0% | +41.4% |
| 5 high | 1,702.0 | +2.54 | +21.6% | +47.5% |

### Q1 z quintile → next S&P total return (1991–end)

| Quintile | Days | Mean z | Next 1 year | Next 3 years |
|---|---:|---:|---:|---:|
| 1 low | 1,846.0 | -1.62 | +12.9% | +58.8% |
| 2 | 1,821.0 | -0.68 | +12.8% | +52.0% |
| 3 | 1,765.0 | -0.38 | +16.1% | +42.9% |
| 4 | 1,839.0 | +0.01 | +14.8% | +42.5% |
| 5 high | 1,760.0 | +1.51 | +4.7% | +9.5% |

### Q2 z quintile → next S&P total return (full)

| Quintile | Days | Mean z | Next 1 year | Next 3 years |
|---|---:|---:|---:|---:|
| 1 low | 3,577.0 | -2.00 | +11.4% | +34.8% |
| 2 | 3,519.0 | -0.86 | +10.5% | +30.6% |
| 3 | 3,524.0 | -0.20 | +4.9% | +23.8% |
| 4 | 3,531.0 | +0.53 | +14.4% | +39.4% |
| 5 high | 3,475.0 | +1.90 | +18.5% | +64.9% |

### Q2 z quintile → next S&P total return (1955–1990)

| Quintile | Days | Mean z | Next 1 year | Next 3 years |
|---|---:|---:|---:|---:|
| 1 low | 1,751.0 | -2.22 | +3.1% | +28.3% |
| 2 | 1,699.0 | -0.80 | +10.8% | +26.1% |
| 3 | 1,754.0 | +0.12 | +9.8% | +33.7% |
| 4 | 1,688.0 | +0.78 | +16.7% | +42.9% |
| 5 high | 1,703.0 | +2.24 | +18.3% | +57.0% |

### Q2 z quintile → next S&P total return (1991–end)

| Quintile | Days | Mean z | Next 1 year | Next 3 years |
|---|---:|---:|---:|---:|
| 1 low | 1,824.0 | -1.79 | +19.2% | +41.9% |
| 2 | 1,826.0 | -0.90 | +12.5% | +35.5% |
| 3 | 1,771.0 | -0.37 | +1.7% | +20.2% |
| 4 | 1,826.0 | +0.23 | +7.3% | +27.6% |
| 5 high | 1,784.0 | +1.48 | +20.2% | +73.2% |

Forward windows overlap; day counts overstate the independent evidence (about 70 years ≈ 70 independent 1-year periods).
