# Fast stress de-risking: merged research (3 agents, 2026-11-23)

Sources: three research agents (academic papers; hedge fund / QIS practice; lesser-known practitioner rules). Links are in their reports; marked "unverified" where the agent saw only a snippet. Nothing below has been tested in this project.

## What the literature says first
- Most stress measures are **risk-premium signals**: when high, future returns are *higher* (variance risk premium, tail risk, average correlation, illiquidity, VIX backwardation). Used as exits they cut the fall but miss the rebound — the trap that sank the VIX-level key.
- **Fast exits lose in V-shaped sell-offs** (Feb 2018, Aug 2024, Apr 2025) and pay in steady slides (Oct 2008, Q4 2018, Mar 2020). Momentum Turning Points (Goulding–Harvey–Mazzoleni, JFE 2023): after a "correction" month (12-month up, 1-month down) the *slower* trend is better. Do not just make the trend key faster.
- The fast channel that survives out of sample is **volatility persistence**, which the product already uses: vol sizing is daily, not locked (it cut 100% → 75% on 6 Nov and 75% → 65% on 17 Nov 2026).

## Merged ranking (agreement across agents × independence from what failed)

| # | Idea | Rule (fix before any run) | Agents | Data | Main risk |
|---|---|---|---|---|---|
| 1 | Tranched month lock | 4 sub-books lock on trading days 1, 6, 11, 16; exposure = average; DMAS < 30 exit kept | 2 of 3 (top) | none new | no new information; small gain |
| 2 | Bond / EM "canary" (Keller DAA) | 13612W momentum of a bond index and EM equities; one negative AND DMAS < 45 → apply the cash rule today | 2 of 3 | AGG/EEM (EODHD) 2003+; bond proxy from yields before | canary chosen by the authors; often negative |
| 3 | Credit-confirmed fast exit | HY OAS 20-day change above its 90th percentile AND S&P below 50-day average → exit or cap 50% inside the month | 1 of 3 (top), 2 mention | `BBG_LF98OAS` (stored, 1994+) or Baa−10y 1986+ | weak academic support; energy-led spread moves (2015–16) |
| 4 | Persistence filter (jump model) | 2-state model on downside deviation and Sortino; state change charged a penalty; replaces DMAS < 30 | 1 | S&P only | one paper; Sharpe 0.68 < product 0.71 |
| 5 | Confirmed 5% canary (Thrasher, NAAIM 2023) | −5% from the 1-year high within 15 days, then 2 closes below the 200-day within 42 days → unlock the month | 1 | S&P only | n = 15 since 1980 |
| 6 | Turbulence / absorption-ratio shift (Kritzman) | 15-day vs 1-year absorption ratio on 49 French industries, z > +1 → 50% | 2 | French 49 industries 1926+ (stored) | authors' own evidence; correlation level predicts *higher* returns |
| 7 | Negative variance risk premium | VIX²/12 − 21-day realized variance < 0 for 3 days → cap 50% | 1 | VIX 1990+, S&P | rare, fires after the first leg |
| 8 | Downside semi-vol sizing | 15% / (√2 × 21-day downside deviation) | 1 | S&P | close to the failed vol-estimator family |

Low value or rejected by all agents: Hindenburg Omen, IBD distribution days, 90% down days (late), VIX/VIX3M inversion (marks lows), VVIX, SKEW, dealer gamma (paid, intraday), stress indices (overlap, revised), tail hedges (cost), stop-loss/CPPI (= trend), NAAIM, margin debt, LPPL, copper/gold, utilities beta rotation (poor live record).

## Readings the agents saw (before the 20 Nov drop)
- HY OAS 3.23% (13 Nov) → 3.52% (19 Nov): rising, far from stress levels (FRED snippet).
- St. Louis FSI −0.87 (13 Nov): below-average stress.
