# ML sizing of the shock re-entry — the one run (spec 2026-11-18, frozen in section 12)

**Verdict (M-L, the gate model): NO CHANGE: gates failed outside the declared fallbacks; result recorded**

M-G replaces M-L: no (cluster bootstrap P(M-G log-loss >= M-L) = 0.388).

Frozen features.csv SHA-256 `afc30914086be91c7b8e98518be8dcd45cb104628ead82e7f6dc0c8cbf8a0706`; data cut 2026-11-11. Out-of-sample with a label: 156 events, 43 clusters, failed clusters (cluster-weighted) 14.1.

| Gate | M-L | Evidence (M-L) | M-G (report unless M-L passes all) |
|---|---|---|---|
| G1 | FAIL | AUC 0.563 (cluster-level 0.586), permutation p 0.0748 | fail: AUC 0.511 (cluster-level 0.486), permutation p 0.6382 |
| G2 | FAIL | slope 0.11; Brier 0.2278 vs base rate 0.2278 | fail: slope -0.31; Brier 0.2314 vs base rate 0.2278 |
| G3 | FAIL | K +0.00464, permutation p 0.0250 | fail: K -0.00469, permutation p 0.9966 |
| G4 | FAIL | mean Sharpe diff -0.0087, positive in 4 of 21; worst fall within 2 pts in 17 of 21, deepest extra -7.6% | fail: mean Sharpe diff -0.0018, positive in 3 of 21; worst fall within 2 pts in 19 of 21, deepest extra -3.8% |
| G5 | PASS | K 2000–09 +0.00113; 2010–26 +0.00694; without 2020 +0.00093; without 2008–09 +0.00438; regions asia_pac_dev +0.00088, emerging +0.00249, europe +0.00813, north_america +0.00000 | fail: K 2000–09 -0.01024; 2010–26 -0.00106; without 2020 -0.00489; without 2008–09 -0.00458; regions asia_pac_dev -0.00452, emerging -0.00639, europe -0.00347, north_america -0.00011 |
| G6 | FAIL | 94 events / 32 clusters: AUC 0.564, p 0.0836, K +0.00603 | fail: 94 events / 32 clusters: AUC 0.535, p 0.5234, K -0.00596 |
| G7 | PASS | Sharpe 0.715 vs core 0.715; worst fall -20.1% vs -20.1%. Core S episodes scored: 2020-03-25 p 0.41 (pi 0.39) normal; 2025-04-24 p 0.46 (pi 0.39) normal | pass: Sharpe 0.715 vs core 0.715; worst fall -20.1% vs -20.1%. Core S episodes scored: 2020-03-25 p 0.27 (pi 0.39) normal; 2025-04-24 p 0.43 (pi 0.39) normal |
| G8 | FAIL | K +0.00464 vs only days_le +0.00469, only vol_rel -0.00022, only runup_2y +0.00034 | fail: K -0.00469 vs only days_le +0.00469, only vol_rel -0.00022, only runup_2y +0.00034 |

## Benchmarks (pooled out-of-sample 2000–2026, cluster weights)

| Model | AUC | Brier | Log-loss | K |
|---|---:|---:|---:|---:|
| M-L | 0.563 | 0.2278 | 0.6811 | +0.00464 |
| M-G | 0.511 | 0.2314 | 0.6570 | -0.00469 |
| base rate | 0.457 | 0.2278 | 0.6486 | +0.00000 |
| only days_le | 0.635 | 0.2094 | 0.6095 | +0.00469 |
| only vol_rel | 0.527 | 0.2243 | 0.6410 | -0.00022 |
| only runup_2y | 0.481 | 0.2289 | 0.6546 | +0.00034 |

M-L does not beat its best single feature (only days_le) on K: not worth its complexity.

## By block (M-L)

| Block | Events | Clusters | pi (train) | AUC | K | Buckets |
|---|---:|---:|---:|---:|---:|---|
| 2000–2004 | 28 | 10 | 0.352 | 0.522 | -0.00007 | {'low': 14, 'normal': 13, 'high': 1} |
| 2005–2009 | 29 | 7 | 0.402 | 0.610 | +0.00284 | {'normal': 24, 'high': 3, 'low': 2} |
| 2010–2015 | 31 | 10 | 0.416 | 0.635 | +0.00124 | {'normal': 18, 'low': 13} |
| 2016–2026 | 68 | 16 | 0.393 | 0.671 | +0.01050 | {'normal': 47, 'low': 21} |

## G4 by market (M-L, 2000 → data end)

| Market | Episodes scored | Sharpe 11d | Sharpe model | Worst fall 11d | model |
|---|---:|---:|---:|---:|---:|
| GSPC | 3 | 0.596 | 0.596 | -18.7% | -18.7% |
| SXXP | 7 | 0.219 | 0.204 | -33.4% | -33.4% |
| N225 | 7 | 0.224 | 0.224 | -48.1% | -48.1% |
| TWII | 4 | 0.452 | 0.452 | -28.9% | -28.9% |
| GSPTSE | 1 | 0.655 | 0.655 | -21.2% | -21.2% |
| HSI | 6 | 0.086 | 0.076 | -44.4% | -44.4% |
| KS11 | 6 | 0.519 | 0.519 | -39.6% | -39.6% |
| AXJO | 5 | 0.192 | 0.192 | -35.4% | -35.4% |
| AEX | 5 | 0.267 | 0.244 | -30.4% | -32.8% |
| ATX | 7 | 0.655 | 0.668 | -28.0% | -27.9% |
| BFX | 6 | 0.333 | 0.310 | -27.4% | -30.0% |
| BSESN | 7 | 0.581 | 0.578 | -38.8% | -38.8% |
| BVSP | 10 | 0.375 | 0.379 | -34.5% | -34.5% |
| FCHI | 7 | 0.105 | 0.091 | -41.2% | -41.2% |
| GDAXI | 4 | 0.359 | 0.317 | -33.0% | -40.6% |
| IBEX | 6 | 0.204 | 0.193 | -48.5% | -47.5% |
| MXX | 7 | 0.383 | 0.383 | -34.0% | -34.0% |
| NZ50 | 1 | 0.843 | 0.843 | -15.7% | -15.7% |
| OMXS30 | 5 | 0.406 | 0.411 | -30.5% | -30.5% |
| SSMI | 5 | 0.229 | 0.187 | -29.6% | -36.7% |
| STOXX50E | 6 | 0.035 | 0.011 | -43.7% | -43.7% |

## US (never evidence)

US walk-forward events (in-sample for design): 2010-08-02 p 0.34 normal (stopped 0); 2020-03-25 p 0.41 normal (stopped 0); 2023-04-24 p 0.22 low (stopped 0); 2025-04-24 p 0.46 normal (stopped 0)

G7 detail: Sharpe 0.715 vs core 0.715; worst fall -20.1% vs -20.1%. Core S episodes scored: 2020-03-25 p 0.41 (pi 0.39) normal; 2025-04-24 p 0.46 (pi 0.39) normal

## Report-only lines

- Leave-one-calendar-year-out (M-L, trains on later data too): AUC 0.573. US 1987-10-21: p 0.32.
- M-L on all labelled events (expected sign; share of 1,000 cluster bootstraps with the same sign): runup_2y +0.055 (expected +; 62%); attempt +0.263 (expected +; 90%); days_le -0.633 (expected -; 100%); vol_rel -0.049 (expected open; 56%); fx_shock +0.013 (expected +; 50%); d_credit -0.037 (expected +; 58%).
- US 1929–1954 events, M-L fitted on all labelled events (pi 0.35): 1929-10-30 p 0.37 normal (stopped 1); 1929-11-15 p 0.44 normal (stopped 0); 1930-09-18 p 0.19 low (stopped 1); 1932-12-09 p 0.29 normal (stopped 1); 1933-07-27 p 0.46 normal (stopped 1); 1941-01-09 p 0.18 low (stopped 0).
- Extended model (report only, REVISED DATA, never a gate): AUC 0.487, K -0.00453 (coverage: local 10y 50%, curve 47%, CLI 67%).

## Data notes (daily moves above 25%, not removed)

- GSPC: none
- SXXP: none
- N225: none
- TWII: none
- GSPTSE: none
- HSI: 1987-10-26 -33%
- KS11: none
- AXJO: none
- AEX: none
- ATX: none
- BFX: none
- BSESN: none
- BVSP: 1995-03-10 +26%, 1999-01-15 +33%
- FCHI: none
- GDAXI: none
- IBEX: none
- MXX: 1987-11-18 +27%
- NZ50: none
- OMXS30: 1998-04-24 -75%
- SSMI: none
- STOXX50E: none

Code: `research/us_gauge/ml/` (run_m.py). One run; no second run with other features, labels, thresholds, blocks, buckets or markets.
