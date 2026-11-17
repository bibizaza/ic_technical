Re-run 2026-11-17 with corrected cash leg (BIL.US_TR); see spec correction log.

# US quant gauge — step 1 results

Verdict: FAIL

First decision: 1993-04-06. Common return sample: 1993-04-08 to 2026-11-13 (8,459 trading days).
H1 ends 2009-12-31; H2 starts 2010-01-01. Both are slices of the FULL simulation.

## Pre-registered gate (all version)

| Condition | Evidence | Result |
|---|---|---|
| Sharpe R1 > R0 in FULL, H1 and H2 | FULL: 0.705737 > 0.602609 (PASS); H1: 0.598689 > 0.440433 (PASS); H2: 0.802953 > 0.755099 (PASS) | PASS |
| FULL maxdd R1 >= maxdd R0 - 0.02 | -0.350627 >= -0.259724 | FAIL |
| Paired bootstrap not-better share < 0.25 | 0.1255 < 0.25 | PASS |
| FULL switches/year R1 <= R0 | 1.370375 <= 2.353470 | PASS |

## All / half comparison (FULL)

| Rule | Version | Sharpe | CAGR | Max DD | Calmar | Volatility | Switches/year | Share in Cash |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BH | none | 0.518978 | 10.84% | -55.25% | 0.196169 | 18.35% | 0.000000 | 0.00% |
| R0 | all | 0.602609 | 8.78% | -23.97% | 0.366396 | 10.93% | 2.353470 | 32.42% |
| R0 | half | 0.611259 | 10.11% | -34.71% | 0.291216 | 13.17% | 2.353470 | 32.42% |
| R1 | all | 0.705737 | 11.26% | -35.06% | 0.321170 | 12.84% | 1.370375 | 19.94% |
| R1 | half | 0.644741 | 11.29% | -36.88% | 0.306139 | 14.41% | 1.370375 | 19.94% |

Bootstrap R1 <= R0: all 0.1255; half 0.1815.
Paired moving blocks: 63 trading days, 2,000 draws, numpy.random.default_rng(7).
All versus half remains the user’s decision; the gate applies only to all.

## Latest gauge

Decision date: 2026-11-10. State: Invested. G: 0.929475. Band: 0.5-1. Switch-risk grade: low. DMAS: 39. REV_z: 0.929475.

## Sensitivity — information only

Only the pre-registered REV_z threshold -0.5 and revision window 126 trading days were run, separately, for all and half. FULL/H1/H2 metrics are in out/step1_sensitivity.csv. They do not select the rule or affect the gate.

## Method and validation

- All six source series were loaded with load(name, refresh=False); no fetch or store writes.
- DMAS calls the production four-factor functions without a benchmark.
- The five-date 1995–2025 signal truncation check passed before results were generated; trading tests verify t+2 return exposure.
- Standard deviations use ddof=1; annualization uses 252 trading days. Cash-state share counts days with equity weight below 1, including the half version.
- Switch-risk grades use evaluation-period decision weeks and exclude terminal observations without a complete forward horizon, separately for 4 and 13 weeks; denominators are in the CSV. Grades are descriptive full-sample estimates.
- Drawdowns are non-overlapping peak-to-recovery episodes; blank recovery means unrecovered. Initial capital is included in the high-water mark.

## Deviations

- Per user instruction, skipped the already completed cash-data fetch and all commit/push steps.
- The plan conflicts on cost timing: global constraints say execution day (t+1), but Task 4 explicitly charges on the new-weight return day (t+2). This implementation follows Task 4. Exposure also starts at t+2, after execution at the next close.
- For a holiday Tuesday mapped into the preceding month (for example January 1 mapped to December 31), monthly grouping follows Task 4: the actual decision date’s calendar month. This differs from a literal calendar-Tuesday interpretation of spec 6.2; no alternate rule was run.
- The initial position is undefined until the first decision has executed. All rules use the common first defined return (two trading rows after the first April 1993 decision); no initial-allocation switch cost is assumed.
- No thresholds, windows, trading rules, or gates were fitted to results; only the two approved information-only sensitivities were run.
