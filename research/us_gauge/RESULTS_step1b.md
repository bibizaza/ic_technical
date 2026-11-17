Re-run 2026-11-17 with corrected cash leg (BIL.US_TR); see spec correction log.

# US quant gauge — step 1b results

First decision: 1993-04-06. Common return sample: 1993-04-08 to 2026-11-13 (8,459 trading days).
H1 ends 2009-12-31; H2 starts 2010-01-01. Halves slice the FULL simulation without restarting state.

Three candidates (R1, R2, R3) have now been tried on the US. A US PASS qualifies for the unchanged Europe check; it is not adoption.

## R2 vs R0 — FAIL (all)

| Condition | Evidence | Result |
|---|---|---|
| FULL Sharpe > R0 | 0.562861 > 0.602609 | FAIL |
| H1 Sharpe > R0 | 0.386853 > 0.440433 | FAIL |
| H2 Sharpe > R0 | 0.735078 > 0.755099 | FAIL |
| FULL maxDD >= R0 maxDD - 0.02 | -0.370661 >= -0.259724 | FAIL |
| Bootstrap not-better share < 0.25 | 0.8355 < 0.25 | FAIL |
| FULL switches/year <= R0 | 2.770540 <= 2.353470 | FAIL |

Bootstrap R2 <= R0: all 0.8355; half 0.8745.

## R3 vs R0 — FAIL (all)

| Condition | Evidence | Result |
|---|---|---|
| FULL Sharpe > R0 | 0.642434 > 0.602609 | PASS |
| H1 Sharpe > R0 | 0.527174 > 0.440433 | PASS |
| H2 Sharpe > R0 | 0.746235 > 0.755099 | FAIL |
| FULL maxDD >= R0 maxDD - 0.02 | -0.299371 >= -0.259724 | FAIL |
| Bootstrap not-better share < 0.25 | 0.3630 < 0.25 | FAIL |
| FULL switches/year <= R0 | 1.549119 <= 2.353470 | PASS |

Bootstrap R3 <= R0: all 0.3630; half 0.3885.

Paired moving blocks: 63 trading days, 2,000 draws, numpy.random.default_rng(7), separately for each candidate/version.
All versus half remains the user’s decision; only all enters the gate.

## R3 vs BH

| Period | R3 version | R3 Sharpe | BH Sharpe | R3 maxDD | BH maxDD |
|---|---|---:|---:|---:|---:|
| FULL | all | 0.642434 | 0.518978 | -0.299371 | -0.552546 |
| FULL | half | 0.623915 | 0.518978 | -0.356591 | -0.552546 |
| H1 | all | 0.527174 | 0.297232 | -0.299371 | -0.552546 |
| H1 | half | 0.425281 | 0.297232 | -0.356591 | -0.552546 |
| H2 | all | 0.746235 | 0.771910 | -0.230503 | -0.337936 |
| H2 | half | 0.820119 | 0.771910 | -0.230503 | -0.337936 |

## R2 earnings re-entries

Count: 11. These are Cash-to-Invested transitions into earnings mode, including any on a monthly decision date (spec 6b allows any decision date).

1994-04-05, 2000-03-21, 2000-06-06, 2002-03-05, 2002-07-09, 2002-12-31, 2003-02-18, 2015-06-16, 2015-07-21, 2016-01-05, 2022-02-22

## Latest R2 reading

Decision date: 2026-11-10. pos: Cash; mode: trend; armed: False; DMAS: 39.000000; REV_z: 0.929475.

## Method and validation

- Six cached source series loaded with load(name, refresh=False); no network, Bloomberg, or Dropbox writes.
- Existing step-1 signals, return construction, metrics, bootstrap, and drawdown functions are reused unchanged. No sensitivities or other variants were run.
- Signal truncation and state-machine prefix tests, plus t+2 exposure tests, passed before this run.
- Full metrics for BH/R0/R1/R2/R3 and both exposure versions are in out/step1b_metrics.csv; five worst drawdown episodes per run are in out/step1b_drawdowns.csv; decision states are in out/step1b_positions_r2.csv.

## Deviations

- No step-1b threshold, rule, window, or gate changes.
- Inherited step-1 mechanics: costs are debited on the new-weight return day (t+2); the initial allocation has no switch cost; evaluation starts at the first effective weight. This preserves the existing convention despite the previously documented step-1 plan conflict about cost timing.
- Monthly grouping uses the actual decision date’s calendar month, including holiday adjustments, as specified in 6b.
- Normal trend re-entry leaves armed unchanged: 6b explicitly resets it only on earnings re-entry and on every exit. The retained flag has no effect while invested.
