Re-run 2026-11-17 with corrected cash leg (BIL.US_TR); see spec correction log.

# US quant gauge — step 1c results

Verdict: FAIL

First decision: 1993-04-06. Common return sample: 1993-04-08 to 2026-11-13 (8,459 trading days).
H1 ends 2009-12-31; H2 starts 2010-01-01; P2021 starts 2021-01-01. All sub-periods slice the FULL simulation without restarting state. P2021 is information only.

## Pre-registered gate (all version)

| Condition | Evidence | Result |
|---|---|---|
| FULL Sharpe R4 > R1 | 0.623880 > 0.705737 | FAIL |
| H1 Sharpe R4 > R1 | 0.458468 > 0.598689 | FAIL |
| H2 Sharpe R4 > R1 | 0.778472 > 0.802953 | FAIL |
| FULL Sharpe R4 > BH | 0.623880 > 0.518978 | PASS |
| Bootstrap R4 <= R1 share < 0.25 | 0.9245 < 0.25 | FAIL |
| FULL switches/year R4 <= R0 | 1.727864 <= 2.353470 | PASS |

Bootstrap R4 <= R1: all 0.9245; half 0.8970.
Paired moving blocks: 63 trading days, 2,000 draws, numpy.random.default_rng(7), separately for each version. Only all enters the gate; max drawdown is reported, not gated.
Four candidates have now been tried on the US. Any PASS still needs the unchanged Europe check; earlier verdicts remain unchanged.

## Metrics

CAGR and maxDD are percentages. Sharpe uses excess returns over the unchanged cash series.

| Period | Rule | Version | Sharpe | CAGR | maxDD | Switches/year |
|---|---|---|---:|---:|---:|---:|
| FULL | BH | none | 0.518978 | 10.8392% | -55.2546% | 0.000000 |
| FULL | R0 | all | 0.602609 | 8.7834% | -23.9724% | 2.353470 |
| FULL | R1 | all | 0.705737 | 11.2611% | -35.0627% | 1.370375 |
| FULL | R1 | half | 0.644741 | 11.2911% | -36.8823% | 1.370375 |
| FULL | R2 | all | 0.562861 | 8.5294% | -37.0661% | 2.770540 |
| FULL | R3 | all | 0.642434 | 9.6670% | -29.9371% | 1.549119 |
| FULL | R4 | all | 0.623880 | 9.6048% | -35.7339% | 1.727864 |
| FULL | R4 | half | 0.614039 | 10.4904% | -35.3801% | 1.727864 |
| H1 | BH | none | 0.297232 | 7.7100% | -55.2546% | 0.000000 |
| H1 | R0 | all | 0.440433 | 7.9201% | -23.9724% | 2.390892 |
| H1 | R1 | all | 0.598689 | 10.5534% | -35.0627% | 1.195446 |
| H1 | R1 | half | 0.455509 | 9.4434% | -36.8823% | 1.195446 |
| H1 | R2 | all | 0.386853 | 7.5116% | -37.0661% | 2.749526 |
| H1 | R3 | all | 0.527174 | 9.0633% | -29.9371% | 1.554080 |
| H1 | R4 | all | 0.458468 | 8.4834% | -35.7339% | 1.554080 |
| H1 | R4 | half | 0.397062 | 8.4320% | -35.3801% | 1.554080 |
| H2 | BH | none | 0.771910 | 14.0386% | -33.7936% | 0.000000 |
| H2 | R0 | all | 0.755099 | 9.6481% | -15.7888% | 2.316286 |
| H2 | R1 | all | 0.802953 | 11.9688% | -21.7041% | 1.544190 |
| H2 | R1 | half | 0.830306 | 13.1580% | -22.3587% | 1.544190 |
| H2 | R2 | all | 0.735078 | 9.5503% | -15.7888% | 2.791421 |
| H2 | R3 | all | 0.746235 | 10.2702% | -23.0503% | 1.544190 |
| H2 | R4 | all | 0.778472 | 10.7306% | -17.4828% | 1.900542 |
| H2 | R4 | half | 0.834117 | 12.5745% | -21.1835% | 1.900542 |
| P2021 | BH | none | 0.699303 | 14.2748% | -24.5330% | 0.000000 |
| P2021 | R0 | all | 0.843617 | 12.2152% | -11.4533% | 1.880597 |
| P2021 | R1 | all | 0.763435 | 13.1676% | -21.7041% | 1.367707 |
| P2021 | R1 | half | 0.766609 | 13.8585% | -22.3587% | 1.367707 |
| P2021 | R2 | all | 0.903563 | 13.3397% | -10.5532% | 2.222524 |
| P2021 | R3 | all | 0.691806 | 11.6977% | -23.0503% | 1.367707 |
| P2021 | R4 | all | 0.763927 | 12.0856% | -13.9241% | 1.709634 |
| P2021 | R4 | half | 0.783577 | 13.3667% | -17.2195% | 1.709634 |

## Latest R4 reading

Decision date: 2026-11-10. State: Invested (scheduled decision state). DMAS: 39.000000; REV_z: 0.929475; VIX 21d average: 19.082381.

## Method and validation

- Seven cached source series loaded with load(name, refresh=False); no network, Bloomberg, or Dropbox writes.
- VIX is aligned to S&P trading days using the last observation on or before each day; the 21-day trailing mean includes the decision day. Unavailable means are not high.
- Existing signals, sleeve_weights (deep = DMAS < 30), returns, metrics, bootstrap and drawdown functions are reused. No sensitivities or variants were run.
- out/step1c_metrics.csv contains all rules and versions for all four periods; out/step1c_drawdowns.csv contains the five worst FULL-run drawdown episodes per rule/version.

## Deviations

- No step-1c threshold, rule, window, or gate changes.
- Inherited step-1 convention: costs are debited on the new-weight return day (t+2); initial allocation has no switch cost; evaluation starts at the first effective weight. This retains the previously documented step-1 cost-timing convention.
