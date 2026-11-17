# US quant gauge — step 1c results

Verdict: FAIL

First decision: 1993-04-06. Common return sample: 1993-04-08 to 2026-11-13 (8,459 trading days).
H1 ends 2009-12-31; H2 starts 2010-01-01; P2021 starts 2021-01-01. All sub-periods slice the FULL simulation without restarting state. P2021 is information only.

## Pre-registered gate (all version)

| Condition | Evidence | Result |
|---|---|---|
| FULL Sharpe R4 > R1 | 0.244155 > 0.313954 | FAIL |
| H1 Sharpe R4 > R1 | 0.468245 > 0.607989 | FAIL |
| H2 Sharpe R4 > R1 | 0.176426 > 0.219392 | FAIL |
| FULL Sharpe R4 > BH | 0.244155 > 0.293512 | FAIL |
| Bootstrap R4 <= R1 share < 0.25 | 0.9635 < 0.25 | FAIL |
| FULL switches/year R4 <= R0 | 1.727864 <= 2.353470 | PASS |

Bootstrap R4 <= R1: all 0.9635; half 0.9490.
Paired moving blocks: 63 trading days, 2,000 draws, numpy.random.default_rng(7), separately for each version. Only all enters the gate; max drawdown is reported, not gated.
Four candidates have now been tried on the US. Any PASS still needs the unchanged Europe check; earlier verdicts remain unchanged.

## Metrics

CAGR and maxDD are percentages. Sharpe uses excess returns over the unchanged cash series.

| Period | Rule | Version | Sharpe | CAGR | maxDD | Switches/year |
|---|---|---|---:|---:|---:|---:|
| FULL | BH | none | 0.293512 | 10.8392% | -55.2546% | 0.000000 |
| FULL | R0 | all | 0.205355 | 8.5170% | -23.9724% | 2.353470 |
| FULL | R1 | all | 0.313954 | 11.0657% | -35.0627% | 1.370375 |
| FULL | R1 | half | 0.314728 | 11.1935% | -36.8823% | 1.370375 |
| FULL | R2 | all | 0.193964 | 8.2637% | -37.0661% | 2.770540 |
| FULL | R3 | all | 0.239800 | 9.2945% | -29.9371% | 1.549119 |
| FULL | R4 | all | 0.244155 | 9.3995% | -35.7339% | 1.727864 |
| FULL | R4 | half | 0.283256 | 10.3870% | -35.3801% | 1.727864 |
| H1 | BH | none | 0.309898 | 7.7100% | -55.2546% | 0.000000 |
| H1 | R0 | all | 0.449011 | 7.7513% | -23.9724% | 2.390892 |
| H1 | R1 | all | 0.607989 | 10.4063% | -35.0627% | 1.195446 |
| H1 | R1 | half | 0.468073 | 9.3706% | -36.8823% | 1.195446 |
| H1 | R2 | all | 0.394935 | 7.3435% | -37.0661% | 2.749526 |
| H1 | R3 | all | 0.537479 | 8.9180% | -29.9371% | 1.554080 |
| H1 | R4 | all | 0.468245 | 8.3388% | -35.7339% | 1.554080 |
| H1 | R4 | half | 0.409947 | 8.3597% | -35.3801% | 1.554080 |
| H2 | BH | none | 0.292921 | 14.0386% | -33.7936% | 0.000000 |
| H2 | R0 | all | 0.134455 | 9.2833% | -15.7887% | 2.316286 |
| H2 | R1 | all | 0.219392 | 11.7247% | -21.7041% | 1.544190 |
| H2 | R1 | half | 0.261755 | 13.0348% | -22.3587% | 1.544190 |
| H2 | R2 | all | 0.131580 | 9.1859% | -15.7887% | 2.791421 |
| H2 | R3 | all | 0.149760 | 9.6699% | -23.0503% | 1.544190 |
| H2 | R4 | all | 0.176426 | 10.4638% | -17.4828% | 1.900542 |
| H2 | R4 | half | 0.242461 | 12.4390% | -21.1835% | 1.900542 |
| P2021 | BH | none | 0.886882 | 14.2748% | -24.5330% | 0.000000 |
| P2021 | R0 | all | 1.070509 | 11.4191% | -12.9403% | 1.880597 |
| P2021 | R1 | all | 0.959095 | 12.5802% | -21.7041% | 1.367707 |
| P2021 | R1 | half | 0.968068 | 13.5634% | -22.3587% | 1.367707 |
| P2021 | R2 | all | 1.121234 | 12.5356% | -10.5532% | 2.222524 |
| P2021 | R3 | all | 0.844381 | 10.3734% | -23.0503% | 1.367707 |
| P2021 | R4 | all | 0.979062 | 11.4300% | -14.0805% | 1.709634 |
| P2021 | R4 | half | 0.998405 | 13.0355% | -17.4329% | 1.709634 |

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
