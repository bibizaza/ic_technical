Re-run 2026-11-17 with corrected cash leg (BIL.US_TR); see spec correction log.

# US quant gauge — step 2 results

Verdict: FAIL

First decision: 2008-07-01. Common return sample: 2008-07-03 to 2026-11-13 (4,621 trading days).
WH1 ends 2017-06-30; WH2 starts 2017-07-01; P2021 starts 2021-01-01 (information only). R0/R1/BH are slices of their full runs; R5 starts at the July 2008 monthly decision. Sub-periods do not restart state.

## Pre-registered gate (all version)

| Condition | Evidence | Result |
|---|---|---|
| WIN Sharpe R5 > R1 | 0.811494 > 0.830796 | FAIL |
| WH1 Sharpe R5 > R1 | 0.739989 > 0.830632 | FAIL |
| WH2 Sharpe R5 > R1 | 0.875423 > 0.830986 | PASS |
| WIN Sharpe R5 > BH | 0.811494 > 0.619510 | PASS |
| Bootstrap R5 <= R1 share < 0.25 | 0.5865 < 0.25 | FAIL |
| WIN switches/year R5 <= R0 | 2.508548 <= 2.290413 | FAIL |

Bootstrap R5 <= R1: all 0.5865; half 0.4745.
Paired moving blocks: 63 trading days, 2,000 draws, numpy.random.default_rng(7), separately for each version. Only all enters the gate; max drawdown is reported, not gated.
About 18 years and few bear markets imply low power. Five candidates have now been tried on the US; any PASS still needs a Europe check with EUR-relevant CoT.

## Metrics

CAGR and maxDD are percentages. Sharpe uses excess returns over cash. BH is identical in both exposure versions.

| Period | Rule | Version | Sharpe | CAGR | maxDD | Switches/year |
|---|---|---|---:|---:|---:|---:|
| WIN | BH | all | 0.619510 | 12.3139% | -47.2603% | 0.000000 |
| WIN | BH | half | 0.619510 | 12.3139% | -47.2603% | 0.000000 |
| WIN | R0 | all | 0.731058 | 9.2037% | -15.7888% | 2.290413 |
| WIN | R0 | half | 0.738491 | 11.1228% | -25.6118% | 2.290413 |
| WIN | R1 | all | 0.830796 | 12.1510% | -21.7041% | 1.472409 |
| WIN | R1 | half | 0.767582 | 12.5362% | -25.6118% | 1.472409 |
| WIN | R5 | all | 0.811494 | 9.7990% | -19.0283% | 2.508548 |
| WIN | R5 | half | 0.772895 | 11.4406% | -25.6118% | 2.508548 |
| WH1 | BH | all | 0.547892 | 9.8528% | -47.2603% | 0.000000 |
| WH1 | BH | half | 0.547892 | 9.8528% | -47.2603% | 0.000000 |
| WH1 | R0 | all | 0.668566 | 6.8297% | -15.7888% | 2.558940 |
| WH1 | R0 | half | 0.666604 | 8.7768% | -25.6118% | 2.558940 |
| WH1 | R1 | all | 0.830632 | 10.6349% | -18.7048% | 1.668874 |
| WH1 | R1 | half | 0.724152 | 10.6145% | -25.6118% | 1.668874 |
| WH1 | R5 | all | 0.739989 | 7.2778% | -19.0283% | 2.558940 |
| WH1 | R5 | half | 0.694589 | 9.0199% | -25.6118% | 2.558940 |
| WH2 | BH | all | 0.699045 | 14.7320% | -33.7936% | 0.000000 |
| WH2 | BH | half | 0.699045 | 14.7320% | -33.7936% | 0.000000 |
| WH2 | R0 | all | 0.787184 | 11.5358% | -12.7094% | 2.032258 |
| WH2 | R0 | half | 0.808609 | 13.4258% | -21.1835% | 2.032258 |
| WH2 | R1 | all | 0.830986 | 13.6282% | -21.7041% | 1.283531 |
| WH2 | R1 | half | 0.810676 | 14.4152% | -22.3587% | 1.283531 |
| WH2 | R5 | all | 0.875423 | 12.2786% | -13.5591% | 2.460102 |
| WH2 | R5 | half | 0.849309 | 13.8185% | -17.8648% | 2.460102 |
| P2021 | BH | all | 0.699303 | 14.2748% | -24.5330% | 0.000000 |
| P2021 | BH | half | 0.699303 | 14.2748% | -24.5330% | 0.000000 |
| P2021 | R0 | all | 0.843617 | 12.2152% | -11.4533% | 1.880597 |
| P2021 | R0 | half | 0.829864 | 13.4680% | -16.5821% | 1.880597 |
| P2021 | R1 | all | 0.763435 | 13.1676% | -21.7041% | 1.367707 |
| P2021 | R1 | half | 0.766609 | 13.8585% | -22.3587% | 1.367707 |
| P2021 | R5 | all | 0.836654 | 12.6663% | -13.5591% | 2.906377 |
| P2021 | R5 | half | 0.820790 | 13.6755% | -16.5257% | 2.906377 |

## Latest R5 reading

Decision date: 2026-11-10. T: out (DMAS 39.000000); R: in (REV_z 0.929475); C: out (C_z -0.574204). Report date used: 2026-11-03. Gauge state: Cash. Sleeve state: Cash (scheduled decision state).

## Method and validation

- All inputs loaded read-only with load(name, refresh=False); no network, Bloomberg, or Dropbox writes.
- CoT net is (leveraged long - leveraged short) / open interest. Z-score uses the previous 156 reports, minimum 104, excluding the current report; standard deviation uses ddof=1, as in earlier steps.
- Each decision selects the latest report with report date + stored release lag <= decision date. Undefined C remains undecided; every decision in the specified window has defined C_z.
- Existing T/R signals, sleeve_weights with deep = DMAS < 30, returns, metrics and bootstrap are reused. No variants or extra rules were run.
- Synthetic release-delay, warm-up, z-score, eight-combination vote, real-data truncation and runner window/state tests passed before the result run.

## Deviations

- No step-2 threshold, rule, window, or gate changes.
- Inherited conventions: costs are debited on the new-weight return day (t+2); initial allocation has no switch cost; the common evaluation sample starts at R5’s first effective weight. Monthly grouping uses the actual decision date’s calendar month, including holiday adjustments.
- BH is labeled all and half, with identical results, to satisfy the requested rules × versions output.
