# US quant gauge — step 2 results

Verdict: FAIL

First decision: 2008-07-01. Common return sample: 2008-07-03 to 2026-11-13 (4,621 trading days).
WH1 ends 2017-06-30; WH2 starts 2017-07-01; P2021 starts 2021-01-01 (information only). R0/R1/BH are slices of their full runs; R5 starts at the July 2008 monthly decision. Sub-periods do not restart state.

## Pre-registered gate (all version)

| Condition | Evidence | Result |
|---|---|---|
| WIN Sharpe R5 > R1 | 0.156868 > 0.249792 | FAIL |
| WH1 Sharpe R5 > R1 | 0.742439 > 0.832977 | FAIL |
| WH2 Sharpe R5 > R1 | 0.019077 > 0.073981 | FAIL |
| WIN Sharpe R5 > BH | 0.156868 > 0.266513 | FAIL |
| Bootstrap R5 <= R1 share < 0.25 | 0.8405 < 0.25 | FAIL |
| WIN switches/year R5 <= R0 | 2.508548 <= 2.290413 | FAIL |

Bootstrap R5 <= R1: all 0.8405; half 0.7895.
Paired moving blocks: 63 trading days, 2,000 draws, numpy.random.default_rng(7), separately for each version. Only all enters the gate; max drawdown is reported, not gated.
About 18 years and few bear markets imply low power. Five candidates have now been tried on the US; any PASS still needs a Europe check with EUR-relevant CoT.

## Metrics

CAGR and maxDD are percentages. Sharpe uses excess returns over cash. BH is identical in both exposure versions.

| Period | Rule | Version | Sharpe | CAGR | maxDD | Switches/year |
|---|---|---|---:|---:|---:|---:|
| WIN | BH | all | 0.266513 | 12.3139% | -47.2603% | 0.000000 |
| WIN | BH | half | 0.266513 | 12.3139% | -47.2603% | 0.000000 |
| WIN | R0 | all | 0.141566 | 8.8278% | -15.7887% | 2.290413 |
| WIN | R0 | half | 0.218057 | 10.9316% | -25.8092% | 2.290413 |
| WIN | R1 | all | 0.249792 | 11.8834% | -21.7041% | 1.472409 |
| WIN | R1 | half | 0.267049 | 12.4019% | -25.8092% | 1.472409 |
| WIN | R5 | all | 0.156868 | 9.2804% | -19.0282% | 2.508548 |
| WIN | R5 | half | 0.226000 | 11.1774% | -25.8092% | 2.508548 |
| WH1 | BH | all | 0.553221 | 9.8528% | -47.2603% | 0.000000 |
| WH1 | BH | half | 0.553221 | 9.8528% | -47.2603% | 0.000000 |
| WH1 | R0 | all | 0.670924 | 6.7396% | -15.7887% | 2.558940 |
| WH1 | R0 | half | 0.671506 | 8.7307% | -25.8092% | 2.558940 |
| WH1 | R1 | all | 0.832977 | 10.5477% | -18.7048% | 1.668874 |
| WH1 | R1 | half | 0.728759 | 10.5707% | -25.8092% | 1.668874 |
| WH1 | R5 | all | 0.742439 | 7.1871% | -19.0282% | 2.558940 |
| WH1 | R5 | half | 0.699592 | 8.9735% | -25.8092% | 2.558940 |
| WH2 | BH | all | 0.127820 | 14.7320% | -33.7936% | 0.000000 |
| WH2 | BH | half | 0.127820 | 14.7320% | -33.7936% | 0.000000 |
| WH2 | R0 | all | 0.008647 | 10.8738% | -13.8063% | 2.032258 |
| WH2 | R0 | half | 0.072141 | 13.0896% | -21.1835% | 2.032258 |
| WH2 | R1 | all | 0.073981 | 13.1827% | -21.7041% | 1.283531 |
| WH2 | R1 | half | 0.103322 | 14.1910% | -22.3587% | 1.283531 |
| WH2 | R5 | all | 0.019077 | 11.3314% | -13.5590% | 2.460102 |
| WH2 | R5 | half | 0.077453 | 13.3382% | -17.9075% | 2.460102 |
| P2021 | BH | all | 0.886882 | 14.2748% | -24.5330% | 0.000000 |
| P2021 | BH | half | 0.886882 | 14.2748% | -24.5330% | 0.000000 |
| P2021 | R0 | all | 1.070509 | 11.4191% | -12.9403% | 1.880597 |
| P2021 | R0 | half | 1.052905 | 13.0660% | -16.7971% | 1.880597 |
| P2021 | R1 | all | 0.959095 | 12.5802% | -21.7041% | 1.367707 |
| P2021 | R1 | half | 0.968068 | 13.5634% | -22.3587% | 1.367707 |
| P2021 | R5 | all | 1.021239 | 11.4941% | -13.5590% | 2.906377 |
| P2021 | R5 | half | 1.023683 | 13.0840% | -16.7265% | 2.906377 |

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
