# Stress test on synthetic US paths (spec 6ST, information only)

Stationary block bootstrap (mean block 63 days) of S&P 500 daily returns 1928–2026. Price-only rules: no earnings key, no VIX floor, no breadth re-entry, Tuesday decisions with the old band rule, cash 0%, 10 bp. So these numbers are for the price-driven skeleton of the core, not the full product. Code: `stress_synth.py`.

Paths that failed to compute and were skipped: 0.

## World A (blocks from any year), 1000 paths of 30 years

| Rule | Sharpe median | Sharpe bad case (5%) | Worst fall median | Worst fall bad case (5%) | Single worst | Paths < −30% | Paths < −40% | Beats B&H Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Buy & hold | 0.61 | 0.29 | -49.7% | -71.3% | -87.5% | 99% | 86% |  |
| R0 (trend only) | 0.54 | 0.22 | -36.7% | -54.3% | -72.7% | 81% | 34% | 31% |
| V = R0 x vol size | 0.59 | 0.26 | -30.4% | -47.0% | -64.9% | 52% | 15% | 44% |
| V + S live | 0.60 | 0.26 | -31.8% | -49.3% | -61.6% | 57% | 19% | 46% |
| V + S 50% floor (product) | 0.59 | 0.25 | -32.2% | -49.9% | -62.8% | 59% | 21% | 43% |
| V + S 100% | 0.57 | 0.23 | -34.9% | -53.9% | -73.4% | 71% | 30% | 38% |

- S episodes per path: median 7; share stopped 23%; worst single episode (index) median -11.5%, bad case (5%) -22.4%.
- 50% floor vs live: Sharpe better in 43% of paths (median -0.001).
- 100% vs 50% floor: Sharpe better in 26% of paths (median -0.016); worst fall deeper in 68% of paths, bad case (5%) -10.3%.

## World B (blocks from crisis years only), 500 paths of 30 years

| Rule | Sharpe median | Sharpe bad case (5%) | Worst fall median | Worst fall bad case (5%) | Single worst | Paths < −30% | Paths < −40% | Beats B&H Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Buy & hold | -0.12 | -0.41 | -94.9% | -99.3% | -99.9% | 100% | 100% |  |
| R0 (trend only) | -0.05 | -0.38 | -72.5% | -91.6% | -97.6% | 100% | 99% | 63% |
| V = R0 x vol size | -0.07 | -0.41 | -60.5% | -83.3% | -94.5% | 99% | 91% | 62% |
| V + S live | -0.08 | -0.42 | -64.5% | -85.0% | -94.2% | 100% | 94% | 59% |
| V + S 50% floor (product) | -0.08 | -0.42 | -65.2% | -86.3% | -93.8% | 100% | 95% | 59% |
| V + S 100% | -0.08 | -0.42 | -67.7% | -88.2% | -93.6% | 100% | 97% | 58% |

- S episodes per path: median 10; share stopped 37%; worst single episode (index) median -16.3%, bad case (5%) -24.9%.
- 50% floor vs live: Sharpe better in 51% of paths (median +0.000).
- 100% vs 50% floor: Sharpe better in 48% of paths (median -0.002); worst fall deeper in 83% of paths, bad case (5%) -9.2%.

## Reading (added after the run)

- A block bootstrap cuts trend persistence beyond the block length (mean 63 days). So trend rules look weaker here than in real history (R0 Sharpe 0.54 vs buy & hold 0.61, while R0 beats buy & hold in real US history). Read the Sharpe columns as a floor, not an estimate.
- The drawdown protection survives: in world A the price skeleton cuts the median 30-year worst fall from −50% to −30–32%, and the bad case (5%) from −71% to −47–50%.
- S at 100% adds drawdown risk: worst fall deeper than the 50% floor in 68% of paths (83% in the crisis world), Sharpe lower in 74%. The 50% floor costs almost nothing against the live size (median worst fall −32.2% vs −31.8%). This supports spec 11d.
- Planning number for clients (price skeleton, no earnings key): a worst fall around −30% over 30 years is normal; −50% is the bad case. The full product's earnings key lowered drawdowns in real history (−20%), but it is not in this test.
- World B (only crisis years) is extreme: buy & hold loses ~95%; the rules lose ~60–65%. Timing reduces damage but cannot make money in a permanent crisis.
