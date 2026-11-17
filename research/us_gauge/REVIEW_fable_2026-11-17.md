# Independent review — Fable 5.2, 2026-11-17

**Verdict: do not sign the K1 headline as written.** Arithmetic reproduced exactly (K1 0.671 / −24.5% / 8.86%;
BH 0.423 / −55.2% / 8.60%; R1 vs BH 1993–2026 0.706 vs 0.519; REV_z and DMAS rebuilt independently). No look-ahead
bug found; cash fix complete; C5 macro inputs are real-time vintages.

## Issues
1. **Blocking — the edge is two episodes.** Excluding 2000-03→2003-03 and 2007-10→2009-06: K1 0.88 vs BH 0.83,
   CAGR 11.6% vs 15.0%. R1: 23 cash spells since 1993; two (2000–02, 2008–09) earned +56.7 pp, the other 21 net −50 pp.
   K1 lagged BH in every 5-year block since 2009.
2. **Blocking — the sector layer is not a macro-model result.** Picks: XLP 68%, XLU 66%, XLE 62% of months. A constant
   XLP/XLU/XLE tilt gives K1 0.679 (> 0.671 with C5); all 84 static triples average 0.648. Claude check: static
   70/10/10/10 XLP/XLU/XLE vs SPY, monthly 1999–2026 = 0.536 vs 0.496 (S2 with C5: 0.545).
3. **Blocking for "pre-registered" wording — S2/K1 not fresh.** `macro_health/results/regime_etf_test.md`
   (commit d000bb7, 2026-10-21) already ran C5 on the same ETFs 1999–2026 at 20/50/100% weights.
4. **Material — the gate changed after R1 failed** (drawdown dropped at step 1c); all 11 candidates were a sequential
   search on one history in one morning.
5. **Material — not robust / not significant.** R1 Sharpe by decision weekday: Mon 0.68, Tue 0.71, Wed 0.59, Thu 0.60,
   Fri 0.65 (mean 0.645). DMAS alone 1955–1993: 0.436 vs BH 0.408. Plain 200-day rule: 0.56. Bootstrap p: R1 vs BH
   0.055; K1 vs BH 0.04 single test, ~0.4 after Bonferroni (11 tries). Vol sizing alone +0.035 (p 0.17).
   25 bp costs: K1 0.637.
6. **Material, unverifiable — EPS point-in-time.** Chart export of 2026-11-16; no proof of as-of history before the
   mid-2000s. Lagging REV_z 5/10/21 days: R1 0.695 / 0.652 / 0.627.
7. **Material — implementation gap.** Delphi holds UCITS ETFs; backtest uses US SPDRs, gross dividends, USD T-bills.
8. Minor: DTB3 discount yield; Shiller dividend interpolation; IYZ/XLC inception availability; DMAS 45/30 set with
   knowledge of S&P history.

## What to believe
Timing helps in slow multi-year bears (2000–02, 2008), not otherwise. Vol sizing lowers risk; Sharpe gain small.
Sectors = a defensive/energy low-beta tilt, not C5 skill. Forward Sharpe edge estimate 0.05–0.10, with ~3–4%/yr lag
in bull markets. Signable wording: "a rule that cut the 2000–02 and 2008 drawdowns by half, at the cost of lagging in
every 5-year block since 2009; Sharpe evidence not significant after multiple testing."

## Europe must pass (frozen rules, one run)
1. Same R1 + V2 code on STOXX 600 with EUR cash; report all five decision weekdays and results excluding 2000–03 and
   2008–09; beat BH on the weekday average and beat a plain 200-day rule.
2. Sector layer vs placebo: beat the constant most-picked-three tilt and the distribution of all static triples on
   beta-adjusted return.
3. EPS point-in-time evidence; survive a 10-day revision lag and 25 bp costs with tradable UCITS instruments.

Fable's scripts (temporary): scratchpad/fable_check/.
