# Subtitle Directive V2 — IC Technical Pipeline

## Context

Each slide shows a 5-month price chart with 50d/100d/200d MAs, RSI(14), and a DMAS scorecard.
The subtitle sits below the instrument name and rating — 2 lines, written like a technical analyst
briefing a committee.

---

## Core Philosophy

- **Line 1**: What is the chart structure telling us right now?
- **Line 2**: What is the next technical level to watch, and what happens there?

---

## Hard Rules (violations = automatic reject)

**R1 — No scores in text**: Never mention DMAS, Technical score, Momentum score, breadth rank,
or fundamental rank by name. RSI level is OK as a technical indicator (e.g. "RSI at 30") but
NOT "RSI score at 30".

**R2 — Max 2 numbers per line.** Prioritize MA distance % and RSI over raw prices.

**R3 — No promotional adjectives**: extraordinary, exceptional, remarkable, outstanding, perfect,
relentless, impressive, stunning, incredible, powerful.

**R4 — No instrument name as first words.**

**R5 — All numbers must be verifiable from the data provided.**

**R6 — Streak counts**: use EXACTLY the `Rating streak: X for N consecutive weeks` field.
Never count history entries yourself.

**R7 — No investment recommendations** (buy, sell, add, reduce, take profits).

**R8 — Both lines end with a period.**

**R9 — No false regional generalizations**: Any group claim (EM, DM, metals) must be supported
by ALL members. When performance diverges within a group, name specific indices instead of
regional labels.

**R10 — MA position verification (CRITICAL)**: When `vs_50d`, `vs_100d`, or `vs_200d` is
**NEGATIVE**, price is **BELOW** that MA — the MA is **overhead resistance**, NOT support.
When **POSITIVE**, price is **ABOVE** — the MA is **support below**. Verify every MA sentence
against the sign of the corresponding `vs_Xd` field before finalizing.

---

## Technical Narrative — MA Stack (primary tool)

| Chart Structure | Vocabulary |
|---|---|
| Above all 3 MAs, fanning up | "All MAs rising and well-spaced" |
| Above 200d, below 50d | "Corrected to the 100d zone but 200d underpins" |
| Below all 3 MAs, curling down | "Sliced through all MAs with 50d curling lower" |
| Testing a specific MA | "Clinging to the 200d as last support" |
| Between MAs | "Trapped between the 100d and 200d" |

---

## RSI Guidance

- RSI ≤ 30: "deeply oversold" / "approaching oversold"
- RSI 30-40: "nearing oversold — room for a bounce attempt"
- RSI 60-70: "approaching overbought"
- RSI ≥ 70: "extended — overbought conditions"
- RSI 40-60: neutral, mention only if supporting another point

---

## MA Slope Awareness

When the MA stack is clear (all above/below), also consider slope:
- Slope up + price above → structural support ("rising 200d underpins")
- Slope down + price below → structural resistance ("declining 50d capping")

---

## DMAS Tone Calibration

Match vocabulary to DMAS range **without naming the score**:

| DMAS Range | Tone | Sample vocabulary |
|---|---|---|
| 80-100 | Confident | "textbook structure", "next leg", "resets for continuation" |
| 60-79 | Balanced | "broader uptrend intact", "needs to reclaim", "would restore" |
| 40-59 | Cautious | "approaching support", "line in the sand", "pivotal zone" |
| 20-39 | Urgent | "rapid succession", "losing levels", "last support" |
| 0-19 | Structural damage | "freefall", "no base forming", "stabilization needed" |

---

## WoW Delta Modulation

Use the week-over-week DMAS change to modulate intensity:

- Large negative (↓15+): "sharp reversal", "violent correction"
- Small negative (↓1-5): "continues to drift", "edging lower"
- Flat (±0): neutral tone
- Positive: "beginning to stabilize", "first uptick in X weeks"

---

## Tech vs Momentum Divergence (>30pt gap)

- **High Momentum + Low Technical**: frame as sharp correction within broader trend
- **Low Momentum + High Technical**: near-term improved but broader trend weak

---

## Overview Subtitle Rules (YTD evolution slides)

Single-line subtitles for the 3 YTD evolution slides (equity, commodity, crypto).
Max 12 words. End with a period.

1. Frame the narrative around **regions, sectors, or themes** — not individual names.
   Think "EM vs DM", "energy vs metals", "Bitcoin dominance vs altcoin weakness".
2. Highlight what is **UNUSUAL or SURPRISING** — the story a PM would tell a client.
3. Quantify only if the number is striking (e.g. "+60% YTD"). At most 1 number per line.
4. FORBIDDEN words: divergence, mixed, varied, dynamics, remarkable, exceptional, extraordinary, outstanding.
5. Must be factually verifiable from the data provided.
6. Do NOT name more than one individual instrument per subtitle. If you name one (e.g. "Oil's +61%"),
   the rest should be a category ("industrial metals", "altcoins") — not a second name.
7. Numbers must match actual data to within 1 percentage point. Either use the exact number or drop it.
8. Never generalize a region when performance diverges within it. If Brazil is +13% and India is -13%,
   "emerging markets" are NOT down — they are split. Name specific indices, not regional labels.

---

## Banned Phrases

- "recovery hinges on"
- "recovery depends on"
- "outlook persists"
- "dynamics continue"
- "exceptional strength remains"
- "momentum supports further gains"
- "bullish streak remains unbroken"
- "maintains bullish trend"
- "resilience despite DMAS decline"
- "momentum strength key to"
- "hinges on reclaiming 50d MA"
- "downgraded from X to Y"
- "score floored at zero"
- "momentum score at X"
- "technical score at X"

---

## Before / After Examples

**DMAS 6 (structural damage):**
```
✅ "In freefall with all MAs now sloping down and price 10.2% below the 200d."
   "RSI nearing oversold at 34 but no base forming — stabilization needed before any recovery attempt."
```

**DMAS 91 (strong bullish):**
```
✅ "All three MAs rising and well-spaced — textbook bullish structure."
   "Pulled back to the 50d after 14 weeks of gains; RSI at 41 resets overbought conditions for the next leg."
```

**DMAS 42 (cautious, ↓38 WoW):**
```
✅ "Violent correction through the 50d and 100d after a parabolic run."
   "RSI at 25 is deeply oversold — the rising 200d near 4,200 is the first real support for a bounce attempt."
```

**R10 violation example (vs_50d=-5.0%, vs_100d=-3.2%, vs_200d=+8.7%):**
```
❌ "The 100d just below current price is the near-term pivot." (WRONG: 100d is ABOVE price)
✅ "Corrected 5.0% below the 50d with the 100d also overhead — the 200d 8.7% below remains the structural floor."
```

**Overview — equity (IBOV +13%, Sensex -10%):**
```
❌ "Emerging markets lead as Western indices struggle." (WRONG: India is EM and down sharply)
✅ "IBOV's +13% rally stands alone as broad selloff sweeps from Sensex to the DAX."
```

**Overview — commodity (Oil +61%, metals flat/negative):**
```
❌ "Commodities are performing well YTD." (vague, ignores metals weakness)
✅ "Oil's +61% parabolic surge dwarfs everything as precious and industrial metals correct sharply."
```

---

## Generation Workflow

1. Read all instrument data first — batch awareness prevents repetition across slides.
2. For each instrument, note the **SIGN** of `vs_50d` / `vs_100d` / `vs_200d` before writing.
3. **R10 CHECK**: List which MAs are overhead (negative) and which are below (positive).
4. Draft Line 1 using MA stack vocabulary.
5. Draft Line 2 with a forward-looking scenario.
6. Check DMAS range → calibrate tone vocabulary.
7. Check WoW delta → adjust intensity.
8. **R10 CHECK again**: Re-read every MA sentence and verify direction matches the sign.
9. Apply all R1–R9 checks.
10. Confirm both lines end with a period.
