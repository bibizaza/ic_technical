# Next ideas from Fable 5.2 (2026-11-17, brainstorm only, nothing run)

Ranked by evidence x fit with Delphi x client story. Bloomberg tickers are from memory: check on the terminal first.

1. **Faster re-entry**: in Cash, re-enter on any Tuesday if DMAS >= 60 (mirror of the early exit at 30). No new data. Tests the F placebo lesson (the base rule stays in cash too long). Re-judge F against it. Story: "we leave slowly, but we do not wait for the calendar to come back".
2. **Recession filter on the exit**: Cash only if DMAS < 45 AND (3-month avg unemployment >= its 12-month low + 0.5 pp OR ISM < 50), lagged 1 month. Long history (1970+ panel stored; USURTOT Index, NAPMPMI Index; Europe UMRTEMU Index, MPMIEZMA Index). Fails in 1987 and 2022. Story: "we cut risk only when the economy confirms the market".
3. **Treasuries as the cash leg**: in Cash hold 7-10y Treasuries if above their 200-day average, else bills (LUATTRUU Index, LEATTREU Index). Check the Delphi mandate. Story: "when we step out, the money still works".
4. **Regional tilt on relative earnings revisions**: +/-10 points US vs Europe by revision z. Small breadth.
5. **Variance risk premium**: floor 50% in a cash spell once 21-day realised vol falls back below VIX. Probably the same information as F.
6. **Put/call ratio**: weakest (short history, 0DTE distortion, duplicates VIX, "it is cheap" flavour like C). PCUSEQTR Index, PCUSTOTL Index. Fable: do not spend a test on it.

Fable's order: idea 1 first (no data, US and Europe), idea 2 second.
