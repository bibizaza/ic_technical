# Data for the machine-learning (meta-labelling) study — no Bloomberg

Goal: a model that sizes a re-entry ("this looks like 2020 → buy more", "this looks like 1987/1929 → buy less").
It needs many crash events (pooled across markets) and features that exist on the day of the decision.

## Already in the store (nothing to do)

| Data | History | Used for |
|---|---|---|
| S&P 500 close + OHLC (`GSPC.INDX`) | 1927 | US events, intraday range |
| 19 other indices (EODHD: N225, TWII, GSPTSE, HSI, KS11, AXJO, GDAXI, FCHI, SSMI, AEX, IBEX, …) | 1965–2000 | pooled crash events |
| STOXX 600 (`BBG_SXXP`, already stored) | 1987 | Europe events |
| VIX (`FRED_VIXCLS`), VIX3M (`FRED_VXVCLS`) | 1990 / 2007 | fear level, curve |
| 2y, 10y, fed funds, Baa–10y spread (FRED) | 1976 / 1962 / 1954 / 1986 | rates, credit stress |
| T-bill (`FRED_DTB3`) | 1954 | cash |
| NYSE advances / declines | 1965 | breadth |
| S&P EPS revisions (already stored) | 1990 | earnings trend |
| M2, unemployment, NBER recession flag, Fed balance sheet | 1959 / 1948 / 1854 / 2002 | macro state |
| CFTC E-mini positioning | 2006 | positioning |
| Shiller dividends | 1871 | total return |

## To add (all free; I can download them myself if you agree)

| # | Data | Source | History | Why |
|---|---|---|---|---|
| 1 | OHLC + volume for the 19 non-US indices | EODHD (our key) | same as the closes | capitulation volume, intraday range = better crash features |
| 2 | FX of each market vs USD (JPY, CAD, KRW, AUD, HKD, TWD, CHF, EUR, GBP) | FRED H.10 daily (`DEXJPUS`, …) | 1971 | FX shock is a main driver (your CHF / SMI point) |
| 3 | Short rate and 10y yield of each market | FRED / OECD monthly (`IR3TIB01xxM156N`, `IRLTLT01xxM156N`) | 1960s–1990s | local cash, local yield curve |
| 4 | OECD leading indicator and recession flag per country | FRED (OECD) monthly | 1960s | "recession bear" vs "liquidity shock" |
| 5 | Ken French daily factors + risk-free + 49 industries | Ken French library | 1926 | cash before 1954; breadth proxy before 1965 (share of industries above their 200-day average); momentum crashes |
| 6 | Shiller monthly earnings and CAPE | Shiller `ie_data.xls` (columns E, M) | 1871 | valuation and earnings trend before 1990 |
| 7 | Moody's Aaa / Baa yields, monthly | FRED (`AAA`, `BAA`) | 1919 | credit stress back to 1929 |
| 8 | VXO (old VIX) | FRED (`VXOCLS`) | 1986 | fear level in 1987 |
| 9 | CBOE put/call ratios | CBOE free archive | 1995 / 2006 | panic hedging |
| 10 | Margin debt | FINRA (NYSE history) | 1959 | leverage before crashes |
| 11 | AAII sentiment survey | AAII (check if download is free) | 1987 | retail panic |
| 12 | High-yield spread (ICE BofA) | FRED (`BAMLH0A0HYM2`) — check history length; the store already has the Bloomberg version from 1994 | 1996 | credit stress |

## How it will be used (short)

- Events: every fall of 10% or more with a +x% rebound, in every market, 1970–2026. Expect a few hundred.
- Label: did the re-entry make money before its stop / time limit?
- Features: only data known on the day (no look-ahead), plus the market name so the model learns that markets differ.
- Train 1970–2015, test once on 2016–2026, then the US alone as the final check. Rules fixed before the run.
