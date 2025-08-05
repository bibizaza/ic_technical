"""
Robust loader for sheet *bql_formula* + modern colour‑scaled Styler.
See doc‑string for layout assumptions (tickers in AB, data in AC‑AE).
"""

from __future__ import annotations
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
from matplotlib import cm, colors


###############################################################################
# 1. Load sparse rows (ticker + three % columns, 6 blank rows in‑between)
###############################################################################
def _load_and_prepare(xlsx: Union[str, Path]) -> pd.DataFrame:
    xlsx = Path(xlsx)
    try:
        raw = pd.read_excel(xlsx, sheet_name="bql_formula", header=None)
    except Exception:
        return pd.DataFrame()

    # real data rows = non‑blank in column 27 (AB)
    rows: List[int] = raw.index[raw.iloc[:, 27].astype(str).str.strip() != ""].tolist()
    if not rows:
        return pd.DataFrame()

    df = raw.loc[rows, [27, 28, 29, 30]].copy()
    df.columns = ["Index", "% Above 20 & 50", "% Above 50", "% Above 20"]

    # numeric conversion + 0‑1 → 0‑100
    pct_cols = ["% Above 20 & 50", "% Above 50", "% Above 20"]
    df[pct_cols] = df[pct_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=pct_cols)
    if df.empty:
        return df

    if df[pct_cols].max().max() <= 1 + 1e-9:
        df[pct_cols] = (df[pct_cols] * 100).round(1)
    else:
        df[pct_cols] = df[pct_cols].round(1)

    # Breadth rank (1 = best) inserted between Index and first % column
    df["Breadth Rank"] = df["% Above 20 & 50"].rank(method="min", ascending=False).astype(int)
    df = df.sort_values("% Above 20 & 50", ascending=False).reset_index(drop=True)
    df = df[["Index", "Breadth Rank"] + pct_cols]
    return df


###############################################################################
# 2. Three‑colour scale (green‑yellow‑red) – Index column left untouched
###############################################################################
def _apply_matrix_style(df: pd.DataFrame) -> pd.Styler:
    if df.empty:
        return df.style

    cmap = cm.get_cmap("RdYlGn")          # red (low) → yellow → green (high)

    def _colour_series(s: pd.Series, invert: bool = False) -> list[str]:
        """Return a list of 'background-color:#RRGGBB;color:#fff|#000'."""
        rng = s.max() - s.min()
        norm = (s - s.min()) / rng if rng else 0.5
        if invert:                         # for Breadth Rank (1 best → green)
            norm = 1 - norm
        out = []
        for v in norm:
            r, g, b, _ = cmap(float(v))
            hexclr = colors.to_hex((r, g, b))
            lum = 0.2126*r + 0.7152*g + 0.0722*b
            txt = "#000000" if lum > 0.5 else "#ffffff"
            out.append(f"background-color:{hexclr};color:{txt};")
        return out

    style_df = pd.DataFrame("", index=df.index, columns=df.columns)
    for col in df.columns:
        if col == "Index":
            continue                       # leave ticker column uncoloured
        invert = (col == "Breadth Rank")
        style_df[col] = _colour_series(df[col].astype(float), invert=invert)

    return (
        df.style
          .apply(lambda _: style_df, axis=None)
          .format({c: "{:,.1f}%" for c in df.columns if "Above" in c})
          .hide(axis="index")
    )


###############################################################################
# 3. Tiny debugger (optional use in app.py)
###############################################################################
def debug_first_rows(xlsx: Union[str, Path], n: int = 10) -> pd.DataFrame:
    return _load_and_prepare(xlsx).head(n)
