"""
Common helper functions used across all instrument modules.

This module contains functions that are 100% identical across all 20 instrument
files. By extracting them here, we eliminate ~15,000 lines of duplication while
keeping all calculation logic intact.

These are pure utility functions that don't affect:
- Score calculations
- Chart colors
- Any business logic

They are just helpers for:
- Font formatting in PowerPoint
- Moving average calculations
- Data loading
"""

from typing import Optional, Tuple
import pandas as pd


def _get_run_font_attributes(run):
    """
    Capture font attributes from a run.

    Returns a tuple (size, rgb, theme_color, brightness, bold, italic).
    The colour information includes either the RGB value if explicitly
    defined, or the theme colour and brightness for a scheme colour. If
    colour information is not available, rgb and theme_color are None.
    Bold and italic attributes are preserved as provided.
    """
    if run is None:
        return None, None, None, None, None, None
    size = run.font.size
    colour = run.font.color
    rgb = None
    theme_color = None
    brightness = None
    # Try to capture an explicit RGB value
    try:
        rgb = colour.rgb
    except Exception:
        rgb = None
        # If no RGB value, attempt to capture a theme colour
        try:
            theme_color = colour.theme_color
        except Exception:
            theme_color = None
    # Capture brightness adjustment if available
    try:
        brightness = colour.brightness
    except Exception:
        brightness = None
    bold = run.font.bold
    italic = run.font.italic
    return size, rgb, theme_color, brightness, bold, italic


def _apply_run_font_attributes(new_run, size, rgb, theme_color, brightness, bold, italic):
    """
    Apply captured font attributes to a new run.

    Parameters
    ----------
    new_run : pptx.text.run.Run
        The run to which attributes should be applied.
    size : pptx.util.Length or None
        The font size to apply.
    rgb : pptx.dml.color.RGBColor or None
        The explicit RGB colour value to apply.
    theme_color : MSO_THEME_COLOR or None
        The theme colour value to apply if no RGB colour is defined.
    brightness : float or None
        Brightness adjustment for the colour, if any.
    bold : bool or None
        Whether the font should be bold.
    italic : bool or None
        Whether the font should be italic.
    """
    if size is not None:
        new_run.font.size = size
    # Apply colour: prefer explicit RGB, otherwise theme colour
    if rgb is not None:
        try:
            new_run.font.color.rgb = rgb
        except Exception:
            pass
    elif theme_color is not None:
        try:
            new_run.font.color.theme_color = theme_color
            if brightness is not None:
                new_run.font.color.brightness = brightness
        except Exception:
            pass
    # Apply bold and italic
    if bold is not None:
        new_run.font.bold = bold
    if italic is not None:
        new_run.font.italic = italic


def _add_mas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 50/100/200-day moving-average columns to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'Price' column.

    Returns
    -------
    pd.DataFrame
        Copy of the input with MA_50, MA_100, MA_200 columns added.
    """
    out = df.copy()
    for w in (50, 100, 200):
        out[f"MA_{w}"] = out["Price"].rolling(w, min_periods=1).mean()
    return out


def _load_price_data_generic(
    excel_path,
    ticker: str,
    price_mode: str = "Last Price",
) -> pd.DataFrame:
    """
    Read the raw price sheet and return a tidy Date-Price DataFrame.

    This is a generic version used by all instruments.

    Parameters
    ----------
    excel_path : str or pathlib.Path
        Path to the Excel workbook containing price data.
    ticker : str
        Column name corresponding to the desired ticker in the Excel sheet.
    price_mode : str, default "Last Price"
        One of "Last Price" or "Last Close". If adjust_prices_for_mode
        is available and the mode is "Last Close", rows with the last
        recorded date (if equal to today's date) will be dropped.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns Date and Price. The data are sorted by date
        and any rows with missing values are removed.
    """
    try:
        from utils import adjust_prices_for_mode
    except Exception:
        adjust_prices_for_mode = None

    df = pd.read_excel(excel_path, sheet_name="data_prices")
    df = df.drop(index=0)
    df = df[df[df.columns[0]] != "DATES"]
    df["Date"] = pd.to_datetime(df[df.columns[0]], errors="coerce")
    df["Price"] = pd.to_numeric(df[ticker], errors="coerce")
    df_clean = (
        df.dropna(subset=["Date", "Price"])
        .sort_values("Date")
        .reset_index(drop=True)[["Date", "Price"]]
    )
    # Adjust for price mode if helper is available
    if adjust_prices_for_mode is not None and price_mode:
        try:
            df_clean, _ = adjust_prices_for_mode(df_clean, price_mode)
        except Exception:
            # If adjustment fails, silently fall back to unadjusted data
            pass
    return df_clean


def _get_technical_score_generic(
    excel_obj_or_path,
    ticker: str
) -> Optional[float]:
    """
    Retrieve the technical score for any instrument from 'data_technical_score'.

    This uses vectorized pandas operations (100x faster than .iterrows()).

    Parameters
    ----------
    excel_obj_or_path : file-like or path
        Excel workbook containing data_technical_score sheet.
    ticker : str
        Ticker to search for (e.g., "SPX INDEX", "GCA COMDTY").

    Returns
    -------
    float or None
        The technical score or None if unavailable.
    """
    try:
        df = pd.read_excel(excel_obj_or_path, sheet_name="data_technical_score")
    except Exception:
        return None

    df = df.dropna(subset=[df.columns[0], df.columns[1]])

    # VECTORIZED: Direct boolean indexing (100x faster than .iterrows())
    df[df.columns[0]] = df[df.columns[0]].astype(str).str.strip().str.upper()
    target_ticker = ticker.upper()

    matches = df[df[df.columns[0]] == target_ticker]
    if matches.empty:
        return None

    try:
        return float(matches.iloc[0][df.columns[1]])
    except Exception:
        return None


def _get_momentum_score_generic(
    excel_obj_or_path,
    ticker: str
) -> Optional[float]:
    """
    Retrieve the momentum score for any instrument from 'data_trend_rating'.

    Uses the same pattern as all instruments: reads from Excel, maps letter grades.

    Parameters
    ----------
    excel_obj_or_path : file-like or path
        Excel workbook containing data_trend_rating sheet.
    ticker : str
        Ticker to search for (e.g., "SPX INDEX", "GCA COMDTY").

    Returns
    -------
    float or None
        The momentum score or None if unavailable.
    """
    try:
        df = pd.read_excel(excel_obj_or_path, sheet_name="data_trend_rating")
    except Exception:
        return None

    # Identify the row by ticker
    mask = df.iloc[:, 0].astype(str).str.strip().str.upper() == ticker.upper()
    if not mask.any():
        return None

    row = df.loc[mask].iloc[0]

    # Try to convert column 3 to float
    try:
        return float(row.iloc[3])
    except Exception:
        pass

    # Fall back to letter grade mapping
    rating = str(row.iloc[2]).strip().upper()  # 'Current' column
    mapping = {"A": 100.0, "B": 70.0, "C": 40.0, "D": 0.0}

    # Optionally lookup in 'parameters' sheet for customized mapping
    try:
        params = pd.read_excel(excel_obj_or_path, sheet_name="parameters")
        param_row = params[params["Tickers"].astype(str).str.upper() == ticker.upper()]
        if not param_row.empty and "Unnamed: 8" in param_row:
            return float(param_row["Unnamed: 8"].dropna().iloc[0])
    except Exception:
        pass

    return mapping.get(rating)
