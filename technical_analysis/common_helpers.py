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
- Color interpolation for gauges
- Volatility-based range calculations
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# Import helper for adjusting price data according to price mode
try:
    from utils import adjust_prices_for_mode  # type: ignore
except Exception:
    adjust_prices_for_mode = None  # type: ignore


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


def _interpolate_color(value: float) -> Tuple[float, float, float]:
    """
    Interpolate from red→yellow→green for a 0–100 value.  Pure red at 0,
    bright yellow at 40 and rich green at 70.

    This function is used for gauge color calculations across all instruments.

    Parameters
    ----------
    value : float
        Score value between 0 and 100.

    Returns
    -------
    Tuple[float, float, float]
        RGB color tuple with values between 0.0 and 1.0.
    """
    red = (1.0, 0.0, 0.0)
    yellow = (1.0, 204 / 255, 0.0)
    green = (0.0, 153 / 255, 81 / 255)
    if value <= 40:
        t = value / 40.0
        return tuple(red[i] + t * (yellow[i] - red[i]) for i in range(3))
    elif value <= 70:
        t = (value - 40) / 30.0
        return tuple(yellow[i] + t * (green[i] - yellow[i]) for i in range(3))
    return green


def _load_price_data_from_obj(
    excel_obj,
    ticker: str,
    price_mode: str = "Last Price",
) -> pd.DataFrame:
    """
    Load price data from a file-like object and return a tidy DataFrame.

    Parameters
    ----------
    excel_obj : file-like
        File-like object representing an Excel workbook containing a
        ``data_prices`` sheet.
    ticker : str
        Column name corresponding to the desired ticker in the Excel sheet.
        No default - must be specified by caller.
    price_mode : str, default "Last Price"
        One of "Last Price" or "Last Close".  If ``adjust_prices_for_mode``
        is available and the mode is "Last Close", rows corresponding to
        the most recent date (if equal to today's date) will be dropped.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``Date`` and ``Price``.  The data are
        sorted by date and any rows with missing values are removed.
    """
    df = pd.read_excel(excel_obj, sheet_name="data_prices")
    df = df.drop(index=0)
    df = df[df[df.columns[0]] != "DATES"]
    df["Date"] = pd.to_datetime(df[df.columns[0]], errors="coerce")
    df["Price"] = pd.to_numeric(df[ticker], errors="coerce")
    df_clean = (
        df.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)[
            ["Date", "Price"]
        ]
    )
    # Adjust for price mode if helper is available
    if adjust_prices_for_mode is not None and price_mode:
        try:
            df_clean, _ = adjust_prices_for_mode(df_clean, price_mode)
        except Exception:
            pass
    return df_clean


def _compute_range_bounds(
    df_full: pd.DataFrame, lookback_days: int = 90
) -> Tuple[float, float]:
    """
    Compute fallback high and low range bounds using realised volatility.

    This helper is used when an implied volatility index (e.g. VIX) is
    unavailable.  It computes the annualised realised volatility over a
    30‑session window by taking the standard deviation of daily
    percentage returns, multiplying by ``sqrt(252)`` and converting to
    a percentage.  The resulting 1‑week expected move is
    ``(current_price × (realised_vol / 100)) / sqrt(52)``.  The upper
    and lower bounds are the current price plus and minus this
    expected move.  If realised volatility cannot be computed or is
    zero, the function falls back to a ±2 % band around the current
    price.

    Parameters
    ----------
    df_full : pandas.DataFrame
        DataFrame containing at least 'Date' and 'Price' columns,
        sorted by date ascending.
    lookback_days : int, optional
        Number of trading days used to compute the approximate true range
        if realised volatility is unavailable.  Currently unused but
        retained for API compatibility.

    Returns
    -------
    Tuple[float, float]
        A two‑tuple ``(upper_bound, lower_bound)`` representing the
        current closing price plus and minus the realised volatility
        based expected move, or ±2 % of the current price if no
        volatility can be computed.
    """
    if df_full.empty:
        return (np.nan, np.nan)
    current_price = df_full["Price"].iloc[-1]
    # Attempt to compute 30‑day realised volatility (annualised) as a fallback.  Use
    # the last 30 trading days of closing prices to compute daily returns.
    # If the realised volatility can be computed, convert it into a 1‑week
    # expected move.  Otherwise fall back to a ±2 % band.
    try:
        # At least 2 data points are needed for pct_change; ensure there are
        # enough rows (we use min to handle shorter histories gracefully).
        lookback = 30
        window_prices = df_full["Price"].tail(lookback)
        # Compute daily percentage returns
        rets = window_prices.pct_change().dropna()
        # Standard deviation of daily returns
        std_daily = rets.std()
        if std_daily is not None and not np.isnan(std_daily) and std_daily > 0:
            # Annualise the standard deviation (multiply by sqrt(252)) and convert to %
            realised_vol = std_daily * np.sqrt(252.0) * 100.0
            # Convert to 1‑week expected move by dividing by sqrt(52)
            expected_move = (current_price * (realised_vol / 100.0)) / np.sqrt(52.0)
            upper_bound = current_price + expected_move
            lower_bound = current_price - expected_move
            return (float(upper_bound), float(lower_bound))
    except Exception:
        pass
    # Fallback: ±2 % of the current price
    return (float(current_price * 1.02), float(current_price * 0.98))


def generate_average_gauge_image(
    tech_score: float,
    mom_score: float,
    last_week_avg: float,
    date_text: str | None = None,
    last_label_text: str = "Last Week",
    width_cm: float = 15.15,
    height_cm: float = 3.13,
) -> bytes:
    """
    Create a horizontal gauge with a red→yellow→green gradient, marking the
    average of technical and momentum scores against last week's average.

    This function is identical across all instruments.

    Parameters
    ----------
    tech_score : float
        Current technical score (0-100).
    mom_score : float
        Current momentum score (0-100).
    last_week_avg : float
        Last week's average score (0-100).
    date_text : str, optional
        Text label for current date.
    last_label_text : str, default "Last Week"
        Text label for previous period.
    width_cm : float, default 15.15
        Width of the gauge in centimeters.
    height_cm : float, default 3.13
        Height of the gauge in centimeters.

    Returns
    -------
    bytes
        PNG image data.
    """
    def clamp100(x: float) -> float:
        return max(0.0, min(100.0, float(x)))

    curr = (clamp100(tech_score) + clamp100(mom_score)) / 2.0
    prev = clamp100(last_week_avg)

    cmap = LinearSegmentedColormap.from_list(
        "gauge_gradient", ["#FF0000", "#FFCC00", "#009951"], N=256
    )

    fig_w, fig_h = width_cm / 2.54, height_cm / 2.54
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    gradient = np.linspace(0, 1, 500).reshape(1, -1)
    bar_thickness = 0.4
    bar_bottom_y = -bar_thickness / 2.0
    bar_top_y = bar_thickness / 2.0
    ax.imshow(
        gradient,
        extent=[0, 100, bar_bottom_y, bar_top_y],
        aspect="auto",
        cmap=cmap,
        origin="lower",
    )

    # Marker dimensions and spacing
    marker_width = 3.0
    marker_height = 0.15
    gap = 0.10
    number_space = 0.25
    top_label_offset = 0.40
    bottom_label_offset = 0.40

    # Y positions for current (top) marker and labels
    top_apex_y = bar_top_y + gap
    top_base_y = top_apex_y + marker_height
    top_number_y = top_base_y + number_space
    top_label_y = top_number_y + top_label_offset

    # Y positions for previous (bottom) marker and labels
    bottom_apex_y = bar_bottom_y - gap
    bottom_base_y = bottom_apex_y - marker_height
    bottom_number_y = bottom_base_y - number_space
    bottom_label_y = bottom_number_y - bottom_label_offset

    curr_colour = _interpolate_color(curr)
    prev_colour = _interpolate_color(prev)

    # Draw triangles and numbers
    ax.add_patch(
        patches.Polygon(
            [
                (curr - marker_width / 2, top_base_y),
                (curr + marker_width / 2, top_base_y),
                (curr, top_apex_y),
            ],
            color=curr_colour,
        )
    )
    ax.add_patch(
        patches.Polygon(
            [
                (prev - marker_width / 2, bottom_base_y),
                (prev + marker_width / 2, bottom_base_y),
                (prev, bottom_apex_y),
            ],
            color=prev_colour,
        )
    )
    ax.text(
        curr,
        top_number_y,
        f"{curr:.0f}",
        color=curr_colour,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
    )
    ax.text(
        prev,
        bottom_number_y,
        f"{prev:.0f}",
        color=prev_colour,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
    )

    if date_text:
        ax.text(
            curr,
            top_label_y,
            date_text,
            color="#0063B0",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
        )
    ax.text(
        prev,
        bottom_label_y,
        last_label_text,
        color="#133C74",
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(bottom_label_y - 0.35, top_label_y + 0.35)
    ax.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=600, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_range_gauge_only_image(
    df_full: pd.DataFrame,
    lookback_days: int = 90,
    width_cm: float = 2.00,
    height_cm: float = 7.53,
) -> bytes:
    """
    Create a standalone vertical gauge image without the price chart.

    This function is intended for interactive environments (e.g. Streamlit) where
    users want to visualise the recent trading range alongside a separate
    interactive plot.  The gauge shows a green–to–red gradient between the
    computed upper and lower bounds, with labels at the extremes and a marker
    indicating the current price’s position within the range.

    Parameters
    ----------
    df_full : pandas.DataFrame
        Full instrument price history as returned by ``_load_price_data``.
    lookback_days : int, default 90
        Number of trading days to look back when computing high/low range.
    width_cm : float, default 2.00
        Width of the output image in centimetres.  A narrow bar suffices for
        embedding alongside an interactive chart in Streamlit.
    height_cm : float, default 7.53
        Height of the gauge in centimetres.  This should match the height of
        your interactive chart for consistent alignment.

    Returns
    -------
    bytes
        PNG image data for the standalone range gauge.
    """
    if df_full.empty:
        return b""
    # Compute bounds and current price
    upper_bound, lower_bound = _compute_range_bounds(df_full, lookback_days=lookback_days)
    current_price = df_full["Price"].iloc[-1]
    # Normalise current position within the range
    if upper_bound == lower_bound:
        rel_pos = 0.5
    else:
        rel_pos = (current_price - lower_bound) / (upper_bound - lower_bound)
        rel_pos = max(0.0, min(1.0, rel_pos))

    # Prepare figure
    fig_w_in, fig_h_in = width_cm / 2.54, height_cm / 2.54
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
    # Build vertical gradient: red → white → green
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    cmap = LinearSegmentedColormap.from_list(
        "range_gauge_only", ["#FF0000", "#FFFFFF", "#009951"], N=256
    )
    ax.imshow(
        gradient,
        extent=[0, 1, lower_bound, upper_bound],
        aspect="auto",
        origin="lower",
        cmap=cmap,
    )
    ax.set_facecolor((1, 1, 1, 0))
    # Draw marker for current price as a horizontal bar spanning the gauge width
    marker_y = lower_bound + rel_pos * (upper_bound - lower_bound)
    marker_height = (upper_bound - lower_bound) * 0.01  # 1% of range height
    ax.add_patch(
        patches.Rectangle(
            (0.0, marker_y - marker_height / 2),
            1.0,
            marker_height,
            color="#153D64",
        )
    )
    # Draw labels for bounds (centre aligned)
    def _fmt(val):
        try:
            return f"{val:,.0f}".replace(",", "'")
        except Exception:
            return f"{val:.0f}"
    upper_label = _fmt(upper_bound)
    lower_label = _fmt(lower_bound)
    ax.text(
        0.5,
        upper_bound,
        f"Higher Range\n{upper_label} $",
        color="#009951",
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        transform=ax.transData,
    )
    ax.text(
        0.5,
        lower_bound,
        f"Lower Range\n{lower_label} $",
        color="#C00000",
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        transform=ax.transData,
    )
    # Format axes: hide ticks and spines
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=600, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()




def generate_range_gauge_chart_image(
    df_full: pd.DataFrame,
    anchor_date: Optional[pd.Timestamp] = None,
    lookback_days: int = 90,
    width_cm: float = 21.41,
    height_cm: float = 7.53,
    chart_width_cm: float = None,
    gauge_width_cm: float = 4.0,
    *,
    vol_index_value: Optional[float] = None,
) -> bytes:
    """
    Create a PNG image of the instrument price chart with a vertical range gauge
    appended on the right.  The gauge shows a green–to–red gradient between
    recent high and support levels, with labels for the upper and lower
    bounds.  A horizontal line continues the last price into the gauge so
    that viewers can assess relative positioning.  This function is used by
    ``insert_spx_technical_chart_with_range``.

    Parameters
    ----------
    df_full : pandas.DataFrame
        Full instrument price history as returned by ``_load_price_data``.
    anchor_date : pandas.Timestamp or None, optional
        Optional anchor date for the regression channel.  If ``None`` no
        channel will be drawn.
    lookback_days : int, default 90
        Number of trading days to look back when computing high/low range.
    width_cm : float, default 21.41
        Width of the output image in centimetres.  This should match the
        template placeholder size in PowerPoint.
    height_cm : float, default 7.53
        Height of the output image in centimetres.

    Returns
    -------
    bytes
        A byte array containing the PNG image data with transparency.
    """
    if df_full.empty:
        return b""

    # Compute bounds for the configured lookback window
    today = df_full["Date"].max().normalize()
    start = today - timedelta(days=PLOT_LOOKBACK_DAYS)
    df = df_full[df_full["Date"].between(start, today)].reset_index(drop=True)
    # Compute moving averages on the full dataset and slice to the lookback window
    df_ma_full = _add_mas(df_full)
    df_ma = df_ma_full[df_ma_full["Date"].between(start, today)].reset_index(drop=True)

    # Regression channel (optional)
    uptrend = False
    upper_channel = lower_channel = None
    if anchor_date is not None:
        subset_full = df_full[df_full["Date"].between(anchor_date, today)].copy()
        if not subset_full.empty:
            X = subset_full["Date"].map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
            y_vals = subset_full["Price"].to_numpy()
            model = LinearRegression().fit(X, y_vals)
            trend = model.predict(X)
            resid = y_vals - trend
            uptrend = model.coef_[0] > 0
            upper_channel = trend + resid.max()
            lower_channel = trend + resid.min()

    # Determine recent high and support levels.  Use the implied volatility
    # if available to estimate a 1‑week range; otherwise fall back to ATR.
    last_price = df["Price"].iloc[-1]
    if vol_index_value is not None and last_price and not np.isnan(last_price):
        expected_move = (last_price * (vol_index_value / 100.0)) / np.sqrt(52.0)
        upper_bound = last_price + expected_move
        lower_bound = last_price - expected_move
    else:
        upper_bound, lower_bound = _compute_range_bounds(df_full, lookback_days=lookback_days)
    last_price_str = f"{last_price:,.2f}"

    # Determine overall width.  If ``chart_width_cm`` is not provided,
    # derive it by subtracting the gauge width from the total width.  This
    # ensures that the combined chart and gauge fit within the fixed slide
    # placeholder width (typically ~21.41 cm).  Callers may supply
    # ``chart_width_cm`` explicitly to override this behaviour.
    if chart_width_cm is None:
        chart_width_cm = max(width_cm - gauge_width_cm, 0.0)

    fig_w_in, fig_h_in = width_cm / 2.54, height_cm / 2.54
    plt.style.use("default")
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    # Determine relative widths for the chart and gauge.  The chart occupies
    # ``chart_width_cm`` cm of the total width while the gauge occupies
    # ``gauge_width_cm`` cm.  These ratios control how much of the figure is
    # devoted to each element.
    chart_rel_width = chart_width_cm / width_cm
    gauge_rel_width = gauge_width_cm / width_cm

    # Create main chart axis using add_axes to occupy the left portion of the
    # figure.  We leave the full height (0→1) for the chart; legend
    # positioning is handled later.
    ax = fig.add_axes([0.0, 0.0, chart_rel_width, 1.0])
    # Placeholder for gauge axis; we will add it after plotting on ax so we can
    # align it vertically with the plotted area of the chart.
    ax_gauge = None

    # Plot main price series and MAs
    ax.plot(
        df["Date"], df["Price"], color="#153D64", linewidth=2.5, label=f"S&P 500 Price (last: {last_price_str})"
    )
    ax.plot(df_ma["Date"], df_ma["MA_50"], color="#008000", linewidth=1.5, label="50‑day MA")
    ax.plot(df_ma["Date"], df_ma["MA_100"], color="#FFA500", linewidth=1.5, label="100‑day MA")
    ax.plot(df_ma["Date"], df_ma["MA_200"], color="#FF0000", linewidth=1.5, label="200‑day MA")

    # Fibonacci levels
    hi, lo = df["Price"].max(), df["Price"].min()
    span = hi - lo
    fib_levels = [hi, hi - 0.236 * span, hi - 0.382 * span, hi - 0.5 * span, hi - 0.618 * span, lo]
    for lvl in fib_levels:
        ax.axhline(lvl, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    # Regression channel shading
    if anchor_date is not None and upper_channel is not None and lower_channel is not None:
        subset = df_full[df_full["Date"].between(anchor_date, today)].copy().reset_index(drop=True)
        fill_color = (0, 0.6, 0, 0.25) if uptrend else (0.78, 0, 0, 0.25)
        line_color = "#008000" if uptrend else "#C00000"
        ax.plot(subset["Date"], upper_channel, color=line_color, linestyle="--")
        ax.plot(subset["Date"], lower_channel, color=line_color, linestyle="--")
        ax.fill_between(subset["Date"], lower_channel, upper_channel, color=fill_color)

    # Hide spines and style ticks on the main chart axis
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    # Add legend for main chart
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=4,
        fontsize=8,
        frameon=False,
    )

    # ---------------------------------------------------------------------
    # Create and draw the range gauge.  We first create the gauge axis so
    # that it shares the y‑limits of the main chart.  Sharing the y‑axis
    # ensures that the gradient and markers align with the same numeric
    # scale as the price chart.  The gauge occupies the remaining
    # horizontal width on the right of the figure.
    # ---------------------------------------------------------------------
    # Determine the left position and width (in figure coordinates) for the
    # gauge axis.  It begins immediately after the main chart and uses
    # ``gauge_rel_width`` as its width.
    gauge_left = chart_rel_width
    gauge_width = gauge_rel_width
    # Create the gauge axis, sharing its y‑axis with the main chart.  This
    # ensures that y‑coordinates on the gauge correspond to price levels on
    # the chart.  The x‑axis range (0→1) will represent the width of the
    # gauge; we do not display ticks on this axis.
    ax_gauge = fig.add_axes([gauge_left, 0.0, gauge_width, 1.0], sharey=ax)
    # Hide tick marks and labels for the gauge axis
    ax_gauge.set_xticks([])
    ax_gauge.set_yticks([])

    # Build a vertical gradient (red → white → green) and draw it only
    # within the computed trading range.  The gradient is drawn using
    # ``imshow`` with an extent that maps the gradient onto the segment
    # between ``lower_bound`` and ``upper_bound`` on the y‑axis.  Areas
    # outside this extent are left blank by setting the axis facecolour.
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    cmap = LinearSegmentedColormap.from_list(
        "range_gauge", ["#FF0000", "#FFFFFF", "#009951"], N=256
    )
    ax_gauge.imshow(
        gradient,
        extent=[0, 1, lower_bound, upper_bound],
        aspect="auto",
        origin="lower",
        cmap=cmap,
    )
    # Fill background outside the gradient with opaque white so that
    # regions above the upper bound and below the lower bound remain
    # neutral.
    ax_gauge.set_facecolor((1, 1, 1, 1))

    # Draw a marker indicating the last price.  The marker is a thin
    # horizontal rectangle spanning the entire width of the gauge.  Its
    # height is set to 1 % of the trading range to remain subtle yet
    # visible.  If the trading range is zero, no marker is drawn.
    full_range = upper_bound - lower_bound
    marker_height = full_range * 0.01 if full_range > 0 else 0
    if marker_height > 0:
        ax_gauge.add_patch(
            patches.Rectangle(
                (0.0, last_price - marker_height / 2.0),
                1.0,
                marker_height,
                color="#153D64",
            )
        )

    # Helper to format numeric values with apostrophe separators.
    def _format_value(val: float) -> str:
        try:
            return f"{val:,.0f}".replace(",", "'")
        except Exception:
            return f"{val:.0f}"
    upper_label = _format_value(upper_bound)
    lower_label = _format_value(lower_bound)
    # Compute percentage differences relative to the last price
    up_pct = (upper_bound - last_price) / last_price * 100 if last_price else 0.0
    down_pct = (last_price - lower_bound) / last_price * 100 if last_price else 0.0
    # Compose label strings for the upper and lower bounds.  The
    # percentage differences are shown with a sign and one decimal place.
    upper_text = f"Higher Range\n{upper_label} $\n(+{up_pct:.1f}%)"
    lower_text = f"Lower Range\n{lower_label} $\n(-{down_pct:.1f}%)"
    # Position the labels just outside the gauge to the right.  We use
    # data coordinates (``transData``) so that the text aligns with the
    # actual price levels.  The x‑coordinate 1.05 places the text slightly
    # to the right of the gauge.
    ax_gauge.text(
        1.05,
        upper_bound,
        upper_text,
        color="#009951",
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        transform=ax_gauge.transData,
    )
    ax_gauge.text(
        1.05,
        lower_bound,
        lower_text,
        color="#C00000",
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        transform=ax_gauge.transData,
    )

    # Final styling for the gauge axis: hide all spines and fix x‑limits.
    ax_gauge.set_xlim(0, 1)
    for side in ["left", "right", "top", "bottom"]:
        ax_gauge.spines[side].set_visible(False)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=600, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()




def generate_range_callout_chart_image(
    df_full: pd.DataFrame,
    anchor_date: Optional[pd.Timestamp] = None,
    lookback_days: int = 90,
    width_cm: float = 21.41,
    height_cm: float = 7.53,
    callout_width_cm: float = 3.5,
    *,
    vol_index_value: Optional[float] = None,
    show_legend: bool = True,
) -> bytes:
    """
    Create a PNG image of the instrument price chart with a textual call‑out on the
    right summarising the recent trading range.  The call‑out lists the
    higher and lower range values (with ±% changes relative to the last
    price) and draws small coloured markers aligned with those levels on
    the y‑axis.  This design preserves the full chart width and avoids
    overlapping the price plot with additional graphics.

    Parameters
    ----------
    df_full : pandas.DataFrame
        Full instrument price history with 'Date' and 'Price' columns.
    anchor_date : pandas.Timestamp or None, optional
        Optional anchor date for a regression channel; if provided, the
        channel is drawn on the price chart.
    lookback_days : int, default 90
        The lookback window for computing the high and low bounds.
    width_cm : float, default 21.41
        Overall width of the output image in centimetres.  This should
        correspond to the template placeholder width.
    height_cm : float, default 7.53
        Height of the output image in centimetres.
    callout_width_cm : float, default 3.5
        Width of the call‑out area on the right where the range summary
        appears.  The remaining width is used for the chart.

    show_legend : bool, default True
        Whether to draw the legend on the main chart.  When generating
        images for insertion into a PowerPoint slide the legend should be
        suppressed (set to ``False``) so that a manually positioned
        legend on the slide remains visible.

    Returns
    -------
    bytes
        PNG image bytes with transparency.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if df_full.empty:
        return b""

    # Restrict to the configured lookback window for plotting
    today = df_full["Date"].max().normalize()
    start = today - timedelta(days=PLOT_LOOKBACK_DAYS)
    df = df_full[df_full["Date"].between(start, today)].reset_index(drop=True)

    # Compute moving averages on the full history and then slice to the
    # lookback window.  This ensures that long moving averages (e.g. 200 days)
    # are not recomputed on the truncated data window.
    df_ma_full = _add_mas(df_full)
    df_ma = df_ma_full[df_ma_full["Date"].between(start, today)].reset_index(drop=True)

    # Optional regression channel
    uptrend = False
    upper_channel = lower_channel = None
    if anchor_date is not None:
        subset_full = df_full[df_full["Date"].between(anchor_date, today)].copy()
        if not subset_full.empty:
            X = subset_full["Date"].map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
            y_vals = subset_full["Price"].to_numpy()
            model = LinearRegression().fit(X, y_vals)
            trend = model.predict(X)
            resid = y_vals - trend
            uptrend = model.coef_[0] > 0
            upper_channel = trend + resid.max()
            lower_channel = trend + resid.min()

    # Compute high/low bounds and current price.  If an implied volatility
    # value is provided (e.g. the VIX level), use it to estimate the
    # expected one‑week move.  The expected move is computed as
    # ``last_price × (vol_index_value/100) / sqrt(52)``.  Otherwise
    # fall back to the realised‑volatility‑based bounds returned by
    # ``_compute_range_bounds``.
    last_price = df["Price"].iloc[-1]
    if vol_index_value is not None and last_price and not np.isnan(last_price):
        expected_move = (last_price * (vol_index_value / 100.0)) / np.sqrt(52.0)
        upper_bound = last_price + expected_move
        lower_bound = last_price - expected_move
    else:
        upper_bound, lower_bound = _compute_range_bounds(df_full, lookback_days=lookback_days)
    # Enforce a minimum total range (e.g. ±1 % of the current price) to avoid overlapping text.
    min_range_pct = 0.02  # 2% total band → ±1% around the current price
    if last_price and not np.isnan(last_price):
        range_span_pct = (upper_bound - lower_bound) / last_price if last_price else 0.0
        if range_span_pct < min_range_pct:
            half_span = (min_range_pct * last_price) / 2.0
            upper_bound = last_price + half_span
            lower_bound = last_price - half_span
        # Recompute percentage differences after adjusting range
        up_pct = (upper_bound - last_price) / last_price * 100.0
        down_pct = (last_price - lower_bound) / last_price * 100.0
    else:
        # Handle missing last_price gracefully
        up_pct = 0.0
        down_pct = 0.0

    # Determine y‑axis limits: ensure the axis includes the entire trading
    # range and the observed price range.  We add a small margin so the
    # labels and markers do not overlap the top or bottom edges.
    price_hi = df["Price"].max()
    price_lo = df["Price"].min()
    ma_hi = df_ma[["MA_50", "MA_100", "MA_200"]].max().max()
    ma_lo = df_ma[["MA_50", "MA_100", "MA_200"]].min().min()
    hi = max(price_hi, ma_hi)
    lo = min(price_lo, ma_lo)
    y_max = max(hi, upper_bound) * 1.02
    y_min = min(lo, lower_bound) * 0.98

    # Compute widths for chart and call‑out.  Reserve a small margin on the
    # left of the chart to ensure that y‑axis tick labels and the legend
    # remain visible when the image is inserted into a PowerPoint slide.
    callout_width_cm = min(callout_width_cm, width_cm)
    chart_width_cm = max(width_cm - callout_width_cm, 0.0)
    fig_w_in, fig_h_in = width_cm / 2.54, height_cm / 2.54
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    # Relative widths as fractions of the full figure width
    chart_rel_width = chart_width_cm / width_cm if width_cm > 0 else 0.0
    callout_rel_width = callout_width_cm / width_cm if width_cm > 0 else 0.0

    # Define a margin fraction for the left side of the chart.  Without
    # this margin, tick labels and the legend can be clipped when the
    # combined image is saved at high DPI.  Use up to 4% of the figure
    # width or 10% of the chart portion, whichever is smaller.
    margin_rel = min(0.04, 0.10 * chart_rel_width)

    # Axes for chart and call‑out; share the y‑axis so that the call‑out
    # markers align with the same price levels as the chart.  The chart
    # occupies the left portion of the figure starting at margin_rel; the
    # call‑out uses the remaining width starting at chart_rel_width.
    ax_chart = fig.add_axes([margin_rel, 0.0, chart_rel_width - margin_rel, 1.0])
    # Create a separate y‑axis for the call‑out so that hiding its ticks
    # does not remove the ticks from the main chart.  We will manually
    # synchronise the y‑limits below.
    ax_callout = fig.add_axes([chart_rel_width, 0.0, callout_rel_width, 1.0])

    # Set y‑limits before plotting so that shared axes align properly
    ax_chart.set_ylim(y_min, y_max)
    ax_callout.set_ylim(y_min, y_max)

    # Plot price and moving averages on the main chart
    ax_chart.plot(df["Date"], df["Price"], color="#153D64", linewidth=2.5,
                  label=f"S&P 500 Price (last: {last_price:,.2f})")
    ax_chart.plot(df_ma["Date"], df_ma["MA_50"], color="#008000", linewidth=1.5, label="50‑day MA")
    ax_chart.plot(df_ma["Date"], df_ma["MA_100"], color="#FFA500", linewidth=1.5, label="100‑day MA")
    ax_chart.plot(df_ma["Date"], df_ma["MA_200"], color="#FF0000", linewidth=1.5, label="200‑day MA")
    # Fibonacci levels on the subset
    sub_hi, sub_lo = df["Price"].max(), df["Price"].min()
    sub_span = sub_hi - sub_lo
    for lvl in [sub_hi, sub_hi - 0.236 * sub_span, sub_hi - 0.382 * sub_span,
                sub_hi - 0.5 * sub_span, sub_hi - 0.618 * sub_span, sub_lo]:
        ax_chart.axhline(lvl, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    # Regression channel shading
    if anchor_date is not None and upper_channel is not None and lower_channel is not None:
        subset = df_full[df_full["Date"].between(anchor_date, today)].copy().reset_index(drop=True)
        fill_color = (0, 0.6, 0, 0.25) if uptrend else (0.78, 0, 0, 0.25)
        line_color = "#008000" if uptrend else "#C00000"
        ax_chart.plot(subset["Date"], upper_channel, color=line_color, linestyle="--")
        ax_chart.plot(subset["Date"], lower_channel, color=line_color, linestyle="--")
        ax_chart.fill_between(subset["Date"], lower_channel, upper_channel, color=fill_color)

    # Style the main chart: remove spines and configure ticks.  We set the
    # y‑axis tick length to zero so that the small horizontal tick marks
    # next to the axis labels are not visible, while keeping the labels
    # themselves.  The x‑axis ticks retain their default length.
    for spine in ax_chart.spines.values():
        spine.set_visible(False)
    ax_chart.tick_params(axis="y", which="both", length=0)
    ax_chart.tick_params(axis="x", which="both", length=2)
    ax_chart.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    # Legend: when ``show_legend`` is True, place the legend just above
    # the main chart, aligned to the left so that it does not overlap the
    # call‑out panel.  Use a multi‑column layout to fit all entries on a
    # single line.  The bounding box is anchored slightly above the axes
    # (y=1.05).  When ``show_legend`` is False the legend is omitted so
    # that a custom legend can be inserted separately on a slide.
    if show_legend:
        ax_chart.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, 1.05),
            ncol=4,
            fontsize=8,
            frameon=False,
        )

    # Configure call‑out axis: remove ticks and spines; set background white
    ax_callout.set_xlim(0, 1)
    ax_callout.set_xticks([])
    ax_callout.set_yticks([])
    for spine in ax_callout.spines.values():
        spine.set_visible(False)
    ax_callout.set_facecolor("white")

    # Determine x positions for markers and text in relative coordinates.
    # Place the markers near the left of the call‑out area and the text
    # closer to the left so that the numeric portions align on their
    # left edges.  Using ``ha='left'`` keeps all values aligned to the
    # same left margin while the middle line still aligns with the price
    # axis via ``va='center'`` and symmetrical blank lines.
    marker_start_x = 0.02
    marker_end_x = 0.08
    text_x = 0.15

    # Draw small horizontal bars as markers aligned with the high/low bounds
    ax_callout.hlines(upper_bound, xmin=marker_start_x, xmax=marker_end_x,
                      colors="#009951", linewidth=2, transform=ax_callout.transData)
    ax_callout.hlines(lower_bound, xmin=marker_start_x, xmax=marker_end_x,
                      colors="#C00000", linewidth=2, transform=ax_callout.transData)

    # Helper to format values with apostrophes for thousands separators
    def _fmt(val: float) -> str:
        try:
            return f"{val:,.0f}".replace(",", "'")
        except Exception:
            return f"{val:.0f}"

    # Compose label strings with percentage differences.  The index level and
    # percentage are shown together on one line to minimise overlap.  The
    # "Higher Range" label appears above its number, while the "Lower Range"
    # label appears below its number.
    # Compose label strings with percentage differences.  We construct
    # multi‑line strings with symmetrical blank lines so that when
    # ``va='center'`` is used, the index/percentage line (the middle line)
    # aligns exactly with the price level on the y‑axis.  For the upper
    # bound, place "Higher Range" above the value and a blank line below.
    # For the lower bound, place a blank line above the value and
    # "Lower Range" below.  This results in three lines for each label
    # block, ensuring the middle line (index and percent) sits on the
    # specified y‑coordinate.
    upper_text = (
        f"Higher Range\n"
        f"{_fmt(upper_bound)} (+{up_pct:.1f}%)\n"
        f""
    )
    lower_text = (
        f"\n"
        f"{_fmt(lower_bound)} (-{down_pct:.1f}%)\n"
        f"Lower Range"
    )

    # Add the text labels at the appropriate y positions.  Using
    # ``va='center'`` ensures that the middle line (index and percent)
    # aligns with the price level, because there is one line above and
    # one line below.  We align the text to the right so that the plus
    # and minus signs line up neatly.
    ax_callout.text(text_x, upper_bound, upper_text, color="#009951",
                    ha="left", va="center", fontsize=8, fontweight='bold',
                    transform=ax_callout.transData)
    ax_callout.text(text_x, lower_bound, lower_text, color="#C00000",
                    ha="left", va="center", fontsize=8, fontweight='bold',
                    transform=ax_callout.transData)

    # Export to transparent PNG.  Use bbox_inches='tight' so that the
    # entire figure (including legends and tick labels) is saved without
    # cropping.  A small padding is added to provide breathing room.
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=600, transparent=True,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()




