"""
Streamlit application for technical dashboard and presentation generation.

This application allows users to upload data, configure year‑to‑date (YTD)
charts for various asset classes, perform technical analysis on the S&P 500
index (including a selectable assessment title and a table of scores) and
generate a customised PowerPoint presentation.  The app persists
configuration selections in the session state and leverages helper functions
from the ``technical_analysis.equity.spx`` module for chart creation and
PowerPoint editing.

Key modifications relative to the original application:

* The SPX “view” title is no longer automatically derived from the average
  of technical and momentum scores.  Instead, users can select a view
  (e.g., “Strongly Bullish”) via a dropdown.  The chosen view is
  prepended with “S&P 500:” and inserted into the PowerPoint slide.
* The Streamlit interface no longer displays an average gauge for the SPX
  scores.  Instead, a simple table shows the technical score, momentum
  score and their average (DMAS), helping users judge the trend.
* The selected view is stored in ``st.session_state["spx_selected_view"]``
  and passed to ``insert_spx_technical_assessment`` when generating the
  presentation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from io import BytesIO
from pptx import Presentation
import tempfile
from pathlib import Path

# Import SPX functions from the dedicated module.  The SPX module
# resides in ``technical_analysis/equity/spx.py`` and provides all helper
# functions for building charts, inserting data into slides, and
# computing scores.  Note that ``insert_spx_technical_assessment``
# accepts a manual description and ``insert_spx_source`` inserts the
# source footnote based on the selected price mode.
from technical_analysis.equity.spx import (
    make_spx_figure,
    insert_spx_technical_chart_with_callout,
    insert_spx_technical_chart,
    insert_spx_technical_score_number,
    insert_spx_momentum_score_number,
    insert_spx_subtitle,
    insert_spx_average_gauge,
    insert_spx_technical_assessment,
    insert_spx_source,
    _get_spx_technical_score,
    _get_spx_momentum_score,
    generate_range_gauge_only_image,
)

# Import helper to adjust price data according to price mode.  The utils
# module resides at the project root (e.g. ``ic/utils.py``) so that it can
# be shared across technical analysis and performance modules.
from utils import adjust_prices_for_mode

# Import performance dashboard helpers (unchanged)
from performance.equity_perf import (
    create_weekly_performance_chart,
    create_historical_performance_table,
    insert_equity_performance_bar_slide,
    insert_equity_performance_histo_slide,
)

# Import FX performance functions
try:
    from performance.fx_perf import (
        create_weekly_performance_chart as create_weekly_fx_performance_chart,
        create_historical_performance_table as create_historical_fx_performance_table,
        insert_fx_performance_bar_slide,
        insert_fx_performance_histo_slide,
    )
except Exception:
    # If FX module not available, define no-op placeholders
    def create_weekly_fx_performance_chart(*args, **kwargs):
        return (b"", None)
    def create_historical_fx_performance_table(*args, **kwargs):
        return (b"", None)
    def insert_fx_performance_bar_slide(prs, image_bytes, *args, **kwargs):
        return prs
    def insert_fx_performance_histo_slide(prs, image_bytes, *args, **kwargs):
        return prs

# Import Crypto performance functions
try:
    from performance.crypto_perf import (
        create_weekly_performance_chart as create_weekly_crypto_performance_chart,
        create_historical_performance_table as create_historical_crypto_performance_table,
        insert_crypto_performance_bar_slide,
        insert_crypto_performance_histo_slide,
    )
except Exception:
    # If Crypto module not available, define no-op placeholders
    def create_weekly_crypto_performance_chart(*args, **kwargs):
        return (b"", None)
    def create_historical_crypto_performance_table(*args, **kwargs):
        return (b"", None)
    def insert_crypto_performance_bar_slide(prs, image_bytes, *args, **kwargs):
        return prs
    def insert_crypto_performance_histo_slide(prs, image_bytes, *args, **kwargs):
        return prs

###############################################################################
# Synthetic data helpers (fallback when no Excel is loaded)
###############################################################################

def _create_synthetic_spx_series() -> pd.DataFrame:
    """Create a synthetic SPX price series for demonstration purposes."""
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=730)
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    np.random.seed(42)
    returns = np.random.normal(loc=0, scale=0.01, size=len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"Date": dates, "Price": prices})


def _add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add moving averages to a DataFrame with a Price column."""
    out = df.copy()
    for w in (50, 100, 200):
        out[f"MA_{w}"] = out["Price"].rolling(w, min_periods=1).mean()
    return out


def _build_fallback_figure(
    df_full: pd.DataFrame, anchor_date: pd.Timestamp | None = None
) -> go.Figure:
    """
    Build a Plotly figure using synthetic data when no Excel file is loaded.
    """
    if df_full.empty:
        return go.Figure()

    today = df_full["Date"].max().normalize()
    start = today - pd.Timedelta(days=365)
    df = df_full[df_full["Date"].between(start, today)].reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Price"],
            mode="lines",
            name="S&P 500 Price",
            line=dict(color="#153D64", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df.get("MA_50", df["Price"]),
            mode="lines",
            name="50-day MA",
            line=dict(color="#008000", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df.get("MA_100", df["Price"]),
            mode="lines",
            name="100-day MA",
            line=dict(color="#FFA500", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df.get("MA_200", df["Price"]),
            mode="lines",
            name="200-day MA",
            line=dict(color="#FF0000", width=1.5),
        )
    )

    hi, lo = df["Price"].max(), df["Price"].min()
    span = hi - lo
    for lvl in [hi, hi - 0.236 * span, hi - 0.382 * span, hi - 0.5 * span, hi - 0.618 * span, lo]:
        fig.add_hline(
            y=lvl, line=dict(color="grey", dash="dash", width=1), opacity=0.6
        )

    if anchor_date is not None:
        subset = df_full[df_full["Date"].between(anchor_date, today)].copy()
        if not subset.empty:
            X = subset["Date"].map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
            y_vals = subset["Price"].to_numpy()
            model = LinearRegression().fit(X, y_vals)
            trend = model.predict(X)
            resid = y_vals - trend
            upper = trend + resid.max()
            lower = trend + resid.min()
            uptrend = model.coef_[0] > 0
            lineclr = "green" if uptrend else "red"
            fillclr = "rgba(0,150,0,0.25)" if uptrend else "rgba(200,0,0,0.25)"
            fig.add_trace(
                go.Scatter(
                    x=subset["Date"],
                    y=upper,
                    mode="lines",
                    line=dict(color=lineclr, dash="dash"),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=subset["Date"],
                    y=lower,
                    mode="lines",
                    line=dict(color=lineclr, dash="dash"),
                    fill="tonexty",
                    fillcolor=fillclr,
                    showlegend=False,
                )
            )

    fig.update_layout(
        margin=dict(l=30, r=30, t=60, b=40),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        xaxis_title=None,
        yaxis_title=None,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )
    return fig


###############################################################################
# Streamlit configuration
###############################################################################

st.set_page_config(page_title="IC Technical", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select page", ["Upload", "YTD Update", "Technical Analysis", "Generate Presentation"]
)


def show_upload_page():
    """Handle file uploads for Excel and PowerPoint templates."""
    st.sidebar.header("Upload files")
    excel_file = st.sidebar.file_uploader(
        "Upload consolidated Excel file", type=["xlsx", "xlsm", "xls"], key="excel_upload"
    )
    if excel_file is not None:
        st.session_state["excel_file"] = excel_file
    pptx_file = st.sidebar.file_uploader(
        "Upload PowerPoint template", type=["pptx", "pptm"], key="ppt_upload"
    )
    if pptx_file is not None:
        st.session_state["pptx_file"] = pptx_file
    st.sidebar.success("Files uploaded. Navigate to other pages to continue.")

    # Allow the user to choose between using the last recorded price (which may
    # be an intraday or current price) and the last close price (i.e. the
    # previous trading day's close).  The choice is stored in session state
    # and will affect how data is loaded and displayed elsewhere in the app.
    # Persist the selected price mode across pages.  Use the previously selected
    # value from session state (if any) to determine the default index.  If no
    # value has been stored yet, default to "Last Price".
    current_mode = st.session_state.get("price_mode", "Last Price")
    options = ["Last Price", "Last Close"]
    default_index = options.index(current_mode) if current_mode in options else 0
    price_mode = st.sidebar.radio(
        "Price mode",
        options=options,
        index=default_index,
        help=(
            "Select 'Last Close' to use the previous day's closing prices for all markets. "
            "Select 'Last Price' to use the most recent price in the data (which may be intraday)."
        ),
        key="price_mode_select",
    )
    st.session_state["price_mode"] = price_mode


def show_ytd_update_page():
    """Display YTD update charts and configuration."""
    st.sidebar.header("YTD Update")
    if "excel_file" not in st.session_state:
        st.sidebar.error("Please upload an Excel file on the Upload page first.")
        st.stop()

    # Lazy import heavy modules
    from ytd_perf.loader_update import load_data
    from utils import adjust_prices_for_mode
    from ytd_perf.equity_ytd import get_equity_ytd_series, create_equity_chart
    from ytd_perf.commodity_ytd import get_commodity_ytd_series, create_commodity_chart
    from ytd_perf.crypto_ytd import get_crypto_ytd_series, create_crypto_chart

    prices_df, params_df = load_data(st.session_state["excel_file"])
    # Determine whether to use the last price or the last close using
    # the centralised adjust_prices_for_mode helper.  This returns an
    # adjusted DataFrame and the effective date used for YTD calculations.
    used_date = None
    if not prices_df.empty:
        price_mode = st.session_state.get("price_mode", "Last Price")
        prices_df, used_date = adjust_prices_for_mode(prices_df, price_mode)
    # Display a caption indicating which date's prices are being used
    if used_date is not None:
        price_mode = st.session_state.get("price_mode", "Last Price")
        if price_mode == "Last Close":
            st.sidebar.caption(f"Prices as of {used_date.strftime('%d/%m/%Y')} close")
        else:
            st.sidebar.caption(f"Prices as of {used_date.strftime('%d/%m/%Y')}")

    # Equities configuration
    st.sidebar.subheader("Equities")
    eq_params = params_df[params_df["Asset Class"] == "Equity"]
    eq_name_to_ticker = {row["Name"]: row["Tickers"] for _, row in eq_params.iterrows()}
    eq_names_available = eq_params["Name"].tolist()
    default_eq = [
        name
        for name in [
            "Dax",
            "Ibov",
            "S&P 500",
            "Sensex",
            "SMI",
            "Shenzen CSI 300",
            "Nikkei 225",
            "TASI",
        ]
        if name in eq_names_available
    ]
    selected_eq_names = st.sidebar.multiselect(
        "Select equity indices",
        options=eq_names_available,
        default=st.session_state.get("selected_eq_names", default_eq),
        key="eq_indices",
    )
    st.session_state["selected_eq_names"] = selected_eq_names
    eq_tickers = [eq_name_to_ticker[name] for name in selected_eq_names]
    eq_subtitle = st.sidebar.text_input(
        "Equity subtitle", value=st.session_state.get("eq_subtitle", ""), key="eq_subtitle_input"
    )
    st.session_state["eq_subtitle"] = eq_subtitle

    # Commodities configuration
    st.sidebar.subheader("Commodities")
    co_params = params_df[params_df["Asset Class"] == "Commodity"]
    co_name_to_ticker = {row["Name"]: row["Tickers"] for _, row in co_params.iterrows()}
    co_names_available = co_params["Name"].tolist()
    default_co = [
        name
        for name in ["Gold", "Silver", "Oil (WTI)", "Platinum", "Copper", "Uranium"]
        if name in co_names_available
    ]
    selected_co_names = st.sidebar.multiselect(
        "Select commodity indices",
        options=co_names_available,
        default=st.session_state.get("selected_co_names", default_co),
        key="co_indices",
    )
    st.session_state["selected_co_names"] = selected_co_names
    co_tickers = [co_name_to_ticker[name] for name in selected_co_names]
    co_subtitle = st.sidebar.text_input(
        "Commodity subtitle", value=st.session_state.get("co_subtitle", ""), key="co_subtitle_input"
    )
    st.session_state["co_subtitle"] = co_subtitle

    # Crypto configuration
    st.sidebar.subheader("Cryptocurrencies")
    cr_params = params_df[params_df["Asset Class"] == "Crypto"]
    cr_name_to_ticker = {row["Name"]: row["Tickers"] for _, row in cr_params.iterrows()}
    cr_names_available = cr_params["Name"].tolist()
    default_cr = [
        name
        for name in ["Ripple", "Bitcoin", "Binance", "Ethereum", "Solana"]
        if name in cr_names_available
    ]
    selected_cr_names = st.sidebar.multiselect(
        "Select crypto indices",
        options=cr_names_available,
        default=st.session_state.get("selected_cr_names", default_cr),
        key="cr_indices",
    )
    st.session_state["selected_cr_names"] = selected_cr_names
    cr_tickers = [cr_name_to_ticker[name] for name in selected_cr_names]
    cr_subtitle = st.sidebar.text_input(
        "Crypto subtitle", value=st.session_state.get("cr_subtitle", ""), key="cr_subtitle_input"
    )
    st.session_state["cr_subtitle"] = cr_subtitle

    # Persist selections
    st.session_state["selected_eq_tickers"] = eq_tickers
    st.session_state["selected_co_tickers"] = co_tickers
    st.session_state["selected_cr_tickers"] = cr_tickers

    st.header("YTD Performance Charts")
    with st.expander("Equity Chart", expanded=True):
        # Pass the selected price mode to compute YTD using either
        # intraday (Last Price) or previous close (Last Close).  This
        # ensures the chart reflects the user's choice in the sidebar.
        price_mode = st.session_state.get("price_mode", "Last Price")
        df_eq = get_equity_ytd_series(
            st.session_state["excel_file"], tickers=eq_tickers, price_mode=price_mode
        )
        st.pyplot(create_equity_chart(df_eq))
    with st.expander("Commodity Chart", expanded=False):
        price_mode = st.session_state.get("price_mode", "Last Price")
        df_co = get_commodity_ytd_series(
            st.session_state["excel_file"], tickers=co_tickers, price_mode=price_mode
        )
        st.pyplot(create_commodity_chart(df_co))
    with st.expander("Crypto Chart", expanded=False):
        # Pass price mode to ensure crypto YTD uses the same intraday/close setting
        price_mode = st.session_state.get("price_mode", "Last Price")
        df_cr = get_crypto_ytd_series(
            st.session_state["excel_file"], tickers=cr_tickers, price_mode=price_mode
        )
        st.pyplot(create_crypto_chart(df_cr))

    st.sidebar.success("Configure YTD charts, then go to 'Generate Presentation'.")


def show_technical_analysis_page():
    """Display the technical analysis interface for Equity (SPX) and other asset classes."""
    st.sidebar.header("Technical Analysis")
    asset_class = st.sidebar.radio(
        "Asset class", ["Equity", "Commodity", "Crypto"], index=0
    )

    # Provide a clear channel button
    if st.sidebar.button("Clear channel", key="ta_clear_global"):
        if "ta_anchor" in st.session_state:
            st.session_state.pop("ta_anchor")
        st.experimental_rerun()

    excel_available = "excel_file" in st.session_state

    if asset_class == "Equity":
        # Load data for interactive chart (real or synthetic)
        if excel_available:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                tmp.write(st.session_state["excel_file"].getbuffer())
                tmp.flush()
                temp_path = Path(tmp.name)
            df_prices = pd.read_excel(temp_path, sheet_name="data_prices")
            df_prices = df_prices.drop(index=0)
            df_prices = df_prices[df_prices[df_prices.columns[0]] != "DATES"]
            df_prices["Date"] = pd.to_datetime(
                df_prices[df_prices.columns[0]], errors="coerce"
            )
            df_prices["Price"] = pd.to_numeric(df_prices["SPX Index"], errors="coerce")
            df_prices = df_prices.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(
                drop=True
            )
            # Adjust the prices according to the selected price mode using the helper.
            price_mode = st.session_state.get("price_mode", "Last Price")
            df_prices, spx_used_date = adjust_prices_for_mode(df_prices, price_mode)
            df_full = df_prices.copy()
            # Store the used date for later caption display
            st.session_state["spx_used_date"] = spx_used_date
        else:
            df_prices = _create_synthetic_spx_series()
            df_full = df_prices.copy()

        min_date = df_prices["Date"].min().date()
        max_date = df_prices["Date"].max().date()

        # Chart with controls in expander
        with st.expander("S&P 500 Technical Chart", expanded=True):
            # Display a caption indicating which date's prices are being used for SPX
            spx_used_date = st.session_state.get("spx_used_date")
            price_mode = st.session_state.get("price_mode", "Last Price")
            if spx_used_date is not None:
                if price_mode == "Last Close":
                    st.caption(f"Prices as of {spx_used_date.strftime('%d/%m/%Y')} close")
                else:
                    st.caption(f"Prices as of {spx_used_date.strftime('%d/%m/%Y')}")
            # -------------------------------------------------------------------
            # Display technical and momentum scores first
            # -------------------------------------------------------------------
            st.subheader("Technical and momentum scores")
            tech_score = None
            mom_score = None
            if excel_available:
                try:
                    tech_score = _get_spx_technical_score(st.session_state["excel_file"])
                except Exception:
                    tech_score = None
                try:
                    mom_score = _get_spx_momentum_score(st.session_state["excel_file"])
                except Exception:
                    mom_score = None

            # Prepare DMAS and table if both scores are available
            dmas = None
            if tech_score is not None and mom_score is not None:
                dmas = round((float(tech_score) + float(mom_score)) / 2.0, 1)
                df_scores = pd.DataFrame(
                    {
                        "Technical Score": [tech_score],
                        "Momentum Score": [mom_score],
                        "Average (DMAS)": [dmas],
                    }
                )
                st.table(df_scores)
            else:
                st.info(
                    "Technical or momentum score not available in the uploaded Excel. "
                    "Please ensure sheets 'data_technical_score' and 'data_trend_rating' exist."
                )

            # -------------------------------------------------------------------
            # Regression channel controls second
            # -------------------------------------------------------------------
            enable_channel = st.checkbox(
                "Enable regression channel",
                value=bool(st.session_state.get("ta_anchor")),
                key="spx_enable_channel",
            )

            anchor_ts = None
            if enable_channel:
                default_anchor = st.session_state.get(
                    "ta_anchor",
                    (max_date - pd.Timedelta(days=180)),
                )
                anchor_input = st.date_input(
                    "Select anchor date",
                    value=default_anchor,
                    min_value=min_date,
                    max_value=max_date,
                    key="spx_anchor_date_input",
                )
                anchor_ts = pd.to_datetime(anchor_input)
                st.session_state["ta_anchor"] = anchor_ts
            else:
                if "ta_anchor" in st.session_state:
                    st.session_state.pop("ta_anchor")
                anchor_ts = None

            # -------------------------------------------------------------------
            # Assessment selection third
            # -------------------------------------------------------------------
            if tech_score is not None and mom_score is not None and dmas is not None:
                options = [
                    "Strongly Bearish",
                    "Bearish",
                    "Slightly Bearish",
                    "Neutral",
                    "Slightly Bullish",
                    "Bullish",
                    "Strongly Bullish",
                ]
                def _default_index_from_dmas(val: float) -> int:
                    if val >= 80:
                        return options.index("Strongly Bullish")
                    elif val >= 70:
                        return options.index("Bullish")
                    elif val >= 60:
                        return options.index("Slightly Bullish")
                    elif val >= 40:
                        return options.index("Neutral")
                    elif val >= 30:
                        return options.index("Slightly Bearish")
                    elif val >= 20:
                        return options.index("Bearish")
                    else:
                        return options.index("Strongly Bearish")

                default_idx = _default_index_from_dmas(dmas)
                user_view = st.selectbox(
                    "Select your assessment", options, index=default_idx, key="spx_view_select"
                )
                st.session_state["spx_selected_view"] = user_view
                st.caption(
                    "Your selection will override the automatically computed view in the presentation."
                )

            # -------------------------------------------------------------------
            # Subtitle input fourth
            # -------------------------------------------------------------------
            spx_subtitle = st.text_input(
                "SPX subtitle",
                value=st.session_state.get("spx_subtitle", ""),
                key="spx_subtitle_input",
            )
            st.session_state["spx_subtitle"] = spx_subtitle

            # -------------------------------------------------------------------
            # Finally, build and show the interactive chart
            # -------------------------------------------------------------------
            if excel_available:
                # Pass the selected price mode to make_spx_figure so that the
                # interactive chart reflects the chosen last-price or last-close
                # setting.  The price mode is persisted in session state.
                pmode = st.session_state.get("price_mode", "Last Price")
                fig = make_spx_figure(temp_path, anchor_date=anchor_ts, price_mode=pmode)
            else:
                df_ma = _add_moving_averages(df_full)
                fig = _build_fallback_figure(df_ma, anchor_date=anchor_ts)

            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Use the controls above to enable and configure the regression channel. "
                "Green shading indicates an uptrend; red shading indicates a downtrend."
            )

    else:
        with st.expander(f"{asset_class} technical charts", expanded=False):
            st.info(f"{asset_class} technical analysis not implemented yet.")


def show_generate_presentation_page():
    """Generate a customised PowerPoint presentation based on user selections."""
    st.sidebar.header("Generate Presentation")
    if "excel_file" not in st.session_state or "pptx_file" not in st.session_state:
        st.sidebar.error(
            "Please upload both an Excel file and a PowerPoint template in the Upload page."
        )
        st.stop()

    # Lazy import functions for inserting charts into PPT
    from ytd_perf.equity_ytd import insert_equity_chart
    from ytd_perf.commodity_ytd import insert_commodity_chart
    from ytd_perf.crypto_ytd import insert_crypto_chart

    st.sidebar.write("### Summary of selections")
    st.sidebar.write("Equities:", st.session_state.get("selected_eq_names", []))
    st.sidebar.write("Commodities:", st.session_state.get("selected_co_names", []))
    st.sidebar.write("Cryptos:", st.session_state.get("selected_cr_names", []))
    # Display FX pairs being analysed (fixed list) for user awareness
    st.sidebar.write(
        "FX:",
        [
            "DXY",
            "EUR/USD",
            "EUR/CHF",
            "EUR/GBP",
            "EUR/JPY",
            "EUR/AUD",
            "EUR/CAD",
            "EUR/BRL",
            "EUR/RUB",
            "EUR/ZAR",
            "EUR/MXN",
        ],
    )

    if st.sidebar.button("Generate updated PPTX", key="gen_ppt_button"):
        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as tmp_input:
            tmp_input.write(st.session_state["pptx_file"].getbuffer())
            tmp_input.flush()
            prs = Presentation(tmp_input.name)

        # Insert YTD charts
        prs = insert_equity_chart(
            prs,
            st.session_state["excel_file"],
            subtitle=st.session_state.get("eq_subtitle", ""),
            tickers=st.session_state.get("selected_eq_tickers", []),
            price_mode=st.session_state.get("price_mode", "Last Price"),
        )
        prs = insert_commodity_chart(
            prs,
            st.session_state["excel_file"],
            subtitle=st.session_state.get("co_subtitle", ""),
            tickers=st.session_state.get("selected_co_tickers", []),
            price_mode=st.session_state.get("price_mode", "Last Price"),
        )
        prs = insert_crypto_chart(
            prs,
            st.session_state["excel_file"],
            subtitle=st.session_state.get("cr_subtitle", ""),
            tickers=st.session_state.get("selected_cr_tickers", []),
            price_mode=st.session_state.get("price_mode", "Last Price"),
        )

        # Insert SPX technical-analysis chart with the call-out range gauge.
        anchor_dt = st.session_state.get("ta_anchor")
        # Use the selected price mode when inserting the SPX technical chart into
        # the presentation.  This ensures that the exported chart reflects
        # either the last price or the previous close, matching the
        # interactive chart in Streamlit.
        pmode = st.session_state.get("price_mode", "Last Price")
        prs = insert_spx_technical_chart_with_callout(
            prs,
            st.session_state["excel_file"],
            anchor_dt,
            price_mode=pmode,
        )

        # Insert SPX technical score number
        prs = insert_spx_technical_score_number(
            prs,
            st.session_state["excel_file"],
        )

        # Insert SPX momentum score number
        prs = insert_spx_momentum_score_number(
            prs,
            st.session_state["excel_file"]
        )

        # Insert SPX subtitle (from user input in UI)
        prs = insert_spx_subtitle(
            prs,
            st.session_state.get("spx_subtitle", ""),
        )

        # Insert SPX average gauge (last week's average is 0–100)
        last_week_avg = st.session_state.get("spx_last_week_avg", 50.0)
        prs = insert_spx_average_gauge(
            prs,
            st.session_state["excel_file"],
            last_week_avg,
        )

        # Insert the technical assessment text into the 'spx_view' textbox.
        # Use the user-selected view if available; otherwise fall back to computed view.
        manual_view = st.session_state.get("spx_selected_view")
        prs = insert_spx_technical_assessment(
            prs,
            st.session_state["excel_file"],
            manual_desc=manual_view,
        )

        # ------------------------------------------------------------------
        # Insert the SPX source footnote.  The text varies depending on
        # whether the user selected "Last Price" or "Last Close".  We
        # compute the used date by loading the SPX price data, applying
        # the same price mode adjustment, and taking the maximum date.
        # If the date cannot be determined, the footnote is left unchanged.
        # ------------------------------------------------------------------
        try:
            # Read raw price data for SPX
            import pandas as pd
            temp_file = st.session_state["excel_file"]
            df_prices = pd.read_excel(temp_file, sheet_name="data_prices")
            df_prices = df_prices.drop(index=0)
            df_prices = df_prices[df_prices[df_prices.columns[0]] != "DATES"]
            df_prices["Date"] = pd.to_datetime(df_prices[df_prices.columns[0]], errors="coerce")
            df_prices["Price"] = pd.to_numeric(df_prices["SPX Index"], errors="coerce")
            df_prices = df_prices.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)[
                ["Date", "Price"]
            ]
            # Apply price mode adjustment to get the used date
            price_mode = st.session_state.get("price_mode", "Last Price")
            df_adj, used_date = adjust_prices_for_mode(df_prices, price_mode)
        except Exception:
            used_date = None
            price_mode = st.session_state.get("price_mode", "Last Price")

        prs = insert_spx_source(
            prs,
            used_date,
            price_mode,
        )

        # ------------------------------------------------------------------
        # Insert Equity performance charts
        # ------------------------------------------------------------------
        try:
            # Generate the weekly performance bar chart with price-mode adjustment
            bar_bytes, perf_used_date = create_weekly_performance_chart(
                st.session_state["excel_file"],
                price_mode=st.session_state.get("price_mode", "Last Price"),
            )
            prs = insert_equity_performance_bar_slide(
                prs,
                bar_bytes,
                used_date=perf_used_date,
                price_mode=st.session_state.get("price_mode", "Last Price"),
                left_cm=1.63,
                top_cm=4.73,
                width_cm=22.48,
                height_cm=10.61,
            )
            # Generate the historical performance heatmap with price-mode adjustment
            histo_bytes, histo_used_date = create_historical_performance_table(
                st.session_state["excel_file"],
                price_mode=st.session_state.get("price_mode", "Last Price"),
            )
            prs = insert_equity_performance_histo_slide(
                prs,
                histo_bytes,
                used_date=histo_used_date,
                price_mode=st.session_state.get("price_mode", "Last Price"),
                left_cm=2.16,
                top_cm=4.70,
                width_cm=19.43,
                height_cm=10.61,
            )

            # ------------------------------------------------------------------
            # Insert FX performance charts
            # ------------------------------------------------------------------
            # Generate the weekly FX performance bar chart with price-mode adjustment
            fx_bar_bytes, fx_used_date = create_weekly_fx_performance_chart(
                st.session_state["excel_file"],
                price_mode=st.session_state.get("price_mode", "Last Price"),
            )
            prs = insert_fx_performance_bar_slide(
                prs,
                fx_bar_bytes,
                used_date=fx_used_date,
                price_mode=st.session_state.get("price_mode", "Last Price"),
                left_cm=1.63,
                top_cm=4.73,
                width_cm=22.48,
                height_cm=10.61,
            )

            # Generate the FX historical performance heatmap with price-mode adjustment
            fx_histo_bytes, fx_used_date2 = create_historical_fx_performance_table(
                st.session_state["excel_file"],
                price_mode=st.session_state.get("price_mode", "Last Price"),
            )
            prs = insert_fx_performance_histo_slide(
                prs,
                fx_histo_bytes,
                used_date=fx_used_date2,
                price_mode=st.session_state.get("price_mode", "Last Price"),
                left_cm=2.16,
                top_cm=4.70,
                width_cm=19.43,
                height_cm=10.61,
            )

            # ------------------------------------------------------------------
            # Insert cryptocurrency performance charts
            # ------------------------------------------------------------------
            # Generate the weekly crypto performance bar chart with price-mode adjustment
            crypto_bar_bytes, crypto_used_date = create_weekly_crypto_performance_chart(
                st.session_state["excel_file"],
                price_mode=st.session_state.get("price_mode", "Last Price"),
            )
            prs = insert_crypto_performance_bar_slide(
                prs,
                crypto_bar_bytes,
                used_date=crypto_used_date,
                price_mode=st.session_state.get("price_mode", "Last Price"),
                left_cm=1.63,
                top_cm=4.73,
                width_cm=22.48,
                height_cm=10.61,
            )

            # Generate the cryptocurrency historical performance heatmap with price-mode adjustment
            crypto_histo_bytes, crypto_used_date2 = create_historical_crypto_performance_table(
                st.session_state["excel_file"],
                price_mode=st.session_state.get("price_mode", "Last Price"),
            )
            prs = insert_crypto_performance_histo_slide(
                prs,
                crypto_histo_bytes,
                used_date=crypto_used_date2,
                price_mode=st.session_state.get("price_mode", "Last Price"),
                left_cm=2.16,
                top_cm=4.70,
                width_cm=19.43,
                height_cm=10.61,
            )
        except Exception:
            # If anything fails, continue without the performance slides
            pass

        out_stream = BytesIO()
        prs.save(out_stream)
        out_stream.seek(0)
        updated_bytes = out_stream.getvalue()

        if st.session_state["pptx_file"].name.lower().endswith(".pptm"):
            fname = "updated_presentation.pptm"
            mime = "application/vnd.ms-powerpoint.presentation.macroEnabled.12"
        else:
            fname = "updated_presentation.pptx"
            mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

        st.sidebar.success("Updated presentation created successfully.")
        st.sidebar.download_button(
            "Download updated presentation",
            data=updated_bytes,
            file_name=fname,
            mime=mime,
            key="download_ppt_button",
        )

    st.write("Click the button in the sidebar to generate your updated presentation.")


# -----------------------------------------------------------------------------
# Main navigation dispatch
# -----------------------------------------------------------------------------
if page == "Upload":
    show_upload_page()
elif page == "YTD Update":
    show_ytd_update_page()
elif page == "Technical Analysis":
    show_technical_analysis_page()
elif page == "Generate Presentation":
    show_generate_presentation_page()