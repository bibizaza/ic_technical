"""
Streamlit application for technical dashboard and presentation generation.

This application allows users to upload data, configure year‑to‑date (YTD)
charts for various asset classes, perform technical analysis on the S&P 500
index (including a new higher‑range/lower‑range gauge) and generate a
customised PowerPoint presentation.  The app persists configuration
selections in the session state and leverages helper functions from the
``technical_analysis.equity.spx`` module for chart creation and PowerPoint
editing.
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

# Import SPX functions from the dedicated module
from technical_analysis.equity.spx import (
    make_spx_figure,
    insert_spx_technical_chart_with_callout,
    insert_spx_technical_score_number,
    insert_spx_momentum_score_number,
    insert_spx_subtitle,
    generate_average_gauge_image,
    _get_spx_technical_score,
    _get_spx_momentum_score,
    insert_spx_average_gauge,
    insert_spx_technical_assessment,
    generate_range_gauge_only_image,
)

# -----------------------------------------------------------------------------
# Fallback helpers for interactive chart if no Excel
# -----------------------------------------------------------------------------
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
    """Build a Plotly figure using synthetic data when no Excel file is loaded."""
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


# -----------------------------------------------------------------------------
# Streamlit configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="IC Technical", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select page", ["Upload", "YTD Update", "Technical Analysis", "Generate Presentation"]
)

# ---------------------------------------------------------------------------
# UPLOAD page
# ---------------------------------------------------------------------------
if page == "Upload":
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

# ---------------------------------------------------------------------------
# YTD UPDATE page
# ---------------------------------------------------------------------------
elif page == "YTD Update":
    st.sidebar.header("YTD Update")
    if "excel_file" not in st.session_state:
        st.sidebar.error("Please upload an Excel file on the Upload page first.")
        st.stop()

    # Lazy import heavy modules
    from ytd_perf.loader_update import load_data
    from ytd_perf.equity_ytd import get_equity_ytd_series, create_equity_chart
    from ytd_perf.commodity_ytd import get_commodity_ytd_series, create_commodity_chart
    from ytd_perf.crypto_ytd import get_crypto_ytd_series, create_crypto_chart

    prices_df, params_df = load_data(st.session_state["excel_file"])

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
        df_eq = get_equity_ytd_series(st.session_state["excel_file"], tickers=eq_tickers)
        st.pyplot(create_equity_chart(df_eq))
    with st.expander("Commodity Chart", expanded=False):
        df_co = get_commodity_ytd_series(st.session_state["excel_file"], tickers=co_tickers)
        st.pyplot(create_commodity_chart(df_co))
    with st.expander("Crypto Chart", expanded=False):
        df_cr = get_crypto_ytd_series(st.session_state["excel_file"], tickers=cr_tickers)
        st.pyplot(create_crypto_chart(df_cr))

    st.sidebar.success("Configure YTD charts, then go to 'Generate Presentation'.")

# ---------------------------------------------------------------------------
# TECHNICAL ANALYSIS page
# ---------------------------------------------------------------------------
elif page == "Technical Analysis":
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
            df_full = df_prices.copy()
        else:
            df_prices = _create_synthetic_spx_series()
            df_full = df_prices.copy()

        min_date = df_prices["Date"].min().date()
        max_date = df_prices["Date"].max().date()

        # Chart with controls in expander
        with st.expander("S&P 500 Technical Chart", expanded=True):
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

            # Subtitle
            spx_subtitle = st.text_input(
                "SPX subtitle",
                value=st.session_state.get("spx_subtitle", ""),
                key="spx_subtitle_input",
            )
            st.session_state["spx_subtitle"] = spx_subtitle

            # Build interactive figure
            if excel_available:
                fig = make_spx_figure(temp_path, anchor_date=anchor_ts)
            else:
                df_ma = _add_moving_averages(df_full)
                fig = _build_fallback_figure(df_ma, anchor_date=anchor_ts)

            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Use the controls above to enable and configure the regression channel. "
                "Green shading indicates an uptrend; red shading indicates a downtrend."
            )

            # -------------------------------------------------------------------
            # Gauge for average of technical & momentum scores
            # -------------------------------------------------------------------
            st.markdown("---")
            st.subheader("Average technical and momentum score")
            # Request last week's average on a 0–100 scale. Persist in session.
            last_week_avg_input = st.number_input(
                "Enter last week's average (0–100)",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.get("spx_last_week_avg", 50.0)),
                step=1.0,
                key="spx_last_week_avg_input",
            )
            st.session_state["spx_last_week_avg"] = float(last_week_avg_input)

            # Compute current technical and momentum scores from Excel if available
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

            # If scores are found, draw gauge; otherwise inform user
            if tech_score is not None and mom_score is not None:
                # Determine date text from loaded data: for Excel use the last date in df_full; otherwise today's date
                try:
                    if excel_available and not df_full.empty:
                        max_dt = pd.to_datetime(df_full["Date"]).max()
                        date_label = (
                            max_dt.strftime("As of %d.%m.%Y") if pd.notnull(max_dt) else None
                        )
                    else:
                        date_label = pd.Timestamp.today().strftime("As of %d.%m.%Y")
                except Exception:
                    date_label = None

                gauge_bytes = generate_average_gauge_image(
                    tech_score,
                    mom_score,
                    float(last_week_avg_input),
                    date_text=date_label,
                    last_label_text="Previous Week",
                )
                st.image(gauge_bytes, caption="Average score gauge")
            else:
                st.info(
                    "Technical or momentum score not available in the uploaded Excel. "
                    "Please ensure sheets 'data_technical_score' and 'data_trend_rating' exist."
                )

            # Note: the vertical range gauge is integrated into the PPT slide only.
            # To keep the Streamlit interface clean, we omit displaying the gauge
            # separately here.  Users will see the full trading range gauge in
            # the generated presentation.
    else:
        with st.expander(f"{asset_class} technical charts", expanded=False):
            st.info(f"{asset_class} technical analysis not implemented yet.")

# ---------------------------------------------------------------------------
# GENERATE PRESENTATION page
# ---------------------------------------------------------------------------
elif page == "Generate Presentation":
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
        )
        prs = insert_commodity_chart(
            prs,
            st.session_state["excel_file"],
            subtitle=st.session_state.get("co_subtitle", ""),
            tickers=st.session_state.get("selected_co_tickers", []),
        )
        prs = insert_crypto_chart(
            prs,
            st.session_state["excel_file"],
            subtitle=st.session_state.get("cr_subtitle", ""),
            tickers=st.session_state.get("selected_cr_tickers", []),
        )

        # Insert SPX technical-analysis chart with call-out range
        anchor_dt = st.session_state.get("ta_anchor")
        prs = insert_spx_technical_chart_with_callout(
            prs,
            st.session_state["excel_file"],
            anchor_dt,
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

        # Insert SPX subtitle
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

        # Insert the technical assessment text into the 'tech_spx' textbox
        prs = insert_spx_technical_assessment(
            prs,
            st.session_state["excel_file"],
        )

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