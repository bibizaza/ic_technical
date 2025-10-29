# Phase 6 Configuration Template
# Add this configuration before show_technical_analysis_page


# ============================================================================
# Instrument Configuration for Technical Analysis
# ============================================================================
# This configuration defines all available instruments for technical analysis
# across Equity, Commodity, and Crypto asset classes.

INSTRUMENT_CONFIG = {
    "Equity": {
        "instruments": {
            "S&P 500": {"ticker": "SPX Index", "key": "spx", "module": "spx", "vol_index": "VIX Index"},
            "CSI 300": {"ticker": "SHSZ300 INDEX", "key": "csi", "module": "csi", "vol_index": None},
            "DAX": {"ticker": "DAX INDEX", "key": "dax", "module": "dax", "vol_index": "V2X Index"},
            "IBOVESPA": {"ticker": "IBOV INDEX", "key": "ibov", "module": "ibov", "vol_index": None},
            "MEXBOL": {"ticker": "MEXBOL INDEX", "key": "mexbol", "module": "mexbol", "vol_index": None},
            "NIKKEI": {"ticker": "NKY INDEX", "key": "nikkei", "module": "nikkei", "vol_index": "VNKY Index"},
            "SENSEX": {"ticker": "SENSEX INDEX", "key": "sensex", "module": "sensex", "vol_index": None},
            "SMI": {"ticker": "SMI INDEX", "key": "smi", "module": "smi", "vol_index": "V2TX Index"},
            "TASI": {"ticker": "SASEIDX INDEX", "key": "tasi", "module": "tasi", "vol_index": None},
        },
        "session_key": "ta_equity_index",
        "default": "S&P 500",
    },
    "Commodity": {
        "instruments": {
            "Gold": {"ticker": "GCA Comdty", "key": "gold", "module": "gold", "vol_index": "XAUUSDV1M BGN Curncy"},
            "Silver": {"ticker": "SIA Comdty", "key": "silver", "module": "silver", "vol_index": None},
            "Platinum": {"ticker": "XPT Comdty", "key": "platinum", "module": "platinum", "vol_index": None},
            "Palladium": {"ticker": "XPD Curncy", "key": "palladium", "module": "palladium", "vol_index": None},
            "Oil": {"ticker": "CL1 Comdty", "key": "oil", "module": "oil", "vol_index": "OVX Index"},
            "Copper": {"ticker": "LP1 Comdty", "key": "copper", "module": "copper", "vol_index": None},
        },
        "session_key": "ta_commodity_index",
        "default": "Gold",
    },
    "Crypto": {
        "instruments": {
            "Bitcoin": {"ticker": "XBTUSD Curncy", "key": "bitcoin", "module": "bitcoin", "vol_index": "BVXS Index"},
            "Ethereum": {"ticker": "XETUSD Curncy", "key": "ethereum", "module": "ethereum", "vol_index": None},
            "Ripple": {"ticker": "XRPUSD Curncy", "key": "ripple", "module": "ripple", "vol_index": None},
            "Solana": {"ticker": "SOLUSD Curncy", "key": "solana", "module": "solana", "vol_index": None},
            "Binance": {"ticker": "XBNCUR Curncy", "key": "binance", "module": "binance", "vol_index": None},
        },
        "session_key": "ta_crypto_index",
        "default": "Bitcoin",
    },
}


# Generic function to replace the 3 asset-specific functions

def show_instrument_analysis(asset_class: str) -> None:
    """
    Generic technical analysis interface for any asset class (Equity, Commodity, Crypto).

    This consolidated function replaces the previous separate implementations for
    each asset class, reducing code duplication by ~1,000 lines.

    Parameters
    ----------
    asset_class : str
        One of "Equity", "Commodity", or "Crypto"
    """
    from importlib import import_module

    excel_available = "excel_file" in st.session_state

    # Get configuration for this asset class
    config = INSTRUMENT_CONFIG[asset_class]
    instruments = config["instruments"]
    session_key = config["session_key"]
    default_instrument = config["default"]

    # Instrument selection
    instrument_options = list(instruments.keys())
    default_idx = st.session_state.get(session_key, default_instrument)
    if default_idx not in instrument_options:
        default_idx = default_instrument

    selected_instrument = st.sidebar.selectbox(
        f"Select {asset_class.lower()} for technical analysis",
        options=instrument_options,
        index=instrument_options.index(default_idx),
        key=f"{session_key}_select",
    )
    st.session_state[session_key] = selected_instrument

    # Get instrument details
    inst = instruments[selected_instrument]
    ticker = inst["ticker"]
    ticker_key = inst["key"]
    module_name = inst["module"]
    vol_index = inst["vol_index"]
    chart_title = f"{selected_instrument} Technical Chart"

    # Dynamically import the module functions
    if asset_class == "Equity":
        module = import_module(f"technical_analysis.equity.{module_name}")
    elif asset_class == "Commodity":
        module = import_module(f"technical_analysis.commodity.{module_name}")
    else:  # Crypto
        module = import_module(f"technical_analysis.crypto.{module_name}")

    # Get functions from module
    make_figure_func = getattr(module, f"make_{ticker_key}_figure")
    get_tech_score = getattr(module, f"_get_{ticker_key}_technical_score")
    get_mom_score = getattr(module, f"_get_{ticker_key}_momentum_score")

    # Load price data
    if excel_available:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(st.session_state["excel_file"].getbuffer())
            tmp.flush()
            temp_path = Path(tmp.name)

        df_prices = pd.read_excel(temp_path, sheet_name="data_prices")
        df_prices = df_prices.drop(index=0)
        df_prices = df_prices[df_prices[df_prices.columns[0]] != "DATES"]
        df_prices["Date"] = pd.to_datetime(df_prices[df_prices.columns[0]], errors="coerce")
        df_prices["Price"] = pd.to_numeric(df_prices[ticker], errors="coerce")
        df_prices = df_prices.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)

        price_mode = st.session_state.get("price_mode", "Last Price")
        if adjust_prices_for_mode is not None and price_mode:
            try:
                df_prices, used_date = adjust_prices_for_mode(df_prices, price_mode)
            except Exception:
                used_date = None
        else:
            used_date = None
        df_full = df_prices.copy()
        st.session_state[f"{ticker_key}_used_date"] = used_date
    else:
        df_prices = _create_synthetic_spx_series()
        df_full = df_prices.copy()
        used_date = None

    min_date = df_prices["Date"].min().date()
    max_date = df_prices["Date"].max().date()

    # Chart and controls
    with st.expander(chart_title, expanded=True):
        used_date = st.session_state.get(f"{ticker_key}_used_date")
        price_mode = st.session_state.get("price_mode", "Last Price")
        if used_date is not None:
            if price_mode == "Last Close":
                st.caption(f"Using closing price as of {used_date.strftime('%Y-%m-%d')}")
            else:
                st.caption(f"Using last available price as of {used_date.strftime('%Y-%m-%d')}")

        # Regression channel controls
        enable_channel = st.checkbox(
            f"Enable regression channel for {selected_instrument}",
            value=st.session_state.get(f"{ticker_key}_enable_channel", False),
            key=f"{ticker_key}_channel_checkbox",
        )
        st.session_state[f"{ticker_key}_enable_channel"] = enable_channel

        if enable_channel:
            default_anchor = st.session_state.get(f"{ticker_key}_anchor_date", min_date)
            if not (min_date <= default_anchor <= max_date):
                default_anchor = min_date

            anchor_date_input = st.date_input(
                f"Select anchor date for {selected_instrument} regression channel",
                value=default_anchor,
                min_value=min_date,
                max_value=max_date,
                key=f"{ticker_key}_anchor_date_input",
            )
            st.session_state[f"{ticker_key}_anchor_date"] = anchor_date_input
            anchor_ts = pd.Timestamp(anchor_date_input)
        else:
            anchor_ts = None

        # Generate chart
        pmode = st.session_state.get("price_mode", "Last Price")
        try:
            fig = make_figure_func(temp_path if excel_available else None, anchor_date=anchor_ts, price_mode=pmode)
        except Exception:
            fig = go.Figure()
            fig.add_annotation(text=f"Unable to load {selected_instrument} data", showarrow=False)

        st.plotly_chart(fig, use_container_width=True)
        st.caption("Use the controls above to enable and configure the regression channel.")

    # Technical assessment
    if excel_available:
        with st.expander(f"{selected_instrument} Technical Assessment", expanded=True):
            assessment_options = [
                "Bullish", "Moderately Bullish", "Neutral",
                "Moderately Bearish", "Bearish"
            ]
            default_assessment = st.session_state.get(f"{ticker_key}_assessment", "Neutral")
            if default_assessment not in assessment_options:
                default_assessment = "Neutral"

            selected_assessment = st.selectbox(
                f"Select {selected_instrument} assessment",
                options=assessment_options,
                index=assessment_options.index(default_assessment),
                key=f"{ticker_key}_assessment_select",
            )
            st.session_state[f"{ticker_key}_assessment"] = selected_assessment
            st.write(f"**Current assessment**: {selected_assessment}")

        # Subtitle
        with st.expander(f"{selected_instrument} Subtitle", expanded=False):
            default_subtitle = st.session_state.get(f"{ticker_key}_subtitle", "")
            subtitle_input = st.text_input(
                f"Enter subtitle for {selected_instrument}",
                value=default_subtitle,
                key=f"{ticker_key}_subtitle_input",
            )
            st.session_state[f"{ticker_key}_subtitle"] = subtitle_input
            if subtitle_input:
                st.write(f"**Subtitle**: {subtitle_input}")

        # Display scores
        try:
            tech_score = get_tech_score(temp_path)
            mom_score = get_mom_score(temp_path)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Technical Score", f"{tech_score:.0f}" if tech_score is not None else "N/A")
            with col2:
                st.metric("Momentum Score", f"{mom_score:.0f}" if mom_score is not None else "N/A")
        except Exception:
            st.warning(f"Could not calculate scores for {selected_instrument}")
