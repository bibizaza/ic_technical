#!/usr/bin/env python3
"""
Automatically apply Phase 6 refactoring to app.py

This script:
1. Adds INSTRUMENT_CONFIG dictionary
2. Adds show_instrument_analysis generic function
3. Removes show_commodity_technical_analysis
4. Removes show_crypto_technical_analysis
5. Updates routing in show_technical_analysis_page
"""

from pathlib import Path
import shutil

def apply_phase6():
    """Apply Phase 6 refactoring."""

    # Read app.py
    with open('app.py') as f:
        lines = f.readlines()

    # Find insertion point (before def show_technical_analysis_page)
    insert_line = None
    for i, line in enumerate(lines):
        if line.startswith('def show_technical_analysis_page():'):
            insert_line = i
            break

    if not insert_line:
        print("ERROR: Could not find show_technical_analysis_page")
        return

    # Instrument configuration
    config_code = '''
# ============================================================================
# Instrument Configuration for Technical Analysis
# ============================================================================

INSTRUMENT_CONFIG = {
    "Equity": {
        "instruments": {
            "S&P 500": {"ticker": "SPX Index", "key": "spx", "module": "spx"},
            "CSI 300": {"ticker": "SHSZ300 INDEX", "key": "csi", "module": "csi"},
            "DAX": {"ticker": "DAX INDEX", "key": "dax", "module": "dax"},
            "IBOVESPA": {"ticker": "IBOV INDEX", "key": "ibov", "module": "ibov"},
            "MEXBOL": {"ticker": "MEXBOL INDEX", "key": "mexbol", "module": "mexbol"},
            "NIKKEI": {"ticker": "NKY INDEX", "key": "nikkei", "module": "nikkei"},
            "SENSEX": {"ticker": "SENSEX INDEX", "key": "sensex", "module": "sensex"},
            "SMI": {"ticker": "SMI INDEX", "key": "smi", "module": "smi"},
            "TASI": {"ticker": "SASEIDX INDEX", "key": "tasi", "module": "tasi"},
        },
        "session_key": "ta_equity_index",
        "default": "S&P 500",
        "module_path": "technical_analysis.equity",
    },
    "Commodity": {
        "instruments": {
            "Gold": {"ticker": "GCA Comdty", "key": "gold", "module": "gold"},
            "Silver": {"ticker": "SIA Comdty", "key": "silver", "module": "silver"},
            "Platinum": {"ticker": "XPT Comdty", "key": "platinum", "module": "platinum"},
            "Palladium": {"ticker": "XPD Curncy", "key": "palladium", "module": "palladium"},
            "Oil": {"ticker": "CL1 Comdty", "key": "oil", "module": "oil"},
            "Copper": {"ticker": "LP1 Comdty", "key": "copper", "module": "copper"},
        },
        "session_key": "ta_commodity_index",
        "default": "Gold",
        "module_path": "technical_analysis.commodity",
    },
    "Crypto": {
        "instruments": {
            "Bitcoin": {"ticker": "XBTUSD Curncy", "key": "bitcoin", "module": "bitcoin"},
            "Ethereum": {"ticker": "XETUSD Curncy", "key": "ethereum", "module": "ethereum"},
            "Ripple": {"ticker": "XRPUSD Curncy", "key": "ripple", "module": "ripple"},
            "Solana": {"ticker": "SOLUSD Curncy", "key": "solana", "module": "solana"},
            "Binance": {"ticker": "XBNCUR Curncy", "key": "binance", "module": "binance"},
        },
        "session_key": "ta_crypto_index",
        "default": "Bitcoin",
        "module_path": "technical_analysis.crypto",
    },
}


def show_instrument_analysis_generic(asset_class: str) -> None:
    """Generic technical analysis for any asset class."""
    from importlib import import_module

    excel_available = "excel_file" in st.session_state
    config = INSTRUMENT_CONFIG[asset_class]
    instruments = config["instruments"]
    session_key = config["session_key"]
    default_instrument = config["default"]
    module_path = config["module_path"]

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
    chart_title = f"{selected_instrument} Technical Chart"

    # Dynamically import module
    module = import_module(f"{module_path}.{module_name}")
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

    # Chart
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
            assessment_options = ["Bullish", "Moderately Bullish", "Neutral", "Moderately Bearish", "Bearish"]
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


'''

    # Insert config and generic function
    lines.insert(insert_line, config_code)

    # Find and replace the equity routing section
    # Find "if asset_class == "Equity":"
    equity_start = None
    for i, line in enumerate(lines):
        if 'if asset_class == "Equity":' in line:
            equity_start = i
            break

    if equity_start:
        # Replace with simple call
        # Find the elif for Commodity
        elif_line = None
        for i in range(equity_start + 1, len(lines)):
            if 'elif asset_class == "Commodity":' in line[i]:
                elif_line = i
                break

        if elif_line:
            # Replace all lines between with simple call
            lines[equity_start] = '    if asset_class == "Equity":\n'
            # Remove lines between equity_start+1 and elif_line
            del lines[equity_start+1:elif_line]
            # Insert the call
            lines.insert(equity_start+1, '        show_instrument_analysis_generic("Equity")\n')

    # Find and remove show_commodity_technical_analysis
    comm_start = None
    for i, line in enumerate(lines):
        if line.startswith('def show_commodity_technical_analysis'):
            comm_start = i
            break

    if comm_start:
        # Find next def
        comm_end = None
        for i in range(comm_start + 1, len(lines)):
            if lines[i].startswith('def '):
                comm_end = i
                break
        if comm_end:
            del lines[comm_start:comm_end]

    # Find and remove show_crypto_technical_analysis
    crypto_start = None
    for i, line in enumerate(lines):
        if line.startswith('def show_crypto_technical_analysis'):
            crypto_start = i
            break

    if crypto_start:
        # Find next def
        crypto_end = None
        for i in range(crypto_start + 1, len(lines)):
            if lines[i].startswith('def '):
                crypto_end = i
                break
        if crypto_end:
            del lines[crypto_start:crypto_end]

    # Update Commodity routing
    for i, line in enumerate(lines):
        if 'show_commodity_technical_analysis()' in line:
            lines[i] = '        show_instrument_analysis_generic("Commodity")\n'
        elif 'show_crypto_technical_analysis()' in line:
            lines[i] = '        show_instrument_analysis_generic("Crypto")\n'

    # Write back
    with open('app.py', 'w') as f:
        f.writelines(lines)

    new_lines = len(lines)
    original_lines = len(open('app_phase6_backup.py').readlines())
    reduction = original_lines - new_lines

    print()
    print("=" * 80)
    print("PHASE 6 APPLIED!")
    print("=" * 80)
    print(f"Original: {original_lines:,} lines")
    print(f"After:    {new_lines:,} lines")
    print(f"Reduction: {reduction:,} lines ({reduction/original_lines*100:.1f}%)")
    print()
    print("✓ Added INSTRUMENT_CONFIG")
    print("✓ Added show_instrument_analysis_generic function")
    print("✓ Removed show_commodity_technical_analysis")
    print("✓ Removed show_crypto_technical_analysis")
    print("✓ Updated routing calls")
    print()
    print("⚠️  IMPORTANT: Test thoroughly with 'streamlit run app.py'")

if __name__ == '__main__':
    apply_phase6()
