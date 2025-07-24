"""
app.py

Streamlit app with sidebar navigation: Upload → YTD Update → Generate Presentation.
Data and selections are persisted in st.session_state to allow separate steps.
"""

import streamlit as st
from pptx import Presentation
from io import BytesIO
import tempfile

# Import our asset-specific modules
from ytd_perf.loader_update import load_data
from ytd_perf.equity_ytd import (
    get_equity_ytd_series,
    create_equity_chart,
    insert_equity_chart,
)
from ytd_perf.commodity_ytd import (
    get_commodity_ytd_series,
    create_commodity_chart,
    insert_commodity_chart,
)
from ytd_perf.crypto_ytd import (
    get_crypto_ytd_series,
    create_crypto_chart,
    insert_crypto_chart,
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select page", ["Upload", "YTD Update", "Generate Presentation"])

# UPLOAD page
if page == "Upload":
    st.sidebar.header("Upload files")
    excel_file = st.sidebar.file_uploader("Upload consolidated Excel file", type=["xlsx", "xlsm", "xls"])
    if excel_file is not None:
        st.session_state["excel_file"] = excel_file
    pptx_file = st.sidebar.file_uploader("Upload PowerPoint template", type=["pptx", "pptm"])
    if pptx_file is not None:
        st.session_state["pptx_file"] = pptx_file
    st.sidebar.success("Files uploaded. Navigate to 'YTD Update' to configure.")

# YTD UPDATE page
elif page == "YTD Update":
    st.sidebar.header("YTD Update")
    if "excel_file" not in st.session_state:
        st.sidebar.error("Please upload an Excel file on the Upload page first.")
        st.stop()
    # Load data
    prices_df, params_df = load_data(st.session_state["excel_file"])

    # Equities configuration
    st.sidebar.subheader("Equities")
    eq_params = params_df[params_df["Asset Class"] == "Equity"]
    eq_name_to_ticker = {row["Name"]: row["Tickers"] for _, row in eq_params.iterrows()}
    eq_names_available = eq_params["Name"].tolist()
    eq_default_names = [
        name for name in ["Dax", "Ibov", "S&P 500", "Sensex", "SMI", "Shenzen CSI 300", "Nikkei 225", "TASI"]
        if name in eq_names_available
    ]
    selected_eq_names = st.sidebar.multiselect(
        "Select equity indices",
        options=eq_names_available,
        default=st.session_state.get("selected_eq_names", eq_default_names),
    )
    st.session_state["selected_eq_names"] = selected_eq_names
    eq_tickers = [eq_name_to_ticker[name] for name in selected_eq_names]
    eq_subtitle = st.sidebar.text_input(
        "Equity subtitle", value=st.session_state.get("eq_subtitle", ""), key="eq_sub"
    )
    st.session_state["eq_subtitle"] = eq_subtitle

    # Commodities configuration
    st.sidebar.subheader("Commodities")
    co_params = params_df[params_df["Asset Class"] == "Commodity"]
    co_name_to_ticker = {row["Name"]: row["Tickers"] for _, row in co_params.iterrows()}
    co_names_available = co_params["Name"].tolist()
    co_default_names = [
        name for name in ["Gold", "Silver", "Oil (WTI)", "Platinum", "Copper", "Uranium"]
        if name in co_names_available
    ]
    selected_co_names = st.sidebar.multiselect(
        "Select commodity indices",
        options=co_names_available,
        default=st.session_state.get("selected_co_names", co_default_names),
    )
    st.session_state["selected_co_names"] = selected_co_names
    co_tickers = [co_name_to_ticker[name] for name in selected_co_names]
    co_subtitle = st.sidebar.text_input(
        "Commodity subtitle", value=st.session_state.get("co_subtitle", ""), key="co_sub"
    )
    st.session_state["co_subtitle"] = co_subtitle

    # Crypto configuration
    st.sidebar.subheader("Cryptocurrencies")
    cr_params = params_df[params_df["Asset Class"] == "Crypto"]
    cr_name_to_ticker = {row["Name"]: row["Tickers"] for _, row in cr_params.iterrows()}
    cr_names_available = cr_params["Name"].tolist()
    cr_default_names = [
        name for name in ["Ripple", "Bitcoin", "Binance", "Ethereum", "Solana"]
        if name in cr_names_available
    ]
    selected_cr_names = st.sidebar.multiselect(
        "Select crypto indices",
        options=cr_names_available,
        default=st.session_state.get("selected_cr_names", cr_default_names),
    )
    st.session_state["selected_cr_names"] = selected_cr_names
    cr_tickers = [cr_name_to_ticker[name] for name in selected_cr_names]
    cr_subtitle = st.sidebar.text_input(
        "Crypto subtitle", value=st.session_state.get("cr_subtitle", ""), key="cr_sub"
    )
    st.session_state["selected_eq_tickers"] = eq_tickers
    st.session_state["selected_co_tickers"] = co_tickers
    st.session_state["selected_cr_tickers"] = cr_tickers

    # Show charts on main page inside expanders
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

# GENERATE PRESENTATION page
elif page == "Generate Presentation":
    st.sidebar.header("Generate Presentation")
    if "excel_file" not in st.session_state or "pptx_file" not in st.session_state:
        st.sidebar.error("Please upload both an Excel file and a PowerPoint template in the Upload page.")
        st.stop()
    # Show summary of selected tickers
    st.sidebar.write("### Summary of selections")
    st.sidebar.write("Equities:", st.session_state.get("selected_eq_names", []))
    st.sidebar.write("Commodities:", st.session_state.get("selected_co_names", []))
    st.sidebar.write("Cryptos:", st.session_state.get("selected_cr_names", []))
    # Button to generate updated PPT
    if st.sidebar.button("Generate updated PPTX"):
        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as tmp_input:
            tmp_input.write(st.session_state["pptx_file"].getbuffer())
            tmp_input.flush()
            prs = Presentation(tmp_input.name)
        # Insert each chart
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

        # Save to bytes
        out_stream = BytesIO()
        prs.save(out_stream)
        out_stream.seek(0)
        updated_bytes = out_stream.getvalue()
        # Determine output extension
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
        )
    # Display a message on main page
    st.write("Click the button in the sidebar to generate your updated presentation.")
