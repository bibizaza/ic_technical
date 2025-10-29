"""
Instrument factory for creating configured instruments.

This module provides a factory pattern that eliminates the need for 37
try/except import blocks in app.py. Instead, instruments are registered
with their metadata and MARS scoring functions, and can be retrieved
dynamically.

Usage example:
    factory = InstrumentFactory()
    spx = factory.get_instrument('spx')
    fig = spx.make_figure(excel_path, anchor_date, price_mode)
"""

from __future__ import annotations
from typing import Dict, Optional
from technical_analysis.base_instrument import BaseInstrument, InstrumentConfig


# ============================================================================
# MARS Engine Imports
# ============================================================================

def _import_mars_scorers():
    """
    Import all MARS scoring functions.

    Returns a dictionary mapping instrument names to their scoring functions.
    If imports fail, returns an empty dict (graceful degradation).
    """
    scorers = {}

    try:
        from mars_engine.mars_lite_scorer import (
            generate_spx_score_history,
            generate_csi_score_history,
            generate_dax_score_history,
            generate_ibov_score_history,
            generate_mexbol_score_history,
            generate_nikkei_score_history,
            generate_sensex_score_history,
            generate_smi_score_history,
            generate_tasi_score_history,
            generate_gold_score_history,
            generate_silver_score_history,
            generate_copper_score_history,
            generate_oil_score_history,
            generate_palladium_score_history,
            generate_platinum_score_history,
            generate_bitcoin_score_history,
            generate_ethereum_score_history,
            generate_binance_score_history,
            generate_solana_score_history,
            generate_ripple_score_history,
        )

        scorers = {
            'spx': generate_spx_score_history,
            'csi': generate_csi_score_history,
            'dax': generate_dax_score_history,
            'ibov': generate_ibov_score_history,
            'mexbol': generate_mexbol_score_history,
            'nikkei': generate_nikkei_score_history,
            'sensex': generate_sensex_score_history,
            'smi': generate_smi_score_history,
            'tasi': generate_tasi_score_history,
            'gold': generate_gold_score_history,
            'silver': generate_silver_score_history,
            'copper': generate_copper_score_history,
            'oil': generate_oil_score_history,
            'palladium': generate_palladium_score_history,
            'platinum': generate_platinum_score_history,
            'bitcoin': generate_bitcoin_score_history,
            'ethereum': generate_ethereum_score_history,
            'binance': generate_binance_score_history,
            'solana': generate_solana_score_history,
            'ripple': generate_ripple_score_history,
        }
    except Exception:
        # If MARS engine is unavailable, return empty dict
        # Instruments will still work but momentum scores will be N/A
        pass

    return scorers


# ============================================================================
# Peer Groups
# ============================================================================

EQUITY_PEER_GROUP = [
    "CCMP Index",
    "IBOV Index",
    "MEXBOL Index",
    "SXXP Index",
    "UKX Index",
    "SMI Index",
    "HSI Index",
    "SHSZ300 Index",
    "NKY Index",
    "SENSEX Index",
    "DAX Index",
    "MXWO Index",
    "USGG10YR Index",
    "GECU10YR Index",
    "CL1 Comdty",
    "GCA Comdty",
    "DXY Curncy",
    "XBTUSD Curncy",
]

COMMODITY_PEER_GROUP = [
    "GCA Comdty",
    "SI1 Comdty",
    "HG1 Comdty",
    "CL1 Comdty",
    "PA1 Comdty",
    "PL1 Comdty",
    "SPX Index",
    "USGG10YR Index",
]

CRYPTO_PEER_GROUP = [
    "XBTUSD Curncy",
    "XETUSD Curncy",
    "XBNCUR Curncy",
    "SOLUSD Curncy",
    "XRPUSD Curncy",
    "SPX Index",
    "GCA Comdty",
]


# ============================================================================
# Instrument Registry
# ============================================================================

class InstrumentFactory:
    """
    Factory for creating and managing instruments.

    This replaces the need for 37 try/except import blocks in app.py.
    """

    def __init__(self):
        """Initialize the factory and register all instruments."""
        self._instruments: Dict[str, BaseInstrument] = {}
        self._mars_scorers = _import_mars_scorers()
        self._register_all_instruments()

    def _register_all_instruments(self):
        """Register all instruments with their configurations."""

        # ==================================================================
        # EQUITY INDICES
        # ==================================================================

        equity_configs = [
            InstrumentConfig(
                name='spx',
                display_name='S&P 500',
                ticker='SPX Index',
                vol_ticker='VIX Index',
                peer_group=EQUITY_PEER_GROUP,
                mars_scorer_func=self._mars_scorers.get('spx'),
            ),
            InstrumentConfig(
                name='csi',
                display_name='CSI 300',
                ticker='SHSZ300 Index',
                vol_ticker='VXFXI Index',
                peer_group=EQUITY_PEER_GROUP,
                mars_scorer_func=self._mars_scorers.get('csi'),
            ),
            InstrumentConfig(
                name='dax',
                display_name='DAX',
                ticker='DAX Index',
                vol_ticker='V2X Index',
                peer_group=EQUITY_PEER_GROUP,
                mars_scorer_func=self._mars_scorers.get('dax'),
            ),
            InstrumentConfig(
                name='ibov',
                display_name='Bovespa',
                ticker='IBOV Index',
                vol_ticker='VXEWZ Index',
                peer_group=EQUITY_PEER_GROUP,
                mars_scorer_func=self._mars_scorers.get('ibov'),
            ),
            InstrumentConfig(
                name='mexbol',
                display_name='Mexbol',
                ticker='MEXBOL Index',
                vol_ticker='VIX Index',  # Fallback to VIX
                peer_group=EQUITY_PEER_GROUP,
                mars_scorer_func=self._mars_scorers.get('mexbol'),
            ),
            InstrumentConfig(
                name='nikkei',
                display_name='Nikkei',
                ticker='NKY Index',
                vol_ticker='VNKY Index',
                peer_group=EQUITY_PEER_GROUP,
                mars_scorer_func=self._mars_scorers.get('nikkei'),
            ),
            InstrumentConfig(
                name='sensex',
                display_name='Sensex',
                ticker='SENSEX Index',
                vol_ticker='IVXSENSEX Index',
                peer_group=EQUITY_PEER_GROUP,
                mars_scorer_func=self._mars_scorers.get('sensex'),
            ),
            InstrumentConfig(
                name='smi',
                display_name='SMI',
                ticker='SMI Index',
                vol_ticker='V2X Index',  # Fallback
                peer_group=EQUITY_PEER_GROUP,
                mars_scorer_func=self._mars_scorers.get('smi'),
            ),
            InstrumentConfig(
                name='tasi',
                display_name='TASI',
                ticker='SASEIDX Index',
                vol_ticker='VIX Index',  # Fallback
                peer_group=EQUITY_PEER_GROUP,
                mars_scorer_func=self._mars_scorers.get('tasi'),
            ),
        ]

        # ==================================================================
        # COMMODITIES
        # ==================================================================

        commodity_configs = [
            InstrumentConfig(
                name='gold',
                display_name='Gold',
                ticker='GCA Comdty',
                vol_ticker='GVZ Index',
                peer_group=COMMODITY_PEER_GROUP,
                mars_scorer_func=self._mars_scorers.get('gold'),
            ),
            InstrumentConfig(
                name='silver',
                display_name='Silver',
                ticker='SI1 Comdty',
                vol_ticker='VXSLV Index',
                peer_group=COMMODITY_PEER_GROUP,
                mars_scorer_func=self._mars_scorers.get('silver'),
            ),
            InstrumentConfig(
                name='copper',
                display_name='Copper',
                ticker='HG1 Comdty',
                vol_ticker='VIX Index',  # Fallback
                peer_group=COMMODITY_PEER_GROUP,
                mars_scorer_func=self._mars_scorers.get('copper'),
            ),
            InstrumentConfig(
                name='oil',
                display_name='WTI Crude Oil',
                ticker='CL1 Comdty',
                vol_ticker='OVX Index',
                peer_group=COMMODITY_PEER_GROUP,
                mars_scorer_func=self._mars_scorers.get('oil'),
            ),
            InstrumentConfig(
                name='palladium',
                display_name='Palladium',
                ticker='PA1 Comdty',
                vol_ticker='VIX Index',  # Fallback
                peer_group=COMMODITY_PEER_GROUP,
                mars_scorer_func=self._mars_scorers.get('palladium'),
            ),
            InstrumentConfig(
                name='platinum',
                display_name='Platinum',
                ticker='PL1 Comdty',
                vol_ticker='VIX Index',  # Fallback
                peer_group=COMMODITY_PEER_GROUP,
                mars_scorer_func=self._mars_scorers.get('platinum'),
            ),
        ]

        # ==================================================================
        # CRYPTOCURRENCIES
        # ==================================================================

        crypto_configs = [
            InstrumentConfig(
                name='bitcoin',
                display_name='Bitcoin',
                ticker='XBTUSD Curncy',
                vol_ticker='DVOL Index',  # Deribit BTC volatility
                peer_group=CRYPTO_PEER_GROUP,
                mars_scorer_func=self._mars_scorers.get('bitcoin'),
            ),
            InstrumentConfig(
                name='ethereum',
                display_name='Ethereum',
                ticker='XETUSD Curncy',
                vol_ticker='DVOL Index',  # Fallback
                peer_group=CRYPTO_PEER_GROUP,
                mars_scorer_func=self._mars_scorers.get('ethereum'),
            ),
            InstrumentConfig(
                name='binance',
                display_name='BNB',
                ticker='XBNCUR Curncy',
                vol_ticker='DVOL Index',  # Fallback
                peer_group=CRYPTO_PEER_GROUP,
                mars_scorer_func=self._mars_scorers.get('binance'),
            ),
            InstrumentConfig(
                name='solana',
                display_name='Solana',
                ticker='SOLUSD Curncy',
                vol_ticker='DVOL Index',  # Fallback
                peer_group=CRYPTO_PEER_GROUP,
                mars_scorer_func=self._mars_scorers.get('solana'),
            ),
            InstrumentConfig(
                name='ripple',
                display_name='XRP',
                ticker='XRPUSD Curncy',
                vol_ticker='DVOL Index',  # Fallback
                peer_group=CRYPTO_PEER_GROUP,
                mars_scorer_func=self._mars_scorers.get('ripple'),
            ),
        ]

        # Register all instruments
        for config in equity_configs + commodity_configs + crypto_configs:
            self._instruments[config.name] = BaseInstrument(config)

    def get_instrument(self, name: str) -> Optional[BaseInstrument]:
        """
        Get an instrument by name.

        Parameters
        ----------
        name : str
            Instrument name (e.g., 'spx', 'gold', 'bitcoin')

        Returns
        -------
        BaseInstrument or None
            The instrument instance or None if not found.
        """
        return self._instruments.get(name.lower())

    def get_all_instruments(self) -> Dict[str, BaseInstrument]:
        """
        Get all registered instruments.

        Returns
        -------
        dict
            Dictionary mapping instrument names to instances.
        """
        return self._instruments.copy()

    def get_equity_instruments(self) -> Dict[str, BaseInstrument]:
        """Get all equity instruments."""
        equity_names = ['spx', 'csi', 'dax', 'ibov', 'mexbol', 'nikkei', 'sensex', 'smi', 'tasi']
        return {name: inst for name, inst in self._instruments.items() if name in equity_names}

    def get_commodity_instruments(self) -> Dict[str, BaseInstrument]:
        """Get all commodity instruments."""
        commodity_names = ['gold', 'silver', 'copper', 'oil', 'palladium', 'platinum']
        return {name: inst for name, inst in self._instruments.items() if name in commodity_names}

    def get_crypto_instruments(self) -> Dict[str, BaseInstrument]:
        """Get all crypto instruments."""
        crypto_names = ['bitcoin', 'ethereum', 'binance', 'solana', 'ripple']
        return {name: inst for name, inst in self._instruments.items() if name in crypto_names}

    def set_lookback_days(self, days: int):
        """
        Set the plot lookback days for all instruments.

        Parameters
        ----------
        days : int
            Number of days to display in charts.
        """
        import technical_analysis.base_instrument as base_module
        base_module.PLOT_LOOKBACK_DAYS = days

    def get_lookback_days(self) -> int:
        """Get the current plot lookback days."""
        import technical_analysis.base_instrument as base_module
        return base_module.PLOT_LOOKBACK_DAYS


# ============================================================================
# Convenience Functions for Backward Compatibility
# ============================================================================

def create_instrument_functions(instrument_name: str):
    """
    Create backward-compatible function wrappers for an instrument.

    This generates functions like make_spx_figure, insert_spx_technical_score_number,
    etc. to maintain compatibility with existing code.

    Parameters
    ----------
    instrument_name : str
        Name of the instrument (e.g., 'spx', 'gold')

    Returns
    -------
    dict
        Dictionary of function names to function objects.
    """
    factory = InstrumentFactory()
    instrument = factory.get_instrument(instrument_name)

    if instrument is None:
        # Return no-op functions if instrument not found
        return {
            f'make_{instrument_name}_figure': lambda *args, **kwargs: None,
            f'insert_{instrument_name}_technical_chart': lambda prs, *args, **kwargs: prs,
            f'insert_{instrument_name}_technical_score_number': lambda prs, *args, **kwargs: prs,
            f'insert_{instrument_name}_momentum_score_number': lambda prs, *args, **kwargs: prs,
            f'insert_{instrument_name}_subtitle': lambda prs, *args, **kwargs: prs,
            f'_get_{instrument_name}_technical_score': lambda *args, **kwargs: None,
            f'_get_{instrument_name}_momentum_score': lambda *args, **kwargs: None,
        }

    # Create wrapper functions
    return {
        f'make_{instrument_name}_figure': instrument.make_figure,
        f'insert_{instrument_name}_technical_chart': instrument.insert_technical_chart,
        f'insert_{instrument_name}_technical_score_number': instrument.insert_technical_score_number,
        f'insert_{instrument_name}_momentum_score_number': instrument.insert_momentum_score_number,
        f'insert_{instrument_name}_subtitle': instrument.insert_subtitle,
        f'_get_{instrument_name}_technical_score': instrument._get_technical_score,
        f'_get_{instrument_name}_momentum_score': instrument._get_momentum_score,
    }
