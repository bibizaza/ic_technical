"""
Lightweight MARS engine package (ic_technical.mars_engine).

Public API:
    - generate_spx_score_history(prices_df: pd.DataFrame) -> pd.Series
    - generate_csi_score_history(prices_df: pd.DataFrame) -> pd.Series
    - load_prices_for_mars(excel_obj_or_path) -> pd.DataFrame
    - PEER_GROUP_SPX: list[str]
    - PEER_GROUP_CSI: list[str]
"""
from .mars_lite_scorer import (
    generate_spx_score_history,
    generate_csi_score_history,
    PEER_GROUP_SPX,
    PEER_GROUP_CSI,
)
from .data_loader import load_prices_for_mars

__all__ = [
    "generate_spx_score_history",
    "generate_csi_score_history",
    "load_prices_for_mars",
    "PEER_GROUP_SPX",
    "PEER_GROUP_CSI",
]
