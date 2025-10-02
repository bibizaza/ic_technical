"""
Lightweight MARS engine package (ic_technical.mars_engine).

Public API:
    - generate_spx_score_history(prices_df: pd.DataFrame) -> pd.Series
    - PEER_GROUP: list[str]
"""
from .mars_lite_scorer import generate_spx_score_history, PEER_GROUP

__all__ = ["generate_spx_score_history", "PEER_GROUP"]
