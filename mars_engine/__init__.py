"""
Lightweight MARS engine package (ic_technical.mars_engine).

Public API:
    - generate_spx_score_history(prices_df: pd.DataFrame, agg_method, weights) -> pd.Series
    - generate_csi_score_history(prices_df: pd.DataFrame, agg_method, weights) -> pd.Series
    - load_prices_for_mars(excel_obj_or_path) -> pd.DataFrame
    - DEFAULT_WEIGHTS: dict[str, float]
    - PEER_GROUP_SPX: list[str]
    - PEER_GROUP_CSI: list[str]

    LASSO weighting (advanced):
    - perform_walk_forward_validation(...) -> pd.DataFrame
    - train_lasso_model(training_df) -> (weights_dict, optimal_alpha)
"""
from .mars_lite_scorer import (
    generate_spx_score_history,
    generate_csi_score_history,
    PEER_GROUP_SPX,
    PEER_GROUP_CSI,
    DEFAULT_WEIGHTS,
)
from .data_loader import load_prices_for_mars
from .lasso_weighting import (
    perform_walk_forward_validation,
    train_lasso_model,
    prepare_training_data,
)

__all__ = [
    "generate_spx_score_history",
    "generate_csi_score_history",
    "load_prices_for_mars",
    "DEFAULT_WEIGHTS",
    "PEER_GROUP_SPX",
    "PEER_GROUP_CSI",
    "perform_walk_forward_validation",
    "train_lasso_model",
    "prepare_training_data",
]
