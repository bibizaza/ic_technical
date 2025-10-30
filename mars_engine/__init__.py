"""
Lightweight MARS engine package (ic_technical.mars_engine).

Public API:
    - generate_spx_score_history(prices_df: pd.DataFrame, agg_method, weights) -> pd.Series
    - generate_csi_score_history(prices_df: pd.DataFrame, agg_method, weights) -> pd.Series
    - get_csi_lasso_score(excel_path_or_df) -> float  # CSI with dynamic LASSO
    - load_prices_for_mars(excel_obj_or_path) -> pd.DataFrame
    - load_mars_scores(excel_obj_or_path) -> dict[str, float]  # Pre-computed scores
    - DEFAULT_WEIGHTS: dict[str, float]
    - PEER_GROUP_SPX: list[str]
    - PEER_GROUP_CSI: list[str]

    LASSO weighting (advanced):
    - perform_walk_forward_validation(...) -> pd.DataFrame
    - train_lasso_model(training_df) -> (weights_dict, optimal_alpha)

    Raw components (for advanced usage):
    - compute_raw_components(df, target_col, hi_col, lo_col, bench_col) -> pd.DataFrame
"""
from .mars_lite_scorer import (
    generate_spx_score_history,
    generate_csi_score_history,
    PEER_GROUP_SPX,
    PEER_GROUP_CSI,
    DEFAULT_WEIGHTS,
)
from .data_loader import load_prices_for_mars, load_mars_scores
from .lasso_weighting import (
    perform_walk_forward_validation,
    train_lasso_model,
    prepare_training_data,
)
from .csi_lasso_scorer import get_csi_lasso_score
from .raw_components import compute_raw_components

__all__ = [
    "generate_spx_score_history",
    "generate_csi_score_history",
    "get_csi_lasso_score",
    "load_prices_for_mars",
    "load_mars_scores",
    "DEFAULT_WEIGHTS",
    "PEER_GROUP_SPX",
    "PEER_GROUP_CSI",
    "perform_walk_forward_validation",
    "train_lasso_model",
    "prepare_training_data",
    "compute_raw_components",
]
