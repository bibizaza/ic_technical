"""Global Indices Insight - Combined Breadth & Fundamental tables."""

from .slide_generator import (
    generate_global_indices_slide,
    prepare_breadth_data,
    compute_fundamental_ranks,
    BreadthRow,
    FundamentalRow,
)

__all__ = [
    "generate_global_indices_slide",
    "prepare_breadth_data",
    "compute_fundamental_ranks",
    "BreadthRow",
    "FundamentalRow",
]
