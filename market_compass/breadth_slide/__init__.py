"""Composite Breadth Score slide generator module."""

from .slide_generator import (
    generate_composite_breadth_slide,
    # Legacy exports (backwards compat)
    BreadthRow,
    prepare_breadth_data,
    insert_breadth_rank,
    generate_breadth_slide,
)

__all__ = [
    "generate_composite_breadth_slide",
    "BreadthRow",
    "prepare_breadth_data",
    "insert_breadth_rank",
    "generate_breadth_slide",
]
