"""Evaluation package for recommendation systems."""

from .metrics import (
    coverage,
    diversity,
    evaluate_model,
    hit_rate_at_k,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "map_at_k",
    "hit_rate_at_k",
    "coverage",
    "diversity",
    "evaluate_model",
]
