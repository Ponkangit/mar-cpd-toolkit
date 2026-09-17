"""MAR change-point detection baselines."""

from .core import DetectionResult, cusum_scores, detect_change, row_col_features, simulate_mar1

__all__ = [
    "DetectionResult",
    "cusum_scores",
    "detect_change",
    "row_col_features",
    "simulate_mar1",
]

__version__ = "0.1.0"
