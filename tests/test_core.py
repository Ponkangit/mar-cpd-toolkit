import numpy as np
import pytest

from marcp import cusum_scores, detect_change, row_col_features, simulate_mar1


def test_row_col_feature_shape() -> None:
    x = np.arange(5 * 2 * 3, dtype=float).reshape(5, 2, 3)
    features = row_col_features(x)
    assert features.shape == (5, 2 * 2 + 3 * 3)


def test_cusum_detects_deterministic_level_change() -> None:
    x = np.zeros((20, 2, 2), dtype=float)
    x[10:] = np.eye(2) * 4.0
    result = detect_change(x, min_segment=3)
    assert result.change_index == 10
    assert result.score > 0


def test_simulator_reproducible() -> None:
    a = np.eye(2) * 0.4
    b = np.eye(2) * 0.5
    x1 = simulate_mar1(12, a, b, seed=123, burnin=5)
    x2 = simulate_mar1(12, a, b, seed=123, burnin=5)
    np.testing.assert_allclose(x1, x2)


def test_unstable_model_rejected() -> None:
    a = np.eye(2) * 1.1
    b = np.eye(2)
    with pytest.raises(ValueError, match="stable"):
        simulate_mar1(10, a, b, seed=1)


def test_cusum_rejects_bad_min_segment() -> None:
    z = np.ones((6, 2))
    with pytest.raises(ValueError, match="min_segment"):
        cusum_scores(z, min_segment=4)
