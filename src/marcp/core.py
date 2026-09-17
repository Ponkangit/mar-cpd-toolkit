from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    """Result of a row/column second-moment CUSUM scan."""

    change_index: int
    score: float
    scores: np.ndarray


def _as_square(name: str, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return x


def _stability_radius(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(np.kron(b, a)))))


def simulate_mar1(
    n: int,
    a1: np.ndarray,
    b1: np.ndarray,
    *,
    change_index: int | None = None,
    a2: np.ndarray | None = None,
    b2: np.ndarray | None = None,
    noise_scale: float = 1.0,
    burnin: int = 100,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate a Gaussian MAR(1) series.

    The recursion is X_t = A X_{t-1} B^T + E_t.  If ``change_index`` is
    supplied, (A2, B2) is used from that returned observation onward.
    Innovations have independent N(0, noise_scale^2) entries.
    """

    if n < 3:
        raise ValueError("n must be at least 3")
    if burnin < 0:
        raise ValueError("burnin must be non-negative")
    if noise_scale <= 0:
        raise ValueError("noise_scale must be positive")

    a1 = _as_square("a1", a1)
    b1 = _as_square("b1", b1)
    if _stability_radius(a1, b1) >= 1.0:
        raise ValueError("(a1, b1) must define a stable MAR(1) process")

    p, q = a1.shape[0], b1.shape[0]
    has_change = change_index is not None
    if has_change:
        if not 1 <= int(change_index) <= n - 1:
            raise ValueError("change_index must lie in {1, ..., n-1}")
        if a2 is None or b2 is None:
            raise ValueError("a2 and b2 are required when change_index is set")
        a2 = _as_square("a2", a2)
        b2 = _as_square("b2", b2)
        if a2.shape != a1.shape or b2.shape != b1.shape:
            raise ValueError("pre- and post-change coefficient dimensions must match")
        if _stability_radius(a2, b2) >= 1.0:
            raise ValueError("(a2, b2) must define a stable MAR(1) process")

    rng = np.random.default_rng(seed)
    x = np.zeros((p, q), dtype=float)

    # Burn in under the first regime, then generate exactly n returned states.
    for _ in range(burnin):
        x = a1 @ x @ b1.T + rng.normal(scale=noise_scale, size=(p, q))

    out = np.empty((n, p, q), dtype=float)
    for t in range(n):
        if has_change and t >= int(change_index):
            aa, bb = a2, b2
        else:
            aa, bb = a1, b1
        x = aa @ x @ bb.T + rng.normal(scale=noise_scale, size=(p, q))
        out[t] = x
    return out


def row_col_features(x: np.ndarray) -> np.ndarray:
    """Return concatenated row/column second-moment features.

    For X_t in R^{p x q}, the feature is vec(X_t X_t^T / q) concatenated
    with vec(X_t^T X_t / p).  The full matrices are retained instead of only
    their upper triangles to keep the implementation transparent.
    """

    x = np.asarray(x, dtype=float)
    if x.ndim != 3:
        raise ValueError("x must have shape (n, p, q)")
    n, p, q = x.shape
    if n < 2 or p < 1 or q < 1:
        raise ValueError("x has invalid dimensions")

    row = np.einsum("tpq,trq->tpr", x, x) / q
    col = np.einsum("tpq,tpr->tqr", x, x) / p
    return np.concatenate((row.reshape(n, -1), col.reshape(n, -1)), axis=1)


def cusum_scores(features: np.ndarray, min_segment: int = 1) -> np.ndarray:
    """Compute Euclidean two-sample CUSUM scores for every admissible split.

    For split k, score(k) = sqrt(k(n-k)/n) * ||mean_left - mean_right||_2.
    Non-admissible positions are filled with NaN.  Index k means the second
    segment starts at observation k.
    """

    z = np.asarray(features, dtype=float)
    if z.ndim != 2:
        raise ValueError("features must have shape (n, d)")
    n = z.shape[0]
    if n < 3:
        raise ValueError("at least three observations are required")
    if min_segment < 1 or 2 * min_segment > n:
        raise ValueError("min_segment is incompatible with sample size")

    csum = np.vstack((np.zeros((1, z.shape[1])), np.cumsum(z, axis=0)))
    total = csum[-1]
    scores = np.full(n + 1, np.nan, dtype=float)

    for k in range(min_segment, n - min_segment + 1):
        left = csum[k] / k
        right = (total - csum[k]) / (n - k)
        scale = np.sqrt(k * (n - k) / n)
        scores[k] = scale * np.linalg.norm(left - right)
    return scores


def detect_change(x: np.ndarray, min_segment: int = 5) -> DetectionResult:
    """Estimate a single change point using row/column second moments."""

    features = row_col_features(x)
    scores = cusum_scores(features, min_segment=min_segment)
    admissible = np.flatnonzero(np.isfinite(scores))
    if admissible.size == 0:
        raise ValueError("no admissible split points")
    best = int(admissible[np.argmax(scores[admissible])])
    return DetectionResult(change_index=best, score=float(scores[best]), scores=scores)
