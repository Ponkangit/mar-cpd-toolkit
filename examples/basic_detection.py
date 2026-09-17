"""Reproducible single-change MAR(1) detection example.

Run from the repository root with:

    python examples/basic_detection.py
"""

import numpy as np

from marcp import detect_change, simulate_mar1


def main() -> None:
    n = 240
    true_change = 120

    # Both regimes satisfy rho(B \u2297 A) < 1.  The post-change regime is
    # deliberately more persistent so that its second-moment change is clear
    # in a small, reproducible demonstration.
    a1 = np.eye(2) * 0.20
    b1 = np.eye(2) * 0.20
    a2 = np.array([[0.90, 0.00], [0.00, 0.85]])
    b2 = np.array([[0.90, 0.00], [0.00, 0.85]])

    x = simulate_mar1(
        n,
        a1,
        b1,
        change_index=true_change,
        a2=a2,
        b2=b2,
        seed=7,
        burnin=100,
    )
    result = detect_change(x, min_segment=30)

    print(f"true_change={true_change}")
    print(f"estimated_change={result.change_index}")
    print(f"absolute_error={abs(result.change_index - true_change)}")
    print(f"max_score={result.score:.6f}")


if __name__ == "__main__":
    main()
