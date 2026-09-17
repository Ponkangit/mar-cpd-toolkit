from __future__ import annotations

import argparse

import numpy as np

from .core import detect_change, simulate_mar1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small MAR(1) change-point demonstration.")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--change", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-segment", type=int, default=20)
    args = parser.parse_args()

    a1 = np.array([[0.45, 0.05], [0.00, 0.35]])
    b1 = np.array([[0.50, 0.00], [0.05, 0.40]])
    a2 = np.array([[0.15, 0.00], [0.05, 0.55]])
    b2 = np.array([[0.35, 0.10], [0.00, 0.55]])

    x = simulate_mar1(
        args.n,
        a1,
        b1,
        change_index=args.change,
        a2=a2,
        b2=b2,
        seed=args.seed,
    )
    result = detect_change(x, min_segment=args.min_segment)
    print(f"estimated_change={result.change_index}")
    print(f"max_score={result.score:.6f}")


if __name__ == "__main__":
    main()
