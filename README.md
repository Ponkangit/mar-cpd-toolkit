# mar-cpd-toolkit

`mar-cpd-toolkit` is a small open-source Python toolkit for reproducible experiments with single change-point detection in matrix autoregressive time series.

The initial release intentionally stays narrow. It provides:

- simulation of stable Gaussian MAR(1) processes, including one coefficient change;
- row/column second-moment features;
- a transparent two-sample CUSUM scan;
- a small command-line demonstration;
- unit tests and GitHub Actions CI.

This repository is a standalone software project. It does **not** contain unpublished manuscripts, formal Monte Carlo outputs, private datasets, or internal research audit material from any separate research workspace.

## Model

The simulator uses

\[
X_t = A X_{t-1} B^\top + E_t,
\]

with Gaussian entrywise innovations. Stability is checked through the spectral radius of `kron(B, A)`.

For each observed matrix `X_t`, the baseline detector constructs

\[
\operatorname{vec}(X_tX_t^\top/q) \oplus \operatorname{vec}(X_t^\top X_t/p),
\]

then scans admissible split points with the standard two-sample Euclidean CUSUM score

\[
\sqrt{\frac{k(n-k)}{n}}\,\|\bar Z_{1:k}-\bar Z_{k+1:n}\|_2.
\]

The implementation is deliberately explicit rather than optimized so that statistical assumptions and indexing are easy to audit.

## Install

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Quick start

```python
import numpy as np
from marcp import detect_change, simulate_mar1

A1 = np.array([[0.45, 0.05], [0.00, 0.35]])
B1 = np.array([[0.50, 0.00], [0.05, 0.40]])
A2 = np.array([[0.15, 0.00], [0.05, 0.55]])
B2 = np.array([[0.35, 0.10], [0.00, 0.55]])

x = simulate_mar1(
    200,
    A1,
    B1,
    change_index=100,
    a2=A2,
    b2=B2,
    seed=7,
)

result = detect_change(x, min_segment=20)
print(result.change_index, result.score)
```

Or run:

```bash
marcp-demo --n 200 --change 100 --seed 7
```

## Scope and roadmap

Version 0.1.0 is a minimal baseline intended to make matrix-valued change-point examples easy to reproduce and inspect. Natural next contributions include bootstrap calibration, multiple-change methods, alternative matrix features, benchmark datasets, performance profiling, and comparison adapters for other detectors.

Issues and pull requests are welcome. Please keep methodological claims tied to tests, simulations, or cited theory.

## Reproducibility policy

- Randomized examples expose explicit seeds.
- Unit tests avoid long Monte Carlo jobs.
- Changes to statistical definitions should include a regression test.
- New benchmark claims should record configuration and software versions.

## License

MIT. See `LICENSE`.
