# Contributing

Contributions are welcome through issues and pull requests.

Please keep changes focused and reproducible:

1. Explain the statistical or software motivation.
2. Add or update tests for behavior changes.
3. Avoid long-running Monte Carlo jobs in the default test suite.
4. Record random seeds in examples and benchmark scripts.
5. Do not add private datasets, unpublished manuscripts, credentials, or generated secrets.

Development setup:

```bash
python -m pip install -e ".[dev]"
pytest
```

For methodological changes, describe the exact statistic, assumptions, and expected behavior in the pull request.
