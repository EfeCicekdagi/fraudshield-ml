# CI Pipeline

The CI pipeline runs via GitHub Actions (`.github/workflows/ci.yml`).

## Workflow Steps
1. **Linting and Setup**: Provisions a Python 3.10 environment.
2. **Synthetic Bundle**: Runs `scripts/generate_ci_bundle.py` to create a lightweight mock model bundle for testing.
3. **Tests**: Executes `pytest tests/` to validate FastAPI endpoints and application logic.
