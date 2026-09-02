# Configuration

Runtime configuration is intentionally environment-driven through `.env.example` and `src/config.py` so the same code works from Windows PowerShell, tests, Docker, and CI without hard-coded machine paths.

The main configuration groups are:

- data paths and sampling
- chronological split fractions
- model iteration/estimator budgets
- business-cost assumptions
- API/dashboard settings
- MLflow settings
- optional PostgreSQL settings
- monitoring thresholds

Copy `.env.example` to `.env` for local execution. `.env` is gitignored.
