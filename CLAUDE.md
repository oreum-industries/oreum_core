# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

`oreum_core` is a Python 3.13 package of core data science tools for Oreum Industries projects. It provides workflows
for data curation, EDA, Bayesian modeling (via PyMC), and gradient-boosted tree modeling (via XGBoost/CatBoost).
Built and published to PyPI using `flit`, managed with `uv`.

## Development Environment

+ **Machine**: Developed on a Macbook Air M2, with 24GB RAM
+ **Python Env**: Uses `uv` for virtual environment management with a local `.venv/`

```zsh
# Create dev environment (also installs pre-commit hooks)
make dev

# Lint (ruff, interrogate, bandit)
make lint

# Unit tests
make test

# Remove dev environment
make dev-uninstall
```

## Linting & Code Quality

Pre-commit runs automatically on `git commit`. Direct commands (also what CI runs):

```zsh
# Lint and format check
ruff check --config pyproject.toml oreum_core/
ruff format --config pyproject.toml --diff --no-cache oreum_core/

# Docstring coverage (must be ≥80%)
interrogate --config pyproject.toml oreum_core/

# Security scan
bandit --config pyproject.toml -r oreum_core/ -f json -o reports/bandit-report.json
```

Key linting rules: `no-commit-to-branch` prevents direct commits to `master`. `no-print-statements` is enforced in
`oreum_core/` — use the logger instead. Every source file must have an Apache 2.0 license header.

## Architecture

### Package Structure

```
oreum_core/
├── curate/         # Data ingestion & transformation
│   ├── data_io.py       # File I/O: CSV, Excel, Parquet (Pandas+Dask), Pickle
│   ├── data_transform.py # DatatypeConverter, DatasetReshaper, Transformer, Standardizer
│   └── text_clean.py    # Text cleaning utilities
├── eda/            # Exploratory data analysis
│   ├── calc.py     # Statistical calculations (SVD, etc.)
│   ├── describe.py # Feature-type introspection
│   ├── eda_io.py   # EDA file I/O
│   └── plot.py     # Matplotlib/Seaborn plot functions
├── model_pymc/     # Bayesian modeling (optional dep: pip install oreum_core[pymc])
│   ├── base.py     # BasePYMCModel — the central base class
│   ├── calc.py     # PyMC calculations
│   ├── describe.py # Model description utilities
│   ├── distributions.py # Custom PyMC distributions
│   ├── plot.py     # ArviZ/PyMC plot helpers
│   └── pymc_io.py  # PYMCIO: NetCDF read/write, model graph export
├── model_tree/     # Gradient-boosted trees (optional dep: pip install oreum_core[tree])
│   └── tree_io.py  # XGBIO
└── utils/
    ├── file_io.py        # BaseFileIO base class for all I/O handlers
    └── snakey_lowercaser.py # String sanitization
```

### Key Design Patterns

**Inheritance chain for I/O**: `BaseFileIO` (utils) → `DaskParquetIO`, `PandasCSVIO`, `PYMCIO`, etc. All I/O classes
accept an optional `rootdir: Path` for notebook-friendly relative paths.

**`BasePYMCModel`** (`model_pymc/base.py`): Central class for all Bayesian models. Subclasses must:
+ Define `name`, `version`, `obs_nm` attributes
+ Implement `_build(**kwargs)` containing the PyMC model definition
+ Optionally implement `_extend_build()` for out-of-sample PPC extensions
+ Use `pm.Data` containers for observations to enable `replace_obs()`

Typical PyMC model lifecycle:
```python
mdl.build()
mdl.sample_prior_predictive()
mdl.sample()                    # NUTS sampler, default 2000 tune + 500 draws × 4 chains
mdl.sample_posterior_predictive()
```

Model persistence via `PYMCIO`: saves/loads ArviZ `InferenceData` as NetCDF (`.nc` files).

**Logging**: Uses named logger `'oreum_core'` with `NullHandler` by default. All submodules use
`logging.getLogger(__name__)`. Never use `print()` in `oreum_core/`.

### Optional Dependency Groups

+ `[pymc]`: adds `pymc`, `nutpie`, `graphviz` — required for `model_pymc/`
+ `[tree]`: adds `catboost`, `xgboost`, `optuna`, `shap`, `category_encoders` — required for `model_tree/`
+ `[dev]`: dev tooling (`ruff`, `pytest`, `bandit`, `interrogate`, `pre-commit`, etc.)

### Build & Publish

Uses `flit` for building/publishing (not setuptools). Package version is set in `pyproject.toml` directly. CI publishes
to PyPI via GitHub Actions on release.

```zsh
make build          # build dist/
make publish        # build + publish to PyPI (requires .env with credentials)
make publish-test   # build + publish to TestPyPI
```

---

## Coding Rules

### Python (all `.py` files)

+ **Python 3.13**: use latest idiomatic python, managed via `uv` with a virtual env at `.venv/`
+ **Keep it simple and DRY**: modular, efficient, avoid unnecessary complexity
+ **Exceptions**: use `try/except` with logging; never leave empty `pass` blocks
+ **Type hints**: use throughout for all function signatures; use the most appropriate types
+ **Naming**: use PascalCase for classes; snake_case for functions and variables
+ **Comprehensions**: prefer list/dict comprehensions over loops where appropriate
+ **Vectorization**: prefer numpy / xarray / numba vectorisation over loops when dealing with array data
+ **Global variables**: avoid these to prevent unwanted side effects
+ **Mathematics**: double-check all mathematical operations — e.g. `0.83^7 * 4 = 1.085` not `1.185`
+ **Linting and formatting**: use PEP8,and also follow `ruff` rules in `pyproject.toml` at `[tool.ruff]`
+ **Docstrings**: required on all public functions and classes (enforced by `interrogate` ≥80%); use inline comments
  sparingly, only for non-obvious logic
+ **Markdown**: word-wrap all markdown at 100 chars, avoid em-dash, use `+` for bullets

### Tests (all `tests/` files)

+ Use `pytest` with fixtures and `pytest-mock`; never `unittest`
+ Parameterize using `pytest-csv-params`; store CSV inputs under `tests/data/` named after the test case
+ Be extremely careful with expected values: a function taking 2 inputs and returning 1 value contributes only 1 value
  to the expected result, not 2
+ Test happy path, sad path, boundary and edge conditions
+ Test input validity according to Pydantic models and datatypes
+ After writing tests, tidy unused imports; then run with `uv run pytest` and fix using actual results, not manual
  calculations
+ For API endpoint tests, respect rate limits on inbound calls
