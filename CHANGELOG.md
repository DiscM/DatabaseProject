# Changelog

## [Unreleased]

### Added
- **`src/features.py`** — shared feature constants (`FEATURE_COLUMNS`, `COMPUTED_FEATURE_NAMES`) and helpers (`to_float`, `to_float_strict`, `compute_derived`). Both `train_oil_model.py` and `predict_oil_price.py` now import from this single source, eliminating duplicated code.
- **`src/pipeline.py`** — shared pipeline orchestration (`run_train`, `run_predict`, `run_visualize`). Both entry points (`main.py`, `main_with_db.py`) import from this module instead of from each other.
- **Test suite** — 52 pytest tests covering features, training, forecast helpers, and MySQL loader utilities.
- **`ruff.toml`** — linter configuration for consistent code style.
- **`pyrightconfig.json`** — type checker configuration.
- **`.github/workflows/ci.yml`** — GitHub Actions CI workflow that runs lint, type check, and tests on every push and pull request.
- `nbformat` to `requirements.txt` (needed by notebook cell compilation).

### Changed
- **Training optimization**: Feature matrix is now built once (vectorized pandas/numpy) instead of 10 times via row-by-row Python loops. `StandardScaler` is fit once on the h=1 train split and reused across all 10 horizons. Each horizon's target is constructed as a single numpy slice (`prices[h:n]`). Running time reduced from ~2.7s to ~1.4s (warm cache).
- **Metadata CSVs excluded from MySQL loader**: `data_dictionary.csv` and `source_catalog.csv` are no longer loaded as MySQL tables. Both `src/load_mysql.py` and `main_with_db.py` exclude them via `_METADATA_TABLES`.
- **Hardcoded paths fixed**: `src/load_mysql.py` now resolves `DATASET_DIR` and `SQL_DIR` relative to `__file__` instead of relying on the working directory.
- **`main.py`** rewritten to use shared `src/pipeline.py` — the duplicated training loop is removed.
- **`main_with_db.py`** imports from `src/pipeline.py` instead of from `main.py`, eliminating the cross-module dependency.
- **Benchmark metrics** updated to reflect current model output.
- **`requirements.txt`** pins upper bounds (`<` versions) to prevent future breakage.
- **Project layout** updated in README to include new files.

### Fixed
- **`train_date_range` in model metadata** — previously wrote `h1_test_dates` instead of `h1_train_dates`. Now correctly captures the training date range.
- **Broad `except Exception`** in `main_with_db.py:wait_for_mysql_ready` narrowed to `mysql.errors.Error`.
- **Missing `import csv`** in test helper (`tests/test_predict.py`).
- **`compute_derived` None-handling** when all values are missing with `strict=True` — `vol_regime` now defaults to `1.0` instead of `0.0`.

### Linted / Type-checked
- All C408 `dict()` calls replaced with `{}` literals in `Visualization/visualize_predictions.py`
- All E701 multi-statement if/elif lines expanded to separate lines in `src/predict_oil_price.py`
- E402 (sys.path before import) suppressed with `# noqa: E402` in `main.py` and `main_with_db.py`
- N806 uppercase variables renamed to lowercase in `src/train_oil_model.py`
- Return type annotations fixed in `apply_momentum_blend` and `_build_feature_matrix`
- Numpy float64 returns explicitly converted to Python `float` in `metrics()`
