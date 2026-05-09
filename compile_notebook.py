"""compile_notebook.py
Re-generates oil_news_project_demo.ipynb from the four source files.

Run from the project root:
    python compile_notebook.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source_lines(text: str) -> list[str]:
    """Convert a multiline string into the JSON source-line format nbformat uses."""
    lines = text.splitlines(keepends=True)
    # nbformat: last line must NOT have a trailing newline
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return lines


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source_lines(text.strip()),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source_lines(text.strip()),
    }


def strip_main_block(code: str) -> str:
    """Remove `if __name__ == '__main__': ...` guard (and anything after it)."""
    # Split on the guard line and drop everything from it onward
    pattern = r'\nif __name__ == ["\']__main__["\']:'
    parts = re.split(pattern, code, maxsplit=1)
    return parts[0].rstrip()


def patch_argparse(code: str) -> str:
    """Make argparse work inside Jupyter (no CLI args present)."""
    return code.replace("parser.parse_args()", "parser.parse_args([])")


def patch_file_dunder(code: str, replacement: str = "Path('src')") -> str:
    """Replace __file__-based path resolution (undefined in Jupyter kernels)."""
    # Replace the entire sys.path guard block we added in load_mysql.py
    guard = (
        "import sys\n"
        "from pathlib import Path as _Path\n"
        "if str(_Path(__file__).parent) not in sys.path:\n"
        "    sys.path.insert(0, str(_Path(__file__).parent))\n"
    )
    return code.replace(guard, "")


# ---------------------------------------------------------------------------
# Cell builders — one per source file, mirroring the standalone scripts
# ---------------------------------------------------------------------------

def section_setup() -> list[dict]:
    """
    Notebook-level setup: working-directory normalisation.
    This replaces the __file__ / sys.path machinery used by the .py files.
    """
    setup_code = """\
import sys, os
from pathlib import Path

# Ensure the notebook CWD is the project root so relative paths work
# (same assumption the standalone scripts make when run from the root).
_root = Path.cwd()
if (_root / "src").exists():
    # Already at project root
    pass
elif (_root.parent / "src").exists():
    os.chdir(_root.parent)
    _root = Path.cwd()

# Make src/ importable (mirrors the sys.path guard in load_mysql.py)
_src = str(_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

print(f"Project root: {Path.cwd()}")
"""
    return [
        md_cell(
            "## Setup\n"
            "Ensures the notebook's working directory matches the project root "
            "and that `src/` is on the import path — mirroring the environment "
            "the standalone scripts assume."
        ),
        code_cell(setup_code),
    ]


def section_db_config() -> list[dict]:
    raw = Path("src/db_config.py").read_text(encoding="utf-8")
    # db_config has no argparse or __file__ usage — embed as-is
    code = strip_main_block(raw)
    return [
        md_cell(
            "## 1. Database Configuration\n"
            "Loads MySQL connection details from `.env` (or environment variables). "
            "Source: `src/db_config.py`"
        ),
        code_cell(code),
    ]


def section_load_mysql() -> list[dict]:
    raw = Path("src/load_mysql.py").read_text(encoding="utf-8")
    code = raw
    # 1. Remove the sys.path guard (the Setup cell above handles it)
    code = patch_file_dunder(code)
    # 2. Patch argparse
    code = patch_argparse(code)
    # 3. Strip __main__ block (we call main() explicitly below)
    code = strip_main_block(code)
    return [
        md_cell(
            "## 2. Load Datasets into MySQL\n"
            "Reads every CSV in `datasets/` and upserts it into MySQL, then "
            "applies the analytics views SQL. Source: `src/load_mysql.py`"
        ),
        code_cell(code),
        code_cell("# Execute: load all CSVs into MySQL\nmain()"),
    ]


def section_train_model() -> list[dict]:
    raw = Path("src/train_oil_model.py").read_text(encoding="utf-8")
    code = patch_argparse(raw)
    code = strip_main_block(code)
    return [
        md_cell(
            "## 3. Train Oil Price Model\n"
            "Builds feature/label pairs with a configurable horizon, trains a "
            "`StandardScaler → Ridge` pipeline, and saves the model + metrics. "
            "Source: `src/train_oil_model.py`"
        ),
        code_cell(code),
        code_cell("# Execute: train the model and save artifacts\nmain()"),
    ]


def section_predict() -> list[dict]:
    raw = Path("src/predict_oil_price.py").read_text(encoding="utf-8")
    code = patch_argparse(raw)
    code = strip_main_block(code)
    return [
        md_cell(
            "## 4. Forward Price Forecast\n"
            "Loads the trained model and feeds the last `horizon` rows of actual "
            "market data to produce a day-by-day forward forecast. "
            "Source: `src/predict_oil_price.py`"
        ),
        code_cell(code),
        code_cell("# Execute: generate the forward forecast CSV\nmain()"),
    ]


def section_visualize() -> list[dict]:
    raw = Path("Visualization/visualize_predictions.py").read_text(encoding="utf-8")
    # Strip the __main__ block; we provide our own invocation below
    code = strip_main_block(raw)

    run_code = """\
# Paths mirror the defaults used by the visualize_predictions.py __main__ block.
# Charts are rendered inline (Plotly) — no files are written to disk.
_predictions_csv = str(Path("model_artifacts") / "test_predictions.csv")
_forecast_csv    = str(Path("model_artifacts") / "forward_forecast.csv")
_market_csv      = str(Path("datasets")        / "ops_market_daily.csv")

create_visualizations(
    _predictions_csv,
    forecast_csv_path=_forecast_csv,
    market_csv_path=_market_csv,
)
"""
    return [
        md_cell(
            "## 5. Visualizations\n"
            "Generates four interactive Plotly charts rendered inline:\n"
            "- Actual vs Predicted (line, zoomable week-by-week)\n"
            "- Prediction accuracy scatter\n"
            "- Model vs Baseline vs Actual\n"
            "- Forward forecast (recent actual + future predictions)\n\n"
            "Use the **1W / 2W / 1M … All** buttons or drag the range slider to zoom.\n"
            "Source: `Visualization/visualize_predictions.py`"
        ),
        code_cell(code),
        code_cell(run_code),
    ]


# ---------------------------------------------------------------------------
# Notebook assembly
# ---------------------------------------------------------------------------

def main() -> None:
    cells: list[dict] = []

    cells.append(md_cell(
        "# Oil News Project — Demo Notebook\n\n"
        "End-to-end walkthrough of the Brent crude oil price prediction pipeline:\n\n"
        "1. Database configuration\n"
        "2. Load datasets into MySQL\n"
        "3. Train the Ridge regression model\n"
        "4. Generate a multi-day forward forecast\n"
        "5. Visualize actual vs predicted prices\n\n"
        "> **Run cells top-to-bottom.** MySQL must be reachable for step 2; "
        "steps 3–5 only need the CSV datasets."
    ))

    cells.extend(section_setup())
    cells.extend(section_db_config())
    cells.extend(section_load_mysql())
    cells.extend(section_train_model())
    cells.extend(section_predict())
    cells.extend(section_visualize())

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out = Path("oil_news_project_demo.ipynb")
    out.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Notebook written -> {out}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
