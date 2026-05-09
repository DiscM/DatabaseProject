"""compile_notebook.py
Regenerates oil_news_project_demo.ipynb from the project's source .py files.

Run from the project root:
    python compile_notebook.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# nbformat cell helpers
# ---------------------------------------------------------------------------

def _lines(text: str) -> list[str]:
    """Convert a string to the line-list format nbformat requires."""
    lines = text.splitlines(keepends=True)
    # nbformat: last line must NOT end with a newline
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return lines


def md(text: str) -> dict:
    """Create a Markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text.strip())}


def code(text: str) -> dict:
    """Create a Code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text.strip()),
    }


# ---------------------------------------------------------------------------
# Source loading — reads the original .py files with minimal Jupyter patches
# ---------------------------------------------------------------------------

def _strip_main_guard(src: str) -> str:
    """Drop `if __name__ == '__main__': ...` and everything after it."""
    parts = re.split(r'\nif __name__ == ["\']__main__["\']:', src, maxsplit=1)
    return parts[0].rstrip()


def _patch_argparse(src: str) -> str:
    """Make argparse work inside Jupyter (no CLI args present)."""
    return src.replace("parser.parse_args()", "parser.parse_args([])")


def _patch_file_dunder(src: str) -> str:
    """
    Remove the sys.path/__file__ guard used in load_mysql.py.
    The notebook's Setup cell handles sys.path instead.
    """
    guard = (
        "import sys\n"
        "from pathlib import Path as _Path\n"
        "if str(_Path(__file__).parent) not in sys.path:\n"
        "    sys.path.insert(0, str(_Path(__file__).parent))\n"
    )
    return src.replace(guard, "")


def load_source(path: str) -> str:
    """
    Read a .py source file and apply the minimal patches required to run
    it as a Jupyter cell:

      1. Strip the `if __name__ == '__main__':` guard (we provide our own
         invocation cell beneath each section).
      2. Replace `parser.parse_args()` with `parser.parse_args([])` so
         argparse doesn't try to read Jupyter's sys.argv.
      3. Remove the __file__-based sys.path guard in load_mysql.py; the
         notebook's Setup cell handles that instead.

    Everything else is taken verbatim from the source file.
    """
    raw = Path(path).read_text(encoding="utf-8")
    src = _patch_argparse(raw)
    src = _patch_file_dunder(src)
    src = _strip_main_guard(src)
    return src


# ---------------------------------------------------------------------------
# Notebook sections
# ---------------------------------------------------------------------------

def section_title() -> list[dict]:
    return [md("""\
# Oil News Project — Demo Notebook

End-to-end walkthrough of the Brent crude oil price prediction pipeline:

1. **Setup** — working directory, import paths, Plotly renderer
2. **Database Configuration** — MySQL connection via `.env`
3. **Load Datasets** — CSV → MySQL upsert + analytics views
4. **Train Model** — `StandardScaler → Ridge` with 10 forecast horizons
5. **Forward Forecast** — iterative 10-day Brent price projection
6. **Visualizations** — interactive Plotly charts (zoomable, pannable)

> **Run cells top-to-bottom.**  MySQL must be reachable for step 3;
> steps 4–6 only need the CSV datasets in `datasets/`.
""")]


def section_setup() -> list[dict]:
    src = """\
import sys, os
from pathlib import Path

# ── Working directory ─────────────────────────────────────────────────────────
# All scripts assume they are run from the project root.  Adjust CWD if the
# notebook kernel started somewhere else (e.g. inside Visualization/).
_root = Path.cwd()
if not (_root / "src").exists() and (_root.parent / "src").exists():
    os.chdir(_root.parent)
    _root = Path.cwd()

# ── Import path ───────────────────────────────────────────────────────────────
# Add src/ so db_config and friends can be imported without installation.
_src = str(_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# ── Plotly renderer ───────────────────────────────────────────────────────────
# Force the interactive HTML renderer.  Without this, Jupyter may auto-detect
# the wrong backend and fall back to a static PNG image instead of a live chart.
import plotly.io as pio
pio.renderers.default = "notebook"

print(f"Project root : {_root}")
print(f"Python       : {sys.version.split()[0]}")
print(f"Plotly render: {pio.renderers.default}")
"""
    return [
        md("""\
## 1. Setup
Normalises the working directory, adds `src/` to the Python import path, and
configures the Plotly renderer so `fig.show()` produces interactive HTML charts
rather than static images."""),
        code(src),
    ]


def section_db_config() -> list[dict]:
    src = load_source("src/db_config.py")
    return [
        md("""\
## 2. Database Configuration
Loads MySQL connection details from `.env` (or environment variables).

Source: `src/db_config.py`"""),
        code(src),
    ]


def section_load_mysql() -> list[dict]:
    src = load_source("src/load_mysql.py")
    return [
        md("""\
## 3. Load Datasets into MySQL
Reads every CSV in `datasets/` and upserts it into MySQL, then applies the
analytics-views SQL.

Source: `src/load_mysql.py`"""),
        code(src),
        code("# ── Execute ──────────────────────────────────────────────────────────────────\nmain()"),
    ]


def section_train_model() -> list[dict]:
    src = load_source("src/train_oil_model.py")
    return [
        md("""\
## 4. Train Oil Price Model
Builds feature/label pairs for **10 forecast horizons** (h=1 through h=10),
trains a `StandardScaler → Ridge` pipeline per horizon, and saves models +
metrics to `model_artifacts/`.

Source: `src/train_oil_model.py`"""),
        code(src),
        code("# ── Execute ──────────────────────────────────────────────────────────────────\nmain()"),
    ]


def section_predict() -> list[dict]:
    src = load_source("src/predict_oil_price.py")
    return [
        md("""\
## 5. Forward Price Forecast
Uses a **Monte Carlo stochastic forecast**: 500 independent simulation paths
apply the h=1 Ridge model iteratively. At each step Gaussian noise
(sigma = h=1 test RMSE) is injected so paths diverge, producing a realistic
probability fan that widens naturally with horizon.

Output: `model_artifacts/forward_forecast.csv` — median + P10/P25/P75/P90.
Source: `src/predict_oil_price.py`"""),
        code(src),
        code("# ── Execute ──────────────────────────────────────────────────────────────────\nmain()"),
    ]


def section_visualize() -> list[dict]:
    src = load_source("Visualization/visualize_predictions.py")
    run = """\
# ── Execute ───────────────────────────────────────────────────────────────────
# Paths mirror the defaults used when visualize_predictions.py is run standalone.
# Charts render inline as interactive Plotly HTML — no PNG files are written.
_predictions_csv = str(Path("model_artifacts") / "test_predictions.csv")
_forecast_csv    = str(Path("model_artifacts") / "forward_forecast.csv")

create_visualizations(
    _predictions_csv,
    forecast_csv_path=_forecast_csv,
)
"""
    return [
        md("""\
## 6. Visualizations
Generates **two interactive Plotly charts** (zoomable and pannable):

- **Prediction accuracy scatter** — actual vs predicted price correlation
- **Model vs Baseline vs Actual** — test-set time-series with the 10-day
  forward forecast bridged on as a dashed line with a shaded forecast window

Use the **1W / 2W / 1M … All** range buttons or drag the slider to zoom.

Source: `Visualization/visualize_predictions.py`"""),
        code(src),
        code(run),
    ]


# ---------------------------------------------------------------------------
# Notebook assembly
# ---------------------------------------------------------------------------

def build_notebook() -> dict:
    cells: list[dict] = []
    cells.extend(section_title())
    cells.extend(section_setup())
    cells.extend(section_db_config())
    cells.extend(section_load_mysql())
    cells.extend(section_train_model())
    cells.extend(section_predict())
    cells.extend(section_visualize())

    return {
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


def main() -> None:
    out = Path("oil_news_project_demo.ipynb")
    nb = build_notebook()
    out.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Written -> {out}  ({len(nb['cells'])} cells)")
    for i, cell in enumerate(nb["cells"]):
        tag = "MD  " if cell["cell_type"] == "markdown" else "CODE"
        preview = "".join(cell["source"])[:60].replace("\n", " ")
        preview = preview.encode("ascii", errors="replace").decode("ascii")
        print(f"  [{i:02d}] {tag}  {preview}")


if __name__ == "__main__":
    main()
