"""main_with_db.py
Full Brent oil price pipeline WITH MySQL integration.

Pre-requisite
-------------
Start the Docker container first:
    docker-compose up -d

Then run this script from the project root:
    python main_with_db.py

Pipeline
--------
  0. Wait  — poll MySQL until the container is accepting connections
  1. Load   — upsert all CSVs in datasets/ into MySQL + apply analytics views
  2. Train  — fit the Ridge model on ops_market_daily.csv
  3. Predict — Monte Carlo stochastic forecast (500 paths, 40 pct momentum blend)
  4. Visualize — open two interactive Plotly charts in the browser
"""
from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

# ── Import path setup ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
_SRC  = str(_ROOT / "src")
_VIZ  = str(_ROOT / "Visualization")

for _p in (_SRC, _VIZ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Local imports ─────────────────────────────────────────────────────────────
from db_config           import get_mysql_config
import load_mysql        as _loader
import train_oil_model   as _train
import predict_oil_price as _predict
import visualize_predictions as _viz

# Pull the core pipeline functions from main.py (avoids duplication)
from main import run_train, run_predict, run_visualize


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(step: int | str, total: int, label: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  STEP {step}/{total} — {label}")
    print(f"{'=' * 62}")


def wait_for_mysql(
    host: str,
    port: int,
    timeout_s: int = 60,
    poll_interval_s: float = 2.0,
) -> None:
    """
    Block until the MySQL port is reachable (TCP connection succeeds) or
    `timeout_s` seconds have elapsed.

    Raises RuntimeError if the container is still not reachable after timeout.
    """
    deadline = time.time() + timeout_s
    attempt  = 0
    print(f"  Waiting for MySQL at {host}:{port}  (timeout={timeout_s}s) ...")
    while time.time() < deadline:
        attempt += 1
        try:
            with socket.create_connection((host, port), timeout=2):
                print(f"  MySQL is reachable after {attempt} poll(s).")
                return
        except OSError:
            print(f"    [{attempt}] Not ready yet — retrying in {poll_interval_s}s ...")
            time.sleep(poll_interval_s)

    raise RuntimeError(
        f"MySQL at {host}:{port} did not become reachable within {timeout_s}s.\n"
        "Make sure the Docker container is running:\n"
        "    docker-compose up -d"
    )


def wait_for_mysql_ready(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    timeout_s: int = 30,
    poll_interval_s: float = 2.0,
) -> None:
    """
    After the TCP port is open, MySQL may still be initialising.
    Poll until an actual connection + query succeeds.
    """
    mysql = _loader.require_connector()
    deadline = time.time() + timeout_s
    attempt  = 0
    print(f"  Waiting for MySQL to finish initialising ...")
    while time.time() < deadline:
        attempt += 1
        try:
            conn = mysql.connect(
                host=host, port=port, user=user, password=password,
                connection_timeout=3,
            )
            conn.close()
            print(f"  MySQL is ready after {attempt} poll(s).")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"    [{attempt}] Init in progress ({exc}) — retrying ...")
            time.sleep(poll_interval_s)

    raise RuntimeError(
        f"MySQL accepted TCP connections but login timed out after {timeout_s}s.\n"
        "Check container logs:  docker-compose logs mysql"
    )


# ── MySQL load step ───────────────────────────────────────────────────────────

def run_load_mysql(
    dataset_dir: Path,
    replace: bool = False,
    only: list[str] | None = None,
) -> None:
    """Upsert all CSVs in dataset_dir into MySQL and apply analytics views."""
    mysql  = _loader.require_connector()
    config = get_mysql_config()

    connection = mysql.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        autocommit=False,
    )
    cursor = connection.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {_loader.q(config.database)}")
    cursor.execute(f"USE {_loader.q(config.database)}")

    dictionary = _loader.load_data_dictionary(dataset_dir / "data_dictionary.csv")
    csv_files  = sorted(dataset_dir.glob("*.csv"))
    if only:
        wanted    = set(only)
        csv_files = [p for p in csv_files if p.stem in wanted]

    total_rows = 0
    for csv_path in csv_files:
        table  = csv_path.stem
        loaded = _loader.load_csv(
            cursor, table, csv_path, dictionary.get(table, {}), replace
        )
        connection.commit()
        print(f"  Loaded {loaded:>6,} rows  ->  {table}")
        total_rows += loaded

    sql_views = _ROOT / "sql" / "analytics_views.sql"
    _loader.apply_sql_file(cursor, sql_views, config.database)
    connection.commit()
    cursor.close()
    connection.close()
    print(f"  Total: {total_rows:,} rows across {len(csv_files)} table(s)")
    print(f"  Database `{config.database}` is ready.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    config = get_mysql_config()

    market_csv      = _ROOT / "datasets"        / "ops_market_daily.csv"
    output_dir      = _ROOT / "model_artifacts"
    artifact_path   = output_dir / "oil_price_model.json"
    predictions_csv = output_dir / "test_predictions.csv"
    forecast_csv    = output_dir / "forward_forecast.csv"

    t0 = time.time()

    # ── Step 0: Wait for MySQL ────────────────────────────────────────────────
    _banner(0, 4, f"Wait for MySQL  ({config.host}:{config.port})")
    wait_for_mysql(config.host, config.port, timeout_s=90)
    wait_for_mysql_ready(
        config.host, config.port,
        config.user, config.password, config.database,
        timeout_s=60,
    )

    # ── Step 1: Load CSVs into MySQL ──────────────────────────────────────────
    _banner(1, 4, "Load datasets into MySQL")
    run_load_mysql(_ROOT / "datasets")

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    _banner(2, 4, "Train Ridge model  (10 horizons, h=1..10)")
    run_train(market_csv, output_dir)

    # ── Step 3: Predict ───────────────────────────────────────────────────────
    _banner(3, 4, "Monte Carlo forecast  (500 paths, 40% momentum blend)")
    run_predict(artifact_path, market_csv)

    # ── Step 4: Visualize ─────────────────────────────────────────────────────
    _banner(4, 4, "Interactive visualizations  (charts open in browser)")
    run_visualize(predictions_csv, forecast_csv)

    print(f"\nFull pipeline (with DB) complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
