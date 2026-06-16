from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from pathlib import Path as _Path

if str(_Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(_Path(__file__).parent))

from db_config import get_mysql_config

_SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = _SCRIPT_DIR.parent / "datasets"
SQL_DIR     = _SCRIPT_DIR.parent / "sql"

_METADATA_TABLES = {"data_dictionary", "source_catalog"}

DATE_COLUMNS = {
    "trade_date",
    "event_date",
    "gpr_date",
    "month_start",
    "snapshot_date",
    "market_date",
    "full_date",
}


def require_connector():
    try:
        import mysql.connector  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "mysql-connector-python is required for loading MySQL. "
            "Install with: python -m pip install -r requirements.txt"
        ) from exc
    return mysql.connector


def q(identifier: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise ValueError(f"Unsafe identifier: {identifier}")
    return f"`{identifier}`"


def load_data_dictionary(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    mapping: dict[str, dict[str, str]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            mapping[row["table_name"]][row["column_name"]] = row["dtype"]
    return mapping


def mysql_type(column: str, dtype: str | None) -> str:
    if column in DATE_COLUMNS or (dtype and "datetime" in dtype):
        return "DATE"
    if dtype and "int" in dtype:
        return "INT"
    if dtype and "float" in dtype:
        return "DOUBLE"
    if column.endswith("_description") or column in {"description", "policy_response"}:
        return "TEXT"
    return "VARCHAR(512)"


def primary_key_for(table: str, columns: list[str]) -> str | None:
    candidates = [column for column in columns if column.endswith("_id") or column.endswith("_key")]
    if candidates and candidates[0] in columns:
        return candidates[0]
    if table.startswith("ops_"):
        candidate = table.removeprefix("ops_").rstrip("s") + "_id"
        return candidate if candidate in columns else None
    return None


def create_table_sql(table: str, columns: list[str], dtypes: dict[str, str]) -> str:
    pk = primary_key_for(table, columns)
    definitions = []
    for column in columns:
        col_type = mysql_type(column, dtypes.get(column))
        nullable = "NOT NULL" if column == pk else "NULL"
        definitions.append(f"  {q(column)} {col_type} {nullable}")
    if pk:
        definitions.append(f"  PRIMARY KEY ({q(pk)})")
    return f"CREATE TABLE IF NOT EXISTS {q(table)} (\n" + ",\n".join(definitions) + "\n) ENGINE=InnoDB;"


def iter_csv_rows(path: Path, columns: list[str]) -> Iterable[tuple[object, ...]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            values: list[object] = []
            for column in columns:
                value = row.get(column, "")
                values.append(None if value == "" else value)
            yield tuple(values)


def load_csv(cursor, table: str, path: Path, dtypes: dict[str, str], replace: bool) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        columns = list(next(csv.reader(handle)))

    cursor.execute(create_table_sql(table, columns, dtypes))
    if replace:
        cursor.execute(f"TRUNCATE TABLE {q(table)}")

    placeholders = ", ".join(["%s"] * len(columns))
    column_sql = ", ".join(q(column) for column in columns)
    insert_sql = f"INSERT INTO {q(table)} ({column_sql}) VALUES ({placeholders})"

    batch: list[tuple[object, ...]] = []
    total = 0
    for values in iter_csv_rows(path, columns):
        batch.append(values)
        if len(batch) >= 1000:
            cursor.executemany(insert_sql, batch)
            total += len(batch)
            batch.clear()
    if batch:
        cursor.executemany(insert_sql, batch)
        total += len(batch)
    return total


def apply_sql_file(cursor, path: Path, database: str) -> None:
    if not path.exists():
        return
    sql = path.read_text(encoding="utf-8").replace("oil_news_project", database)
    for statement in [part.strip() for part in sql.split(";") if part.strip()]:
        cursor.execute(statement)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load workspace CSV datasets into MySQL.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--replace", action="store_true", help="Truncate tables before loading.")
    parser.add_argument("--only", nargs="*", help="Optional list of CSV stem/table names to load.")
    args = parser.parse_args()

    mysql = require_connector()
    config = get_mysql_config()
    connection = mysql.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        autocommit=False,
    )
    cursor = connection.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {q(config.database)}")
    cursor.execute(f"USE {q(config.database)}")

    dictionary = load_data_dictionary(args.dataset_dir / "data_dictionary.csv")
    csv_files = sorted(
        p for p in args.dataset_dir.glob("*.csv")
        if p.stem not in _METADATA_TABLES
    )
    if args.only:
        wanted = set(args.only)
        csv_files = [path for path in csv_files if path.stem in wanted]

    for csv_path in csv_files:
        table = csv_path.stem
        loaded = load_csv(cursor, table, csv_path, dictionary.get(table, {}), args.replace)
        connection.commit()
        print(f"Loaded {loaded:>6} rows into {table}")

    apply_sql_file(cursor, SQL_DIR / "analytics_views.sql", config.database)
    connection.commit()
    cursor.close()
    connection.close()
    print(f"Done. Database `{config.database}` is ready.")


if __name__ == "__main__":
    main()
