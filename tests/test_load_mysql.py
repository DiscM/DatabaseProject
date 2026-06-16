from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from load_mysql import _METADATA_TABLES, load_data_dictionary, mysql_type, q


class TestQuoteIdentifier:
    def test_simple_name(self):
        assert q("my_table") == "`my_table`"

    def test_alphanumeric(self):
        assert q("col_123") == "`col_123`"

    def test_rejects_spaces(self):
        with pytest.raises(ValueError, match="Unsafe identifier"):
            q("my table")

    def test_rejects_semicolon(self):
        with pytest.raises(ValueError, match="Unsafe identifier"):
            q("table; DROP TABLE users")


class TestMysqlType:
    def test_date_column_by_name(self):
        assert mysql_type("market_date", None) == "DATE"

    def test_int_dtype(self):
        assert mysql_type("some_count", "int") == "INT"

    def test_float_dtype(self):
        assert mysql_type("some_rate", "float") == "DOUBLE"

    def test_text_description(self):
        assert mysql_type("description", None) == "TEXT"
        assert mysql_type("policy_response", None) == "TEXT"

    def test_default_varchar(self):
        assert mysql_type("country_name", None) == "VARCHAR(512)"


class TestMetadataExclusion:
    def test_data_dictionary_is_excluded(self):
        assert "data_dictionary" in _METADATA_TABLES

    def test_source_catalog_is_excluded(self):
        assert "source_catalog" in _METADATA_TABLES


class TestLoadDataDictionary:
    def test_parses_valid_csv(self, tmp_path):
        path = tmp_path / "data_dictionary.csv"
        path.write_text(
            "table_name,column_name,dtype,description\n"
            "ops_market_daily,brent_price_usd,float,Price in USD\n"
            "ops_market_daily,market_date,datetime,Trade date\n",
            encoding="utf-8",
        )
        result = load_data_dictionary(path)
        assert "ops_market_daily" in result
        assert result["ops_market_daily"]["brent_price_usd"] == "float"
        assert result["ops_market_daily"]["market_date"] == "datetime"

    def test_missing_file_returns_empty(self):
        result = load_data_dictionary(Path("/no/such/file.csv"))
        assert result == {}
