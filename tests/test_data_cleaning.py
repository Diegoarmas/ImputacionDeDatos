import sys
from pathlib import Path
import io
import csv

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_cleaning import (
    TARGET_COLUMN,
    _normalize_number_string,
    to_float_series,
    prepare_dataframe,
    load_csv_resilient,
    _repaired_rows,
)


class TestNormalizeNumberString:
    def test_empty_string(self):
        assert _normalize_number_string("") == ""

    def test_nan_value(self):
        assert _normalize_number_string(np.nan) == ""

    def test_plain_integer(self):
        assert _normalize_number_string("1234") == "1234"

    def test_comma_decimal_european(self):
        # "153,000" matches THOUSANDS_COMMA_RE (1-3 digits + ,3digits) → thousands sep
        assert _normalize_number_string("153,000") == "153000"

    def test_dot_thousands_european(self):
        # "1.234" matches THOUSANDS_DOT_RE → strip dot
        assert _normalize_number_string("1.234") == "1234"

    def test_comma_thousands_anglosaxon(self):
        # "1,234" matches THOUSANDS_COMMA_RE → strip comma
        assert _normalize_number_string("1,234") == "1234"

    def test_mixed_european_format(self):
        # "1.234,56" → last sep is comma → European decimal
        assert _normalize_number_string("1.234,56") == "1234.56"

    def test_mixed_anglosaxon_format(self):
        # "1,234.56" → last sep is dot → Anglo-Saxon decimal
        assert _normalize_number_string("1,234.56") == "1234.56"

    def test_plain_decimal_dot(self):
        assert _normalize_number_string("3.14") == "3.14"

    def test_nbsp_stripped(self):
        result = _normalize_number_string("\u00a01\u00a0234")
        assert result == "1234"


class TestToFloatSeries:
    def test_basic_integers(self):
        s = pd.Series(["1", "2", "3"])
        result = to_float_series(s)
        assert list(result) == [1.0, 2.0, 3.0]

    def test_european_decimals(self):
        # "153,000" → THOUSANDS_COMMA_RE matches → thousands sep → 153000
        s = pd.Series(["153,000", "122,000"])
        result = to_float_series(s)
        assert result.iloc[0] == 153000.0
        assert result.iloc[1] == 122000.0

    def test_nan_handling(self):
        s = pd.Series(["", "nan", "None", "1.5"])
        result = to_float_series(s)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])
        assert result.iloc[3] == 1.5

    def test_thousands_comma(self):
        s = pd.Series(["1,234"])
        result = to_float_series(s)
        assert result.iloc[0] == 1234.0

    def test_coerce_invalid(self):
        s = pd.Series(["abc", "1.5"])
        result = to_float_series(s)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == 1.5


class TestPrepareDataframe:
    def _sample_df(self, co2_values=None):
        n = 5
        data = {
            TARGET_COLUMN: co2_values or ["100", "200", "150", "120", "180"],
            "TARA": ["1000", "1200", "900", "1100", "1050"],
            "MARCA": ["VW", "FORD", "SEAT", "RENAULT", "BMW"],
        }
        return pd.DataFrame(data)

    def test_target_removed_from_features(self):
        df = self._sample_df()
        features, target = prepare_dataframe(df)
        assert TARGET_COLUMN not in features.columns

    def test_target_is_numeric(self):
        df = self._sample_df()
        _, target = prepare_dataframe(df)
        assert pd.api.types.is_numeric_dtype(target)

    def test_numeric_columns_converted(self):
        df = self._sample_df()
        features, _ = prepare_dataframe(df)
        assert pd.api.types.is_numeric_dtype(features["TARA"])

    def test_date_columns_expanded(self):
        df = self._sample_df()
        df["FECHA_MATR"] = ["01/01/2020"] * 5
        features, _ = prepare_dataframe(df)
        assert "FECHA_MATR_YEAR" in features.columns
        assert "FECHA_MATR_MONTH" in features.columns
        assert "FECHA_MATR" not in features.columns

    def test_returns_correct_lengths(self):
        df = self._sample_df()
        features, target = prepare_dataframe(df)
        assert len(features) == 5
        assert len(target) == 5


class TestLoadCsvResilient:
    def _write_csv(self, tmp_path, rows, sep=","):
        path = tmp_path / "test.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=sep)
            writer.writerows(rows)
        return path

    def test_normal_csv(self, tmp_path):
        rows = [
            ["A", "B", "C"],
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]
        path = self._write_csv(tmp_path, rows)
        df = load_csv_resilient(path, sep=",", encoding="utf-8")
        assert list(df.columns) == ["A", "B", "C"]
        assert len(df) == 2

    def test_semicolon_separator(self, tmp_path):
        rows = [["X", "Y"], ["10", "20"]]
        path = self._write_csv(tmp_path, rows, sep=";")
        df = load_csv_resilient(path, sep=";", encoding="utf-8")
        assert list(df.columns) == ["X", "Y"]
        assert df["X"].iloc[0] == "10"


class TestRepairedRows:
    def test_clean_file(self, tmp_path):
        path = tmp_path / "clean.csv"
        path.write_text("A,B,C\n1,2,3\n4,5,6\n", encoding="utf-8")
        rows, broken = _repaired_rows(path, sep=",", encoding="utf-8")
        assert broken == 0
        assert len(rows) == 3  # header + 2 data rows

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("", encoding="utf-8")
        rows, broken = _repaired_rows(path, sep=",", encoding="utf-8")
        assert rows == []
        assert broken == 0
