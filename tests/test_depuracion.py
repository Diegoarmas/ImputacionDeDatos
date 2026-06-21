import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from depuracion_txt import _has_value


class TestHasValue:
    def test_positive_integer_string(self):
        s = pd.Series(["120"])
        assert _has_value(s).iloc[0]

    def test_zero_string_excluded(self):
        s = pd.Series(["0"])
        assert not _has_value(s).iloc[0]

    def test_zero_float_string_excluded(self):
        s = pd.Series(["0.0"])
        assert not _has_value(s).iloc[0]

    def test_empty_string_excluded(self):
        s = pd.Series([""])
        assert not _has_value(s).iloc[0]

    def test_nan_excluded(self):
        s = pd.Series([np.nan])
        assert not _has_value(s).iloc[0]

    def test_nan_string_excluded(self):
        s = pd.Series(["nan"])
        assert not _has_value(s).iloc[0]

    def test_none_string_excluded(self):
        s = pd.Series(["None"])
        assert not _has_value(s).iloc[0]

    def test_null_string_excluded(self):
        s = pd.Series(["null"])
        assert not _has_value(s).iloc[0]

    def test_positive_float_string(self):
        s = pd.Series(["99.5"])
        assert _has_value(s).iloc[0]

    def test_mixed_series(self):
        s = pd.Series(["100", "0", "", "nan", "150"])
        result = _has_value(s)
        assert result.iloc[0]       # 100 → válido
        assert not result.iloc[1]   # 0   → excluido
        assert not result.iloc[2]   # ""  → excluido
        assert not result.iloc[3]   # nan → excluido
        assert result.iloc[4]       # 150 → válido
