import numpy as np
import pandas as pd
import pytest

from imputacion_co2_ml import _normalize_missing_rate


class TestNormalizeMissingRate:
    def test_zero(self):
        assert _normalize_missing_rate(0) == 0.0

    def test_fraction(self):
        assert _normalize_missing_rate(0.2) == pytest.approx(0.2)

    def test_one_is_fraction(self):
        assert _normalize_missing_rate(1) == pytest.approx(1.0)

    def test_percentage_twenty(self):
        assert _normalize_missing_rate(20) == pytest.approx(0.2)

    def test_percentage_hundred(self):
        assert _normalize_missing_rate(100) == pytest.approx(1.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="negativo"):
            _normalize_missing_rate(-1)

    def test_over_hundred_raises(self):
        with pytest.raises(ValueError, match="entre 0 y 1"):
            _normalize_missing_rate(101)

    def test_fraction_zero_point_five(self):
        assert _normalize_missing_rate(0.5) == pytest.approx(0.5)

    def test_percentage_fifty(self):
        assert _normalize_missing_rate(50) == pytest.approx(0.5)


class TestCo2ZeroFilter:
    def test_zero_co2_excluded_from_training_mask(self):
        # La mascara de entrenamiento debe excluir CO2=0 ademas de NaN.
        target = pd.Series([100.0, 0.0, 150.0, np.nan, 120.0])
        mask = target.notna() & (target > 0)
        assert mask.iloc[0]       # 100  → incluido
        assert not mask.iloc[1]   # 0    → excluido
        assert mask.iloc[2]       # 150  → incluido
        assert not mask.iloc[3]   # NaN  → excluido
        assert mask.iloc[4]       # 120  → incluido
        assert mask.sum() == 3

    def test_only_nan_excluded_without_zero_filter(self):
        # Verifica que sin el filtro >0, los ceros pasarían.
        target = pd.Series([100.0, 0.0, 150.0])
        mask_old = target.notna()
        mask_new = target.notna() & (target > 0)
        assert mask_old.sum() == 3  # el filtro antiguo dejaba pasar el cero
        assert mask_new.sum() == 2  # el nuevo lo excluye


class TestNegativeClip:
    def test_clip_removes_negative_predictions(self):
        raw = np.array([120.0, -5.0, 0.0, 85.3, -0.001])
        clipped = np.maximum(raw, 0.0)
        assert (clipped >= 0).all()
        assert clipped[0] == pytest.approx(120.0)
        assert clipped[1] == pytest.approx(0.0)
        assert clipped[2] == pytest.approx(0.0)
        assert clipped[3] == pytest.approx(85.3)
        assert clipped[4] == pytest.approx(0.0)

    def test_positive_predictions_unchanged(self):
        raw = np.array([50.0, 100.0, 200.0])
        clipped = np.maximum(raw, 0.0)
        np.testing.assert_array_almost_equal(clipped, raw)
