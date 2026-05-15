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
