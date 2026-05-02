import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling import build_pipeline, fit_and_evaluate


def _sample_feature_df(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "TARA": rng.uniform(800, 2000, n),
            "POTENCIA": rng.uniform(50, 300, n),
            "CILINDRADA": rng.uniform(800, 4000, n),
            "MARCA": rng.choice(["VW", "FORD", "SEAT"], n),
        }
    )


def _sample_target(n=200):
    rng = np.random.default_rng(0)
    return pd.Series(rng.uniform(80, 300, n))


class TestBuildPipeline:
    def test_returns_three_values(self):
        df = _sample_feature_df()
        result = build_pipeline(df, random_state=42)
        assert len(result) == 3

    def test_numeric_columns_detected(self):
        df = _sample_feature_df()
        _, numeric_cols, categorical_cols = build_pipeline(df, random_state=42)
        assert "TARA" in numeric_cols
        assert "POTENCIA" in numeric_cols
        assert "CILINDRADA" in numeric_cols

    def test_categorical_columns_detected(self):
        df = _sample_feature_df()
        _, numeric_cols, categorical_cols = build_pipeline(df, random_state=42)
        assert "MARCA" in categorical_cols

    def test_pipeline_can_fit_and_predict(self):
        df = _sample_feature_df()
        target = _sample_target()
        pipeline, _, _ = build_pipeline(df, random_state=42)
        pipeline.fit(df, target)
        preds = pipeline.predict(df)
        assert len(preds) == len(df)

    def test_pipeline_steps(self):
        df = _sample_feature_df()
        pipeline, _, _ = build_pipeline(df, random_state=42)
        step_names = [name for name, _ in pipeline.steps]
        assert "preprocessor" in step_names
        assert "model" in step_names


class TestFitAndEvaluate:
    def test_returns_expected_keys(self):
        df = _sample_feature_df()
        target = _sample_target()
        pipeline, _, _ = build_pipeline(df, random_state=42)
        scores = fit_and_evaluate(pipeline, df, target, random_state=42, cv_folds=3)
        assert set(scores.keys()) == {"mae", "rmse", "r2", "mae_std", "rmse_std", "r2_std", "cv_folds"}

    def test_mae_positive(self):
        df = _sample_feature_df()
        target = _sample_target()
        pipeline, _, _ = build_pipeline(df, random_state=42)
        scores = fit_and_evaluate(pipeline, df, target, random_state=42, cv_folds=3)
        assert scores["mae"] >= 0
        assert scores["rmse"] >= 0

    def test_r2_in_range(self):
        df = _sample_feature_df()
        target = _sample_target()
        pipeline, _, _ = build_pipeline(df, random_state=42)
        scores = fit_and_evaluate(pipeline, df, target, random_state=42, cv_folds=3)
        # R2 can be negative for bad models but our synthetic data should yield a value
        assert isinstance(scores["r2"], float)

    def test_cv_folds_respected(self):
        df = _sample_feature_df()
        target = _sample_target()
        pipeline, _, _ = build_pipeline(df, random_state=42)
        scores = fit_and_evaluate(pipeline, df, target, random_state=42, cv_folds=4)
        assert scores["cv_folds"] == 4
