import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

try:
  from xgboost import XGBRegressor
except ImportError:
  XGBRegressor = None


def build_pipeline(
  feature_df: pd.DataFrame,
  random_state: int,
  device: str,
) -> tuple[Pipeline, list[str], list[str], str]:
  # Detecta dinamicamente tipos para aplicar preprocesado distinto por bloque.
  numeric_columns = [
    c for c in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[c])
  ]
  categorical_columns = [c for c in feature_df.columns if c not in numeric_columns]

  numeric_pipeline = Pipeline(
    steps=[
      ("imputer", SimpleImputer(strategy="median")),
    ]
  )

  categorical_pipeline = Pipeline(
    steps=[
      ("imputer", SimpleImputer(strategy="constant", fill_value="DESCONOCIDO")),
      (
        "encoder",
        OrdinalEncoder(
          handle_unknown="use_encoded_value",
          unknown_value=-1,
        ),
      ),
    ]
  )

  preprocessor = ColumnTransformer(
    transformers=[
      ("num", numeric_pipeline, numeric_columns),
      ("cat", categorical_pipeline, categorical_columns),
    ]
  )

  if device == "cuda":
    if XGBRegressor is None:
      raise ImportError(
        "Falta dependencia xgboost. Instala requirements.txt y vuelve a ejecutar."
      )

    model = XGBRegressor(
      objective="reg:squarederror",
      tree_method="hist",
      device="cuda",
      n_estimators=600,
      learning_rate=0.05,
      max_depth=8,
      min_child_weight=3,
      subsample=0.9,
      colsample_bytree=0.9,
      reg_lambda=1.0,
      random_state=random_state,
      n_jobs=1,
    )
    model_backend = "xgboost-cuda"
  elif device == "cpu":
    model = HistGradientBoostingRegressor(
      random_state=random_state,
      max_iter=350,
      learning_rate=0.05,
      max_depth=8,
      min_samples_leaf=20,
    )
    model_backend = "sklearn-hgbt-cpu"
  else:
    raise ValueError("device debe ser 'cuda' o 'cpu'.")

  pipeline = Pipeline(
    steps=[
      ("preprocessor", preprocessor),
      ("model", model),
    ]
  , memory=None)

  return pipeline, numeric_columns, categorical_columns, model_backend


def fit_and_evaluate(
  pipeline: Pipeline,
  x_known: pd.DataFrame,
  y_known: pd.Series,
  random_state: int,
  cv_folds: int,
  n_jobs_cv: int,
) -> dict[str, float]:
  # Usa validacion cruzada para reducir la varianza de una sola particion.
  cv = KFold(
    n_splits=cv_folds,
    shuffle=True,
    random_state=random_state,
  )

  scores = cross_validate(
    pipeline,
    x_known,
    y_known,
    cv=cv,
    scoring={
      "mae": "neg_mean_absolute_error",
      "rmse": "neg_root_mean_squared_error",
      "r2": "r2",
    },
    n_jobs=n_jobs_cv,
    return_train_score=False,
  )

  mae_values = -scores["test_mae"]
  rmse_values = -scores["test_rmse"]
  r2_values = scores["test_r2"]

  return {
    "mae": float(mae_values.mean()),
    "rmse": float(rmse_values.mean()),
    "r2": float(r2_values.mean()),
    "mae_std": float(mae_values.std(ddof=0)),
    "rmse_std": float(rmse_values.std(ddof=0)),
    "r2_std": float(r2_values.std(ddof=0)),
    "cv_folds": int(cv_folds),
  }
