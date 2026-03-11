import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def build_pipeline(feature_df: pd.DataFrame) -> tuple[Pipeline, list[str], list[str]]:
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

  model = HistGradientBoostingRegressor(
    random_state=42,
    max_iter=350,
    learning_rate=0.05,
    max_depth=8,
    min_samples_leaf=20,
  )

  pipeline = Pipeline(
    steps=[
      ("preprocessor", preprocessor),
      ("model", model),
    ]
  )

  return pipeline, numeric_columns, categorical_columns


def fit_and_evaluate(
  pipeline: Pipeline,
  x_known: pd.DataFrame,
  y_known: pd.Series,
  test_size: float,
  random_state: int,
) -> dict[str, float]:
  # Evalua calidad antes de imputar para tener metrica real del modelo.
  x_train, x_valid, y_train, y_valid = train_test_split(
    x_known,
    y_known,
    test_size=test_size,
    random_state=random_state,
  )

  pipeline.fit(x_train, y_train)
  y_pred = pipeline.predict(x_valid)

  mae = mean_absolute_error(y_valid, y_pred)
  rmse = mean_squared_error(y_valid, y_pred) ** 0.5
  r2 = r2_score(y_valid, y_pred)

  return {
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
  }
