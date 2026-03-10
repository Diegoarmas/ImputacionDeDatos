import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


TARGET_COLUMN = "EMISIONES_CO2"
DATE_COLUMNS = ["FECHA_MATR", "FEC_PRIM_MATR"]
KNOWN_NUMERIC_COLUMNS = [
    "TARA",
    "PESO_MAX",
    "MOM",
    "MMTA",
    "CILINDRADA",
    "POTENCIA",
    "KW",
    "CONSUMO",
    "AUTONOMIA",
    "DISTANCIA_EJES",
    "EJE_ANTERIOR",
    "EJE_POSTERIOR",
    "PLAZAS",
    "PLAZAS_MAX",
    "PLAZAS_PIE",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un modelo para imputar EMISIONES_CO2 faltantes."
    )
    parser.add_argument(
        "--input",
        default="data/processed/muestra_50k.csv",
        help="CSV de entrada.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/muestra_50k_co2_imputado.csv",
        help="CSV de salida con imputaciones.",
    )
    parser.add_argument(
        "--sep",
        default=",",
        help="Separador del CSV de entrada y salida.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Codificacion del CSV de entrada y salida.",
    )
    parser.add_argument(
        "--model-output",
        default="artifacts/models/co2_model.joblib",
        help="Ruta para guardar el modelo entrenado.",
    )
    parser.add_argument(
        "--metrics-output",
        default="artifacts/metrics/co2_metrics.json",
        help="Ruta para guardar metricas en JSON.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporcion de validacion para metricas.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla para reproducibilidad.",
    )
    return parser.parse_args()


def to_float_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = df.copy()

    target = to_float_series(data[TARGET_COLUMN])
    data = data.drop(columns=[TARGET_COLUMN])

    for col in DATE_COLUMNS:
        if col in data.columns:
            parsed = pd.to_datetime(data[col], format="%d/%m/%Y", errors="coerce")
            data[f"{col}_YEAR"] = parsed.dt.year
            data[f"{col}_MONTH"] = parsed.dt.month
            data = data.drop(columns=[col])

    numeric_columns = [c for c in KNOWN_NUMERIC_COLUMNS if c in data.columns]
    for col in numeric_columns:
        data[col] = to_float_series(data[col])

    for col in data.columns:
        if col not in numeric_columns and not col.endswith("_YEAR") and not col.endswith("_MONTH"):
            data[col] = data[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})

    return data, target


def build_pipeline(feature_df: pd.DataFrame) -> tuple[Pipeline, list[str], list[str]]:
    numeric_columns = [
        c
        for c in feature_df.columns
        if pd.api.types.is_numeric_dtype(feature_df[c])
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


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_output_path = Path(args.model_output)
    metrics_output_path = Path(args.metrics_output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: no existe el archivo {input_path}")
        return 1

    df = pd.read_csv(
        input_path,
        sep=args.sep,
        encoding=args.encoding,
        dtype=str,
        low_memory=False,
    )

    if TARGET_COLUMN not in df.columns:
        print(f"Error: no existe la columna objetivo {TARGET_COLUMN}")
        return 1

    features, target = prepare_dataframe(df)

    known_mask = target.notna()
    missing_mask = target.isna()

    if known_mask.sum() < 100:
        print("Error: hay muy pocas filas con EMISIONES_CO2 para entrenar.")
        return 1

    pipeline, numeric_columns, categorical_columns = build_pipeline(features)

    x_known = features.loc[known_mask]
    y_known = target.loc[known_mask]

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_known,
        y_known,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_valid)

    mae = mean_absolute_error(y_valid, y_pred)
    rmse = mean_squared_error(y_valid, y_pred) ** 0.5
    r2 = r2_score(y_valid, y_pred)

    print("Metricas de validacion:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")

    # Reentrenamos con todas las filas conocidas antes de imputar.
    pipeline.fit(x_known, y_known)

    imputed_values = target.copy()
    if missing_mask.sum() > 0:
        x_missing = features.loc[missing_mask]
        imputed_values.loc[missing_mask] = pipeline.predict(x_missing)

    output_df = df.copy()
    output_df["EMISIONES_CO2_NUM"] = target
    output_df["EMISIONES_CO2_IMPUTADA"] = imputed_values.round(3)
    output_df["EMISIONES_CO2_ESTIMADA_POR_ML"] = missing_mask.astype(int)

    output_df.to_csv(output_path, index=False, sep=args.sep, encoding=args.encoding)

    model_artifact = {
        "pipeline": pipeline,
        "feature_columns": list(features.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "target_column": TARGET_COLUMN,
        "separator": args.sep,
        "encoding": args.encoding,
    }
    joblib.dump(model_artifact, model_output_path)

    metrics = {
        "rows_total": int(len(df)),
        "rows_with_known_target": int(known_mask.sum()),
        "rows_with_missing_target": int(missing_mask.sum()),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "input": str(args.input),
        "output": str(output_path),
        "model_output": str(model_output_path),
    }

    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Artefactos generados:")
    print(f"  CSV imputado: {output_path}")
    print(f"  Modelo:       {model_output_path}")
    print(f"  Metricas:     {metrics_output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
