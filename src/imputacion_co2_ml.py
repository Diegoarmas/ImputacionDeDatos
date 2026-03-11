import argparse
import json
from pathlib import Path

import joblib
from data_cleaning import TARGET_COLUMN, load_csv_resilient, prepare_dataframe
from modeling import build_pipeline, fit_and_evaluate


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

  df = load_csv_resilient(
    input_path,
    sep=args.sep,
    encoding=args.encoding,
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

  scores = fit_and_evaluate(
    pipeline,
    x_known,
    y_known,
    test_size=args.test_size,
    random_state=args.random_state,
  )
  mae = scores["mae"]
  rmse = scores["rmse"]
  r2 = scores["r2"]

  print("Metricas de validacion:")
  print(f"  MAE:  {mae:.4f}")
  print(f"  RMSE: {rmse:.4f}")
  print(f"  R2:   {r2:.4f}")

  # Reentrenamos con todas las filas conocidas para maximizar informacion.
  pipeline.fit(x_known, y_known)

  imputed_values = target.copy()
  if missing_mask.sum() > 0:
    x_missing = features.loc[missing_mask]
    imputed_values.loc[missing_mask] = pipeline.predict(x_missing)

  # Conserva datos originales y anade columnas de trazabilidad de imputacion.
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
