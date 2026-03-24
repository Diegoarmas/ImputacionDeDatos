import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from data_cleaning import TARGET_COLUMN, load_csv_resilient, prepare_dataframe
from modeling import build_pipeline, fit_and_evaluate


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Entrena un modelo para imputar EMISIONES_CO2 faltantes."
  )
  parser.add_argument(
    "--input",
    default="data/processed/muestra_50k_con_co2.csv",
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
    help="Parametro legado (sin uso): ahora se evalua con validacion cruzada.",
  )
  parser.add_argument(
    "--cv-folds",
    type=int,
    default=5,
    help="Numero de folds para validacion cruzada.",
  )
  parser.add_argument(
    "--random-state",
    type=int,
    default=42,
    help="Semilla para reproducibilidad.",
  )
  parser.add_argument(
    "--missing-rate",
    type=float,
    default=0.2,
    help=(
      "Porcentaje/proporcion de EMISIONES_CO2 a ocultar para imputar. "
      "Acepta 0.2 o 20 para 20%."
    ),
  )
  return parser.parse_args()


def _normalize_missing_rate(value: float) -> float:
  if value < 0:
    raise ValueError("--missing-rate no puede ser negativo.")
  if value <= 1:
    return value
  if value <= 100:
    return value / 100.0
  raise ValueError("--missing-rate debe estar entre 0 y 1, o entre 0 y 100.")


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

  try:
    missing_rate = _normalize_missing_rate(args.missing_rate)
  except ValueError as exc:
    print(f"Error: {exc}")
    return 1

  features, target = prepare_dataframe(df)

  full_target_mask = target.notna()
  if full_target_mask.sum() < 100:
    print("Error: hay muy pocas filas con EMISIONES_CO2 para entrenar.")
    return 1

  # Aplica missing artificial sobre la serie completa de CO2 conocida.
  co2_complete = target.copy()
  co2_with_missing = co2_complete.copy()

  rng = np.random.default_rng(args.random_state)
  known_indexes = co2_complete.index[full_target_mask]
  missing_count = int(round(len(known_indexes) * missing_rate))
  missing_count = min(missing_count, len(known_indexes))

  if missing_count > 0:
    masked_indexes = rng.choice(known_indexes.to_numpy(), size=missing_count, replace=False)
    co2_with_missing.loc[masked_indexes] = np.nan

  known_mask = co2_with_missing.notna()
  missing_mask = co2_with_missing.isna()

  if known_mask.sum() < 100:
    print("Error: hay muy pocas filas con EMISIONES_CO2 para entrenar.")
    return 1

  if args.cv_folds < 2:
    print("Error: --cv-folds debe ser al menos 2.")
    return 1

  pipeline, numeric_columns, categorical_columns = build_pipeline(
    features,
    random_state=args.random_state,
  )

  x_known = features.loc[known_mask]
  y_known = co2_with_missing.loc[known_mask]

  scores = fit_and_evaluate(
    pipeline,
    x_known,
    y_known,
    random_state=args.random_state,
    cv_folds=args.cv_folds,
  )
  mae = scores["mae"]
  rmse = scores["rmse"]
  r2 = scores["r2"]
  mae_std = scores["mae_std"]
  rmse_std = scores["rmse_std"]
  r2_std = scores["r2_std"]

  print(f"Metricas de validacion cruzada ({args.cv_folds} folds):")
  print(f"  MAE:  {mae:.4f} +/- {mae_std:.4f}")
  print(f"  RMSE: {rmse:.4f} +/- {rmse_std:.4f}")
  print(f"  R2:   {r2:.4f} +/- {r2_std:.4f}")

  # Reentrenamos con todas las filas conocidas para maximizar informacion.
  pipeline.fit(x_known, y_known)

  imputed_values = co2_with_missing.copy()
  if missing_mask.sum() > 0:
    x_missing = features.loc[missing_mask]
    imputed_values.loc[missing_mask] = pipeline.predict(x_missing)

  # Conserva datos originales y agrega columnas solicitadas de objetivo/imputacion.
  output_df = df.copy()
  output_df["EMISIONES_CO2_COMPLETA"] = co2_complete
  output_df["EMISIONES_CO2_CON_MISSING_PCT"] = co2_with_missing
  output_df["EMISIONES_CO2_IMPUTADA"] = imputed_values.round(3)

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
    "rows_with_known_target": int(full_target_mask.sum()),
    "rows_with_missing_applied": int(missing_mask.sum()),
    "missing_rate": float(missing_rate),
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "mae_std": float(mae_std),
    "rmse_std": float(rmse_std),
    "r2_std": float(r2_std),
    "cv_folds": int(args.cv_folds),
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
