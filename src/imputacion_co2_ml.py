import argparse
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from data_cleaning import TARGET_COLUMN, load_csv_resilient, prepare_dataframe
from modeling import build_pipeline, fit_and_evaluate


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Entrena un modelo para imputar EMISIONES_CO2 faltantes."
  )
  parser.add_argument(
    "--input",
    default="",
    help="CSV de entrada individual. Si se omite, se usa --input-dir.",
  )
  parser.add_argument(
    "--input-dir",
    default="data/processed/pool",
    help="Directorio con multiples CSV de entrada. Se concatenan todos los ficheros encontrados.",
  )
  parser.add_argument(
    "--input-pattern",
    default="*.csv",
    help="Patron glob para filtrar ficheros dentro de --input-dir.",
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
  parser.add_argument(
    "--device",
    choices=["cuda", "cpu"],
    default="cuda",
    help="Dispositivo para entrenar el modelo: cuda (GPU) o cpu.",
  )
  parser.add_argument(
    "--simplificado",
    action="store_true",
    help=(
      "Genera un segundo CSV simplificado con solo filas donde se aplico "
      "missing rate y su imputacion."
    ),
  )
  parser.add_argument(
    "--simplificado-output",
    default="",
    help="Ruta del CSV simplificado. Si se omite, usa data/processed/datos_simpl.csv.",
  )
  return parser.parse_args()


def _load_input_data(
  input_path: Path,
  input_dir: Path,
  input_pattern: str,
  sep: str,
  encoding: str,
) -> tuple[pd.DataFrame, list[Path]]:
  if input_path.exists():
    df = load_csv_resilient(
      input_path,
      sep=sep,
      encoding=encoding,
    )
    return df, [input_path]

  if not input_dir.exists() or not input_dir.is_dir():
    raise FileNotFoundError(
      f"No existe el archivo {input_path} ni el directorio de entrada {input_dir}"
    )

  input_files = sorted(
    path for path in input_dir.glob(input_pattern) if path.is_file()
  )
  if not input_files:
    raise FileNotFoundError(
      f"No se encontraron ficheros con patron {input_pattern} en {input_dir}"
    )

  dataframes = [
    load_csv_resilient(path, sep=sep, encoding=encoding) for path in input_files
  ]
  combined_df = pd.concat(dataframes, ignore_index=True)
  return combined_df, input_files


def _build_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
  output_path = Path(args.output)
  simplified_output_path = (
    Path(args.simplificado_output)
    if args.simplificado_output
    else Path("data/processed/datos_simpl.csv")
  )
  model_output_path = Path(args.model_output)
  metrics_output_path = Path(args.metrics_output)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  if args.simplificado:
    simplified_output_path.parent.mkdir(parents=True, exist_ok=True)
  model_output_path.parent.mkdir(parents=True, exist_ok=True)
  metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

  return output_path, simplified_output_path, model_output_path, metrics_output_path


def _train_and_generate_outputs(
  df: pd.DataFrame,
  args: argparse.Namespace,
  loaded_inputs: list[Path],
  output_path: Path,
  simplified_output_path: Path,
  model_output_path: Path,
  metrics_output_path: Path,
) -> int:
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

  co2_complete = target.copy()
  co2_with_missing = co2_complete.copy()

  rng = np.random.default_rng(args.random_state)
  known_indexes = co2_complete.index[full_target_mask]
  missing_count = int(round(len(known_indexes) * missing_rate))
  missing_count = min(missing_count, len(known_indexes))

  if missing_count > 0:
    masked_indexes = rng.choice(
      known_indexes.to_numpy(), size=missing_count, replace=False
    )
    co2_with_missing.loc[masked_indexes] = np.nan  # type: ignore

  known_mask = co2_with_missing.notna()
  missing_mask = co2_with_missing.isna()

  if known_mask.sum() < 100:
    print("Error: hay muy pocas filas con EMISIONES_CO2 para entrenar.")
    return 1

  if args.cv_folds < 2:
    print("Error: --cv-folds debe ser al menos 2.")
    return 1

  try:
    pipeline, numeric_columns, categorical_columns, model_backend = build_pipeline(
      features,
      random_state=args.random_state,
      device=args.device,
    )
  except ImportError as exc:
    print(f"Error: {exc}")
    return 1

  n_jobs_cv = 1 if args.device == "cuda" else -1

  x_known = features.loc[known_mask]
  y_known = co2_with_missing.loc[known_mask]

  scores = fit_and_evaluate(
    pipeline,
    x_known,
    y_known,
    random_state=args.random_state,
    cv_folds=args.cv_folds,
    n_jobs_cv=n_jobs_cv,
  )
  mae = scores["mae"]
  rmse = scores["rmse"]
  r2 = scores["r2"]
  mae_std = scores["mae_std"]
  rmse_std = scores["rmse_std"]
  r2_std = scores["r2_std"]

  print(f"Metricas de validacion cruzada ({args.cv_folds} folds):")
  print(f"  Backend: {model_backend} (device={args.device})")
  print(f"  MAE:  {mae:.4f} +/- {mae_std:.4f}")
  print(f"  RMSE: {rmse:.4f} +/- {rmse_std:.4f}")
  print(f"  R2:   {r2:.4f} +/- {r2_std:.4f}")

  pipeline.fit(x_known, y_known)

  imputed_values = co2_with_missing.copy()
  if missing_mask.sum() > 0:
    x_missing = features.loc[missing_mask]
    imputed_values.loc[missing_mask] = pipeline.predict(x_missing)

  output_df = df.copy()
  output_df["EMISIONES_CO2_COMPLETA"] = co2_complete
  output_df["EMISIONES_CO2_CON_MISSING_PCT"] = co2_with_missing
  output_df["EMISIONES_CO2_IMPUTADA"] = imputed_values.round(3)

  output_df.to_csv(output_path, index=False, sep=args.sep, encoding=args.encoding)

  simplified_rows = 0
  if args.simplificado:
    simplified_df = output_df.loc[
      missing_mask,
      [
        "EMISIONES_CO2_COMPLETA",
        "EMISIONES_CO2_CON_MISSING_PCT",
        "EMISIONES_CO2_IMPUTADA",
      ],
    ].copy()
    simplified_rows = int(len(simplified_df))
    simplified_df.to_csv(
      simplified_output_path,
      index=False,
      sep=args.sep,
      encoding=args.encoding,
    )

  model_artifact = {
    "pipeline": pipeline,
    "feature_columns": list(features.columns),
    "numeric_columns": numeric_columns,
    "categorical_columns": categorical_columns,
    "target_column": TARGET_COLUMN,
    "separator": args.sep,
    "encoding": args.encoding,
    "model_backend": model_backend,
    "device": args.device,
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
    "model_backend": model_backend,
    "device": args.device,
    "cv_n_jobs": int(n_jobs_cv),
    "input": str(args.input) if args.input else "",
    "input_dir": str(args.input_dir) if not args.input else "",
    "input_pattern": args.input_pattern if not args.input else "",
    "inputs_loaded": [str(path) for path in loaded_inputs],
    "input_files_count": int(len(loaded_inputs)),
    "output": str(output_path),
    "simplified_output": str(simplified_output_path) if args.simplificado else "",
    "rows_in_simplified_output": int(simplified_rows),
    "model_output": str(model_output_path),
  }

  with open(metrics_output_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

  print("Artefactos generados:")
  print(f"  CSV imputado: {output_path}")
  if args.simplificado:
    print(f"  CSV simplif.: {simplified_output_path}")
  print(f"  Modelo:       {model_output_path}")
  print(f"  Metricas:     {metrics_output_path}")

  return 0


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

  input_path = Path(args.input) if args.input else Path("__missing_input__")
  input_dir = Path(args.input_dir)
  output_path, simplified_output_path, model_output_path, metrics_output_path = _build_output_paths(args)

  try:
    df, loaded_inputs = _load_input_data(
      input_path=input_path,
      input_dir=input_dir,
      input_pattern=args.input_pattern,
      sep=args.sep,
      encoding=args.encoding,
    )
  except FileNotFoundError as exc:
    print(f"Error: {exc}")
    return 1

  return _train_and_generate_outputs(
    df=df,
    args=args,
    loaded_inputs=loaded_inputs,
    output_path=output_path,
    simplified_output_path=simplified_output_path,
    model_output_path=model_output_path,
    metrics_output_path=metrics_output_path,
  )


if __name__ == "__main__":
  raise SystemExit(main())
