import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from data_cleaning import TARGET_COLUMN, load_csv_resilient, prepare_dataframe
from modeling import build_pipeline, fit_and_evaluate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Entrena modelo de imputacion con splits train/val/test."
  )
  parser.add_argument(
    "--input",
    default="",
    help="CSV de entrada individual. Si se omite, se usa --input-dir.",
  )
  parser.add_argument(
    "--input-dir",
    default="data/processed/pool_train",
    help="Directorio de entrenamiento.",
  )
  parser.add_argument(
    "--input-pattern",
    default="*.csv",
    help="Patron glob para train.",
  )
  parser.add_argument(
    "--val-dir",
    default="",
    help="Directorio de validacion. Si se omite, se salta la evaluacion en validacion.",
  )
  parser.add_argument(
    "--val-pattern",
    default="*.csv",
    help="Patron glob para validacion.",
  )
  parser.add_argument(
    "--test-dir",
    default="",
    help="Directorio de test. Si se omite, se salta la evaluacion en test.",
  )
  parser.add_argument(
    "--test-pattern",
    default="*.csv",
    help="Patron glob para test.",
  )
  parser.add_argument(
    "--period-column",
    default="FECHA_MATR",
    help="Columna de fecha para analizar periodos de train/val/test.",
  )
  parser.add_argument(
    "--period-report-output",
    default="artifacts/metrics/period_split_report.json",
    help="Ruta de salida del reporte de periodos temporales.",
  )
  parser.add_argument(
    "--output",
    default="data/processed/muestra_50k_co2_imputado.csv",
    help="CSV de salida con imputaciones sobre train.",
  )
  parser.add_argument(
    "--sep",
    default=",",
    help="Separador de CSV.",
  )
  parser.add_argument(
    "--encoding",
    default="utf-8",
    help="Codificacion de CSV.",
  )
  parser.add_argument(
    "--model-output",
    default="artifacts/models/co2_model.joblib",
    help="Ruta del modelo entrenado.",
  )
  parser.add_argument(
    "--metrics-output",
    default="artifacts/metrics/co2_metrics.json",
    help="Ruta de metricas JSON.",
  )
  parser.add_argument(
    "--cv-folds",
    type=int,
    default=5,
    help="Numero de folds para CV en train.",
  )
  parser.add_argument(
    "--random-state",
    type=int,
    default=42,
    help="Semilla de reproducibilidad.",
  )
  parser.add_argument(
    "--missing-rate",
    type=float,
    default=0.2,
    help="Porcentaje/proporcion de missing artificial para imputacion en train.",
  )
  parser.add_argument(
    "--device",
    choices=["cuda", "cpu"],
    default="cpu",
    help="Dispositivo para entrenamiento (default: cpu; usa cuda para GPU).",
  )
  parser.add_argument(
    "--simplificado",
    action="store_true",
    help="Genera CSV simplificado con filas imputadas.",
  )
  parser.add_argument(
    "--simplificado-output",
    default="",
    help="Ruta de CSV simplificado.",
  )
  return parser.parse_args()


def _load_input_data(
  input_path: Path | None,
  input_dir: Path,
  input_pattern: str,
  sep: str,
  encoding: str,
) -> tuple[pd.DataFrame, list[Path]]:
  if input_path is not None and input_path.exists():
    df = load_csv_resilient(input_path, sep=sep, encoding=encoding)
    return df, [input_path]

  if not input_dir.exists() or not input_dir.is_dir():
    raise FileNotFoundError(
      f"No existe el directorio {input_dir}"
      if input_path is None
      else f"No existe el archivo {input_path} ni el directorio {input_dir}"
    )

  input_files = sorted(path for path in input_dir.glob(input_pattern) if path.is_file())
  if not input_files:
    raise FileNotFoundError(
      f"No se encontraron ficheros con patron {input_pattern} en {input_dir}"
    )

  dataframes = [load_csv_resilient(path, sep=sep, encoding=encoding) for path in input_files]
  combined_df = pd.concat(dataframes, ignore_index=True)
  return combined_df, input_files


def _build_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
  output_path = Path(args.output)
  simplified_output_path = (
    Path(args.simplificado_output)
    if args.simplificado_output
    else Path("data/processed/datos_simpl.csv")
  )
  model_output_path = Path(args.model_output)
  metrics_output_path = Path(args.metrics_output)
  period_report_output_path = Path(args.period_report_output)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  if args.simplificado:
    simplified_output_path.parent.mkdir(parents=True, exist_ok=True)
  model_output_path.parent.mkdir(parents=True, exist_ok=True)
  metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
  period_report_output_path.parent.mkdir(parents=True, exist_ok=True)

  return (
    output_path,
    simplified_output_path,
    model_output_path,
    metrics_output_path,
    period_report_output_path,
  )


def _normalize_missing_rate(value: float) -> float:
  if value < 0:
    raise ValueError("--missing-rate no puede ser negativo.")
  if value <= 1:
    return value
  if value <= 100:
    return value / 100.0
  raise ValueError("--missing-rate debe estar entre 0 y 1, o entre 0 y 100.")


def _period_stats(df: pd.DataFrame, period_column: str, target_column: str) -> dict:
  stats = {
    "rows_total": int(len(df)),
    "rows_with_target": int(df[target_column].notna().sum()) if target_column in df.columns else 0,
    "period_column": period_column,
    "date_min": None,
    "date_max": None,
    "year_distribution": {},
  }

  if period_column not in df.columns:
    return stats

  parsed = pd.to_datetime(df[period_column], dayfirst=True, errors="coerce")
  valid = parsed.dropna()
  if valid.empty:
    return stats

  stats["date_min"] = valid.min().strftime("%Y-%m-%d")
  stats["date_max"] = valid.max().strftime("%Y-%m-%d")

  year_counts = Counter(valid.dt.year.astype(int).tolist())
  stats["year_distribution"] = {
    str(year): count for year, count in sorted(year_counts.items())
  }
  return stats


def _evaluate_on_dataset(
  model_artifact: dict,
  eval_df: pd.DataFrame,
  dataset_name: str,
) -> dict:
  pipeline = model_artifact["pipeline"]
  feature_columns = model_artifact["feature_columns"]

  if TARGET_COLUMN not in eval_df.columns:
    return {}

  features, target = prepare_dataframe(eval_df)
  valid_mask = target.notna()
  if valid_mask.sum() < 10:
    print(
      f"Advertencia: muy pocas filas con {TARGET_COLUMN} en {dataset_name} ({valid_mask.sum()})"
    )
    return {}

  x_eval = features.loc[valid_mask].reindex(columns=feature_columns)
  y_eval = target.loc[valid_mask]
  y_pred = pipeline.predict(x_eval)

  mae = float(mean_absolute_error(y_eval, y_pred))
  rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
  r2 = float(r2_score(y_eval, y_pred))

  return {
    f"{dataset_name}_mae": mae,
    f"{dataset_name}_rmse": rmse,
    f"{dataset_name}_r2": r2,
    f"{dataset_name}_rows_total": int(len(eval_df)),
    f"{dataset_name}_rows_with_target": int(valid_mask.sum()),
  }


def _train_and_generate_outputs(
  train_df: pd.DataFrame,
  args: argparse.Namespace,
  loaded_inputs: list[Path],
  output_path: Path,
  simplified_output_path: Path,
  model_output_path: Path,
  metrics_output_path: Path,
) -> int:
  if TARGET_COLUMN not in train_df.columns:
    print(f"Error: no existe la columna objetivo {TARGET_COLUMN}")
    return 1

  try:
    missing_rate = _normalize_missing_rate(args.missing_rate)
  except ValueError as exc:
    print(f"Error: {exc}")
    return 1

  features, target = prepare_dataframe(train_df)
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
    masked_indexes = rng.choice(known_indexes.to_numpy(), size=missing_count, replace=False)
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

  print(f"Metricas CV train ({args.cv_folds} folds):")
  print(f"  Backend: {model_backend} (device={args.device})")
  print(f"  MAE:  {scores['mae']:.4f} +/- {scores['mae_std']:.4f}")
  print(f"  RMSE: {scores['rmse']:.4f} +/- {scores['rmse_std']:.4f}")
  print(f"  R2:   {scores['r2']:.4f} +/- {scores['r2_std']:.4f}")

  pipeline.fit(x_known, y_known)

  imputed_values = co2_with_missing.copy()
  if missing_mask.sum() > 0:
    x_missing = features.loc[missing_mask]
    imputed_values.loc[missing_mask] = pipeline.predict(x_missing)

  output_df = train_df.copy()
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
    "rows_total": int(len(train_df)),
    "rows_with_known_target": int(full_target_mask.sum()),
    "rows_with_missing_applied": int(missing_mask.sum()),
    "missing_rate": float(missing_rate),
    "mae": float(scores["mae"]),
    "rmse": float(scores["rmse"]),
    "r2": float(scores["r2"]),
    "mae_std": float(scores["mae_std"]),
    "rmse_std": float(scores["rmse_std"]),
    "r2_std": float(scores["r2_std"]),
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


def _update_json(path: Path, updates: dict) -> None:
  with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)
  payload.update(updates)
  with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)


def _evaluate_split_dataset(
  split_dir: Path | None,
  split_pattern: str,
  split_name: str,
  model_artifact: dict,
  args: argparse.Namespace,
  metrics_output_path: Path,
  period_report: dict,
) -> None:
  if split_dir is None or not split_dir.exists() or not split_dir.is_dir():
    return

  try:
    df, inputs = _load_input_data(
      input_path=None,
      input_dir=split_dir,
      input_pattern=split_pattern,
      sep=args.sep,
      encoding=args.encoding,
    )
  except FileNotFoundError as exc:
    print(f"Advertencia: no se pudo cargar {split_name}: {exc}")
    return

  metrics = _evaluate_on_dataset(model_artifact, df, split_name)
  if metrics:
    print(f"\n=== Evaluacion en {split_name.upper()} ===")
    print(
      f"MAE={metrics[f'{split_name}_mae']:.4f} "
      f"RMSE={metrics[f'{split_name}_rmse']:.4f} "
      f"R2={metrics[f'{split_name}_r2']:.4f}"
    )
    _update_json(
      metrics_output_path,
      {
        **metrics,
        f"{split_name}_dir": str(split_dir),
        f"{split_name}_pattern": split_pattern,
        f"{split_name}_inputs_loaded": [str(p) for p in inputs],
      },
    )

  period_report[split_name] = _period_stats(df, args.period_column, TARGET_COLUMN)


def main() -> int:
  args = parse_args()

  (
    output_path,
    simplified_output_path,
    model_output_path,
    metrics_output_path,
    period_report_output_path,
  ) = _build_output_paths(args)

  input_path = Path(args.input) if args.input else None
  train_dir = Path(args.input_dir)

  try:
    train_df, train_inputs = _load_input_data(
      input_path=input_path,
      input_dir=train_dir,
      input_pattern=args.input_pattern,
      sep=args.sep,
      encoding=args.encoding,
    )
  except FileNotFoundError as exc:
    print(f"Error: {exc}")
    return 1

  train_result = _train_and_generate_outputs(
    train_df=train_df,
    args=args,
    loaded_inputs=train_inputs,
    output_path=output_path,
    simplified_output_path=simplified_output_path,
    model_output_path=model_output_path,
    metrics_output_path=metrics_output_path,
  )
  if train_result != 0:
    return train_result

  period_report = {
    "train": _period_stats(train_df, args.period_column, TARGET_COLUMN),
    "period_column": args.period_column,
  }

  model_artifact = joblib.load(model_output_path)

  val_dir = Path(args.val_dir) if args.val_dir else None
  _evaluate_split_dataset(val_dir, args.val_pattern, "val", model_artifact, args, metrics_output_path, period_report)

  test_dir = Path(args.test_dir) if args.test_dir else None
  _evaluate_split_dataset(test_dir, args.test_pattern, "test", model_artifact, args, metrics_output_path, period_report)

  with open(period_report_output_path, "w", encoding="utf-8") as f:
    json.dump(period_report, f, ensure_ascii=False, indent=2)

  _update_json(
    metrics_output_path,
    {
      "period_column": args.period_column,
      "period_report_output": str(period_report_output_path),
    },
  )

  print("\nReporte temporal generado:")
  print(f"  {period_report_output_path}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
