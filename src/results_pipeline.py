from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {"imputer", "missing_rate", "mse", "mae"}


def _validate_results(df: pd.DataFrame) -> None:
  missing = REQUIRED_COLUMNS - set(df.columns)
  if missing:
    cols = ", ".join(sorted(missing))
    raise ValueError(f"Faltan columnas requeridas en results: {cols}")


def _prepare_output_dirs(base_dir: Path) -> dict[str, Path]:
  tables_dir = base_dir / "tables"
  plots_dir = base_dir / "plots"
  logs_dir = base_dir / "logs"

  for directory in (base_dir, tables_dir, plots_dir, logs_dir):
    directory.mkdir(parents=True, exist_ok=True)

  return {
    "base": base_dir,
    "tables": tables_dir,
    "plots": plots_dir,
    "logs": logs_dir,
  }


def save_tables(df: pd.DataFrame, tables_dir: Path) -> tuple[Path, Path]:
  """Guarda tabla plana y tabla pivote de MSE."""
  flat_path = tables_dir / "experiment_results.csv"
  pivot_path = tables_dir / "mse_comparison.csv"

  df_sorted = df.sort_values(["imputer", "missing_rate"]).reset_index(drop=True)
  df_sorted.to_csv(flat_path, index=False)

  pivot_df = df.pivot_table(
    index="imputer",
    columns="missing_rate",
    values="mse",
    aggfunc="mean",
  ).sort_index(axis=0).sort_index(axis=1)
  pivot_df.to_csv(pivot_path)

  return flat_path, pivot_path


def _line_plot_by_imputer(
  df: pd.DataFrame,
  metric: str,
  output_path: Path,
  title: str,
  y_label: str,
) -> None:
  plt.figure(figsize=(10, 6))

  for imputer_name, group in df.groupby("imputer"):
    group_sorted = group.sort_values("missing_rate")
    plt.plot(
      group_sorted["missing_rate"],
      group_sorted[metric],
      marker="o",
      linewidth=2,
      label=imputer_name,
    )

  plt.title(title)
  plt.xlabel("Missing Rate")
  plt.ylabel(y_label)
  plt.grid(True, alpha=0.3)
  plt.legend(title="Imputer")
  plt.tight_layout()
  plt.savefig(output_path, dpi=150)
  plt.close()


def generate_plots(df: pd.DataFrame, plots_dir: Path) -> tuple[Path, Path]:
  """Genera graficas de MSE y MAE en funcion del missing_rate."""
  mse_path = plots_dir / "mse_vs_missing_rate.png"
  mae_path = plots_dir / "mae_vs_missing_rate.png"

  _line_plot_by_imputer(
    df=df,
    metric="mse",
    output_path=mse_path,
    title="MSE vs Missing Rate por Imputador",
    y_label="MSE",
  )
  _line_plot_by_imputer(
    df=df,
    metric="mae",
    output_path=mae_path,
    title="MAE vs Missing Rate por Imputador",
    y_label="MAE",
  )

  return mse_path, mae_path


def write_log(df: pd.DataFrame, logs_dir: Path) -> Path:
  """Escribe un resumen de ejecucion en texto plano."""
  log_path = logs_dir / "experiment_log.txt"

  timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  imputers = sorted(df["imputer"].astype(str).unique().tolist())
  missing_rates = sorted(df["missing_rate"].astype(float).unique().tolist())
  total_experiments = int(len(df))

  content = [
    "=== Experiment Log ===",
    f"Timestamp: {timestamp}",
    f"Imputers evaluados: {', '.join(imputers)}",
    "Missing rates utilizados: "
    + ", ".join(f"{rate:.2f}" for rate in missing_rates),
    f"Numero total de experimentos: {total_experiments}",
  ]

  log_path.write_text("\n".join(content) + "\n", encoding="utf-8")
  return log_path


def run_results_pipeline(results: Iterable[dict], base_results_dir: str | Path = "results") -> dict[str, Path]:
  """
  Ejecuta pipeline de post-procesado de resultados de imputacion.

  Params:
    results: iterable de dicts con claves imputer, missing_rate, mse, mae.
    base_results_dir: carpeta raiz de salida (por defecto: results).

  Returns:
    Diccionario con rutas de artefactos generados.
  """
  df = pd.DataFrame(list(results))
  if df.empty:
    raise ValueError("La lista de resultados esta vacia.")

  _validate_results(df)

  dirs = _prepare_output_dirs(Path(base_results_dir))
  experiment_csv, pivot_csv = save_tables(df, dirs["tables"])
  mse_plot, mae_plot = generate_plots(df, dirs["plots"])
  log_file = write_log(df, dirs["logs"])

  return {
    "experiment_results_csv": experiment_csv,
    "mse_comparison_csv": pivot_csv,
    "mse_plot": mse_plot,
    "mae_plot": mae_plot,
    "log_file": log_file,
  }


if __name__ == "__main__":
  example_results = [
    {"imputer": "mean", "missing_rate": 0.05, "mse": 0.12, "mae": 0.21},
    {"imputer": "knn", "missing_rate": 0.05, "mse": 0.08, "mae": 0.16},
    {"imputer": "iterative", "missing_rate": 0.05, "mse": 0.06, "mae": 0.13},
    {"imputer": "mean", "missing_rate": 0.10, "mse": 0.16, "mae": 0.25},
    {"imputer": "knn", "missing_rate": 0.10, "mse": 0.11, "mae": 0.19},
    {"imputer": "iterative", "missing_rate": 0.10, "mse": 0.09, "mae": 0.17},
  ]

  outputs = run_results_pipeline(example_results)
  print("Artefactos generados:")
  for name, path in outputs.items():
    print(f"- {name}: {path}")
