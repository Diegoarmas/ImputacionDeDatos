"""
Compara multiples tecnicas de imputacion para EMISIONES_CO2.

Tecnicas evaluadas:
  - mean/median: DummyRegressor (baseline estadistico sin features)
  - linear:      LinearRegression
  - knn:         KNeighborsRegressor (KNN sobre el espacio de features)
  - rf:          RandomForestRegressor
  - hgbt:        HistGradientBoostingRegressor (modelo actual en produccion)

Salidas:
  - JSON con metricas por tecnica
  - CSV tabla resumen
  - PNG grafico comparativo (MAE, RMSE, R2)
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# Importaciones locales — mismo modulo de limpieza que el pipeline principal.
sys.path.insert(0, str(Path(__file__).parent))
from data_cleaning import TARGET_COLUMN, load_csv_resilient, prepare_dataframe


# ---------------------------------------------------------------------------
# Definicion de tecnicas
# ---------------------------------------------------------------------------

def _build_preprocessor(feature_df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
  numeric_cols = [c for c in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[c])]
  categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

  num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
  cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="DESCONOCIDO")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
  ])

  preprocessor = ColumnTransformer([
    ("num", num_pipe, numeric_cols),
    ("cat", cat_pipe, categorical_cols),
  ])
  return preprocessor, numeric_cols, categorical_cols


def _build_techniques(feature_df: pd.DataFrame, random_state: int, knn_neighbors: int) -> dict[str, Pipeline]:
  def _pre():
    # Cada tecnica necesita su propia instancia del preprocessor.
    p, _, _ = _build_preprocessor(feature_df)
    return p

  return {
    "mean": Pipeline([("pre", _pre()), ("model", DummyRegressor(strategy="mean"))]),
    "median": Pipeline([("pre", _pre()), ("model", DummyRegressor(strategy="median"))]),
    "linear": Pipeline([("pre", _pre()), ("model", LinearRegression())]),
    "knn": Pipeline([("pre", _pre()), ("model", KNeighborsRegressor(n_neighbors=knn_neighbors))]),
    "rf": Pipeline([
      ("pre", _pre()),
      ("model", RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
      )),
    ]),
    "hgbt": Pipeline([
      ("pre", _pre()),
      ("model", HistGradientBoostingRegressor(
        max_iter=350,
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=20,
        random_state=random_state,
      )),
    ]),
  }


# ---------------------------------------------------------------------------
# Evaluacion
# ---------------------------------------------------------------------------

def _evaluate_technique(
  name: str,
  pipeline: Pipeline,
  x: pd.DataFrame,
  y: pd.Series,
  cv_folds: int,
  random_state: int,
) -> dict:
  print(f"  Evaluando: {name} ...", end=" ", flush=True)
  cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
  # KNN con n_jobs=-1 clona el dataset entero por cada job → OOM en datasets grandes.
  cv_n_jobs = 1 if name == "knn" else -1
  scores = cross_validate(
    pipeline,
    x,
    y,
    cv=cv,
    scoring={
      "mae": "neg_mean_absolute_error",
      "rmse": "neg_root_mean_squared_error",
      "r2": "r2",
    },
    n_jobs=cv_n_jobs,
    return_train_score=False,
  )
  mae_vals = -scores["test_mae"]
  rmse_vals = -scores["test_rmse"]
  r2_vals = scores["test_r2"]

  result = {
    "technique": name,
    "mae": float(mae_vals.mean()),
    "mae_std": float(mae_vals.std(ddof=0)),
    "rmse": float(rmse_vals.mean()),
    "rmse_std": float(rmse_vals.std(ddof=0)),
    "r2": float(r2_vals.mean()),
    "r2_std": float(r2_vals.std(ddof=0)),
    "cv_folds": cv_folds,
  }
  print(f"MAE={result['mae']:.4f}  RMSE={result['rmse']:.4f}  R2={result['r2']:.4f}")
  return result


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def _load_data(
  input_path: str,
  input_dir: str,
  input_pattern: str,
  sep: str,
  encoding: str,
  max_files: int,
) -> pd.DataFrame:
  if input_path:
    p = Path(input_path)
    if not p.exists():
      raise FileNotFoundError(f"No existe el archivo: {p}")
    return load_csv_resilient(p, sep=sep, encoding=encoding)

  d = Path(input_dir)
  if not d.exists():
    raise FileNotFoundError(f"No existe el directorio: {d}")

  files = sorted(f for f in d.glob(input_pattern) if f.is_file())
  if not files:
    raise FileNotFoundError(f"Sin ficheros con patron '{input_pattern}' en {d}")
  if max_files > 0:
    files = files[:max_files]

  frames = []
  for f in files:
    frames.append(load_csv_resilient(f, sep=sep, encoding=encoding))
  combined = pd.concat(frames, ignore_index=True)
  del frames
  gc.collect()
  return combined


# ---------------------------------------------------------------------------
# Salidas: tabla, plot, JSON
# ---------------------------------------------------------------------------

def _print_table(results: list[dict]) -> None:
  print("\n" + "=" * 70)
  print(f"{'Tecnica':<12}  {'MAE':>10}  {'±':>8}  {'RMSE':>10}  {'±':>8}  {'R2':>8}  {'±':>6}")
  print("-" * 70)
  for r in sorted(results, key=lambda x: x["mae"]):
    print(
      f"{r['technique']:<12}  "
      f"{r['mae']:>10.4f}  {r['mae_std']:>8.4f}  "
      f"{r['rmse']:>10.4f}  {r['rmse_std']:>8.4f}  "
      f"{r['r2']:>8.4f}  {r['r2_std']:>6.4f}"
    )
  print("=" * 70)
  best = min(results, key=lambda x: x["mae"])
  print(f"\nMejor tecnica (menor MAE): {best['technique']}  (MAE={best['mae']:.4f})")


def _save_csv(results: list[dict], path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  df = pd.DataFrame(results).sort_values("mae")
  df.to_csv(path, index=False)
  print(f"  Tabla CSV: {path}")


def _save_json(results: list[dict], path: Path, meta: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = {"meta": meta, "results": results}
  with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
  print(f"  JSON:      {path}")


def _save_plot(results: list[dict], path: Path) -> None:
  import matplotlib.pyplot as plt
  import matplotlib.ticker as ticker

  path.parent.mkdir(parents=True, exist_ok=True)

  df = pd.DataFrame(results).sort_values("mae").reset_index(drop=True)
  techniques = df["technique"].tolist()
  x = np.arange(len(techniques))

  fig, axes = plt.subplots(1, 3, figsize=(14, 5))
  fig.suptitle("Comparacion de tecnicas de imputacion — EMISIONES_CO2", fontsize=13)

  metrics = [
    ("mae", "MAE (menor = mejor)", "steelblue"),
    ("rmse", "RMSE (menor = mejor)", "darkorange"),
    ("r2", "R² (mayor = mejor)", "seagreen"),
  ]

  for ax, (metric, ylabel, color) in zip(axes, metrics):
    vals = df[metric].values
    errs = df[f"{metric}_std"].values
    bars = ax.bar(x, vals, yerr=errs, color=color, alpha=0.8, capsize=4, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(techniques, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, vals):
      ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + (max(vals) * 0.02),
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=7.5,
      )

  fig.tight_layout()
  fig.savefig(path, dpi=150, bbox_inches="tight")
  plt.close(fig)
  print(f"  Plot:      {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Compara tecnicas de imputacion para EMISIONES_CO2.")
  p.add_argument("--input", default="", help="CSV de entrada individual.")
  p.add_argument("--input-dir", default="data/processed/pool_train", help="Directorio de CSVs de entrenamiento.")
  p.add_argument("--input-pattern", default="*.csv", help="Patron glob.")
  p.add_argument("--sep", default=",")
  p.add_argument("--encoding", default="utf-8")
  p.add_argument("--max-files", type=int, default=10, help="Max ficheros a cargar (0=sin limite).")
  p.add_argument("--cv-folds", type=int, default=5)
  p.add_argument("--random-state", type=int, default=42)
  p.add_argument("--knn-neighbors", type=int, default=5, help="K para KNeighborsRegressor.")
  p.add_argument("--missing-rate", type=float, default=0.2, help="Porcentaje de missing artificial (0-1).")
  p.add_argument("--json-output", default="artifacts/metrics/imputer_comparison.json")
  p.add_argument("--csv-output", default="results/tables/imputer_comparison.csv")
  p.add_argument("--plot-output", default="results/plots/imputer_comparison.png")
  p.add_argument(
    "--techniques",
    nargs="+",
    default=["mean", "median", "linear", "knn", "rf", "hgbt"],
    choices=["mean", "median", "linear", "knn", "rf", "hgbt"],
    help="Subconjunto de tecnicas a evaluar.",
  )
  return p.parse_args()


def main() -> int:
  args = parse_args()

  print("Cargando datos...")
  try:
    df = _load_data(
      input_path=args.input,
      input_dir=args.input_dir,
      input_pattern=args.input_pattern,
      sep=args.sep,
      encoding=args.encoding,
      max_files=args.max_files,
    )
  except FileNotFoundError as exc:
    print(f"Error: {exc}")
    return 1

  print(f"  Filas cargadas: {len(df):,}")

  if TARGET_COLUMN not in df.columns:
    print(f"Error: no existe la columna '{TARGET_COLUMN}'.")
    return 1

  features, target = prepare_dataframe(df)
  del df
  gc.collect()

  valid_mask = target.notna() & (target > 0)
  print(f"  Filas con CO2 conocido (>0): {valid_mask.sum():,}")

  if valid_mask.sum() < 100:
    print("Error: demasiado pocas filas con target conocido.")
    return 1

  missing_rate = args.missing_rate
  if missing_rate < 0 or missing_rate > 1:
    print("Error: --missing-rate debe estar entre 0 y 1.")
    return 1

  rng = np.random.default_rng(args.random_state)
  known_idx = target.index[valid_mask]
  n_mask = int(round(len(known_idx) * missing_rate))
  if n_mask > 0:
    masked_idx = rng.choice(known_idx.to_numpy(), size=n_mask, replace=False)
    target_cv = target.copy()
    target_cv.loc[masked_idx] = np.nan
  else:
    target_cv = target.copy()

  train_mask = target_cv.notna() & (target_cv > 0)
  x_train = features.loc[train_mask]
  y_train = target_cv.loc[train_mask]

  print(f"\nEvaluando con CV ({args.cv_folds} folds), missing_rate={missing_rate:.0%}...")
  all_techniques = _build_techniques(features, args.random_state, args.knn_neighbors)
  selected = {k: v for k, v in all_techniques.items() if k in args.techniques}

  with mlflow.start_run():
    mlflow.log_params(
      {
        "missing_rate": missing_rate,
        "knn_neighbors": args.knn_neighbors,
        "cv_folds": args.cv_folds,
        "random_state": args.random_state,
        "techniques": ",".join(args.techniques),
        "input_dir": args.input_dir,
      }
    )

    results = []
    for name, pipeline in selected.items():
      r = _evaluate_technique(name, pipeline, x_train, y_train, args.cv_folds, args.random_state)
      results.append(r)
      mlflow.log_metrics(
        {
          f"{name}_mae": r["mae"],
          f"{name}_rmse": r["rmse"],
          f"{name}_r2": r["r2"],
        }
      )
      gc.collect()

    _print_table(results)

    meta = {
      "input_dir": args.input_dir,
      "cv_folds": args.cv_folds,
      "random_state": args.random_state,
      "missing_rate": missing_rate,
      "knn_neighbors": args.knn_neighbors,
      "rows_with_target": int(valid_mask.sum()),
      "rows_used_for_cv": int(train_mask.sum()),
    }

    print("\nGuardando resultados...")
    _save_json(results, Path(args.json_output), meta)
    _save_csv(results, Path(args.csv_output))
    _save_plot(results, Path(args.plot_output))

    mlflow.log_artifact(args.json_output)
    mlflow.log_artifact(args.csv_output)
    mlflow.log_artifact(args.plot_output)

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
