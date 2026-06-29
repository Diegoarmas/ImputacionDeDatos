"""
Estudio de la influencia de MARCA y MODELO en la imputación de EMISIONES_CO2.

Analiza cómo varía la precisión de imputación según la marca/modelo del vehículo
y cuantifica el aporte de estas features mediante un experimento de ablación.

Salidas:
  - results/tables/brand_analysis.csv        — métricas por marca
  - results/tables/model_analysis.csv        — métricas por modelo
  - results/tables/ablation_brand_model.csv  — comparativa con/sin features (si --ablation)
  - results/plots/mae_by_brand.png           — gráfico MAE por marca
  - results/plots/mae_by_model.png           — gráfico MAE por modelo (top N)
  - results/plots/volume_vs_mae.png          — scatter volumen vs. error
  - results/reports/brand_model_report.json  — reporte completo
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from data_cleaning import TARGET_COLUMN, load_csv_resilient, prepare_dataframe


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def _load_pool(
    pool_dir: Path,
    pattern: str,
    sep: str,
    encoding: str,
    max_files: int,
) -> pd.DataFrame:
    files = sorted(f for f in pool_dir.glob(pattern) if f.is_file())
    if not files:
        raise FileNotFoundError(f"Sin ficheros con patrón '{pattern}' en {pool_dir}")
    if max_files > 0:
        files = files[:max_files]

    frames = []
    for f in files:
        print(f"  Cargando {f.name} ...")
        frames.append(load_csv_resilient(f, sep=sep, encoding=encoding))
    combined = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    print(f"  Total filas cargadas: {len(combined):,}")
    return combined


# ---------------------------------------------------------------------------
# Métricas por grupo
# ---------------------------------------------------------------------------

def _group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    min_samples: int,
) -> pd.DataFrame:
    """Calcula MAE, RMSE, R² por cada grupo (marca o modelo)."""
    records = []
    unique_groups = np.unique(groups)

    for g in unique_groups:
        mask = groups == g
        n = int(mask.sum())
        if n < min_samples:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        mae = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        # R² puede ser negativo o indefinido con pocas muestras / varianza nula.
        try:
            r2 = float(r2_score(yt, yp))
        except Exception:
            r2 = np.nan

        records.append({
            "grupo": g,
            "n_filas": n,
            "co2_real_medio": float(yt.mean()),
            "co2_pred_medio": float(yp.mean()),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "error_relativo_pct": float(mae / yt.mean() * 100) if yt.mean() > 0 else np.nan,
        })

    df = pd.DataFrame(records).sort_values("mae", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def _plot_mae_by_group(
    df: pd.DataFrame,
    title: str,
    xlabel: str,
    output_path: Path,
    top_n: int,
    global_mae: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = df.head(top_n).sort_values("mae", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.35)))
    colors = []
    for _, row in plot_df.iterrows():
        if row["mae"] > global_mae * 1.5:
            colors.append("#e74c3c")  # rojo — error alto
        elif row["mae"] > global_mae:
            colors.append("#f39c12")  # naranja — por encima de media
        else:
            colors.append("#27ae60")  # verde — igual o mejor que media

    bars = ax.barh(plot_df["grupo"], plot_df["mae"], color=colors, edgecolor="white", height=0.6)
    ax.axvline(global_mae, color="#3498db", linestyle="--", linewidth=1.5, label=f"MAE global = {global_mae:.2f}")

    for bar, val in zip(bars, plot_df["mae"]):
        ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8)

    ax.set_xlabel("MAE (g CO₂/km)")
    ax.set_ylabel(xlabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {output_path}")


def _plot_volume_vs_mae(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        df["n_filas"],
        df["mae"],
        c=df["mae"],
        cmap="RdYlGn_r",
        s=60,
        alpha=0.7,
        edgecolors="grey",
        linewidths=0.5,
    )

    # Etiquetas para las marcas más extremas.
    for _, row in df.nlargest(5, "mae").iterrows():
        ax.annotate(
            row["grupo"],
            (row["n_filas"], row["mae"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=7.5,
            color="#555",
        )
    for _, row in df.nsmallest(3, "mae").iterrows():
        ax.annotate(
            row["grupo"],
            (row["n_filas"], row["mae"]),
            textcoords="offset points",
            xytext=(8, -6),
            fontsize=7.5,
            color="#555",
        )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("MAE (g CO₂/km)")

    ax.set_xlabel("Nº de vehículos (filas)")
    ax.set_ylabel("MAE (g CO₂/km)")
    ax.set_title("Volumen de datos vs. error de imputación por marca", fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {output_path}")


# ---------------------------------------------------------------------------
# Experimento de ablación
# ---------------------------------------------------------------------------

ABLATION_DROP_COLS = ["MARCA", "MODELO"]


def _run_ablation(
    train_dir: Path,
    test_features: pd.DataFrame,
    test_target: pd.Series,
    valid_mask: pd.Series,
    device: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Re-entrena CON y SIN MARCA/MODELO usando el mismo pool/algoritmo y compara.

    Ambas configuraciones se entrenan desde cero sobre el mismo train_dir y con el
    mismo backend (device) que el modelo de producción, para que la única variable
    que cambie entre filas de la tabla sea la presencia de MARCA/MODELO.
    """
    from modeling import build_pipeline, fit_and_evaluate

    print(f"\n=== Experimento de ablación: re-entrenando CON y SIN MARCA/MODELO (device={device}) ===")

    # --- Cargar datos de train ---
    train_df = _load_pool(
        pool_dir=train_dir,
        pattern="*.csv",
        sep=args.sep,
        encoding=args.encoding,
        max_files=args.max_files,
    )
    train_features, train_target = prepare_dataframe(train_df)
    del train_df
    gc.collect()

    train_valid = train_target.notna() & (train_target > 0)
    x_train_full = train_features.loc[train_valid]
    y_train = train_target.loc[train_valid]
    x_test_full = test_features.loc[valid_mask]
    y_test = test_target.loc[valid_mask]

    drop_cols_present = [c for c in ABLATION_DROP_COLS if c in x_train_full.columns]
    print(f"  Features eliminadas en la config 'sin': {drop_cols_present}")

    configs = [
        ("Con MARCA/MODELO (original)", x_train_full, x_test_full),
        ("Sin MARCA/MODELO (ablación)", x_train_full.drop(columns=drop_cols_present),
         x_test_full.drop(columns=drop_cols_present)),
    ]

    rows = []
    metrics_by_config = {}
    for label, x_tr, x_te in configs:
        pipeline, _, _, _ = build_pipeline(x_tr, random_state=args.random_state, device=device)

        scores = fit_and_evaluate(
            pipeline,
            x_tr,
            y_train,
            random_state=args.random_state,
            cv_folds=args.cv_folds,
            n_jobs_cv=-1,
        )
        print(f"  CV [{label}]: MAE={scores['mae']:.4f}  RMSE={scores['rmse']:.4f}  R²={scores['r2']:.4f}")

        pipeline.fit(x_tr, y_train)
        y_pred = np.maximum(pipeline.predict(x_te), 0.0)
        test_mae = float(mean_absolute_error(y_test, y_pred))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        test_r2 = float(r2_score(y_test, y_pred))
        print(f"  Test [{label}]: MAE={test_mae:.4f}  RMSE={test_rmse:.4f}  R²={test_r2:.4f}")

        metrics = {
            "config": label,
            "cv_mae": scores["mae"],
            "cv_rmse": scores["rmse"],
            "cv_r2": scores["r2"],
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
        }
        rows.append(metrics)
        metrics_by_config[label] = metrics

    con = metrics_by_config["Con MARCA/MODELO (original)"]
    sin = metrics_by_config["Sin MARCA/MODELO (ablación)"]

    delta_mae = sin["test_mae"] - con["test_mae"]
    delta_r2 = con["test_r2"] - sin["test_r2"]
    rows.append({
        "config": "Δ (sin − con)",
        "cv_mae": sin["cv_mae"] - con["cv_mae"],
        "cv_rmse": sin["cv_rmse"] - con["cv_rmse"],
        "cv_r2": con["cv_r2"] - sin["cv_r2"],
        "test_mae": delta_mae,
        "test_rmse": sin["test_rmse"] - con["test_rmse"],
        "test_r2": delta_r2,
    })

    ablation_df = pd.DataFrame(rows)

    print("\n  Impacto de MARCA/MODELO:")
    print(f"    ΔMAE  test = +{delta_mae:.4f} g/km (sin MARCA/MODELO imputa peor en {delta_mae:.4f} g/km)" if delta_mae > 0
          else f"    ΔMAE  test = {delta_mae:.4f} g/km (eliminar MARCA/MODELO mejora ligeramente)")
    print(f"    ΔR²   test = -{delta_r2:.4f}" if delta_r2 > 0
          else f"    ΔR²   test = +{abs(delta_r2):.4f}")

    return ablation_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estudio de influencia de MARCA/MODELO en la imputación de CO2."
    )
    p.add_argument("--test-dir", default="data/processed/pool_test",
                   help="Directorio con datos de test.")
    p.add_argument("--model-path", default="artifacts/models/co2_model.joblib",
                   help="Ruta al modelo entrenado (.joblib).")
    p.add_argument("--max-files", type=int, default=5,
                   help="Máximo de ficheros CSV a cargar por pool.")
    p.add_argument("--sep", default=",")
    p.add_argument("--encoding", default="utf-8")
    p.add_argument("--top-brands", type=int, default=20,
                   help="Número de marcas a mostrar en gráficos.")
    p.add_argument("--top-models", type=int, default=30,
                   help="Número de modelos a mostrar en gráficos.")
    p.add_argument("--min-samples", type=int, default=100,
                   help="Mínimo de filas por grupo para incluirlo en el análisis.")
    p.add_argument("--ablation", action="store_true",
                   help="Ejecuta el experimento de ablación (re-entrena sin MARCA/MODELO).")
    p.add_argument("--train-dir", default="data/processed/pool_train",
                   help="Directorio de train (necesario para --ablation).")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)

    # Rutas de salida
    p.add_argument("--brand-csv", default="results/tables/brand_analysis.csv")
    p.add_argument("--model-csv", default="results/tables/model_analysis.csv")
    p.add_argument("--ablation-csv", default="results/tables/ablation_brand_model.csv")
    p.add_argument("--brand-plot", default="results/plots/mae_by_brand.png")
    p.add_argument("--model-plot", default="results/plots/mae_by_model.png")
    p.add_argument("--volume-plot", default="results/plots/volume_vs_mae.png")
    p.add_argument("--report-json", default="results/reports/brand_model_report.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    import joblib

    # --- Cargar modelo ---
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: no se encuentra el modelo en {model_path}")
        return 1

    print(f"Cargando modelo desde {model_path} ...")
    model_artifact = joblib.load(model_path)
    pipeline = model_artifact["pipeline"]
    feature_columns = model_artifact["feature_columns"]

    # --- Cargar datos de test ---
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"Error: no existe el directorio de test {test_dir}")
        return 1

    print(f"\nCargando datos de test desde {test_dir} ...")
    test_df = _load_pool(test_dir, "*.csv", args.sep, args.encoding, args.max_files)

    if TARGET_COLUMN not in test_df.columns:
        print(f"Error: columna '{TARGET_COLUMN}' no encontrada.")
        return 1

    # Guardar columnas de marca/modelo antes de prepare_dataframe.
    raw_marca = test_df["MARCA"].fillna("DESCONOCIDO").astype(str).str.strip().values if "MARCA" in test_df.columns else None
    raw_modelo = test_df["MODELO"].fillna("DESCONOCIDO").astype(str).str.strip().values if "MODELO" in test_df.columns else None

    features, target = prepare_dataframe(test_df)
    valid_mask = target.notna() & (target > 0)
    n_valid = int(valid_mask.sum())
    print(f"  Filas con CO2 conocido (>0): {n_valid:,}")

    if n_valid < 100:
        print("Error: muy pocas filas con target válido.")
        return 1

    # --- Predicción global ---
    print("\nPrediciendo CO2 en test set ...")
    x_test = features.loc[valid_mask].reindex(columns=feature_columns)
    y_test = target.loc[valid_mask].values
    y_pred = np.maximum(pipeline.predict(x_test), 0.0)

    global_mae = float(mean_absolute_error(y_test, y_pred))
    global_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    global_r2 = float(r2_score(y_test, y_pred))
    print(f"  Métricas globales test: MAE={global_mae:.4f}  RMSE={global_rmse:.4f}  R²={global_r2:.4f}")

    report: dict = {
        "global": {
            "n_filas": n_valid,
            "mae": global_mae,
            "rmse": global_rmse,
            "r2": global_r2,
        },
    }

    # --- Análisis por MARCA ---
    brand_df = None
    if raw_marca is not None:
        print("\n=== Análisis por MARCA ===")
        marcas_valid = raw_marca[valid_mask.values]
        brand_df = _group_metrics(y_test, y_pred, marcas_valid, args.min_samples)
        print(f"  Marcas con >= {args.min_samples} filas: {len(brand_df)}")

        brand_path = Path(args.brand_csv)
        brand_path.parent.mkdir(parents=True, exist_ok=True)
        brand_df.to_csv(brand_path, index=False)
        print(f"  CSV: {brand_path}")

        _plot_mae_by_group(
            brand_df,
            title="MAE de imputación de CO₂ por MARCA",
            xlabel="Marca",
            output_path=Path(args.brand_plot),
            top_n=args.top_brands,
            global_mae=global_mae,
        )

        _plot_volume_vs_mae(brand_df, Path(args.volume_plot))

        # Top 5 peores y mejores.
        print("\n  Top 5 peor imputadas (mayor MAE):")
        for _, r in brand_df.head(5).iterrows():
            print(f"    {r['grupo']:<20s}  MAE={r['mae']:.2f}  n={r['n_filas']:,}")
        print("  Top 5 mejor imputadas (menor MAE):")
        for _, r in brand_df.tail(5).iterrows():
            print(f"    {r['grupo']:<20s}  MAE={r['mae']:.2f}  n={r['n_filas']:,}")

        report["por_marca"] = brand_df.to_dict(orient="records")
    else:
        print("Advertencia: columna MARCA no encontrada, se omite análisis por marca.")

    # --- Análisis por MODELO ---
    model_df_result = None
    if raw_modelo is not None:
        print("\n=== Análisis por MODELO ===")
        modelos_valid = raw_modelo[valid_mask.values]
        model_df_result = _group_metrics(y_test, y_pred, modelos_valid, args.min_samples)
        print(f"  Modelos con >= {args.min_samples} filas: {len(model_df_result)}")

        model_path_out = Path(args.model_csv)
        model_path_out.parent.mkdir(parents=True, exist_ok=True)
        model_df_result.to_csv(model_path_out, index=False)
        print(f"  CSV: {model_path_out}")

        _plot_mae_by_group(
            model_df_result,
            title="MAE de imputación de CO₂ por MODELO",
            xlabel="Modelo",
            output_path=Path(args.model_plot),
            top_n=args.top_models,
            global_mae=global_mae,
        )

        print("\n  Top 5 modelos peor imputados (mayor MAE):")
        for _, r in model_df_result.head(5).iterrows():
            print(f"    {r['grupo']:<30s}  MAE={r['mae']:.2f}  n={r['n_filas']:,}")
        print("  Top 5 modelos mejor imputados (menor MAE):")
        for _, r in model_df_result.tail(5).iterrows():
            print(f"    {r['grupo']:<30s}  MAE={r['mae']:.2f}  n={r['n_filas']:,}")

        report["por_modelo"] = model_df_result.head(args.top_models).to_dict(orient="records")
    else:
        print("Advertencia: columna MODELO no encontrada, se omite análisis por modelo.")

    # --- Ablación ---
    ablation_df = None
    if args.ablation:
        train_dir = Path(args.train_dir)
        if not train_dir.exists():
            print(f"Error: directorio de train '{train_dir}' no encontrado (necesario para --ablation).")
            return 1

        ablation_df = _run_ablation(
            train_dir=train_dir,
            test_features=features,
            test_target=target,
            valid_mask=valid_mask,
            device=model_artifact.get("device", "cpu"),
            args=args,
        )

        abl_path = Path(args.ablation_csv)
        abl_path.parent.mkdir(parents=True, exist_ok=True)
        ablation_df.to_csv(abl_path, index=False)
        print(f"  CSV ablación: {abl_path}")

        report["ablation"] = ablation_df.to_dict(orient="records")

    # --- Reporte JSON ---
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReporte JSON: {report_path}")

    # --- Resumen final ---
    print("\n" + "=" * 60)
    print("RESUMEN DEL ESTUDIO")
    print("=" * 60)
    print(f"MAE global: {global_mae:.4f} g CO₂/km")
    if brand_df is not None and len(brand_df) > 0:
        worst_brand = brand_df.iloc[0]
        best_brand = brand_df.iloc[-1]
        print(f"Peor marca:  {worst_brand['grupo']} (MAE={worst_brand['mae']:.2f}, n={worst_brand['n_filas']:,})")
        print(f"Mejor marca: {best_brand['grupo']} (MAE={best_brand['mae']:.2f}, n={best_brand['n_filas']:,})")
        ratio = worst_brand["mae"] / best_brand["mae"] if best_brand["mae"] > 0 else float("inf")
        print(f"Ratio peor/mejor: {ratio:.1f}x")
    if ablation_df is not None:
        abl_row = ablation_df[ablation_df["config"].str.contains("Δ")]
        if len(abl_row) > 0:
            d_mae = abl_row.iloc[0]["test_mae"]
            print(f"Impacto ablación MARCA/MODELO: ΔMAE = {d_mae:+.4f} g/km")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
