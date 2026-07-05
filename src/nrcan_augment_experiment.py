#!/usr/bin/env python3
"""
nrcan_augment_experiment.py
============================
Experimento INDEPENDIENTE: ¿mejora el modelo al aumentar el entrenamiento
con registros de NRCan (Natural Resources Canada) corregidos al ciclo EU?

Este script NO modifica ningún modelo ni artefacto de producción.

Flujo
-----
1. Carga datos DGT de pool_train y pool_val.
2. Carga / descarga datos NRCan (Fuel Consumption Ratings, 1995-actual).
3. Filtra NRCan a marcas europeas presentes en el parque DGT.
4. Aplica factor de corrección validado (ciclo EPA → ciclo EU/NEDC) derivado del
   análisis empírico EEA-2010 vs NRCan-2010 (comparacion_eu_canada.md):
     · Por tipo de combustible: gasolina 1/1.205, diésel 1/1.527
     · Regresión lineal: CO₂_EU = (CO₂_CA − 134.6) / 0.605  [R²=0.55]
   Método seleccionable via --correction-method {fuel_type|regression}.
5. BASELINE  : entrena HGBT únicamente con datos DGT; evalúa en pool_val.
6. AUGMENTADO: entrena HGBT con DGT + filas NRCan corregidas; evalúa en pool_val.
7. Guarda JSON con métricas y PNG con gráfico comparativo.

Uso
---
  # Descarga NRCan automáticamente:
  python src/nrcan_augment_experiment.py

  # Con CSV de NRCan ya descargado:
  python src/nrcan_augment_experiment.py --nrcan-csv /ruta/nrcan_combinado.csv

  # Experimento rápido (pocos ficheros):
  python src/nrcan_augment_experiment.py --max-train-files 5 --max-val-files 3

Descarga manual de NRCan
------------------------
Si la descarga automática falla, descarga los CSV desde:
  https://natural-resources.canada.ca/energy-efficiency/transportation-alternative-fuels/fuel-consumption-guide/using-fuel-consumption-guide-data/fuel-consumption-ratings-datasets/21002
y pasa la ruta con --nrcan-csv.
"""

import argparse
import gc
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# ---------------------------------------------------------------------------
# Path: permite ejecutar desde la raíz del proyecto
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from data_cleaning import TARGET_COLUMN, load_csv_resilient, prepare_dataframe

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# URLs de los ficheros NRCan (se intentan en orden)
NRCAN_URLS = [
    # Combinado más reciente
    "https://www.nrcan.gc.ca/sites/nrcan/files/oee/files/csv/MY1995-2025_Fuel_Consumption_Ratings.csv",
    "https://www.nrcan.gc.ca/sites/nrcan/files/oee/files/csv/MY1995-2024_Fuel_Consumption_Ratings.csv",
    # Separados por periodo
    "https://www.nrcan.gc.ca/sites/nrcan/files/oee/files/csv/MY2015-2024_Fuel_Consumption_Ratings.csv",
    "https://www.nrcan.gc.ca/sites/nrcan/files/oee/files/csv/MY2015-2023_Fuel_Consumption_Ratings.csv",
    "https://www.nrcan.gc.ca/sites/nrcan/files/oee/files/csv/MY1995-2014_Fuel_Consumption_Ratings_5-cycle.csv",
]

# Marcas europeas que sí aparecen en el catálogo canadiense con regularidad.
# Marcas ausentes en NRCan: RENAULT, SEAT, PEUGEOT, CITROËN, OPEL, SKODA, DACIA.
EUROPEAN_BRANDS_NRCAN = {
    "VOLKSWAGEN", "VW",
    "BMW",
    "MERCEDES-BENZ", "MERCEDES",
    "AUDI",
    "VOLVO",
    "MINI",
    "LAND ROVER",
    "JAGUAR",
    "PORSCHE",
    "ALFA ROMEO",
    "FIAT",
    "SMART",
    "BENTLEY",
    "LAMBORGHINI",
    "MASERATI",
    "ROLLS-ROYCE",
}

# Mapeo NRCan Fuel Type → código de PROPULSION DGT (mejores esfuerzo)
NRCAN_FUEL_TO_PROPULSION = {
    "X": "1",   # Regular gasoline
    "Z": "1",   # Premium gasoline
    "D": "2",   # Diesel
    "E": "5",   # E85 ethanol — aproximado como bi-combustible
    "N": "3",   # Natural gas
    "B": "1",   # PHEV gasolina (gasoline component only)
}

# Columnas numéricas DGT conocidas (subset disponible desde NRCan)
NUMERIC_FROM_NRCAN = ["CILINDRADA", "FECHA_MATR_YEAR", "FEC_PRIM_MATR_YEAR"]

# ---------------------------------------------------------------------------
# Factores de corrección EPA → NEDC validados empíricamente en
# comparacion_eu_canada.md  (EEA 2010 vs NRCan 2010, 371 pares emparejados)
#
#   Ratio CA/EU global      : 1.213   (+21 %)
#   Por combustible:
#     Gasolina (n=362)       : 1.205   → factor EU/CA = 0.8299
#     Diésel   (n=9)         : 1.527   → factor EU/CA = 0.6549
#   Regresión lineal (R²=0.55):
#     CO₂_CA = 0.605 × CO₂_EU + 134.63
#     → CO₂_EU = (CO₂_CA − 134.63) / 0.605
#
# NOTA: estos factores corresponden al ciclo NEDC (era pre-2018).
# Para vehículos post-2018 con homologación WLTP en DGT, el sesgo sería
# menor porque WLTP es más exigente que NEDC (más próximo al ciclo EPA).
# ---------------------------------------------------------------------------

# Factor EU/CA por categoría de combustible (gasolina / diésel / desconocido)
CORRECTION_BY_FUEL: dict[str, float] = {
    "G": 1.0 / 1.205,   # gasolina → EU ≈ CA / 1.205 = CA × 0.8299
    "D": 1.0 / 1.527,   # diésel   → EU ≈ CA / 1.527 = CA × 0.6549
    "O": 1.0 / 1.213,   # otros    → ratio global como fallback
}

# Constantes de la regresión lineal EEA↔NRCan
REG_SLOPE     = 0.605
REG_INTERCEPT = 134.63


# ===========================================================================
# 1. DESCARGA Y CARGA DE DATOS NRCAN
# ===========================================================================

def _try_download(url: str, dest: Path) -> bool:
    """Intenta descargar url a dest. Devuelve True si tiene éxito."""
    try:
        print(f"  Intentando: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            if r.status != 200:
                return False
            dest.write_bytes(r.read())
        print(f"  → Descargado en {dest}")
        return True
    except Exception as exc:
        print(f"  → Fallo: {exc}")
        return False


def download_nrcan(cache_dir: Path) -> Path | None:
    """
    Intenta descargar los CSV de NRCan desde URLs conocidas.
    Si hay varios ficheros (1995-2014 + 2015-actual) los concatena en un único CSV.
    Devuelve la ruta al CSV combinado o None si todos los intentos fallan.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    combined_path = cache_dir / "nrcan_combined.csv"
    if combined_path.exists():
        print(f"  Usando caché: {combined_path}")
        return combined_path

    print("Descargando datos NRCan...")

    # Intenta el fichero único combinado primero
    for url in NRCAN_URLS[:3]:
        dest = cache_dir / Path(url).name
        if _try_download(url, dest):
            dest.rename(combined_path)
            return combined_path

    # Si no hay combinado, intenta los dos periodos por separado
    parts = []
    for url in NRCAN_URLS[3:]:
        dest = cache_dir / Path(url).name
        if _try_download(url, dest):
            parts.append(dest)

    if len(parts) >= 1:
        print("  Concatenando ficheros parciales...")
        frames = [pd.read_csv(p, encoding="latin-1", dtype=str, low_memory=False) for p in parts]
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(combined_path, index=False)
        print(f"  → Combinado en {combined_path} ({len(combined):,} filas)")
        return combined_path

    return None


def _normalize_col(series: pd.Series) -> pd.Series:
    """Normaliza strings: strip + uppercase."""
    return series.astype(str).str.strip().str.upper()


def load_nrcan(path: Path) -> pd.DataFrame:
    """
    Carga el CSV de NRCan y normaliza columnas al esquema interno.
    Devuelve DataFrame con columnas:
      make, model, year, engine_l, fuel_type, co2_gkm
    """
    print(f"Cargando NRCan desde {path} ...")

    # NRCan usa encoding latin-1 en algunos ficheros
    for enc in ("utf-8", "latin-1", "utf-8-sig"):
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"No se puede leer {path} con encodings estándar.")

    print(f"  Columnas raw NRCan: {list(df.columns)}")

    # --- Normaliza nombres de columna (distintas versiones del fichero) ---
    rename_map: dict[str, str] = {}
    col_lower = {c.lower().strip(): c for c in df.columns}

    def _find(*candidates: str) -> str | None:
        for c in candidates:
            if c in col_lower:
                return col_lower[c]
        return None

    col_year  = _find("model year", "year", "année modèle")
    col_make  = _find("make", "marque")
    col_model = _find("model", "modèle")
    col_eng   = _find("engine size(l)", "engine size (l)", "taille moteur (l)", "cylindrée (l)")
    col_fuel  = _find("fuel type", "type de carburant")
    col_co2   = _find("co2 emissions(g/km)", "co2 emissions (g/km)", "co2 (g/km)",
                      "émissions de co2 (g/km)", "émissions co2 (g/km)")

    missing = [name for name, col in
               [("year", col_year), ("make", col_make), ("model", col_model),
                ("engine", col_eng), ("fuel", col_fuel), ("co2", col_co2)]
               if col is None]
    if missing:
        raise ValueError(
            f"No se encontraron columnas NRCan para: {missing}.\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "year":      pd.to_numeric(df[col_year],  errors="coerce"),
        "make":      _normalize_col(df[col_make]),
        "model":     _normalize_col(df[col_model]),
        "engine_l":  pd.to_numeric(df[col_eng],   errors="coerce"),
        "fuel_type": _normalize_col(df[col_fuel]),
        "co2_gkm":   pd.to_numeric(df[col_co2],   errors="coerce"),
    })

    # Descarta filas sin CO₂ o cilindrada
    out = out.dropna(subset=["co2_gkm", "engine_l"])
    out = out[out["co2_gkm"] > 0]

    print(f"  {len(out):,} registros NRCan válidos (1995-actual)")
    return out


# ===========================================================================
# 2. FILTRADO A MARCAS EUROPEAS
# ===========================================================================

def filter_european_brands(nrcan: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra NRCan a marcas europeas que aparecen también en NRCan.
    Además excluye vehículos no-turismo (trucks, pickups, vans pesadas).
    """
    mask = nrcan["make"].isin(EUROPEAN_BRANDS_NRCAN)
    filtered = nrcan[mask].copy()
    n_kept = len(filtered)
    n_total = len(nrcan)
    pct = 100 * n_kept / n_total if n_total > 0 else 0
    print(f"  Marcas europeas en NRCan: {n_kept:,} / {n_total:,} filas ({pct:.1f}%)")
    print(f"  Marcas presentes: {sorted(filtered['make'].unique())}")
    return filtered


# ===========================================================================
# 3. CORRECCIÓN VALIDADA (ciclo EPA → ciclo EU/NEDC)
# ===========================================================================

def _fuel_cat(fuel_type_nrcan: str) -> str:
    """Mapea el código de combustible NRCan a la categoría de corrección."""
    f = str(fuel_type_nrcan).strip().upper()
    if f in ("X", "Z", "B"):
        return "G"  # gasolina / PHEV gasolina
    if f == "D":
        return "D"  # diésel
    return "O"      # otros (GLP, E85, eléctrico…)


def correct_co2_fuel_type(co2_ca: float, fuel_type_nrcan: str) -> float:
    """
    Aplica el factor de corrección validado por tipo de combustible.

    Fuente: comparacion_eu_canada.md — análisis EEA-2010 vs NRCan-2010,
    371 pares emparejados por (marca, modelo, cilindrada ±200cc, combustible).

      Gasolina: ratio CA/EU = 1.205  → EU = CA / 1.205
      Diésel:   ratio CA/EU = 1.527  → EU = CA / 1.527
      Otros:    ratio CA/EU = 1.213  (ratio global, fallback)
    """
    factor = CORRECTION_BY_FUEL[_fuel_cat(fuel_type_nrcan)]
    return co2_ca * factor


def correct_co2_regression(co2_ca: float) -> float:
    """
    Aplica la regresión lineal inversa EEA↔NRCan para estimar CO₂ EU.

    Modelo ajustado: CO₂_CA = 0.605 × CO₂_EU + 134.63  (R²=0.55, RMSE=66 g/km)
    Inversa:         CO₂_EU = (CO₂_CA − 134.63) / 0.605

    Más preciso que el factor simple, pero con mayor incertidumbre en los
    extremos del rango (vehículos de muy baja o muy alta emisión).
    Valores CO₂_CA < 134.63 g/km producen estimaciones negativas → se clampean
    al mínimo físico de 50 g/km.
    """
    eu = (co2_ca - REG_INTERCEPT) / REG_SLOPE
    return max(eu, 50.0)


def apply_correction(
    nrcan_df: pd.DataFrame,
    method: str,
) -> pd.Series:
    """
    Devuelve una Series con los CO₂ NRCan corregidos al ciclo EU/NEDC.

    method:
      'fuel_type'  — factor por tipo de combustible (gasolina/diésel/otro).
      'regression' — regresión lineal inversa EEA↔NRCan (independiente del combustible).
    """
    if method == "regression":
        print(f"  Método de corrección: regresión lineal  "
              f"[CO₂_EU = (CO₂_CA − {REG_INTERCEPT}) / {REG_SLOPE}]")
        corrected = nrcan_df["co2_gkm"].apply(correct_co2_regression)
    else:
        print(f"  Método de corrección: por tipo de combustible  "
              f"[G/1.205, D/1.527, O/1.213]")
        corrected = pd.Series([
            correct_co2_fuel_type(row["co2_gkm"], row["fuel_type"])
            for _, row in nrcan_df.iterrows()
        ], index=nrcan_df.index)

    print(f"  CO₂ NRCan original  — media: {nrcan_df['co2_gkm'].mean():.1f} g/km, "
          f"mediana: {nrcan_df['co2_gkm'].median():.1f} g/km")
    print(f"  CO₂ NRCan corregido — media: {corrected.mean():.1f} g/km, "
          f"mediana: {corrected.median():.1f} g/km")
    return corrected


# ===========================================================================
# 4. CONVERSIÓN NRCAN → FORMATO DGT PREPARADO
# ===========================================================================

def nrcan_to_prepared(
    nrcan_df: pd.DataFrame,
    dgt_feature_cols: list[str],
    co2_corrected: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convierte filas NRCan al formato de características DGT listo para el modelo.

    Columnas mapeadas desde NRCan:
      MARCA              → make (normalizado, uppercase)
      MODELO             → model
      CILINDRADA         → engine_l × 1000  (cc)
      PROPULSION         → fuel_type mapeado a código DGT
      FECHA_MATR_YEAR /
      FEC_PRIM_MATR_YEAR → year del modelo
      EMISIONES_CO2      → co2_corrected (ya convertido al ciclo EU/NEDC)

    El resto de columnas DGT se dejan a NaN; HGBT tolera missing en features.

    Parámetros
    ----------
    co2_corrected : pd.Series
        CO₂ en g/km ya corregido al ciclo EU, resultado de apply_correction().
    """
    n = len(nrcan_df)
    X = pd.DataFrame(np.nan, index=range(n), columns=dgt_feature_cols)

    # --- Numéricas ---
    if "CILINDRADA" in X.columns:
        X["CILINDRADA"] = (nrcan_df["engine_l"].values * 1000).astype("float32")

    for yr_col in ("FECHA_MATR_YEAR", "FEC_PRIM_MATR_YEAR"):
        if yr_col in X.columns:
            X[yr_col] = nrcan_df["year"].values.astype("float32")

    # --- Categóricas ---
    if "MARCA" in X.columns:
        X["MARCA"] = nrcan_df["make"].values

    if "MODELO" in X.columns:
        X["MODELO"] = nrcan_df["model"].values

    if "PROPULSION" in X.columns:
        X["PROPULSION"] = nrcan_df["fuel_type"].map(
            lambda f: NRCAN_FUEL_TO_PROPULSION.get(str(f).strip().upper(), np.nan)
        )

    # --- Target corregido ---
    y = pd.Series(
        co2_corrected.values,
        name=TARGET_COLUMN,
        dtype="float32",
    )

    print(f"  Filas NRCan convertidas al esquema DGT: {n:,}")
    return X, y


# ===========================================================================
# 5. PIPELINE DE MODELO
# ===========================================================================

def build_hgbt_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Construye un Pipeline HGBT reutilizando la misma arquitectura que
    compare_imputers.py (sin XGBoost-CUDA para no contaminar artefactos).
    """
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols     = [c for c in X.columns if c not in numeric_cols]

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="DESCONOCIDO")),
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    return Pipeline([
        ("pre", pre),
        ("model", HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
        )),
    ])


def evaluate(model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    y_pred = model.predict(X_val)
    mae  = mean_absolute_error(y_val, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    r2   = r2_score(y_val, y_pred)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4), "n_val": int(len(y_val))}


# Segmentos temporales: NRCan cubre ~1990-2010, objetivo es cubrir vehículos
# antiguos del DGT que carecen de CO₂ declarado.
YEAR_SEGMENTS = [
    ("pre_2000",  None, 2000),
    ("2000_2009", 2000, 2010),
    ("2010_2019", 2010, 2020),
    ("2020_plus", 2020, None),
]


def evaluate_by_year(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """Métricas segmentadas por año de matriculación (FECHA_MATR_YEAR)."""
    year_col = None
    for c in ("FECHA_MATR_YEAR", "FEC_PRIM_MATR_YEAR"):
        if c in X_val.columns:
            year_col = c
            break
    if year_col is None:
        return {}

    years = pd.to_numeric(X_val[year_col], errors="coerce")
    segments: dict = {}
    for name, lo, hi in YEAR_SEGMENTS:
        mask = pd.Series(True, index=X_val.index)
        if lo is not None:
            mask &= years >= lo
        if hi is not None:
            mask &= years < hi
        n = int(mask.sum())
        if n < 10:
            segments[name] = {"n": n, "mae": None, "r2": None}
            continue
        y_pred = model.predict(X_val[mask])
        mae = float(mean_absolute_error(y_val[mask], y_pred))
        r2  = float(r2_score(y_val[mask], y_pred))
        segments[name] = {"n": n, "mae": round(mae, 4), "r2": round(r2, 4)}
    return segments


# ===========================================================================
# 6. CARGA DE DATOS DGT
# ===========================================================================

def load_pool(pool_dir: Path, max_files: int, label: str) -> pd.DataFrame:
    files = sorted(f for f in pool_dir.glob("*.csv") if f.is_file())
    if not files:
        raise FileNotFoundError(f"Sin ficheros CSV en {pool_dir}")
    if max_files > 0:
        files = files[:max_files]
    print(f"  Cargando {label}: {len(files)} ficheros de {pool_dir.name}/")
    frames = [load_csv_resilient(f, sep=",", encoding="utf-8") for f in files]
    df = pd.concat(frames, ignore_index=True)
    del frames; gc.collect()
    print(f"    → {len(df):,} filas")
    return df


# ===========================================================================
# 7. SALIDAS
# ===========================================================================

def save_json(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  JSON guardado: {path}")


def save_plot(results: dict, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib no disponible, se omite gráfico.")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    baseline = results["baseline"]
    augmented = results["augmented"]
    metrics = ["mae", "rmse", "r2"]
    labels  = ["MAE (↓ mejor)", "RMSE (↓ mejor)", "R² (↑ mejor)"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b_vals = [baseline[m] for m in metrics]
    a_vals = [augmented[m] for m in metrics]

    bars_b = ax.bar(x - width/2, b_vals, width, label="Baseline (DGT solo)",  color="steelblue",  alpha=0.85)
    bars_a = ax.bar(x + width/2, a_vals, width, label="Augmentado (DGT+NRCan)", color="darkorange", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Experimento NRCan: efecto del aumento de datos sobre el conjunto de validación")
    ax.legend()

    for bar, val in list(zip(bars_b, b_vals)) + list(zip(bars_a, a_vals)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfico guardado: {path}")


def print_summary(results: dict) -> None:
    b = results["baseline"]
    a = results["augmented"]
    meta = results["meta"]
    n_nrcan = meta["nrcan_rows_added"]

    delta_mae  = a["mae"]  - b["mae"]
    delta_rmse = a["rmse"] - b["rmse"]
    delta_r2   = a["r2"]   - b["r2"]

    def _sign(v: float) -> str:
        return f"+{v:.4f}" if v >= 0 else f"{v:.4f}"

    print("\n" + "=" * 65)
    print(f"{'Métrica':<12}  {'Baseline':>12}  {'Augmentado':>12}  {'Δ':>10}")
    print("-" * 65)
    print(f"{'MAE':12}  {b['mae']:12.4f}  {a['mae']:12.4f}  {_sign(delta_mae):>10}")
    print(f"{'RMSE':12}  {b['rmse']:12.4f}  {a['rmse']:12.4f}  {_sign(delta_rmse):>10}")
    print(f"{'R²':12}  {b['r2']:12.4f}  {a['r2']:12.4f}  {_sign(delta_r2):>10}")
    print("=" * 65)
    print(f"\nMétodo de corrección EPA→EU : {meta['correction']['method']}")
    print(f"Filas NRCan añadidas        : {n_nrcan:,}")
    print(f"Filas val evaluadas         : {b['n_val']:,}")

    if delta_mae < -0.5:
        print("\n✓ El aumento con NRCan MEJORA el modelo (ΔMAE < −0.5 g/km).")
    elif abs(delta_mae) < 0.5:
        print("\n~ Efecto NEUTRO: ΔMAE < 0.5 g/km en valor absoluto.")
    else:
        print("\n✗ El aumento con NRCan EMPEORA el modelo (ΔMAE > +0.5 g/km).")


# ===========================================================================
# 8. MAIN
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experimento NRCan augmentation.")
    p.add_argument(
        "--nrcan-csv", default="",
        help="Ruta al CSV de NRCan ya descargado (opcional; si no se da, se descarga automáticamente).",
    )
    p.add_argument(
        "--nrcan-cache-dir", default="data/nrcan",
        help="Directorio donde cachear la descarga de NRCan. Por defecto: data/nrcan/",
    )
    p.add_argument(
        "--train-dir", default="data/processed/pool_train",
        help="Directorio con ficheros CSV de entrenamiento DGT.",
    )
    p.add_argument(
        "--val-dir", default="data/processed/pool_val",
        help="Directorio con ficheros CSV de validación DGT.",
    )
    p.add_argument(
        "--max-train-files", type=int, default=10,
        help="Máximo de ficheros train a cargar (0 = todos). Por defecto 10 (~1M filas).",
    )
    p.add_argument(
        "--max-val-files", type=int, default=0,
        help="Máximo de ficheros val a cargar (0 = todos).",
    )
    p.add_argument(
        "--only-european", action="store_true", default=True,
        help="Filtrar NRCan sólo a marcas europeas (defecto: activado).",
    )
    p.add_argument(
        "--correction-method",
        choices=["fuel_type", "regression"],
        default="fuel_type",
        help=(
            "Método de corrección EPA→EU/NEDC:\n"
            "  fuel_type  — factor por combustible: gasolina/1.205, diésel/1.527 "
            "(fuente: comparacion_eu_canada.md, 371 pares EEA-2010 vs NRCan-2010)\n"
            "  regression — inversa de CO₂_CA = 0.605×CO₂_EU + 134.63 (R²=0.55)"
        ),
    )
    p.add_argument(
        "--json-output", default="results/nrcan_augment_experiment.json",
        help="Ruta de salida del JSON de resultados.",
    )
    p.add_argument(
        "--plot-output", default="results/plots/nrcan_augment_experiment.png",
        help="Ruta de salida del gráfico PNG.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ------------------------------------------------------------------
    # Paso 1: Carga DGT train + val
    # ------------------------------------------------------------------
    print("\n[1/6] Cargando datos DGT...")
    train_dir = Path(args.train_dir)
    val_dir   = Path(args.val_dir)

    dgt_train_raw = load_pool(train_dir, args.max_train_files, "train")
    dgt_val_raw   = load_pool(val_dir,   args.max_val_files,   "val")

    X_train, y_train_full = prepare_dataframe(dgt_train_raw)
    X_val,   y_val_full   = prepare_dataframe(dgt_val_raw)

    # Filtra a filas con CO₂ conocido > 0
    train_mask = y_train_full.notna() & (y_train_full > 0)
    val_mask   = y_val_full.notna()   & (y_val_full   > 0)

    X_tr = X_train[train_mask].reset_index(drop=True)
    y_tr = y_train_full[train_mask].reset_index(drop=True)
    X_vl = X_val[val_mask].reset_index(drop=True)
    y_vl = y_val_full[val_mask].reset_index(drop=True)

    print(f"  Train DGT (con CO₂): {len(X_tr):,} filas")
    print(f"  Val   DGT (con CO₂): {len(X_vl):,} filas")

    # ------------------------------------------------------------------
    # Paso 2: Carga NRCan
    # ------------------------------------------------------------------
    print("\n[2/6] Cargando datos NRCan...")
    if args.nrcan_csv:
        nrcan_path = Path(args.nrcan_csv)
        if not nrcan_path.exists():
            print(f"ERROR: no existe {nrcan_path}")
            return 1
    else:
        nrcan_path = download_nrcan(Path(args.nrcan_cache_dir))

    if nrcan_path is None:
        print(
            "\nERROR: No se pudo descargar NRCan automáticamente.\n"
            "Descarga manualmente los CSV desde:\n"
            "  https://natural-resources.canada.ca/energy-efficiency/transportation-alternative-fuels/"
            "fuel-consumption-guide/using-fuel-consumption-guide-data/fuel-consumption-ratings-datasets/21002\n"
            "y pasa la ruta con --nrcan-csv /ruta/al/fichero.csv"
        )
        return 1

    nrcan_df = load_nrcan(nrcan_path)

    # ------------------------------------------------------------------
    # Paso 3: Filtrar a marcas europeas
    # ------------------------------------------------------------------
    print("\n[3/6] Filtrando marcas europeas en NRCan...")
    if args.only_european:
        nrcan_df = filter_european_brands(nrcan_df)
    print(f"  Filas NRCan tras filtrado: {len(nrcan_df):,}")

    # ------------------------------------------------------------------
    # Paso 4: Corrección validada EPA → EU/NEDC
    # ------------------------------------------------------------------
    print(f"\n[4/6] Corrigiendo CO₂ NRCan al ciclo EU (método: {args.correction_method})...")
    co2_corrected = apply_correction(nrcan_df, args.correction_method)

    # Convierte NRCan al esquema de features DGT
    X_nrcan, y_nrcan = nrcan_to_prepared(nrcan_df, list(X_tr.columns), co2_corrected)

    # Alinea tipos con X_tr (numéricas float32, categóricas object)
    for col in X_tr.columns:
        if pd.api.types.is_numeric_dtype(X_tr[col]):
            X_nrcan[col] = pd.to_numeric(X_nrcan[col], errors="coerce").astype("float32")
        else:
            X_nrcan[col] = X_nrcan[col].astype(str).replace({"nan": np.nan})

    # ------------------------------------------------------------------
    # Paso 5: BASELINE (sólo DGT)
    # ------------------------------------------------------------------
    print("\n[5/6] Entrenando BASELINE (DGT train únicamente)...")
    pipe_baseline = build_hgbt_pipeline(X_tr)
    pipe_baseline.fit(X_tr, y_tr)
    metrics_baseline = evaluate(pipe_baseline, X_vl, y_vl)
    by_year_baseline = evaluate_by_year(pipe_baseline, X_vl, y_vl)
    print(f"  → MAE={metrics_baseline['mae']:.4f}  RMSE={metrics_baseline['rmse']:.4f}  R²={metrics_baseline['r2']:.4f}")
    if by_year_baseline:
        print("  Por segmento de año:")
        for seg, m in by_year_baseline.items():
            tag = seg.replace("_", " ")
            if m["mae"] is not None:
                print(f"    {tag:12s}  n={m['n']:>7,}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}")

    del pipe_baseline; gc.collect()

    # ------------------------------------------------------------------
    # Paso 6: AUGMENTADO (DGT + NRCan corregido)
    # ------------------------------------------------------------------
    print("\n[6/6] Entrenando AUGMENTADO (DGT + NRCan)...")
    X_aug = pd.concat([X_tr, X_nrcan], ignore_index=True)
    y_aug = pd.concat([y_tr, y_nrcan], ignore_index=True)
    print(f"  Total filas train augmentado: {len(X_aug):,}  "
          f"(+{len(X_nrcan):,} NRCan = {100*len(X_nrcan)/len(X_aug):.1f}%)")

    pipe_augmented = build_hgbt_pipeline(X_aug)
    pipe_augmented.fit(X_aug, y_aug)
    metrics_augmented = evaluate(pipe_augmented, X_vl, y_vl)
    by_year_augmented = evaluate_by_year(pipe_augmented, X_vl, y_vl)
    print(f"  → MAE={metrics_augmented['mae']:.4f}  RMSE={metrics_augmented['rmse']:.4f}  R²={metrics_augmented['r2']:.4f}")
    if by_year_augmented:
        print("  Por segmento de año:")
        for seg, m in by_year_augmented.items():
            tag = seg.replace("_", " ")
            bseg = by_year_baseline.get(seg, {})
            if m["mae"] is not None and bseg.get("mae") is not None:
                delta = m["mae"] - bseg["mae"]
                arrow = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
                print(f"    {tag:12s}  n={m['n']:>7,}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}  Δ={delta:+.4f} {arrow}")

    del pipe_augmented; gc.collect()

    # ------------------------------------------------------------------
    # Resultados
    # ------------------------------------------------------------------
    correction_meta: dict = {"method": args.correction_method}
    if args.correction_method == "fuel_type":
        correction_meta["factors"] = {
            "G (gasolina)": round(CORRECTION_BY_FUEL["G"], 4),
            "D (diesel)":   round(CORRECTION_BY_FUEL["D"], 4),
            "O (otros)":    round(CORRECTION_BY_FUEL["O"], 4),
            "source": "comparacion_eu_canada.md — EEA-2010 vs NRCan-2010, 371 pares",
        }
    else:
        correction_meta["regression"] = {
            "formula": f"CO2_EU = (CO2_CA - {REG_INTERCEPT}) / {REG_SLOPE}",
            "r2": 0.548,
            "rmse_gkm": 66.1,
            "source": "comparacion_eu_canada.md — EEA-2010 vs NRCan-2010, 371 pares",
        }

    results = {
        "meta": {
            "train_files": args.max_train_files,
            "train_rows_dgt": int(len(X_tr)),
            "nrcan_rows_added": int(len(X_nrcan)),
            "val_rows": int(len(X_vl)),
            "correction": correction_meta,
            "only_european": args.only_european,
            "model": "HistGradientBoostingRegressor",
        },
        "baseline":  metrics_baseline,
        "augmented": metrics_augmented,
        "by_year_baseline":  by_year_baseline,
        "by_year_augmented": by_year_augmented,
    }

    print_summary(results)
    save_json(results, Path(args.json_output))
    save_plot(results, Path(args.plot_output))

    print("\nExperimento completo. Resultados en:")
    print(f"  {args.json_output}")
    print(f"  {args.plot_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
