[![CI](https://github.com/Diegoarmas/ImputacionDeDatos/actions/workflows/ci.yml/badge.svg)](https://github.com/Diegoarmas/ImputacionDeDatos/actions/workflows/ci.yml)

# ImputacionDeDatos

Pipeline en Python para imputar valores faltantes de `EMISIONES_CO2` en el registro del parque de vehículos español (38,6 millones de filas). Utiliza un modelo de regresión supervisada basado en Gradient Boosting que alcanza **MAE ≈ 3 g/km** y **R² ≈ 0,97**.

---

## Análisis de variables y selección de modelo

### Problema

**Regresión supervisada**: predecir `EMISIONES_CO2` (variable continua, 0–204 g/km) a partir de las características técnicas y administrativas de cada vehículo.

La distribución del target es **multimodal y con discontinuidades**:
- Vehículos eléctricos → CO2 = 0 (ruptura dura)
- Motocicletas → 50–120 g/km
- Turismos gasolina/diesel → 90–180 g/km
- Furgonetas/camiones → 120–204+ g/km

### Tipología de variables (45 features)

| Tipo | Variables | Notas |
|---|---|---|
| **Continuas** | TARA, PESO_MAX, CILINDRADA, KW, POTENCIA, CONSUMO, AUTONOMIA, DISTANCIA_EJES, EJE_ANTERIOR, EJE_POSTERIOR | Escalas muy distintas (kg, cc, kW) |
| **Discretas** | PLAZAS, PLAZAS_MAX, PLAZAS_PIE | Rango 1–7, tratadas como numéricas |
| **Nominales alta cardinalidad** | FABRICANTE, MARCA, MODELO, VARIANTE, VERSION | Cientos de valores únicos |
| **Nominales baja cardinalidad** | PROPULSION, ALIMENTACION, CARROCERIA, TIPO_DGT, CLASE_MATR | Pocas categorías |
| **Ordinales** | CAT_EURO, EMISIONES_EURO | Euro I–VI, tienen orden lógico |
| **Temporales** | FECHA_MATR, FEC_PRIM_MATR | Año y mes extraídos como features numéricas |
| **Geográficas** | PROVINCIA, MUNICIPIO | Alta cardinalidad nominal |

### Relaciones físicas clave

```
ALIMENTACION / PROPULSION → CO2    # eléctrico = 0, diesel ≠ gasolina
CONSUMO       → CO2                # CO2 ≈ 23,5 × L/100km (gasolina), 26,5 × L/100km (diesel)
CILINDRADA    → KW → CO2           # motor más grande → más potencia → más emisiones
CAT_EURO ↑    → CO2 ↓             # normas más recientes = vehículos más limpios
TARA / PESO   → CO2               # más pesado → más consumo
AÑO_MATR ↑   → CO2 ↓             # vehículos más nuevos son más eficientes
MARCA / MODELO → CO2              # efectos de fabricante
```

Existe **multicolinealidad alta** entre CILINDRADA, KW, POTENCIA y TARA. Los modelos de árbol la gestionan de forma natural, a diferencia de la regresión lineal (que requeriría regularización o PCA previo).

### Comparativa de modelos

| Modelo | Apto | Motivo |
|---|---|---|
| **Regresión Lineal** | No | Asume linealidad. PROPULSION=Eléctrico crea una ruptura dura (CO2=0) que una recta no puede modelar. Con MODELO/MARCA habría explosión dimensional tras one-hot encoding. |
| **KNN** | No | Maldición de la dimensionalidad con 45+ features. Distancia euclidiana no tiene sentido mezclando kg, cc y categorías. O(n) en inferencia, inviable con millones de filas. |
| **Árbol de decisión simple** | Solo baseline | Interpretable, capta no-linealidades, pero alta varianza y sobreajuste con tantas categorías. |
| **Random Forest** | Sí, alternativa | Reduce varianza por bagging. Mayor memoria y latencia que Gradient Boosting; peor con alta cardinalidad. |
| **Gradient Boosting (HistGBT / XGBoost / LightGBM)** | **Sí — opción óptima** | Estado del arte en tabular heterogéneo. Maneja natively missings, alta cardinalidad (OrdinalEncoder), interacciones (CILINDRADA × ALIMENTACION) y escala a millones de filas. GPU disponible vía XGBoost. |
| **Redes Neuronales (MLP)** | No recomendado | Para datos tabulares heterogéneos, Gradient Boosting supera sistemáticamente a MLP. Requieren normalización exhaustiva y más hiperparámetros. |
| **SVR** | No | Complejidad O(n²–n³), inviable con millones de registros. |
| **Modelos no supervisados** | Solo complemento | Clustering previo por tipo de vehículo podría estratificar el problema, pero no son el modelo principal dado que disponemos de etiquetas. |

### Conclusión

**`HistGradientBoostingRegressor` es la elección correcta** por la combinación de:

- Target continuo con distribución multimodal y discontinuidades (eléctrico vs. combustión)
- 45 features heterogéneas (continuas + nominales de alta cardinalidad + ordinales + temporales)
- Valores faltantes presentes en múltiples columnas
- Escala de producción (decenas de millones de filas)

La única alternativa que merece evaluación experimental es **LightGBM** con soporte nativo de categoricals, que puede dar mejoras marginales en features de alta cardinalidad como MODELO o MUNICIPIO.

---

## Estructura actual

- [src/imputacion_co2_ml.py](src/imputacion_co2_ml.py): script principal (orquesta carga, entrenamiento, imputacion y guardado).
- [src/data_cleaning.py](src/data_cleaning.py): funciones de limpieza y reparacion de CSV malformado.
- [src/modeling.py](src/modeling.py): construccion del pipeline y evaluacion del modelo (CPU con HistGradientBoosting o GPU con XGBoost via `--device`).
- [src/depuracion_txt.py](src/depuracion_txt.py): utilidades para depurar/convertir TXT por bloques y separar train/val/test.
- [src/compare_imputers.py](src/compare_imputers.py): compara varias técnicas de imputación (mean/median/linear/knn/rf/hgbt).
- [src/brand_model_analysis.py](src/brand_model_analysis.py): estudio de influencia de MARCA/MODELO en el error de imputación (ver [ANALISIS_MARCA_MODELO.md](ANALISIS_MARCA_MODELO.md)).
- [src/nrcan_augment_experiment.py](src/nrcan_augment_experiment.py): experimento de aumento de datos con NRCan Canadá (ver [EXPERIMENTO_NRCAN.md](EXPERIMENTO_NRCAN.md)).
- [src/results_pipeline.py](src/results_pipeline.py): utilidad opcional de post-procesado (tablas/plots/log a partir de una lista de resultados); no forma parte del pipeline DVC actual.
- [generate_pools.sh](generate_pools.sh): script para generar automáticamente pools separados train/val/test.
- [data/raw/](data/raw/): datos originales (parque_vehiculos_202503.txt)
- [data/processed/pool_train/](data/processed/pool_train/): muestras para **ENTRENAMIENTO** (70% por defecto)
- [data/processed/pool_val/](data/processed/pool_val/): muestras para **VALIDACIÓN** (15% por defecto)
- [data/processed/pool_test/](data/processed/pool_test/): muestras para **PRUEBA** (15% por defecto, sin overlap con train/val)
- [artifacts/models/](artifacts/models/): modelos serializados.
- [artifacts/metrics/](artifacts/metrics/): metricas en JSON.
- [results/](results/): tablas y gráficas generadas por `compare_imputers.py` y `brand_model_analysis.py`.
- [tests/](tests/): tests unitarios para los modulos principales.

Documentación adicional: [DOCUMENTACION_MODELO.md](DOCUMENTACION_MODELO.md) (detalle interno del modelo), [ANALISIS_MARCA_MODELO.md](ANALISIS_MARCA_MODELO.md), [EXPERIMENTO_NRCAN.md](EXPERIMENTO_NRCAN.md), [EstudioVarYModelos.md](EstudioVarYModelos.md), [RESULTADOS_MODELO.md](RESULTADOS_MODELO.md).

## Requisitos

- Python 3.12+
- Dependencias en [requirements.txt](requirements.txt)

Instalacion:

```bash
./.venv/bin/pip install -r requirements.txt
```

## Reproducibilidad (DVC)

El pipeline completo (depuración → entrenamiento → comparación de imputadores → análisis marca/modelo) está definido como un DAG en [dvc.yaml](dvc.yaml), parametrizado vía [params.yaml](params.yaml).

```bash
# Ejecutar todo el pipeline (solo corre los stages cuyas deps/params cambiaron)
./.venv/bin/dvc repro

# Ver el grafo de stages
./.venv/bin/dvc dag

# Ver/comparar métricas trackeadas por DVC (co2_metrics.json, imputer_comparison.json, etc.)
./.venv/bin/dvc metrics show
./.venv/bin/dvc metrics diff

# Probar otro hiperparámetro y reproducir solo lo afectado
# (editar params.yaml, p.ej. train.missing_rate)
./.venv/bin/dvc repro
```

El dataset crudo (`data/raw/parque_vehiculos_202503.txt`) está versionado con `dvc add` (ver `data/raw/*.dvc`); el binario no se sube a git, solo su hash. No hay remote DVC configurado por defecto — para compartir datos entre máquinas, añadir uno con `dvc remote add`.

## Paso 1: Generar pools separados de TRAIN, VAL y TEST

Separar automáticamente los datos en pool_train (70%), pool_val (15%) y pool_test (15%)
para garantizar que no haya overlap entre entrenamiento, validación y evaluación. El reparto es
**determinístico** (bucketing por índice de fila módulo 10 000), garantizando reproducibilidad.

### Opción A: Usando el script automatizado (recomendado)

```bash
chmod +x generate_pools.sh
./generate_pools.sh
```

Esto genera:
- `data/processed/pool_train/`: ~70% de los datos para entrenar
- `data/processed/pool_val/`: ~15% de los datos para validar durante el desarrollo
- `data/processed/pool_test/`: ~15% de los datos para evaluar el modelo final
- `artifacts/metrics/split_period_report.json`: cobertura temporal de cada split

### Opción B: Comando manual

```bash
./.venv/bin/python src/depuracion_txt.py \
  --input data/raw/parque_vehiculos_202503.txt \
  --pool-dir data/processed/pool_train \
  --pool-val-dir data/processed/pool_val \
  --pool-test-dir data/processed/pool_test \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --rows-per-sample 200000 \
  --keep-remainder \
  --in-sep '|' \
  --out-sep ',' \
  --in-encoding 'iso-8859-1' \
  --out-encoding 'utf-8'
```

Parámetros clave:
- `--train-ratio` / `--val-ratio`: proporciones para train y validación (default: 0.7 / 0.15; el resto, 0.15, va a test)
- `--pool-dir` / `--pool-val-dir` / `--pool-test-dir`: directorios de salida para ENTRENAMIENTO, VALIDACIÓN y PRUEBA
- `--rows-per-sample`: filas por archivo CSV generado (default: 200 000)
- `--keep-remainder`: guarda las filas restantes que no completan una muestra

## Paso 2: Entrenar e imputar con datos de TRAIN

Entrenar el modelo con pool_train (validación cruzada) y evaluarlo también en pool_val y pool_test:

```bash
./.venv/bin/python src/imputacion_co2_ml.py \
  --input-dir data/processed/pool_train \
  --val-dir data/processed/pool_val \
  --test-dir data/processed/pool_test \
  --missing-rate 0.2 \
  --device cpu \
  --simplificado \
  --sep ',' \
  --encoding utf-8
```

Parámetros clave:
- `--input-dir`: pool de ENTRENAMIENTO (default: `data/processed/pool_train`)
- `--val-dir`: pool de VALIDACIÓN (default: `data/processed/pool_val`; opcional, omitir para saltar esa evaluación)
- `--test-dir`: pool de TEST **separado** para evaluar el modelo final (opcional pero recomendado)
- `--missing-rate`: proporción de CO2 a ocultar para imputación artificial (0–1)
- `--device`: `cpu` (HistGradientBoostingRegressor) o `cuda` (XGBoost en GPU)
- `--simplificado`: genera CSV con solo filas imputadas

**Salidas generadas**:
- Modelo entrenado: `artifacts/models/co2_model.joblib`
- Métricas (train + val + test): `artifacts/metrics/co2_metrics.json`
- Reporte temporal por split: `artifacts/metrics/period_split_report.json`
- CSV imputado (train): `data/processed/22_05.csv` (ruta configurable con `--output`)
- CSV simplificado: `data/processed/simplificado/datos_simpl.csv` (si `--simplificado`)

Las métricas JSON incluirán:
- Métricas de validación cruzada (train)
- Métricas de evaluación en val/test (si `--val-dir`/`--test-dir` se proporcionan)

## Paso 3: comparar técnicas de imputación

[src/compare_imputers.py](src/compare_imputers.py) evalúa por validación cruzada varias técnicas
(`mean`, `median`, `linear`, `knn`, `rf`, `hgbt`) sobre pool_train con un missing rate artificial fijo:

```bash
./.venv/bin/python src/compare_imputers.py \
  --input-dir data/processed/pool_train \
  --missing-rate 0.2 \
  --cv-folds 5 \
  --knn-neighbors 5
```

Salidas:

- [artifacts/metrics/imputer_comparison.json](artifacts/metrics/imputer_comparison.json): metadatos + métricas por técnica.
- [results/tables/imputer_comparison.csv](results/tables/imputer_comparison.csv): tabla ordenada por MAE.
- [results/plots/imputer_comparison.png](results/plots/imputer_comparison.png): comparación gráfica de MAE/RMSE/R² por técnica.

## Paso 4: análisis por marca/modelo (opcional)

[src/brand_model_analysis.py](src/brand_model_analysis.py) evalúa el error del modelo entrenado
desagregado por MARCA y MODELO, y un estudio de ablación con/sin esas variables. Ver
[ANALISIS_MARCA_MODELO.md](ANALISIS_MARCA_MODELO.md) para el detalle completo de uso e interpretación.

## Como interpretar la comparación de imputadores

### 1) Tabla de resultados

Archivo: [results/tables/imputer_comparison.csv](results/tables/imputer_comparison.csv)

- Cada fila es una técnica (`mean`, `median`, `linear`, `knn`, `rf`, `hgbt`) con su `mae`, `rmse`, `r2` (y desviaciones estándar de CV).
- Menor `mae`/`rmse` y mayor `r2` significan mejor imputación.
- Ordenada por `mae` ascendente: la primera fila es la mejor técnica al `missing_rate` evaluado.

### 2) Plot comparativo

Archivo: [results/plots/imputer_comparison.png](results/plots/imputer_comparison.png)

- Tres subgráficas de barras (MAE, RMSE, R²) con una barra por técnica.
- Permite ver de un vistazo si el ranking por MAE se mantiene en RMSE y R².

## Criterio practico para elegir imputador

- Primero, compara MAE y RMSE al mismo `missing_rate`.
- Luego, revisa si el ranking se mantiene al repetir la comparación con otro `missing_rate`
  (editando `compare.missing_rate` en [params.yaml](params.yaml) y volviendo a ejecutar).
- El mejor candidato suele ser el que combina error bajo y ranking estable entre missing rates.

## Notas de calidad de datos

- El cargador en [src/data_cleaning.py](src/data_cleaning.py) intenta leer el CSV de forma estandar.
- Si encuentra filas partidas o malformadas, aplica una reparacion de respaldo para continuar el proceso.

## Tests

Ejecutar tests unitarios:

```bash
python -m pytest tests/ -v
```

