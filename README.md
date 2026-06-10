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
- [src/modeling.py](src/modeling.py): construccion del pipeline y evaluacion del modelo.
- [src/depuracion_txt.py](src/depuracion_txt.py): utilidades para depurar/convertir TXT por bloques y separar train/test.
- [generate_pools.sh](generate_pools.sh): script para generar automáticamente pools separados train/test.
- [data/raw/](data/raw/): datos originales (parque_vehiculos_202503.txt)
- [data/processed/pool_train/](data/processed/pool_train/): muestras para **ENTRENAMIENTO** (80%)
- [data/processed/pool_test/](data/processed/pool_test/): muestras para **PRUEBA** (20%, sin overlap con train)
- [artifacts/models/](artifacts/models/): modelos serializados.
- [artifacts/metrics/](artifacts/metrics/): metricas en JSON.
- [tests/](tests/): tests unitarios para los modulos principales.

## Requisitos

- Python 3.12+
- Dependencias en [requirements.txt](requirements.txt)

Instalacion:

```bash
./.venv/bin/pip install -r requirements.txt
```

## Paso 1: Generar pools separados de TRAIN y TEST

**Nuevo flujo recomendado**: Separar automáticamente los datos en pool_train (80%) y pool_test (20%) 
para garantizar que no haya overlap entre entrenamiento y evaluación.

### Opción A: Usando el script automatizado (recomendado)

```bash
chmod +x generate_pools.sh
./generate_pools.sh
```

Esto genera:
- `data/processed/pool_train/`: ~80% de los datos para entrenar
- `data/processed/pool_test/`: ~20% de los datos para evaluar

La separación es **determinística** basada en el índice de fila, garantizando reproducibilidad y sin overlap.

### Opción B: Comando manual

```bash
./.venv/bin/python src/depuracion_txt.py \
  --input data/raw/parque_vehiculos_202503.txt \
  --pool-dir data/processed/pool_train \
  --pool-test-dir data/processed/pool_test \
  --train-ratio 0.8 \
  --rows-per-sample 100000 \
  --keep-remainder \
  --in-sep '|' \
  --out-sep ',' \
  --in-encoding 'iso-8859-1' \
  --out-encoding 'utf-8'
```

Parámetros clave:
- `--train-ratio`: proporción de datos para entrenamiento (default: 0.8 = 80% train, 20% test)
- `--pool-dir`: directorio de salida para pool de ENTRENAMIENTO
- `--pool-test-dir`: directorio de salida para pool de PRUEBA
- `--keep-remainder`: guarda las filas restantes que no completan una muestra

## Paso 2: Entrenar e imputar con datos de TRAIN

Entrenar el modelo **solo** con pool_train (sin tocar pool_test):

```bash
./.venv/bin/python src/imputacion_co2_ml.py \
  --input-dir data/processed/pool_train \
  --test-dir data/processed/pool_test \
  --missing-rate 20 \
  --simplificado \
  --sep ',' \
  --encoding utf-8
```

Parámetros clave:
- `--input-dir`: pool de ENTRENAMIENTO (default: `data/processed/pool_train`)
- `--test-dir`: pool de TEST **separado** para evaluar el modelo final (opcional pero recomendado)
- `--missing-rate`: porcentaje de CO2 a ocultar para imputación artificial
- `--simplificado`: genera CSV con solo filas imputadas

**Salidas generadas**:
- Modelo entrenado: `artifacts/models/co2_model.joblib`
- Métricas (incluyendo test): `artifacts/metrics/co2_metrics.json`
- CSV imputado (train): `data/processed/muestra_50k_co2_imputado.csv`
- CSV simplificado: `data/processed/datos_simpl.csv` (si `--simplificado`)

Las métricas JSON incluirán:
- Métricas de validación cruzada (train)
- Métricas de evaluación en test set (si `--test-dir` se proporciona)

## Paso 3: generar tablas, graficas y log de experimentos

El modulo [src/results_pipeline.py](src/results_pipeline.py) crea automaticamente:

- Tabla plana de resultados por experimento.
- Tabla pivote de comparacion de MSE por imputador y missing rate.
- Graficas MSE y MAE vs missing rate.
- Log con fecha, imputadores, missing rates y numero de experimentos.

Por defecto, toma metricas reales desde
[artifacts/metrics/co2_metrics.json](artifacts/metrics/co2_metrics.json)
(generado por [src/imputacion_co2_ml.py](src/imputacion_co2_ml.py)) y construye
las tablas/plots para el modelo unico actual (`HistGradientBoostingRegressor`).

Comando (modo real por defecto):

```bash
./.venv/bin/python src/results_pipeline.py
```

CSV real de resultados (si se define, tiene prioridad sobre --mode):

```bash
./.venv/bin/python src/results_pipeline.py \
  --input results/tables/experiment_results.csv
```

Modo demo (datos ficticios de ejemplo):

```bash
./.venv/bin/python src/results_pipeline.py --mode demo
```

Salidas:

- [results/tables/experiment_results.csv](results/tables/experiment_results.csv)
- [results/tables/mse_comparison.csv](results/tables/mse_comparison.csv)
- [results/plots/mse_vs_missing_rate.png](results/plots/mse_vs_missing_rate.png)
- [results/plots/mae_vs_missing_rate.png](results/plots/mae_vs_missing_rate.png)
- [results/logs/experiment_log.txt](results/logs/experiment_log.txt)

## Como interpretar tablas y plots

### 1) Tabla de experimentos

Archivo: [results/tables/experiment_results.csv](results/tables/experiment_results.csv)

- Cada fila representa un experimento (`imputer`, `missing_rate`, `mse`, `mae`).
- En el flujo actual, normalmente veras una sola fila/modelo
  (`HistGradientBoostingRegressor`) en modo real.
- Menor `mse` y menor `mae` significan mejor imputacion.
- Sirve para comparar metodos en un missing rate especifico.

### 2) Tabla pivote de MSE

Archivo: [results/tables/mse_comparison.csv](results/tables/mse_comparison.csv)

- Filas: imputadores.
- Columnas: niveles de `missing_rate`.
- Valores: MSE.
- Regla rapida: cuanto menor sea el valor, mejor comportamiento del imputador.

### 3) Plot de MSE

Archivo: [results/plots/mse_vs_missing_rate.png](results/plots/mse_vs_missing_rate.png)

- Muestra como cambia el error cuadratico al aumentar faltantes.
- Linea mas baja: mejor precision global.
- Pendiente mas suave: mayor robustez al missing.

### 4) Plot de MAE

Archivo: [results/plots/mae_vs_missing_rate.png](results/plots/mae_vs_missing_rate.png)

- Mide error absoluto medio, mas interpretable en unidades de CO2.
- Si MSE sube mucho mas que MAE, puede haber errores grandes puntuales.

### 5) Log de experimento

Archivo: [results/logs/experiment_log.txt](results/logs/experiment_log.txt)

- Resume trazabilidad de ejecucion.
- Incluye fecha/hora, metodos evaluados y cobertura de experimentos.

## Criterio practico para elegir imputador

- Primero, compara MAE y MSE al mismo `missing_rate`.
- Luego, revisa si el ranking se mantiene cuando sube el missing.
- El mejor candidato suele ser el que combina error bajo y estabilidad de curva.

## Notas de calidad de datos

- El cargador en [src/data_cleaning.py](src/data_cleaning.py) intenta leer el CSV de forma estandar.
- Si encuentra filas partidas o malformadas, aplica una reparacion de respaldo para continuar el proceso.

## Tests

Ejecutar tests unitarios:

```bash
python -m pytest tests/ -v
```

