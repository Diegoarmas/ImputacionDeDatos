[![CI](https://github.com/Diegoarmas/ImputacionDeDatos/actions/workflows/ci.yml/badge.svg)](https://github.com/Diegoarmas/ImputacionDeDatos/actions/workflows/ci.yml)

# ImputacionDeDatos

Pipeline en Python para imputar valores faltantes de EMISIONES_CO2 en datos de vehiculos.

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

