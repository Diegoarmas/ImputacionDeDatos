# ImputacionDeDatos

Pipeline en Python para imputar valores faltantes de EMISIONES_CO2 en datos de vehiculos.

## Estructura actual

- [src/imputacion_co2_ml.py](src/imputacion_co2_ml.py): script principal (orquesta carga, entrenamiento, imputacion y guardado).
- [src/data_cleaning.py](src/data_cleaning.py): funciones de limpieza y reparacion de CSV malformado.
- [src/modeling.py](src/modeling.py): construccion del pipeline y evaluacion del modelo.
- [src/depuracion_txt.py](src/depuracion_txt.py): utilidades para depurar/convertir TXT por bloques.
- [data/processed/muestra_50k.csv](data/processed/muestra_50k.csv): dataset de entrada para imputacion.
- [artifacts/models/](artifacts/models/): modelos serializados.
- [artifacts/metrics/](artifacts/metrics/): metricas en JSON.

## Requisitos

- Python 3.12+
- Dependencias en [requirements.txt](requirements.txt)

Instalacion:

```bash
./.venv/bin/pip install -r requirements.txt
```

## Paso 1: depurar desde raw (solo filas con CO2 informado)

Comando recomendado:

```bash
./.venv/bin/python src/depuracion_txt.py \
  --input data/raw/muestra_50k.txt \
  --output data/processed/muestra_50k_con_co2.csv \
  --in-sep '|' \
  --out-sep ,
```

Este paso genera un CSV en el que solo se conservan filas con valor en `EMISIONES_CO2`.

## Paso 2: entrenar e imputar aplicando % de missing artificial

Comando recomendado:

```bash
./.venv/bin/python src/imputacion_co2_ml.py \
  --input data/processed/muestra_50k_con_co2.csv \
  --output data/processed/muestra_50k_co2_imputado.csv \
  --missing-rate 20 \
  --device cuda \
  --simplificado \
  --sep , \
  --encoding utf-8
```

Notas de dispositivo:

- `--device cuda`: usa GPU con XGBoost (recomendado si tienes NVIDIA compatible).
- `--device cpu`: usa el modelo de scikit-learn en CPU.

Ayuda de opciones:

```bash
./.venv/bin/python src/imputacion_co2_ml.py --help
```

Salidas por defecto:

- CSV imputado: [data/processed/muestra_50k_co2_imputado.csv](data/processed/muestra_50k_co2_imputado.csv)
- CSV simplificado (si usas `--simplificado`): [data/processed/datos_simpl.csv](data/processed/datos_simpl.csv)
- Modelo: [artifacts/models/co2_model.joblib](artifacts/models/co2_model.joblib)
- Metricas: [artifacts/metrics/co2_metrics.json](artifacts/metrics/co2_metrics.json)

Columnas agregadas en el CSV de salida:

- EMISIONES_CO2_COMPLETA: valor original de CO2 (base completa).
- EMISIONES_CO2_CON_MISSING_PCT: CO2 tras aplicar el porcentaje de missing.
- EMISIONES_CO2_IMPUTADA: resultado final tras imputar los faltantes artificiales.

## Paso 3: generar tablas, graficas y log de experimentos

El modulo [src/results_pipeline.py](src/results_pipeline.py) crea automaticamente:

- Tabla plana de resultados por experimento.
- Tabla pivote de comparacion de MSE por imputador y missing rate.
- Graficas MSE y MAE vs missing rate.
- Log con fecha, imputadores, missing rates y numero de experimentos.

Comando:

```bash
./.venv/bin/python src/results_pipeline.py
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

