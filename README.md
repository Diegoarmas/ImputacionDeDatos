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
- [tests/](tests/): tests unitarios para los modulos principales.

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
  --simplificado \
  --sep , \
  --encoding utf-8
```

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

## Notas de calidad de datos

- El cargador en [src/data_cleaning.py](src/data_cleaning.py) intenta leer el CSV de forma estandar.
- Si encuentra filas partidas o malformadas, aplica una reparacion de respaldo para continuar el proceso.

## Tests

Ejecutar tests unitarios:

```bash
python -m pytest tests/ -v
```

