# ImputacionDeDatos

Script en Python para procesar archivos grandes por bloques (chunks) y convertirlos a CSV con separador configurable.

## Archivos del proyecto

- `deouracion_txt.py`: script principal de procesamiento.
- `imputacion_co2_ml.py`: entrenamiento e imputacion de `EMISIONES_CO2` faltante.
- `muestra_50k.txt`: muestra de datos (separador `|`, encoding `iso-8859-1`).
- `parque_vehiculos_202503.txt`: dataset original grande.

## Requisitos

- Python 3.12+
- dependencias en `requirements.txt` instaladas en el entorno virtual del proyecto

Instalar dependencias:

```bash
./.venv/bin/pip install -r requirements.txt
```

## Uso rapido

Ejecutar con valores por defecto:

```bash
./.venv/bin/python deouracion_txt.py
```

Valores por defecto del script:

- input: `muestra_50k.txt`
- output: `dataset_limpio.csv`
- in-sep: `|`
- out-sep: `,`
- in-encoding: `iso-8859-1`
- out-encoding: `utf-8`
- chunksize: `100000`

## Opciones disponibles

```bash
./.venv/bin/python deouracion_txt.py --help
```

Parametros principales:

- `--input`: archivo de entrada
- `--output`: archivo de salida
- `--in-sep`: separador de entrada
- `--out-sep`: separador de salida
- `--in-encoding`: codificacion de entrada
- `--out-encoding`: codificacion de salida
- `--chunksize`: filas por bloque
- `--columns`: columnas a conservar, separadas por coma

## Ejemplos

Convertir de `|` a `;`:

```bash
./.venv/bin/python deouracion_txt.py \
  --input muestra_50k.txt \
  --output salida.csv \
  --in-sep '|' \
  --out-sep ';' \
  --in-encoding iso-8859-1 \
  --out-encoding utf-8 \
  --chunksize 20000
```

Conservar solo algunas columnas:

```bash
./.venv/bin/python deouracion_txt.py \
  --input muestra_50k.txt \
  --output salida_filtrada.csv \
  --in-sep '|' \
  --columns PROVINCIA,MARCA,MODELO
```

## Notas

- El script escribe la salida por bloques para reducir uso de memoria.
- Si hay error de codificacion, prueba cambiar `--in-encoding`.
- Si ejecutas `python3 deouracion_txt.py` y falla por pandas, usa el entorno virtual del proyecto:

```bash
./.venv/bin/python deouracion_txt.py
```

## ML para imputar CO2 faltante

El script `imputacion_co2_ml.py`:

- toma filas con `EMISIONES_CO2` conocido para entrenar un modelo de regresion,
- evalua metricas en un conjunto de validacion,
- predice `EMISIONES_CO2` donde falta,
- guarda CSV imputado, modelo entrenado y metricas JSON.

Ejecucion recomendada:

```bash
./.venv/bin/python imputacion_co2_ml.py \
  --input muestra_50k.csv \
  --output muestra_50k_co2_imputado.csv \
  --sep ',' \
  --encoding utf-8
```

Artefactos que genera por defecto:

- `muestra_50k_co2_imputado.csv`
- `co2_model.joblib`
- `co2_metrics.json`

Columnas nuevas en el CSV imputado:

- `EMISIONES_CO2_NUM`: objetivo original convertido a numerico.
- `EMISIONES_CO2_IMPUTADA`: objetivo final (real o estimado).
- `EMISIONES_CO2_ESTIMADA_POR_ML`: `1` si fue imputado por el modelo, `0` si venia informado.
