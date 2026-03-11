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

## Ejecutar imputacion de CO2

Comando recomendado:

```bash
./.venv/bin/python src/imputacion_co2_ml.py \
  --input data/processed/muestra_50k.csv \
  --output data/processed/muestra_50k_co2_imputado.csv \
  --sep , \
  --encoding utf-8
```

Ayuda de opciones:

```bash
./.venv/bin/python src/imputacion_co2_ml.py --help
```

Salidas por defecto:

- CSV imputado: [data/processed/muestra_50k_co2_imputado.csv](data/processed/muestra_50k_co2_imputado.csv)
- Modelo: [artifacts/models/co2_model.joblib](artifacts/models/co2_model.joblib)
- Metricas: [artifacts/metrics/co2_metrics.json](artifacts/metrics/co2_metrics.json)

Columnas agregadas en el CSV de salida:

- EMISIONES_CO2_NUM: valor original convertido a numerico.
- EMISIONES_CO2_IMPUTADA: valor final (real o estimado).
- EMISIONES_CO2_ESTIMADA_POR_ML: 1 si fue imputado, 0 si ya venia informado.

## Notas de calidad de datos

- El cargador en [src/data_cleaning.py](src/data_cleaning.py) intenta leer el CSV de forma estandar.
- Si encuentra filas partidas o malformadas, aplica una reparacion de respaldo para continuar el proceso.

