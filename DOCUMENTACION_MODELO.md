# Documentacion Detallada Del Modelo De Imputacion CO2

## 1. Objetivo del proyecto

El objetivo es imputar valores faltantes de `EMISIONES_CO2` usando un modelo supervisado.

El flujo actual tiene dos etapas:

1. Depurar datos crudos y conservar solo filas con CO2 informado.
2. Simular faltantes (missing rate), entrenar el modelo con las filas no ocultadas e imputar las filas ocultadas.

Esto permite evaluar de forma controlada que tan bien recupera el modelo valores de CO2.

## 2. Archivos principales y rol de cada uno

- `src/depuracion_txt.py`: convierte el archivo raw por bloques y filtra filas con CO2 informado.
- `src/data_cleaning.py`: lectura robusta del CSV, limpieza y tipado de columnas.
- `src/modeling.py`: construye pipeline de preprocesado + modelo y calcula metricas con validacion cruzada.
- `src/imputacion_co2_ml.py`: orquestador principal (carga, missing artificial, entrenamiento, imputacion, guardado).

## 3. Flujo end-to-end

### 3.1 Depuracion del raw

Script: `src/depuracion_txt.py`

Entrada tipica:

- `data/raw/muestra_50k.txt`

Salida tipica:

- `data/processed/muestra_50k_con_co2.csv`

Que hace:

1. Lee por chunks para no cargar todo a memoria.
2. Opcionalmente reduce columnas (`--columns`).
3. Filtra filas donde `EMISIONES_CO2` esta informado (no vacio, no `nan`, no `none`, no `null`).
4. Escribe CSV final en UTF-8.

## 4. Limpieza y preparacion de features

Modulo: `src/data_cleaning.py`

### 4.1 Constantes de esquema

- `TARGET_COLUMN = "EMISIONES_CO2"`
- `DATE_COLUMNS = ["FECHA_MATR", "FEC_PRIM_MATR"]`
- `KNOWN_NUMERIC_COLUMNS`: lista de columnas esperadas como numericas.

### 4.2 Normalizacion numerica

Funcion: `_normalize_number_string(value)`

Problema que resuelve:

- El dataset mezcla formatos numericos con coma y punto.

Reglas aplicadas:

1. Si hay coma y punto:
- Si la coma aparece mas a la derecha, se interpreta como decimal europeo.
- Si el punto aparece mas a la derecha, se asume punto decimal y se eliminan comas de miles.
2. Si solo hay coma:
- Se interpreta como decimal y se convierte a punto.
3. Si solo hay punto con patron de miles (`1.234.567`):
- Se quitan puntos.

Luego `to_float_series` convierte el resultado a float con `pd.to_numeric(errors="coerce")`.

### 4.3 Construccion de matriz de modelado

Funcion: `prepare_dataframe(df)`

Pasos:

1. Separa target (`EMISIONES_CO2`) de las features.
2. Convierte fechas a:
- `*_YEAR`
- `*_MONTH`
3. Convierte columnas numericas conocidas con `to_float_series`.
4. Normaliza texto en categoricas (trim, vacios a NaN).

Retorna:

- `data`: DataFrame de features listo para el pipeline.
- `target`: Serie numerica de CO2.

### 4.4 Carga robusta de CSV

Funcion: `load_csv_resilient(path, sep, encoding)`

1. Intenta `pd.read_csv` normal.
2. Si falla por `ParserError`, usa fallback `_repaired_rows`:
- Repara filas partidas en multiples lineas fisicas.
- Descarta filas con mas columnas de las esperadas.

Esto reduce caidas del proceso por formato irregular de entrada.

## 5. Modelo y evaluacion

Modulo: `src/modeling.py`

### 5.1 Deteccion automatica de tipos

Funcion: `build_pipeline(feature_df, random_state)`

- `numeric_columns`: columnas con dtype numerico.
- `categorical_columns`: resto de columnas.

### 5.2 Preprocesado

Bloque numerico:

- `SimpleImputer(strategy="median")`

Bloque categorico:

- `SimpleImputer(strategy="constant", fill_value="DESCONOCIDO")`
- `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`

Union de ambos:

- `ColumnTransformer` con transformadores `num` y `cat`.

### 5.3 Modelo final

- `HistGradientBoostingRegressor`

Hiperparametros actuales:

- `max_iter=350`
- `learning_rate=0.05`
- `max_depth=8`
- `min_samples_leaf=20`
- `random_state` configurable por CLI

### 5.4 Evaluacion

Funcion: `fit_and_evaluate(...)`

- Usa `KFold` con `shuffle=True` y semilla fija.
- Usa `cross_validate` con metricas:
- MAE (`neg_mean_absolute_error`)
- RMSE (`neg_root_mean_squared_error`)
- R2

Retorna promedio y desviacion estandar de cada metrica.

## 6. Orquestacion principal

Script: `src/imputacion_co2_ml.py`

### 6.1 Parametros CLI relevantes

- `--input`: CSV con CO2 conocido (por defecto `muestra_50k_con_co2.csv`).
- `--output`: CSV completo con columnas de CO2 e imputacion.
- `--missing-rate`: porcentaje/proporcion a ocultar (`0.2` o `20`).
- `--cv-folds`: folds de validacion cruzada.
- `--random-state`: semilla global del proceso.
- `--simplificado`: genera CSV adicional con solo filas ocultadas.
- `--simplificado-output`: ruta de ese CSV simplificado.

### 6.2 Missing artificial controlado

1. Se parte de `co2_complete` (target completo conocido).
2. Se calcula `missing_count = round(n_filas * missing_rate)`.
3. Se seleccionan indices aleatorios sin reemplazo.
4. Esos indices se ponen en NaN dentro de `co2_with_missing`.

Interpretacion:

- `co2_complete`: verdad de referencia.
- `co2_with_missing`: escenario con faltantes simulados.

### 6.3 Entrenamiento e imputacion

1. Se entrena con filas no NaN de `co2_with_missing`.
2. Se imputan solo filas NaN (las ocultadas artificialmente).
3. Se guarda `imputed_values`.

### 6.4 Salidas de datos

CSV principal (`--output`) incluye:

- `EMISIONES_CO2_COMPLETA`
- `EMISIONES_CO2_CON_MISSING_PCT`
- `EMISIONES_CO2_IMPUTADA`

CSV simplificado (si `--simplificado`) incluye solo filas donde se aplico missing rate:

- `EMISIONES_CO2_COMPLETA`
- `EMISIONES_CO2_CON_MISSING_PCT` (NaN en esas filas)
- `EMISIONES_CO2_IMPUTADA`

### 6.5 Artefactos guardados

- Modelo serializado: `artifacts/models/co2_model.joblib`
- Metricas JSON: `artifacts/metrics/co2_metrics.json`

Metricas incluyen, entre otras:

- `rows_with_missing_applied`
- `missing_rate`
- `mae`, `rmse`, `r2`
- `mae_std`, `rmse_std`, `r2_std`
- datos del simplificado cuando aplica

## 7. Como ejecutar

### 7.1 Depurar raw

```bash
python3 src/depuracion_txt.py \
  --input data/raw/muestra_50k.txt \
  --output data/processed/muestra_50k_con_co2.csv \
  --in-sep '|' \
  --out-sep ','
```

### 7.2 Entrenar e imputar con 20% missing artificial

```bash
python3 src/imputacion_co2_ml.py \
  --input data/processed/muestra_50k_con_co2.csv \
  --output data/processed/muestra_50k_co2_imputado.csv \
  --missing-rate 20 \
  --cv-folds 5 \
  --random-state 42 \
  --simplificado
```

## 8. Decisiones tecnicas clave

1. Se filtran solo filas con CO2 conocido para poder evaluar imputacion de forma objetiva.
2. Se usa missing artificial para comparar imputado vs valor real conocido.
3. Se usa pipeline de sklearn para evitar fugas de preprocesado entre train/valid.
4. Se usa validacion cruzada para metricas mas estables que un unico holdout.
5. Se guarda CSV simplificado para auditoria rapida de casos imputados.

## 9. Limitaciones y recomendaciones

1. `OrdinalEncoder` en categoricas es simple y robusto, pero puede no capturar bien relaciones complejas entre categorias.
2. Si el dominio crece, probar codificacion target/catboost encoding o modelos alternativos.
3. Revisar periodicamente calidad de parseo del raw, porque filas malformadas pueden sesgar distribuciones.
4. Para comparabilidad entre corridas, mantener fijo `--random-state`.

## 10. Resumen corto

El sistema actual esta disenado para imputacion reproducible y auditable de CO2.

- Primero limpia y filtra.
- Luego simula faltantes controlados.
- Entrena y evalua con CV.
- Imputa solo lo ocultado.
- Exporta version completa y version simplificada para inspeccion.
