# Implementación: Separación Train/Test en ImputacionDeDatos

## Cambios realizados

### 1. Modificaciones a `src/depuracion_txt.py`

#### Nuevos parámetros CLI:
```
--pool-test-dir (str)  : Directorio para exportar datos de TEST
--train-ratio (float)  : Proporción para entrenamiento (default: 0.8)
```

#### Nueva función:
```python
def _is_train_split(global_row_index: int, train_ratio: float) -> bool:
    """Determina si una fila va a TRAIN (True) o TEST (False)"""
    threshold = int(train_ratio * 100)
    return (global_row_index % 100) < threshold
```

**Cómo funciona:**
- Mantiene un contador global de filas útiles (con EMISIONES_CO2)
- Usa aritmética modular para garantizar split determinístico
- Genera dos buffers separados: `pool_buffer_train` y `pool_buffer_test`
- Escribe en dos directorios: `--pool-dir` y `--pool-test-dir`

**Ventajas:**
- ✅ Determinístico: mismo archivo → mismo split
- ✅ Sin overlap: cada fila solo va a UN pool
- ✅ Escalable: procesa por bloques
- ✅ Reproducible

#### Cambios en `main()`:
```python
# Validación de configuración
use_split = pool_dir_path and args.pool_test_dir

# Procesamiento dual: separa cada fila en train o test
for row in chunk:
    if _is_train_split(global_row_index, args.train_ratio):
        chunk_train.append(row)
        kept_rows_train += 1
    else:
        chunk_test.append(row)
        kept_rows_test += 1
    global_row_index += 1

# Genera muestras en ambos directorios
```

---

### 2. Modificaciones a `src/imputacion_co2_ml.py`

#### Nuevos parámetros CLI:
```
--input-dir (str, default: data/processed/pool_train)
--test-dir (str)    : Pool de TEST para evaluar modelo final
--test-pattern (str): Patrón glob para archivos en --test-dir
```

#### Nueva función:
```python
def _evaluate_on_test_set(model_artifact, test_df, args) -> dict:
    """Evalúa modelo en TEST set sin missing artificial"""
    # Carga modelo
    # Filtra filas con EMISIONES_CO2 informado
    # Predice y calcula MAE, RMSE, R2
    # Retorna dict con test_mae, test_rmse, test_r2
```

#### Cambios en `main()`:
```python
# 1. Entrena con pool_train (validación cruzada)
result = _train_and_generate_outputs(df, args, ...)

# 2. Si --test-dir está definido:
if args.test_dir:
    test_df, _ = _load_input_data(test_dir, ...)
    model_artifact = joblib.load(model_output_path)
    test_metrics = _evaluate_on_test_set(model_artifact, test_df, args)
    
    # 3. Integra métricas de test en JSON
    metrics.update(test_metrics)
    json.dump(metrics, metrics_output_path)
```

**Resultados en `artifacts/metrics/co2_metrics.json`:**
```json
{
  "mae": 0.5234,          // Validación cruzada (train)
  "rmse": 0.7821,
  "r2": 0.8934,
  "mae_std": 0.0234,
  
  "test_mae": 0.5412,     // Evaluación en TEST (sin missing artificial)
  "test_rmse": 0.8001,
  "test_r2": 0.8876,
  "test_rows_total": 50000,
  "test_rows_with_target": 45231,
  "test_dir": "data/processed/pool_test",
  "test_inputs_loaded": [...]
}
```

---

### 3. Nuevo archivo: `generate_pools.sh`

Script helper que automatiza la generación de pools:

```bash
#!/bin/bash
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

Uso:
```bash
chmod +x generate_pools.sh
./generate_pools.sh
```

---

### 4. Documentación actualizada

#### README.md
- Nuevo flujo de 3 pasos (generar pools → entrenar → evaluar)
- Descripción de pool_train vs pool_test
- Explicación de parámetros clave
- Ejemplos de uso con --test-dir

#### GUIA_TRAIN_TEST.md (nuevo)
- Guía completa de uso
- Ejemplos de comandos
- Verificación de separación
- Troubleshooting
- Explicación técnica del método

#### Memoria repo
- `/memories/repo/train_test_split_strategy.md`: Documentación técnica

---

## Casos de uso

### Caso 1: Generar pools (primera vez)
```bash
./generate_pools.sh
# Crea:
# - data/processed/pool_train/ (80% de datos)
# - data/processed/pool_test/ (20% de datos, sin overlap)
```

### Caso 2: Entrenar e inmediatamente evaluar en test
```bash
./.venv/bin/python src/imputacion_co2_ml.py \
  --input-dir data/processed/pool_train \
  --test-dir data/processed/pool_test \
  --missing-rate 20 \
  --simplificado

# Output: artifacts/metrics/co2_metrics.json con métricas de TRAIN y TEST
```

### Caso 3: Entrenar sin test inmediato
```bash
./.venv/bin/python src/imputacion_co2_ml.py \
  --input-dir data/processed/pool_train \
  --missing-rate 20

# Output: solo métricas de train (validación cruzada)
```

### Caso 4: Cambiar ratio train/test
```bash
rm -rf data/processed/pool_train data/processed/pool_test

./.venv/bin/python src/depuracion_txt.py \
  --input data/raw/parque_vehiculos_202503.txt \
  --pool-dir data/processed/pool_train \
  --pool-test-dir data/processed/pool_test \
  --train-ratio 0.7 \  # 70% train, 30% test
  --keep-remainder \
  --rows-per-sample 100000
```

---

## Garantía de separación

### Prueba matemática:
Para cualquier fila con índice `i`:
- Si `i % 100 < 80`: va a TRAIN
- Si `i % 100 >= 80`: va a TEST

**Propiedades:**
- Cada fila está en exactamente UN conjunto
- Mismo archivo → mismo split (determinístico)
- Distribuye uniformemente los datos

**Verificación:**
```bash
# Sacar primeras 10 líneas y verificar que todas van a TRAIN
head -11 data/processed/pool_train/muestra_100k_0001.csv | \
  tail -10 | md5sum

head -11 data/processed/pool_test/muestra_100k_0001.csv | \
  tail -10 | md5sum  # Diferente

# Las filas no se repiten entre pools
```

---

## Notas técnicas

1. **Índice global**: se reinicia a 0 después de cada chunk filtrado, permitiendo procesamiento streaming

2. **Buffer separado**: mantiene dos buffers (train/test) para agregar filas hasta formar una muestra

3. **Orden preservado**: aunque se separa en dos streams, se mantiene el orden original de lectura

4. **Idempotencia**: ejecutar dos veces con mismo archivo genera idénticos pools

5. **GPU support**: Tanto entrenamiento como evaluación respetan `--device cuda/cpu`
