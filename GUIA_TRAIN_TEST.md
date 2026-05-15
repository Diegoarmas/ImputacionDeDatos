# Guía: Separación Train/Test en ImputacionDeDatos

## Problema resuelto
Anteriormente: Todos los datos (train + test) estaban mezclados.
Ahora: Train y test están **100% separados sin overlap**.

## Flujo rápido

### 1️⃣ Generar pools (Una sola vez)
```bash
./generate_pools.sh
```
**Qué hace:**
- Lee `data/raw/parque_vehiculos_202503.txt`
- Filtra solo filas con `EMISIONES_CO2` informado
- Separa en 80% train / 20% test (determinístico)
- Genera:
  - `data/processed/pool_train/` (~1M filas en múltiples archivos)
  - `data/processed/pool_test/` (~250k filas en múltiples archivos)

**Garantía:** Las filas en pool_test NUNCA están en pool_train ✓

### 2️⃣ Entrenar modelo CON datos de train
```bash
./.venv/bin/python src/imputacion_co2_ml.py \
  --input-dir data/processed/pool_train \
  --test-dir data/processed/pool_test \
  --missing-rate 20 \
  --simplificado
```

**Qué hace:**
- Carga TODOS los CSVs de `pool_train`
- Entrena con validación cruzada (5 folds) SOLO en train
- Evalúa el modelo final en `pool_test` (datos que NO vio)
- Guarda métricas en `artifacts/metrics/co2_metrics.json`

**Salidas importantes:**
```json
{
  "mae": 0.5234,           // Error promedio (train CV)
  "rmse": 0.7821,
  "r2": 0.8934,
  "test_mae": 0.5412,      // Error promedio en TEST (datos nuevos)
  "test_rmse": 0.8001,
  "test_r2": 0.8876,
  "test_rows_with_target": 45231  // Cuántas filas evaluadas en test
}
```

### 3️⃣ Validar que no hay overlap
```bash
# Verificar estructura
ls -la data/processed/pool_train/ | head
ls -la data/processed/pool_test/ | head

# Ver estadísticas
wc -l data/processed/pool_train/*.csv  # Total filas train
wc -l data/processed/pool_test/*.csv   # Total filas test
```

**Esperado:** 
- pool_train: ~80% de filas totales
- pool_test: ~20% de filas totales
- Ningún archivo repite nombre entre directorios

---

## Personalización

### Cambiar ratio train/test
Para 70% train / 30% test:
```bash
./.venv/bin/python src/depuracion_txt.py \
  --input data/raw/parque_vehiculos_202503.txt \
  --pool-dir data/processed/pool_train \
  --pool-test-dir data/processed/pool_test \
  --train-ratio 0.7 \  # ← Cambia aquí
  --keep-remainder
```

### Cambiar tamaño de muestras
Para 50k filas por archivo (en lugar de 100k):
```bash
./.venv/bin/python src/depuracion_txt.py \
  --input data/raw/parque_vehiculos_202503.txt \
  --pool-dir data/processed/pool_train \
  --pool-test-dir data/processed/pool_test \
  --rows-per-sample 50000 \  # ← Cambia aquí
  --keep-remainder
```

### Entrenar sin evaluar en test
```bash
./.venv/bin/python src/imputacion_co2_ml.py \
  --input-dir data/processed/pool_train \
  --missing-rate 20 \
  --simplificado
# Sin --test-dir, solo muestra métricas de train
```

---

## Cómo se garantiza sin overlap

La separación usa **índice modular determinístico**:

```python
def _is_train_split(global_row_index: int, train_ratio: float) -> bool:
    threshold = int(train_ratio * 100)  # 80
    return (global_row_index % 100) < threshold
    # Resultado: índices 0-79 → TRAIN, 80-99 → TEST
```

**Implicaciones:**
- Fila #0: `0 % 100 = 0 < 80` → TRAIN ✓
- Fila #50: `50 % 100 = 50 < 80` → TRAIN ✓
- Fila #81: `81 % 100 = 81 < 80` → FALSE → TEST ✓
- Fila #100: `100 % 100 = 0 < 80` → TRAIN ✓
- Fila #181: `181 % 100 = 81 < 80` → FALSE → TEST ✓

**Ventajas:**
- ✓ Determinístico: Mismo archivo → mismo split siempre
- ✓ Sin overlap: Cada fila va a UN solo pool
- ✓ Escalable: Procesa por bloques sin cargar todo
- ✓ Reproducible: Otros pueden reproducir exactamente

---

## Verificación final

Para confirmar que todo está bien:

```bash
# 1. Ver cuántas filas en cada pool
echo "=== TRAIN ===" && wc -l data/processed/pool_train/*.csv | tail -1
echo "=== TEST ===" && wc -l data/processed/pool_test/*.csv | tail -1

# 2. Ver primeras filas de métrica
cat artifacts/metrics/co2_metrics.json | grep -E "test_|rows_with_known"

# 3. Verificar modelo entrenado
ls -lh artifacts/models/co2_model.joblib
```

**Esperado:**
```
=== TRAIN ===
8234567 total    # ~80% del total
=== TEST ===
2058934 total    # ~20% del total

"test_mae": 0.5412
"test_rmse": 0.8001
"test_rows_with_target": 45231
```

---

## Problemas comunes

**P: ¿Por qué pool_test tiene X filas si puse 20%?**
A: Porque hay muestras de exactamente 100k filas. Si total es 10.1M:
- 80% = 8.08M (≈ 80 archivos de 100k)
- 20% = 2.02M (≈ 20 archivos de 100k)

**P: ¿Puedo regenerar pools con otro train-ratio?**
A: Sí, pero debe eliminar los directorios anteriores primero:
```bash
rm -rf data/processed/pool_train data/processed/pool_test
./generate_pools.sh  # O ejecutar con otro ratio
```

**P: ¿Qué pasa si corro gen
