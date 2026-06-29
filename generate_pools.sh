#!/bin/bash
# Script para generar pools separados de entrenamiento, validacion y prueba
# Uso: ./generate_pools.sh

set -e

echo "=== Generando pools TRAIN/VAL/TEST separados ==="

# Configuración
INPUT="data/raw/parque_vehiculos_202503.txt"
POOL_TRAIN="data/processed/pool_train"
POOL_VAL="data/processed/pool_val"
POOL_TEST="data/processed/pool_test"
ROWS_PER_SAMPLE=100000
TRAIN_RATIO=0.70
VAL_RATIO=0.15

# Verificar que el archivo de entrada existe
if [ ! -f "$INPUT" ]; then
    echo "Error: Archivo de entrada no encontrado: $INPUT"
    exit 1
fi

echo "Archivo de entrada: $INPUT"
echo "Pool TRAIN: $POOL_TRAIN (70%)"
echo "Pool VAL:   $POOL_VAL (15%)"
echo "Pool TEST:  $POOL_TEST (15%)"
echo "Filas por muestra: $ROWS_PER_SAMPLE"
echo ""

# Ejecutar el script de depuración con split
./.venv/bin/python src/depuracion_txt.py \
  --input "$INPUT" \
  --pool-dir "$POOL_TRAIN" \
  --pool-val-dir "$POOL_VAL" \
  --pool-test-dir "$POOL_TEST" \
  --train-ratio "$TRAIN_RATIO" \
  --val-ratio "$VAL_RATIO" \
  --period-column 'FECHA_MATR' \
  --split-report-output 'artifacts/metrics/split_period_report.json' \
  --rows-per-sample "$ROWS_PER_SAMPLE" \
  --max-samples 150 \
  --keep-remainder \
  --in-sep '|' \
  --out-sep ',' \
  --in-encoding 'iso-8859-1' \
  --out-encoding 'utf-8'

echo ""
echo "=== Pool generados correctamente ==="
echo "Directorio TRAIN: $POOL_TRAIN"
echo "Directorio VAL:   $POOL_VAL"
echo "Directorio TEST: $POOL_TEST"
echo "Reporte periodo: artifacts/metrics/split_period_report.json"
echo ""
echo "Próximo paso: Entrenar con pool_train y validar/testear separado"
