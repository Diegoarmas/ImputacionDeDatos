# Resultados e Interpretación del Modelo de Imputación CO₂

Pipeline XGBoost-CUDA entrenado sobre el registro DGT. Métricas globales,
comportamiento por marca/modelo y experimento de ablación.

**Configuración**: 40 ficheros train (8M filas) · Val + Test 15 ficheros c/u (2.56M) · missing_rate=20% · CV 5 folds

---

## 1. Resumen ejecutivo

| Métrica | Valor |
|---|---|
| **Test MAE** | **6.41 g CO₂/km** |
| **Test R²** | **0.822** |
| CV MAE (train, 5 folds) | 6.88 ± 0.02 g/km |
| CV R² (train) | 0.897 ± 0.001 |
| Filas con CO₂ imputadas | 1.36M (20% del train) |
| Filas excluidas (CO₂ = 0, eléctricos) | 1.21M |

Un MAE de **6.41 g/km** sobre 2.56M vehículos de test es un rendimiento sólido.
El límite físico alcanzable está acotado por la variación intrínseca de la
medición CO₂ en ITV (~5–10 g/km según condiciones de prueba). Val y test
dan prácticamente los mismos resultados (6.43 vs 6.41 g/km), señal de que
el modelo generaliza de forma estable.

---

## 2. Dataset y particiones

| Split | Ficheros | Filas totales | Filas con CO₂ | Uso |
|---|---|---|---|---|
| pool_train | 40 / 74 | 8 000 000 | 6 790 350 | Fit del modelo + CV |
| pool_val | 15 / 15 | 3 000 000 | 2 563 939 | Comparar configs, elegir hiperparámetros |
| pool_test | 15 / 15 | 3 000 000 | 2 563 525 | Reporte final (usar una sola vez) |

> **Advertencia sobre el test set.** Las métricas de test expuestas aquí son
> el reporte final. Para comparar futuras configuraciones deben usarse las
> métricas de **val**. Cada consulta al test introduce sesgo acumulativo en la
> selección del modelo.

---

## 3. Métricas globales e interpretación

| | CV train (5 folds) | Test (hold-out) |
|---|---|---|
| MAE | 6.88 ± 0.02 g/km | **6.41 g/km** |
| RMSE | 19.14 ± 0.10 g/km | 16.83 g/km |
| R² | 0.897 ± 0.001 | **0.822** |

**¿Por qué el CV da MAE mayor que el test?**
El CV evalúa el modelo sobre el 20% oculto *dentro de los mismos 40 ficheros
de train* (shuffle aleatorio). El test usa 15 ficheros completamente distintos.
La diferencia (6.88 vs 6.41) no indica sobreajuste sino que el CV es una
estimación más conservadora. La varianza del CV es muy baja (σ=±0.02 g/km),
confirmando que el modelo es estable.

**RMSE vs MAE: ratio 2.6x.**
La diferencia grande entre RMSE (16.83) y MAE (6.41) revela una cola de
errores grandes: la mayoría de vehículos se imputan bien (~6 g/km) pero hay
una minoría con errores sustancialmente mayores. Esos outliers corresponden a
los segmentos identificados en el análisis por marca/modelo (motos de nicho,
SUVs específicos).

---

## 4. Análisis por marca

De las **93 marcas** con ≥100 vehículos en test, el MAE va de 2.28 g/km
(BYD) a 98.04 g/km (CFMOTO) — ratio de **43×**.

**Top 5 peor imputadas:**

| Marca | n | CO₂ medio | MAE | R² | Interpretación |
|---|---|---|---|---|---|
| CFMOTO | 347 | 325 g/km | **98.0** | 0.68 | Cuadriciclo con CO₂ muy alto, poca representación |
| DFSK | 174 | 198 g/km | **42.4** | −1.76 | Furgoneta china de nicho, distribución atípica |
| CHRYSLER | 102 | 221 g/km | **31.9** | −1.02 | Volumen mínimo (102), alta varianza interna |
| RODRIGUEZ LOPEZ AUTO | 325 | 236 g/km | 28.5 | 0.84 | Importador atípico, rango CO₂ amplio |
| IVECO | 1 496 | 215 g/km | 27.9 | −0.30 | Industriales: distribución multimodal distinta a turismos |

**Top 5 mejor imputadas:**

| Marca | n | CO₂ medio | MAE | R² |
|---|---|---|---|---|
| BYD | 344 | 15 g/km | **2.28** | 0.88 |
| LYNK&CO | 392 | 27 g/km | 2.89 | 0.38 |
| OMODA | 855 | 145 g/km | 3.08 | 0.99 |
| SYM | 7 060 | 61 g/km | 3.38 | 0.16 |
| QJMOTOR | 634 | 67 g/km | 3.41 | 0.96 |

**20 de 93 marcas tienen R² negativo** (el modelo predice peor que la media
del grupo). Afecta casi exclusivamente a motos (Daelim, Montesa, Gas Gas,
Harley-Davidson, Sherco, Guzzi, Keeway, Rieju…) y microcoches (Aixam,
Microcar, Ligier). El dataset de entrenamiento está dominado por turismos,
cuya relación cilindrada→CO₂ no se traslada a motos de combustión.

---

## 5. Análisis por modelo

**650 modelos** con ≥100 vehículos evaluados. **172 modelos (26%) tienen R²
negativo**, replicando el patrón de marcas a nivel granular.

**Peores modelos por MAE:**

| Modelo | MAE | R² | Problema |
|---|---|---|---|
| YP125R (scooter) | **47.5** | −29.5 | Predice 16 g/km, real es 63 g/km — infraestimación 4× |
| Grand Vitara | 35.8 | −0.27 | SUV/4x4, distribución CO₂ de diésel todoterreno |
| Defender | 24.8 | 0.80 | Alta varianza interna de versiones (diesel/V8) |
| Wrangler | 22.7 | −0.19 | Combustible y versiones muy dispares |
| Sprinter | 17.4 | −0.11 | Furgoneta industrial, poco representada |

Los "mejores" modelos (DTX 125: 0.24 g/km, NSC110: 0.38 g/km) tienen
R²≈0: todos los vehículos del grupo tienen exactamente el mismo CO₂ (misma
versión/año). El bajo MAE no refleja capacidad predictiva sino varianza real
cero en el grupo.

---

## 6. Experimento de ablación: MARCA/MODELO

| Configuración | CV MAE | Test MAE | Test R² |
|---|---|---|---|
| Con MARCA/MODELO | 5.34 | **5.27** | **0.907** |
| Sin MARCA/MODELO | 5.41 | 5.34 | 0.905 |
| Δ (sin − con) | +0.06 | **+0.067** | −0.002 |

Quitar MARCA/MODELO **empeora** el modelo en +0.067 g/km (~1.3% relativo).
El efecto es pequeño pero reproducible (3 ejecuciones bit-idénticas con
`random_state=42`).

No justifica eliminar esas features por accuracy. La decisión de quitarlas
debería basarse en otros criterios: reducir cardinalidad del encoder,
simplificar mantenimiento del vocabulario de marcas/modelos.

> **Nota metodológica.** Un resultado previo (commit `79a1b88`) reportaba
> erróneamente Δ = −0.80 g/km a favor de eliminar MARCA/MODELO. Era
> inválido: comparaba el modelo de producción pre-entrenado (datos distintos)
> contra una config re-entrenada en CPU — comparación asimétrica. Corregido
> en el commit `42e8340`.

---

## 7. Optimización de memoria aplicada

| | Antes | Después |
|---|---|---|
| RAM por fichero (200k filas) | 455 MB | **22 MB** |
| 40 ficheros estimado | ~18 GB → OOM kill | ~0.9 GB → OK |
| Reducción | — | **21×** |

Cambios en [src/data_cleaning.py](src/data_cleaning.py):
- Columnas numéricas: `float64` → `float32` (mitad de bytes)
- Columnas de texto: `object` → `category` (índice entero + tabla de strings única)

XGBoost y sklearn manejan ambos tipos de forma nativa sin cambios de interfaz.

---

## 8. Artefactos generados

| Archivo | Contenido |
|---|---|
| [artifacts/models/co2_model.joblib](artifacts/models/co2_model.joblib) | Pipeline serializado listo para inferencia |
| [artifacts/metrics/co2_metrics.json](artifacts/metrics/co2_metrics.json) | CV, val y test MAE/RMSE/R², lista de ficheros, backend |
| [results/tables/brand_analysis.csv](results/tables/brand_analysis.csv) | Métricas por marca (93 marcas, ≥100 filas) |
| [results/tables/model_analysis.csv](results/tables/model_analysis.csv) | Métricas por modelo (650 modelos, ≥100 filas) |
| [results/tables/ablation_brand_model.csv](results/tables/ablation_brand_model.csv) | Comparativa con/sin MARCA/MODELO |
| [results/plots/mae_by_brand.png](results/plots/mae_by_brand.png) | Barras MAE por marca (top 20) |
| [results/plots/volume_vs_mae.png](results/plots/volume_vs_mae.png) | Scatter volumen vs MAE por marca |
| [data/processed/simplificado/datos_simpl.csv](data/processed/simplificado/datos_simpl.csv) | 1.36M filas con CO₂ imputado |
