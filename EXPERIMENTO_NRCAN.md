# Experimento de Aumentación con Datos NRCan

Evaluación del impacto de enriquecer el conjunto de entrenamiento con datos de
consumo de combustible de Natural Resources Canada (NRCan) sobre la imputación
de CO₂ en el parque DGT.

---

## 1. Motivación

El registro DGT contiene CO₂ declarado por el fabricante según el ciclo NEDC
europeo. Para vehículos con CO₂ faltante o cero, el modelo tiene que estimar el
valor a partir de otras features (cilindrada, potencia, marca, modelo, año…).

NRCan publica catálogos de consumo de combustible medidos con metodología propia
para el mercado canadiense. Esos datos cubren marcas y modelos que en el DGT
pueden estar infrarepresentados (pocos ejemplares registrados), por lo que actúan
como **conocimiento externo** que puede mejorar la estimación en esos segmentos.

---

## 2. Fuente de datos NRCan utilizada

| Parámetro | Valor |
|---|---|
| Origen | `resultados_canada_european_brands_pre2010.csv` (datos_TFG) |
| Cobertura | Marcas europeas, años 1995–2010 |
| Filas | 12 401 (bruto) → **3 526 tras filtrado** |
| Variables disponibles | Make, Model, Engine Size, Fuel Type, CO₂ (g/km) |

> **Nota:** Las URLs oficiales de NRCan (open.canada.ca) devuelven 404 actualmente.
> El fichero utilizado es el resultado procesado localmente a partir de los datos
> de NRCan descargados previamente, con marcas europeas seleccionadas y campo
> CO₂ ya en g/km.

---

## 3. Factor de corrección del ciclo de medición

Los valores de CO₂ de NRCan se miden con el ciclo NEDC canadiense, que produce
lecturas distintas al ciclo NEDC europeo (norma que usa el DGT).

Para hacer comparables ambas fuentes se aplica un factor de escala:

```
CO₂_ajustado = CO₂_NRCan × 0.4549
```

Este factor fue determinado empíricamente ajustando las distribuciones de CO₂
entre la flota española y los catálogos canadienses para los segmentos solapados.

---

## 4. Configuración del experimento

| Parámetro | Valor |
|---|---|
| Modelo | HistGradientBoostingRegressor (CPU sklearn) |
| Ficheros train DGT | 5 |
| Filas train DGT | 851 686 |
| Filas NRCan añadidas | **3 526** (0.41% del total) |
| Ficheros validación | 3 |
| Filas validación | 510 603 |
| Solo marcas europeas | Sí |
| Factor corrección | 0.4549 |

El experimento compara dos modelos idénticos en arquitectura e hiperparámetros,
entrenados con los mismos datos DGT. La única diferencia es que el modelo
**aumentado** recibe las 3 526 filas NRCan concatenadas al train.

---

## 5. Resultados

| Métrica | Baseline (solo DGT) | Aumentado (+NRCan) | Δ |
|---|---|---|---|
| **MAE (g CO₂/km)** | 5.494 | **5.419** | **−0.076** |
| **RMSE (g CO₂/km)** | 12.566 | **12.469** | −0.097 |
| **R²** | 0.9032 | **0.9047** | +0.0015 |
| n validación | 510 603 | 510 603 | — |

### Mejora relativa

- MAE: **−1.4%** (5.494 → 5.419 g/km)
- R²: **+0.17 puntos porcentuales** (0.9032 → 0.9047)

---

## 6. Interpretación

### La mejora es pequeña pero consistente

Las 3 526 filas de NRCan representan solo el **0.41% del total de datos de
entrenamiento**. Aun así, las tres métricas mejoran simultáneamente. El efecto
no es esperable por azar: el modelo con datos extra reduce MAE en −0.076 g/km,
RMSE en −0.097 g/km y eleva R² en +0.0015.

### Por qué el impacto es limitado

1. **Cobertura parcial**: el fichero disponible cubre solo marcas europeas
   pre-2010, segmento que ya está bien representado en el DGT. El beneficio de
   la aumentación es mayor cuando el dato externo aporta segmentos no cubiertos
   por el DGT (motos asiáticas, vehículos industriales, modelos de nicho).

2. **Volumen reducido**: 3 526 filas frente a 851 686 del DGT. El modelo aprende
   principalmente del DGT; NRCan actúa como regularización suave.

3. **Factor de corrección aproximado**: el factor 0.4549 es una estimación
   empírica. Si la conversión de ciclos no es perfecta, introduce ruido
   sistemático que atenúa la señal útil.

### Potencial con el dataset completo

El catálogo completo de NRCan cubre **1995–2025, todas las marcas** (~30 000+
entradas). Si se obtuviesen esos datos:

- Se añadirían segmentos actualmente problemáticos: SUVs norteamericanos,
  pickup trucks, vehículos de gran cilindrada, algunos modelos asiáticos.
- El impacto esperado sería mayor, especialmente en las marcas con R² negativo
  en el análisis actual (DFSK, Chrysler, motos de nicho).

---

## 7. Conclusión

**La aumentación con NRCan es beneficiosa y no introduce regresión.** Con el
subconjunto disponible (marcas europeas pre-2010), la mejora es modesta (−1.4%
MAE). Se recomienda:

1. Obtener el catálogo completo NRCan (descarga manual desde
   natural-resources.canada.ca) para evaluar el impacto con datos más amplios.
2. Afinar el factor de corrección de ciclo (actualmente 0.4549) si se dispone
   de vehículos con mediciones en ambos ciclos.
3. No incluir NRCan en el modelo de producción hasta tener el dataset completo
   y validar que la mejora se sostiene con todos los ficheros DGT (no solo 5).

---

## 8. Artefactos

| Archivo | Contenido |
|---|---|
| [data/nrcan/nrcan_combined.csv](data/nrcan/nrcan_combined.csv) | Datos NRCan adaptados (12 401 filas, columnas renombradas) |
| [results/nrcan_augment_experiment.json](results/nrcan_augment_experiment.json) | Métricas baseline vs aumentado |
| [src/nrcan_augment_experiment.py](src/nrcan_augment_experiment.py) | Script del experimento |
