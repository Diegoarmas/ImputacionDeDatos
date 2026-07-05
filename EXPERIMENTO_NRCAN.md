# Experimento de Aumentación con Datos NRCan

Evaluación del uso de datos de Natural Resources Canada (NRCan) como fuente
auxiliar de entrenamiento para cubrir el hueco temporal de vehículos antiguos
en el parque DGT.

---

## 1. Motivación: el problema de los vehículos antiguos

El DGT tiene millones de vehículos matriculados antes del año 2000 cuyo campo
CO₂ está vacío o es cero. Esto ocurre porque:

- La normativa de declaración obligatoria de emisiones CO₂ en la ficha técnica
  no existía antes de ~2004 (Directiva 1999/94/CE).
- Los vehículos históricos o de colección no pasan ITV con medición de CO₂.
- Muchas fichas antiguas nunca fueron digitalizadas con ese dato.

El modelo de imputación necesita **ejemplos de entrenamiento de esa época** para
poder estimar CO₂ en esos vehículos. El DGT por sí solo no los tiene: los
ficheros pool_train están dominados por vehículos modernos con CO₂ conocido.

**NRCan resuelve parcialmente eso.** El catálogo canadiense cubre modelos desde
1995 con valores de CO₂ medidos. Sin embargo, como se detalla en la sección 6,
las marcas que cubre NRCan no coinciden con las que dominan el parque español
antiguo, lo que limita su utilidad real.

---

## 2. Fuente de datos NRCan utilizada

| Parámetro | Valor |
|---|---|
| Origen | `resultados_canada_european_brands_pre2010.csv` (datos_TFG) |
| Cobertura | Marcas europeas disponibles en Canadá, años 1995–2010 |
| Filas brutas | 12 401 |
| Filas tras filtrado | **3 526** (solo marcas europeas) |
| Variables disponibles | Make, Model, Model Year, Engine Size, Fuel Type, CO₂ (g/km) |

> Las URLs oficiales de NRCan (open.canada.ca) devuelven 404 actualmente.
> El fichero procede de datos NRCan descargados previamente y procesados
> localmente para el TFG.

---

## 3. Factor de corrección de ciclo

Los valores CO₂ de NRCan se miden con metodología canadiense (ciclo combinado
FTP/HWFET, derivado del ciclo estadounidense). El DGT usa el ciclo NEDC europeo.
Son ciclos estructuralmente distintos: el canadiense tiene aceleraciones más
agresivas y es más representativo del uso real, por lo que produce valores de
CO₂ sistemáticamente **más altos** que el NEDC.

### Factores validados empíricamente

El análisis comparativo EEA-2010 vs NRCan-2010 identificó **371 pares de
vehículos** presentes en ambos catálogos (mismo modelo, cilindrada ±200 cm³,
mismo combustible). Ratios medidos:

| Combustible | n pares | Ratio CA/EU (mediana) | Factor divisor |
|---|---|---|---|
| Gasolina | 362 | **1.205** | /1.205 |
| Diésel | 9 | **1.527** | /1.527 |
| Otros | — | **1.213** (ratio global) | /1.213 |

La regresión lineal da: `CO₂_EU = (CO₂_CA − 134.6) / 0.605` (R²=0.55,
RMSE=66 g/km), útil para estimaciones pero con demasiada incertidumbre para
imputación precisa vehículo a vehículo.

### El factor 0.4549 — por qué era incorrecto

Una versión anterior del script calculaba el factor de corrección dinámicamente
comparando medianas de CO₂ entre marcas del DGT train y marcas de NRCan. El
resultado fue **0.4549** (ratio CA/EU implícito = 2.2×).

Este valor es un artefacto de comparar **segmentos distintos**:

- DGT train incluye muchas marcas de volumen europeas (Renault, SEAT, Citroën,
  Peugeot) con CO₂ bajo (~130–150 g/km).
- NRCan solo tiene marcas globales/premium (BMW, Mercedes, Audi) con CO₂
  alto (~280–300 g/km).
- Al hacer matching por nombre de marca, el ratio calculado fue
  ~130/285 ≈ **0.456**, que no refleja la diferencia de ciclos sino la
  diferencia de composición de flota.

Con el factor 0.4549, el CO₂ NRCan corregido queda en ~135 g/km (demasiado
bajo). Con los factores validados por tipo de combustible, queda en ~247 g/km,
alineado con la media EU de los pares emparejados (245.7 g/km).

| | Factor 0.4549 | Factor G/1.205, D/1.527 |
|---|---|---|
| Origen | Matching cruzado entre flotas distintas | 371 pares EEA vs NRCan mismo modelo |
| Ratio CA/EU implícito | 2.2× | 1.21× (validado por literatura) |
| CO₂ NRCan corregido (media) | 135 g/km | **247 g/km** |
| Correcto | ❌ | ✅ |

---

## 4. Métricas de referencia correctas

> **El MAE global no es la métrica relevante para este experimento.**
>
> El val set está dominado por vehículos modernos (2010–2023): más de 500k
> de los 510k vehículos son post-2010. El efecto de NRCan en ese segmento
> es nulo por diseño. La métrica que importa es el MAE segmentado por año
> de fabricación, especialmente en **pre-2000**.

---

## 5. Configuración del experimento

| Parámetro | Valor |
|---|---|
| Modelo | HistGradientBoostingRegressor (CPU sklearn) |
| Ficheros train DGT | 5 |
| Filas train DGT | 851 686 |
| Filas NRCan añadidas | **3 526** (0.41% del total train) |
| Ficheros validación | 3 |
| Filas validación | 510 603 |
| Solo marcas europeas | Sí |
| Corrección ciclo | Por tipo de combustible (G/1.205, D/1.527, O/1.213) |

---

## 6. Resultados

### 6.1 Métricas globales

| Métrica | Baseline (solo DGT) | Aumentado (+NRCan) | Δ |
|---|---|---|---|
| MAE (g CO₂/km) | 5.494 | 5.518 | +0.024 |
| RMSE (g CO₂/km) | 12.566 | 12.532 | −0.034 |
| R² | 0.9032 | 0.9038 | +0.0006 |

El MAE global es prácticamente idéntico. Esperado: 3 526 filas de vehículos
de marcas premium antiguas no alteran la predicción del segmento mayoritario
(vehículos modernos de todo tipo de marcas).

### 6.2 Métricas por segmento de año (métrica clave)

| Segmento | n val | Baseline MAE | Aumentado MAE | Δ |
|---|---|---|---|---|
| **pre-2000** | 295 | **22.75** | 23.54 | +0.78 |
| **2000–2009** | 738 | **17.56** | 18.55 | +0.99 |
| 2010–2019 | 305 762 | 5.37 | 5.37 | ≈ 0 |
| 2020+ | 203 808 | 5.61 | 5.66 | +0.05 |

**Observaciones:**

- El MAE para vehículos pre-2000 es **22.75 g/km** frente a 5.4 g/km global:
  el modelo base ya tiene dificultad enorme con vehículos de esa época, lo que
  confirma la necesidad de datos auxiliares.
- Añadir NRCan **no mejora** el segmento antiguo — lo empeora ligeramente
  (ver sección 7 para la causa).
- Los vehículos modernos (2010+) apenas se ven afectados (Δ ≈ 0), lo cual
  indica que NRCan no interfiere con el segmento bien cubierto por el DGT.
- La muestra pre-2000 es muy pequeña (295 vehículos): la diferencia de ±1 g/km
  tiene alta varianza estadística.

---

## 7. Por qué NRCan no mejora el segmento pre-2000

El problema no es el factor de corrección (que sí es correcto) sino la
**cobertura de marcas**.

Las 15 marcas europeas disponibles en NRCan son exclusivamente **premium o
globales**:

> Alfa Romeo, Audi, Bentley, BMW, Jaguar, Lamborghini, Land Rover, Maserati,
> Mercedes-Benz, MINI, Porsche, Rolls-Royce, Smart, Volkswagen, Volvo

Los vehículos pre-2000 más comunes en el parque español son **Renault, SEAT,
Citroën, Peugeot, Opel** — exactamente las marcas que no tienen presencia en
el mercado canadiense y que por tanto **no están en NRCan**.

El efecto negativo se explica así: NRCan añade datos de coches premium de los
90 (BMW Serie 5 de 1995, Mercedes Clase C, Audi A6…) con CO₂ alto (~247 g/km
de media tras corrección). El modelo aprende que "vehículo europeo de esa época
= CO₂ alto". Cuando después intenta imputar un SEAT Ibiza 1.4 de 1998 o un
Renault Clio 1.2 de 1999 — que tendrían ~130–150 g/km — predice demasiado
alto porque los únicos ejemplos de esa época en el train son premium.

---

## 8. Conclusión

**NRCan no resuelve el problema de los vehículos pre-2000 del parque español**
porque el catálogo canadiense no cubre las marcas que dominan ese segmento
(Renault, SEAT, Citroën, Peugeot, Opel representan ~50% del parque antiguo
español). Añadir solo datos de marcas premium de esa época introduce sesgo.

El factor de corrección de ciclo actual (**G/1.205, D/1.527**) es el correcto
y está validado empíricamente sobre 371 pares de vehículos. El factor 0.4549
que aparecía en una versión anterior del script era un artefacto incorrecto.

### Fuentes alternativas recomendadas

Para mejorar el segmento pre-2000 se necesitan fuentes con cobertura de marcas
europeas de volumen:

| Fuente | Cobertura | Ventaja |
|---|---|---|
| **EEA NEDC dataset** (monitoring CO₂) | Renault, PSA, Fiat, SEAT, VW, Ford EU | Mismo ciclo NEDC, sin corrección necesaria |
| **ACEA data** | Todos los fabricantes europeos | Datos oficiales de industria |
| **WLTP type-approval DB** (Comisión Europea) | Post-2017, todas las marcas | Ciclo moderno, cubre modelos actuales |

---

## 9. Artefactos

| Archivo | Contenido |
|---|---|
| [data/nrcan/nrcan_combined.csv](data/nrcan/nrcan_combined.csv) | Datos NRCan adaptados (12 401 filas) |
| [results/nrcan_augment_experiment.json](results/nrcan_augment_experiment.json) | Métricas globales + segmentadas por año |
| [src/nrcan_augment_experiment.py](src/nrcan_augment_experiment.py) | Script del experimento |
| [datos_TFG/comparacion_eu_canada.md](../datos_TFG/comparacion_eu_canada.md) | Análisis EEA-2010 vs NRCan-2010 (371 pares, derivación de factores) |
