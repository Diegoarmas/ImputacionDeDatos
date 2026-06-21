# ImputacionDeDatos

Pipeline en Python para imputar valores faltantes de `EMISIONES_CO2` en el registro del parque de vehículos español (38,6 millones de filas). Utiliza un modelo de regresión supervisada basado en Gradient Boosting que alcanza **MAE ≈ 3 g/km** y **R² ≈ 0,97**.

> **Scope del modelo**: solo vehículos de combustión con `EMISIONES_CO2 > 0`. Los vehículos eléctricos (CO2 = 0) se descartan en la fase de depuración de datos y quedan fuera del modelo.

---

## Análisis de variables y selección de modelo

### Problema

**Regresión supervisada**: predecir `EMISIONES_CO2` (variable continua, >0–204 g/km) a partir de las características técnicas y administrativas de cada vehículo de combustión.

La distribución del target es **multimodal**:
- Motocicletas → 50–120 g/km
- Turismos gasolina/diesel → 90–180 g/km
- Furgonetas/camiones → 120–204+ g/km

### Tipología de variables (45 features)

| Tipo | Variables | Notas |
|---|---|---|
| **Continuas** | TARA, PESO_MAX, CILINDRADA, KW, POTENCIA, CONSUMO, AUTONOMIA, DISTANCIA_EJES, EJE_ANTERIOR, EJE_POSTERIOR | Escalas muy distintas (kg, cc, kW) |
| **Discretas** | PLAZAS, PLAZAS_MAX, PLAZAS_PIE | Rango 1–7, tratadas como numéricas |
| **Nominales alta cardinalidad** | FABRICANTE, MARCA, MODELO, VARIANTE, VERSION | Cientos de valores únicos |
| **Nominales baja cardinalidad** | PROPULSION, ALIMENTACION, CARROCERIA, TIPO_DGT, CLASE_MATR | Pocas categorías |
| **Ordinales** | CAT_EURO, EMISIONES_EURO | Euro I–VI, tienen orden lógico |
| **Temporales** | FECHA_MATR, FEC_PRIM_MATR | Año y mes extraídos como features numéricas |
| **Geográficas** | PROVINCIA, MUNICIPIO | Alta cardinalidad nominal |

### Relaciones físicas clave

```
ALIMENTACION / PROPULSION → CO2    # diesel ≠ gasolina (vehículos eléctricos excluidos)
CONSUMO       → CO2                # CO2 ≈ 23,5 × L/100km (gasolina), 26,5 × L/100km (diesel)
CILINDRADA    → KW → CO2           # motor más grande → más potencia → más emisiones
CAT_EURO ↑    → CO2 ↓             # normas más recientes = vehículos más limpios
TARA / PESO   → CO2               # más pesado → más consumo
AÑO_MATR ↑   → CO2 ↓             # vehículos más nuevos son más eficientes
MARCA / MODELO → CO2              # efectos de fabricante
```

Existe **multicolinealidad alta** entre CILINDRADA, KW, POTENCIA y TARA. Los modelos de árbol la gestionan de forma natural, a diferencia de la regresión lineal (que requeriría regularización o PCA previo).

### Comparativa de modelos

| Modelo | Apto | Motivo |
|---|---|---|
| **Regresión Lineal** | No | Asume linealidad. La relación CILINDRADA/KW → CO2 es no lineal, y con MODELO/MARCA habría explosión dimensional tras one-hot encoding. |
| **KNN** | No | Maldición de la dimensionalidad con 45+ features. Distancia euclidiana no tiene sentido mezclando kg, cc y categorías. O(n) en inferencia, inviable con millones de filas. |
| **Árbol de decisión simple** | Solo baseline | Interpretable, capta no-linealidades, pero alta varianza y sobreajuste con tantas categorías. |
| **Random Forest** | Sí, alternativa | Reduce varianza por bagging. Mayor memoria y latencia que Gradient Boosting; peor con alta cardinalidad. |
| **Gradient Boosting (HistGBT / XGBoost / LightGBM)** | **Sí — opción óptima** | Estado del arte en tabular heterogéneo. Maneja natively missings, alta cardinalidad (OrdinalEncoder), interacciones (CILINDRADA × ALIMENTACION) y escala a millones de filas. GPU disponible vía XGBoost. |
| **Redes Neuronales (MLP)** | No recomendado | Para datos tabulares heterogéneos, Gradient Boosting supera sistemáticamente a MLP. Requieren normalización exhaustiva y más hiperparámetros. |
| **SVR** | No | Complejidad O(n²–n³), inviable con millones de registros. |
| **Modelos no supervisados** | Solo complemento | Clustering previo por tipo de vehículo podría estratificar el problema, pero no son el modelo principal dado que disponemos de etiquetas. |

### Conclusión

**`HistGradientBoostingRegressor` es la elección correcta** para este problema. A continuación se justifica cada aspecto.

#### Cómo funciona internamente

El modelo construye una secuencia de árboles de decisión donde cada árbol nuevo aprende a corregir el error del anterior (gradient boosting). La predicción final es la suma de todos los árboles:

```
CO2_predicho = árbol₁ + árbol₂ + ... + árbol₃₅₀
               (cada uno reduce el error residual)
```

El prefijo **Hist** indica que antes de entrenar agrupa los valores continuos en 256 intervalos (histograma), lo que permite operar sobre millones de filas sin evaluar cada valor individualmente. Esto lo hace entre 10 y 100 veces más rápido que la versión estándar sin pérdida de precisión apreciable.

#### Por qué se ajusta a los datos de este proyecto

| Característica del dato | Cómo la gestiona HistGBT |
|---|---|
| Missings en features | Los trata nativamente: en cada corte decide si los NaN van a la rama izquierda o derecha, sin necesidad de imputación previa |
| Alta cardinalidad (MODELO, MARCA) | Funciona con OrdinalEncoder; los árboles no asumen ningún orden, solo comparan bins |
| Relación no lineal (CILINDRADA → CO2) | Cascadas de cortes capturan no-linealidades sin transformar las variables |
| Multicolinealidad (KW ↔ POTENCIA ↔ CILINDRADA) | No le afecta: en cada nodo usa la feature más informativa y descarta las redundantes |
| Escalas muy distintas (kg, cc, kW, g/km) | Irrelevante para árboles, que solo comparan valores dentro de la misma columna |
| Distribución multimodal del target | Los árboles profundos (max_depth=8) pueden segmentar el espacio en tantas regiones como necesiten |

#### Por qué los modelos alternativos no son adecuados

- **Regresión lineal**: no puede capturar la relación no lineal entre cilindrada/potencia y CO2, ni las interacciones entre tipo de combustible y consumo. Requeriría además codificar cientos de modelos y marcas en columnas binarias.
- **KNN**: con 45 features la distancia euclidiana pierde significado (maldición de la dimensionalidad), y buscar los vecinos más cercanos en millones de registros es inviable en producción.
- **Redes neuronales**: en datos tabulares heterogéneos como este, gradient boosting supera sistemáticamente a MLP según la literatura. Requieren además normalización exhaustiva de todas las variables y ciclos de ajuste mucho más largos.
- **SVR**: su complejidad de entrenamiento es O(n²)–O(n³), lo que hace inviable su uso con el volumen de datos del parque de vehículos.

La única alternativa que merece evaluación experimental es **LightGBM** con soporte nativo de categoricals, que puede dar mejoras marginales en features de alta cardinalidad como MODELO o MUNICIPIO.

---

### Modelo en uso (configuración exacta)

**Modo CPU (por defecto):** `sklearn.ensemble.HistGradientBoostingRegressor`

| Hiperparámetro | Valor |
|---|---|
| `max_iter` | 350 |
| `learning_rate` | 0,05 |
| `max_depth` | 8 |
| `min_samples_leaf` | 20 |
| `random_state` | 42 (configurable) |

**Modo GPU (`--device cuda`):** `xgboost.XGBRegressor`

| Hiperparámetro | Valor |
|---|---|
| `n_estimators` | 400 |
| `learning_rate` | 0,05 |
| `max_depth` | 6 |
| `min_child_weight` | 5 |
| `subsample` | 0,8 |
| `colsample_bytree` | 0,8 |
| `tree_method` | `hist` |
| `device` | `cuda` |

**Pipeline de preprocesado (común a ambos modos):**

| Bloque | Pasos |
|---|---|
| Numéricas | `SimpleImputer(strategy="median")` |
| Categóricas | `SimpleImputer(fill_value="DESCONOCIDO")` → `OrdinalEncoder(unknown_value=-1)` |
| Fechas | Extraídas como `_YEAR` y `_MONTH` antes del pipeline |

---