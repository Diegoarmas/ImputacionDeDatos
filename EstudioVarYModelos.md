# ImputacionDeDatos

Pipeline en Python para imputar valores faltantes de `EMISIONES_CO2` en el registro del parque de vehículos español (38,6 millones de filas). Utiliza un modelo de regresión supervisada basado en Gradient Boosting que alcanza **MAE ≈ 3 g/km** y **R² ≈ 0,97**.

---

## Análisis de variables y selección de modelo

### Problema

**Regresión supervisada**: predecir `EMISIONES_CO2` (variable continua, 0–204 g/km) a partir de las características técnicas y administrativas de cada vehículo.

La distribución del target es **multimodal y con discontinuidades**:
- Vehículos eléctricos → CO2 = 0 (ruptura dura)
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
ALIMENTACION / PROPULSION → CO2    # eléctrico = 0, diesel ≠ gasolina
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
| **Regresión Lineal** | No | Asume linealidad. PROPULSION=Eléctrico crea una ruptura dura (CO2=0) que una recta no puede modelar. Con MODELO/MARCA habría explosión dimensional tras one-hot encoding. |
| **KNN** | No | Maldición de la dimensionalidad con 45+ features. Distancia euclidiana no tiene sentido mezclando kg, cc y categorías. O(n) en inferencia, inviable con millones de filas. |
| **Árbol de decisión simple** | Solo baseline | Interpretable, capta no-linealidades, pero alta varianza y sobreajuste con tantas categorías. |
| **Random Forest** | Sí, alternativa | Reduce varianza por bagging. Mayor memoria y latencia que Gradient Boosting; peor con alta cardinalidad. |
| **Gradient Boosting (HistGBT / XGBoost / LightGBM)** | **Sí — opción óptima** | Estado del arte en tabular heterogéneo. Maneja natively missings, alta cardinalidad (OrdinalEncoder), interacciones (CILINDRADA × ALIMENTACION) y escala a millones de filas. GPU disponible vía XGBoost. |
| **Redes Neuronales (MLP)** | No recomendado | Para datos tabulares heterogéneos, Gradient Boosting supera sistemáticamente a MLP. Requieren normalización exhaustiva y más hiperparámetros. |
| **SVR** | No | Complejidad O(n²–n³), inviable con millones de registros. |
| **Modelos no supervisados** | Solo complemento | Clustering previo por tipo de vehículo podría estratificar el problema, pero no son el modelo principal dado que disponemos de etiquetas. |

### Conclusión

**`HistGradientBoostingRegressor` es la elección correcta** por la combinación de:

- Target continuo con distribución multimodal y discontinuidades (eléctrico vs. combustión)
- 45 features heterogéneas (continuas + nominales de alta cardinalidad + ordinales + temporales)
- Valores faltantes presentes en múltiples columnas
- Escala de producción (decenas de millones de filas)

La única alternativa que merece evaluación experimental es **LightGBM** con soporte nativo de categoricals, que puede dar mejoras marginales en features de alta cardinalidad como MODELO o MUNICIPIO.

---