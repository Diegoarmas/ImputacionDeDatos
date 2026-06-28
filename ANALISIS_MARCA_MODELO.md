# Análisis de MARCA/MODELO en la imputación de CO2

Estudio de cómo varía la precisión de imputación de `EMISIONES_CO2` según
la marca y el modelo del vehículo, y experimento de ablación para cuantificar
si estas dos features aportan o restan capacidad predictiva al modelo.

Script: [src/brand_model_analysis.py](src/brand_model_analysis.py)

## Cómo ejecutarlo

```bash
./.venv/bin/python src/brand_model_analysis.py \
  --test-dir data/processed/pool_test \
  --model-path artifacts/models/co2_model.joblib \
  --ablation \
  --train-dir data/processed/pool_train
```

`--ablation` re-entrena el pipeline desde cero, una vez **con** MARCA/MODELO
y otra vez **sin** ellas, usando el mismo pool y el mismo backend (CPU/GPU)
que el modelo de producción.

## Salidas

| Archivo | Contenido |
|---|---|
| [results/tables/brand_analysis.csv](results/tables/brand_analysis.csv) | MAE/RMSE/R² por marca (≥100 filas) |
| [results/tables/model_analysis.csv](results/tables/model_analysis.csv) | MAE/RMSE/R² por modelo (≥100 filas) |
| [results/tables/ablation_brand_model.csv](results/tables/ablation_brand_model.csv) | Comparativa con/sin MARCA/MODELO |
| [results/plots/mae_by_brand.png](results/plots/mae_by_brand.png) | MAE por marca (top N) |
| [results/plots/mae_by_model.png](results/plots/mae_by_model.png) | MAE por modelo (top N) |
| [results/plots/volume_vs_mae.png](results/plots/volume_vs_mae.png) | Volumen de filas vs. error, por marca |
| [results/reports/brand_model_report.json](results/reports/brand_model_report.json) | Reporte completo en JSON |

## Resultados de la última ejecución

### Rendimiento global

| Métrica | Valor |
|---|---|
| Filas de test evaluadas | 848.888 |
| MAE global | 6.31 g CO₂/km |
| RMSE global | 15.92 g CO₂/km |
| R² global | 0.841 |

### Por marca: alta heterogeneidad, ligada al volumen de datos

- El MAE entre marcas varía de **2.28** (BYD) a **98.04** (CFMOTO), más de 40x de diferencia.
- Peores marcas (MAE más alto): CFMOTO (98.0, n=347), DFSK (42.4, n=174),
  CHRYSLER (31.9, n=102), IVECO (27.9, n=1.496), ISUZU (27.6, n=118) — todas
  con volumen bajo-medio.
- Mejores marcas (MAE más bajo): BYD, LYNK&CO, OMODA, SYM, QJMOTOR, DACIA,
  SEAT, HONDA — en su mayoría con bastante volumen.
- Correlación entre volumen (nº de filas) y MAE: **-0.24** (débil pero
  negativa). Más datos ayuda algo, pero no es el factor dominante.
- Varias marcas de motos/nicho (DAELIM, MONTESA, GAS GAS, HARLEY-DAVIDSON,
  INDIAN MOTORCYCLE, MICROCAR, LIGIER) presentan **R² negativo**: el modelo
  lo hace peor que predecir directamente la media de ese grupo. El modelo no
  generaliza bien a segmentos de motos/microcoches.
- Se detecta un valor de marca corrupto `¡` con 27.560 filas (MAE 18.6),
  probablemente un problema de encoding/normalización de texto a revisar en
  [src/data_cleaning.py](src/data_cleaning.py).

### Por modelo: SUVs/4x4/furgonetas son el punto débil

- Peor modelo con diferencia: **YP125R** (scooter), MAE=47.5, R²=**-29.5**;
  predice ~16 g/km cuando la media real es 63 g/km (infraestimación severa).
- Resto de modelos problemáticos: GRAND VITARA, DEFENDER, WRANGLER (×2),
  CAPTIVA, SPRINTER, FREELANDER 2, MASTER, RODIUS — casi todos SUV/4x4/
  furgonetas, muchos con R² negativo.
- Los modelos "mejor imputados" (Vespa, scooters 50-125cc, C3, NSC110...)
  tienen MAE muy bajo pero R²≈0: son grupos con CO2 casi constante, así que
  el bajo error refleja poca varianza dentro del grupo, no capacidad
  predictiva real.

### Ablación: quitar MARCA/MODELO mejora el modelo

| Config | CV MAE | Test MAE | Test R² |
|---|---|---|---|
| Con MARCA/MODELO (original) | 9.76 | 6.45 | 0.833 |
| Sin MARCA/MODELO (ablación) | 5.59 | 5.65 | 0.890 |
| Δ (sin − con) | -4.17 | -0.80 | +0.057 |

**Quitar MARCA/MODELO mejora el modelo**, tanto en validación cruzada como
en test. Con esas features hay además una brecha grande entre CV MAE (9.76)
y test MAE (6.45) — síntoma de que son categóricas de alta cardinalidad que
generan ruido/overfitting en la validación cruzada (categorías raras que
caen en un fold y no en otro). Sin ellas, CV y test casi coinciden
(5.59 vs 5.65), señal de un modelo más estable y mejor generalizado.

## Conclusión práctica

MARCA/MODELO no aportan señal incremental real más allá de lo que ya
capturan las features numéricas (motor, peso, combustible, etc.), y sí
añaden ruido. Vale la pena considerar:

- Eliminarlas del pipeline de producción, o
- Sustituirlas por una variable de categoría/segmento de menor cardinalidad
  (p. ej. agrupando por tipo de vehículo) si se quiere conservar parte de
  esa información sin el overfitting asociado a la alta cardinalidad.
