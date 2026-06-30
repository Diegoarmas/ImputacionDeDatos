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

### Ablación: efecto pequeño pero confirmado y reproducible

El experimento se ejecutó **tres veces** con el mismo `random_state=42` y
los mismos datos. Las dos últimas ejecuciones dieron resultados
**bit-idénticos** (16 cifras significativas), confirmando que la
implementación actual es totalmente determinista en este entorno:

| Config | CV MAE | Test MAE | Test R² |
|---|---|---|---|
| Con MARCA/MODELO (original) | 5.34 | 5.27 | 0.907 |
| Sin MARCA/MODELO (ablación) | 5.41 | 5.34 | 0.905 |
| **Δ (sin − con)** | **+0.06** | **+0.07** | **-0.002** |

**Nota sobre una discrepancia previa**: una versión anterior de este
documento citaba un resultado muy distinto (Δ test MAE = **-0.80**, "quitar
MARCA/MODELO mejora mucho"), tomado del CSV generado por el commit
`79a1b88`. Al investigar la causa, se encontró que **no era ruido de
entrenamiento sino un bug de metodología** ya corregido en el commit
`42e8340` ("...including device configuration"):

- En la versión vieja (`79a1b88`), el config "Con MARCA/MODELO" **no se
  reentrenaba**: reutilizaba métricas de `artifacts/metrics/co2_metrics.json`
  (un experimento distinto, con datos/configuración diferentes) o las
  métricas globales del modelo de producción. El config "Sin MARCA/MODELO"
  sí se entrenaba desde cero, pero siempre en **CPU**
  (`device="cpu"` fijo), sin importar que el modelo de producción usara GPU.
  Era una comparación de peras con manzanas.
- En la versión actual (`42e8340`, la que usa este documento), ambos
  configs se reentrenan desde cero, con el mismo pool, mismo `device`
  (cuda) y mismo seed — comparación correcta y controlada. Ver
  [src/brand_model_analysis.py:255-259](src/brand_model_analysis.py#L255-L259).

Con la implementación corregida, el resultado es estable: **quitar
MARCA/MODELO empeora ligeramente** el modelo (+0.067 g/km de MAE, -0.002 de
R² en test), un efecto pequeño (~1.3% relativo sobre el MAE de 5.27) pero
consistente en las tres ejecuciones.

## Conclusión práctica

- MARCA/MODELO aportan una contribución **positiva pero pequeña** a la
  precisión: sin ellas, el MAE de test sube de 5.27 a 5.34 g/km (+0.067) y
  el R² baja de 0.907 a 0.905. El efecto es real y reproducible, no ruido.
- La magnitud es demasiado pequeña para justificar eliminarlas solo por
  precisión. Si se quitaran, sería por otras razones (reducir cardinalidad
  de categóricas, simplificar el pipeline, evitar mantenimiento de un
  encoder con cientos de marcas/modelos), asumiendo ese pequeño coste en
  error.
- Lo que sí es sólido y no depende de la ablación (viene de inferencia con
  el modelo de producción ya entrenado, determinista) son los hallazgos por
  marca/modelo de las secciones anteriores: la alta heterogeneidad de error
  entre marcas/modelos y la mala generalización a motos/SUV de nicho.
