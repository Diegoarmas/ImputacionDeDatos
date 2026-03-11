import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMN = "EMISIONES_CO2"
DATE_COLUMNS = ["FECHA_MATR", "FEC_PRIM_MATR"]
KNOWN_NUMERIC_COLUMNS = [
  "TARA",
  "PESO_MAX",
  "MOM",
  "MMTA",
  "CILINDRADA",
  "POTENCIA",
  "KW",
  "CONSUMO",
  "AUTONOMIA",
  "DISTANCIA_EJES",
  "EJE_ANTERIOR",
  "EJE_POSTERIOR",
  "PLAZAS",
  "PLAZAS_MAX",
  "PLAZAS_PIE",
]


def to_float_series(series: pd.Series) -> pd.Series:
  # Normaliza formato europeo (miles con punto y decimal con coma) a float.
  cleaned = (
    series.astype(str)
    .str.strip()
    .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
  )
  return pd.to_numeric(cleaned, errors="coerce")


def prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
  data = df.copy()

  # Separa la variable objetivo para no usarla como feature.
  target = to_float_series(data[TARGET_COLUMN])
  data = data.drop(columns=[TARGET_COLUMN])

  # Convierte fechas en variables de calendario para el modelo.
  for col in DATE_COLUMNS:
    if col in data.columns:
      parsed = pd.to_datetime(data[col], format="%d/%m/%Y", errors="coerce")
      data[f"{col}_YEAR"] = parsed.dt.year
      data[f"{col}_MONTH"] = parsed.dt.month
      data = data.drop(columns=[col])

  numeric_columns = [c for c in KNOWN_NUMERIC_COLUMNS if c in data.columns]
  for col in numeric_columns:
    data[col] = to_float_series(data[col])

  for col in data.columns:
    if (
      col not in numeric_columns
      and not col.endswith("_YEAR")
      and not col.endswith("_MONTH")
    ):
      data[col] = (
        data[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
      )

  return data, target


def _repaired_rows(path: Path, sep: str, encoding: str) -> tuple[list[list[str]], int]:
  """Repair rows split across physical lines by joining until field count matches header."""
  with open(path, "r", encoding=encoding, newline="") as f:
    raw_lines = f.readlines()

  if not raw_lines:
    return [], 0

  header_row = next(csv.reader([raw_lines[0]], delimiter=sep))
  expected_fields = len(header_row)
  repaired = [header_row]
  broken_rows = 0

  # Acumula texto cuando una fila esta cortada en varias lineas fisicas.
  buffer = ""
  for line in raw_lines[1:]:
    if not buffer:
      buffer = line.rstrip("\n")
    else:
      buffer = f"{buffer} {line.lstrip().rstrip('\n')}"

    parsed = next(csv.reader([buffer], delimiter=sep))

    if len(parsed) < expected_fields:
      continue

    if len(parsed) > expected_fields:
      # Keep malformed wide rows out of training data instead of crashing.
      broken_rows += 1
      buffer = ""
      continue

    repaired.append(parsed)
    if " " in buffer and line != raw_lines[-1]:
      broken_rows += 1
    buffer = ""

  if buffer:
    parsed = next(csv.reader([buffer], delimiter=sep))
    if len(parsed) == expected_fields:
      repaired.append(parsed)
    else:
      broken_rows += 1

  return repaired, broken_rows


def load_csv_resilient(path: Path, sep: str, encoding: str) -> pd.DataFrame:
  try:
    return pd.read_csv(
      path,
      sep=sep,
      encoding=encoding,
      dtype=str,
      low_memory=False,
    )
  except pd.errors.ParserError as exc:
    print(
      f"Aviso: parseo CSV estandar fallo ({exc}). Intentando reparacion de lineas..."
    )

  # Fallback: repara filas rotas antes de volver a parsear con pandas.
  repaired_rows, broken_rows = _repaired_rows(path, sep, encoding)
  if not repaired_rows:
    raise ValueError("No se pudo leer el CSV: archivo vacio o corrupto.")

  repaired_csv = io.StringIO()
  writer = csv.writer(repaired_csv, delimiter=sep)
  writer.writerows(repaired_rows)
  repaired_csv.seek(0)

  df = pd.read_csv(
    repaired_csv,
    sep=sep,
    dtype=str,
    low_memory=False,
  )
  if broken_rows > 0:
    print(f"Aviso: se repararon o descartaron {broken_rows} filas mal formadas.")
  return df
