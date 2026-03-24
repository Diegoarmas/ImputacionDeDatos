import argparse
import sys
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Procesa archivos grandes por bloques y exporta a CSV."
  )
  parser.add_argument(
    "--input",
    default="data/raw/muestra_50k.txt",
    help="Ruta del archivo de entrada.",
  )
  parser.add_argument(
    "--output",
    default="data/processed/muestra_50k_con_co2.csv",
    help="Ruta del archivo de salida.",
  )
  parser.add_argument(
    "--in-sep",
    default="|",
    help="Separador del archivo de entrada.",
  )
  parser.add_argument(
    "--out-sep",
    default=",",
    help="Separador del archivo de salida.",
  )
  parser.add_argument(
    "--in-encoding",
    default="iso-8859-1",
    help="Codificacion del archivo de entrada.",
  )
  parser.add_argument(
    "--out-encoding",
    default="utf-8",
    help="Codificacion del archivo de salida.",
  )
  parser.add_argument(
    "--chunksize",
    type=int,
    default=100_000,
    help="Numero de filas por bloque.",
  )
  parser.add_argument(
    "--columns",
    default="",
    help="Columnas separadas por coma a conservar. Si se omite, se conservan todas.",
  )
  parser.add_argument(
    "--target-column",
    default="EMISIONES_CO2",
    help="Columna objetivo usada para filtrar filas con valor informado.",
  )
  parser.add_argument(
    "--keep-only-with-target",
    action="store_true",
    default=True,
    help="Conserva solo filas con valor en la columna objetivo.",
  )
  return parser


def _has_value(series: pd.Series) -> pd.Series:
  cleaned = series.astype(str).str.strip().str.lower()
  return ~cleaned.isin({"", "nan", "none", "null"})


def main() -> int:
  parser = build_parser()
  args = parser.parse_args()

  output_path = Path(args.output)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  selected_columns = [c.strip() for c in args.columns.split(",") if c.strip()]
  total_rows = 0
  kept_rows = 0

  try:
    reader = pd.read_csv(
      args.input,
      sep=args.in_sep,
      chunksize=args.chunksize,
      low_memory=False,
      encoding=args.in_encoding,
      dtype=str,
    )

    for i, chunk in enumerate(reader):
      if selected_columns:
        missing = [c for c in selected_columns if c not in chunk.columns]
        if missing:
          print(
            "Error: columnas no encontradas en la entrada: "
            + ", ".join(missing)
          )
          print("Columnas disponibles: " + ", ".join(chunk.columns))
          return 1
        chunk = chunk[selected_columns]

      if args.keep_only_with_target:
        if args.target_column not in chunk.columns:
          print(
            f"Error: no existe la columna objetivo {args.target_column} en la entrada."
          )
          return 1
        chunk = chunk[_has_value(chunk[args.target_column])]

      mode = "w" if i == 0 else "a"
      write_header = i == 0
      chunk.to_csv(
        output_path,
        index=False,
        mode=mode,
        header=write_header,
        sep=args.out_sep,
        encoding=args.out_encoding,
      )

      total_rows += len(chunk)
      kept_rows += len(chunk)
      print(f"Chunk {i}: {len(chunk)} filas exportadas con {args.target_column} informado")

  except FileNotFoundError:
    print(f"Error: no se encontro el archivo de entrada: {args.input}")
    return 1
  except UnicodeDecodeError as exc:
    print(f"Error de codificacion leyendo {args.input}: {exc}")
    print(
      "Prueba con --in-encoding iso-8859-1 o --in-encoding utf-8-sig segun tu archivo."
    )
    return 1
  except Exception as exc:
    print(f"Error inesperado: {exc}")
    return 1

  print(
    f"Proceso completado. Filas de salida con {args.target_column}: {kept_rows}. "
    f"Salida: {output_path}"
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
