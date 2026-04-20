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
    help="Ruta del archivo de salida (modo archivo unico).",
  )
  parser.add_argument(
    "--pool-dir",
    default="",
    help=(
      "Directorio para exportar multiples muestras CSV. "
      "Si se define, se generan archivos por bloques y se ignora --output."
    ),
  )
  parser.add_argument(
    "--rows-per-sample",
    type=int,
    default=100_000,
    help="Numero de filas por muestra CSV cuando se usa --pool-dir.",
  )
  parser.add_argument(
    "--max-samples",
    type=int,
    default=0,
    help="Maximo de muestras a generar en --pool-dir. 0 = sin limite.",
  )
  parser.add_argument(
    "--pool-prefix",
    default="muestra_100k_",
    help="Prefijo de nombre de archivo para muestras en --pool-dir.",
  )
  parser.add_argument(
    "--keep-remainder",
    action="store_true",
    help="Guarda una muestra final incompleta si quedan filas en buffer.",
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
    help="Numero de filas por bloque de lectura.",
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
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
      "Conserva solo filas con valor en la columna objetivo (default: True). "
      "Usa --no-keep-only-with-target para desactivarlo."
    ),
  )
  return parser


def _has_value(series: pd.Series) -> pd.Series:
  cleaned = series.fillna("").astype(str).str.strip().str.lower()
  return ~cleaned.isin({"", "nan", "none", "null", "<na>"})


def _validate_args(args: argparse.Namespace) -> None:
  if args.rows_per_sample < 1:
    raise ValueError("--rows-per-sample debe ser >= 1")
  if args.max_samples < 0:
    raise ValueError("--max-samples debe ser >= 0")


def _select_columns(chunk: pd.DataFrame, selected_columns: list[str]) -> pd.DataFrame:
  if not selected_columns:
    return chunk

  missing = [c for c in selected_columns if c not in chunk.columns]
  if missing:
    raise ValueError(
      "Error: columnas no encontradas en la entrada: "
      + ", ".join(missing)
      + "\nColumnas disponibles: "
      + ", ".join(chunk.columns)
    )
  return chunk[selected_columns]


def _filter_target_rows(
  chunk: pd.DataFrame,
  keep_only_with_target: bool,
  target_column: str,
) -> pd.DataFrame:
  if not keep_only_with_target:
    return chunk

  if target_column not in chunk.columns:
    raise ValueError(
      f"Error: no existe la columna objetivo {target_column} en la entrada."
    )
  return chunk[_has_value(chunk[target_column])]


def _write_pool_sample(
  sample_df: pd.DataFrame,
  sample_index: int,
  pool_dir: Path,
  pool_prefix: str,
  out_sep: str,
  out_encoding: str,
) -> Path:
  sample_path = pool_dir / f"{pool_prefix}{sample_index:04d}.csv"
  sample_df.to_csv(sample_path, index=False, sep=out_sep, encoding=out_encoding)
  return sample_path


def _write_single_chunk(
  chunk: pd.DataFrame,
  chunk_index: int,
  output_path: Path,
  out_sep: str,
  out_encoding: str,
) -> None:
  mode = "w" if chunk_index == 0 else "a"
  header = chunk_index == 0
  chunk.to_csv(
    output_path,
    index=False,
    mode=mode,
    header=header,
    sep=out_sep,
    encoding=out_encoding,
  )


def main() -> int:
  parser = build_parser()
  args = parser.parse_args()

  try:
    _validate_args(args)
  except ValueError as exc:
    print(f"Error: {exc}")
    return 1

  output_path = Path(args.output)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  pool_dir_path = Path(args.pool_dir) if args.pool_dir else None
  if pool_dir_path:
    pool_dir_path.mkdir(parents=True, exist_ok=True)

  selected_columns = [c.strip() for c in args.columns.split(",") if c.strip()]
  input_rows = 0
  kept_rows = 0
  generated_samples = 0
  sample_index = 1
  pool_buffer = pd.DataFrame()

  try:
    reader = pd.read_csv(
      args.input,
      sep=args.in_sep,
      chunksize=args.chunksize,
      low_memory=False,
      encoding=args.in_encoding,
      dtype=str,
    )

    for i, raw_chunk in enumerate(reader):
      if args.max_samples > 0 and generated_samples >= args.max_samples:
        print(f"Info: alcanzado maximo de muestras ({args.max_samples}).")
        break

      input_rows += len(raw_chunk)

      chunk = _select_columns(raw_chunk, selected_columns)
      chunk = _filter_target_rows(
        chunk,
        keep_only_with_target=args.keep_only_with_target,
        target_column=args.target_column,
      )
      kept_rows += len(chunk)

      if pool_dir_path is None:
        _write_single_chunk(
          chunk,
          chunk_index=i,
          output_path=output_path,
          out_sep=args.out_sep,
          out_encoding=args.out_encoding,
        )
        print(
          f"Chunk {i}: {len(chunk)} filas exportadas con {args.target_column} informado"
        )
        continue

      if chunk.empty:
        print(f"Chunk {i}: 0 filas utiles")
        continue

      if pool_buffer.empty:
        combined = chunk
      else:
        combined = pd.concat([pool_buffer, chunk], ignore_index=True)

      while len(combined) >= args.rows_per_sample:
        if args.max_samples > 0 and generated_samples >= args.max_samples:
          break

        sample_df = combined.iloc[:args.rows_per_sample].copy()
        sample_path = _write_pool_sample(
          sample_df=sample_df,
          sample_index=sample_index,
          pool_dir=pool_dir_path,
          pool_prefix=args.pool_prefix,
          out_sep=args.out_sep,
          out_encoding=args.out_encoding,
        )
        generated_samples += 1
        sample_index += 1
        print(f"Muestra {generated_samples}: {len(sample_df)} filas -> {sample_path}")

        combined = combined.iloc[args.rows_per_sample:].reset_index(drop=True)

      pool_buffer = combined
      print(
        f"Chunk {i}: {len(chunk)} filas utiles, acumulado pendiente: {len(pool_buffer)}"
      )

    if pool_dir_path and args.keep_remainder and not pool_buffer.empty:
      if args.max_samples == 0 or generated_samples < args.max_samples:
        sample_path = _write_pool_sample(
          sample_df=pool_buffer,
          sample_index=sample_index,
          pool_dir=pool_dir_path,
          pool_prefix=args.pool_prefix,
          out_sep=args.out_sep,
          out_encoding=args.out_encoding,
        )
        generated_samples += 1
        print(
          f"Muestra final (remanente): {len(pool_buffer)} filas -> {sample_path}"
        )

  except FileNotFoundError:
    print(f"Error: no se encontro el archivo de entrada: {args.input}")
    return 1
  except UnicodeDecodeError as exc:
    print(f"Error de codificacion leyendo {args.input}: {exc}")
    print(
      "Prueba con --in-encoding iso-8859-1 o --in-encoding utf-8-sig segun tu archivo."
    )
    return 1
  except ValueError as exc:
    print(str(exc))
    return 1
  except Exception as exc:
    print(f"Error inesperado: {exc}")
    return 1

  if pool_dir_path:
    print(
      f"Proceso completado. Filas leidas: {input_rows}. Filas utiles: {kept_rows}. "
      f"Muestras generadas: {generated_samples}. Directorio: {pool_dir_path}"
    )
  else:
    print(
      f"Proceso completado. Filas leidas: {input_rows}. Filas de salida con "
      f"{args.target_column}: {kept_rows}. Salida: {output_path}"
    )
  return 0


if __name__ == "__main__":
  sys.exit(main())
