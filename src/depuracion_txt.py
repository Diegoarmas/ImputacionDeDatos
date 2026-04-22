import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class SplitStats:
  rows: int = 0
  min_date: pd.Timestamp | None = None
  max_date: pd.Timestamp | None = None
  year_counter: Counter[int] | None = None

  def __post_init__(self) -> None:
    if self.year_counter is None:
      self.year_counter = Counter()

  def update_with_dates(self, date_series: pd.Series) -> None:
    parsed = pd.to_datetime(date_series, dayfirst=True, errors="coerce")
    valid = parsed.dropna()
    if valid.empty:
      return

    current_min = valid.min()
    current_max = valid.max()

    if self.min_date is None or current_min < self.min_date:
      self.min_date = current_min
    if self.max_date is None or current_max > self.max_date:
      self.max_date = current_max

    years = valid.dt.year.astype(int)
    self.year_counter.update(years.tolist())

  def as_dict(self) -> dict:
    years_sorted = {
      str(year): count for year, count in sorted(self.year_counter.items())
    }
    return {
      "rows": int(self.rows),
      "date_min": self.min_date.strftime("%Y-%m-%d") if self.min_date is not None else None,
      "date_max": self.max_date.strftime("%Y-%m-%d") if self.max_date is not None else None,
      "year_distribution": years_sorted,
    }


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Procesa archivos grandes por bloques y exporta a CSV/pools."
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
      "Directorio para exportar multiples muestras CSV de entrenamiento. "
      "Si se define, se generan archivos por bloques y se ignora --output."
    ),
  )
  parser.add_argument(
    "--pool-val-dir",
    default="",
    help="Directorio para exportar pool de validacion.",
  )
  parser.add_argument(
    "--pool-test-dir",
    default="",
    help="Directorio para exportar pool de test.",
  )
  parser.add_argument(
    "--train-ratio",
    type=float,
    default=0.7,
    help="Proporcion para entrenamiento (default: 0.7).",
  )
  parser.add_argument(
    "--val-ratio",
    type=float,
    default=0.15,
    help="Proporcion para validacion (default: 0.15).",
  )
  parser.add_argument(
    "--rows-per-sample",
    type=int,
    default=100_000,
    help="Numero de filas por muestra CSV cuando se usa modo pool.",
  )
  parser.add_argument(
    "--max-samples",
    type=int,
    default=0,
    help="Maximo total de muestras a generar entre todos los pools. 0 = sin limite.",
  )
  parser.add_argument(
    "--pool-prefix",
    default="muestra_100k_",
    help="Prefijo de nombre de archivo para muestras en pools.",
  )
  parser.add_argument(
    "--keep-remainder",
    action="store_true",
    help="Guarda una muestra final incompleta si quedan filas en buffer.",
  )
  parser.add_argument(
    "--period-column",
    default="FECHA_MATR",
    help="Columna de fecha para estudiar el periodo temporal por split.",
  )
  parser.add_argument(
    "--split-report-output",
    default="artifacts/metrics/split_period_report.json",
    help="Ruta JSON del reporte temporal de train/val/test.",
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

  pool_mode = bool(args.pool_dir)
  has_any_secondary_pool = bool(args.pool_val_dir) or bool(args.pool_test_dir)
  if has_any_secondary_pool and not pool_mode:
    raise ValueError("--pool-val-dir/--pool-test-dir requieren --pool-dir")

  if pool_mode:
    if not (0.0 < args.train_ratio < 1.0):
      raise ValueError("--train-ratio debe estar entre 0 y 1")
    if not (0.0 <= args.val_ratio < 1.0):
      raise ValueError("--val-ratio debe estar entre 0 y 1")
    if args.train_ratio + args.val_ratio >= 1.0:
      raise ValueError("train_ratio + val_ratio debe ser < 1.0 (deja espacio para test)")


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


def _split_name_for_index(
  global_row_index: int,
  train_threshold: int,
  val_threshold: int,
) -> str:
  # 10_000 buckets da precision suficiente para ratios como 0.733
  bucket = global_row_index % 10_000
  if bucket < train_threshold:
    return "train"
  if bucket < val_threshold:
    return "val"
  return "test"


def _update_period_stats_for_split(
  stats: dict[str, SplitStats],
  split_name: str,
  split_df: pd.DataFrame,
  period_column: str,
) -> None:
  if split_name not in stats:
    return

  split_stats = stats[split_name]
  split_stats.rows += len(split_df)
  if period_column in split_df.columns and not split_df.empty:
    split_stats.update_with_dates(split_df[period_column])


def _build_split_report(
  input_rows: int,
  kept_rows: int,
  train_ratio: float,
  val_ratio: float,
  stats: dict[str, SplitStats],
) -> dict:
  return {
    "rows_read": int(input_rows),
    "rows_usable": int(kept_rows),
    "ratios_requested": {
      "train": float(train_ratio),
      "val": float(val_ratio),
      "test": float(1.0 - train_ratio - val_ratio),
    },
    "splits": {
      split_name: split_stats.as_dict()
      for split_name, split_stats in stats.items()
    },
  }


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

  train_pool_dir = Path(args.pool_dir) if args.pool_dir else None
  val_pool_dir = Path(args.pool_val_dir) if args.pool_val_dir else None
  test_pool_dir = Path(args.pool_test_dir) if args.pool_test_dir else None

  if train_pool_dir:
    train_pool_dir.mkdir(parents=True, exist_ok=True)
  if val_pool_dir:
    val_pool_dir.mkdir(parents=True, exist_ok=True)
  if test_pool_dir:
    test_pool_dir.mkdir(parents=True, exist_ok=True)

  split_report_path = Path(args.split_report_output)
  split_report_path.parent.mkdir(parents=True, exist_ok=True)

  selected_columns = [c.strip() for c in args.columns.split(",") if c.strip()]
  input_rows = 0
  kept_rows = 0

  sample_indices = {"train": 1, "val": 1, "test": 1}
  sample_counts = {"train": 0, "val": 0, "test": 0}
  buffers = {"train": pd.DataFrame(), "val": pd.DataFrame(), "test": pd.DataFrame()}
  split_stats = {
    "train": SplitStats(),
    "val": SplitStats(),
    "test": SplitStats(),
  }

  global_row_index = 0
  train_threshold = int(args.train_ratio * 10_000)
  val_threshold = int((args.train_ratio + args.val_ratio) * 10_000)

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
      if args.max_samples > 0 and sum(sample_counts.values()) >= args.max_samples:
        print(f"Info: alcanzado maximo total de muestras ({args.max_samples}).")
        break

      input_rows += len(raw_chunk)
      chunk = _select_columns(raw_chunk, selected_columns)
      chunk = _filter_target_rows(
        chunk,
        keep_only_with_target=args.keep_only_with_target,
        target_column=args.target_column,
      )
      kept_rows += len(chunk)

      if train_pool_dir is None:
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

      split_rows = {"train": [], "val": [], "test": []}
      for _, row in chunk.iterrows():
        split_name = _split_name_for_index(
          global_row_index=global_row_index,
          train_threshold=train_threshold,
          val_threshold=val_threshold,
        )
        split_rows[split_name].append(row)
        global_row_index += 1

      split_frames = {
        k: pd.DataFrame(v) if v else pd.DataFrame(columns=chunk.columns)
        for k, v in split_rows.items()
      }

      if not val_pool_dir:
        split_frames["train"] = pd.concat(
          [split_frames["train"], split_frames["val"]], ignore_index=True
        )
        split_frames["val"] = pd.DataFrame(columns=chunk.columns)

      if not test_pool_dir:
        split_frames["train"] = pd.concat(
          [split_frames["train"], split_frames["test"]], ignore_index=True
        )
        split_frames["test"] = pd.DataFrame(columns=chunk.columns)

      for split_name, split_df in split_frames.items():
        _update_period_stats_for_split(
          stats=split_stats,
          split_name=split_name,
          split_df=split_df,
          period_column=args.period_column,
        )

      target_dirs = {
        "train": train_pool_dir,
        "val": val_pool_dir,
        "test": test_pool_dir,
      }

      for split_name, split_df in split_frames.items():
        target_dir = target_dirs[split_name]
        if target_dir is None or split_df.empty:
          continue

        if buffers[split_name].empty:
          combined = split_df
        else:
          combined = pd.concat([buffers[split_name], split_df], ignore_index=True)

        while len(combined) >= args.rows_per_sample:
          if args.max_samples > 0 and sum(sample_counts.values()) >= args.max_samples:
            break

          sample_df = combined.iloc[:args.rows_per_sample].copy()
          sample_path = _write_pool_sample(
            sample_df=sample_df,
            sample_index=sample_indices[split_name],
            pool_dir=target_dir,
            pool_prefix=args.pool_prefix,
            out_sep=args.out_sep,
            out_encoding=args.out_encoding,
          )
          sample_indices[split_name] += 1
          sample_counts[split_name] += 1
          print(
            f"Muestra {split_name.upper()} {sample_counts[split_name]}: {len(sample_df)} filas -> {sample_path}"
          )

          combined = combined.iloc[args.rows_per_sample:].reset_index(drop=True)

        buffers[split_name] = combined

      print(
        f"Chunk {i}: train={len(split_frames['train'])}, val={len(split_frames['val'])}, "
        f"test={len(split_frames['test'])}. Buffers: "
        f"train={len(buffers['train'])}, val={len(buffers['val'])}, test={len(buffers['test'])}"
      )

    if train_pool_dir and args.keep_remainder:
      target_dirs = {
        "train": train_pool_dir,
        "val": val_pool_dir,
        "test": test_pool_dir,
      }
      for split_name in ("train", "val", "test"):
        target_dir = target_dirs[split_name]
        if target_dir is None:
          continue
        if buffers[split_name].empty:
          continue
        if args.max_samples > 0 and sum(sample_counts.values()) >= args.max_samples:
          break

        sample_path = _write_pool_sample(
          sample_df=buffers[split_name],
          sample_index=sample_indices[split_name],
          pool_dir=target_dir,
          pool_prefix=args.pool_prefix,
          out_sep=args.out_sep,
          out_encoding=args.out_encoding,
        )
        sample_counts[split_name] += 1
        print(
          f"Muestra final {split_name.upper()} (remanente): "
          f"{len(buffers[split_name])} filas -> {sample_path}"
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

  if train_pool_dir:
    report = _build_split_report(
      input_rows=input_rows,
      kept_rows=kept_rows,
      train_ratio=args.train_ratio,
      val_ratio=args.val_ratio,
      stats=split_stats,
    )
    with open(split_report_path, "w", encoding="utf-8") as f:
      json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== RESULTADO FINAL ===")
    print(f"Filas leidas: {input_rows}")
    print(f"Filas utiles: {kept_rows}")
    print(
      f"Ratios solicitados -> train={args.train_ratio:.2f}, "
      f"val={args.val_ratio:.2f}, test={1.0 - args.train_ratio - args.val_ratio:.2f}"
    )
    print(
      f"Muestras -> train={sample_counts['train']}, "
      f"val={sample_counts['val']}, test={sample_counts['test']}"
    )
    print(f"Reporte temporal: {split_report_path}")
  else:
    print(
      f"Proceso completado. Filas leidas: {input_rows}. Filas de salida con "
      f"{args.target_column}: {kept_rows}. Salida: {output_path}"
    )

  return 0


if __name__ == "__main__":
  sys.exit(main())
