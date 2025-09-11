import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Set

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _iter_json_chunks(path: Path, chunksize: int):
    return pd.read_json(str(path), lines=True, chunksize=chunksize)


def _type_category(t: pa.DataType) -> str:
    if pa.types.is_null(t):
        return "null"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "string"
    if pa.types.is_floating(t):
        return "float"
    if pa.types.is_integer(t):
        return "int"
    if pa.types.is_boolean(t):
        return "bool"
    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        return "binary"
    return "other"


def _promote_type(categories: Set[str]) -> pa.DataType:
    cats = set(categories)
    cats.discard("null")
    if not cats:
        # all nulls; use string to allow nulls and be safe
        return pa.string()
    if "string" in cats:
        return pa.string()
    if "binary" in cats:
        # mixed with anything else is ambiguous; fall back to string
        return pa.binary() if cats == {"binary"} else pa.string()
    if "float" in cats or ("int" in cats and "bool" in cats):
        return pa.float64()
    if "int" in cats:
        # prefer float64 to survive later floats appearing
        return pa.float64()
    if "bool" in cats:
        return pa.bool_()
    # default
    return pa.string()


def _unified_schema_and_cols(path: Path, chunksize: int) -> tuple[pa.Schema, list[str], int]:
    total_rows = 0
    cols: list[str] = []
    colset: Set[str] = set()
    # Track categories seen per column
    seen: Dict[str, Set[str]] = {}
    for df in _iter_json_chunks(path, chunksize):
        n = len(df)
        if n == 0:
            continue
        total_rows += n
        table = pa.Table.from_pandas(df, preserve_index=False)
        for field in table.schema:
            name = field.name
            cat = _type_category(field.type)
            if name not in seen:
                seen[name] = set()
            seen[name].add(cat)
        for c in df.columns:
            if c not in colset:
                colset.add(c)
                cols.append(c)
    if total_rows == 0:
        raise ValueError(f"NDJSON is empty: {path}")

    # Build promoted schema
    fields = []
    for name in cols:
        cats = seen.get(name, {"null"})
        typ = _promote_type(cats)
        fields.append(pa.field(name, typ, nullable=True))
    schema = pa.schema(fields)
    return schema, cols, total_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to NDJSON produced by C++ logger")
    ap.add_argument("--out", dest="out", required=True, help="Output Parquet path")
    ap.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Number of rows per JSON chunk to process in memory",
    )
    ap.add_argument(
        "--row-group-size",
        type=int,
        default=100_000,
        help="Target Parquet row group size (rows) for each written chunk",
    )
    ap.add_argument(
        "--compression",
        type=str,
        default="snappy",
        choices=["snappy", "gzip", "zstd", "brotli", "lz4", "none"],
        help="Parquet compression codec",
    )
    ap.add_argument(
        "--one-pass",
        action="store_true",
        help=("Derive schema from first non-empty chunk and write in one pass. " "Faster but drops columns that appear only in later chunks."),
    )
    args = ap.parse_args()

    in_path = Path(args.inp).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"NDJSON not found: {in_path}")
    if in_path.stat().st_size == 0:
        raise ValueError(f"NDJSON is empty: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # We could do this in one line, but it would load entire file into memory
    # df = pd.read_json(str(in_path), lines=True)
    # df.to_parquet(str(out_path), index=False)

    # Determine schema and columns
    if args.one_pass:
        # Read first non-empty chunk to infer schema
        base_schema: Optional[pa.Schema] = None
        base_cols: Optional[list[str]] = None
        total_rows = 0
        writer: Optional[pq.ParquetWriter] = None
        try:
            for df in _iter_json_chunks(in_path, args.chunksize):
                if writer is None:
                    if len(df) == 0:
                        continue
                    total_rows += len(df)
                    # Infer and widen initial schema to be robust to later floats
                    inferred = pa.Table.from_pandas(df, preserve_index=False).schema
                    widened_fields = []
                    for f in inferred:
                        t = f.type
                        if pa.types.is_integer(t):
                            t = pa.float64()
                        widened_fields.append(pa.field(f.name, t, nullable=True))
                    base_schema = pa.schema(widened_fields)
                    base_cols = list(df.columns)
                    writer = pq.ParquetWriter(str(out_path), base_schema, compression=(None if args.compression == "none" else args.compression))
                    # Align columns and write
                    df = df.reindex(columns=base_cols)
                    table = pa.Table.from_pandas(df, schema=base_schema, preserve_index=False)
                    writer.write_table(table, row_group_size=args.row_group_size)
                else:
                    n = len(df)
                    if n == 0:
                        continue
                    total_rows += n
                    # align to base columns, dropping unseen columns
                    assert base_cols is not None and base_schema is not None
                    df = df.reindex(columns=base_cols)
                    table = pa.Table.from_pandas(df, schema=base_schema, preserve_index=False)
                    writer.write_table(table, row_group_size=args.row_group_size)
        finally:
            if "writer" in locals() and writer is not None:
                writer.close()
        if total_rows == 0:
            raise ValueError(f"NDJSON is empty: {in_path}")
        print(f"wrote {total_rows} rows to {out_path}")
        return

    # Two-pass: safer schema across all chunks
    schema, cols, total_rows = _unified_schema_and_cols(in_path, args.chunksize)

    writer = pq.ParquetWriter(str(out_path), schema, compression=(None if args.compression == "none" else args.compression))
    try:
        for df in _iter_json_chunks(in_path, args.chunksize):
            if len(df) == 0:
                continue
            # Reindex to full column set and order
            df = df.reindex(columns=cols)
            table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
            writer.write_table(table, row_group_size=args.row_group_size)
    finally:
        writer.close()
    print(f"wrote {total_rows} rows to {out_path}")


if __name__ == "__main__":
    main()
