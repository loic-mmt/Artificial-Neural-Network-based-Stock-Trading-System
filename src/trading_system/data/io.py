from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

try:
    import pyarrow.dataset as ds
except ImportError:  # pragma: no cover - optional runtime dependency
    ds = None


def read_parquet_dataset(
    base_dir: str | Path,
    columns: Sequence[str] | None = None,
    filter_expr: Any | None = None,
) -> pd.DataFrame:
    """Load a parquet file or hive-partitioned parquet directory."""

    path = Path(base_dir).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)

    projected = list(columns) if columns is not None else None
    if ds is not None:
        dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
        return dataset.to_table(columns=projected, filter=filter_expr).to_pandas()

    if filter_expr is not None:
        raise RuntimeError("Parquet filters require pyarrow.dataset.")
    if path.is_file():
        return pd.read_parquet(path, columns=projected)

    parquet_files = sorted(path.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {path}")
    parts = [
        pd.read_parquet(file_path, columns=projected) for file_path in parquet_files
    ]
    return pd.concat(parts, ignore_index=True)


__all__ = ["read_parquet_dataset"]
