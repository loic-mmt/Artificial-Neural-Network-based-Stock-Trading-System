from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def src_root() -> Path:
    return project_root() / "src"


def data_root() -> Path:
    return project_root() / "data"


def processed_data_dir() -> Path:
    return data_root() / "processed"


def derived_data_dir() -> Path:
    return data_root() / "derived"


def artifacts_root() -> Path:
    return project_root() / "artifacts"


def runs_dir() -> Path:
    return artifacts_root() / "runs"


def gridsearch_dir() -> Path:
    return artifacts_root() / "gridsearch"


def comparisons_dir() -> Path:
    return artifacts_root() / "comparisons"


def default_market_dataset_path(filename: str = "cac40_daily.parquet") -> Path:
    return processed_data_dir() / filename


def default_oracle_labels_path(filename: str = "oracle_labels_train_EN_PA.csv") -> Path:
    return derived_data_dir() / filename
