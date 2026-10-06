"""Small, training-independent contracts for persisted visualization data."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class RunRecord:
    run_id: str
    label: str
    family: str
    path: Path
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, Path] = field(default_factory=dict)
    selectors: dict[str, Any] = field(default_factory=dict)
    status: str = "complete"
    warnings: tuple[str, ...] = ()


@dataclass
class RunData:
    record: RunRecord
    equity: pd.DataFrame = field(default_factory=pd.DataFrame)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    history: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    market: pd.DataFrame = field(default_factory=pd.DataFrame)
    orders: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class Catalog:
    records: list[RunRecord] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
