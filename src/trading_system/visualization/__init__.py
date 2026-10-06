"""Read-only visualization of persisted research and backtest results."""

from .adapters import load_run
from .catalog import catalog_frame, catalog_signature, comparison_issues, discover_runs
from .schemas import Catalog, RunData, RunRecord
from .positions import asset_names, asset_frame, position_events

__all__ = [
    "Catalog", "RunData", "RunRecord", "catalog_frame", "catalog_signature",
    "comparison_issues", "discover_runs", "load_run",
    "asset_names", "asset_frame", "position_events",
]
