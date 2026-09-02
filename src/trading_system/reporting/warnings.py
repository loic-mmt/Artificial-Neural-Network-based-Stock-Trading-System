from __future__ import annotations

from pathlib import Path

SURVIVOR_BIAS_WARNING = (
    "WARNING: current-universe historical results are survivor-biased and "
    "provisional until point-in-time membership data is available."
)


def current_universe_warning(data_path: str | Path | None) -> str | None:
    if data_path is None:
        return SURVIVOR_BIAS_WARNING
    name = Path(data_path).name.lower()
    if "cac40" in name or "market_universe" in name:
        return SURVIVOR_BIAS_WARNING
    return None


__all__ = ["SURVIVOR_BIAS_WARNING", "current_universe_warning"]
