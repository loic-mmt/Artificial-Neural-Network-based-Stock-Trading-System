"""Preserve ECB RTD database vintages without inventing publication times.

ECB RTD VALID_FROM is a database-version timestamp. The archived record is not
automatically eligible as a point-in-time trading feature at that timestamp.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd


ECB_RTD_BASE = "https://data-api.ecb.europa.eu/service/data/RTD/"
DEFAULT_SERIES = {
    "RTD.M.S0.N.P_C_OV.X": "euro_area_hicp_index",
    "RTD.Q.S0.S.G_GDPM_TO_C.E": "euro_area_real_gdp",
}
_KEY = re.compile(r"RTD\.[AMQ]\.[A-Z0-9_.]+\Z")


def series_url(series_key: str) -> str:
    if not _KEY.fullmatch(series_key):
        raise ValueError("Invalid ECB RTD series key.")
    return f"{ECB_RTD_BASE}{quote(series_key[4:], safe='.')}?includeHistory=true&format=csvdata"


def fetch_csv(series_key: str, *, timeout: int = 120) -> bytes:
    if timeout <= 0:
        raise ValueError("timeout must be positive.")
    request = Request(series_url(series_key), headers={"User-Agent": "TradingSystemVintageResearch/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_csv(content: bytes, series_key: str, *, name: str,
              retrieved_at: datetime | None = None) -> pd.DataFrame:
    """Normalize historical replacements/deletions, retaining source provenance."""
    if not content:
        raise ValueError("Empty ECB RTD response.")
    source = pd.read_csv(BytesIO(content), dtype=str)
    required = {"KEY", "TIME_PERIOD", "OBS_VALUE", "ACTION", "VALID_FROM", "VALID_TO", "TITLE"}
    if required - set(source):
        raise ValueError(f"ECB RTD response lacks columns: {sorted(required - set(source))}")
    if source.empty or set(source["KEY"].dropna()) != {series_key}:
        raise ValueError("ECB RTD response does not match the requested series.")
    if source["TIME_PERIOD"].isna().any() or not bool(source["ACTION"].isin(["Replace", "Delete"]).all()):
        raise ValueError("ECB RTD response contains invalid periods or actions.")
    values = pd.to_numeric(source["OBS_VALUE"], errors="coerce")
    if (source["OBS_VALUE"].notna() & values.isna()).any():
        raise ValueError("ECB RTD response has a nonnumeric observation.")
    valid_from = pd.to_datetime(source["VALID_FROM"], utc=True, errors="coerce")
    valid_to = pd.to_datetime(source["VALID_TO"], utc=True, errors="coerce")
    if (source["VALID_FROM"].notna() & valid_from.isna()).any() or (
        source["VALID_TO"].notna() & valid_to.isna()
    ).any():
        raise ValueError("ECB RTD response contains invalid version timestamps.")
    if valid_from[source["ACTION"].eq("Replace")].isna().any():
        raise ValueError("ECB RTD replacement lacks VALID_FROM.")
    stamp = pd.Timestamp(retrieved_at or datetime.now(timezone.utc))
    if stamp.tzinfo is None:
        raise ValueError("retrieved_at must be timezone-aware.")
    result = pd.DataFrame({
        "series_key": series_key,
        "series_name": name,
        "observation_period": source["TIME_PERIOD"],
        "observation_value": values.astype(float),
        "action": source["ACTION"],
        "database_valid_from_utc": valid_from,
        "database_valid_to_utc": valid_to,
        "title": source["TITLE"],
        "source_url": series_url(series_key),
        "source_sha256": sha256(content).hexdigest(),
        "retrieved_at_utc": stamp.tz_convert("UTC"),
    })
    if result.duplicated(["series_key", "observation_period", "database_valid_from_utc",
                          "database_valid_to_utc", "action"]).any():
        raise ValueError("ECB RTD response contains duplicate version records.")
    return result


def save_vintages(sources: dict[str, bytes], output: str | Path,
                  *, names: dict[str, str] | None = None) -> dict:
    """Write a new research archive. No available_at field is synthesized."""
    path = Path(output).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Vintage archive already exists: {path}")
    if not sources:
        raise ValueError("At least one ECB RTD series is required.")
    labels = names or DEFAULT_SERIES
    retrieved_at = datetime.now(timezone.utc)
    parts = [parse_csv(content, key, name=labels.get(key, key), retrieved_at=retrieved_at)
             for key, content in sources.items()]
    archive = pd.concat(parts, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.staging")
    if temporary.exists():
        raise FileExistsError(f"Vintage staging path already exists: {temporary}")
    try:
        archive.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(path), "rows": len(archive), "series": {
        key: {"rows": len(part), "versions": int(part["database_valid_from_utc"].nunique()),
              "first_version": part["database_valid_from_utc"].min().isoformat(),
              "last_version": part["database_valid_from_utc"].max().isoformat()}
        for key, part in zip(sources, parts)
    }, "point_in_time_ready": False}


__all__ = ["DEFAULT_SERIES", "fetch_csv", "parse_csv", "save_vintages", "series_url"]
