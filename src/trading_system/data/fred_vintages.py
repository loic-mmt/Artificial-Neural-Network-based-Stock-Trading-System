"""Download ALFRED observation histories, keeping date-level release metadata raw."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


API = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_SERIES = {
    "CPIAUCSL": "us_cpi_index",
    "UNRATE": "us_unemployment_rate",
    "PAYEMS": "us_nonfarm_payrolls",
    "INDPRO": "us_industrial_production_index",
}
_SERIES_ID = re.compile(r"[A-Z][A-Z0-9_]{0,39}\Z")
_COLUMNS = ["series_id", "series_name", "observation_date", "value",
            "realtime_start", "realtime_end", "retrieved_at_utc"]


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO date: {value}") from exc


def _page(api_key: str, series_id: str, *, observation_start: str,
          realtime_start: str, realtime_end: str, offset: int, limit: int,
          timeout: int) -> dict:
    params = {
        "api_key": api_key,
        "series_id": series_id,
        "file_type": "json",
        "output_type": 1,
        "observation_start": observation_start,
        "realtime_start": realtime_start,
        "realtime_end": realtime_end,
        "limit": limit,
        "offset": offset,
    }
    request = Request(f"{API}?{urlencode(params)}",
                      headers={"User-Agent": "TradingSystemVintageResearch/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except HTTPError as exc:
        # HTTPError and its URL include the API key. Never chain or print it.
        try:
            detail = json.loads(exc.read()).get("error_message", "")
        except (ValueError, AttributeError):
            detail = ""
        detail = str(detail).replace(api_key, "[redacted]")[:240]
        raise RuntimeError(f"FRED HTTP {exc.code} for {series_id} offset {offset}: {detail}") from None
    except (URLError, TimeoutError):
        raise RuntimeError(f"FRED network error for {series_id} offset {offset}") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("observations"), list):
        raise ValueError(f"Invalid FRED response for {series_id} offset {offset}")
    return payload


def _merge_window_fragments(frame: pd.DataFrame) -> pd.DataFrame:
    """Undo artificial real-time interval boundaries introduced by query windows."""
    merged = []
    for _, group in frame.groupby("observation_date", sort=False):
        previous = None
        for row in group.sort_values("realtime_start").to_dict("records"):
            same_value = previous is not None and (
                row["value"] == previous["value"] or
                (pd.isna(row["value"]) and pd.isna(previous["value"])))
            adjacent = (previous is not None and
                        previous["realtime_end"] != "9999-12-31" and
                        row["realtime_start"].date() ==
                        _date(previous["realtime_end"]) + timedelta(days=1))
            if same_value and adjacent:
                previous["realtime_end"] = row["realtime_end"]
            else:
                merged.append(row)
                previous = row
    return pd.DataFrame.from_records(merged)


def fetch_series(api_key: str, series_id: str, *, name: str | None = None,
                 observation_start: str = "2000-01-01", realtime_start: str = "2000-01-01",
                 realtime_end: str | None = None, limit: int = 100000,
                 timeout: int = 120) -> pd.DataFrame:
    """Fetch every paginated observation interval for one ALFRED series."""
    if not api_key or not _SERIES_ID.fullmatch(series_id):
        raise ValueError("A FRED key and valid series ID are required.")
    end = realtime_end or date.today().isoformat()
    if _date(realtime_start) > _date(end) or limit < 1 or limit > 100000 or timeout <= 0:
        raise ValueError("Invalid real-time range, limit, or timeout.")
    _date(observation_start)
    stamp = datetime.now(timezone.utc)
    records: list[dict] = []
    window_start = _date(realtime_start)
    final_day = _date(end)
    while window_start <= final_day:
        # FRED JSON accepts at most 2000 vintage dates per request. Four-year
        # windows also accommodate dense daily series such as DGS2.
        window_end = min(window_start + timedelta(days=1460), final_day)
        offset = 0
        while True:
            payload = _page(api_key, series_id, observation_start=observation_start,
                            realtime_start=window_start.isoformat(),
                            realtime_end=("9999-12-31" if realtime_end is None and
                                          window_end == final_day else window_end.isoformat()),
                            offset=offset, limit=limit, timeout=timeout)
            batch = payload["observations"]
            count = int(payload.get("count", -1))
            if count < 0 or (offset < count and not batch):
                raise ValueError(f"Incomplete FRED pagination for {series_id} offset {offset}")
            records.extend(batch)
            offset += len(batch)
            if offset >= count:
                break
        window_start = window_end + timedelta(days=1)
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError(f"No ALFRED observations for {series_id}.")
    required = {"date", "value", "realtime_start", "realtime_end"}
    if required - set(frame):
        raise ValueError(f"Missing ALFRED fields for {series_id}.")
    dates = {column: pd.to_datetime(frame[column], errors="coerce")
             for column in ("date", "realtime_start")}
    if any(values.isna().any() for values in dates.values()):
        raise ValueError(f"Invalid ALFRED dates for {series_id}.")
    try:
        frame["realtime_end"].map(_date)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid ALFRED real-time end for {series_id}.") from exc
    values = pd.to_numeric(frame["value"].replace(".", pd.NA), errors="coerce")
    if (frame["value"].ne(".") & values.isna()).any():
        raise ValueError(f"Invalid ALFRED values for {series_id}.")
    result = pd.DataFrame({
        "series_id": series_id,
        "series_name": name or series_id,
        "observation_date": dates["date"],
        "value": values.astype(float),
        "realtime_start": dates["realtime_start"],
        # ALFRED uses 9999-12-31 for open intervals, outside pandas' datetime range.
        "realtime_end": frame["realtime_end"],
        "retrieved_at_utc": stamp,
    })
    keys = ["series_id", "observation_date", "realtime_start", "realtime_end"]
    if result.groupby(keys, dropna=False)["value"].nunique(dropna=False).gt(1).any():
        raise ValueError(f"Conflicting ALFRED values for {series_id}.")
    # The same interval may intersect two adjacent real-time windows.
    result = result.drop_duplicates(keys)
    result = _merge_window_fragments(result)
    return result[_COLUMNS].sort_values(["observation_date", "realtime_start"]).reset_index(drop=True)


def save_vintages(frames: list[pd.DataFrame], output: str | Path) -> dict:
    """Create a complete raw archive atomically; refuse to overwrite existing data."""
    path = Path(output).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"FRED vintage archive already exists: {path}")
    if not frames:
        raise ValueError("At least one series is required.")
    archive = pd.concat(frames, ignore_index=True)
    if archive.empty or set(_COLUMNS) - set(archive):
        raise ValueError("Empty or invalid FRED archive.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.staging")
    if temporary.exists():
        raise FileExistsError(f"Vintage staging path already exists: {temporary}")
    try:
        archive.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(path), "rows": len(archive),
        "series": {sid: {"rows": len(part), "first_observation": part["observation_date"].min().date().isoformat(),
                         "last_observation": part["observation_date"].max().date().isoformat(),
                         "first_realtime_start": part["realtime_start"].min().date().isoformat(),
                         "last_realtime_start": part["realtime_start"].max().date().isoformat()}
                   for sid, part in archive.groupby("series_id")},
        "point_in_time_ready": False,
    }
