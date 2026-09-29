"""Scheduled events known at decision time; no surprise data is consumed."""

from pathlib import Path
import pandas as pd


def load_events(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def prepare_events(events, bars, config):
    if not config.enabled or config.event_policy == "none":
        return pd.DataFrame()
    if isinstance(events, (str, Path)):
        events = load_events(events)
    required = {"event_type", "timestamp", "known_at", "scope", "scope_value"}
    if events is None or events.empty or not required.issubset(events):
        raise ValueError("Enabled event filter requires a usable calendar: event_type, timestamp, known_at, scope, scope_value.")
    work = events.copy().reset_index(drop=True)
    for name in ("timestamp", "known_at"):
        work[name] = pd.to_datetime(work[name], utc=True, errors="raise")
    if work[["timestamp", "known_at"]].isna().any().any():
        raise ValueError("Event times cannot be missing.")
    if not work.scope.isin(["global", "ticker", "sector"]).all():
        raise ValueError("Event scope must be global, ticker or sector.")
    if work.event_type.isna().any() or work.event_type.astype(str).str.strip().eq("").any():
        raise ValueError("event_type cannot be empty.")
    targeted = work.scope.ne("global")
    if (work.loc[targeted, "scope_value"].isna().any() or work.loc[targeted, "scope_value"].astype(str).str.strip().eq("").any()):
        raise ValueError("Targeted events require scope_value.")
    if work.scope.eq("sector").any() and bars.sector.isna().any():
        raise ValueError("Sector events require sector metadata on every bar.")
    if config.event_types:
        work = work.loc[work.event_type.isin(config.event_types)].copy()
    if work.empty:
        raise ValueError("No events match configured event_types.")
    work["event_id"] = work["event_id"].astype(str) if "event_id" in work else [f"event-{i}" for i in work.index]
    return work.sort_values(["timestamp", "event_id"]).reset_index(drop=True)


def active_events(events, ticker, sector, now, next_action, config):
    if events.empty:
        return []
    # An action's holdings persist until the next action. Close before a window
    # that falls entirely between two available prices, rather than pretend an
    # order could be filled inside that interval using daily OHLC.
    begin = events.timestamp - pd.to_timedelta(config.event_pre_hours, unit="h")
    end = events.timestamp + pd.to_timedelta(config.event_post_hours, unit="h")
    scope = events.scope.eq("global") | (events.scope.eq("ticker") & events.scope_value.eq(ticker)) | (events.scope.eq("sector") & events.scope_value.eq(sector))
    mask = scope & events.known_at.le(now) & end.ge(now) & begin.lt(next_action)
    return events.loc[mask, "event_id"].tolist()
