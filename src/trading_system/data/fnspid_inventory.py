"""Frozen current-price-universe FNSPID inventory, without scoring or returns.

Observation uses exactly the publication-delay window of the exploratory daily
export. Absence of an observation never establishes historical source coverage.
The news eligibility snapshot is restricted to the earliest TRAIN interval;
complete full-period prices are an explicit, retrospective data-quality filter.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .fnspid_import import import_fnspid
from .news_collection import _output_lock
from .news_pilot_scoring import _write_frame, _write_json, file_sha256
from .purged_cv import expanding_calendar_folds


SCHEMA_VERSION = 1
TIMING_FLAGS = ("version_timing_uncertain", "publication_timing_uncertain")
_OUTPUT_NAMES = ("inventory-report.json", "monthly-inventory.parquet",
                 "monthly-inventory.manifest.json", "tickers-selected.json")
_CV_SPEC = {"n_splits": 3, "initial_train_fraction": .5,
            "inner_val_fraction": .2, "final_test_fraction": .15,
            "gap_bars": 5, "embargo_bars": 0}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _date(value: Any, name: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a valid midnight session date.") from error
    if pd.isna(stamp) or stamp != stamp.normalize():
        raise ValueError(f"{name} must be a valid midnight session date.")
    if stamp.tzinfo is not None and stamp.utcoffset().total_seconds() != 0:
        raise ValueError(f"{name} must be timezone-naive or UTC.")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}.")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _price_calendar(path: Path, begin: pd.Timestamp, finish: pd.Timestamp):
    # Deliberately do not load labels, returns, positions or other model features.
    prices = pd.read_parquet(path, columns=["date", "ticker", "close"])
    if prices.columns.has_duplicates or prices.empty:
        raise ValueError("Price inventory requires a non-empty unique date/ticker/close schema.")
    names = prices.ticker
    if not names.map(lambda item: isinstance(item, str) and
                     re.fullmatch(r"[A-Z0-9][A-Z0-9.\-^=]*", item) is not None).all():
        raise ValueError("Price inventory requires normalized individual ticker names.")
    universe = sorted(names.unique())
    dates = pd.to_datetime(prices.date, utc=True, errors="raise")
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Price calendar requires non-missing midnight session dates.")
    prices["date"] = dates
    prices = prices.loc[dates.ge(begin) & dates.lt(finish)].copy()
    if prices.empty or prices.duplicated(["date", "ticker"]).any():
        raise ValueError("Price period is empty or has duplicate date/ticker keys.")
    close = pd.to_numeric(prices.close, errors="coerce")
    prices["price_available"] = np.isfinite(close.to_numpy(dtype=float)) & close.gt(0)
    calendar = pd.DatetimeIndex(prices.date.unique()).sort_values().as_unit("ns")
    return prices, universe, calendar


def _selection_bounds(calendar, begin, finish, selection_start, selection_end, warmup_sessions):
    folds, _ = expanding_calendar_folds(pd.DataFrame({"date": calendar}), **_CV_SPEC)
    validation = pd.Timestamp(folds[0]["split"].validation_start)
    before_validation = calendar[calendar < validation]
    # Keep the calendar gap out of the eligibility snapshot as well as training.
    end = before_validation[-_CV_SPEC["gap_bars"]] if _CV_SPEC["gap_bars"] else validation
    if selection_end is not None:
        end = _date(selection_end, "selection_end")
    if warmup_sessions >= len(calendar):
        raise ValueError("Price calendar is too short for the declared TRAIN warmup.")
    start = calendar[warmup_sessions] if selection_start is None else _date(selection_start, "selection_start")
    if start < begin or end > validation or end > finish or start >= end:
        raise ValueError("Selection must be a non-empty interval inside the earliest TRAIN, before validation.")
    if not ((calendar >= start) & (calendar < end)).any():
        raise ValueError("Selection interval must contain at least one observed price session.")
    return start, end, validation


def _gaps(calendar: pd.DatetimeIndex, present: np.ndarray, minimum: int) -> list[dict]:
    absent = ~present
    starts = np.flatnonzero(absent & np.r_[True, ~absent[:-1]])
    ends = np.flatnonzero(absent & np.r_[~absent[1:], True])
    return [{"start": calendar[first].isoformat(), "end": calendar[last].isoformat(),
             "session_days": int(last - first + 1),
             "calendar_days": int((calendar[last] - calendar[first]).days + 1)}
            for first, last in zip(starts, ends) if last - first + 1 >= minimum]


def _article_statistics(path: Path, universe: list[str], begin, finish):
    """Read imported columns in bounded batches; never parse the source again."""
    deduplicated, usable = Counter(), Counter()
    times = {ticker: [] for ticker in universe}
    ranges = {ticker: {"first": None, "last": None, "first_usable": None,
                      "last_usable": None} for ticker in universe}
    columns = ["ticker", "published_at", *TIMING_FLAGS]
    parquet = pq.ParquetFile(path)
    if not set(columns).issubset(parquet.schema_arrow.names):
        raise ValueError("Imported FNSPID inventory needs publication dates and both timing flags.")
    for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
        frame = batch.to_pandas()
        if not frame.ticker.isin(universe).all():
            raise ValueError("Imported articles contain tickers outside the frozen price universe.")
        published = pd.to_datetime(frame.published_at, utc=True, errors="raise")
        if published.isna().any():
            raise ValueError("Imported FNSPID publication dates cannot be missing.")
        clean = pd.Series(True, index=frame.index)
        for flag in TIMING_FLAGS:
            if not pd.api.types.is_bool_dtype(frame[flag]) or frame[flag].isna().any():
                raise ValueError(f"Imported {flag} must be a non-null boolean.")
            clean &= ~frame[flag]
        months = published.dt.strftime("%Y-%m")
        within = published.ge(begin) & published.lt(finish)
        deduplicated.update(zip(frame.loc[within, "ticker"], months[within]))
        usable.update(zip(frame.loc[within & clean, "ticker"], months[within & clean]))
        # Include imported pre-start trailing articles when computing decisions.
        for ticker, indices in frame.groupby("ticker", sort=False).groups.items():
            stamps = published.loc[indices]
            eligible = stamps.loc[clean.loc[indices]]
            if len(eligible):
                times[ticker].append(eligible.astype("int64").to_numpy())
            for name, values, operation in (("first", stamps, min), ("last", stamps, max),
                                           ("first_usable", eligible, min), ("last_usable", eligible, max)):
                if len(values):
                    candidate = values.min() if operation is min else values.max()
                    prior = ranges[ticker][name]
                    ranges[ticker][name] = candidate if prior is None else operation(prior, candidate)
    arrays = {ticker: np.sort(np.concatenate(parts)) if parts else np.array([], dtype="int64")
              for ticker, parts in times.items()}
    return deduplicated, usable, arrays, ranges


def _registered(destination: Path, identity: dict, *, resume: bool):
    manifest_path = destination / "inventory.manifest.json"
    paths = [destination / name for name in (*_OUTPUT_NAMES, "inventory.manifest.json")]
    if not any(path.exists() for path in paths):
        return None
    if not resume:
        raise FileExistsError("Inventory artifacts exist; use resume with unchanged inputs.")
    if not all(path.is_file() for path in paths):
        raise ValueError("Cannot resume incomplete inventory artifacts; restore them or choose a new destination.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("Cannot resume an unreadable inventory manifest.") from error
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "complete":
        raise ValueError("Cannot resume an incompatible or incomplete inventory manifest.")
    if manifest.get("identity") != identity:
        raise ValueError("Cannot resume: inventory settings or price input changed.")
    artifacts = manifest.get("artifacts", {})
    if set(artifacts) != set(_OUTPUT_NAMES):
        raise ValueError("Cannot resume a changed inventory artifact registry.")
    for name in _OUTPUT_NAMES:
        entry = artifacts[name]
        if entry.get("path") != name or entry.get("sha256") != file_sha256(destination / name):
            raise ValueError(f"Cannot resume corrupted inventory artifact: {name}.")
    return manifest


def _result(destination: Path, report: dict, selection: dict):
    manifest_path = destination / "inventory.manifest.json"
    return {"report": report, "selection": selection, "tickers": selection["tickers"],
            "selected_tickers": selection["tickers"],
            "report_path": destination / "inventory-report.json",
            "monthly_path": destination / "monthly-inventory.parquet",
            "selection_path": destination / "tickers-selected.json",
            "manifest_path": manifest_path, "manifest_sha256": file_sha256(manifest_path),
            "articles_path": destination / "import" / "articles.parquet",
            "import_report_path": destination / "import" / "import-report.json"}


def load_fnspid_inventory(destination: str | Path, *, verify_sources: bool = False) -> dict[str, Any]:
    """Verify the frozen inventory and reusable import before expanded preparation.

    All inventory/import artifacts are always checksum-validated. Optionally
    rehash original CSV/price/metadata sources, which can require substantial I/O.
    This loader never parses CSVs, modifies files or infers historical coverage.
    """
    output = Path(destination).resolve(strict=True)
    report_path = output / "inventory-report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        identity = report["identity"]
        selection = json.loads((output / "tickers-selected.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError("Cannot load an incomplete or unreadable FNSPID inventory.") from error
    manifest = _registered(output, identity, resume=True)
    if (manifest is None or report.get("protocol") != "fnspid-exploratory" or
            report.get("status") != "complete" or report.get("coverage_status") != "unknown" or
            report.get("historical_coverage_claim") is not False or
            selection != report.get("selection") or selection.get("historical_coverage_claim") is not False):
        raise ValueError("Inventory protocol or selection differs from its frozen report.")
    names = selection.get("tickers")
    universe = identity.get("settings", {}).get("tickers")
    if (not isinstance(names, list) or names != sorted(set(names)) or
            not isinstance(universe, list) or universe != selection.get("universe") or
            set(names) - set(universe)):
        raise ValueError("Inventory selection has invalid tickers or a changed universe.")
    expected = {"articles": output / "import" / "articles.parquet",
                "report": output / "import" / "import-report.json",
                "manifest": output / "import" / "articles.manifest.json"}
    records = manifest.get("import_artifacts")
    if not isinstance(records, dict) or set(records) != set(expected) or records != report.get("import_artifacts"):
        raise ValueError("Inventory import artifact registry differs from its report.")
    for name, path in expected.items():
        record = records[name]
        if (not isinstance(record, dict) or Path(record.get("path", "")).resolve() != path or
                not path.is_file() or record.get("sha256") != file_sha256(path)):
            raise ValueError(f"Inventory import artifact checksum mismatch: {name}.")
    if verify_sources:
        for record in [identity["price_input"], identity["dataset_info"], *report["source_files"]]:
            path = Path(record["path"])
            if not path.is_file() or file_sha256(path) != record.get("sha256"):
                raise ValueError(f"Inventory original source checksum mismatch: {path}.")
    return _result(output, report, selection)


def build_fnspid_inventory(
    price_path: str | Path, csv_paths: Sequence[str | Path], destination: str | Path, *,
    start: str = "2020-03-09", end: str = "2024-01-01", dataset_info: str | Path,
    delay_hours: float = 24, lookback_hours: float = 24,
    selection_start: str | None = None, selection_end: str | None = None,
    warmup_sessions: int = 80, min_observed_train_days: int = 60,
    min_train_observed_fraction: float = .1, long_gap_days: int = 20,
    timezone_assumption: str = "UTC", chunksize: int = 100_000,
    resume: bool = False, dry_run: bool = False,
) -> dict[str, Any]:
    """Import all current price tickers once, audit and freeze TRAIN eligibility.

    ``end`` and ``selection_end`` are exclusive. The default eligibility interval
    skips 80 price sessions (20 label/feature + 60 context sessions) and ends
    before the first inner-validation boundary, including a five-session gap.
    Explicit bounds can match the runner's prepared TRAIN dates more closely.
    Full-period price completeness is intentionally retrospective and recorded;
    news observations, thresholds and eligibility never use returns or scores.

    ``dry_run`` reads the price calendar and source metadata but never scans or
    hashes CSV contents, imports, creates outputs, or scores articles. It reports
    a plan rather than inventing an eligible universe from unread news sources.
    """
    begin, finish = _date(start, "start"), _date(end, "end")
    if begin >= finish:
        raise ValueError("start must precede the exclusive end.")
    delay = _finite(delay_hours, "delay_hours")
    lookback = _finite(lookback_hours, "lookback_hours", positive=True)
    warmup = _integer(warmup_sessions, "warmup_sessions")
    minimum = _integer(min_observed_train_days, "min_observed_train_days", minimum=1)
    gap_minimum = _integer(long_gap_days, "long_gap_days", minimum=1)
    _integer(chunksize, "chunksize", minimum=1)
    fraction = _finite(min_train_observed_fraction, "min_train_observed_fraction")
    if fraction > 1:
        raise ValueError("min_train_observed_fraction must be in [0, 1].")
    if isinstance(csv_paths, (str, Path)) or not csv_paths:
        raise ValueError("csv_paths must contain individual CSV files.")
    paths = sorted({Path(path).resolve() for path in csv_paths}, key=str)
    if len(paths) != len(csv_paths) or any(not path.is_file() for path in paths):
        raise ValueError("CSV inputs must be distinct existing files.")
    price_source = Path(price_path).resolve(strict=True)
    info_path = Path(dataset_info).resolve(strict=True)
    info = json.loads(info_path.read_text(encoding="utf-8-sig"))
    revision = info.get("sha") if isinstance(info, dict) else None
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("dataset_info requires an immutable lowercase 40-hex sha revision.")
    output = Path(destination).resolve()
    if output.exists() and not output.is_dir():
        raise ValueError("Inventory destination must be a directory.")
    prices, universe, calendar = _price_calendar(price_source, begin, finish)
    train_start, train_end, validation = _selection_bounds(
        calendar, begin, finish, selection_start, selection_end, warmup)
    news_begin = begin - pd.Timedelta(hours=delay + lookback)
    settings = {"start": begin.isoformat(), "end_exclusive": finish.isoformat(),
                "news_start": news_begin.isoformat(), "tickers": universe,
                "delay_hours": delay, "lookback_hours": lookback,
                "include_at_cutoff": False, "include_at_window_start": True,
                "decision_time": "session midnight UTC", "timezone_assumption": timezone_assumption,
                "chunksize": chunksize, "long_gap_days": gap_minimum,
                "selection": {"start": train_start.isoformat(), "end_exclusive": train_end.isoformat(),
                              "first_inner_validation_start": validation.isoformat(), "cv_spec": _CV_SPEC,
                              "warmup_sessions": warmup, "min_observed_train_days": minimum,
                              "min_train_observed_fraction": fraction,
                              "require_complete_price_calendar": True,
                              "uses_performance": False,
                              "news_eligibility_basis": "earliest TRAIN observations only",
                              "price_completeness_basis": "retrospective full study calendar; survivorship remains unaudited"}}
    identity = {"schema_version": SCHEMA_VERSION, "protocol": "fnspid-exploratory",
                "settings": settings, "price_input": {"path": str(price_source), "sha256": file_sha256(price_source)},
                "dataset_info": {"path": str(info_path), "sha256": file_sha256(info_path)},
                "source_revision": revision, "csv_paths": [str(path) for path in paths]}
    if dry_run:
        registered = _registered(output, identity, resume=True) if (output / "inventory.manifest.json").exists() else None
        return {"status": "validated" if registered else "planned", "dry_run": True,
                "settings": settings, "universe_count": len(universe), "tickers": universe,
                "selected_tickers": None, "manifest_path": output / "inventory.manifest.json",
                "csv_inputs": [{"path": str(path), "size_bytes": path.stat().st_size} for path in paths],
                "source_contents_verified": False, "historical_coverage_claim": False}
    output.mkdir(parents=True, exist_ok=True)
    with _output_lock(output):
        registered = _registered(output, identity, resume=resume)
        import_root = output / "import"
        has_import = (import_root / "import-report.json").exists()
        print(f"FNSPID inventory: {len(universe)} price tickers; {'validate frozen import' if has_import else 'strict CSV import'}", flush=True)
        imported = import_fnspid(paths, import_root, tickers=universe,
                                  start=news_begin.isoformat(), end=finish.isoformat(),
                                  dataset_info=info_path, timezone_assumption=timezone_assumption,
                                  chunksize=chunksize, resume=resume and has_import)
        import_report = imported["report"]
        import_artifacts = {"articles": {"path": str(imported["articles_path"]), "sha256": file_sha256(imported["articles_path"])},
                            "report": {"path": str(imported["report_path"]), "sha256": file_sha256(imported["report_path"])},
                            "manifest": {"path": str(import_root / "articles.manifest.json"),
                                         "sha256": file_sha256(import_root / "articles.manifest.json")}}
        if registered is not None:
            if registered.get("import_artifacts") != import_artifacts:
                raise ValueError("Cannot resume: imported FNSPID artifacts changed.")
            report = json.loads((output / "inventory-report.json").read_text(encoding="utf-8"))
            selection = json.loads((output / "tickers-selected.json").read_text(encoding="utf-8"))
            return _result(output, report, selection)
        print("FNSPID inventory: imported Parquet -> monthly observations and frozen TRAIN eligibility", flush=True)
        dedup, usable, published, ranges = _article_statistics(imported["articles_path"], universe, news_begin, finish)
        months = [str(month) for month in pd.period_range(begin.tz_localize(None).to_period("M"),
                                                         (finish - pd.Timedelta(nanoseconds=1)).tz_localize(None).to_period("M"), freq="M")]
        calendar_months = calendar.strftime("%Y-%m")
        train_mask = (calendar >= train_start) & (calendar < train_end)
        train_days = int(train_mask.sum())
        delta = pd.Timedelta(hours=delay).value
        window = pd.Timedelta(hours=lookback).value
        rows, audit, eligible, rejected = [], {}, [], {}
        for ticker in universe:
            ticker_prices = prices.loc[prices.ticker.eq(ticker)]
            valid_dates = pd.DatetimeIndex(ticker_prices.loc[ticker_prices.price_available, "date"])
            price_mask = calendar.isin(valid_dates)
            available = published[ticker] + delta
            observed = ((np.searchsorted(available, calendar.asi8, side="left") -
                         np.searchsorted(available, calendar.asi8 - window, side="left")) > 0) & price_mask
            observed_train = int((observed & train_mask).sum())
            observed_fraction = observed_train / train_days
            complete = bool(price_mask.all())
            reasons = []
            if not complete:
                reasons.append("incomplete_price_calendar")
            if observed_train < minimum:
                reasons.append("insufficient_observed_train_days")
            if observed_fraction < fraction:
                reasons.append("insufficient_train_observed_fraction")
            if reasons:
                rejected[ticker] = reasons
            else:
                eligible.append(ticker)
            raw_months = import_report["coverage"][ticker].get("raw_counts_by_month")
            if raw_months is None:
                raise ValueError("FNSPID import report lacks raw_counts_by_month; exact pre-dedup inventory requires a fresh compatible import.")
            for month in months:
                mask = calendar_months == month
                expected = int(mask.sum())
                observed_days = int((observed & mask).sum())
                available_days = int((price_mask & mask).sum())
                rows.append({"ticker": ticker, "month": month,
                             "raw_news_rows": int(raw_months.get(month, 0)),
                             "deduplicated_news_associations": dedup[ticker, month],
                             "usable_news_associations": usable[ticker, month],
                             "timing_rejected_associations": dedup[ticker, month] - usable[ticker, month],
                             "expected_price_days": expected, "available_price_days": available_days,
                             "missing_price_days": expected - available_days,
                             "observed_decision_days": observed_days,
                             "unobserved_decision_days": available_days - observed_days,
                             "observed_decision_fraction": observed_days / available_days if available_days else 0.,
                             "train_decision_days": int((train_mask & mask).sum()),
                             "observed_train_decision_days": int((train_mask & observed & mask).sum()),
                             "coverage_status": "unknown", "historical_coverage_proven": False})
            audit[ticker] = {"selected": not reasons, "rejection_reasons": reasons,
                             "coverage_status": "unknown", "historical_coverage_proven": False,
                             "raw_news_rows": sum(int(raw_months.get(month, 0)) for month in months),
                             "deduplicated_news_associations": sum(dedup[ticker, month] for month in months),
                             "usable_news_associations": sum(usable[ticker, month] for month in months),
                             "first_news_at": ranges[ticker]["first"].isoformat() if ranges[ticker]["first"] is not None else None,
                             "last_news_at": ranges[ticker]["last"].isoformat() if ranges[ticker]["last"] is not None else None,
                             "first_usable_news_at": ranges[ticker]["first_usable"].isoformat() if ranges[ticker]["first_usable"] is not None else None,
                             "last_usable_news_at": ranges[ticker]["last_usable"].isoformat() if ranges[ticker]["last_usable"] is not None else None,
                             "price_days": int(price_mask.sum()), "expected_price_days": len(calendar),
                             "complete_price_calendar": complete,
                             "invalid_close_rows": int((~ticker_prices.price_available).sum()),
                             "observed_decision_days": int(observed.sum()),
                             "observed_decision_fraction": float(observed.sum() / price_mask.sum()) if price_mask.any() else 0.,
                             "train_decision_days": train_days, "observed_train_days": observed_train,
                             "train_observed_fraction": observed_fraction,
                             "long_unobserved_gaps": _gaps(calendar, observed, gap_minimum),
                             "long_price_gaps": _gaps(calendar, price_mask, gap_minimum),
                             "months_without_usable_news": [month for month in months if not usable[ticker, month]]}
        monthly = pd.DataFrame.from_records(rows)
        snapshot = {"rules": settings["selection"], "price_calendar_sha256": _digest(calendar.astype(str).tolist()),
                    "import_artifacts": import_artifacts}
        selection = {"schema_version": SCHEMA_VERSION, "tickers": eligible,
                     "universe": universe, "rejected": rejected, "rules": settings["selection"],
                     "snapshot_sha256": _digest(snapshot), "historical_coverage_claim": False,
                     "selection_basis": "price availability and earliest TRAIN news observations; no performance ranking"}
        report = {"schema_version": SCHEMA_VERSION, "status": "complete", "identity": identity,
                  "protocol": "fnspid-exploratory", "point_in_time": False,
                  "coverage_status": "unknown", "historical_coverage_claim": False,
                  "universe_count": len(universe), "selected_count": len(eligible),
                  "rejected_count": len(rejected), "tickers": audit,
                  "study_calendar": {"first": calendar[0].isoformat(), "last": calendar[-1].isoformat(), "sessions": len(calendar)},
                  "selection": selection, "import_artifacts": import_artifacts,
                  "source_files": import_report["input_files"],
                  "raw_news_count_basis": "accepted source CSV rows before article/ticker deduplication, by declared publication month; includes imported pre-study trailing window",
                  "monthly_news_count_period": {"start": news_begin.isoformat(), "end_exclusive": finish.isoformat()},
                  "limitations": ["Historical source coverage remains unknown, including all empty windows.",
                                  "Publication plus delay is assumed availability, not first-seen evidence.",
                                  "Only imported titles without publication/version timing flags are usable.",
                                  "Full-period price completeness uses future data availability; current-universe survivorship is not audited.",
                                  "The session calendar is the union observed in the frozen price file, not an independently audited exchange calendar."]}
        _write_frame(output / "monthly-inventory.parquet", monthly,
                     {"protocol": "fnspid-exploratory", "coverage_status": "unknown", "identity": identity})
        _write_json(output / "tickers-selected.json", selection)
        _write_json(output / "inventory-report.json", report)
        _write_json(output / "inventory.manifest.json", {
            "schema_version": SCHEMA_VERSION, "status": "complete", "identity": identity,
            "import_artifacts": import_artifacts,
            "artifacts": {name: {"path": name, "sha256": file_sha256(output / name)} for name in _OUTPUT_NAMES}})
        print(f"FNSPID inventory ready: {len(eligible)}/{len(universe)} tickers pass frozen TRAIN rules", flush=True)
        return _result(output, report, selection)


__all__ = ["build_fnspid_inventory", "load_fnspid_inventory"]
