"""FNSPID features under an explicit publication-delay assumption.

These archives provide publication dates, not historical first-seen or source
coverage evidence. The distinct schema and loader keep this exploratory route
separate from point-in-time exports, even when their numerical features agree.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from trading_system.artifacts.experiment import hash_dataframe
from trading_system.artifacts.multimodal_study import atomic_write_json
from .news_sentiment import (
    SENTIMENT_COLUMNS, SENTIMENT_STATISTICS, SentimentExport,
    _aware_utc, _sha256, build_news_decision_points,
)


FNSPID_PROTOCOL = "fnspid-exploratory"
FNSPID_SCHEMA = "fnspid-exploratory-1"
AVAILABILITY_ASSUMPTION = "publication_plus_delay_assumption"
_RAW_COLUMNS = (
    "ticker", "decision_at", "news_count", "coverage_status", "observation_status",
    "availability_kind", "last_contributing_assumed_available_at", *SENTIMENT_STATISTICS,
)
EXPLORATORY_LIMITATIONS = [
    "Publication plus a fixed delay is an assumption, not historical availability evidence.",
    "Source coverage is unknown; empty windows are unobserved and unavailable.",
    "Archived text may include revisions, backfills, date errors or timestamp precision loss.",
    "A modern FinBERT encoder can postdate the evaluated period and contain future knowledge.",
    "Ticker associations and the selected current universe can introduce attribution and survivorship biases.",
    "This comparison is exploratory and cannot establish a historical deployable trading result.",
]


def _parameters(delay_hours, lookback_hours, short_lookback_hours, half_life_hours):
    result = {}
    for name, value in (("delay_hours", delay_hours), ("lookback_hours", lookback_hours),
                        ("short_lookback_hours", short_lookback_hours), ("half_life_hours", half_life_hours)):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite number.")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite number.") from exc
        if not math.isfinite(value) or value < 0 or (name != "delay_hours" and value == 0):
            raise ValueError(f"{name} must be {'non-negative' if name == 'delay_hours' else 'positive'} and finite.")
        result[name] = value
    if result["short_lookback_hours"] > result["lookback_hours"]:
        raise ValueError("short_lookback_hours cannot exceed lookback_hours.")
    return result


def _scored_articles(path: Path, checkpoint: str) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    required = {"news_id", "ticker", "published_at", "p_positive", "p_neutral", "p_negative",
                "sentiment_score", "confidence", "checkpoint"}
    if raw.columns.has_duplicates or not required.issubset(raw):
        raise ValueError(f"Scored FNSPID input is missing unique required columns: {sorted(required - set(raw))}")
    if not raw["ticker"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Scored FNSPID input contains invalid tickers.")
    if not raw["news_id"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Scored FNSPID input contains invalid article IDs.")
    if raw.duplicated(["news_id", "ticker"]).any():
        raise ValueError("Scored FNSPID input has duplicate article/ticker associations.")
    if not raw["checkpoint"].eq(checkpoint).all():
        raise ValueError("Scored FNSPID checkpoints disagree with the frozen checkpoint.")
    for flag in ("version_timing_uncertain", "publication_timing_uncertain"):
        if flag in raw and not raw[flag].eq(False).all():
            raise ValueError(f"Scored FNSPID input retains uncertain article timing: {flag}.")
    result = raw.loc[:, list(required)].copy()
    result["published_at"] = _aware_utc(raw["published_at"], name="published_at", nullable=False)
    numeric = ("p_positive", "p_neutral", "p_negative", "sentiment_score", "confidence")
    for name in numeric:
        result[name] = pd.to_numeric(raw[name], errors="coerce")
    values = result.loc[:, numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Scored FNSPID scores must be finite.")
    probabilities = result.loc[:, ["p_positive", "p_neutral", "p_negative"]].to_numpy(dtype=float)
    if (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(
        probabilities.sum(axis=1), 1, atol=1e-5, rtol=0
    ):
        raise ValueError("Scored FNSPID probabilities must lie in [0, 1] and sum to one.")
    if not np.allclose(result.sentiment_score, result.p_positive - result.p_negative, atol=1e-5, rtol=0):
        raise ValueError("Scored FNSPID sentiment_score must equal p_positive - p_negative.")
    if not np.allclose(result.confidence, probabilities.max(axis=1), atol=1e-5, rtol=0):
        raise ValueError("Scored FNSPID confidence must equal the maximum class probability.")
    return result


def _aggregate(scored: pd.DataFrame, decisions: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    scored = scored.copy()
    scored["assumed_available_at"] = scored.published_at + pd.Timedelta(hours=parameters["delay_hours"])
    if scored.assumed_available_at.isna().any():
        raise ValueError("Publication-delay assumption overflowed timestamp bounds.")
    rows = []
    lookback = pd.Timedelta(hours=parameters["lookback_hours"]).value
    short = pd.Timedelta(hours=parameters["short_lookback_hours"]).value
    for ticker, points in decisions.groupby("ticker", sort=False):
        articles = scored.loc[scored.ticker.eq(ticker)].sort_values("assumed_available_at", kind="stable")
        times = articles.assumed_available_at.astype("int64").to_numpy()
        signals = articles.sentiment_score.to_numpy(dtype=float)
        probabilities = articles[["p_positive", "p_neutral", "p_negative"]].to_numpy(dtype=float)
        confidence = articles.confidence.to_numpy(dtype=float)
        for decision in points.decision_at:
            cutoff = decision.value
            begin = np.searchsorted(times, cutoff - lookback, side="left")
            end = np.searchsorted(times, cutoff, side="left")
            count = int(end - begin)
            row = {"ticker": ticker, "decision_at": decision, "news_count": count,
                   "coverage_status": "unknown", "observation_status": "observed" if count else "unobserved",
                   "availability_kind": AVAILABILITY_ASSUMPTION,
                   "last_contributing_assumed_available_at": pd.Timestamp(times[end - 1], tz="UTC") if count else pd.NaT,
                   **dict.fromkeys(SENTIMENT_STATISTICS, np.nan)}
            if count:
                signal = signals[begin:end]
                probs = probabilities[begin:end]
                # Match the upstream classifier's negative/neutral/positive
                # class order when maximum probabilities tie.
                predicted = probs[:, ::-1].argmax(axis=1)
                age_hours = (cutoff - times[begin:end]) / pd.Timedelta(hours=1).value
                # Normalize by the youngest article to avoid all-zero weights
                # for long windows and small half-lives; the ratio is unchanged.
                weights = confidence[begin:end] * np.exp2(
                    -(age_hours - age_hours.min()) / parameters["half_life_hours"]
                )
                short_begin = max(begin, np.searchsorted(times, cutoff - short, side="left"))
                row.update(
                    sentiment_mean=float(signal.mean()), sentiment_std=float(signal.std(ddof=0)),
                    p_positive_mean=float(probs[:, 0].mean()), p_neutral_mean=float(probs[:, 1].mean()),
                    p_negative_mean=float(probs[:, 2].mean()),
                    positive_share=float((predicted == 2).mean()),
                    negative_share=float((predicted == 0).mean()),
                    confidence_mean=float(confidence[begin:end].mean()),
                    sentiment_ewm=float(np.average(signal, weights=weights)),
                    sentiment_momentum=float(signals[short_begin:end].mean() - signal.mean()) if short_begin < end else np.nan,
                    hours_since_last_news=float(age_hours[-1]),
                )
            rows.append(row)
    raw = pd.DataFrame(rows, columns=_RAW_COLUMNS)
    raw["last_contributing_assumed_available_at"] = pd.to_datetime(
        raw.last_contributing_assumed_available_at, utc=True
    )
    return raw.sort_values(["decision_at", "ticker"], kind="stable").reset_index(drop=True)


def export_fnspid_sentiment(
    scored_path: str | Path, market_frame: pd.DataFrame, destination: str | Path, *,
    checkpoint: str, input_identifiers: dict, delay_hours=24, lookback_hours=24,
    short_lookback_hours=6, half_life_hours=6,
) -> dict[str, Any]:
    """Write exploratory daily Parquet and its immutable-assumption manifest."""
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("A frozen scoring checkpoint identifier is required.")
    if not isinstance(input_identifiers, dict):
        raise TypeError("input_identifiers must be an object.")
    parameters = _parameters(delay_hours, lookback_hours, short_lookback_hours, half_life_hours)
    source = Path(scored_path).expanduser().resolve(strict=True)
    target = Path(destination).expanduser().resolve()
    if target.suffix.lower() != ".parquet":
        raise ValueError("FNSPID export must be a Parquet file.")
    manifest_path = target.with_suffix(".manifest.json")
    if target.exists() or manifest_path.exists():
        raise FileExistsError(f"FNSPID export or manifest already exists: {target}")
    scored = _scored_articles(source, checkpoint)
    if "ticker" not in market_frame:
        raise ValueError("Market data is missing ticker.")
    tickers = tuple(sorted(market_frame.ticker.dropna().unique()))
    decisions = build_news_decision_points(market_frame, tickers=tickers)
    raw = _aggregate(scored, decisions, parameters)
    identifiers = {"domain": "ticker", **input_identifiers}
    manifest = {
        "schema_version": FNSPID_SCHEMA, "protocol": FNSPID_PROTOCOL,
        "point_in_time": False, "historical_coverage_claim": False,
        "availability_kind": AVAILABILITY_ASSUMPTION, "checkpoint": checkpoint,
        "input_identifiers": identifiers,
        "aggregation": {**parameters, "include_at_cutoff": False, "include_at_window_start": True,
                        "time_basis": "published_at + delay_hours", "timezone": "UTC",
                        "decision_time": "session midnight UTC", "std_ddof": 0,
                        "decay_weight": "confidence * 2**(-article_age_hours / half_life_hours)",
                        "class_share": "argmax(p_negative, p_neutral, p_positive); ties prefer negative then neutral",
                        "momentum": "short-window mean minus full-window mean; missing without short-window news",
                        "recency": "hours since last contributing assumed-available article"},
        "inputs": {"scored": {"path": str(source), "sha256": _sha256(source), "rows": len(scored)},
                   "market": {"sha256": hash_dataframe(market_frame), "rows": len(market_frame)},
                   "decisions": {"sha256": hash_dataframe(decisions), "rows": len(decisions)}},
        "feature_columns": list(raw.columns), "model_columns": list(SENTIMENT_COLUMNS),
        "feature_rows": len(raw), "observed_rows": int(raw.observation_status.eq("observed").sum()),
        "unobserved_rows": int(raw.observation_status.eq("unobserved").sum()),
        "coverage_status": "unknown", "tickers": list(tickers),
        "historical_version_integrity_verified": False, "encoder_training_cutoff_verified": False,
        "decision_range": {"start": decisions.decision_at.min().isoformat(), "end": decisions.decision_at.max().isoformat()},
        "limitations": EXPLORATORY_LIMITATIONS,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".parquet", dir=target.parent)
    os.close(handle)
    try:
        raw.to_parquet(temporary, index=False)
        manifest["parquet_sha256"] = _sha256(Path(temporary))
        # A completed manifest is required by the loader. Write it after the
        # data so an interrupted export cannot be mistaken for a ready input.
        os.replace(temporary, target)
        atomic_write_json(manifest_path, manifest)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return manifest


def load_fnspid_sentiment_export(path: str | Path) -> SentimentExport:
    """Verify an exploratory export without asserting historical coverage."""
    target = Path(path).expanduser().resolve(strict=True)
    if target.suffix.lower() != ".parquet":
        raise ValueError("FNSPID export must be a Parquet file.")
    with target.with_suffix(".manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != FNSPID_SCHEMA
            or manifest.get("protocol") != FNSPID_PROTOCOL
            or manifest.get("point_in_time") is not False
            or manifest.get("historical_coverage_claim") is not False
            or manifest.get("coverage_status") != "unknown"
            or manifest.get("availability_kind") != AVAILABILITY_ASSUMPTION):
        raise ValueError("Unsupported or misleading FNSPID exploratory manifest.")
    aggregation = manifest.get("aggregation")
    if not isinstance(aggregation, dict) or aggregation.get("include_at_cutoff") is not False:
        raise ValueError("FNSPID export must use include_at_cutoff=False.")
    parameters = _parameters(*(aggregation.get(name) for name in (
        "delay_hours", "lookback_hours", "short_lookback_hours", "half_life_hours"
    )))
    if (aggregation.get("include_at_window_start") is not True
            or aggregation.get("time_basis") != "published_at + delay_hours"
            or aggregation.get("timezone") != "UTC"):
        raise ValueError("FNSPID aggregation does not declare the publication-delay window assumption.")
    if not isinstance(manifest.get("checkpoint"), str) or not manifest["checkpoint"].strip():
        raise ValueError("FNSPID export must identify its frozen checkpoint.")
    if manifest.get("parquet_sha256") != _sha256(target):
        raise ValueError("FNSPID export checksum does not match its manifest.")
    raw = pd.read_parquet(target)
    if (raw.columns.has_duplicates or list(raw.columns) != list(_RAW_COLUMNS)
            or list(raw.columns) != manifest.get("feature_columns")
            or list(SENTIMENT_COLUMNS) != manifest.get("model_columns")):
        raise ValueError("FNSPID feature columns do not match the declared schema.")
    if raw.empty or len(raw) != manifest.get("feature_rows"):
        raise ValueError("FNSPID export row count does not match its manifest.")
    if not raw.ticker.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("FNSPID export has invalid tickers.")
    decisions = _aware_utc(raw.decision_at, name="decision_at", nullable=False)
    if not decisions.eq(decisions.dt.normalize()).all():
        raise ValueError("FNSPID decisions must be session midnight UTC.")
    if pd.DataFrame({"ticker": raw.ticker, "date": decisions}).duplicated().any():
        raise ValueError("FNSPID export has duplicate decision keys.")
    contributed = _aware_utc(raw.last_contributing_assumed_available_at,
                             name="last_contributing_assumed_available_at", nullable=True)
    if (contributed.notna() & contributed.ge(decisions)).any():
        raise ValueError("FNSPID article assumption reaches or follows decision midnight.")
    lower = decisions - pd.Timedelta(hours=parameters["lookback_hours"])
    if (contributed.notna() & contributed.lt(lower)).any():
        raise ValueError("FNSPID contributing article falls outside the rolling window.")
    counts = pd.to_numeric(raw.news_count, errors="coerce")
    if counts.isna().any() or not np.isfinite(counts.to_numpy(dtype=float)).all() or (
        counts.lt(0) | counts.ne(np.floor(counts))
    ).any():
        raise ValueError("FNSPID news_count must contain non-negative integers.")
    observed = counts.gt(0)
    if (not raw.coverage_status.eq("unknown").all()
            or not raw.availability_kind.eq(AVAILABILITY_ASSUMPTION).all()
            or not raw.observation_status.isin(("observed", "unobserved")).all()
            or not np.array_equal(observed.to_numpy(), raw.observation_status.eq("observed").to_numpy())
            or not np.array_equal(observed.to_numpy(), contributed.notna().to_numpy())):
        raise ValueError("FNSPID observation status and timestamps disagree; historical coverage must remain unknown.")
    if (int(observed.sum()) != manifest.get("observed_rows")
            or int((~observed).sum()) != manifest.get("unobserved_rows")):
        raise ValueError("FNSPID observed row counts do not match its manifest.")
    frame = pd.DataFrame({
        "date": decisions, "ticker": raw.ticker, "source_available": observed,
        "available_at": contributed, "coverage_status": raw.coverage_status,
        "observation_status": raw.observation_status, "availability_kind": raw.availability_kind,
        "news_count": counts.to_numpy(dtype=np.float32),
    })
    for name in SENTIMENT_STATISTICS:
        values = pd.to_numeric(raw[name], errors="coerce")
        if (raw[name].notna() & values.isna()).any() or np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError(f"FNSPID export has invalid {name} values.")
        if ((~observed & values.notna()).any()
                or (name != "sentiment_momentum" and (observed & values.isna()).any())):
            raise ValueError(f"FNSPID {name} disagrees with article observation status.")
        if name == "hours_since_last_news" and not np.allclose(
            values[observed], (decisions[observed] - contributed[observed]).dt.total_seconds() / 3600,
            atol=1e-6, rtol=0,
        ):
            raise ValueError("FNSPID recency disagrees with its contributing timestamp.")
        frame[name] = values.fillna(0).to_numpy(dtype=np.float32)
        frame[f"{name}_present"] = values.notna().to_numpy(dtype=np.float32)
    return SentimentExport(frame=frame, columns=SENTIMENT_COLUMNS, manifest=manifest, protocol=FNSPID_PROTOCOL)


__all__ = ["FNSPID_PROTOCOL", "FNSPID_SCHEMA", "AVAILABILITY_ASSUMPTION", "EXPLORATORY_LIMITATIONS",
           "export_fnspid_sentiment", "load_fnspid_sentiment_export"]
