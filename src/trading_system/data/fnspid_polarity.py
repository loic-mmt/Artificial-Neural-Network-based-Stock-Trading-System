"""Fold-local article polarity controls for exploratory FNSPID exports.

An article belongs to the partition of its first contributing decision, not its
publication date. Permutations are independent within ticker and TRAIN/INNER/
OUTER partitions. The coherent classifier packet includes confidence, which is
also used by the confidence-weighted aggregate. Activity, timing, recency and
all presence masks retain their original values.

Neutralization is declared here and applied by the runner *after* fitting and
applying TRAIN standardization: zero the nine polarity value channels while
retaining confidence, recency, activity and their original presence channels.
It does not fabricate neutral articles or change the audited daily export.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from trading_system.artifacts.experiment import hash_dataframe
from trading_system.artifacts.multimodal_study import stable_digest
from .fnspid_export import AVAILABILITY_ASSUMPTION, FNSPID_PROTOCOL, _aggregate, _parameters
from .news_sentiment import SENTIMENT_STATISTICS, SentimentExport, _aware_utc


POLARITY_CONTROLS = ("original", "shuffled", "neutralized")
POLARITY_STATISTICS = tuple(
    name for name in SENTIMENT_STATISTICS
    if name not in ("confidence_mean", "hours_since_last_news")
)
ARTICLE_SCORE_COLUMNS = (
    "p_positive", "p_neutral", "p_negative", "sentiment_score", "confidence",
)


def _boundary(value: Any, name: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
        if pd.isna(result):
            raise ValueError("missing timestamp")
        result = result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid fold {name} boundary.") from error
    if result != result.normalize():
        raise ValueError(f"Fold {name} must be midnight UTC.")
    return result


def _boundaries(fold: dict) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    split = fold["split"]
    get = split.__getitem__ if isinstance(split, Mapping) else lambda name: getattr(split, name)
    validation = _boundary(get("validation_start"), "validation_start")
    test = _boundary(get("test_start"), "test_start")
    end = _boundary(fold["end"], "end")
    if not validation < test <= end:
        raise ValueError("Fold boundaries must satisfy validation_start < test_start <= end.")
    return validation, test, end


def _digest(frame: pd.DataFrame) -> str:
    return hash_dataframe(frame.reset_index(drop=True)) if len(frame) else stable_digest([])


def _articles_for_decisions(scored, decisions, parameters, validation, test):
    """Select actual contributors; a weekend article outside the window stays out."""
    required = {"news_id", "ticker", "published_at", *ARTICLE_SCORE_COLUMNS}
    if not isinstance(scored, pd.DataFrame) or scored.columns.has_duplicates:
        raise ValueError("Shuffled control requires scored articles with unique columns.")
    if not required.issubset(scored):
        raise ValueError(f"Scored control input is missing columns: {sorted(required - set(scored))}")
    if scored.duplicated(["news_id", "ticker"]).any():
        raise ValueError("Scored control input has duplicate article/ticker associations.")
    articles = scored.copy(deep=True)
    articles["published_at"] = _aware_utc(articles.published_at, name="published_at", nullable=False)
    articles["assumed_available_at"] = articles.published_at + pd.Timedelta(hours=parameters["delay_hours"])
    if articles.assumed_available_at.isna().any():
        raise ValueError("Publication-delay assumption overflowed timestamp bounds.")
    selected = []
    lookback = pd.Timedelta(hours=parameters["lookback_hours"]).value
    for ticker, calendar in decisions.groupby("ticker", sort=True):
        group = articles.loc[articles.ticker.eq(ticker)].sort_values(
            ["assumed_available_at", "news_id"], kind="stable"
        ).reset_index(drop=True)
        if group.empty:
            continue
        points = calendar.decision_at.astype("int64").to_numpy()
        times = group.assumed_available_at.astype("int64").to_numpy()
        # side=right implements the export's strict availability < decision.
        first = np.searchsorted(points, times, side="right")
        exists = first < len(points)
        bounded = np.minimum(first, len(points) - 1)
        contributing = exists & (times >= points[bounded] - lookback)
        group = group.loc[contributing].copy()
        group["first_contributing_decision"] = pd.to_datetime(points[first[contributing]], utc=True)
        group["partition"] = np.where(
            group.first_contributing_decision < validation, "train",
            np.where(group.first_contributing_decision < test, "inner", "outer"),
        )
        selected.append(group)
    if not selected:
        result = articles.iloc[:0].copy()
        result["first_contributing_decision"] = pd.Series(dtype="datetime64[ns, UTC]")
        result["partition"] = pd.Series(dtype=str)
        return result
    return pd.concat(selected, ignore_index=True)


def _permute_articles(articles, *, shuffle_seed, fold_number, partitions):
    result = articles.copy(deep=True)
    inventories = {}
    for partition in partitions:
        mappings = []
        inventory = {"article_rows": 0, "ticker_groups": 0, "singleton_groups": 0,
                     "changed_assignments": 0, "changed_scores": 0}
        subset = result.loc[result.partition.eq(partition)]
        for ticker, group in subset.groupby("ticker", sort=True):
            index = group.index.to_numpy()
            packet = group.loc[:, ARTICLE_SCORE_COLUMNS].to_numpy(copy=True)
            group_seed = int(stable_digest({"seed": shuffle_seed, "fold": fold_number,
                                           "ticker": ticker, "partition": partition})[:16], 16)
            permutation = np.random.default_rng(group_seed).permutation(len(group))
            result.loc[index, list(ARTICLE_SCORE_COLUMNS)] = packet[permutation]
            inventory["article_rows"] += len(group)
            inventory["ticker_groups"] += 1
            inventory["singleton_groups"] += int(len(group) == 1)
            inventory["changed_assignments"] += int((permutation != np.arange(len(group))).sum())
            inventory["changed_scores"] += int(np.any(packet[permutation] != packet, axis=1).sum())
            mapping = group.loc[:, ["ticker", "news_id", "assumed_available_at",
                                    "first_contributing_decision"]].copy()
            mapping["donor_news_id"] = group.news_id.to_numpy()[permutation]
            mappings.append(mapping)
        mapping = pd.concat(mappings, ignore_index=True) if mappings else pd.DataFrame()
        inventory["mapping_sha256"] = _digest(mapping)
        inventory["input_scores_sha256"] = _digest(subset.loc[:, ["ticker", "news_id", *ARTICLE_SCORE_COLUMNS]])
        inventory["output_scores_sha256"] = _digest(result.loc[result.partition.eq(partition),
                                                              ["ticker", "news_id", *ARTICLE_SCORE_COLUMNS]])
        inventories[partition] = inventory
    return result, inventories


def prepare_fold_news_control(
    exported: SentimentExport, scored: pd.DataFrame | None, fold: dict, *,
    mode: str = "original", shuffle_seed: int = 314159, include_outer: bool = False,
) -> SentimentExport:
    """Return an in-memory fold control, never rewriting an export or manifest.

    ``scored`` must be the already-verified ``_scored_articles`` input matching
    ``exported``. It is required only for ``shuffled``. With ``include_outer=False``
    only decisions strictly before ``test_start`` are selected and aggregated;
    with ``True`` the upper bound is ``fold['end']`` inclusive. Decisions after
    that bound (including the final closed holdout) are never aggregated.

    Shuffle windows must be <=24 hours because daily midnight decisions then
    give disjoint article windows, preventing one article from crossing a split.
    The ordinary random permutation can leave fixed points; its audit reports
    changed assignments, changed score packets and singleton ticker groups.
    """
    if mode not in POLARITY_CONTROLS:
        raise ValueError(f"Unknown polarity control; choose from {POLARITY_CONTROLS}.")
    if (isinstance(shuffle_seed, bool) or not isinstance(shuffle_seed, (int, np.integer))
            or shuffle_seed < 0):
        raise ValueError("shuffle_seed must be a non-negative integer.")
    if not isinstance(include_outer, bool):
        raise TypeError("include_outer must be bool.")
    if (exported.protocol != FNSPID_PROTOCOL or exported.manifest.get("protocol") != FNSPID_PROTOCOL
            or exported.manifest.get("point_in_time") is not False):
        raise ValueError("Article polarity controls require the fnspid-exploratory protocol.")
    validation, test, end = _boundaries(fold)
    source = exported.frame.copy(deep=True)
    source["date"] = _aware_utc(source.date, name="date", nullable=False)
    if not source.date.eq(source.date.dt.normalize()).all():
        raise ValueError("FNSPID control decisions must be midnight UTC.")
    if source.duplicated(["date", "ticker"]).any():
        raise ValueError("FNSPID control input has duplicate decision keys.")
    selected = source.loc[source.date.le(end) if include_outer else source.date.lt(test)]
    selected = selected.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
    if selected.empty:
        raise ValueError("No news decisions fall within the requested fold control scope.")
    if (not selected.coverage_status.eq("unknown").all()
            or not selected.availability_kind.eq(AVAILABILITY_ASSUMPTION).all()):
        raise ValueError("FNSPID controls must preserve unknown historical coverage.")
    result = selected.copy(deep=True)
    partitions = ("train", "inner", "outer") if include_outer else ("train", "inner")
    inventories = {}
    if mode == "shuffled":
        if scored is None:
            raise ValueError("Shuffled polarity control requires scored article data.")
        aggregation = exported.manifest["aggregation"]
        parameters = _parameters(*(aggregation.get(name) for name in (
            "delay_hours", "lookback_hours", "short_lookback_hours", "half_life_hours"
        )))
        if parameters["lookback_hours"] > 24:
            raise ValueError("Shuffled article control requires lookback_hours <= 24 to prevent overlapping partitions.")
        decisions = selected.loc[:, ["ticker", "date"]].rename(columns={"date": "decision_at"})
        articles = _articles_for_decisions(scored, decisions, parameters, validation, test)
        articles, inventories = _permute_articles(
            articles, shuffle_seed=int(shuffle_seed), fold_number=fold["fold"], partitions=partitions,
        )
        raw = _aggregate(articles, decisions, parameters)
        # _aggregate returns the same canonical date/ticker order as selected.
        preserve = {"news_count": raw.news_count.to_numpy(dtype=np.float32),
                    "source_available": raw.news_count.gt(0).to_numpy(),
                    "available_at": raw.last_contributing_assumed_available_at.to_numpy(),
                    "coverage_status": raw.coverage_status.to_numpy(),
                    "observation_status": raw.observation_status.to_numpy(),
                    "availability_kind": raw.availability_kind.to_numpy()}
        for statistic in SENTIMENT_STATISTICS:
            values = raw[statistic]
            preserve[f"{statistic}_present"] = values.notna().to_numpy(dtype=np.float32)
            if statistic == "hours_since_last_news":
                preserve[statistic] = values.fillna(0).to_numpy(dtype=np.float32)
            else:
                result[statistic] = values.fillna(0).to_numpy(dtype=np.float32)
        for name, values in preserve.items():
            expected = pd.Series(values, name=name)
            try:
                pd.testing.assert_series_equal(selected[name], expected, check_dtype=False,
                                               check_names=False, check_exact=True)
            except AssertionError as error:
                raise ValueError(f"Scored control input disagrees with the verified export: {name}.") from error
    manifest = deepcopy(exported.manifest)
    intervals = {
        "train": {"start": None, "end": validation.isoformat(), "end_inclusive": False},
        "inner": {"start": validation.isoformat(), "end": test.isoformat(), "end_inclusive": False},
        "outer": {"start": test.isoformat(), "end": end.isoformat(), "end_inclusive": True},
    }
    manifest["polarity_control"] = {
        "schema_version": 1, "mode": mode, "fold": fold["fold"],
        "shuffle_seed": int(shuffle_seed) if mode == "shuffled" else None,
        "include_outer": include_outer,
        "decision_intervals": {name: intervals[name] for name in partitions},
        "decision_rows": len(result),
        "source_frame_sha256": _digest(selected), "controlled_frame_sha256": _digest(result),
        "partition_inventories": inventories,
        "permuted_article_columns": list(ARTICLE_SCORE_COLUMNS) if mode == "shuffled" else [],
        "seed_policy": "SHA256(shuffle_seed, fold, ticker, partition); numpy default_rng",
        "article_partition_policy": "first contributing decision; strict availability cutoff; inclusive window start",
        "neutralized_statistics": list(POLARITY_STATISTICS) if mode == "neutralized" else [],
        "neutralization_policy": "zero polarity value channels after TRAIN standardization; keep confidence, recency and all masks",
        "source_export_sha256": exported.manifest.get("parquet_sha256"),
        "point_in_time": False, "historical_coverage_claim": False,
    }
    return replace(exported, frame=result, manifest=manifest)


__all__ = ["POLARITY_CONTROLS", "POLARITY_STATISTICS", "ARTICLE_SCORE_COLUMNS", "prepare_fold_news_control"]
