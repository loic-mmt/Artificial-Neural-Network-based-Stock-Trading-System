"""Article controls preserve information availability and split isolation."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trading_system.data import fnspid_polarity
from trading_system.data.fnspid_export import (
    _scored_articles, export_fnspid_sentiment, load_fnspid_sentiment_export,
)
from trading_system.data.fnspid_polarity import (
    ARTICLE_SCORE_COLUMNS, POLARITY_STATISTICS, prepare_fold_news_control,
)
from trading_system.data.news_sentiment import SENTIMENT_STATISTICS
from trading_system.data.purged_cv import PurgedSplit


CHECKPOINT = "fixture-finbert@frozen-revision"
FOLD = {"fold": 2, "split": {"validation_start": "2020-01-04", "test_start": "2020-01-07"},
        "end": "2020-01-09"}


def _scores(dates, tickers, signals):
    signals = np.asarray(signals)
    positive, negative = .35 + signals / 2, .35 - signals / 2
    return pd.DataFrame({
        "news_id": [f"news-{index:04d}" for index in range(len(dates))],
        "ticker": tickers, "published_at": pd.to_datetime(dates, utc=True),
        "p_positive": positive, "p_neutral": .3, "p_negative": negative,
        "sentiment_score": signals,
        "confidence": np.maximum.reduce([positive, negative, np.full(len(dates), .3)]),
        "checkpoint": CHECKPOINT,
    })


def _export(tmp_path, scores, market, **parameters):
    source = tmp_path / "scores.parquet"
    destination = tmp_path / "features.parquet"
    scores.to_parquet(source, index=False)
    export_fnspid_sentiment(source, market, destination, checkpoint=CHECKPOINT,
                           input_identifiers={"fixture": True}, delay_hours=0, **parameters)
    return load_fnspid_sentiment_export(destination), _scored_articles(source, CHECKPOINT)


@pytest.fixture
def fixture(tmp_path):
    dates = pd.date_range("2020-01-01", "2020-01-11", tz="UTC")
    market = pd.DataFrame([(day, ticker) for day in dates for ticker in ("A", "B", "C")],
                          columns=("date", "ticker"))
    news_dates, tickers, signals = [], [], []
    for ticker, sign in (("A", 1), ("B", -1)):
        for day_index, day in enumerate(dates):
            for hour in (1, 5, 14, 21):
                news_dates.append(day + pd.Timedelta(hours=hour))
                tickers.append(ticker)
                signals.append(sign * (.04 + .004 * (day_index * 4 + hour)))
    # A singleton within INNER, plus many unavailable C decision rows.
    news_dates.append(pd.Timestamp("2020-01-03T12:00:00Z"))
    tickers.append("C")
    signals.append(.55)
    return _export(tmp_path, _scores(news_dates, tickers, signals), market)


def _control(fixture, **kwargs):
    exported, scores = fixture
    return prepare_fold_news_control(exported, scores, FOLD, **kwargs)


def test_shuffle_is_reproducible_order_independent_and_leaves_inputs_immutable(fixture):
    exported, scored = fixture
    frame_before, scored_before, manifest_before = exported.frame.copy(deep=True), scored.copy(deep=True), deepcopy(exported.manifest)
    first = _control(fixture, mode="shuffled", shuffle_seed=17, include_outer=True)
    second = _control(fixture, mode="shuffled", shuffle_seed=17, include_outer=True)
    reordered = prepare_fold_news_control(
        replace(exported, frame=exported.frame.sample(frac=1, random_state=5)),
        scored.sample(frac=1, random_state=7), FOLD, mode="shuffled", shuffle_seed=17, include_outer=True,
    )
    pd.testing.assert_frame_equal(first.frame, second.frame)
    pd.testing.assert_frame_equal(first.frame, reordered.frame)
    assert first.manifest == second.manifest == reordered.manifest
    pd.testing.assert_frame_equal(exported.frame, frame_before)
    pd.testing.assert_frame_equal(scored, scored_before)
    assert exported.manifest == manifest_before
    changed = _control(fixture, mode="shuffled", shuffle_seed=18, include_outer=True)
    assert not first.frame.sentiment_mean.equals(changed.frame.sentiment_mean)
    assert first.manifest["polarity_control"]["partition_inventories"]["inner"]["singleton_groups"] == 1


def test_shuffle_preserves_activity_recency_provenance_and_every_presence_mask(fixture):
    original = _control(fixture, include_outer=True)
    shuffled = _control(fixture, mode="shuffled", include_outer=True)
    invariant = [name for name in original.frame if name not in (*POLARITY_STATISTICS, "confidence_mean")]
    pd.testing.assert_frame_equal(shuffled.frame[invariant], original.frame[invariant])
    assert shuffled.frame.sentiment_mean.ne(original.frame.sentiment_mean).any()
    assert shuffled.frame.coverage_status.eq("unknown").all()
    unavailable = ~shuffled.frame.source_available
    assert unavailable.any()
    assert shuffled.frame.loc[unavailable, list(SENTIMENT_STATISTICS)].eq(0).all().all()
    c = shuffled.frame.ticker.eq("C")
    pd.testing.assert_frame_equal(shuffled.frame.loc[c], original.frame.loc[c])
    # Classifier packets are shuffled per ticker, never across A's positive and B's negative scores.
    assert shuffled.frame.loc[shuffled.frame.ticker.eq("A") & ~unavailable, "sentiment_mean"].gt(0).all()
    assert shuffled.frame.loc[shuffled.frame.ticker.eq("B") & ~unavailable, "sentiment_mean"].lt(0).all()


@pytest.mark.parametrize("partition,lower,upper", [
    ("train", "2020-01-01", "2020-01-04"),
    ("inner", "2020-01-04", "2020-01-07"),
    ("outer", "2020-01-07", "2020-01-10"),
])
def test_mutating_one_partition_cannot_change_other_partition_scores(fixture, partition, lower, upper):
    exported, scores = fixture
    reference = _control(fixture, mode="shuffled", include_outer=True)
    changed = scores.copy(deep=True)
    # Availability < decision means a midnight article first contributes one day later.
    first_decision = changed.published_at.dt.floor("D") + pd.Timedelta(days=1)
    within = first_decision.ge(pd.Timestamp(lower, tz="UTC")) & first_decision.lt(pd.Timestamp(upper, tz="UTC"))
    changed.loc[within, list(ARTICLE_SCORE_COLUMNS)] = [.7, .2, .1, .6, .7]
    actual = prepare_fold_news_control(exported, changed, FOLD, mode="shuffled", include_outer=True)
    outside = actual.frame.date.lt(pd.Timestamp(lower, tz="UTC")) | actual.frame.date.ge(pd.Timestamp(upper, tz="UTC"))
    pd.testing.assert_frame_equal(actual.frame.loc[outside], reference.frame.loc[outside])
    for name in ("train", "inner", "outer"):
        if name != partition:
            assert (actual.manifest["polarity_control"]["partition_inventories"][name]
                    == reference.manifest["polarity_control"]["partition_inventories"][name])


def test_inner_preparation_does_not_aggregate_outer_and_adding_outer_keeps_train_inner(fixture, monkeypatch):
    real_aggregate = fnspid_polarity._aggregate
    calls = []

    def recording_aggregate(articles, decisions, parameters):
        calls.append((articles.copy(), decisions.copy()))
        return real_aggregate(articles, decisions, parameters)

    monkeypatch.setattr(fnspid_polarity, "_aggregate", recording_aggregate)
    inner = _control(fixture, mode="shuffled", include_outer=False)
    assert calls[0][1].decision_at.lt(pd.Timestamp("2020-01-07", tz="UTC")).all()
    assert calls[0][0].partition.isin(("train", "inner")).all()
    outer = _control(fixture, mode="shuffled", include_outer=True)
    assert calls[1][1].decision_at.le(pd.Timestamp(FOLD["end"], tz="UTC")).all()
    pd.testing.assert_frame_equal(inner.frame, outer.frame.loc[outer.frame.date.lt(pd.Timestamp("2020-01-07", tz="UTC"))].reset_index(drop=True))
    for partition in ("train", "inner"):
        assert (inner.manifest["polarity_control"]["partition_inventories"][partition]
                == outer.manifest["polarity_control"]["partition_inventories"][partition])
    assert "outer" not in inner.manifest["polarity_control"]["decision_intervals"]


def test_strict_cutoff_inclusive_window_start_and_first_contributor_partition(tmp_path, monkeypatch):
    market = pd.DataFrame({"date": ["2020-01-03", "2020-01-04", "2020-01-05", "2020-01-07", "2020-01-09"],
                           "ticker": ["A"] * 5})
    scores = _scores([
        "2020-01-01T23:59:59Z",  # outside first window: never contributes
        "2020-01-02T00:00:00Z",  # lower bound: TRAIN
        "2020-01-03T00:00:00Z",  # at train cutoff: INNER on January 4
        "2020-01-04T00:00:00Z",  # at inner cutoff: January 5
        "2020-01-05T12:00:00Z",  # missing session and too old for January 7
        "2020-01-06T23:00:00Z",  # January 7 OUTER
        "2020-01-09T00:00:00Z",  # final cutoff: never included
    ], ["A"] * 7, [.1, .2, .3, .4, .5, .6, -.5])
    exported, validated = _export(tmp_path, scores, market)
    captured = []
    real_aggregate = fnspid_polarity._aggregate

    def inspect(articles, decisions, parameters):
        captured.append(articles.copy())
        return real_aggregate(articles, decisions, parameters)

    monkeypatch.setattr(fnspid_polarity, "_aggregate", inspect)
    result = prepare_fold_news_control(exported, validated, FOLD, mode="shuffled", include_outer=True)
    articles = captured[0].set_index("news_id")
    assert set(articles.index) == {"news-0001", "news-0002", "news-0003", "news-0005"}
    assert articles.loc["news-0001", "partition"] == "train"
    assert articles.loc["news-0002", "partition"] == "inner"
    assert articles.loc["news-0005", "partition"] == "outer"
    assert result.frame.news_count.tolist() == [1, 1, 1, 1, 0]
    assert result.frame.hours_since_last_news.tolist() == [24, 24, 24, 1, 0]


def test_neutralization_declares_only_polarity_values_and_keeps_the_source_frame(fixture):
    exported, _ = fixture
    neutral = prepare_fold_news_control(exported, None, FOLD, mode="neutralized")
    original = prepare_fold_news_control(exported, None, FOLD)
    pd.testing.assert_frame_equal(neutral.frame, original.frame)
    control = neutral.manifest["polarity_control"]
    assert control["neutralized_statistics"] == list(POLARITY_STATISTICS)
    assert len(control["neutralized_statistics"]) == 9
    assert "confidence_mean" not in control["neutralized_statistics"]
    assert "hours_since_last_news" not in control["neutralized_statistics"]
    assert control["permuted_article_columns"] == []
    assert neutral.frame.date.lt(pd.Timestamp(FOLD["split"]["test_start"], tz="UTC")).all()


def test_tied_article_times_use_news_id_for_order_independent_permutation(tmp_path):
    market = pd.DataFrame({"date": pd.date_range("2020-01-01", "2020-01-09", tz="UTC"), "ticker": "A"})
    scores = _scores(["2020-01-02T12:00:00Z"] * 8, ["A"] * 8, np.linspace(-.4, .4, 8))
    exported, scores = _export(tmp_path, scores, market)
    first = prepare_fold_news_control(exported, scores, FOLD, mode="shuffled")
    second = prepare_fold_news_control(exported, scores.iloc[::-1], FOLD, mode="shuffled")
    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert first.manifest == second.manifest


def test_empty_article_input_preserves_all_unavailable_rows_and_accepts_dataclass_split(tmp_path):
    market = pd.DataFrame({"date": pd.date_range("2020-01-01", "2020-01-09", tz="UTC"), "ticker": "A"})
    exported, scores = _export(tmp_path, _scores([], [], []), market)
    fold = {**FOLD, "split": PurgedSplit("2020-01-04", "2020-01-07")}
    actual = prepare_fold_news_control(exported, scores, fold, mode="shuffled", include_outer=True)
    pd.testing.assert_frame_equal(actual.frame, exported.frame)
    assert actual.frame.news_count.eq(0).all()
    assert not actual.frame.source_available.any()
    for inventory in actual.manifest["polarity_control"]["partition_inventories"].values():
        assert inventory["article_rows"] == inventory["changed_assignments"] == inventory["singleton_groups"] == 0


def test_control_rejects_duplicate_article_keys_missing_scores_and_overlapping_windows(fixture):
    exported, scores = fixture
    with pytest.raises(ValueError, match="duplicate article"):
        prepare_fold_news_control(exported, pd.concat((scores, scores.iloc[:1])), FOLD, mode="shuffled")
    with pytest.raises(ValueError, match="requires scored article"):
        prepare_fold_news_control(exported, None, FOLD, mode="shuffled")
    manifest = deepcopy(exported.manifest)
    manifest["aggregation"]["lookback_hours"] = 48
    with pytest.raises(ValueError, match="lookback_hours <= 24"):
        prepare_fold_news_control(replace(exported, manifest=manifest), scores, FOLD, mode="shuffled")


def test_control_rejects_protocol_provenance_disagreements_and_duplicate_decisions(fixture):
    exported, scores = fixture
    with pytest.raises(ValueError, match="fnspid-exploratory"):
        prepare_fold_news_control(replace(exported, protocol="pit"), scores, FOLD, mode="shuffled")
    with pytest.raises(ValueError, match="duplicate decision"):
        prepare_fold_news_control(replace(exported, frame=pd.concat((exported.frame, exported.frame.iloc[:1]))), scores, FOLD)
    changed = scores.copy()
    changed.loc[changed.ticker.eq("C"), "published_at"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="disagrees with the verified export"):
        prepare_fold_news_control(exported, changed, FOLD, mode="shuffled")


@pytest.mark.parametrize("seed", [True, -1, 1.5])
def test_control_rejects_invalid_seed(fixture, seed):
    with pytest.raises(ValueError, match="shuffle_seed"):
        _control(fixture, mode="shuffled", shuffle_seed=seed)
