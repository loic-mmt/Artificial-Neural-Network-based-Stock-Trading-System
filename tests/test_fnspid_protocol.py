"""Publication-delay windows stay distinct from historical PIT evidence."""

from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from trading_system.data.fnspid_export import (
    AVAILABILITY_ASSUMPTION, FNSPID_PROTOCOL, export_fnspid_sentiment,
    load_fnspid_sentiment_export,
)
from trading_system.data.multimodal import build_multimodal_dataset
from trading_system.data.news_sentiment import SENTIMENT_COLUMNS, load_news_sentiment_export


CHECKPOINT = "fixture-finbert@frozen-revision"


def _scores(dates, tickers=None, signals=None):
    count = len(dates)
    if tickers is None:
        tickers = ["A"] * count
    if signals is None:
        signals = np.linspace(-.4, .4, count)
    signals = np.asarray(signals)
    positive, negative = .3 + signals / 2, .3 - signals / 2
    return pd.DataFrame({
        "news_id": [f"id-{index}" for index in range(count)], "ticker": tickers,
        "published_at": pd.to_datetime(dates, utc=True),
        "p_positive": positive, "p_neutral": .4, "p_negative": negative,
        "sentiment_score": signals, "confidence": np.maximum.reduce([positive, negative, np.full(count, .4)]),
        "checkpoint": CHECKPOINT, "version_timing_uncertain": False, "publication_timing_uncertain": False,
    })


def _export(tmp_path, scores, market, *, name="features", **parameters):
    scored = tmp_path / f"{name}-scored.parquet"
    scores.to_parquet(scored, index=False)
    target = tmp_path / f"{name}.parquet"
    manifest = export_fnspid_sentiment(scored, market, target, checkpoint=CHECKPOINT,
                                      input_identifiers={"fixture": True}, **parameters)
    return target, load_fnspid_sentiment_export(target), manifest


def test_assumed_windows_have_strict_cutoff_inclusive_lower_bound_and_no_future(tmp_path):
    market = pd.DataFrame({"ticker": ["A", "B", "A"],
                           "date": ["2020-01-03", "2020-01-03", "2020-01-04"]})
    scores = _scores([
        "2020-01-01T00:00:00Z", "2020-01-01T18:00:00Z", "2020-01-02T00:00:00Z",
        "2019-12-31T23:59:00Z", "2020-01-02T10:00:00Z",
    ], signals=[-.4, .4, -.2, .5, .2])
    path, exported, manifest = _export(tmp_path, scores, market)
    raw = pd.read_parquet(path)
    first = raw.iloc[0]
    assert first.news_count == 2
    assert first.sentiment_mean == pytest.approx(0)
    assert first.sentiment_std == pytest.approx(.4)
    assert first.sentiment_ewm == pytest.approx((-.4 * .125 + .4) / 1.125)
    assert first.sentiment_momentum == pytest.approx(.4)
    assert first.hours_since_last_news == 6
    assert first.last_contributing_assumed_available_at == pd.Timestamp("2020-01-02T18:00:00Z")
    assert raw.iloc[2].news_count == 2
    assert raw.iloc[1].observation_status == "unobserved"
    assert raw.iloc[1][list(SENTIMENT_COLUMNS[1::2])].isna().all()
    assert exported.frame.source_available.tolist() == [True, False, True]
    assert exported.frame.coverage_status.eq("unknown").all()
    assert exported.frame.loc[1, list(SENTIMENT_COLUMNS)].eq(0).all()
    assert manifest["point_in_time"] is False
    assert manifest["historical_coverage_claim"] is False
    assert manifest["inputs"]["scored"]["sha256"]
    assert manifest["inputs"]["market"]["sha256"]
    assert manifest["observed_rows"] == 2
    assert manifest["model_columns"] == list(SENTIMENT_COLUMNS)
    with pytest.raises(ValueError, match="schema"):
        load_news_sentiment_export(path)
    with pytest.raises(ValueError, match="explicit fnspid-exploratory"):
        build_multimodal_dataset(market, tickers=["A", "B"], context_len=1,
                                 sentiment_frame=exported.frame, sentiment_columns=exported.columns)
    batch = build_multimodal_dataset(market, tickers=["A", "B"], context_len=1,
                                    sentiment_frame=exported.frame, sentiment_columns=exported.columns,
                                    sentiment_protocol=FNSPID_PROTOCOL).batch([0, 1])
    np.testing.assert_array_equal(batch.sentiment_mask, [[True, False], [True, False]])


def test_exploratory_export_checksum_and_observation_provenance_are_verified(tmp_path):
    market = pd.DataFrame({"ticker": ["A", "B"], "date": ["2020-01-03"] * 2})
    path, exported, _ = _export(tmp_path, _scores(["2020-01-01T18:00:00Z"]), market)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_fnspid_sentiment_export(path)
    changed = exported.frame.copy()
    changed.loc[1, "source_available"] = True
    with pytest.raises(ValueError, match="disagree"):
        build_multimodal_dataset(market, tickers=["A", "B"], context_len=1,
                                 sentiment_frame=changed, sentiment_columns=exported.columns,
                                 sentiment_protocol=FNSPID_PROTOCOL)
    changed = exported.frame.assign(coverage_status="covered")
    with pytest.raises(ValueError, match="historical coverage"):
        build_multimodal_dataset(market, tickers=["A", "B"], context_len=1,
                                 sentiment_frame=changed, sentiment_columns=exported.columns,
                                 sentiment_protocol=FNSPID_PROTOCOL)


@pytest.mark.parametrize("mutation, message", [
    ("probabilities", "probabilities"), ("checkpoint", "checkpoints"),
    ("duplicates", "duplicate"), ("timezone", "timezone"),
    ("uncertain", "uncertain article timing"), ("confidence", "confidence"),
])
def test_scored_input_rejects_malformed_or_unfrozen_rows(tmp_path, mutation, message):
    scores = _scores(["2020-01-01T18:00:00Z"])
    if mutation == "probabilities":
        scores.loc[0, "p_positive"] = 1.5
    elif mutation == "checkpoint":
        scores.loc[0, "checkpoint"] = "other"
    elif mutation == "duplicates":
        scores = pd.concat([scores, scores], ignore_index=True)
    elif mutation == "timezone":
        scores.published_at = scores.published_at.dt.tz_localize(None)
    elif mutation == "uncertain":
        scores.loc[0, "version_timing_uncertain"] = True
    elif mutation == "confidence":
        scores.loc[0, "confidence"] = .9
    market = pd.DataFrame({"date": ["2020-01-03"], "ticker": ["A"]})
    with pytest.raises(ValueError, match=message):
        _export(tmp_path, scores, market)
    assert not (tmp_path / "features.parquet").exists()


def test_loader_rejects_false_coverage_even_with_valid_checksum(tmp_path):
    market = pd.DataFrame({"date": ["2020-01-03"], "ticker": ["A"]})
    path, _, manifest = _export(tmp_path, _scores(["2020-01-01T18:00:00Z"]), market)
    raw = pd.read_parquet(path).assign(coverage_status="covered")
    raw.to_parquet(path, index=False)
    manifest["parquet_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="historical coverage"):
        load_fnspid_sentiment_export(path)


def _market():
    dates = pd.date_range("2020-01-01", periods=180, freq="B", tz="UTC")
    rows = []
    for asset, ticker in enumerate(("A", "B")):
        for index, day in enumerate(dates):
            close = 80 + 20 * asset + .12 * index + 2 * np.sin(index / (5 + asset))
            rows.append({"date": day, "ticker": ticker, "sector": "Finance", "open": close * .999,
                         "high": close * 1.01, "low": close * .99, "close": close, "adj_close": close,
                         "volume": 1000 + 10 * index + asset})
    return pd.DataFrame(rows)


def test_exploratory_cli_and_matched_training_preserve_missing_policy_and_resume(tmp_path):
    torch = pytest.importorskip("torch")
    from trading_system.experiments.news_sentiment_ablation import (
        NewsSentimentAblationConfig, run_news_sentiment_ablation,
    )
    from trading_system.pipelines.compare_news_sentiment import main
    from trading_system.pipelines.multi_ticker_long_short import DEFAULT_CONFIG
    from trading_system.training.financial_loss import FinancialLossConfig

    market = _market()
    rows = market.loc[np.arange(len(market)) % 4 != 0].reset_index(drop=True)
    scores = _scores(rows.date - pd.Timedelta(hours=27), rows.ticker,
                     .4 * np.sin(np.arange(len(rows)) / 8))
    path, exported, _ = _export(tmp_path, scores, market)
    config = replace(DEFAULT_CONFIG, context_len=5, device="cpu", label_mode="forward_return", forward_horizon=3)
    loss = FinancialLossConfig("sharpe")
    parameters = {"hidden_size": 4, "epochs": 1, "early_stopping_patience": 1}
    ablation = NewsSentimentAblationConfig(candidates=("gru", "sentiment", "gru_sentiment_mean"),
                                        sentiment_hidden_size=4, date_batch_size=16, news_protocol=FNSPID_PROTOCOL)
    destination = tmp_path / "comparison"
    options = dict(sentiment_export=exported, n_splits=2, gap_bars=2, ablation=ablation)
    with pytest.raises(ValueError, match="protocol.*disagree"):
        run_news_sentiment_ablation(market, config, loss, parameters, [1], destination,
                                   **{**options, "ablation": replace(ablation, news_protocol="pit")}, dry_run=True)
    assert not destination.exists()
    price_path = tmp_path / "prices.parquet"
    market.to_parquet(price_path, index=False)
    cli = ["--data", str(price_path), "--news-sentiment-export", str(path),
           "--news-protocol", FNSPID_PROTOCOL, "--output-dir", str(destination),
           "--preset", "multi_ticker_long_short", "--cv-folds", "2", "--cv-gap-bars", "2",
           "--context-len", "5", "--model-parameter-sets", '{"gru":[{"hidden_size":4,"epochs":1}]}',
           "--sentiment-candidates", "gru,sentiment,gru_sentiment_mean", "--seeds", "1", "--device", "cpu", "--dry-run"]
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        dry = main(cli)
        assert dry["metadata"]["news_protocol"] == FNSPID_PROTOCOL
        assert dry["metadata"]["point_in_time"] is False
        for task in dry["task_specs"]:
            scaler = task["spec"]["preprocessing"]["sentiment_scaler"]
            assert scaler["covered_fit_rows"] == 0
            assert scaler["observed_fit_rows"] == scaler["available_fit_rows"] > 0
        assert not destination.exists()
        report = run_news_sentiment_ablation(market, config, loss, parameters, [1], destination, **options)
        assert len(report["folds"]) == 6
        assert report["metadata"]["warnings"]
        assert report["metadata"]["final_holdout_opened"] is False
        for row in report["folds"]:
            predictions = pd.read_parquet(destination / row["prediction_artifacts"]["outer"])
            assert predictions.coverage_status.eq("unknown").all()
            assert predictions.availability_kind.eq(AVAILABILITY_ASSUMPTION).all()
            assert row["coverage"]["outer"]["covered_rows"] == 0
            assert row["coverage"]["outer"]["observed_rows"] > 0
            missing = ~predictions.news_available
            assert missing.any()
            if row["candidate"] == "sentiment":
                assert predictions.loc[missing, "position"].eq(0).all()
                assert not predictions.loc[missing, "available"].any()
            else:
                assert predictions.available.all()
        checkpoint = destination / report["folds"][0]["model_artifact"]
        modified = checkpoint.stat().st_mtime_ns
        run_news_sentiment_ablation(market, config, loss, parameters, [1], destination, **options, resume=True)
        assert checkpoint.stat().st_mtime_ns == modified
        manifest = dict(exported.manifest)
        manifest["aggregation"] = {**manifest["aggregation"], "delay_hours": 48}
        with pytest.raises(ValueError, match="Resume metadata"):
            run_news_sentiment_ablation(market, config, loss, parameters, [1], destination,
                                       **{**options, "sentiment_export": replace(exported, manifest=manifest)}, resume=True)
    finally:
        torch.set_num_threads(old_threads)
