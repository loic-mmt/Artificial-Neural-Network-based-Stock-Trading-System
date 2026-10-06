"""Synthetic offline news controls, masks and immutable matched artifacts."""

from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from trading_system.data.news_sentiment import SENTIMENT_COLUMNS, SENTIMENT_STATISTICS, load_news_sentiment_export
from trading_system.experiments.news_sentiment_ablation import (
    NewsSentimentAblationConfig, _SentimentScaler, _align_news, _dataset,
    _make_model, plan_run_news_sentiment_ablation, run_news_sentiment_ablation,
)
from trading_system.experiments.graph_ablation import _prepare
from trading_system.pipelines.compare_news_sentiment import main
from trading_system.pipelines.multi_ticker_long_short import DEFAULT_CONFIG
from trading_system.training.financial_loss import FinancialLossConfig


def _frame():
    dates = pd.date_range("2020-01-01", periods=180, freq="B", tz="UTC")
    rows = []
    for asset, ticker in enumerate(("A", "B")):
        time = np.arange(len(dates))
        close = 80 + 20 * asset + .12 * time + 2 * np.sin(time / (5 + asset))
        for index, day in enumerate(dates):
            rows.append({"date": day, "ticker": ticker, "sector": "Finance",
                         "open": close[index] * .999, "high": close[index] * 1.01,
                         "low": close[index] * .99, "close": close[index],
                         "adj_close": close[index], "volume": 1000 + 10 * index + asset})
    return pd.DataFrame(rows)


def _export(tmp_path, market):
    rows = []
    for ticker, part in market.groupby("ticker", sort=True):
        for index, day in enumerate(part.date):
            if ticker == "B" and index % 5 == 0:
                continue
            count = 0 if index % 4 == 0 else 1 + index % 3
            sentiment = .7 * np.sin(index / 8)
            row = {"ticker": ticker, "decision_at": day, "news_count": count,
                   "coverage_status": "unknown" if index % 7 == 0 else "covered",
                   "last_contributing_available_at": day - pd.Timedelta(hours=4) if count else pd.NaT}
            for name in SENTIMENT_STATISTICS:
                if not count:
                    row[name] = np.nan
                elif name in ("p_positive_mean", "positive_share"):
                    row[name] = .3 + .1 * sentiment
                elif name in ("p_negative_mean", "negative_share"):
                    row[name] = .3 - .1 * sentiment
                elif name == "p_neutral_mean":
                    row[name] = .4
                elif name == "hours_since_last_news":
                    row[name] = 4.
                elif name == "confidence_mean":
                    row[name] = .8
                elif name == "sentiment_std":
                    row[name] = .1
                else:
                    row[name] = sentiment
            rows.append(row)
    raw = pd.DataFrame(rows)
    path = tmp_path / "news.parquet"
    raw.to_parquet(path, index=False)
    manifest = {"schema_version": "1.0", "checkpoint": "synthetic-finbert@no-model-download",
                "aggregation": {"include_at_cutoff": False, "required_sources": ["fixture"]},
                "inputs": {"coverage": {"rows": len(raw), "sha256": "synthetic-not-real-news"}},
                "input_identifiers": {"domain": "company", "fixture": True},
                "feature_columns": list(raw.columns), "feature_rows": len(raw),
                "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    return path, load_news_sentiment_export(path)


def _inputs(tmp_path):
    frame = _frame()
    _, exported = _export(tmp_path, frame)
    config = replace(DEFAULT_CONFIG, context_len=5, device="cpu", label_mode="forward_return", forward_horizon=3)
    return frame, exported, config, FinancialLossConfig("sharpe"), {
        "hidden_size": 4, "epochs": 1, "early_stopping_patience": 1,
    }


def test_matched_news_comparison_masks_unknowns_and_preserves_rows(tmp_path):
    frame, exported, config, loss, parameters = _inputs(tmp_path)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        target = tmp_path / "comparison"
        options = dict(sentiment_export=exported, n_splits=2, gap_bars=2,
                       ablation=NewsSentimentAblationConfig(sentiment_hidden_size=4, date_batch_size=16))
        plan = run_news_sentiment_ablation(frame, config, loss, parameters, [1], target, **options, dry_run=True)
        assert not target.exists()
        assert len(plan["task_specs"]) == 10
        report = run_news_sentiment_ablation(frame, config, loss, parameters, [1], target, **options)
        assert len(report["folds"]) == 10
        assert report["metadata"]["final_holdout_opened"] is False
        assert report["final_test"] == []
        assert report["metadata"]["news_manifest_sha256"]
        assert report["exposure_controlled"]["selection_uses_adjusted_metrics"] is False
        assert len(report["exposure_controlled"]["tasks"]) == 12
        final_start = pd.Timestamp(report["metadata"]["final_split"]["test_start"])
        for fold in (0, 1):
            group = [row for row in report["folds"] if row["fold"] == fold]
            assert len({tuple(row["eligible_sessions"]["outer"]) for row in group}) == 1
            assert len({tuple(row["eligible_sessions"]["train"]) for row in group}) == 1
            assert len({tuple(row["feature_columns"]) for row in group}) == 1
            reference = next(row for row in group if row["candidate"] == "gru")
            for row in group:
                preds = pd.read_parquet(target / row["prediction_artifacts"]["outer"])
                assert len(preds) == 2 * row["outer_dates"]
                assert preds.backtest_key.is_unique
                assert preds.date.max() < final_start
                assert row["coverage"]["outer"]["covered_zero_news_rows"] > 0
                assert row["coverage"]["outer"]["unknown_rows"] > 0
                assert row["coverage"]["outer"]["missing_export_rows"] > 0
                unknown_labels = ~preds.label_known
                assert unknown_labels.any()
                assert preds.loc[unknown_labels, "label"].eq(-1).all()
                assert row["classification"]["labeled_rows"] == int(preds.label_known.sum())
                missing = ~preds.news_available
                if row["candidate"] == "sentiment":
                    assert (~preds.loc[missing, "available"]).all()
                    assert preds.loc[missing, "position"].eq(0).all()
                    assert preds.loc[missing, ["p_sell", "p_hold", "p_buy"]].eq([0, 1, 0]).all().all()
                    assert preds.loc[missing, ["logit_sell", "logit_hold", "logit_buy"]].isna().all().all()
                    assert preds.loc[missing, "probability_kind"].eq("flat_fallback").all()
                else:
                    assert preds.available.all()
                if row["candidate"] == "gru_features":
                    assert len(row["effective_temporal_columns"]) == len(reference["feature_columns"]) + 24
                if row["candidate"] == "gru_activity":
                    assert len(row["effective_temporal_columns"]) == len(reference["feature_columns"]) + 2
            exposure = [row["metrics"]["mean_abs_position"] for row in report["exposure_controlled"]["tasks"] if row["fold"] == fold]
            np.testing.assert_allclose(exposure, exposure[0], atol=1e-12)
        checkpoint = target / report["folds"][0]["model_artifact"]
        before = checkpoint.stat().st_mtime_ns
        resumed = run_news_sentiment_ablation(frame, config, loss, parameters, [1], target, **options, resume=True)
        assert len(resumed["folds"]) == 10
        assert checkpoint.stat().st_mtime_ns == before
        with pytest.raises(ValueError, match="Resume metadata"):
            run_news_sentiment_ablation(frame, config, loss, parameters, [1], target,
                **{**options, "sentiment_export": replace(exported, manifest={**exported.manifest, "checkpoint": "changed"})}, resume=True)
        prediction = target / report["folds"][0]["prediction_artifacts"]["outer"]
        prediction.write_bytes(prediction.read_bytes() + b"corrupt")
        with pytest.raises(ValueError, match="integrity"):
            run_news_sentiment_ablation(frame, config, loss, parameters, [1], target, **options, resume=True)
    finally:
        torch.set_num_threads(old_threads)


def test_train_only_scaler_presence_and_masked_mean_fallback(tmp_path):
    frame, exported, config, loss, parameters = _inputs(tmp_path)
    ablation = NewsSentimentAblationConfig(sentiment_hidden_size=4)
    plan = plan_run_news_sentiment_ablation(frame, config, loss, parameters, [1],
                                           sentiment_export=exported, ablation=ablation, n_splits=2, gap_bars=2)
    from trading_system.data.purged_cv import PurgedSplit
    fold = plan["metadata"]["cv_folds"][0]
    fold_frame = frame.loc[frame.date <= pd.Timestamp(fold["end"])].copy()
    prepared = _prepare(fold_frame, replace(config, purged_split=PurgedSplit(**fold["split"])), ablation)
    scaler = _SentimentScaler.fit(prepared, exported, config, required=True)
    news = _align_news(prepared.train, exported, config)
    eligible = news.source_available.to_numpy() & prepared.train._fit_eligible.to_numpy()
    assert scaler.means[0] == pytest.approx(news.loc[eligible, "news_count"].mean())
    valid = eligible & news.sentiment_mean_present.astype(bool).to_numpy()
    assert scaler.means[1] == pytest.approx(news.loc[valid, "sentiment_mean"].mean())
    changed = exported.frame.copy()
    changed.loc[changed.date >= pd.Timestamp(fold["split"]["validation_start"]), "sentiment_mean"] = 999.
    assert scaler == _SentimentScaler.fit(prepared, replace(exported, frame=changed), config, required=True)
    transformed = scaler.transform(news)
    assert set(transformed.sentiment_mean_present) <= {0., 1.}
    assert transformed.loc[~news.source_available, list(SENTIMENT_COLUMNS)].eq(0).all().all()
    dataset = _dataset(prepared.validation, prepared.history_validation, prepared, exported, scaler, config, "gru_sentiment_mean")
    batch = next(dataset.iter_batches(100))
    model = _make_model(batch, "gru_sentiment_mean", config, parameters, ablation, 1, torch)
    model.eval()
    with torch.no_grad():
        gru, sentiment, fused = model.gru(batch), model.sentiment(batch), model(batch)
    missing = torch.as_tensor(~batch.sentiment_mask)
    assert missing.any()
    assert not sentiment.availability[missing].any()
    assert fused.availability.all()
    torch.testing.assert_close(fused.logits[missing], gru.logits[missing])


def test_no_covered_train_fails_before_output_and_macro_not_forced_to_tickers(tmp_path):
    frame, exported, config, loss, parameters = _inputs(tmp_path)
    unknown = exported.frame.assign(source_available=False, coverage_status="unknown")
    destination = tmp_path / "no-coverage"
    with pytest.raises(ValueError, match="covered eligible TRAIN"):
        run_news_sentiment_ablation(frame, config, loss, parameters, [1], destination,
            sentiment_export=replace(exported, frame=unknown), n_splits=2, gap_bars=2, dry_run=True)
    assert not destination.exists()
    macro = replace(exported, manifest={**exported.manifest, "input_identifiers": {"domain": "macro"}})
    with pytest.raises(ValueError, match="Macro"):
        plan_run_news_sentiment_ablation(frame, config, loss, parameters, [1], sentiment_export=macro, n_splits=2)
    future = exported.frame.copy()
    row = future.index[future.news_count.gt(0)][0]
    future.loc[row, "available_at"] = future.loc[row, "date"]
    with pytest.raises(ValueError, match="midnight"):
        run_news_sentiment_ablation(frame, config, loss, parameters, [1], destination,
            sentiment_export=replace(exported, frame=future), n_splits=2, gap_bars=2, dry_run=True)
    assert not destination.exists()


def test_standalone_all_missing_validation_stays_flat_with_combined_loss(tmp_path):
    frame, exported, config, _, parameters = _inputs(tmp_path)
    plan = plan_run_news_sentiment_ablation(frame, config, FinancialLossConfig("combined"), parameters, [1],
        sentiment_export=exported, n_splits=2, gap_bars=2,
        ablation=NewsSentimentAblationConfig(candidates=("sentiment",), sentiment_hidden_size=4))
    start = pd.Timestamp(plan["metadata"]["cv_folds"][0]["split"]["validation_start"])
    source = exported.frame.copy()
    mask = source.date >= start
    source.loc[mask, "source_available"] = False
    source.loc[mask, "coverage_status"] = "unknown"
    target = tmp_path / "all-missing-validation"
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        report = run_news_sentiment_ablation(frame, config, FinancialLossConfig("combined"), parameters, [1], target,
            sentiment_export=replace(exported, frame=source), n_splits=2, gap_bars=2,
            ablation=NewsSentimentAblationConfig(candidates=("sentiment",), sentiment_hidden_size=4))
    finally:
        torch.set_num_threads(old_threads)
    for row in report["folds"]:
        assert row["inner_metrics"]["mean_abs_position"] == 0
        assert row["outer_metrics"]["mean_abs_position"] == 0
        assert row["classification_available_only"] is None
        assert row["coverage"]["train"]["covered_rows"] > 0


def test_news_cli_dry_run_is_offline_and_read_only(tmp_path):
    frame = _frame()
    news_path, _ = _export(tmp_path, frame)
    price_path = tmp_path / "prices.parquet"
    frame.to_parquet(price_path, index=False)
    target = tmp_path / "cli-output"
    result = main(["--data", str(price_path), "--news-sentiment-export", str(news_path),
                   "--output-dir", str(target), "--preset", "multi_ticker_long_short",
                   "--cv-folds", "2", "--cv-gap-bars", "2", "--context-len", "5",
                   "--model-parameter-sets", '{"gru":[{"hidden_size":4,"epochs":1}]}',
                   "--seeds", "1", "--device", "cpu", "--dry-run"])
    assert result["dry_run"] is True
    assert result["metadata"]["shared_price_warmup"] == 3
    assert not target.exists()
