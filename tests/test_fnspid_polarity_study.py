"""Matched polarity controls retain news activity and sealed evaluation calendars."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_fnspid_protocol import _export, _market, _scores
from test_fnspid_news_study import _args, _ready_manifest, _tokens, _value

from trading_system.data.fnspid_export import FNSPID_PROTOCOL
from trading_system.data.news_pilot_scoring import file_sha256
from trading_system.data.news_sentiment import SENTIMENT_COLUMNS, SENTIMENT_STATISTICS
from trading_system.experiments.graph_ablation import _prepare
from trading_system.experiments.news_sentiment_ablation import (
    CANDIDATES, NewsSentimentAblationConfig, _SentimentScaler, _dataset,
    _temporal_columns, plan_run_news_sentiment_ablation, run_news_sentiment_ablation,
)
from trading_system.pipelines import fnspid_news_study as study
from trading_system.pipelines.compare_news_sentiment import main as compare_main
from trading_system.pipelines.multi_ticker_long_short import DEFAULT_CONFIG
from trading_system.training.financial_loss import FinancialLossConfig


POLARITY_CANDIDATES = (
    "gru", "gru_activity", "gru_features", "gru_features_shuffled", "gru_features_neutralized",
)
POLARITY_STATISTICS = tuple(
    name for name in SENTIMENT_STATISTICS if name not in ("confidence_mean", "hours_since_last_news")
)


def _inputs(tmp_path):
    market = _market()
    rows = market.loc[np.arange(len(market)) % 4 != 0].reset_index(drop=True)
    scores = _scores(rows.date - pd.Timedelta(hours=27), rows.ticker,
                     .4 * np.sin(np.arange(len(rows)) / 8))
    daily, exported, _ = _export(tmp_path, scores, market)
    scored = tmp_path / "features-scored.parquet"
    config = replace(DEFAULT_CONFIG, context_len=5, device="cpu",
                     label_mode="forward_return", forward_horizon=3)
    parameters = {"hidden_size": 4, "epochs": 1, "early_stopping_patience": 1}
    ablation = NewsSentimentAblationConfig(
        candidates=POLARITY_CANDIDATES, sentiment_hidden_size=4, date_batch_size=16,
        news_protocol=FNSPID_PROTOCOL, scored_articles_path=str(scored), shuffle_seed=314159,
    )
    return market, exported, config, FinancialLossConfig("sharpe"), parameters, ablation, daily, scored


def _fingerprints(root):
    return {str(path.relative_to(root)): (path.stat().st_mtime_ns, file_sha256(path))
            for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.25, "314159"])
def test_shuffle_seed_rejects_invalid_values(seed):
    with pytest.raises(ValueError, match="shuffle_seed"):
        NewsSentimentAblationConfig(shuffle_seed=seed)


def test_polarity_candidates_are_opt_in_and_require_explicit_fnspid_protocol():
    assert CANDIDATES == ("gru", "gru_activity", "gru_features", "sentiment", "gru_sentiment_mean")
    assert NewsSentimentAblationConfig().candidates == CANDIDATES
    assert NewsSentimentAblationConfig().shuffle_seed == 314159
    for candidate in ("gru_features_shuffled", "gru_features_neutralized"):
        with pytest.raises(ValueError, match="fnspid|protocol"):
            NewsSentimentAblationConfig(candidates=(candidate,))


@pytest.mark.parametrize("problem", ["missing", "checksum", "protocol"])
def test_polarity_preflight_rejects_unverified_inputs_without_creating_output(tmp_path, problem):
    market, exported, config, loss, parameters, ablation, _, scored = _inputs(tmp_path)
    if problem == "missing":
        ablation = replace(ablation, scored_articles_path=None)
    elif problem == "checksum":
        changed = pd.read_parquet(scored)
        changed.loc[0, "sentiment_score"] += .01
        changed.to_parquet(scored, index=False)
    else:
        exported = replace(exported, protocol="pit")
    target = tmp_path / "rejected-comparison"
    before = _fingerprints(tmp_path)
    with pytest.raises(ValueError, match="scored|checksum|protocol|SHA"):
        run_news_sentiment_ablation(
            market, config, loss, parameters, [1], target, sentiment_export=exported,
            n_splits=2, gap_bars=2, ablation=ablation, dry_run=True,
        )
    assert not target.exists()
    assert _fingerprints(tmp_path) == before


def test_neutralization_zeros_polarity_after_scaling_in_target_and_history(tmp_path):
    pytest.importorskip("torch")
    from trading_system.data.purged_cv import PurgedSplit

    market, exported, config, loss, parameters, ablation, _, _ = _inputs(tmp_path)
    plan = plan_run_news_sentiment_ablation(
        market, config, loss, parameters, [1], sentiment_export=exported,
        n_splits=2, gap_bars=2, ablation=ablation,
    )
    fold = plan["metadata"]["cv_folds"][0]
    frame = market.loc[market.date <= pd.Timestamp(fold["end"])].copy()
    prepared = _prepare(frame, replace(config, purged_split=PurgedSplit(**fold["split"])), ablation)
    scaler = _SentimentScaler.fit(prepared, exported, config, required=True)
    arguments = (prepared.validation, prepared.history_validation, prepared, exported, scaler, config)
    original = _dataset(*arguments, "gru_features")
    neutral = _dataset(*arguments, "gru_features_neutralized")
    columns = _temporal_columns(prepared, "gru_features", FNSPID_PROTOCOL)
    assert columns == _temporal_columns(prepared, "gru_features_neutralized", FNSPID_PROTOCOL)
    assert columns == _temporal_columns(prepared, "gru_features_shuffled", FNSPID_PROTOCOL)
    assert len(columns) == len(prepared.columns) + len(SENTIMENT_COLUMNS) + 1
    polarity = [columns.index("news_sentiment__" + name) for name in POLARITY_STATISTICS]
    kept = [index for index in range(len(columns)) if index not in polarity]
    news_polarity = [SENTIMENT_COLUMNS.index(name) for name in POLARITY_STATISTICS]
    news_kept = [index for index in range(len(SENTIMENT_COLUMNS)) if index not in news_polarity]
    seen_nonzero, seen_history = False, False
    for source, controlled in zip(original.iter_batches(16), neutral.iter_batches(16), strict=True):
        assert controlled.temporal.shape == source.temporal.shape
        np.testing.assert_array_equal(controlled.temporal[..., polarity], 0)
        np.testing.assert_array_equal(controlled.temporal[..., kept], source.temporal[..., kept])
        np.testing.assert_array_equal(controlled.sentiment[..., news_polarity], 0)
        np.testing.assert_array_equal(controlled.sentiment[..., news_kept], source.sentiment[..., news_kept])
        for name in ("asset_mask", "temporal_mask", "sentiment_mask", "label_mask", "row_positions"):
            np.testing.assert_array_equal(getattr(controlled, name), getattr(source, name))
        seen_nonzero |= bool(np.any(source.temporal[..., polarity] != 0))
        # The first INNER batch's preceding windows include eligible TRAIN history.
        seen_history |= bool(np.any(source.temporal[0, :, :-1, polarity] != 0))
    assert seen_nonzero and seen_history
    assert len(POLARITY_STATISTICS) == 9


def test_compare_cli_polarity_dry_run_is_offline_and_read_only(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from trading_system.data import fnspid_scoring

    market, _, _, _, _, ablation, daily, scored = _inputs(tmp_path)
    price_path = tmp_path / "prices.parquet"
    market.to_parquet(price_path, index=False)
    target = tmp_path / "cli-output"
    monkeypatch.setattr(fnspid_scoring, "score_fnspid",
                        lambda *args, **kwargs: pytest.fail("Polarity controls reuse frozen scores."))
    before = _fingerprints(tmp_path)
    plan = compare_main([
        "--data", str(price_path), "--news-sentiment-export", str(daily),
        "--news-scored-articles", str(scored), "--news-shuffle-seed", "314159",
        "--news-protocol", FNSPID_PROTOCOL, "--output-dir", str(target),
        "--preset", "multi_ticker_long_short", "--cv-folds", "2", "--cv-gap-bars", "2",
        "--context-len", "5", "--model-parameter-sets", '{"gru":[{"hidden_size":4,"epochs":1}]}',
        "--sentiment-candidates", ",".join(ablation.candidates), "--seeds", "1",
        "--device", "cpu", "--dry-run",
    ])
    assert plan["dry_run"] is True and len(plan["task_specs"]) == 10
    assert plan["metadata"]["ablation"]["shuffle_seed"] == 314159
    assert Path(plan["metadata"]["ablation"]["scored_articles_path"]).resolve() == scored.resolve()
    assert plan["metadata"]["final_holdout_opened"] is False
    for task in plan["task_specs"]:
        control = task["spec"]["preprocessing"]["news_control"]
        assert control["mode"] in (
            "original", "shuffled", "neutralized",
        )
        assert len(control["export_sha256"]) == 64
    assert not target.exists() and _fingerprints(tmp_path) == before


@pytest.mark.parametrize("dry_run", [True, False])
def test_fnspid_polarity_stage_forwards_45_tasks_in_one_comparison(tmp_path, monkeypatch, dry_run):
    from trading_system.pipelines import compare_news_sentiment

    args = _args(tmp_path)
    _ready_manifest(args)
    scoring = args.prepared_dir / "scoring"
    scoring.mkdir()
    scored = scoring / "scored_company.parquet"
    scored.write_bytes(b"frozen-scored-articles")
    calls = []
    monkeypatch.setattr(study, "prepare", lambda _: pytest.fail("Polarity uses the prepared pilot."))
    monkeypatch.setattr(compare_news_sentiment, "main",
                        lambda tokens: calls.append(list(tokens)) or {"calls": len(calls)})
    before = _fingerprints(tmp_path)
    flags = ["--shuffle-seed", "23"] + (["--dry-run"] if dry_run else [])
    assert study.main(_tokens(args, "polarity", *flags)) == {"calls": 1}
    assert len(calls) == 1
    tokens = calls[0]
    assert ("--dry-run" in tokens) is dry_run
    assert _value(tokens, "--sentiment-candidates") == ",".join(POLARITY_CANDIDATES)
    assert _value(tokens, "--news-scored-articles") == str(scored)
    assert _value(tokens, "--news-shuffle-seed") == "23"
    assert Path(_value(tokens, "--output-dir")) == Path("artifacts/comparisons/fnspid-news-polarity-controls")
    tasks = len(_value(tokens, "--sentiment-candidates").split(","))
    tasks *= len(_value(tokens, "--seeds").split(",")) * int(_value(tokens, "--cv-folds"))
    assert tasks == 45
    assert "--final-test" not in tokens and "--cv-final-test" not in tokens
    assert _fingerprints(tmp_path) == before


def test_tiny_polarity_fits_have_matched_capacity_and_resume_without_retraining(tmp_path):
    torch = pytest.importorskip("torch")
    market, exported, config, loss, parameters, ablation, daily, scored = _inputs(tmp_path)
    source_paths = (daily, daily.with_suffix(".manifest.json"), scored)
    source_hashes = {path: file_sha256(path) for path in source_paths}
    target = tmp_path / "trained-controls"
    options = dict(sentiment_export=exported, n_splits=2, gap_bars=2, ablation=ablation)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        report = run_news_sentiment_ablation(market, config, loss, parameters, [1], target, **options)
        assert len(report["folds"]) == 10
        assert report["metadata"]["final_holdout_opened"] is False
        assert report["final_test"] == []
        final_start = pd.Timestamp(report["metadata"]["final_split"]["test_start"])
        expected_modes = {"gru_features_shuffled": "shuffled", "gru_features_neutralized": "neutralized"}
        for fold in (0, 1):
            rows = [row for row in report["folds"] if row["fold"] == fold]
            models = [row for row in rows if row["candidate"].startswith("gru_features")]
            assert len(models) == 3
            assert len({row["fit"]["parameter_count"] for row in models}) == 1
            assert len({tuple(row["effective_temporal_columns"]) for row in models}) == 1
            assert len({tuple(row["eligible_sessions"]["train"]) for row in rows}) == 1
            assert len({tuple(row["eligible_sessions"]["inner"]) for row in rows}) == 1
            assert len({tuple(row["eligible_sessions"]["outer"]) for row in rows}) == 1
            masks = []
            for row in rows:
                control = row["news_control"]
                assert control["mode"] == expected_modes.get(row["candidate"], "original")
                assert len(control["export_sha256"]) == 64
                predictions = pd.read_parquet(target / row["prediction_artifacts"]["outer"])
                assert predictions.available.all() and predictions.backtest_key.is_unique
                assert predictions.date.max() < final_start
                masks.append(predictions[["date", "ticker", "news_available", "observation_status", "coverage_status"]])
            for observed in masks[1:]:
                pd.testing.assert_frame_equal(observed, masks[0])
        checkpoint_mtimes = {path: path.stat().st_mtime_ns for path in target.glob("*.pt")}
        assert len(checkpoint_mtimes) == 10
        resumed = run_news_sentiment_ablation(
            market, config, loss, parameters, [1], target, **options, resume=True,
        )
        assert len(resumed["folds"]) == 10
        assert {path: path.stat().st_mtime_ns for path in checkpoint_mtimes} == checkpoint_mtimes
        with pytest.raises(ValueError, match="Resume metadata|signature"):
            run_news_sentiment_ablation(
                market, config, loss, parameters, [1], target,
                **{**options, "ablation": replace(ablation, shuffle_seed=271828)}, resume=True,
            )
        assert {path: path.stat().st_mtime_ns for path in checkpoint_mtimes} == checkpoint_mtimes
    finally:
        torch.set_num_threads(previous_threads)
    assert {path: file_sha256(path) for path in source_paths} == source_hashes
