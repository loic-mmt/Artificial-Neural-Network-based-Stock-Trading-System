"""Exposure controls use executable paths, aligned signals and causal daily scales."""

from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest

from trading_system.experiments.exposure_comparison import (
    _summary, aligned_prediction_group, compare_replay_exposure, normalize_positions,
)
from trading_system.artifacts.multimodal_study import file_record, validate_files
from trading_system.training.financial_loss import FinancialLossConfig, ReturnPanel


def _frame():
    dates = pd.date_range("2020-01-01", periods=40, freq="B", tz="UTC")
    rows = []
    for index, day in enumerate(dates):
        for number, ticker in enumerate(("A", "B")):
            price = (100 + 20 * number) * np.exp(.003 * index + .025 * np.sin(index / (3 + number)))
            rows.append({"date": day, "ticker": ticker, "adj_close": price})
    return pd.DataFrame(rows)


def _inputs(*, delay=1, permutation=False):
    frame = _frame()
    if permutation:
        frame = frame.sample(frac=1, random_state=3).reset_index(drop=True)
    panel = ReturnPanel(frame, group_col="ticker", execution_delay=delay)
    index = frame.date.map({day: i for i, day in enumerate(sorted(frame.date.unique()))}).to_numpy()
    sign = np.where(frame.ticker.eq("A"), 1., -1.)
    positions = {
        "gru": sign * (.65 + .2 * np.sin(index / 4)),
        "gnn": -sign * (.22 + .12 * np.cos(index / 5)),
    }
    return frame, panel, positions


def _predictions():
    frame, _, positions = _inputs()
    return pd.concat([
        frame.assign(candidate=name, position=values, available=True,
                     label_known=np.arange(len(frame)) % 3 != 0,
                     label=np.where(np.arange(len(frame)) % 3 == 0, -1, 2))
        for name, values in positions.items()
    ], ignore_index=True)


@pytest.mark.parametrize("method", ["mean_min", "daily_min"])
@pytest.mark.parametrize("delay", [1, 3])
def test_normalization_matches_executed_exposure_without_upsizing_or_input_mutation(method, delay):
    _, panel, positions = _inputs(delay=delay, permutation=True)
    original = {name: values.copy() for name, values in positions.items()}
    normalized, info = normalize_positions(panel, positions, method=method)
    loss = FinancialLossConfig("sharpe", cost_bps=17)
    means = []
    for name, values in normalized.items():
        scales = np.asarray(info["scales"][name])
        assert np.isfinite(scales).all()
        assert (scales >= 0).all() and (scales <= 1 + 1e-12).all()
        assert scales.shape == (() if method == "mean_min" else (panel.rows,))
        np.testing.assert_allclose(values, positions[name] * scales, atol=1e-12)
        assert (np.abs(values) <= np.abs(positions[name]) + 1e-12).all()
        np.testing.assert_array_equal(np.sign(values[values != 0]), np.sign(positions[name][values != 0]))
        np.testing.assert_array_equal(positions[name], original[name])
        means.append(panel.metrics(values, loss)["mean_abs_position"])
    np.testing.assert_allclose(means, info["target_mean_exposure"], atol=1e-12)
    if method == "mean_min":
        expected = min(panel.metrics(values, loss)["mean_abs_position"] for values in positions.values())
        assert info["target_mean_exposure"] == pytest.approx(expected)
    else:
        target = np.min([np.abs(values[panel.indices]).mean(axis=0)
                         for values in positions.values()], axis=0)
        executable = panel.returns.shape[1] - panel.delay
        for values in normalized.values():
            # The final delay+1 decisions cannot earn a return within this
            # split; their normalization is irrelevant to executable exposure.
            np.testing.assert_allclose(np.abs(values[panel.indices]).mean(axis=0)[:executable],
                                       target[:executable], atol=1e-12)


def test_mean_exposure_ignores_terminal_decisions_that_never_execute():
    _, panel, positions = _inputs(delay=3)
    baseline, baseline_info = normalize_positions(panel, positions, method="mean_min")
    changed = {name: values.copy() for name, values in positions.items()}
    terminal = panel.indices[:, -(panel.delay + 1):]
    changed["gru"][terminal] = 1.
    changed["gnn"][terminal] = 0.
    adjusted, info = normalize_positions(panel, changed, method="mean_min")
    assert info["target_mean_exposure"] == pytest.approx(baseline_info["target_mean_exposure"])
    for name in positions:
        assert info["scales"][name] == pytest.approx(baseline_info["scales"][name])
        np.testing.assert_allclose(panel.path(adjusted[name], FinancialLossConfig())[0],
                                   panel.path(baseline[name], FinancialLossConfig())[0])


def test_constant_scaling_preserves_ordinary_sharpe_and_recalculates_costs_and_compounding():
    frame = _frame()
    panel = ReturnPanel(frame, group_col="ticker", execution_delay=2)
    days = np.repeat(np.arange(40), 2)
    signal = .65 + .2 * np.sin(days / 4)
    adjusted, info = normalize_positions(panel, {"full": signal, "small": .25 * signal}, method="mean_min")
    scale = info["scales"]["full"]
    assert scale == pytest.approx(.25)
    loss = FinancialLossConfig("sharpe", cost_bps=19, sharpe_epsilon=.01)
    original_path = panel.path(signal, loss)
    adjusted_path = panel.path(adjusted["full"], loss)
    for before, after in zip(original_path, adjusted_path):
        np.testing.assert_allclose(after, scale * before, atol=1e-12)
    original_metrics = panel.metrics(signal, loss)
    metrics = panel.metrics(adjusted["full"], loss)
    assert metrics["net_sharpe"] == pytest.approx(original_metrics["net_sharpe"])
    assert metrics["regularized_sharpe"] != pytest.approx(original_metrics["regularized_sharpe"])
    assert metrics["cost_return_sum"] == pytest.approx(scale * original_metrics["cost_return_sum"])
    assert metrics["turnover"] == pytest.approx(scale * original_metrics["turnover"])
    wealth = np.r_[1., np.cumprod(1 + adjusted_path[0])]
    assert metrics["net_return"] == pytest.approx(wealth[-1] - 1)
    assert metrics["net_pnl"] == pytest.approx(10000 * (wealth[-1] - 1))
    assert metrics["max_drawdown"] == pytest.approx(np.min(wealth / np.maximum.accumulate(wealth) - 1))
    assert abs(metrics["net_return"] - scale * original_metrics["net_return"]) > 1e-6


@pytest.mark.parametrize("method", ["mean_min", "daily_min"])
def test_proportional_predictions_become_identical_and_zero_candidate_remains_finite(method):
    _, panel, positions = _inputs(permutation=True)
    signal = positions["gru"]
    adjusted, _ = normalize_positions(panel, {"full": signal, "small": .3 * signal}, method=method)
    np.testing.assert_allclose(adjusted["full"], adjusted["small"], atol=1e-12)
    adjusted, info = normalize_positions(panel, {"full": signal, "zero": np.zeros(panel.rows)}, method=method)
    assert info["target_mean_exposure"] == 0
    for name, values in adjusted.items():
        assert np.isfinite(info["scales"][name]).all()
        np.testing.assert_array_equal(values, np.zeros(panel.rows))
        metrics = panel.metrics(values, FinancialLossConfig("sharpe"))
        assert metrics["net_sharpe"] == 0
        assert metrics["regularized_sharpe"] == 0
        assert metrics["net_return"] == 0


def test_daily_normalization_is_causal_in_future_predictions():
    _, panel, positions = _inputs(permutation=True)
    normalized, info = normalize_positions(panel, positions, method="daily_min")
    changed = {name: values.copy() for name, values in positions.items()}
    past, future = panel.indices[:, :22].reshape(-1), panel.indices[:, 22:].reshape(-1)
    changed["gru"][future] = .01
    changed["gnn"][future] = -.95
    later, later_info = normalize_positions(panel, changed, method="daily_min")
    for name in positions:
        np.testing.assert_array_equal(normalized[name][past], later[name][past])
        np.testing.assert_array_equal(info["scales"][name][past], later_info["scales"][name][past])
    assert not np.allclose(normalized["gru"][future], later["gru"][future])


def test_alignment_preserves_unknown_labels_and_is_invariant_to_permutation():
    predictions = _predictions()
    expected = _frame().sort_values(["date", "ticker"]).reset_index(drop=True)
    frame, positions = aligned_prediction_group(predictions, ("gru", "gnn"))
    permuted, permuted_positions = aligned_prediction_group(
        predictions.sample(frac=1, random_state=8), ("gnn", "gru"))
    pd.testing.assert_frame_equal(frame, expected)
    pd.testing.assert_frame_equal(permuted, expected)
    assert len(frame) == 80
    assert (~predictions.label_known).any()
    for name in positions:
        np.testing.assert_array_equal(positions[name], permuted_positions[name])
        expected_values = predictions.loc[predictions.candidate.eq(name)].sort_values(["date", "ticker"]).position.to_numpy()
        np.testing.assert_array_equal(positions[name], expected_values)


@pytest.mark.parametrize("problem", ["duplicate", "missing_row", "different_ticker", "missing_calendar_date", "missing_candidate"])
def test_alignment_rejects_duplicate_or_incomplete_candidate_keys(problem):
    predictions = _predictions()
    if problem == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    elif problem == "missing_row":
        predictions = predictions.iloc[1:].copy()
    elif problem == "different_ticker":
        predictions.loc[0, "ticker"] = "C"
    elif problem == "missing_calendar_date":
        # Both candidates miss the same asset/day: exact inter-candidate keys
        # alone do not guarantee the complete portfolio calendar.
        predictions = predictions.loc[~(predictions.ticker.eq("B") & predictions.date.eq(predictions.date.iloc[3]))]
    else:
        predictions = predictions.loc[predictions.candidate.ne("gnn")]
    with pytest.raises(ValueError):
        aligned_prediction_group(predictions, ("gru", "gnn"))


@pytest.mark.parametrize("problem", ["mismatched_price", "zero_price", "nonfinite_price", "unavailable", "nan_position", "inf_position", "position_bounds"])
def test_alignment_rejects_incompatible_prices_availability_or_positions(problem):
    predictions = _predictions()
    if problem == "mismatched_price":
        predictions.loc[0, "adj_close"] += 1
    elif problem == "zero_price":
        predictions.loc[predictions.date.eq(predictions.date.iloc[0]), "adj_close"] = 0
    elif problem == "nonfinite_price":
        predictions.loc[0, "adj_close"] = np.nan
    elif problem == "unavailable":
        predictions.loc[0, "available"] = False
    else:
        predictions.loc[0, "position"] = {"nan_position": np.nan, "inf_position": np.inf,
                                           "position_bounds": 1.01}[problem]
    with pytest.raises(ValueError):
        aligned_prediction_group(predictions, ("gru", "gnn"))


def _replay_fixture(tmp_path, *, flat=False):
    """Build only frozen predictions and their reference paths, never models."""
    run, replay = tmp_path / "source-run", tmp_path / "source-replay"
    run.mkdir()
    replay.mkdir()
    candidates, seeds = ("gru", "identity", "rolling_topk"), [7]
    loss = FinancialLossConfig("combined", cost_bps=17, combined_pnl_weight=.25)
    capital, delay = 12500., 2
    predictions, tasks, metrics = [], [], []
    for fold in (0, 1):
        frame, _, signals = _inputs(delay=delay)
        frame = frame.copy()
        frame["date"] += pd.Timedelta(days=90 * fold)
        panel = ReturnPanel(frame, group_col="ticker", execution_delay=delay)
        positions = {"gru": signals["gru"], "identity": .6 * signals["gru"],
                     "rolling_topk": signals["gnn"]}
        if flat:
            positions = {name: np.zeros(panel.rows) for name in candidates}
        for candidate in candidates:
            for seed in seeds:
                tasks.append({"candidate": candidate, "fold": fold, "seed": seed})
                predictions.append(frame.assign(
                    candidate=candidate, fold=fold, seed=seed, partition="outer",
                    position=positions[candidate], available=True,
                    label_known=np.arange(len(frame)) % 4 != 0,
                    label=np.where(np.arange(len(frame)) % 4 == 0, -1, 2),
                ))
                metrics.append({"candidate": candidate, "fold": fold, "seed": seed,
                                "status": "ok", "outer_metrics": panel.metrics(positions[candidate], loss, capital)})
    metadata = {
        "config": {"initial_capital": capital, "execution_delay": delay},
        "ablation": {"candidates": list(candidates)},
        "loss_config": asdict(loss), "n_splits": 2, "seeds": seeds,
        "final_holdout_opened": False,
        "final_split": {"test_start": "2030-01-01T00:00:00+00:00"},
    }
    source = {"metadata": metadata, "folds": metrics, "final_test": []}
    (run / "report.json").write_text(json.dumps(source))
    path = replay / "predictions.parquet"
    pd.concat(predictions, ignore_index=True).to_parquet(path, index=False)
    replay_manifest = {"schema_version": 1, "source_run": str(run.resolve()),
                       "final_holdout_opened": False, "mismatches": [], "reusable": True,
                       "tasks": tasks, "files": [file_record(path, replay)]}
    (replay / "replay.json").write_text(json.dumps(replay_manifest))
    return run, replay, source


def _source_bytes(run, replay):
    return {path: path.read_bytes() for root in (run, replay)
            for path in root.iterdir() if path.is_file()}


def test_fresh_exposure_report_reproduces_raw_metrics_without_mutating_sources(tmp_path):
    run, replay, original = _replay_fixture(tmp_path)
    before = _source_bytes(run, replay)
    output = tmp_path / "exposure-output"
    result = compare_replay_exposure(replay, run, output)
    assert output.is_dir()
    assert {path.name for path in output.iterdir()} == {"report.json", "report.md", "daily_paths.parquet"}
    assert json.loads((output / "report.json").read_text()) == result
    assert result["training_performed"] is False
    assert result["final_holdout_opened"] is False
    assert result["descriptive_only"] is True
    assert result["source_run"] == str(run.resolve())
    assert result["source_replay"] == str(replay.resolve())
    assert result["candidates"] == ["gru", "identity", "rolling_topk", "buy_hold"]
    assert len(result["tasks"]) == 3 * 4 * 2
    validate_files(result["files"], output)
    saved = {(row["candidate"], row["fold"], row["seed"]): row for row in original["folds"]}
    for row in result["tasks"]:
        if row["method"] == "raw" and row["candidate"] != "buy_hold":
            expected = saved[row["candidate"], row["fold"], row["seed"]]["outer_metrics"]
            for key, value in expected.items():
                assert row["metrics"][key] == pytest.approx(value)
        assert 0 <= row["factor_min"] <= row["factor_max"] <= 1 + 1e-12
    daily = pd.read_parquet(output / "daily_paths.parquet")
    assert len(daily) == 39 * len(result["tasks"])
    assert not daily.duplicated(["method", "candidate", "fold", "seed", "date"]).any()
    np.testing.assert_allclose(daily.net_return, daily.gross_return - daily.cost, atol=1e-12)
    assert _source_bytes(run, replay) == before
    with pytest.raises(FileExistsError):
        compare_replay_exposure(replay, run, output)
    assert _source_bytes(run, replay) == before


@pytest.mark.parametrize("problem", [
    "metric_mismatch", "missing_task", "unavailable", "missing_prediction", "checksum_mismatch",
    "source_holdout_opened", "replay_holdout_opened", "replay_metric_mismatch",
    "wrong_source_run", "published_final_test",
])
def test_invalid_replay_or_raw_metrics_publish_no_partial_exposure_report(tmp_path, problem):
    run, replay, source = _replay_fixture(tmp_path)
    if problem == "metric_mismatch":
        # Fail after the first fold has already written temporary daily paths.
        source["folds"][-1]["outer_metrics"]["net_return"] += 1
        (run / "report.json").write_text(json.dumps(source))
    elif problem in {"source_holdout_opened", "published_final_test"}:
        if problem == "source_holdout_opened":
            source["metadata"]["final_holdout_opened"] = True
        else:
            source["final_test"] = [{"net_return": .1}]
        (run / "report.json").write_text(json.dumps(source))
    elif problem in {"replay_holdout_opened", "replay_metric_mismatch", "wrong_source_run"}:
        manifest = json.loads((replay / "replay.json").read_text())
        if problem == "replay_holdout_opened":
            manifest["final_holdout_opened"] = True
        elif problem == "replay_metric_mismatch":
            manifest["mismatches"] = [{"candidate": "gru", "fold": 0, "seed": 7}]
        else:
            manifest["source_run"] = str(tmp_path / "unrelated-run")
        (replay / "replay.json").write_text(json.dumps(manifest))
    elif problem == "missing_task":
        manifest = json.loads((replay / "replay.json").read_text())
        manifest["tasks"] = manifest["tasks"][:-1]
        (replay / "replay.json").write_text(json.dumps(manifest))
    else:
        path = replay / "predictions.parquet"
        if problem == "checksum_mismatch":
            path.write_bytes(b"corrupted prediction export")
        else:
            predictions = pd.read_parquet(path)
            if problem == "unavailable":
                predictions.loc[predictions.index[-1], "available"] = False
            else:
                predictions = predictions.iloc[:-1]
            predictions.to_parquet(path, index=False)
            manifest = json.loads((replay / "replay.json").read_text())
            manifest["files"] = [file_record(path, replay)]
            (replay / "replay.json").write_text(json.dumps(manifest))
    before = _source_bytes(run, replay)
    output = tmp_path / "invalid-exposure"
    with pytest.raises(ValueError):
        compare_replay_exposure(replay, run, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".invalid-exposure-*"))
    assert _source_bytes(run, replay) == before


def test_all_flat_candidates_produce_finite_report_and_zero_normalized_paths(tmp_path):
    run, replay, _ = _replay_fixture(tmp_path, flat=True)
    result = compare_replay_exposure(replay, run, tmp_path / "flat-exposure")
    for row in result["tasks"]:
        if row["method"] != "raw" or row["candidate"] != "buy_hold":
            assert row["metrics"]["mean_abs_position"] == 0
            assert row["metrics"]["net_return"] == 0
            assert row["metrics"]["net_sharpe"] == 0
            assert row["metrics"]["regularized_sharpe"] == 0
    for row in result["summary"]:
        if row["method"] != "raw" or row["candidate"] != "buy_hold":
            assert row["mean_metrics"]["net_sharpe"] == 0
            assert all(fold["net_sharpe"] == 0 for fold in row["by_fold"])


def test_summary_preserves_null_ordinary_sharpe_when_a_fold_has_no_available_value():
    rows = [{"method": "raw", "candidate": "gru", "fold": fold, "seed": seed,
             "metrics": {"net_return": .1, "net_sharpe": sharpe, "max_drawdown": -.02}}
            for fold, seed, sharpe in ((0, 1, None), (0, 7, None), (1, 1, 2.), (1, 7, None))]
    result = _summary(rows)[0]
    assert result["mean_metrics"]["net_sharpe"] == 2
    assert result["by_fold"][0]["net_sharpe"] is None
    assert result["by_fold"][1]["net_sharpe"] == 2
    unavailable = _summary(rows[:2])[0]
    assert unavailable["mean_metrics"]["net_sharpe"] is None
    assert unavailable["by_fold"][0]["net_sharpe"] is None


_AUDIT_FLAGS = (
    "feature_columns_match", "stock_scaler_allclose", "market_scaler_exact",
    "market_columns_match", "train_calendar_match", "inner_calendar_match", "outer_calendar_match",
)


def _audited_replay_fixture(tmp_path):
    run, replay, source = _replay_fixture(tmp_path)
    for field, checksum in (("dataset_sha256", "a" * 64),
                            ("graph_context_sha256", "b" * 64),
                            ("market_context_sha256", "c" * 64)):
        source["metadata"][field] = checksum
    (run / "report.json").write_text(json.dumps(source))
    manifest = json.loads((replay / "replay.json").read_text())
    manifest.update(mismatches=["market_context"], reusable=False,
                    source_dataset_sha256="a" * 64, replay_dataset_sha256="a" * 64)
    (replay / "replay.json").write_text(json.dumps(manifest))
    audit = {
        "schema_version": 1, "source_run": str(run.resolve()),
        "hashes": {
            "dataset": {"expected": "a" * 64, "actual": "a" * 64},
            "graph_context": {"expected": "b" * 64, "actual": "b" * 64},
            "market_context": {"expected": "c" * 64, "actual": "d" * 64},
        },
        "folds": [{"fold": fold, **{flag: True for flag in _AUDIT_FLAGS}}
                  for fold in (0, 1)],
    }
    path = tmp_path / "market-reconstruction-audit.json"
    path.write_text(json.dumps(audit))
    return run, replay, source, path, audit


def test_explicit_audited_market_reconstruction_remains_nonreusable_and_keeps_raw_metrics(tmp_path):
    run, replay, original, audit_path, audit = _audited_replay_fixture(tmp_path)
    before, audit_before = _source_bytes(run, replay), audit_path.read_bytes()
    output = tmp_path / "audited-exposure"
    result = compare_replay_exposure(replay, run, output, market_reconstruction_audit=audit_path)
    assert result["market_reconstruction_audited"] is True
    assert result["source_reusable"] is False
    assert result["replay_mismatches"] == ["market_context"]
    assert result["training_performed"] is False
    assert result["final_holdout_opened"] is False
    copied = output / "input-audit.json"
    assert json.loads(copied.read_text()) == audit
    assert file_record(copied, output) in result["files"]
    validate_files(result["files"], output)
    saved = {(row["candidate"], row["fold"], row["seed"]): row for row in original["folds"]}
    for row in result["tasks"]:
        if row["method"] == "raw" and row["candidate"] != "buy_hold":
            expected = saved[row["candidate"], row["fold"], row["seed"]]["outer_metrics"]
            for key, value in expected.items():
                assert row["metrics"][key] == pytest.approx(value)
    assert _source_bytes(run, replay) == before
    assert audit_path.read_bytes() == audit_before


@pytest.mark.parametrize("problem", [
    "other_mismatch", "extra_mismatch", "no_mismatch", "reusable_true", "reusable_unspecified",
    "wrong_schema", "wrong_source", "actual_dataset", "actual_graph_context",
    "expected_dataset", "expected_graph_context", "expected_market_context",
    "invalid_market_hash", "replay_dataset", "source_dataset", "missing_fold", "duplicate_fold",
    "missing_market_source_hash", "unknown_market_source_hash", "invalid_market_source_hash",
])
def test_market_audit_cannot_override_raw_hashes_or_unrelated_replay_mismatches(tmp_path, problem):
    run, replay, source, audit_path, audit = _audited_replay_fixture(tmp_path)
    manifest_path = replay / "replay.json"
    manifest = json.loads(manifest_path.read_text())
    if problem in {"other_mismatch", "extra_mismatch", "no_mismatch"}:
        manifest["mismatches"] = {
            "other_mismatch": ["dataset"], "extra_mismatch": ["market_context", "graph_context"],
            "no_mismatch": [],
        }[problem]
    elif problem == "reusable_true":
        manifest["reusable"] = True
    elif problem == "reusable_unspecified":
        manifest.pop("reusable")
    elif problem == "wrong_schema":
        audit["schema_version"] = 2
    elif problem == "wrong_source":
        audit["source_run"] = str(tmp_path / "other-run")
    elif problem.startswith("actual_"):
        audit["hashes"][problem.removeprefix("actual_")]["actual"] = "e" * 64
    elif problem.startswith("expected_"):
        audit["hashes"][problem.removeprefix("expected_")]["expected"] = "f" * 64
    elif problem == "invalid_market_hash":
        audit["hashes"]["market_context"]["actual"] = "z" * 64
    elif problem in {"missing_market_source_hash", "unknown_market_source_hash", "invalid_market_source_hash"}:
        if problem == "missing_market_source_hash":
            source["metadata"].pop("market_context_sha256")
            audit["hashes"]["market_context"].pop("expected")
        else:
            expected = None if problem == "unknown_market_source_hash" else "unverified-source"
            source["metadata"]["market_context_sha256"] = expected
            audit["hashes"]["market_context"]["expected"] = expected
        (run / "report.json").write_text(json.dumps(source))
    elif problem == "replay_dataset":
        manifest["replay_dataset_sha256"] = "e" * 64
    elif problem == "source_dataset":
        manifest["source_dataset_sha256"] = "e" * 64
    elif problem == "missing_fold":
        audit["folds"] = audit["folds"][:-1]
    else:
        audit["folds"][1]["fold"] = 0
    manifest_path.write_text(json.dumps(manifest))
    audit_path.write_text(json.dumps(audit))
    before = _source_bytes(run, replay)
    output = tmp_path / "rejected-audit"
    with pytest.raises(ValueError):
        compare_replay_exposure(replay, run, output, market_reconstruction_audit=audit_path)
    assert not output.exists()
    assert not list(tmp_path.glob(".rejected-audit-*"))
    assert _source_bytes(run, replay) == before


@pytest.mark.parametrize("flag", _AUDIT_FLAGS)
@pytest.mark.parametrize("invalid_value", [False, None, 1])
def test_market_reconstruction_audit_requires_every_fold_flag_to_be_explicitly_true(tmp_path, flag, invalid_value):
    run, replay, _, audit_path, audit = _audited_replay_fixture(tmp_path)
    audit["folds"][1][flag] = invalid_value
    audit_path.write_text(json.dumps(audit))
    output = tmp_path / "bad-fold-audit"
    with pytest.raises(ValueError):
        compare_replay_exposure(replay, run, output, market_reconstruction_audit=audit_path)
    assert not output.exists()


def test_market_reconstruction_mismatch_requires_explicit_audit_and_never_skips_raw_metric_check(tmp_path):
    run, replay, source, audit_path, _ = _audited_replay_fixture(tmp_path)
    output = tmp_path / "market-audit-raw-failure"
    with pytest.raises(ValueError, match="explicit market reconstruction audit"):
        compare_replay_exposure(replay, run, output)
    assert not output.exists()
    source["folds"][-1]["outer_metrics"]["net_return"] += 1
    (run / "report.json").write_text(json.dumps(source))
    before = _source_bytes(run, replay)
    with pytest.raises(ValueError, match="metrics differ"):
        compare_replay_exposure(replay, run, output, market_reconstruction_audit=audit_path)
    assert not output.exists()
    assert not list(tmp_path.glob(".market-audit-raw-failure-*"))
    assert _source_bytes(run, replay) == before
