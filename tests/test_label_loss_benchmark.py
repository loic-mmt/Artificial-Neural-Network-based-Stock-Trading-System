"""Integration checks for the frozen post-open label/loss benchmark."""

from dataclasses import replace
import json
import shutil

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from trading_system.data.post_open import build_post_open_frame
from trading_system.experiments.label_loss_benchmark import (
    LabelLossBenchmarkConfig, _supervision, benchmark_counts, calendar_folds,
    candidate_grid, common_exposure_comparison, prepare_common_fold, run_label_loss_benchmark,
)
from trading_system.pipelines.label_loss_benchmark import build_parser, main
from trading_system.training.post_open_panel import PostOpenReturnPanel


def prices(n=190):
    calendar = pd.bdate_range("2010-01-04", periods=n)
    parts = []
    for i, ticker in enumerate(("A", "B")):
        rng = np.random.default_rng(101 + i)
        opening = (80 + 30 * i) * np.exp(np.cumsum(rng.normal(0.0005, 0.009, n)))
        closing = opening * np.exp(rng.normal(0.0003, 0.012, n))
        parts.append(pd.DataFrame({
            "date": calendar, "ticker": ticker, "open": opening,
            "high": np.maximum(opening, closing) * (1 + rng.uniform(0.001, 0.01, n)),
            "low": np.minimum(opening, closing) * (1 - rng.uniform(0.001, 0.01, n)),
            "close": closing, "adj_close": closing,
            "volume": rng.integers(10000, 1000000, n).astype(float),
            "sector": "Technology" if i == 0 else "Financials",
        }))
    return pd.concat(parts, ignore_index=True)


def tiny_config(**changes):
    return LabelLossBenchmarkConfig(
        context_len=3, max_features=4, n_folds=1, folds=(0,), seeds=(1,),
        feature_groups=("technical",), base_epochs=1, epoch_multiplier=1,
        batch_size=32, hidden_size=4, gap_bars=3, volatility_window=10, device="cpu",
        **changes,
    )


def test_default_grid_is_99_unique_fits_and_378_evaluation_paths():
    config = LabelLossBenchmarkConfig()
    assert benchmark_counts(config) == {
        "unique_configurations": 11, "fits": 99, "evaluation_paths": 378,
    }
    grid = candidate_grid(config)
    assert len({candidate["candidate"] for candidate in grid}) == 11
    assert sum(candidate["family"] == "cross_entropy" for candidate in grid) == 3
    assert sum(candidate["family"] == "financial" for candidate in grid) == 2
    assert sum(candidate["family"] == "hybrid" for candidate in grid) == 6
    assert all(candidate["label_method"] is None for candidate in grid if candidate["family"] == "financial")
    assert all(candidate["training_protocol"] is None for candidate in grid if candidate["family"] == "cross_entropy")


def test_parser_and_configuration_apply_tripled_epoch_budget():
    args = build_parser().parse_args([])
    assert args.base_epochs == 100 and args.epoch_multiplier == 3
    config = LabelLossBenchmarkConfig(base_epochs=args.base_epochs, epoch_multiplier=args.epoch_multiplier)
    assert config.model_parameters()["epochs"] == 300
    assert replace(config, epoch_multiplier=2).model_parameters()["epochs"] == 200
    assert config.financial_config().combined_pnl_weight == 0.25
    assert config.hybrid_ce_weight == 0.5


def test_dry_run_prints_grid_without_reading_prices_or_training(capsys, monkeypatch):
    import trading_system.pipelines.label_loss_benchmark as pipeline

    def forbidden(*args, **kwargs):
        raise AssertionError("A dry run attempted price I/O or training.")

    monkeypatch.setattr(pipeline, "read_parquet_dataset", forbidden)
    monkeypatch.setattr(pipeline, "run_label_loss_benchmark", forbidden)
    assert main(["--tickers", "A", "B", "--dry-run", "--data", "does-not-exist.parquet"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["counts"]["fits"] == 99
    assert result["ticker_count"] == 2
    assert result["config"]["epoch_multiplier"] == 3


def test_global_date_partitions_are_disjoint_and_preserve_requested_gaps():
    config = replace(tiny_config(), n_folds=3, folds=None)
    frame = prices().sample(frac=1, random_state=7)
    folds = calendar_folds(frame, config)
    calendar = pd.DatetimeIndex(pd.to_datetime(frame.date, utc=True)).unique().sort_values()
    assert len(folds) == 3
    outer_sets = []
    for fold in folds:
        sets = []
        for part in ("train", "inner", "outer"):
            selected = calendar[(calendar >= pd.Timestamp(fold[f"{part}_start"])) &
                                (calendar <= pd.Timestamp(fold[f"{part}_end"]))]
            sets.append(set(selected))
        assert not sets[0] & sets[1] and not sets[1] & sets[2] and not sets[0] & sets[2]
        assert calendar.get_loc(pd.Timestamp(fold["inner_start"])) - calendar.get_loc(pd.Timestamp(fold["train_end"])) - 1 == config.gap_bars
        assert calendar.get_loc(pd.Timestamp(fold["outer_start"])) - calendar.get_loc(pd.Timestamp(fold["inner_end"])) - 1 == config.gap_bars
        outer_sets.append(sets[2])
    assert all(not outer_sets[i] & outer_sets[j] for i in range(3) for j in range(i))


def test_common_selector_imputer_scaler_are_label_independent_and_train_only():
    config = tiny_config()
    featured, _ = build_post_open_frame(prices(), groups=config.feature_groups)
    featured["date"] = pd.to_datetime(featured.date, utc=True)
    fold = calendar_folds(featured, config)[0]
    lagged = [column for column in featured if column.startswith("lag1_")]
    # Missing TRAIN values exercise fitting and deterministic common fills.
    featured.loc[featured.index[::17], lagged[0]] = np.nan
    first, preprocessing = prepare_common_fold(featured, fold, config)
    changed = featured.copy()
    changed["Label_id"] = np.arange(len(changed)) % 3
    changed["_label_known"] = False
    after_train = changed.date > pd.Timestamp(fold["train_end"])
    changed.loc[after_train, lagged] = np.broadcast_to(
        1e12 + np.arange(int(after_train.sum()))[:, None],
        (int(after_train.sum()), len(lagged)),
    )
    changed.loc[after_train, "night_pct"] = 1234.0
    second, altered = prepare_common_fold(changed, fold, config)
    assert preprocessing == altered
    np.testing.assert_array_equal(first["train"]["X"], second["train"]["X"])
    assert preprocessing["label_independent"]
    assert preprocessing["columns"][-1] == "night_pct"
    assert len(preprocessing["columns"]) <= config.max_features
    assert np.isfinite(first["train"]["X"]).all()
    for phase, part in first.items():
        target_dates = pd.to_datetime(part["aligned"].date, utc=True)
        assert target_dates.between(pd.Timestamp(fold[f"{phase}_start"]), pd.Timestamp(fold[f"{phase}_end"])).all()
    # Inner validation receives a complete first-session context using earlier
    # price features; historical rows do not become inner targets themselves.
    assert pd.to_datetime(first["inner"]["aligned"].date, utc=True).min() == pd.Timestamp(fold["inner_start"])
    for ticker in ("A", "B"):
        aligned = first["inner"]["aligned"]
        location = np.flatnonzero(aligned.ticker.to_numpy() == ticker)[0]
        first_date = pd.Timestamp(aligned.iloc[location].date)
        historical = featured.loc[(featured.ticker == ticker) & (featured.date <= first_date)].tail(config.context_len)
        expected = historical[preprocessing["columns"]].fillna(preprocessing["fill_values"]).to_numpy(dtype=np.float32)
        mean = np.asarray(preprocessing["scaler"]["mean"], dtype=np.float32)
        scale = np.asarray(preprocessing["scaler"]["scale"], dtype=np.float32)
        np.testing.assert_allclose(first["inner"]["X"][location], (expected - mean[0]) / scale[0], atol=1e-6)


def test_missing_asset_bar_invalidates_context_without_compressing_global_calendar():
    config = tiny_config()
    featured, _ = build_post_open_frame(prices(), groups=config.feature_groups)
    featured["date"] = pd.to_datetime(featured.date, utc=True)
    fold = calendar_folds(featured, config)[0]
    calendar = pd.DatetimeIndex(featured.date.unique()).sort_values()
    hole_index = 32
    missing = calendar[hole_index]
    featured = featured.loc[~((featured.ticker == "A") & (featured.date == missing))].copy()
    parts, _ = prepare_common_fold(featured, fold, config)
    train = parts["train"]
    # A shortened per-ticker sequence must not span the missing global session.
    invalid = set(calendar[hole_index:hole_index + config.context_len])
    a_dates = set(pd.to_datetime(train["aligned"].loc[train["aligned"].ticker == "A", "date"], utc=True))
    b_dates = set(pd.to_datetime(train["aligned"].loc[train["aligned"].ticker == "B", "date"], utc=True))
    assert not a_dates & invalid
    assert invalid <= b_dates
    assert missing in train["calendar"]
    # Finance refuses the internal quote hole rather than inventing a daily
    # return across several sessions or filling a non-executable price.
    with pytest.raises(ValueError, match="internal missing price quotes"):
        PostOpenReturnPanel(train["prices"], protocol="overnight", tickers=("A", "B"),
                            calendar=train["calendar"], signal_frame=train["aligned"])


def test_all_asset_missing_open_keeps_original_price_session_and_rejects_internal_hole():
    config = tiny_config()
    raw = prices()
    calendar = pd.DatetimeIndex(pd.to_datetime(raw.date, utc=True)).unique().sort_values()
    missing = calendar[32]
    raw.loc[pd.to_datetime(raw.date, utc=True) == missing, "open"] = np.nan
    featured, _ = build_post_open_frame(raw, groups=config.feature_groups)
    featured["date"] = pd.to_datetime(featured.date, utc=True)
    # The feature constructor omits this session for every ticker. Its absence
    # must not redefine the price calendar, fold boundaries or return horizon.
    assert not (featured.date == missing).any()
    fold = calendar_folds(raw, config)[0]
    parts, preprocessing = prepare_common_fold(featured, fold, config, price_frame=raw)
    train = parts["train"]
    assert preprocessing["calendar_source"] == "original_prices"
    assert missing in train["calendar"]
    assert (pd.to_datetime(train["prices"].date, utc=True) == missing).sum() == 2
    invalid = set(calendar[32:32 + config.context_len])
    assert not set(pd.to_datetime(train["aligned"].date, utc=True)) & invalid
    for protocol in ("intraday", "overnight"):
        with pytest.raises(ValueError, match="internal missing price quotes"):
            PostOpenReturnPanel(train["prices"], protocol=protocol, tickers=("A", "B"),
                                calendar=train["calendar"], signal_frame=train["aligned"])


def exposure_records(*, include_flat=False):
    q = np.array([[0.25, -0.1, 0.4], [0.2, 0.3, -0.15]])
    returns = np.array([[0.02, -0.01, 0.005], [-0.005, 0.015, -0.01]])
    dates = pd.bdate_range("2020-01-02", periods=3)
    records = []
    scales = {"low": 1.0, "high": 2.0, **({"flat": 0.0} if include_flat else {})}
    for candidate, scale in scales.items():
        for asset, ticker in enumerate(("A", "B")):
            for session, date in enumerate(dates):
                position = scale * q[asset, session]
                turnover = 2 * abs(position)
                cost = 0.0005 * turnover
                records.append({"candidate": candidate, "date": date, "ticker": ticker,
                    "position": position, "net_return": position * returns[asset, session] - cost,
                    "turnover": turnover, "cost": cost})
    return pd.DataFrame(records)


def test_common_exposure_scales_net_returns_and_costs_to_lowest_without_leverage():
    config = tiny_config()
    source = exposure_records()
    controls = {row["candidate"]: row for row in common_exposure_comparison(source, config)}
    target_exposure = source.loc[source.candidate == "low", "position"].abs().mean()
    assert controls["low"]["scale"] == pytest.approx(1.0)
    assert controls["high"]["scale"] == pytest.approx(0.5)
    for candidate, row in controls.items():
        assert 0 <= row["scale"] <= 1
        assert row["common_mean_exposure"] == pytest.approx(target_exposure)
        assert not row["degenerate_cash_control"]
        original = source[source.candidate == candidate]
        daily_net = original.groupby("date").net_return.mean().to_numpy()
        expected_return = np.prod(1 + row["scale"] * daily_net) - 1
        assert row["net_return"] == pytest.approx(expected_return)
        assert row["net_pnl"] == pytest.approx(config.initial_capital * expected_return)
        assert row["cost_return_sum"] == pytest.approx(original.groupby("date").cost.mean().sum() * row["scale"])
        assert row["turnover"] == pytest.approx(original.groupby("date").turnover.mean().sum() * row["scale"])
        assert row["matching_scope"] == "mean_only_not_exposure_timing"
        assert row["selection_use"] == "outer_reporting_only"
    # Here the high path is exactly two times the low path, including costs.
    # De-leveraging must make their net paths coincide, not charge full costs.
    assert controls["high"]["net_return"] == pytest.approx(controls["low"]["net_return"])
    assert controls["high"]["cost_return_sum"] == pytest.approx(controls["low"]["cost_return_sum"])


def test_common_exposure_flags_degenerate_cash_when_any_candidate_is_flat():
    controls = common_exposure_comparison(exposure_records(include_flat=True), tiny_config())
    assert len(controls) == 3
    for row in controls:
        assert row["degenerate_cash_control"]
        assert row["common_mean_exposure"] == row["scale"] == 0.0
        assert row["net_return"] == row["net_pnl"] == row["turnover"] == row["cost_return_sum"] == 0.0
        assert row["net_sharpe"] == row["regularized_sharpe"] == 0.0


@pytest.mark.parametrize("method", ["intraday_return", "forward_return", "volatility_position"])
def test_label_dependencies_are_purged_at_each_global_partition_boundary(method):
    config = tiny_config()
    featured, _ = build_post_open_frame(prices(), groups=config.feature_groups)
    featured["date"] = pd.to_datetime(featured.date, utc=True)
    fold = calendar_folds(featured, config)[0]
    parts, _ = prepare_common_fold(featured, fold, config)
    for phase, part in parts.items():
        labels = _supervision(featured, part, method, fold, phase, config)
        selected = labels._label_known.to_numpy(dtype=bool)
        assert selected.any()
        end = pd.to_datetime(labels.loc[selected, "label_end_date"], utc=True)
        assert (end <= pd.Timestamp(fold[f"{phase}_end"])).all()
        if method == "intraday_return":
            assert selected.all()
        else:
            tail = labels.groupby("ticker", sort=False).tail(config.label_horizon - 1)
            assert not tail._label_known.any()


@pytest.fixture(scope="module")
def completed_benchmark(tmp_path_factory):
    directory = tmp_path_factory.mktemp("label-loss-integration")
    frame = prices()
    config = tiny_config()
    report = run_label_loss_benchmark(frame, ["A", "B"], directory, config=config, plots=False, progress=False)
    return directory, frame, config, report


def test_real_gru_end_to_end_all_candidates_paths_and_ce_checkpoint_reuse(completed_benchmark):
    directory, frame, config, report = completed_benchmark
    assert report["completed_fits"] == 11
    assert report["completed_evaluation_paths"] == 42
    assert not report["final_holdout_opened"] and not report["selection_performed_on_outer"]
    results = pd.read_csv(directory / "results.csv")
    assert len(results) == 42 and results.fit_directory.nunique() == 11
    ce = results[results.family == "cross_entropy"]
    assert len(ce) == 18 and ce.fit_directory.nunique() == 3
    assert ce.groupby("fit_directory").protocol.nunique().eq(2).all()
    for fit_directory, group in results.groupby("fit_directory"):
        root = directory / fit_directory
        assert (root / "checkpoint.pt").is_file()
        probabilities = pd.read_parquet(root / "probabilities.parquet")
        assert not probabilities.duplicated(["date", "ticker"]).any()
        np.testing.assert_allclose(probabilities[["p_short", "p_flat", "p_long"]].sum(axis=1), 1, atol=1e-6)
        diagnostics = json.loads((root / "learning_diagnostics.json").read_text())
        assert diagnostics["optimizer_updates"] == 1
        assert diagnostics["checkpoint_selection"] == "inner_validation_total_loss"
        classes = pd.read_csv(root / "classification.csv")
        assert set(classes.label_method) == set(config.label_methods)
        assert set(classes.phase) == {"train", "inner", "outer"}
        assert classes.groupby("label_method").phase.nunique().eq(3).all()
        for row in group.to_dict("records"):
            positions = pd.read_parquet(root / f"{row['artifact']}-positions.parquet")
            assert not positions.empty
            assert set(positions.ticker) == {"A", "B"}
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["counts"]["fits"] == 11
    assert not metadata["final_holdout_opened"]
    assert len(json.loads((directory / "fold-0" / "label_oracles.json").read_text())) == 6
    assert (directory / "oppositions.csv").is_file()
    aggregate = pd.read_csv(directory / "summary.csv")
    assert len(aggregate) == 42
    assert "net_return_mean" in aggregate and "mean_abs_position_mean" in aggregate
    classification = pd.read_csv(directory / "classification.csv")
    assert len(classification) == 11 * 3 * 3
    assert set(classification.phase) == {"train", "inner", "outer"}
    controls = pd.read_csv(directory / "common-exposure.csv")
    assert len(controls) == 42
    assert controls.scale.between(0, 1).all()
    assert set(controls.selection_use) == {"outer_reporting_only"}


def test_completed_resume_reuses_all_fits_and_ignores_closed_holdout(completed_benchmark, monkeypatch):
    directory, frame, config, original = completed_benchmark
    import trading_system.training.label_loss_trainer as trainer

    def forbidden(*args, **kwargs):
        raise AssertionError("Resume attempted to retrain a completed task.")

    monkeypatch.setattr(trainer, "fit_label_loss_model", forbidden)
    poison = frame.groupby("ticker", sort=False).tail(1).copy()
    poison["date"] = pd.Timestamp(config.holdout_start)
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        poison[column] = np.nan
    with_holdout = pd.concat([frame, poison], ignore_index=True)
    resumed = run_label_loss_benchmark(with_holdout, ["A", "B"], directory,
        config=config, resume=True, plots=False, progress=False)
    assert resumed == original
    for label_path in (directory / "fold-0").glob("*-labels.parquet"):
        labels = pd.read_parquet(label_path)
        assert (pd.to_datetime(labels.date, utc=True) < pd.to_datetime(config.holdout_start, utc=True)).all()


def test_resume_rejects_corrupted_completed_fit_before_training(completed_benchmark, tmp_path, monkeypatch):
    original, frame, config, _ = completed_benchmark
    destination = tmp_path / "corrupt"
    shutil.copytree(original, destination)
    checkpoint = next(destination.glob("fold-0/seed-1/*/checkpoint.pt"))
    checkpoint.write_bytes(b"corrupted checkpoint")
    import trading_system.training.label_loss_trainer as trainer

    def forbidden(*args, **kwargs):
        raise AssertionError("Corrupted completed tasks must reject before training.")

    monkeypatch.setattr(trainer, "fit_label_loss_model", forbidden)
    with pytest.raises(ValueError, match="integrity|size|checksum|hash|changed"):
        run_label_loss_benchmark(frame, ["A", "B"], destination,
            config=config, resume=True, plots=False, progress=False)


def test_nonempty_output_without_resume_and_recipe_mismatch_reject(completed_benchmark):
    directory, frame, config, _ = completed_benchmark
    with pytest.raises(FileExistsError, match="resume"):
        run_label_loss_benchmark(frame, ["A", "B"], directory, config=config, plots=False, progress=False)
    with pytest.raises(ValueError, match="recipe differ"):
        run_label_loss_benchmark(frame, ["A", "B"], directory,
            config=replace(config, hybrid_ce_weight=0.4), resume=True, plots=False, progress=False)


def test_calendar_fold_input_cannot_include_closed_holdout():
    frame = prices()
    frame.loc[frame.index[-1], "date"] = pd.Timestamp("2023-06-22")
    with pytest.raises(ValueError, match="closed-holdout"):
        calendar_folds(frame, tiny_config())
