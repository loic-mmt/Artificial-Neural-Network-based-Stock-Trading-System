from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from trading_system.data.purged_cv import PurgedSplit, expanding_calendar_folds, purge_intervals
from trading_system.experiments.config import ExperimentConfig
from trading_system.experiments.runner import _prepare_splits, run_validation_experiment
from trading_system.experiments.purged_search import run_purged_cv
from trading_system.models.specs import ModelSelection
from trading_system.training.financial_loss import FinancialLossConfig


def prices(n=280, ticker="A"):
    rng = np.random.default_rng(7)
    p = 100 * np.exp(np.cumsum(rng.normal(.002, .012, n)))
    return pd.DataFrame({"ticker": ticker, "date": pd.date_range("2020-01-01", periods=n),
                         "open": p, "high": p + 1, "low": p - 1, "close": p, "adj_close": p,
                         "volume": rng.integers(100, 10000, n).astype(float), "sector": "Technology"})


def config(**kwargs):
    return ExperimentConfig(context_len=3, label_mode="triple_barrier", triple_barrier_max_holding=10,
                            triple_barrier_volatility_window=5, triple_barrier_profit_barrier=.01,
                            triple_barrier_stop_barrier=.01, triple_barrier_cost_bps=0., device="cpu",
                            model=ModelSelection("manual_ann", {"epochs": 1, "hidden_size": 4}), **kwargs)


def test_closed_interval_purge_unknowns_embargo_and_date_not_row_count():
    dates = pd.date_range("2020-01-01", periods=10)
    starts = pd.DatetimeIndex([*dates, dates[6]])
    ends = pd.Series(starts)
    ends.iloc[0] = dates[3]  # exact equality with validation entry is purged
    ends.iloc[1] = pd.NaT
    keep, report = purge_intervals(starts, ends, np.array([0, 1, 2, 6, 7, 8, 9, 10]), np.array([3, 4, 5]), embargo_bars=2)
    assert keep.tolist() == [2, 8, 9]
    assert report == dict(candidates=8, kept=3, unknown=1, overlap=1, embargo=3)


def test_purge_matches_brute_force_on_random_variable_event_intervals():
    rng = np.random.default_rng(8)
    starts = pd.date_range("2020-01-01", periods=100)
    ends = starts + pd.to_timedelta(rng.integers(0, 8, 100), unit="D")
    validation = np.array([10, 11, 40, 50, 51, 80])
    train = np.setdiff1d(np.arange(100), validation)
    expected = [i for i in train if not any(starts[i] <= ends[j] and ends[i] >= starts[j] for j in validation)]
    keep, _ = purge_intervals(starts, ends, train, validation)
    assert keep.tolist() == expected


def test_expanding_folds_are_global_nested_and_final_holdout_is_separate():
    data = pd.concat([prices(), prices(ticker="B")], ignore_index=True)
    folds, final = expanding_calendar_folds(data, n_splits=3, embargo_bars=5)
    assert len(folds) == 3
    previous_end = None
    for fold in folds:
        split = fold["split"]
        assert pd.Timestamp(split.validation_start) < pd.Timestamp(split.test_start) <= pd.Timestamp(fold["end"]) < pd.Timestamp(final.test_start)
        if previous_end is not None:
            assert pd.Timestamp(split.test_start) > previous_end
        previous_end = pd.Timestamp(fold["end"])
    assert final.embargo_bars == 5


@pytest.mark.parametrize("options", [{"n_splits": 1}, {"n_splits": 1000}, {"initial_train_fraction": 0},
                                     {"inner_val_fraction": np.nan}, {"gap_bars": -1}, {"embargo_bars": True}])
def test_invalid_folds_fail_explicitly(options):
    with pytest.raises(ValueError):
        expanding_calendar_folds(prices(), **options)


def test_actual_first_touch_keeps_resolved_events_instead_of_fixed_horizon_purge():
    data = prices()
    split = PurgedSplit(str(data.date.iloc[150]), str(data.date.iloc[220]))
    train, _, _, _, _ = _prepare_splits(data, config(purged_split=split))
    near = train[train.date.between(data.date.iloc[142], data.date.iloc[148])]
    assert near._label_known.any()  # fixed max-holding-tail exclusion loses these
    known = train[train._label_known]
    assert (known._information_end < pd.Timestamp(split.validation_start)).all()
    assert train.attrs["purging"]["train"]["overlap"] >= 1


def test_gap_rows_stay_context_but_do_not_fit_or_score():
    data = prices()
    split = PurgedSplit(str(data.date.iloc[150]), str(data.date.iloc[220]), gap_bars=4, embargo_bars=3)
    train, val, _, _, _ = _prepare_splits(data, config(purged_split=split))
    assert len(train.tail(4)) == 4
    assert train.tail(4)._cv_gap.all() and not train.tail(4)._label_known.any()
    assert val.tail(4)._cv_gap.all() and not val.tail(4)._label_known.any()
    assert (train.loc[train._label_known, "_information_end"] < pd.to_datetime(train.date.iloc[-4], utc=True)).all()
    assert train.attrs["purging"]["train"]["embargo"] == 0


@pytest.mark.parametrize("method", ["breakout", "forward_return", "volatility_position", "triple_barrier"])
def test_all_label_methods_purge_and_keep_finite_training(method):
    data = prices()
    cfg = replace(config(), label_mode=method, purged_split=PurgedSplit(str(data.date.iloc[150]), str(data.date.iloc[220])))
    result = run_validation_experiment(data, cfg)
    assert result.bundle.purging_state is not None
    assert np.isfinite(result.val_probabilities).all()


def test_outer_period_perturbation_cannot_change_inner_model_or_preprocessing():
    data = prices()
    cfg = config(purged_split=PurgedSplit(str(data.date.iloc[150]), str(data.date.iloc[220]), gap_bars=3))
    original = run_validation_experiment(data, cfg)
    changed = data.copy()
    changed.loc[220:, ["open", "high", "low", "close", "adj_close"]] *= 100
    modified = run_validation_experiment(changed, cfg)
    np.testing.assert_array_equal(original.val_probabilities, modified.val_probabilities)
    np.testing.assert_array_equal(original.bundle.scaler.mean_, modified.bundle.scaler.mean_)


@pytest.mark.parametrize("losses", [None, [FinancialLossConfig("pnl")], [FinancialLossConfig("cross_entropy")]])
def test_cv_real_end_to_end_seals_final_and_persists_purge(tmp_path, losses):
    report = run_purged_cv(prices(), config(), {"manual_ann": [{"epochs": 1, "hidden_size": 4}]},
                           [1], tmp_path / "cv", n_splits=2, loss_configs=losses,
                           selection_metric="net_return" if losses else "macro_f1", gap_bars=2, fail_fast=True)
    assert report["final_test"] == []
    assert report["metadata"]["selected"] is not None
    assert len(report["folds"]) == 2
    assert all(row["purging"]["train"]["kept_after_gap"] > 0 for row in report["folds"])
    assert (tmp_path / "cv" / "selection.json").exists()


def test_cv_selects_only_complete_candidates_before_final_test(tmp_path, monkeypatch):
    import trading_system.experiments.purged_search as search
    original_fit = search.run_validation_experiment
    original_eval = search.evaluate_experiment_test
    target = tmp_path / "cv"
    final_calls = []
    progress_calls = []
    progress_updates = []

    class FakeProgress:
        def __init__(self, items):
            self.items = list(items)

        def __iter__(self):
            return iter(self.items)

        def set_postfix_str(self, *args, **kwargs):
            progress_updates.append(args[0])
            return None

    def progress(items, **kwargs):
        wrapped = FakeProgress(items)
        progress_calls.append((len(wrapped.items), kwargs))
        return wrapped

    def fit(frame, cfg, **kwargs):
        if cfg.model.parameters["hidden_size"] == 5:
            raise ValueError("Synthetic candidate failure")
        return original_fit(frame, cfg, **kwargs)
    def evaluate(frame, validation):
        if len(frame) == 280:
            selected = json.loads((target / "selection.json").read_text())["metadata"]["selected"]
            assert selected
            assert validation.config.model.parameters["hidden_size"] == 4
            final_calls.append(1)
        return original_eval(frame, validation)
    monkeypatch.setattr(search, "run_validation_experiment", fit)
    monkeypatch.setattr(search, "evaluate_experiment_test", evaluate)
    monkeypatch.setattr(search, "tqdm", progress)
    result = search.run_purged_cv(prices(), config(), {"manual_ann": [{"epochs": 1, "hidden_size": 4}, {"epochs": 1, "hidden_size": 5}]},
                                  [1, 2], target, n_splits=2, final_test=True)
    assert progress_calls == [(8, {"desc": "CV trainings", "unit": "fit", "dynamic_ncols": True, "disable": None})]
    assert any("tr=" in update and "va=" in update for update in progress_updates)
    assert len(final_calls) == 2
    assert len(result["final_test"]) == 2
    assert sum(row["complete"] for row in result["summary"]) == 1


def test_cv_cli_is_opt_in_and_dispatches_both_comparators(tmp_path):
    from trading_system.pipelines.compare_models import main as model_main, build_parser
    from trading_system.pipelines.compare_losses import main as loss_main
    assert build_parser().parse_args([]).cv_folds is None
    source = tmp_path / "data.parquet"
    prices().to_parquet(source)
    arguments = ["--data", str(source), "--preset", "multi_ticker_long_short", "--models", "manual_ann", "--seeds", "1",
                 "--model-parameter-sets", '{"manual_ann":[{"epochs":1,"hidden_size":4}]}', "--cv-folds", "2", "--fail-fast"]
    for fn, name, extra in ((model_main, "models", []), (loss_main, "losses", ["--losses", "sharpe"])):
        result = fn([*arguments, *extra, "--output-dir", str(tmp_path / name)])
        assert result["metadata"]["protocol"] == "nested_expanding_purged_cv"
        assert result["final_test"] == []


def test_final_holdout_changes_cannot_change_fold_scores_or_selection(tmp_path):
    data = prices()
    changed = data.copy()
    _, final = expanding_calendar_folds(data, n_splits=2)
    mask = pd.to_datetime(changed.date, utc=True) >= pd.Timestamp(final.test_start)
    changed.loc[mask, ["open", "high", "low", "close", "adj_close"]] *= 7
    reports = [run_purged_cv(frame, config(), {"manual_ann": [{"epochs": 1, "hidden_size": 4}]}, [1],
                             tmp_path / str(i), n_splits=2, save_artifacts=False, fail_fast=True)
               for i, frame in enumerate((data, changed))]
    assert [row["score"] for row in reports[0]["folds"]] == [row["score"] for row in reports[1]["folds"]]
    assert reports[0]["metadata"]["selected"] == reports[1]["metadata"]["selected"]


def test_multiticker_purged_cv_uses_same_calendar_boundaries(tmp_path):
    data = pd.concat([prices(), prices(ticker="B").iloc[3:]], ignore_index=True)
    report = run_purged_cv(data, config(universe="multi"), {"manual_ann": [{"epochs": 1, "hidden_size": 4}]},
                           [1], tmp_path / "multi", n_splits=2, fail_fast=True)
    for row in report["folds"]:
        assert row["status"] == "ok"
        assert row["purging"]["train"]["overlap"] >= 2


@pytest.mark.parametrize("objective", ["cross_entropy", "sharpe"])
def test_torch_expanded_fracdiff_purged_cv_roundtrip_and_final(tmp_path, objective):
    pytest.importorskip("torch")
    from trading_system.features.fracdiff import FracDiffConfig
    from trading_system.training.overfitting import OverfittingControlConfig
    from trading_system.experiments.position_objectives import load_position_artifact
    cfg = replace(config(), feature_set="expanded", fracdiff=FracDiffConfig(order=.5, threshold=.05),
                  overfitting_control=OverfittingControlConfig(max_features=5))
    report = run_purged_cv(prices(400), cfg, {"gru": [{"epochs": 1, "hidden_size": 4}]}, [1],
                           tmp_path / objective, n_splits=2, gap_bars=2, final_test=True,
                           loss_configs=[FinancialLossConfig(objective)], selection_metric="net_return", fail_fast=True)
    assert len(report["final_test"]) == 1
    loaded = load_position_artifact(tmp_path / objective / "final" / "seed-1")
    assert loaded.config.purged_split is not None
    assert loaded.bundle.purging_state == report["final_test"][0]["purging"]
    assert len(loaded.bundle.feature_columns) <= 5
