"""Evaluation-only contracts: exact keys, equal slots and no label leakage."""

import json

import numpy as np
import pandas as pd
import pytest

from trading_system.analysis.label_loss_evaluation import (
    classification_metrics,
    compare_opposite_positions,
    decode_probabilities,
    evaluate_positions,
    plot_evaluation,
    plot_learning_trace,
    plot_oppositions,
)
from trading_system.training.financial_loss import FinancialLossConfig


class Panel:
    """Minimal post-open panel with a terminal close on the last active row."""

    def __init__(self, returns, protocol="overnight", indices=None):
        self.returns = np.asarray(returns, dtype=float)
        self.indices = (np.arange(self.returns.size).reshape(self.returns.shape)
                        if indices is None else np.asarray(indices))
        self.rows = int(self.indices.max()) + 1
        self.dates = pd.date_range("2020-01-01", periods=self.returns.shape[1], tz="UTC")
        self.tickers = tuple("ABC"[:len(self.returns)])
        self.protocol = protocol

    def path(self, positions, config):
        target = np.asarray(positions, dtype=float)
        if target.shape != (self.rows,) or not np.isfinite(target).all() or (abs(target) > 1).any():
            raise ValueError("Invalid positions")
        executed = np.zeros_like(self.returns)
        observed = self.indices >= 0
        executed[observed] = target[self.indices[observed]]
        if self.protocol == "intraday":
            delta = executed.copy()
            turnover = 2 * abs(executed)
        else:
            delta = np.diff(executed, axis=1, prepend=0.)
            turnover = abs(delta)
            turnover[:, -1] += abs(executed[:, -1])
        costs = config.cost_bps * 1e-4 * turnover
        net = executed * self.returns - costs
        return net.mean(axis=0), executed, delta, turnover, costs


def test_three_decoders_and_exact_zero():
    probabilities = [[.1, .3, .6], [.6, .3, .1], [.2, .6, .2], [.5, 0., .5]]
    np.testing.assert_allclose(decode_probabilities(probabilities), [.5, -.5, 0., 0.])
    np.testing.assert_array_equal(decode_probabilities(probabilities, "sign"), [1., -1., 0., 0.])
    np.testing.assert_array_equal(decode_probabilities(probabilities, "argmax"), [1., -1., 0., -1.])
    with pytest.raises(ValueError, match="decoder"):
        decode_probabilities(probabilities, "different")
    with pytest.raises(ValueError, match="sum to one"):
        decode_probabilities([[.1, .2, .3]])


def test_classification_unknowns_absent_classes_and_train_majority():
    scores = classification_metrics(
        [2, 2, 1, 999], [[0, 0, 1], [.1, .1, .8], [.1, .8, .1], [1, 0, 0]],
        np.array([True, True, True, False]), majority_class=0,
    )
    assert scores["accuracy"] == 1
    assert scores["balanced_accuracy"] == pytest.approx(2 / 3)
    assert scores["macro_f1"] == pytest.approx(2 / 3)
    assert scores["balanced_accuracy_fixed3"] == pytest.approx(2 / 3)
    assert scores["macro_f1_fixed3"] == pytest.approx(2 / 3)
    assert scores["balanced_accuracy_present_classes"] == 1.
    assert scores["macro_f1_present_classes"] == 1.
    assert scores["n_present_classes"] == 2
    assert scores["n_unknown"] == 1
    assert scores["confusion_matrix"] == [[0, 0, 0], [0, 1, 0], [0, 0, 2]]
    assert scores["majority_baseline"]["accuracy"] == 0
    assert scores["majority_baseline"]["class_source"] == "train"
    assert scores["cross_entropy"] == pytest.approx(-2 * np.log(.8) / 3)
    json.dumps(scores, allow_nan=False)


def test_classification_empty_known_is_not_flat_or_nan():
    scores = classification_metrics([None], [[.1, .8, .1]], np.array([False]))
    assert scores["n_known"] == 0
    assert scores["accuracy"] is None and scores["cross_entropy"] is None
    assert scores["n_present_classes"] == 0
    assert scores["balanced_accuracy_present_classes"] is None
    assert scores["majority_baseline"]["class_id"] is None
    json.dumps(scores, allow_nan=False)
    with pytest.raises(ValueError, match="boolean"):
        classification_metrics([1], [[0, 1, 0]], [1])
    with pytest.raises(ValueError, match="class IDs"):
        classification_metrics([3], [[0, 1, 0]], np.array([True]))


@pytest.mark.parametrize("protocol", ["intraday", "overnight"])
def test_portfolio_ticker_and_side_attribution_reconcile(protocol):
    panel = Panel([[.01, -.02, .03, .01], [-.03, .01, -.02, .04]], protocol)
    positions = np.array([.5, .3, -.4, 0., -.5, -.2, .3, .4])
    config = FinancialLossConfig("combined", cost_bps=5)
    result = evaluate_positions(panel, positions, config)
    metrics = result["metrics"]
    assert sum(record["portfolio_pnl_contribution"] for record in result["per_ticker"]) == pytest.approx(metrics["net_pnl"])
    assert metrics["long_pnl_contribution"] + metrics["short_pnl_contribution"] == pytest.approx(metrics["net_pnl"])
    assert sum(record["pnl_contribution"] for record in result["position_records"]) == pytest.approx(metrics["net_pnl"])
    assert sum(record["long_pnl_contribution"] + record["short_pnl_contribution"]
               for record in result["daily_paths"]) == pytest.approx(metrics["net_pnl"])
    assert result["daily_paths"][-1]["equity"] == pytest.approx(10000 + metrics["net_pnl"])
    matched = result["exposure_controls"]["always_long_mean_exposure_matched"]
    assert matched["mean_abs_position"] == pytest.approx(metrics["mean_abs_position"])
    assert result["exposure_controls"]["cash"]["net_pnl"] == 0
    assert result["daily_paths"][-1]["baseline_equity"] == pytest.approx(10000 + result["exposure_controls"]["always_long"]["net_pnl"])
    assert result["daily_paths"][-1]["exposure_matched_baseline_equity"] == pytest.approx(10000 + matched["net_pnl"])
    assert all(row["cash_equity"] == 10000 for row in result["daily_paths"])
    if protocol == "intraday":
        assert all(trade["duration_sessions"] == 1 for trade in result["trades"])
    json.dumps(result, allow_nan=False)


def test_mean_exposure_control_does_not_scale_model_or_add_leverage():
    panel = Panel([[.01, .02, .03]], "intraday")
    config = FinancialLossConfig(cost_bps=5)
    result = evaluate_positions(panel, np.full(3, .25), config)
    controls = result["exposure_controls"]
    assert controls["matching"]["scale"] == .25
    assert controls["always_long_mean_exposure_matched"]["turnover"] == pytest.approx(1.5)
    assert controls["always_long"]["turnover"] == pytest.approx(6)
    assert result["metrics"]["outperformance_vs_exposure_matched_long"] == pytest.approx(0)


def test_trade_episodes_use_direction_not_every_continuous_resize():
    panel = Panel([[.01, .02, .03, -.01, -.02]], "overnight")
    result = evaluate_positions(panel, [.2, .4, .3, -.5, -.2], FinancialLossConfig(cost_bps=0))
    assert [trade["duration_sessions"] for trade in result["trades"]] == [3, 2]
    assert [trade["side"] for trade in result["trades"]] == ["Long", "Short"]


def test_sparse_missing_signal_keeps_exit_cost_and_exact_date_keys():
    panel = Panel([[.01, .02, .03]], indices=[[0, -1, 1]])
    keys = pd.DataFrame({"date": [panel.dates[0], panel.dates[2]], "ticker": ["A", "A"], "candidate": ["x", "x"]})
    result = evaluate_positions(panel, [1, -1], FinancialLossConfig(cost_bps=5), keys=keys)
    records = result["position_records"]
    assert [record["row_index"] for record in records] == [0, None, 1]
    assert records[1]["position"] == 0 and records[1]["cost"] == .0005
    assert records[1]["candidate"] == "x"
    assert records[2]["date"] == panel.dates[2].isoformat()
    with pytest.raises(ValueError, match="ticker"):
        evaluate_positions(panel, [1, -1], FinancialLossConfig(), keys=keys.assign(ticker="B"))
    with pytest.raises(ValueError, match="date"):
        evaluate_positions(panel, [1, -1], FinancialLossConfig(), keys=keys.assign(date=panel.dates[0]))


def test_scalar_identity_keys_propagate_to_all_tables():
    panel = Panel([[.01, .02, .03]])
    identity = {"candidate": "hybrid_forward_overnight", "fold": 1, "seed": 7,
                "protocol": "overnight", "decoder": "continuous", "label_method": None}
    result = evaluate_positions(panel, [.2, -.4, .3], FinancialLossConfig(), keys=identity)
    for table in ("per_ticker", "daily_paths", "position_records", "trades"):
        assert all(record["candidate"] == identity["candidate"] and record["seed"] == 7
                   for record in result[table])
    json.dumps(result, allow_nan=False)


def test_insolvent_individual_short_slot_does_not_hide_solvent_portfolio():
    panel = Panel([[2., 0.], [0., 0.], [0., 0.]], "intraday")
    result = evaluate_positions(panel, [-1, 0, 0, 0, 0, 0], FinancialLossConfig(cost_bps=0))
    assert result["metrics"]["net_return"] == pytest.approx(-2 / 3)
    assert result["per_ticker"][0]["standalone_slot_insolvent"]
    assert result["per_ticker"][0]["net_return"] is None
    assert result["trades"][0]["net_return"] is None
    assert sum(record["portfolio_pnl_contribution"] for record in result["per_ticker"]) == pytest.approx(result["metrics"]["net_pnl"])


def comparison_frame():
    dates = pd.date_range("2020-01-01", periods=5, tz="UTC")
    records = []
    for candidate, positions in (("CE", [1, 1, 0, -1, -1]), ("financial", [-.5, -.5, 1, 1, -1])):
        for index, position in enumerate(positions):
            records.append({"candidate": candidate, "fold": 0, "seed": 7, "protocol": "overnight",
                            "decoder": "sign", "date": dates[index], "ticker": "A", "position": position,
                            "gross_return": position * .01, "net_return": position * .01 - .001,
                            "allocation_weight": 1.})
    return pd.DataFrame(records)


def test_opposition_excludes_flat_and_reports_consecutive_direction_episodes():
    result = compare_opposite_positions(comparison_frame().sample(frac=1, random_state=4))
    summary = result["summary"][0]
    assert summary["both_nonflat"] == 4 and summary["opposed_rows"] == 3
    assert summary["opposition_rate_both_nonflat"] == .75
    assert [episode["duration_sessions"] for episode in result["episodes"]] == [2, 1]
    assert summary["opposed_net_return_contribution_a"] == pytest.approx(.01 - .003)
    assert summary["opposed_net_return_contribution_b"] == pytest.approx(0 - .003)


def test_comparisons_do_not_mix_seeds_or_intersect_missing_calendars():
    work = comparison_frame()
    assert compare_opposite_positions(work.assign(seed=np.where(work.candidate.eq("CE"), 1, 7)))["summary"] == []
    with pytest.raises(ValueError, match="same ticker/session calendar"):
        compare_opposite_positions(work.drop(index=0))
    with pytest.raises(ValueError, match="Duplicate"):
        compare_opposite_positions(pd.concat([work, work.iloc[:1]]))


def test_missing_global_session_breaks_opposition_episode():
    work = comparison_frame()
    # The A rows skip the middle session, which B retains in the global calendar.
    a = work[work.date.isin(work.date.unique()[[0, 2]])].assign(position=lambda rows: np.where(rows.candidate.eq("CE"), 1., -1.))
    b = work[work.date.eq(work.date.unique()[1])].assign(ticker="B", position=0.)
    result = compare_opposite_positions(pd.concat([a, b]))
    assert [episode["duration_sessions"] for episode in result["episodes"]] == [1, 1]


def test_optional_plots_are_bounded_and_exist(tmp_path):
    panel = Panel([[.01, -.01, .02]], "intraday")
    result = evaluate_positions(panel, [1, -1, 0], FinancialLossConfig())
    positions = pd.DataFrame(result["position_records"]).assign(candidate="example", protocol="intraday", decoder="sign", fold=0, seed=1)
    prices = pd.DataFrame({"ticker": "A", "date": panel.dates, "close": [100, 99, 101]})
    paths = plot_evaluation(tmp_path, result["daily_paths"], positions, prices)
    assert len(paths) == 2
    assert all(pd.io.common.file_exists(path) for path in paths)


def test_learning_trace_plots_recorded_components(tmp_path):
    trace = [{"epoch": epoch, "train": {"total_loss": 1 / epoch, "ce_loss": .8 / epoch,
                                        "financial_loss": -.2 * epoch},
              "validation": {"total_loss": 2 / epoch, "ce_loss": 1 / epoch, "financial_loss": -.1 * epoch},
              "weighted_ce_gradient_norm": 0. if epoch == 1 else .01,
              "weighted_financial_gradient_norm": .02,
              "combined_gradient_norm_preclip": .03} for epoch in (1, 2)]
    paths = plot_learning_trace(tmp_path, trace)
    assert len(paths) == 2
    assert all(pd.io.common.file_exists(path) for path in paths)
    assert plot_learning_trace(tmp_path, []) == []


def test_opposition_plot_stacks_models_without_mixing_context(tmp_path):
    positions = comparison_frame()
    prices = positions[["date", "ticker"]].drop_duplicates().assign(open=[99, 100, 101, 102, 103], close=[100, 101, 102, 103, 104])
    paths = plot_oppositions(tmp_path, positions, prices, max_tickers=1)
    assert len(paths) == 1 and pd.io.common.file_exists(paths[0])
    with pytest.raises(ValueError, match="common fold/seed/protocol/decoder"):
        plot_oppositions(tmp_path, positions.assign(seed=lambda rows: np.where(rows.candidate.eq("CE"), 1, 7)), prices)


def test_plot_price_contract_uses_adjusted_open_for_overnight_and_explicit_intraday_caption():
    from trading_system.analysis.label_loss_evaluation import _plot_price_contract
    prices = pd.DataFrame({"open": [100, 120], "close": [110, 125], "adj_close": [55, 125]})
    overnight, label, caption = _plot_price_contract(prices, "overnight")
    np.testing.assert_allclose(overnight._plot_price, [50, 120])
    assert label == "Adjusted open" and "open J+1" in caption
    preferred, _, _ = _plot_price_contract(prices.assign(adj_open_target=[49, 119]), "overnight")
    np.testing.assert_allclose(preferred._plot_price, [49, 119])
    intraday, label, caption = _plot_price_contract(prices, "intraday")
    np.testing.assert_allclose(intraday._plot_price, [110, 125])
    assert label == "close" and "not close to close" in caption
    with pytest.raises(ValueError, match="not closing prices alone"):
        _plot_price_contract(prices.drop(columns="open"), "overnight")


def test_ce_learning_plot_omits_disabled_financial_axis(tmp_path, monkeypatch):
    import matplotlib.figure
    captured = []
    original = matplotlib.figure.Figure.savefig

    def save(figure, *args, **kwargs):
        captured.append([axis.get_title() for axis in figure.axes])
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", save)
    trace = [{"epoch": 1, "train": {"total_loss": 1., "ce_loss": 1., "financial_loss": None},
              "validation": {"total_loss": 1.1, "ce_loss": 1.1, "financial_loss": None}}]
    paths = plot_learning_trace(tmp_path, trace)
    assert len(paths) == 1
    assert captured == [["total_loss", "ce_loss"]]


@pytest.mark.parametrize("protocol", ["intraday", "overnight"])
def test_actual_post_open_panel_matches_its_metrics_and_terminal_contract(protocol):
    from trading_system.training.post_open_panel import PostOpenReturnPanel
    dates = pd.date_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"date": dates, "ticker": "A", "open": [100, 110, 105, 108],
                           "close": [103, 111, 106, 109], "adj_close": [103, 111, 106, 109]})
    panel = PostOpenReturnPanel(prices, protocol=protocol)
    config = FinancialLossConfig(cost_bps=5)
    positions = [1, .4, -.5, 1]
    result = evaluate_positions(panel, positions, config, keys=panel.signal_frame)
    existing = panel.metrics(positions, config)
    assert result["metrics"]["net_pnl"] == pytest.approx(existing["net_pnl"])
    assert result["metrics"]["long_pnl_contribution"] + result["metrics"]["short_pnl_contribution"] == pytest.approx(existing["net_pnl"])
    if protocol == "overnight":
        assert result["position_records"][-1]["position"] == 0
        assert result["position_records"][-1]["cost"] == .00025
    assert len(result["daily_paths"]) == 4
