"""The chart layer must present recorded results without changing accounting."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trading_system.visualization.charts import (
    benchmark_figure,
    capital_drawdown_figure,
    drawdown_figure,
    equity_figure,
    export_html,
    exposure_figure,
    history_figure,
    kpi_bar_figure,
    metric_distribution_figure,
    monthly_returns_figure,
    position_figure,
    trades_figure,
)
from trading_system.visualization.theme import PALETTE, SYSTEM_FONT


def _run(equity=None, **kwargs):
    values = dict(
        record=SimpleNamespace(run_id="run-1", label="GRU · seed 1"),
        equity=pd.DataFrame() if equity is None else equity,
        positions=pd.DataFrame(), trades=pd.DataFrame(), history={}, warnings=(),
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def _equity(values, **columns):
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=len(values), tz="UTC"),
                         "equity": values, **columns})


def test_equity_sorts_rebases_observed_capital_and_preserves_input():
    source = _equity([9986.15, 10050.0, 10100.0], benchmark_equity=[9567.64, 9700.0, 9600.0])
    source = source.iloc[[2, 0, 1]].copy()
    original = source.copy(deep=True)
    data = _run(source)
    figure = equity_figure([data])
    strategy, reference = figure.data
    np.testing.assert_allclose(strategy.y, np.array([9986.15, 10050., 10100.]) / 9986.15 * 100.)
    np.testing.assert_allclose(reference.y, np.array([9567.64, 9700., 9600.]) / 9567.64 * 100.)
    np.testing.assert_allclose(strategy.customdata, [9986.15, 10050., 10100.])
    assert pd.Index(strategy.x).is_monotonic_increasing
    assert strategy.line.color == PALETTE["blue"]
    assert reference.line.color == PALETTE["muted"]
    assert reference.line.dash == "dash"
    raw = equity_figure([data], normalize=False)
    np.testing.assert_allclose(raw.data[0].y, [9986.15, 10050., 10100.])
    pd.testing.assert_frame_equal(source, original)


def test_drawdown_uses_recorded_strategy_column_and_never_reference():
    data = _run(_equity([100., 90., 120.], benchmark_equity=[100., 40., 30.], drawdown=[-.01, -.2, -.03]))
    figure = drawdown_figure([data])
    assert len(figure.data) == 1
    np.testing.assert_allclose(figure.data[0].y, [-1., -20., -3.])
    derived = drawdown_figure([_run(_equity([100., 90., 120., 96.]))])
    np.testing.assert_allclose(derived.data[0].y, [0., -10., 0., -20.])


def test_display_sampling_keeps_endpoints_and_drawdown_extreme():
    values = np.linspace(100., 160., 100)
    values[42] = 3.
    data = _run(_equity(values))
    sampled = equity_figure([data], normalize=False, max_points=10)
    assert len(sampled.data[0].y) <= 10
    assert sampled.data[0].y[0] == 100.
    assert sampled.data[0].y[-1] == 160.
    assert 3. in sampled.data[0].y
    full_dd = drawdown_figure([data], max_points=200)
    sampled_dd = drawdown_figure([data], max_points=10)
    assert len(sampled_dd.data[0].y) <= 10
    assert min(sampled_dd.data[0].y) == min(full_dd.data[0].y)
    np.testing.assert_array_equal(data.equity["equity"], values)


def test_run_color_stable_across_chart_selection_order():
    first = _run(_equity([100., 105.]))
    second = _run(_equity([200., 205.]), record=SimpleNamespace(run_id="run-2", label="Other"))
    assert equity_figure([first, second]).data[0].line.color == equity_figure([second, first]).data[1].line.color
    assert equity_figure([first]).data[0].line.color == drawdown_figure([first]).data[0].line.color


def test_eight_strategies_have_distinct_stable_styles_and_matching_drawdowns():
    runs = [_run(_equity([100., 110., 105.], benchmark_equity=[100., 106., 107.]),
                 record=SimpleNamespace(run_id=f"run-{index}", label=f"Model {index}")) for index in range(8)]
    capital = equity_figure(runs)
    drawdown = drawdown_figure(runs)
    styles = {trace.legendgroup: (trace.line.color, trace.line.dash)
              for trace in capital.data if not trace.legendgroup.startswith("reference:")}
    assert len(set(styles.values())) == 8
    assert styles == {trace.legendgroup: (trace.line.color, trace.line.dash) for trace in drawdown.data}
    reordered = equity_figure(runs[::-1])
    assert styles == {trace.legendgroup: (trace.line.color, trace.line.dash)
                      for trace in reordered.data if not trace.legendgroup.startswith("reference:")}
    references = [trace for trace in capital.data if trace.legendgroup.startswith("reference:")]
    assert all(trace.line.color == PALETTE["muted"] and trace.line.dash == "dash" for trace in references)


def test_multi_run_legend_reserves_space_keeps_full_identity_in_hover():
    full_label = "Long experiment name " * 4 + " seed 1 fold 0 test"
    runs = [_run(_equity([100., 110.], benchmark_equity=[100., 105.]),
                 record=SimpleNamespace(run_id=f"run-{index}", label=full_label)) for index in range(8)]
    figure = capital_drawdown_figure(runs)
    standalone = capital_drawdown_figure(runs[:1])
    assert figure.layout.margin.b > standalone.layout.margin.b
    assert figure.layout.height > standalone.layout.height
    assert figure.layout.legend.entrywidthmode == "fraction"
    assert len({trace.name for trace in figure.data if trace.showlegend is not False}) == 16
    assert all(len(trace.name) < 45 for trace in figure.data)
    strategy = figure.data[0]
    assert strategy.meta == full_label
    assert "%{meta}" in strategy.hovertemplate
    corresponding_dd = next(trace for trace in figure.data if trace.yaxis == "y2" and trace.legendgroup == strategy.legendgroup)
    assert corresponding_dd.line.color == strategy.line.color
    assert corresponding_dd.line.dash == strategy.line.dash


def test_chart_theme_matches_web_and_disables_decorative_motion():
    figure = equity_figure([_run(_equity([100., 105.]))])
    assert figure.layout.font.family == SYSTEM_FONT
    assert figure.layout.font.color == PALETTE["ink"]
    assert figure.layout.paper_bgcolor == PALETTE["white"]
    assert figure.layout.yaxis.gridcolor == PALETTE["line"]
    assert figure.layout.hovermode == "x unified"
    assert figure.layout.transition.duration == 0


def test_capital_drawdown_share_dates_and_do_not_duplicate_legends():
    data = _run(_equity([100., 90., 120.], benchmark_equity=[100., 99., 98.]))
    figure = capital_drawdown_figure([data])
    assert len(figure.data) == 3
    assert figure.data[0].yaxis == "y"
    assert figure.data[1].name == "Buy & Hold"
    assert figure.data[2].yaxis == "y2"
    assert figure.data[2].showlegend is False
    assert figure.layout.xaxis.matches == "x2"
    np.testing.assert_array_equal(figure.data[0].x, figure.data[2].x)
    assert figure.data[2].fillcolor.endswith(",0.09)")


def test_kpi_bars_preserve_exact_metric_name_and_only_scale_percent_whitelist():
    frame = pd.DataFrame({"label": ["GRU", "ANN"], "net_return": [.1, -.2],
                          "regularized_sharpe": [1.3, -.2], "status": ["ok", "partial"]})
    returns = kpi_bar_figure(frame, metric="net_return")
    np.testing.assert_array_equal(returns.data[0].x, [10., -20.])
    assert returns.data[0].orientation == "h"
    assert returns.data[0].marker.color == (PALETTE["blue"], PALETTE["sell-ink"])
    sharpe = kpi_bar_figure(frame, metric="regularized_sharpe")
    np.testing.assert_array_equal(sharpe.data[0].x, [1.3, -.2])
    assert sharpe.layout.title.text == "regularized_sharpe"
    assert sharpe.layout.xaxis.title.text == "regularized_sharpe"
    assert "%" not in sharpe.data[0].text[0]


def test_kpi_compact_labels_keep_distinct_runs_and_full_identity_in_hover():
    labels = ["Long experiment name with a distinct candidate ABC, seed 42",
              "Long experiment name with a distinct candidate XYZ, seed 42"]
    frame = pd.DataFrame({"label": labels, "run_id": ["first", "second"], "net_return": [.1, .2]})
    trace = kpi_bar_figure(frame, metric="net_return").data[0]
    assert len(set(trace.y)) == 2
    assert all(len(label) <= 22 for label in trace.y)
    assert list(trace.customdata[:, 0]) == labels
    assert list(trace.customdata[:, 1]) == ["first", "second"]


def test_position_markers_are_recorded_sides_and_window_is_applied_after_events(monkeypatch):
    from trading_system.visualization import positions

    dates = pd.date_range("2024-01-01", periods=5, tz="UTC")
    full = pd.DataFrame({"date": dates, "price": [100., 101., 102., 103., 104.]})
    # Short opening is a sell; short covering is a buy, regardless of final position.
    events = pd.DataFrame({"date": [dates[1], dates[3]], "price": [101., 103.],
                           "side": ["sell", "buy"], "kind": ["entry_short", "exit_short"],
                           "source": ["executed_orders", "executed_orders"]})
    data = _run(_equity([100., 101., 102., 103., 104.]))
    calls = []
    monkeypatch.setattr(positions, "asset_frame", lambda observed, ticker: full)

    def complete_events(observed, ticker):
        calls.append((len(observed.equity), ticker))
        return events

    monkeypatch.setattr(positions, "position_events", complete_events)
    figure = position_figure(data, "AAA", start="2024-01-02", end="2024-01-04")
    assert calls == [(5, "AAA")]
    price, buy, sell = figure.data
    assert list(price.x) == dates[1:4].tolist()
    assert list(buy.x) == [dates[3]]
    assert list(sell.x) == [dates[1]]
    assert buy.marker.symbol == "triangle-up"
    assert sell.marker.symbol == "triangle-down"
    assert "source" in buy.hovertemplate and "kind" in buy.hovertemplate
    assert buy.customdata[0][0] == "executed_orders"
    assert buy.customdata[0][1] == "exit_short"


def test_position_price_absence_does_not_make_up_market_series(monkeypatch):
    from trading_system.visualization import positions

    monkeypatch.setattr(positions, "asset_frame", lambda *_: pd.DataFrame())
    monkeypatch.setattr(positions, "position_events", lambda *_: pd.DataFrame())
    figure = position_figure(_run(), "AAA")
    assert not figure.data
    assert any("Série de prix absente" in item.text for item in figure.layout.annotations)


def test_missing_dates_or_zero_normalization_do_not_invent_series():
    assert not equity_figure([_run(pd.DataFrame({"equity": [100., 102.]}))]).data
    assert not equity_figure([_run(_equity([0., 100.]))]).data
    assert len(equity_figure([_run(_equity([0., 100.]))], normalize=False).data) == 1
    assert not equity_figure([_run(_equity([np.nan, np.inf]))]).data
    invalid = _equity([100., 102.])
    invalid["date"] = ["bad", "2024-01-02"]
    figure = equity_figure([_run(invalid)])
    assert len(figure.data[0].y) == 1


def test_distribution_keeps_raw_observations_excludes_failures_aggregates():
    frame = pd.DataFrame({
        "model": ["gru"] * 5, "seed": [1, 2, 3, 4, 5],
        "score": [1., 3., 200., 500., np.inf],
        "status": ["ok", "ok", "failed", "ok", "ok"],
        "row_kind": ["run", "run", "run", "summary", "run"],
    })
    figure = metric_distribution_figure(frame, metric="score")
    np.testing.assert_array_equal(figure.data[0].y, [1., 3.])
    assert figure.data[0].boxpoints == "all"
    assert not metric_distribution_figure(pd.DataFrame({"score_mean": [2.]}), metric="score_mean").data
    assert not metric_distribution_figure(pd.DataFrame({"score": [2.], "is_aggregate": [True]}), metric="score").data


@pytest.mark.parametrize("metric", ["trade_count", "order_count", "parameter_count"])
def test_distribution_counts_are_individual_metrics(metric):
    frame = pd.DataFrame({"model": ["gru", "gru"], metric: [10, 20]})
    figure = metric_distribution_figure(frame, metric=metric)
    np.testing.assert_array_equal(figure.data[0].y, [10., 20.])


def test_partial_runs_keep_valid_observations_and_show_status():
    frame = pd.DataFrame({"model": ["gru"] * 4, "status": ["ok", "partial", "failed", "error"],
                          "score": [1., 2., 300., 400.], "pnl": [10., 20., 3000., 4000.]})
    distribution = metric_distribution_figure(frame, metric="score")
    np.testing.assert_array_equal(distribution.data[0].y, [1., 2.])
    assert "status" in distribution.data[0].hovertemplate
    assert "partial" in distribution.data[0].customdata[1]
    scatter = benchmark_figure(frame, x="score", y="pnl")
    np.testing.assert_array_equal(scatter.data[0].x, [1., 2.])
    assert "status" in scatter.data[0].hovertemplate
    assert "partial" in scatter.data[0].customdata[1]


def test_benchmark_scatter_handles_single_missing_axis_and_invalid_rows():
    frame = pd.DataFrame({"label": ["a", "b", "c"], "family": ["comparison"] * 3,
                          "score": [1., np.nan, 3.], "pnl": [5., 8., 9.],
                          "status": ["ok", "ok", "failed"]})
    figure = benchmark_figure(frame, x="score", y="pnl")
    np.testing.assert_array_equal(figure.data[0].x, [1.])
    np.testing.assert_array_equal(figure.data[0].y, [5.])
    assert not benchmark_figure(frame, x="missing", y="pnl").data


def test_exposure_not_implicitly_computed_from_asset_positions():
    data = _run(_equity([100., 101.], gross_exposure=[.3, 1.2], net_exposure=[-.2, .5]))
    figure = exposure_figure(data)
    np.testing.assert_allclose(figure.data[0].y, [30., 120.])
    np.testing.assert_allclose(figure.data[1].y, [-20., 50.])
    positions_only = _run(positions=pd.DataFrame({"date": pd.date_range("2024-01-01", periods=2),
                                                "ticker": ["a", "b"], "position": [1., 1.]}))
    assert not exposure_figure(positions_only).data


def test_monthly_returns_compound_recorded_first_return_before_sampling():
    data = _run(_equity([9000., 9900., 10890.], net_return=[-.1, .1, .1]))
    figure = monthly_returns_figure(data)
    assert figure.data[0].z[0][0] == pytest.approx(8.9)
    assert np.isnan(figure.data[0].z[0][1])
    # Unknown initial return is not filled with zero in the equity-only fallback.
    observed = monthly_returns_figure(_run(_equity([9000., 9900., 10890.])))
    assert observed.data[0].z[0][0] == pytest.approx(21.)
    assert "transitions" in observed.layout.title.text
    missing = monthly_returns_figure(_run(_equity([100., 102., 105.], net_return=[0., np.nan, .03])))
    assert not missing.data


def test_history_accepts_nested_fit_result_and_ignores_scalar_metadata():
    data = _run(history={"seed": 1, "history": {"train_loss": [1., .5], "val_loss": [.9, .6],
                                               "notes": ["a", "b"], "best_epoch": 2}})
    figure = history_figure(data)
    assert [trace.name for trace in figure.data] == ["train_loss", "val_loss"]
    np.testing.assert_array_equal(figure.data[0].x, [1, 2])


def test_trades_show_saved_individual_pnl_only():
    figure = trades_figure(_run(trades=pd.DataFrame({"net_pnl": [-10., 20., np.nan, np.inf]})))
    np.testing.assert_array_equal(figure.data[0].x, [-10., 20.])
    assert not trades_figure(_run(trades=pd.DataFrame({"total_pnl": [500.]}))).data


@pytest.mark.parametrize("build", [
    lambda: equity_figure([]), lambda: drawdown_figure([]),
    lambda: benchmark_figure(pd.DataFrame(), x="x", y="y"),
    lambda: metric_distribution_figure(pd.DataFrame(), metric="x"),
    lambda: exposure_figure(_run()), lambda: trades_figure(_run()),
    lambda: history_figure(_run()), lambda: monthly_returns_figure(_run()),
])
def test_empty_series_show_clear_message(build):
    figure = build()
    assert not figure.data
    assert figure.layout.annotations
    assert figure.layout.annotations[0].text


def test_html_export_embeds_runtime_once_and_escapes_section_names():
    data = _run(_equity([100., 110.]))
    document = export_html({"<test>": equity_figure([data]), "DD": drawdown_figure([data])})
    assert document.startswith("<!doctype html>")
    assert document.count("plotly.js v") == 1
    assert document.count("Plotly.newPlot(") == 2
    assert '<script src="' not in document
    assert "&lt;test&gt;" in document
    assert 'id="run-chart-0"' in document
    assert 'id="run-chart-1"' in document
    assert "Aucune figure" in export_html({})


@pytest.mark.parametrize("max_points", [0, 1, True, 2.5])
def test_invalid_display_budget_has_explicit_error(max_points):
    with pytest.raises(ValueError, match="max_points must"):
        equity_figure([_run(_equity([100., 101., 102.]))], max_points=max_points)
