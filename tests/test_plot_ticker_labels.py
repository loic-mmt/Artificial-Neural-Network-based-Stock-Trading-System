from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def plotter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    return importlib.import_module("plot_ticker_labels")


@pytest.fixture
def history():
    prices = 100 + 3 * np.sin(np.arange(80) * 0.7) + np.arange(80) * 0.1
    return pd.DataFrame({
        "date": pd.bdate_range("2022-01-03", periods=80), "ticker": "TEST",
        "adj_close": prices, "close": prices, "high": prices + 1, "low": prices - 1,
    })


def test_shuffled_input_and_date_crop_preserve_full_history_labels(plotter, history):
    original = history.copy(deep=True)
    full = plotter.prepare_labels(history, "TEST")
    shuffled = plotter.prepare_labels(history.sample(frac=1, random_state=7), "TEST")
    pd.testing.assert_frame_equal(full, shuffled)
    start, end = full["date"].iloc[[30, 50]]
    cropped = plotter.prepare_labels(history, "TEST", start=str(start), end=str(end))
    expected = full.loc[full["date"].between(start, end)].reset_index(drop=True)
    pd.testing.assert_frame_equal(cropped, expected)
    pd.testing.assert_frame_equal(history, original)


def test_line_segments_use_start_label_and_distinguish_unknown_from_hold(plotter):
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    labeled = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=5, tz="UTC"),
        "adj_close": [100, 101, 102, 103, 104],
        "Label_id": [2, 1, 0, 1, 1], "_label_known": [True, True, True, False, False],
    })
    fig = plotter.build_figure(labeled, "TEST")
    ax = fig.axes[0]
    collection = ax.collections[0]
    assert len(collection.get_segments()) == 4
    assert [colors.to_hex(color) for color in collection.get_colors()] == [
        plotter.LABEL_COLORS[2], plotter.LABEL_COLORS[1], plotter.LABEL_COLORS[0],
        plotter.UNKNOWN_COLOR,
    ]
    assert not ax.patches  # No arrows.
    assert not ax.lines  # No markers or extra curves.
    plt.close(fig)


def test_duplicate_dates_and_empty_ticker_are_rejected(plotter, history):
    with pytest.raises(ValueError, match="unique"):
        plotter.prepare_labels(pd.concat([history, history.iloc[:1]]), "TEST")
    with pytest.raises(ValueError, match="No rows"):
        plotter.prepare_labels(history, "MISSING")
    with pytest.raises(ValueError, match="two sessions"):
        plotter.prepare_labels(history, "TEST", start="2030-01-01")
    with pytest.raises(ValueError, match="later"):
        plotter.prepare_labels(history, "TEST", start="2023-01-01", end="2022-01-01")


def test_cli_saves_without_opening_window(plotter, history, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    data, output = tmp_path / "prices.parquet", tmp_path / "plot.png"
    pd.concat([history, history.assign(ticker="OTHER")]).to_parquet(data)
    monkeypatch.setattr(plt, "show", lambda: pytest.fail("Must not open a window."))
    assert plotter.main([
        "--data", str(data), "--ticker", "TEST", "--output", str(output), "--no-show",
    ]) == 0
    assert output.stat().st_size > 1_000


def test_intraday_plot_uses_raw_prices_position_legend_and_actual_method(plotter):
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    history = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=5), "ticker": "TEST",
        "open": [100, 100, 100, 0, 100], "close": [101, 99, 100, 100, 102],
        "adj_close": [10, 11, 12, 13, 14],
    })
    labeled = plotter.prepare_labels(history, "TEST", label_method="intraday-return")
    assert labeled["Label"].tolist() == ["Long", "Short", "Flat", "Flat", "Long"]
    assert labeled["_label_known"].tolist() == [True, True, True, False, True]
    np.testing.assert_allclose(labeled["intraday_ret"].iloc[:3], [0.01, -0.01, 0])
    fig = plotter.build_figure(labeled, "TEST")
    ax = fig.axes[0]
    assert "Intraday return" in ax.get_title()
    assert [text.get_text() for text in ax.get_legend().texts] == [
        "Long", "Flat", "Short", "Unknown",
    ]
    assert "Open J to close J" in fig._supxlabel.get_text()
    assert "visual J to J+1 adj_close" in fig._supxlabel.get_text()
    assert [colors.to_hex(color) for color in ax.collections[0].get_colors()] == [
        plotter.LABEL_COLORS[2], plotter.LABEL_COLORS[0], plotter.LABEL_COLORS[1],
        plotter.UNKNOWN_COLOR,
    ]
    assert not ax.patches and not ax.lines
    plt.close(fig)


def test_intraday_cli_requires_no_atr_columns_and_filters_ticker(plotter, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    history = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=3), "ticker": "TEST",
        "open": [100, 100, 100], "close": [101, 99, 100], "adj_close": [10, 11, 12],
    })
    data, output = tmp_path / "prices.parquet", tmp_path / "intraday.png"
    pd.concat([history, history.assign(ticker="OTHER")]).to_parquet(data)
    monkeypatch.setattr(plt, "show", lambda: pytest.fail("Must not open a window."))
    original_prepare = plotter.prepare_labels

    def check_read(frame, *args, **kwargs):
        assert frame["ticker"].eq("TEST").all()
        assert "high" not in frame and "low" not in frame
        return original_prepare(frame, *args, **kwargs)

    monkeypatch.setattr(plotter, "prepare_labels", check_read)
    assert plotter.main([
        "--data", str(data), "--ticker", "TEST", "--label-method", "intraday-return",
        "--output", str(output), "--no-show",
    ]) == 0
    assert output.stat().st_size > 1_000


def test_intraday_plot_rejects_irrelevant_triple_barrier_parameters(plotter, history):
    with pytest.raises(ValueError, match="do not accept"):
        plotter.prepare_labels(history, "TEST", label_method="intraday-return", max_holding=10)


@pytest.fixture
def history_with_open(history):
    frame = history.copy()
    frame["open"] = frame["close"] * np.where(np.arange(len(frame)) % 2, 1.01, 0.99)
    return frame


@pytest.mark.parametrize("method", [
    "breakout", "forward_return", "volatility_position", "triple_barrier", "intraday_return",
])
def test_each_method_matches_registry_defaults_and_figure_semantics(
    plotter, history_with_open, method,
):
    import matplotlib.pyplot as plt
    from trading_system.labels.config import LabelConfig
    from trading_system.labels.registry import LabelContext, create_default_label_registry

    configs = {
        "breakout": LabelConfig.breakout(window=20, alternating=True),
        "forward_return": LabelConfig.forward_return(horizon=10),
        "volatility_position": LabelConfig.volatility_position(position_mode="long_short"),
        "triple_barrier": LabelConfig.triple_barrier(**plotter.DEFAULT_LABEL_PARAMETERS),
        "intraday_return": LabelConfig.intraday_return(),
    }
    work = history_with_open.copy()
    work["date"] = pd.to_datetime(work["date"], utc=True)
    expected = create_default_label_registry().generate(
        work, configs[method], LabelContext(split_col=None),
    )
    actual = plotter.prepare_labels(
        history_with_open.sample(frac=1, random_state=7), "TEST", label_method=method,
    )
    pd.testing.assert_series_equal(actual["Label"], expected.frame["Label"])
    pd.testing.assert_series_equal(actual["Label_id"], expected.frame["Label_id"])
    np.testing.assert_array_equal(actual["_label_known"], expected.known_mask)
    assert actual.attrs["label_method"] == method
    assert actual.attrs["label_semantics"] == configs[method].semantics

    start, end = actual["date"].iloc[[30, 50]]
    cropped = plotter.prepare_labels(
        history_with_open, "TEST", label_method=method, start=str(start), end=str(end),
    )
    pd.testing.assert_frame_equal(
        cropped, actual.loc[actual["date"].between(start, end)].reset_index(drop=True),
    )
    fig = plotter.build_figure(actual, "TEST")
    ax = fig.axes[0]
    assert method.replace("_", " ").capitalize() in ax.get_title()
    assert [text.get_text() for text in ax.get_legend().texts] == [
        expected.class_names[2], expected.class_names[1], expected.class_names[0], "Unknown",
    ]
    assert not ax.patches and not ax.lines
    plt.close(fig)


@pytest.mark.parametrize("method, parameters", [
    ("breakout", {"window": 5, "alternating": False}),
    ("forward_return", {"horizon": 3, "buy_threshold": 0.01}),
    ("volatility_position", {"horizon": 3, "position_mode": "long_flat"}),
    ("triple_barrier", {"volatility_estimator": "rolling_std", "max_holding": 3}),
])
def test_method_specific_overrides_match_registry(plotter, history_with_open, method, parameters):
    from trading_system.labels.config import LabelConfig
    from trading_system.labels.registry import LabelContext, create_default_label_registry

    defaults = {
        "breakout": {},
        "forward_return": {"horizon": 10},
        "volatility_position": {"position_mode": "long_short"},
        "triple_barrier": plotter.DEFAULT_LABEL_PARAMETERS,
    }
    config = getattr(LabelConfig, method)(**(defaults[method] | parameters))
    work = history_with_open.copy()
    work["date"] = pd.to_datetime(work["date"], utc=True)
    expected = create_default_label_registry().generate(work, config, LabelContext(split_col=None))
    actual = plotter.prepare_labels(work, "TEST", label_method=method, **parameters)
    pd.testing.assert_series_equal(actual["Label_id"], expected.frame["Label_id"])
    np.testing.assert_array_equal(actual["_label_known"], expected.known_mask)


@pytest.mark.parametrize("method", [
    "breakout", "forward-return", "forward_return", "volatility-position", "volatility_position",
    "triple-barrier", "triple_barrier", "intraday-return", "intraday_return",
])
def test_cli_accepts_every_label_method_alias(plotter, history_with_open, tmp_path, method):
    data, output = tmp_path / "prices.parquet", tmp_path / "labels.png"
    history_with_open.to_parquet(data)
    assert plotter.main([
        "--data", str(data), "--ticker", "TEST", "--label-method", method,
        "--output", str(output), "--no-show",
    ]) == 0
    assert output.stat().st_size > 1_000


@pytest.mark.parametrize("method, flags", [
    ("breakout", ["--label-window", "5"]),
    ("forward-return", ["--label-horizon", "3", "--label-buy-threshold", "0.01"]),
    ("volatility-position", ["--label-position-mode", "long_flat"]),
    ("intraday-return", []),
])
def test_cli_non_barrier_methods_do_not_require_atr_columns(
    plotter, history_with_open, tmp_path, monkeypatch, method, flags,
):
    frame = history_with_open.drop(columns=["high", "low"])
    data, output = tmp_path / "prices.parquet", tmp_path / "labels.png"
    pd.concat([frame, frame.assign(ticker="OTHER")]).to_parquet(data)
    original_prepare = plotter.prepare_labels

    def check_projection(frame, *args, **kwargs):
        assert "high" not in frame and "low" not in frame
        assert frame["ticker"].eq("TEST").all()
        return original_prepare(frame, *args, **kwargs)

    monkeypatch.setattr(plotter, "prepare_labels", check_projection)
    assert plotter.main([
        "--data", str(data), "--ticker", "TEST", "--label-method", method,
        *flags, "--output", str(output), "--no-show",
    ]) == 0
    assert output.stat().st_size > 1_000


@pytest.mark.parametrize("method", ["forward-return", "intraday-return"])
def test_cli_rejects_barrier_parameter_for_other_methods_before_data_read(plotter, method):
    with pytest.raises(SystemExit) as error:
        plotter.main([
            "--data", "nonexistent.parquet", "--ticker", "TEST", "--label-method", method,
            "--label-profit-barrier", "0.75", "--no-show",
        ])
    assert error.value.code == 2


def test_forward_return_unknown_tail_is_not_colored_as_hold(plotter, history):
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    labeled = plotter.prepare_labels(history, "TEST", label_method="forward_return", horizon=3)
    assert not labeled["_label_known"].iloc[-3:].any()
    assert labeled["_label_known"].iloc[:-3].all()
    fig = plotter.build_figure(labeled, "TEST")
    plotted_colors = [colors.to_hex(color) for color in fig.axes[0].collections[0].get_colors()]
    assert plotted_colors[-2:] == [plotter.UNKNOWN_COLOR] * 2
    assert plotter.UNKNOWN_COLOR not in plotted_colors[:-2]
    plt.close(fig)
