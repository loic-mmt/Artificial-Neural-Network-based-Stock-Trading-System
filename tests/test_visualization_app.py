"""Optional UI smoke tests: install the visualization extra to run them."""

from datetime import date
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from trading_system.visualization.cli import main as cli_main
from trading_system.visualization.catalog import discover_runs


def render(root):
    from trading_system.visualization.app import main
    main(["--artifacts-root", root])


def saved_mt5(root, name, seed):
    folder = root / name
    folder.mkdir()
    (folder / "metadata.json").write_text(json.dumps({"seed": seed, "model": "gru", "tickers": ["AAA", "BBB"], "initial_capital": 100}))
    (folder / "portfolio_metrics.json").write_text(json.dumps({"model": {"net_return": .08, "net_sharpe": 1.2}}))
    pd.DataFrame({"date": pd.date_range("2025-01-01", periods=3, tz="UTC"), "model_equity": [100., 90., 108.], "buy_hold_equity": [100., 105., 110.]}).to_parquet(folder / "equity_curve.parquet", index=False)
    pd.DataFrame({"date": list(pd.date_range("2025-01-01", periods=3, tz="UTC")) * 2, "ticker": ["AAA"] * 3 + ["BBB"] * 3,
                  "position": [0., 1., 1., 0., -1., 0.], "adj_close": [10., 11., 12., 20., 21., 22.]}).to_parquet(folder / "positions.parquet", index=False)


def widget(elements, label):
    return next(item for item in elements if item.label == label)


def figures(app):
    return [json.loads(element.proto.spec) for element in app.get("plotly_chart")]


def test_empty_and_invalid_directory_show_message(tmp_path):
    app = AppTest.from_function(render, args=(str(tmp_path),)).run()
    assert not app.exception
    assert "Aucun résultat reconnu" in app.info[0].value
    app.sidebar.text_input[0].set_value(str(tmp_path / "missing")).run()
    assert not app.exception
    assert "Dossier introuvable" in app.error[0].value


def test_metrics_only_benchmark_and_cli_without_starting_server(tmp_path, capsys):
    (tmp_path / "report.json").write_text(json.dumps({"runs": [{"model_name": "gru", "seed": 1, "status": "ok", "test_macro_f1": .6}]}))
    app = AppTest.from_function(render, args=(str(tmp_path),)).run()
    assert not app.exception
    assert any("Courbes absentes" in message.value for message in app.info)
    assert app.dataframe[0].value.test_macro_f1.tolist() == [.6]
    assert cli_main(["--artifacts-root", str(tmp_path), "--list"]) == 0
    assert "1 runs" in capsys.readouterr().out


def test_curve_selection_date_window_and_details(tmp_path):
    saved_mt5(tmp_path, "first", 1)
    saved_mt5(tmp_path, "second", 7)
    app = AppTest.from_function(render, args=(str(tmp_path),), default_timeout=20).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Portefeuille", "Comparer les runs"]
    selection = widget(app.multiselect, "Runs")
    ids = [record.run_id for record in discover_runs(tmp_path).records]
    selection.set_value(ids).run()
    assert not app.exception
    assert any("Comparaison descriptive" in message.value for message in app.warning)
    assert not app.tabs[1].warning  # The warning remains visible above both tabs.
    app.date_input[0].set_value((date(2025, 1, 2), date(2025, 1, 3))).run()
    assert not app.exception
    stats = next(element.value for element in app.dataframe if "observed_return" in element.value)
    assert stats.observed_return.tolist() == pytest.approx([.2, .2])
    recorded = next(element.value for element in app.dataframe if "net_sharpe" in element.value)
    assert recorded.net_return.tolist() == [.08, .08]
    assert widget(app.metric, "Stratégie").value == "+20.00%"
    assert widget(app.metric, "Buy & Hold").value == "+4.76%"
    widget(app.multiselect, "Actifs").set_value(["AAA", "BBB"]).run()
    assert not app.exception
    positions = next(element.value for element in app.dataframe if "ticker" in element.value)
    assert len(positions) == 4
    assert set(positions.ticker) == {"AAA", "BBB"}
    rendered = figures(app)
    asset_charts = {figure["layout"]["title"]["text"]: figure for figure in rendered if figure["layout"]["title"]["text"] in ("AAA", "BBB")}
    assert set(asset_charts) == {"AAA", "BBB"}
    assert any(trace.get("mode") == "markers" for trace in asset_charts["BBB"]["data"])
    assert any(trace.get("type") == "bar" for figure in rendered for trace in figure["data"])
    app.date_input[0].set_value((date(2025, 1, 3), date(2025, 1, 3))).run()
    assert not app.exception
    aaa = next(figure for figure in figures(app) if figure["layout"]["title"]["text"] == "AAA")
    assert not any(trace.get("mode") == "markers" for trace in aaa["data"])
    assert any(element.type == "download_button" for element in app.get("download_button"))


def test_universe_filter_matches_whole_ticker_tokens(tmp_path):
    saved_mt5(tmp_path, "first", 1)
    app = AppTest.from_function(render, args=(str(tmp_path),)).run()
    widget(app.sidebar.multiselect, "Univers contient").set_value(["BBB"]).run()
    assert not app.exception
    stats = next(element.value for element in app.dataframe if "observed_return" in element.value)
    assert len(stats) == 1
    assert stats.last_equity.tolist() == [108.]
