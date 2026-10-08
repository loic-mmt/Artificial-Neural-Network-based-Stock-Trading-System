from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from trading_system.analysis.label_benchmark import benchmark_labels, main, refresh_reports, save_benchmark, summarize_probes
from trading_system.labels.config import LabelConfig


@pytest.fixture
def history():
    prices = 100 + np.arange(90) * 0.1 + np.sin(np.arange(90)) * 2
    frame = pd.DataFrame({
        "date": pd.bdate_range("2022-01-03", periods=90), "ticker": "AAA",
        "open": prices * (1 + 0.005 * np.cos(np.arange(90))),
        "close": prices, "adj_close": prices, "high": prices + 2, "low": prices - 2,
    })
    return pd.concat([frame, frame.assign(ticker="BBB")], ignore_index=True)


def configs():
    return [
        LabelConfig.breakout(window=5), LabelConfig.forward_return(horizon=3),
        LabelConfig.volatility_position(horizon=3, volatility_window=5, position_mode="long_short"),
        LabelConfig.triple_barrier(max_holding=3, volatility_window=5, event_filter="all"),
        LabelConfig.intraday_return(),
    ]


def test_methods_align_shuffled_rows_and_same_probe_cohort(history):
    result = benchmark_labels(history.sample(frac=1, random_state=1), configs(), progress=False)
    assert len(result["summary"]) == 5
    assert len(result["by_ticker"]) == 10
    assert {row["n_tickers"] for row in result["summary"]} == {2}
    probes = result["tables"]["horizon_probes"]
    assert probes.groupby(["ticker", "horizon", "threshold"])["n_rows"].nunique().eq(1).all()
    intraday = next(row for row in result["summary"] if row["method"] == "intraday_return")
    assert intraday["position_trade_duration"]["min"] == 1
    assert intraday["position_trade_duration"]["max"] == 1
    assert intraday["turnover_per_session"] == 2


def test_future_targets_never_cross_end_and_start_keeps_warmup(history):
    boundary = history["date"].drop_duplicates().iloc[40]
    start = history["date"].drop_duplicates().iloc[20]
    result = benchmark_labels(history, configs(), start=str(start.date()), end=str(boundary.date()), progress=False)
    events = result["tables"]["native_events"]
    assert events["end_date"].max().tz_localize(None) <= boundary
    assert events["start_date"].min().tz_localize(None) >= start
    altered = history.copy()
    altered.loc[altered["date"] > boundary, ["open", "close", "adj_close", "high", "low"]] *= 10
    same = benchmark_labels(altered, configs(), start=str(start.date()), end=str(boundary.date()), progress=False)
    assert result["summary"] == same["summary"]
    for name in result["tables"]:
        pd.testing.assert_frame_equal(result["tables"][name], same["tables"][name])


def test_summary_and_exports_are_json_serializable(history, tmp_path):
    result = benchmark_labels(history, configs(), progress=False)
    save_benchmark(result, tmp_path, {"training_performed": False})
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["training_performed"] is False
    assert len(saved["summary"]) == 5
    assert "Aucun entraînement" in (tmp_path / "report.md").read_text()
    for filename in ("summary.csv", "by_ticker.csv", "horizon_probes.csv", "horizon_summary.csv",
                     "label_runs.parquet", "position_trades.parquet", "native_events.parquet"):
        assert (tmp_path / filename).stat().st_size > 0
    aggregate = summarize_probes(result["tables"]["horizon_probes"])
    assert aggregate.groupby(["horizon", "threshold"])["n_rows"].nunique().eq(1).all()
    before = saved["summary"]
    refresh_reports(tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text())["summary"] == before


def test_cli_selects_ticker_and_preserves_previous_output(history, tmp_path):
    data, output = tmp_path / "prices.parquet", tmp_path / "results"
    history.to_parquet(data)
    argv = ["--data", str(data), "--tickers", "AAA", "--methods", "intraday-return",
            "--no-progress", "--output-dir", str(output)]
    assert main(argv) == 0
    saved = json.loads((output / "summary.json").read_text())
    assert saved["summary"][0]["n_tickers"] == 1
    with pytest.raises(SystemExit):
        main(argv)


def test_duplicate_dates_or_empty_interval_are_rejected(history):
    with pytest.raises(ValueError, match="unique"):
        benchmark_labels(pd.concat([history, history.iloc[:1]]), configs(), progress=False)
    with pytest.raises(ValueError, match="No ticker sessions"):
        benchmark_labels(history, configs(), end="2000-01-01", progress=False)
