import numpy as np
import pandas as pd
import pytest

from trading_system.analysis.label_diagnostics import analyze_label_result
from trading_system.labels.config import LabelConfig
from trading_system.labels.registry import LabelResult


def _result(ids, *, prices=None, known=None, config=None, dates=None, **columns):
    config = config or LabelConfig.breakout()
    frame = pd.DataFrame({
        "ticker": "TEST", "date": dates if dates is not None else pd.bdate_range("2020-01-01", periods=len(ids)),
        "adj_close": prices if prices is not None else np.arange(len(ids), dtype=float) + 100,
        "Label_id": ids,
        "Label": [config.class_names[index] for index in ids],
        **columns,
    })
    return LabelResult(frame, np.ones(len(ids), dtype=bool) if known is None else np.array(known, dtype=bool),
                       config.class_names, config.semantics), config


def test_hold_runs_are_not_position_durations_and_turnover_includes_liquidation():
    result, config = _result([2, 1, 1, 0, 1], prices=[100, 110, 120, 115, 100])
    report = analyze_label_result(result, config, ticker="TEST", fee_bps=5)
    assert report["label_runs"]["length_sessions"].tolist() == [1, 2, 1, 1]
    trades = report["position_trades"]
    assert trades["side"].tolist() == [1, -1]
    assert trades["duration_sessions"].tolist() == [3, 1]
    assert trades["gross_return"].tolist() == pytest.approx([0.15, 1 - 100 / 115])
    assert trades["net_return"].tolist() == pytest.approx([0.149, 1 - 100 / 115 - 0.001])
    assert trades["right_censored"].tolist() == [False, True]
    assert report["summary"]["turnover_units"] == 4
    assert report["summary"]["long_exposure"] == 0.75
    assert report["summary"]["short_exposure"] == 0.25


def test_target_flat_closes_position_instead_of_carrying():
    config = LabelConfig.volatility_position(horizon=1)
    result, _ = _result([2, 1, 1, 0, 1], config=config,
                        fwd_ret=[0.01, 0.01, 0.01, -0.01, np.nan], known=[1, 1, 1, 1, 0])
    report = analyze_label_result(result, config, ticker="TEST")
    assert report["position_trades"]["duration_sessions"].tolist() == [1, 1]
    assert report["summary"]["flat_exposure"] == 0.5


def test_unknown_resets_carry_and_does_not_bridge_runs_or_transitions():
    result, config = _result([2, 1, 1, 1, 2, 1], known=[1, 1, 0, 1, 1, 1])
    report = analyze_label_result(result, config, ticker="TEST")
    assert report["position_trades"]["duration_sessions"].tolist() == [2, 1]
    assert report["position_trades"]["right_censored"].tolist() == [True, True]
    assert report["summary"]["n_unknown"] == 1
    assert report["summary"]["n_adjacent_known_pairs"] == 3
    assert report["summary"]["class_counts"]["Hold"] == 3
    assert report["summary"]["unknown_exposure"] == 0.2
    assert report["summary"]["n_exposure_sessions"] == 5
    assert report["summary"]["n_known_exposure_sessions"] == 4
    assert report["label_runs"]["length_sessions"].tolist() == [1, 1, 1, 1, 1]


def test_start_preserves_preceding_action_state_but_rebases_return():
    result, config = _result([2, 1, 1, 1, 0], prices=[100, 105, 110, 120, 125])
    report = analyze_label_result(result, config, ticker="TEST", start="2020-01-03", fee_bps=0)
    trades = report["position_trades"]
    assert len(trades) == 1
    assert trades.iloc[0]["duration_sessions"] == 2
    assert bool(trades.iloc[0]["left_censored"])
    assert not bool(trades.iloc[0]["right_censored"])
    assert trades.iloc[0]["gross_return"] == pytest.approx(125 / 110 - 1)
    assert report["summary"]["n_rows"] == 3
    assert report["label_runs"].iloc[0]["length_sessions"] == 2
    assert bool(report["label_runs"].iloc[0]["left_censored"])


def test_triple_barrier_durations_count_sessions_and_short_returns_have_correct_sign():
    config = LabelConfig.triple_barrier()
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    result, _ = _result([2, 0, 1, 1], config=config, dates=dates,
                        label_event_id=[0, 1, pd.NA, pd.NA],
                        label_end_date=[dates[2], dates[3], pd.NaT, pd.NaT],
                        event_return=[0.02, -0.03, np.nan, np.nan])
    report = analyze_label_result(result, config, ticker="TEST", fee_bps=5)
    events = report["native_events"]
    assert events["duration_sessions"].tolist() == [2, 2]
    assert events["gross_return"].tolist() == pytest.approx([0.02, 0.03])
    assert events["net_return"].tolist() == pytest.approx([0.019, 0.029])
    assert events["overlapping"].tolist() == [True, True]
    assert report["summary"]["native_events_compounded"] is False


def test_native_neutral_events_keep_market_move_without_artificial_trade_returns():
    config = LabelConfig.forward_return(horizon=2)
    result, _ = _result([2, 1, 0, 1, 1], config=config,
                        known=[1, 1, 1, 0, 0], fwd_ret=[0.02, 0.003, -0.03, np.nan, np.nan])
    report = analyze_label_result(result, config, ticker="TEST")
    events = report["native_events"]
    assert events["directed"].tolist() == [True, False, True]
    assert pd.isna(events.iloc[1]["net_return"])
    assert report["summary"]["native_event_stats"]["count"] == 2
    assert report["summary"]["native_event_stats"]["n_events_including_neutral"] == 3


def test_intraday_reopens_daily_even_for_consecutive_long_labels():
    config = LabelConfig.intraday_return()
    result, _ = _result([2, 2, 0, 1], config=config,
                        open=[100, 100, 100, 100], close=[102, 103, 98, 100],
                        intraday_ret=[0.02, 0.03, -0.02, 0])
    report = analyze_label_result(result, config, ticker="TEST", fee_bps=5)
    assert report["label_runs"]["length_sessions"].tolist() == [2, 1, 1]
    trades = report["position_trades"]
    assert trades["duration_sessions"].tolist() == [1, 1, 1]
    assert trades["net_return"].tolist() == pytest.approx([0.019, 0.029, 0.019])
    assert report["summary"]["turnover_units"] == 6
    assert not trades["right_censored"].any()


def test_common_mask_only_restricts_probes_and_horizon_tail_is_not_zero_filled():
    result, config = _result([2, 1, 0, 1], prices=[100, 110, 100, 90])
    report = analyze_label_result(result, config, ticker="TEST", probe_horizons=(1, 3, 5),
                                  noise_thresholds=(0.01,), common_mask=np.array([False, True, True, False]))
    assert report["summary"]["turnover_units"] == 4
    probes = report["horizon_probes"]
    assert probes["n_rows"].tolist() == [2, 0, 0]
    assert probes.iloc[0]["n_active"] == 1
    assert probes.iloc[0]["n_neutral"] == 1
    assert probes.iloc[0]["missed_neutral_count"] == 1
    assert probes.iloc[0]["missed_neutral_rate"] == 0.5
    assert probes.iloc[0]["side_accuracy"] == 1


def test_sorting_realigns_known_and_common_masks():
    ordered, config = _result([2, 1, 0, 1], known=[1, 0, 1, 1])
    permutation = np.array([2, 0, 3, 1])
    mixed = LabelResult(ordered.frame.iloc[permutation].reset_index(drop=True),
                        ordered.known_mask[permutation], config.class_names, config.semantics)
    mask = np.array([True, False, True, False])
    baseline = analyze_label_result(ordered, config, ticker="TEST", common_mask=mask)
    reordered = analyze_label_result(mixed, config, ticker="TEST", common_mask=mask[permutation])
    assert baseline["summary"] == reordered["summary"]
    for name in ("label_runs", "position_trades", "native_events", "transitions", "horizon_probes"):
        pd.testing.assert_frame_equal(baseline[name], reordered[name])


def test_empty_visible_range_returns_stable_empty_tables():
    result, config = _result([2, 1, 0])
    report = analyze_label_result(result, config, ticker="TEST", start="2030-01-01")
    assert report["summary"]["n_rows"] == 0
    assert report["summary"]["turnover_units"] == 0
    assert report["summary"]["position_trade_stats"]["net_return"]["mean"] is None
    assert report["position_trades"].empty
    assert "duration_sessions" in report["native_events"]
    assert report["transitions"]["count"].sum() == 0


@pytest.mark.parametrize("case", ["duplicate", "invalid_price", "wrong_ticker", "wrong_common_shape", "wrong_common_dtype"])
def test_validation_rejects_corrupt_inputs(case):
    result, config = _result([2, 1, 0])
    kwargs = {}
    if case == "duplicate":
        result.frame.loc[1, "date"] = result.frame.loc[0, "date"]
    elif case == "invalid_price":
        result.frame.loc[1, "adj_close"] = 0
    elif case == "wrong_ticker":
        result.frame.loc[1, "ticker"] = "OTHER"
    elif case == "wrong_common_shape":
        kwargs["common_mask"] = np.array([True])
    else:
        kwargs["common_mask"] = np.array([1, 1, 1])
    with pytest.raises(ValueError):
        analyze_label_result(result, config, ticker="TEST", **kwargs)


def test_native_event_cannot_end_outside_provided_history():
    config = LabelConfig.triple_barrier()
    result, _ = _result([2, 1, 1], config=config,
                        label_event_id=[0, pd.NA, pd.NA],
                        label_end_date=[pd.Timestamp("2021-01-01"), pd.NaT, pd.NaT],
                        event_return=[0.02, np.nan, np.nan])
    with pytest.raises(ValueError, match="end inside"):
        analyze_label_result(result, config, ticker="TEST")


def test_singleton_label_entirely_neutral_has_no_fabricated_trade():
    result, config = _result([1])
    report = analyze_label_result(result, config, ticker="TEST")
    assert report["summary"]["label_entropy_bits"] == 0
    assert report["summary"]["position_trade_stats"]["count"] == 0
    assert report["horizon_probes"]["n_rows"].sum() == 0
