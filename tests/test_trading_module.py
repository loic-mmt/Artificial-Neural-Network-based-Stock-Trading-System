from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from trading_system.trading import (
    TradingConfig, run_trading_backtest, targets_from_labels,
    targets_from_probabilities,
)
from trading_system.trading.replay import main, replay_signals


def bars(opens=None, closes=None, *, highs=None, lows=None, count=6, ticker="A"):
    opens = np.array(opens if opens is not None else [100.] * count, dtype=float)
    closes = np.array(closes if closes is not None else opens, dtype=float)
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=len(opens)), "ticker": ticker,
        "open": opens, "high": highs if highs is not None else np.maximum(opens, closes) + 1,
        "low": lows if lows is not None else np.minimum(opens, closes) - 1,
        "close": closes,
    })


def cfg(**kwargs):
    return TradingConfig(enabled=True, fees_bps=0, initial_capital=1000, **kwargs)


def event(timestamp="2020-01-02T12:00:00Z", **values):
    row = {"event_type": "ECB", "timestamp": timestamp,
           "known_at": "2019-12-01T00:00:00Z", "scope": "global", "scope_value": ""}
    row.update(values)
    return pd.DataFrame([row])


@pytest.mark.parametrize("invalid", [
    {"stop_loss_pct": .02, "stop_loss_atr": 1}, {"trailing_stop_pct": 0},
    {"fees_bps": float("nan")}, {"initial_capital": -1},
    {"execution": "open_proxy"}, {"event_policy": "magical"},
    {"allocation_mode": "magical"}, {"target_gross_exposure": 0},
    {"atr_window": True}, {"break_even_activation_pct": .02, "break_even_offset_pct": .03},
])
def test_invalid_config(invalid):
    with pytest.raises((ValueError, TypeError)):
        TradingConfig(**invalid)


def test_decoders_reuse_existing_semantics():
    assert targets_from_labels([2, 1, 0], label_semantics="action").tolist() == [1, 1, -1]
    assert targets_from_labels([2, 1, 0], label_semantics="target_position").tolist() == [1, 0, -1]
    np.testing.assert_allclose(targets_from_probabilities([[.2, .3, .5]]), [.3])


def test_optional_rules_disabled_and_no_rule_identity():
    b = pd.concat([bars(ticker="A"), bars(ticker="B")], ignore_index=True)
    q = np.array([.3, .5, 0, -.2, -.3, 0] * 2)
    base = TradingConfig(initial_capital=1000, fees_bps=13, slippage_bps=7)
    off = run_trading_backtest(b, q, replace(base, stop_loss_pct=.001, max_drawdown=.01))
    plain = run_trading_backtest(b, q, base)
    on = run_trading_backtest(b, q, replace(base, enabled=True))
    pd.testing.assert_frame_equal(off.equity, plain.equity)
    pd.testing.assert_frame_equal(on.equity, plain.equity)
    pd.testing.assert_frame_equal(on.orders, plain.orders)
    assert off.state.cash == off.metrics["final_capital"]


def test_next_bar_does_not_earn_previous_return():
    b = bars([100, 150, 150], [150, 150, 150])
    result = run_trading_backtest(b, [1, 1, 1], cfg())
    assert result.metrics["net_return"] == 0
    assert result.orders.iloc[0].base_price == 150
    assert result.orders.iloc[0].date == pd.Timestamp("2020-01-02", tz="UTC")


def test_continuous_quantities_cash_resize_and_flip():
    result = run_trading_backtest(bars(), [.5, .25, -.5, 0, 0, 0], cfg())
    q = result.orders.quantity.tolist()
    assert q == [5., -2.5, -2.5, -5., 5.]
    assert result.orders.iloc[2].timestamp == result.orders.iloc[3].timestamp
    assert result.metrics["net_pnl"] == 0
    assert result.orders.iloc[0].quantity_after == 5
    assert result.state.positions["A"].quantity == 0


def test_fees_and_slippage_are_charged_per_actual_fill():
    result = run_trading_backtest(bars(count=3), [.5, .5, .5],
                                  TradingConfig(fees_bps=10, slippage_bps=20, initial_capital=1000))
    orders = result.orders
    np.testing.assert_allclose(orders.fee, orders.quantity.abs() * orders.execution_price * .001)
    np.testing.assert_allclose(orders.slippage_cost, orders.quantity.abs() * .2)
    assert np.isclose(result.metrics["net_pnl"], -orders.fee.sum() - orders.slippage_cost.sum())
    assert orders.iloc[0].execution_price == pytest.approx(100.2)
    assert orders.iloc[-1].execution_price == pytest.approx(99.8)


@pytest.mark.parametrize("direction,level,high,low,reason", [
    (1, 105, 106, 99, "take_profit"), (1, 95, 101, 94, "stop_loss"),
    (-1, 95, 101, 94, "take_profit"), (-1, 105, 106, 99, "stop_loss"),
])
def test_long_short_protective_exits(direction, level, high, low, reason):
    b = bars(highs=[101, high, 101, 101, 101, 101], lows=[99, low, 99, 99, 99, 99])
    result = run_trading_backtest(b, [direction] * 6, cfg(stop_loss_pct=.05, take_profit_pct=.05))
    assert len(result.trades) == 1
    assert result.trades.iloc[0].exit_price == pytest.approx(level)
    assert result.trades.iloc[0].exit_reason == reason
    assert result.trades.iloc[0].net_pnl == pytest.approx(50 if reason == "take_profit" else -50)
    assert result.orders.iloc[-1].phase == "intrabar_unknown"
    assert result.decisions.raw_target.iloc[-1] == direction
    assert result.positions.weight.iloc[-1] == 0


def test_tp_sl_conflict_chooses_stop_and_reports_ambiguity():
    b = bars(highs=[101, 110, 101, 101, 101, 101], lows=[99, 90, 99, 99, 99, 99])
    r = run_trading_backtest(b, [1] * 6, cfg(stop_loss_pct=.05, take_profit_pct=.05))
    assert r.trades.iloc[0].exit_price == 95
    assert r.trades.iloc[0].exit_reason == "stop_loss_tp_conflict"
    assert r.metadata["ambiguous_tp_stop_bars"] == 1


@pytest.mark.parametrize("direction,opening,expected", [(1, 90, 90), (-1, 110, 110)])
def test_gap_stop_fills_at_adverse_open_before_signal(direction, opening, expected):
    b = bars([100, 100, opening, 100], [100, 100, opening, 100])
    r = run_trading_backtest(b, [direction, -direction, -direction, 0], cfg(stop_loss_pct=.05))
    first = r.trades.iloc[0]
    assert first.exit_price == expected
    assert first.exit_reason == "stop_loss"
    assert first.exit_phase == "open_gap"
    assert len(r.orders.loc[r.orders.date.eq(pd.Timestamp("2020-01-03", tz="UTC"))]) == 1


def test_gap_tp_fills_conservatively_at_limit():
    b = bars([100, 100, 120, 120], [100, 100, 120, 120])
    r = run_trading_backtest(b, [1] * 4, cfg(take_profit_pct=.05))
    assert r.trades.iloc[0].exit_price == 105


def test_trailing_only_uses_completed_extrema_and_never_loosen():
    b = bars([100, 100, 110, 108], [100, 110, 108, 108],
             highs=[101, 120, 112, 109], lows=[99, 99, 107, 107])
    r = run_trading_backtest(b, [1] * 4, cfg(trailing_stop_pct=.1))
    # Current bar high=120 cannot retroactively turn the low=99 into a stop at108.
    assert r.positions.iloc[1].quantity > 0
    assert r.positions.iloc[1].stop_price == 108
    assert r.trades.iloc[0].exit_price == 108
    assert r.trades.iloc[0].exit_reason == "trailing_stop"


def test_trailing_activation_and_break_even_are_next_bar_rules():
    b = bars([100, 100, 101, 101], [100, 103, 101, 101],
             highs=[101, 104, 102, 102], lows=[99, 99, 99, 100])
    r = run_trading_backtest(b, [1] * 4, cfg(trailing_stop_pct=.1,
        trailing_activation_pct=.05, break_even_activation_pct=.03))
    assert r.positions.iloc[1].stop_price == 100
    assert r.trades.iloc[0].exit_price == 100
    assert r.trades.iloc[0].exit_reason == "break_even"


def test_resize_does_not_reset_anchor_or_trailing():
    b = bars([100, 100, 110, 115, 115], [100, 110, 115, 115, 115])
    r = run_trading_backtest(b, [.3, .6, .6, .6, 0], cfg(trailing_stop_pct=.2))
    assert r.trades.iloc[0].entry_price == 100
    assert r.orders.iloc[1].reason == "resize_add"
    assert r.positions.iloc[2].holding_bars == 2


def test_atr_warmup_and_entry_distance_are_causal_and_frozen():
    b = bars(count=6)
    b.loc[3, ["high", "low"]] = [105, 95]
    r = run_trading_backtest(b, [1] * 6, cfg(atr_window=2, stop_loss_atr=1))
    assert r.orders.iloc[0].date == pd.Timestamp("2020-01-03", tz="UTC")
    assert r.positions.iloc[2].stop_price == 98
    assert r.trades.iloc[0].exit_price == 98
    assert "atr_warmup" in r.decisions.reasons.iloc[1]


def test_duration_exit_ignores_band_and_blocks_reentry():
    r = run_trading_backtest(bars(), [1] * 6, cfg(max_holding_bars=1, no_trade_band=.05))
    assert len(r.trades) == 1
    assert r.trades.iloc[0].exit_reason == "max_holding"
    assert r.trades.iloc[0].exit_time == pd.Timestamp("2020-01-03T08:00:00Z")


def test_new_signal_unlocks_forced_exit_after_flat():
    b = bars(highs=[101] * 6, lows=[99, 90, 99, 99, 99, 99])
    r = run_trading_backtest(b, [1, .8, 0, 1, 1, 1], cfg(stop_loss_pct=.05))
    entries = r.orders.loc[r.orders.reason.eq("signal_entry")]
    assert entries.date.tolist() == [pd.Timestamp("2020-01-02", tz="UTC"), pd.Timestamp("2020-01-05", tz="UTC")]


@pytest.mark.parametrize("mode,cooldown,day", [("next_bar", 0, 3), ("cooldown", 1, 4)])
def test_alternative_reentry(mode, cooldown, day):
    b = bars(highs=[101] * 6, lows=[99, 90, 99, 99, 99, 99])
    r = run_trading_backtest(b, [1] * 6, cfg(stop_loss_pct=.05, reentry=mode, cooldown_bars=cooldown))
    entries = r.orders.loc[r.orders.reason.eq("signal_entry")]
    assert entries.iloc[1].date == pd.Timestamp(f"2020-01-0{day}", tz="UTC")


def test_no_trade_band_holds_position():
    r = run_trading_backtest(bars(), [.4, .405, .4, .4, .4, .4], cfg(no_trade_band=.01))
    assert len(r.orders) == 2
    assert r.decisions.reasons.str.contains("no_trade_band").any()


def test_band_does_not_trade_a_held_asset_when_another_asset_pays_fees():
    a, b = bars(ticker="A"), bars(ticker="B")
    a["position"] = .4
    b["position"] = [0, .6, .6, .6, .6, .6]
    market = pd.concat([a, b], ignore_index=True)
    r = run_trading_backtest(market, market[["date", "ticker", "position"]],
                             replace(cfg(no_trade_band=.02), fees_bps=10, slippage_bps=5))
    held = r.orders.loc[r.orders.ticker.eq("A")]
    assert held.reason.tolist() == ["signal_entry", "terminal_close"]
    assert r.orders.loc[r.orders.ticker.eq("B")].iloc[0].date.day == 3
    assert r.metrics["net_pnl"] == pytest.approx(-r.metrics["fees"] - r.metrics["slippage_cost"])


def test_short_trailing_tightens_from_completed_low_only():
    b = bars([100, 100, 96, 96], [100, 95, 96, 96],
             highs=[101, 102, 100, 97], lows=[99, 90, 94, 95])
    r = run_trading_backtest(b, [-1] * 4, cfg(trailing_stop_pct=.1))
    assert r.positions.iloc[1].quantity < 0
    assert r.positions.iloc[1].stop_price == pytest.approx(99)
    assert r.trades.iloc[0].exit_price == pytest.approx(99)
    assert r.trades.iloc[0].exit_reason == "trailing_stop"


def test_short_break_even_activates_without_another_stop_rule():
    b = bars([100, 100, 99, 99], [100, 98, 99, 99],
             highs=[101, 102, 100, 100], lows=[99, 96, 98, 98])
    r = run_trading_backtest(b, [-1] * 4,
                             cfg(break_even_activation_pct=.03, break_even_offset_pct=.001))
    assert r.positions.iloc[1].quantity < 0
    assert r.positions.iloc[1].stop_price == pytest.approx(99.9)
    assert r.trades.iloc[0].exit_price == pytest.approx(99.9)
    assert r.trades.iloc[0].exit_reason == "break_even"


def test_atr_trailing_keeps_entry_atr_when_current_range_grows():
    b = bars([100, 100, 100, 102, 102], [100, 100, 103, 102, 102],
             highs=[101, 101, 104, 103, 103], lows=[99, 99, 99, 101, 101])
    r = run_trading_backtest(b, [1] * 5,
                             cfg(atr_window=2, trailing_stop_atr=1, trailing_activation_pct=.02))
    assert r.positions.iloc[2].stop_price == 102  # high104 minus entry ATR2
    assert r.trades.iloc[0].exit_price == 102
    assert r.trades.iloc[0].exit_time.day == 4


def test_short_atr_take_profit():
    b = bars(count=5)
    b.loc[2, "low"] = 95
    r = run_trading_backtest(b, [-1] * 5, cfg(atr_window=2, take_profit_atr=2))
    assert r.trades.iloc[0].exit_price == 96
    assert r.trades.iloc[0].exit_reason == "take_profit"


@pytest.mark.parametrize("side,rule,blocked", [
    (1, "take_profit_atr", False), (-1, "stop_loss_atr", False),
    (1, "stop_loss_atr", True), (-1, "take_profit_atr", True),
])
def test_wide_atr_rejects_only_nonpositive_barriers(side, rule, blocked):
    b = bars(count=5, highs=[200] * 5, lows=[10] * 5)
    r = run_trading_backtest(b, [side] * 5, cfg(atr_window=2, **{rule: 2}))
    assert r.orders.empty == blocked
    assert r.decisions.reasons.str.contains("invalid_barrier_distance").any() == blocked


@pytest.mark.parametrize("side", [1, -1])
def test_stop_already_crossed_due_to_entry_slippage_fills_at_market_open(side):
    r = run_trading_backtest(bars(), [side] * 6,
                             replace(cfg(stop_loss_pct=.001), slippage_bps=100))
    exit_order = r.orders.iloc[1]
    assert exit_order.reason == "stop_loss"
    assert exit_order.phase == "open_gap"
    assert exit_order.base_price == 100
    assert exit_order.execution_price == pytest.approx(100 * (1 - side * .01))
    assert len(r.trades) == 1


def test_opposite_signal_releases_lock_on_bar_after_exit():
    b = bars([100, 100, 90, 100, 100], [100, 100, 90, 100, 100])
    r = run_trading_backtest(b, [1, -1, -1, -1, 0], cfg(stop_loss_pct=.05))
    entries = r.orders.loc[r.orders.reason.eq("signal_entry")]
    assert entries.date.dt.day.tolist() == [2, 4]
    assert entries.iloc[1].quantity < 0


def test_protection_ignores_band_and_reentry_cooldown():
    b = bars(lows=[99, 90, 99, 99, 99, 99])
    r = run_trading_backtest(b, [1] * 6,
                             cfg(stop_loss_pct=.05, no_trade_band=.9, reentry="cooldown", cooldown_bars=10))
    assert r.trades.exit_reason.tolist() == ["stop_loss"]
    assert len(r.orders) == 2


def test_missing_common_session_is_not_invented():
    b = bars(count=4)
    b.date = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"])
    r = run_trading_backtest(b, [1, 0, 0, 0], cfg())
    assert r.orders.date.dt.day.tolist() == [2, 4]


def test_portfolio_capital_once_and_limits_after_fees():
    b = pd.concat([bars(ticker="A"), bars(ticker="B")], ignore_index=True)
    b["sector"] = "bank"
    configuration = replace(cfg(max_asset_weight=.3, max_sector_weight=.4,
        max_gross_exposure=.5, max_net_exposure=.4), fees_bps=100, slippage_bps=50)
    r = run_trading_backtest(b, np.ones(len(b)), configuration)
    assert r.equity.equity.iloc[0] == 1000
    assert r.positions.weight.abs().max() <= .3 + 1e-8
    assert r.equity.gross_exposure.max() <= .4 + 1e-8
    assert r.decisions.actual_weight.groupby(r.decisions.date).sum().max() <= .4 + 1e-8
    assert r.metrics["net_pnl"] == pytest.approx(-r.metrics["fees"] - r.metrics["slippage_cost"])


def test_equal_active_excludes_flats_and_resizes_when_active_set_changes():
    tickers = list("ABCDEFGH")
    market = pd.concat([bars(count=4, ticker=ticker) for ticker in tickers], ignore_index=True)
    targets = market[["date", "ticker"]].copy()
    by_date = {
        market.date.unique()[0]: [1, 1, 1, -1, -1, -1, 0, 0],
        market.date.unique()[1]: [1, 1, 1, -1, -1, -1, 1, 0],
        market.date.unique()[2]: [1, 1, 1, -1, -1, 0, 0, 0],
        market.date.unique()[3]: [0] * 8,
    }
    targets["target_position"] = [
        by_date[date][tickers.index(ticker)]
        for date, ticker in zip(targets.date, targets.ticker)
    ]
    result = run_trading_backtest(
        market,
        targets,
        cfg(allocation_mode="equal_active", target_gross_exposure=.96),
    )
    decisions = result.decisions.set_index(["date", "ticker"])
    first = pd.Timestamp("2020-01-02", tz="UTC")
    second = pd.Timestamp("2020-01-03", tz="UTC")
    np.testing.assert_allclose(
        decisions.loc[first].requested_weight.to_numpy(),
        [.16, .16, .16, -.16, -.16, -.16, 0, 0],
    )
    np.testing.assert_allclose(
        decisions.loc[second].requested_weight.to_numpy(),
        [*((.96 / 7) * np.array([1, 1, 1, -1, -1, -1, 1])), 0],
    )
    second_orders = result.orders[result.orders.date.eq(second)]
    assert set(second_orders.reason) == {"resize_reduce", "signal_entry"}
    assert second_orders.loc[second_orders.ticker.eq("G"), "quantity_after"].iloc[0] > 0


def test_equal_active_target_is_capped_without_forcing_infeasible_exposure():
    tickers = list("ABCDEFGH")
    market = pd.concat([bars(count=3, ticker=ticker) for ticker in tickers], ignore_index=True)
    directions = dict(zip(tickers, [1, 1, 1, -1, -1, -1, 0, 0]))
    targets = market[["date", "ticker"]].copy()
    targets["target_position"] = targets.ticker.map(directions)
    result = run_trading_backtest(
        market,
        targets,
        cfg(
            allocation_mode="equal_active",
            target_gross_exposure=.96,
            max_asset_weight=.10,
        ),
    )
    action = result.decisions[result.decisions.date.eq(pd.Timestamp("2020-01-02", tz="UTC"))]
    assert action.requested_weight.abs().sum() == pytest.approx(.96)
    assert action.target_weight.abs().sum() == pytest.approx(.60)
    assert action.actual_weight.abs().sum() == pytest.approx(.60)


def test_fixed_universe_remains_the_default_allocation():
    tickers = list("ABCDEFGH")
    market = pd.concat([bars(count=3, ticker=ticker) for ticker in tickers], ignore_index=True)
    directions = dict(zip(tickers, [1, 1, 1, -1, -1, -1, 0, 0]))
    targets = market[["date", "ticker"]].copy()
    targets["target_position"] = targets.ticker.map(directions)
    result = run_trading_backtest(market, targets, cfg())
    action = result.decisions[result.decisions.date.eq(pd.Timestamp("2020-01-02", tz="UTC"))]
    assert action.requested_weight.abs().sum() == pytest.approx(6 / 8)
    assert action.requested_weight.abs().max() == pytest.approx(1 / 8)


def test_risk_limits_override_turnover_buffer():
    b = bars([100, 100, 100, 100], [100, 80, 80, 80])
    r = run_trading_backtest(b, [-1] * 4, cfg(max_asset_weight=.25, no_trade_band=.5))
    assert r.decisions.actual_weight.abs().max() <= .25 + 1e-8


def test_volatility_size_uses_history_before_open():
    b = bars([100, 100, 110, 95, 100, 100], [100, 110, 95, 100, 100, 100])
    r = run_trading_backtest(b, [1] * 6, cfg(volatility_target=.01, volatility_window=2))
    assert "volatility_warmup" in r.decisions.reasons.iloc[1]
    assert r.orders.iloc[0].date == pd.Timestamp("2020-01-04", tz="UTC")
    assert r.decisions.actual_weight.abs().max() < .05


def test_drawdown_halt_liquidates_at_next_action_and_stays_halted():
    b = bars([100, 100, 80, 100, 100], [100, 80, 100, 100, 100])
    r = run_trading_backtest(b, [1] * 5, cfg(max_drawdown=.1))
    assert r.state.halted
    assert r.trades.iloc[0].exit_reason == "drawdown_halt"
    assert r.trades.iloc[0].exit_price == 80
    assert len(r.trades) == 1


@pytest.mark.parametrize("policy,expected", [("block_increases", 0), ("reduce", .5), ("flat", 0)])
def test_calendar_policies(policy, expected):
    r = run_trading_backtest(bars(), [1] * 6, cfg(event_policy=policy), events=event())
    assert r.decisions.actual_weight.iloc[1] == expected
    assert r.decisions.event_ids.iloc[1] == "event-0"


def test_late_known_calendar_does_not_leak_and_skipping_orders_keeps_exposure():
    r = run_trading_backtest(bars(), [1] * 6, cfg(event_policy="flat"),
                            events=event(known_at="2020-01-02T10:00:00Z"))
    assert r.decisions.actual_weight.iloc[1] == 1
    r = run_trading_backtest(bars(), [.5, 1, 1, 1, 1, 1], cfg(event_policy="block_increases"),
                            events=event(timestamp="2020-01-03T12:00:00Z"))
    assert r.decisions.actual_weight.iloc[2] == .5


def test_nocturnal_event_flattens_at_last_available_action():
    r = run_trading_backtest(bars(), [1] * 6, cfg(event_policy="flat", execution="next_close"),
                            events=event(timestamp="2020-01-03T20:00:00Z"))
    assert r.decisions.actual_weight.iloc[1] == 1
    assert r.decisions.actual_weight.iloc[2] == 0
    assert r.trades.iloc[0].exit_reason == "event_flat"


def test_sector_ticker_scope_overlap_and_dst():
    a, b = bars(ticker="A"), bars(ticker="B")
    a["sector"], b["sector"] = "bank", "tech"
    market = pd.concat([a, b], ignore_index=True)
    events = pd.concat([event(scope="sector", scope_value="bank"), event(scope="ticker", scope_value="A")])
    r = run_trading_backtest(market, np.ones(len(market)), cfg(event_policy="flat"), events=events)
    decisions = r.decisions.loc[r.decisions.date.eq(pd.Timestamp("2020-01-02", tz="UTC"))].set_index("ticker")
    assert decisions.loc["A", "actual_weight"] == 0
    assert decisions.loc["B", "actual_weight"] == .5
    assert len(decisions.loc["A", "event_ids"].split("|")) == 2
    market = bars(count=4)
    market.date = pd.to_datetime(["2020-03-27", "2020-03-30", "2020-03-31", "2020-04-01"])
    r = run_trading_backtest(market, np.ones(4), cfg())
    assert r.decisions.timestamp.iloc[0].hour == 8
    assert r.decisions.timestamp.iloc[1].hour == 7


def test_enabled_events_require_real_calendar_and_sector_metadata():
    with pytest.raises(ValueError, match="usable calendar"):
        run_trading_backtest(bars(), [1] * 6, cfg(event_policy="flat"))
    with pytest.raises(ValueError, match="sector metadata"):
        run_trading_backtest(bars(), [1] * 6, cfg(event_policy="flat"), events=event(scope="sector", scope_value="bank"))


def test_raw_split_changes_quantity_and_stops_without_fake_pnl():
    b = bars([100, 100, 50, 50], [100, 100, 50, 50])
    b["stock_splits"] = [0, 0, 2, 0]
    r = run_trading_backtest(b, [1] * 4, cfg(price_basis="raw", stop_loss_pct=.1))
    assert r.metrics["net_pnl"] == 0
    assert r.positions.iloc[2].quantity == 20
    assert r.positions.iloc[2].stop_price == 45
    assert len(r.orders) == 2


def test_split_adjusted_input_does_not_adjust_splits_twice():
    b = bars(count=4)
    b["stock_splits"] = [0, 0, 2, 0]
    b["adj_close"] = [50, 70, 90, 10]  # never used as a trading mark
    r = run_trading_backtest(b, [1] * 4, cfg(stop_loss_pct=.1))
    assert r.metrics["net_pnl"] == 0
    assert r.positions.iloc[2].quantity == 10
    assert r.positions.iloc[2].stop_price == 90


@pytest.mark.parametrize("side", [1, -1])
def test_dividend_accrues_to_prior_holder_including_short(side):
    b = bars(count=4)
    b["dividends"] = [0, 0, 2, 0]
    r = run_trading_backtest(b, [side] * 4, cfg())
    assert r.metrics["net_pnl"] == pytest.approx(side * 20)
    assert r.trades.dividends.sum() == pytest.approx(side * 20)
    assert r.trades.net_pnl.sum() == pytest.approx(r.metrics["net_pnl"])


def test_array_order_and_keyed_targets_are_identical():
    a, b = bars(ticker="A"), bars(ticker="B")
    a["target_position"], b["target_position"] = .3, -.4
    market = pd.concat([b.iloc[::-1], a.iloc[::-1]], ignore_index=True)
    r = run_trading_backtest(market, market.target_position.to_numpy(), cfg())
    keyed = run_trading_backtest(market, market[["date", "ticker", "target_position"]], cfg())
    pd.testing.assert_frame_equal(r.orders, keyed.orders)


@pytest.mark.parametrize("defect", ["missing_bar", "duplicate", "bad_ohlc", "nan", "missing_target"])
def test_bad_inputs_rejected(defect):
    b = pd.concat([bars(ticker="A"), bars(ticker="B")], ignore_index=True)
    targets = b[["date", "ticker"]].assign(target_position=.5)
    if defect == "missing_bar":
        b = b.iloc[:-1]
    elif defect == "duplicate":
        b = pd.concat([b, b.iloc[[0]]])
    elif defect == "bad_ohlc":
        b.loc[0, "high"] = 1
    elif defect == "nan":
        b.loc[0, "close"] = np.nan
    else:
        targets = targets.iloc[:-1]
    with pytest.raises(ValueError):
        run_trading_backtest(b, targets, cfg())


def test_after_open_executions_are_delayed_unless_explicit_proxy():
    b = bars([100, 110, 120], [105, 115, 125])
    delayed = run_trading_backtest(b, [.5] * 3, cfg(signal_timing="after_open"))
    proxy = run_trading_backtest(b, [.5] * 3, cfg(signal_timing="after_open", execution="open_proxy"))
    close = run_trading_backtest(b, [.5] * 3, cfg(signal_timing="after_open", execution="next_close"))
    assert delayed.orders.iloc[0].date.day == 2
    assert proxy.orders.iloc[0].date.day == 1
    assert proxy.metadata["execution_is_proxy"]
    assert close.orders.iloc[0].base_price == 105
    assert not close.metadata["execution_is_proxy"]


def test_intraday_and_late_signal_availability():
    b = bars(count=4)
    b.date = pd.date_range("2020-01-01T10:00:00Z", periods=4, freq="h")
    b["open_time"] = b.date
    b["close_time"] = b.date + pd.Timedelta(minutes=30)
    targets = b[["date", "ticker"]].assign(target_position=1)
    targets["signal_available_at"] = b.close_time + pd.Timedelta(minutes=40)
    r = run_trading_backtest(b, targets, cfg(bar_mode="intraday"))
    assert r.orders.iloc[0].timestamp == b.open_time.iloc[2]
    assert not r.metadata["session_times_inferred"]
    with pytest.raises(ValueError, match="explicit open_time"):
        run_trading_backtest(b.drop(columns=["open_time", "close_time"]), [1] * 4, cfg(bar_mode="intraday"))


def test_contiguous_intraday_bars_allow_shared_boundary_but_delay_observed_close_signal():
    b = bars(count=4)
    b.date = pd.date_range("2020-01-01T10:00:00Z", periods=4, freq="30min")
    b["open_time"] = b.date
    b["close_time"] = b.date + pd.Timedelta(minutes=30)
    r = run_trading_backtest(b, [1] * 4, cfg(bar_mode="intraday"))
    assert r.orders.iloc[0].timestamp == pd.Timestamp("2020-01-01T11:00:00Z")


@pytest.mark.parametrize("defect", ["empty", "missing_date"])
def test_cli_invalid_signal_keys_rejected_before_writing(tmp_path, defect):
    source = bars(count=4).assign(position=.5)
    if defect == "empty":
        source = source.iloc[:0]
    else:
        source.loc[0, "date"] = pd.NaT
    path = tmp_path / "signals.parquet"
    source.to_parquet(path, index=False)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="non-empty with valid"):
        replay_signals(path, cfg(), output)
    assert not output.exists()


def test_cli_separates_groups_writes_hashes_and_guards_holdout(tmp_path):
    b = bars(count=4)
    source = pd.concat([b.assign(position=.5, fold=0, seed=1), b.assign(position=-.5, fold=0, seed=7)])
    signal_path = tmp_path / "signals.parquet"
    source.to_parquet(signal_path, index=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"take_profit_pct": .05, "fees_bps": 0}))
    output = tmp_path / "replay"
    main([
        "--signals", str(signal_path),
        "--trading-config", str(config_path),
        "--allocation-mode", "equal_active",
        "--target-gross-exposure", ".98",
        "--output-dir", str(output),
    ])
    report = json.loads((output / "report.json").read_text())
    assert len(report["comparisons"]) == 2
    assert report["config"]["allocation_mode"] == "equal_active"
    assert report["config"]["target_gross_exposure"] == .98
    assert not report["final_holdout_opened"]
    assert len(report["inputs_sha256"]["signals"]) == 64
    assert (output / "group-001" / "with_rules" / "orders.parquet").exists()
    with pytest.raises(FileExistsError):
        replay_signals(signal_path, cfg(), output)
    source.date = pd.to_datetime("2023-01-01") + pd.to_timedelta(source.date.dt.day, unit="D")
    source.to_parquet(signal_path, index=False)
    with pytest.raises(ValueError, match="sealed final holdout"):
        replay_signals(signal_path, cfg(), tmp_path / "sealed")
    assert not (tmp_path / "sealed").exists()
    report = replay_signals(signal_path, cfg(), tmp_path / "explicit", allow_final_test=True)
    assert report["final_holdout_opened"]
