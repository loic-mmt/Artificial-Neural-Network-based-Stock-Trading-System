from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from trading_system.trading import (
    AffineCosts, OUConfig, OUParameters, TradingConfig,
    estimate_ou, run_trading_backtest, solve_ou_boundaries,
)
from trading_system.trading.ou_policy import OUPolicy
from trading_system.trading.replay import replay_signals
from trading_system.trading.ou_validation import validate_directory


def bars(prices, *, start="2020-01-01"):
    prices = np.asarray(prices, dtype=float)
    return pd.DataFrame(dict(date=pd.date_range(start, periods=len(prices), freq="B"),
                             ticker="A", open=prices, close=prices, high=prices, low=prices))


def config(mode="exit", **kwargs):
    return TradingConfig(enabled=True, trailing_stop_pct=.1, fees_bps=0,
                         ou=OUConfig(mode=mode, window=80, refit_every=5), **kwargs)


def synthetic(n=512, seed=3, rho=.85, mean=1., innovation=.04):
    rng = np.random.default_rng(seed)
    y = np.empty(n)
    y[0] = mean
    for i in range(1, n):
        y[i] = mean + rho*(y[i-1]-mean) + rng.normal(0, innovation)
    return np.exp(y)


def boundary_stub(monkeypatch, *, buy=100., sell=120., valid=True):
    def snapshot(self, ticker, now, bar):
        return dict(valid=valid, reason="ok" if valid else "unit_root_uncertainty",
                    fit_version=1, fit_cutoff=pd.Timestamp("2019-12-31T16:30Z"),
                    buy_price=buy, sell_price=sell, acquisition_value=1.)
    monkeypatch.setattr(OUPolicy, "snapshot", snapshot)


def test_paper_benchmark_and_convergence():
    params = OUParameters(.6, 1., .2)
    costs = AffineCosts(buy_fixed=.02, sell_fixed=.02)
    result = solve_ou_boundaries(params, .3, costs=costs)
    fine = solve_ou_boundaries(params, .3, costs=costs, rtol=1e-10, domain_sigma=11)
    assert result.sell_price == pytest.approx(2.8844631362, abs=2e-8)
    assert result.buy_price == pytest.approx(1.9488307672, abs=2e-7)
    assert result.buy_price == pytest.approx(fine.buy_price, abs=2e-7)
    assert result.acquisition_value > 0 and result.root_residual < 1e-8


def test_costs_and_price_units():
    params = OUParameters(.6, 1., .2)
    original = solve_ou_boundaries(params, .3, costs=AffineCosts(buy_fixed=.02, sell_fixed=.02))
    scaled = solve_ou_boundaries(replace(params, log_mean=1+np.log(100)), .3,
                                  costs=AffineCosts(buy_fixed=2, sell_fixed=2))
    assert scaled.buy_price == pytest.approx(original.buy_price*100)
    assert scaled.sell_price == pytest.approx(original.sell_price*100)
    expensive = solve_ou_boundaries(params, .3, costs=AffineCosts(buy_fixed=.1, sell_fixed=.1))
    assert expensive.buy_price < original.buy_price
    assert expensive.sell_price > original.sell_price
    impossible = solve_ou_boundaries(params, .3, costs=AffineCosts(buy_fixed=50, sell_fixed=50))
    assert impossible.buy_price is None and impossible.status == "no_profitable_entry"
    friction = AffineCosts.from_bps(5, 2)
    assert friction.buy_factor == pytest.approx(1.0005*1.0002)
    assert friction.sell_factor == pytest.approx(.9995*.9998)
    assert friction.buy_fixed == 0


@pytest.mark.parametrize("kwargs", [dict(window=20), dict(refit_every=0), dict(discount_rate=0),
                                    dict(mode="magic"), dict(adf_pvalue=2)])
def test_bad_ou_config(kwargs):
    with pytest.raises(ValueError):
        OUConfig(**kwargs)


def test_policy_config_is_explicit():
    with pytest.raises(ValueError, match="trailing"):
        TradingConfig(ou={"mode": "exit"})
    with pytest.raises(ValueError, match="exclusive"):
        config(take_profit_pct=.1)
    with pytest.raises(ValueError, match="causal"):
        config(execution="open_proxy", signal_timing="after_open")
    assert TradingConfig(ou={"mode": "off"}).ou.mode == "off"


def test_fit_recovers_transition_and_rejects_invalid_regimes():
    settings = OUConfig(window=512)
    params, diag = estimate_ou(synthetic(), settings)
    assert diag["valid"] and params is not None
    assert diag["ar1"] == pytest.approx(.85, abs=.06)
    assert params.log_mean == pytest.approx(1., abs=.03)
    assert params.mean_reversion == pytest.approx(-np.log(.85)*252, rel=.4)
    for values in (np.ones(512), np.exp(np.linspace(0, 2, 512)),
                   np.exp(np.random.default_rng(42).normal(0, .01, 512).cumsum())):
        params, diag = estimate_ou(values, settings)
        assert params is None and not diag["valid"]
    params, diag = estimate_ou(synthetic(20), settings)
    assert params is None and diag["reason"] == "warmup"


def test_disabled_policy_preserves_ledger():
    b = bars([100, 110, 95, 120, 105])
    q = [.5, .3, -.4, 0, .2]
    c = config()
    off = run_trading_backtest(b, q, replace(c, enabled=False))
    old = run_trading_backtest(b, q, replace(c, enabled=False, ou=OUConfig()))
    pd.testing.assert_frame_equal(off.orders, old.orders)
    pd.testing.assert_frame_equal(off.equity, old.equity)


def test_fallback_is_trailing_only_or_entry_rejection():
    b = bars([100, 100, 120, 100, 100])
    fallback = run_trading_backtest(b, [.5]*5, config())
    trail = run_trading_backtest(b, [.5]*5, replace(config(), ou=OUConfig()))
    pd.testing.assert_frame_equal(fallback.equity, trail.equity)
    assert fallback.positions.take_profit.isna().all()
    assert fallback.decisions.reasons.str.contains("ou_fallback_warmup").any()
    blocked = run_trading_backtest(b, [.5]*5, config("entry_exit"))
    assert blocked.orders.empty
    assert blocked.decisions.reasons.str.contains("ou_entry_block").sum() == 4


def test_tp_frozen_and_gaps_use_conservative_boundary(monkeypatch):
    boundary_stub(monkeypatch)
    b = bars([100, 100, 110, 130, 100])
    result = run_trading_backtest(b, [.3, .5, .4, 0, 0], config())
    trade = result.trades.iloc[0]
    assert trade.exit_reason == "ou_take_profit"
    assert trade.exit_price == 120
    assert trade.ou_sell_price == 120
    assert result.orders.loc[result.orders.trade_id == trade.trade_id, "ou_fit_version"].eq(1).all()
    assert result.orders.timestamp.gt(result.orders.ou_fit_cutoff).all()
    assert result.decisions.iloc[3].reasons == "reentry_block"


def test_buy_limit_gaps_reentry_and_no_same_bar_reentry(monkeypatch):
    boundary_stub(monkeypatch)
    b = bars([100, 110, 90, 120, 90, 90])
    result = run_trading_backtest(b, [.5]*6, config("entry_exit"))
    entries = result.orders.query("quantity_before == 0")
    assert entries.date.dt.day.tolist() == [3, 7]
    assert entries.base_price.le(100).all()
    assert "ou_entry_block" in result.decisions.iloc[1].reasons
    assert "reentry_block" in result.decisions.iloc[3].reasons
    assert result.orders.query("reason == 'ou_take_profit'").base_price.tolist() == [120]


def test_conflict_signal_flat_short_and_priority(monkeypatch):
    boundary_stub(monkeypatch)
    b = bars([100, 100, 100, 100, 100, 100])
    b.loc[2, ["high", "low"]] = [125, 85]
    c = config("entry_exit", no_trade_band=.4, stop_loss_pct=.08)
    result = run_trading_backtest(b, [.5, .5, -.5, 0, 0, 0], c)
    assert result.metadata["ambiguous_tp_stop_bars"] == 1
    assert result.trades.iloc[0].exit_reason == "stop_loss_tp_conflict"
    shorts = result.orders.query("quantity_before == 0 and quantity < 0")
    assert len(shorts) == 1 and shorts.ou_sell_price.isna().all()
    assert result.trades.iloc[-1].exit_reason == "signal_exit"


def test_next_close_entry_ignores_earlier_high_and_trailing(monkeypatch):
    boundary_stub(monkeypatch)
    b = bars([100, 100, 100, 100])
    b.loc[1, "high"] = 130
    result = run_trading_backtest(b, [.5]*4, config(execution="next_close"))
    assert result.trades.iloc[0].exit_reason == "signal_exit"  # terminal close rebalance
    assert result.trades.iloc[0].exit_price == 100
    assert result.positions.iloc[1].stop_price == 90


def test_turnover_band_blocks_small_entry_but_not_flat_or_flip(monkeypatch):
    boundary_stub(monkeypatch)
    b = bars([100]*6)
    small = run_trading_backtest(b, [.1]*6, config("entry_exit", no_trade_band=.25))
    assert small.orders.empty
    result = run_trading_backtest(b, [.5, .1, -.1, 0, 0, 0], config(no_trade_band=.25))
    assert result.orders.reason.eq("signal_flip").sum() == 1
    assert result.trades.exit_reason.eq("signal_exit").sum() == 1
    assert result.state.positions["A"].quantity == 0


def test_history_and_future_mutations_do_not_change_prior_decisions():
    history = bars(synthetic(100), start="2019-01-01")
    b = bars(synthetic(25, seed=2))
    q = [.5]*len(b)
    c = config()
    first = run_trading_backtest(b, q, c, history=history)
    changed = b.copy()
    changed.loc[15:, ["open", "high", "low", "close"]] *= 1.8
    second = run_trading_backtest(changed, q, c, history=history)
    pd.testing.assert_frame_equal(first.decisions.iloc[:15], second.decisions.iloc[:15])
    early = first.decisions.dropna(subset=["ou_fit_cutoff"])
    assert early.ou_fit_cutoff.lt(early.timestamp).all()
    assert first.metadata["ou_policy"]["fit_reasons"].get("ok", 0) > 0
    with pytest.raises(ValueError, match="strictly before"):
        run_trading_backtest(b, q, c, history=b)


def test_raw_split_preserves_frozen_barriers_and_dividends(monkeypatch):
    boundary_stub(monkeypatch)
    b = bars([100, 100, 50, 50])
    b["stock_splits"] = [0, 0, 2, 0]
    b["dividends"] = [0, 0, 1, 0]
    raw = run_trading_backtest(b, [.5]*4, config(price_basis="raw"))
    assert raw.positions.iloc[2].take_profit == 60
    assert raw.trades.iloc[0].ou_sell_price == 60
    adjusted = b.copy()
    adjusted.loc[:1, ["open", "close", "high", "low"]] /= 2
    boundary_stub(monkeypatch, buy=50, sell=60)
    result = run_trading_backtest(adjusted, [.5]*4, config(price_basis="split_adjusted"))
    assert raw.metrics["net_pnl"] == pytest.approx(result.metrics["net_pnl"])


def test_actual_fit_split_units_and_future_actions_are_causal():
    history = bars(synthetic(100), start="2019-01-01")
    b = bars(synthetic(25, seed=2))
    b["stock_splits"] = 0.
    b.loc[10:, ["open", "high", "low", "close"]] /= 2
    b.loc[10, "stock_splits"] = 2.
    raw = run_trading_backtest(b, [.4]*25, config(price_basis="raw"), history=history)
    adjusted = b.copy()
    adjusted.loc[:9, ["open", "high", "low", "close"]] /= 2
    past_adjusted = history.copy()
    past_adjusted[["open", "high", "low", "close"]] /= 2
    split_adjusted = run_trading_backtest(adjusted, [.4]*25, config(), history=past_adjusted)
    np.testing.assert_allclose(raw.equity.equity, split_adjusted.equity.equity, rtol=1e-10)
    changed = b.copy()
    changed.loc[20, "stock_splits"] = 3.
    future = run_trading_backtest(changed, [.4]*25, config(price_basis="raw"), history=history)
    pd.testing.assert_frame_equal(raw.decisions.iloc[:20], future.decisions.iloc[:20])


def test_buy_limit_respects_delayed_signal_and_shared_portfolio_caps(monkeypatch):
    boundary_stub(monkeypatch)
    b = pd.concat([bars([90]*5), bars([90]*5).assign(ticker="B")], ignore_index=True)
    signals = b[["date", "ticker"]].copy()
    signals["target_position"] = .8
    # Every signal arrives two days late; earlier eligible prices cannot fill it.
    signals["signal_available_at"] = pd.to_datetime(signals.date, utc=True) + pd.Timedelta(days=2, hours=20)
    result = run_trading_backtest(b, signals, config("entry_exit", max_asset_weight=.2, max_gross_exposure=.3))
    assert result.orders.timestamp.min() > signals.signal_available_at.min()
    assert result.orders.timestamp.gt(result.orders.ou_fit_cutoff).all()
    assert result.decisions.groupby("timestamp").actual_weight.sum().le(.3+1e-9).all()
    assert result.decisions.actual_weight.le(.2+1e-9).all()
    assert result.metrics["net_pnl"] == pytest.approx(result.trades.net_pnl.sum())


def test_cli_history_and_holdout_guard(tmp_path):
    b = bars([100]*5)
    b["position"] = .5
    signals = tmp_path / "signals.parquet"
    b.to_parquet(signals)
    past = bars([100]*100, start="2019-01-01")
    # Future history rows are not made available to a group's fit.
    path = tmp_path / "history.parquet"
    pd.concat([past, bars([100]*3, start="2023-01-01")]).to_parquet(path)
    report = replay_signals(signals, config(), tmp_path / "report", history=path)
    assert not report["final_holdout_opened"]
    metadata = json.loads((tmp_path / "report/group-000/with_rules/metadata.json").read_text())
    assert metadata["history_sha256"] is not None
    assert all(r["fit_cutoff"] < "2022" for r in metadata["ou_policy"]["fit_records"])


def test_validation_driver_cannot_open_holdout(tmp_path):
    signals = tmp_path / "signals"
    signals.mkdir()
    b = bars([100]*5, start="2023-01-01")
    b["position"] = .5
    b.to_parquet(signals / "fold-0-seed-1-open_gap-positions.parquet")
    data = tmp_path / "data.parquet"
    b.to_parquet(data)
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="sealed final holdout"):
        validate_directory(signals, data, output, families=("open_gap",), workers=1)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and not manifest["final_holdout_opened"]
    assert not (output / "comparison.csv").exists()
