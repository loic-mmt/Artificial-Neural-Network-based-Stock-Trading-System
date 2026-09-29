"""Deterministic OHLC replay of optional rules over frozen model targets.

The legacy training and return-panel engines are deliberately not imported or
modified. Both arms of a rule comparison use this same accounting engine.
"""

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from trading_system.artifacts.serialization import stable_config_hash
from .config import TradingConfig
from .data import prepare_bars, prepare_targets
from .events import active_events, prepare_events
from .metrics import frame_fingerprint, portfolio_metrics
from .ou_policy import OUPolicy


EPS = 1e-10


@dataclass
class PositionState:
    quantity: float = 0.0
    entry_price: float | None = None
    entry_time: pd.Timestamp | None = None
    entry_bar: int = -1
    trade_id: int | None = None
    cash_flow: float = 0.0
    fees: float = 0.0
    dividends: float = 0.0
    entry_notional: float = 0.0
    atr_entry: float | None = None
    stop_price: float | None = None
    stop_reason: str | None = None
    take_profit: float | None = None
    favorable_price: float | None = None
    age: int = 0
    pending_duration_exit: bool = False
    blocked_direction: int = 0
    last_forced_exit_bar: int = -1
    ou_fit_version: int | None = None
    ou_fit_cutoff: pd.Timestamp | None = None
    ou_entry_buy: float | None = None
    ou_entry_sell: float | None = None


@dataclass
class TradingState:
    cash: float
    equity: float
    peak_equity: float
    positions: dict[str, PositionState] = field(default_factory=dict)
    halted: bool = False
    bar_index: int = -1


@dataclass
class TradingResult:
    equity: pd.DataFrame
    positions: pd.DataFrame
    orders: pd.DataFrame
    trades: pd.DataFrame
    decisions: pd.DataFrame
    metrics: dict
    metadata: dict
    state: TradingState


def _sign(value):
    return 0 if abs(value) < EPS else (1 if value > 0 else -1)


def _distance(config, prefix, price, atr):
    percent = getattr(config, prefix + "_pct")
    multiple = getattr(config, prefix + "_atr")
    return price * percent if percent is not None else (atr * multiple if multiple is not None else None)


class _Replay:
    def __init__(self, bars, targets, config, events, history=None):
        self.config = config
        self.bars, self.tickers, self.explicit_times = prepare_bars(bars, config)
        if not isinstance(targets, pd.DataFrame):
            values = np.asarray(targets, dtype=float)
            if values.shape != (len(bars),):
                raise ValueError("Target array must have one element per ORIGINAL input bar row.")
            targets = bars[["date", "ticker"]].copy()
            targets["target_position"] = values
        self.targets = prepare_targets(self.bars, targets, config)
        self.events = prepare_events(events, self.bars, config)
        self.calendar = self.bars.drop_duplicates("date").reset_index(drop=True)
        self.groups = {ticker: part.reset_index(drop=True) for ticker, part in self.bars.groupby("ticker", sort=True)}
        self.history = None
        if history is not None and not history.empty:
            self.history, tickers, _ = prepare_bars(history, config)
            if tickers != self.tickers or self.history.close_time.max() >= self.bars.open_time.min():
                raise ValueError("OU history must have the same tickers and finish strictly before replay starts.")
        self.ou_policy = OUPolicy(self.bars, config, self.history) if config.enabled and config.ou.mode != "off" else None
        self.ou_snapshots = {}
        self.signals = {ticker: part.reset_index(drop=True) for ticker, part in self.targets.groupby("ticker", sort=True)}
        phase = "close_time" if config.execution == "next_close" else "open_time"
        self.actions = pd.DatetimeIndex(self.calendar[phase]).as_unit("ns")
        self.scheduled = {}
        for ticker in self.tickers:
            signal = self.signals[ticker]
            available = pd.DatetimeIndex(signal.signal_available_at).as_unit("ns")
            if not available.is_monotonic_increasing:
                raise ValueError("Signal availability must increase monotonically per ticker.")
            slots = np.arange(len(signal)) if config.execution == "open_proxy" else self.actions.searchsorted(available, side="right")
            if config.execution == "open_proxy" and (available > pd.DatetimeIndex(self.groups[ticker].close_time)).any():
                raise ValueError("An open proxy cannot use a signal unavailable until after its bar.")
            scheduled = np.full(len(self.actions), -1, dtype=int)
            for source, slot in enumerate(slots):
                if slot < len(scheduled):
                    scheduled[slot] = source  # latest available target wins a shared slot
            latest = -1
            for i in range(len(scheduled)):
                latest = scheduled[i] if scheduled[i] >= 0 else latest
                scheduled[i] = latest
            self.scheduled[ticker] = scheduled
        self.state = TradingState(config.initial_capital, config.initial_capital,
                                  config.initial_capital, {t: PositionState() for t in self.tickers})
        self.orders, self.trades, self.decisions, self.position_rows, self.equity_rows = [], [], [], [], []
        self.trade_counter = 0
        self.ambiguities = 0

    def equity_at(self, prices):
        return self.state.cash + sum(self.state.positions[t].quantity * prices[t] for t in self.tickers)

    def tighten_stop(self, pos, price, reason):
        direction = _sign(pos.quantity)
        if pos.stop_price is None or direction * (price - pos.stop_price) > EPS:
            pos.stop_price, pos.stop_reason = float(price), reason

    def fill(self, ticker, quantity, price, now, bar, reason, phase, marks, atr=None):
        if abs(quantity) < EPS:
            return
        pos = self.state.positions[ticker]
        old_quantity = pos.quantity
        if _sign(old_quantity) and _sign(old_quantity + quantity) not in (0, _sign(old_quantity)):
            raise RuntimeError("A flip must be two explicit fills.")
        direction = _sign(quantity)
        slip = self.config.slippage_bps * 1e-4
        execution_price = price * (1 + direction * slip)
        fee = abs(quantity * execution_price) * self.config.fees_bps * 1e-4
        equity_before = self.equity_at(marks)
        cash_flow = -quantity * execution_price - fee
        if not _sign(old_quantity):
            self.trade_counter += 1
            pos.trade_id = self.trade_counter
            pos.entry_price, pos.entry_time, pos.entry_bar = execution_price, now, bar
            pos.entry_notional = abs(quantity * execution_price)
            pos.atr_entry = atr
            pos.age = 0
            pos.cash_flow = pos.fees = pos.dividends = 0.
            pos.stop_price = pos.stop_reason = pos.take_profit = None
            pos.favorable_price = execution_price
            pos.pending_duration_exit = False
            snapshot = self.ou_snapshots.get(ticker, {}) if quantity > 0 else {}
            pos.ou_fit_version = snapshot.get("fit_version")
            pos.ou_fit_cutoff = snapshot.get("fit_cutoff")
            pos.ou_entry_buy = snapshot.get("buy_price") if snapshot.get("valid") else None
            pos.ou_entry_sell = snapshot.get("sell_price") if snapshot.get("valid") else None
        self.state.cash += cash_flow
        pos.cash_flow += cash_flow
        pos.fees += fee
        pos.quantity = old_quantity + quantity
        if abs(pos.quantity) < EPS:
            pos.quantity = 0.
        self.orders.append({
            "date": self.calendar.iloc[bar].date, "timestamp": now,
            "ticker": ticker, "trade_id": pos.trade_id, "quantity": quantity,
            "quantity_before": old_quantity, "quantity_after": pos.quantity,
            "base_price": price, "execution_price": execution_price,
            "notional": abs(quantity * execution_price), "fee": fee,
            "slippage_cost": abs(quantity) * abs(execution_price - price),
            "turnover": abs(quantity * price) / equity_before if equity_before > 0 else 0.,
            "reason": reason, "phase": phase,
            "earliest_time": self.calendar.iloc[bar].open_time if phase == "intrabar_unknown" else now,
            "latest_time": now,
            "ou_fit_version": pos.ou_fit_version, "ou_fit_cutoff": pos.ou_fit_cutoff,
            "ou_buy_price": pos.ou_entry_buy, "ou_sell_price": pos.ou_entry_sell,
        })
        if not _sign(pos.quantity):
            self.trades.append({
                "ticker": ticker, "trade_id": pos.trade_id,
                "side": "long" if old_quantity > 0 else "short",
                "entry_time": pos.entry_time, "exit_time": now,
                "entry_price": pos.entry_price, "exit_price": execution_price,
                "entry_notional": pos.entry_notional, "net_pnl": pos.cash_flow,
                "fees": pos.fees, "dividends": pos.dividends,
                "holding_bars": pos.age, "exit_reason": reason,
                "exit_phase": phase,
                "ou_fit_version": pos.ou_fit_version, "ou_fit_cutoff": pos.ou_fit_cutoff,
                "ou_buy_price": pos.ou_entry_buy, "ou_sell_price": pos.ou_entry_sell,
            })
            pos.entry_price = pos.entry_time = pos.favorable_price = None
            pos.stop_price = pos.stop_reason = pos.take_profit = None
            pos.pending_duration_exit = False
            pos.ou_fit_version = pos.ou_fit_cutoff = pos.ou_entry_buy = pos.ou_entry_sell = None
        elif not _sign(old_quantity) and self.config.enabled:
            sign = _sign(pos.quantity)
            distance = _distance(self.config, "stop_loss", execution_price, atr)
            if distance is not None:
                self.tighten_stop(pos, execution_price - sign * distance, "stop_loss")
            distance = _distance(self.config, "take_profit", execution_price, atr)
            if distance is not None:
                pos.take_profit = execution_price + sign * distance
            if pos.ou_entry_sell is not None:
                pos.take_profit = pos.ou_entry_sell
            # Paper X is the tradable price before costs, not the slipped fill.
            trail_anchor = price if self.ou_policy is not None and sign > 0 else execution_price
            if self.ou_policy is not None and sign > 0:
                pos.favorable_price = price
            if self.config.trailing_activation_pct == 0:
                distance = _distance(self.config, "trailing_stop", trail_anchor, atr)
                if distance is not None:
                    self.tighten_stop(pos, trail_anchor - sign * distance, "trailing_stop")

    def close(self, ticker, price, now, bar, reason, phase, marks, forced=False):
        pos = self.state.positions[ticker]
        old_sign = _sign(pos.quantity)
        self.fill(ticker, -pos.quantity, price, now, bar, reason, phase, marks)
        if forced and old_sign:
            pos.blocked_direction = old_sign
            pos.last_forced_exit_bar = bar

    def protections(self, rows, bar, gap_only):
        if not self.config.enabled:
            return
        marks = {t: float(rows[t]["open" if gap_only else "close"]) for t in self.tickers}
        for ticker, row in rows.items():
            pos = self.state.positions[ticker]
            direction = _sign(pos.quantity)
            if not direction:
                continue
            stop, profit = pos.stop_price, pos.take_profit
            low = high = float(row.open) if gap_only else None
            if not gap_only:
                low, high = float(row.low), float(row.high)
            hit_stop = stop is not None and (low <= stop if direction > 0 else high >= stop)
            hit_profit = profit is not None and (high >= profit if direction > 0 else low <= profit)
            if not (hit_stop or hit_profit):
                continue
            if hit_stop and hit_profit:
                self.ambiguities += 1
            if hit_stop:
                reason = pos.stop_reason or "stop_loss"
                crossed_at_open = row.open <= stop if direction > 0 else row.open >= stop
                price = float(row.open) if crossed_at_open else float(stop)
                if hit_profit:
                    reason += "_tp_conflict"
            else:
                reason, price = "ou_take_profit" if pos.ou_entry_sell is not None else "take_profit", float(profit)
                crossed_at_open = row.open >= profit if direction > 0 else row.open <= profit
            # A newly entered, very tight stop may already lie beyond the
            # market open due to entry slippage. Do not invent a better fill at
            # a barrier price that the market never traded.
            at_open = gap_only or crossed_at_open
            now = row.open_time if at_open else row.close_time
            fill_marks = {t: float(rows[t].open) for t in self.tickers} if at_open else marks
            self.close(ticker, price, now, bar, reason,
                       "open_gap" if at_open else "intrabar_unknown", fill_marks, forced=True)

    def corporate_actions(self, rows):
        for ticker, row in rows.items():
            pos = self.state.positions[ticker]
            if self.config.price_basis == "raw" and row.stock_splits not in (0, 1):
                ratio = float(row.stock_splits)
                pos.quantity *= ratio
                for name in ("entry_price", "atr_entry", "stop_price", "take_profit", "favorable_price", "ou_entry_buy", "ou_entry_sell"):
                    value = getattr(pos, name)
                    if value is not None:
                        setattr(pos, name, value / ratio)
            # Ex-date accrual, before opening orders: shorts owe the dividend.
            payment = pos.quantity * float(row.dividends)
            self.state.cash += payment
            if _sign(pos.quantity):
                pos.cash_flow += payment
                pos.dividends += payment

    def capped_weights(self, weights, rows):
        cfg = self.config
        if not cfg.enabled:
            return weights
        weights = weights.copy()
        if cfg.max_asset_weight is not None:
            weights = np.clip(weights, -cfg.max_asset_weight, cfg.max_asset_weight)
        if cfg.max_sector_weight is not None:
            sectors = [rows[t].sector for t in self.tickers]
            for sector in set(sectors):
                mask = np.array([s == sector for s in sectors])
                gross = np.abs(weights[mask]).sum()
                if gross > cfg.max_sector_weight:
                    weights[mask] *= cfg.max_sector_weight / gross
        gross, net = np.abs(weights).sum(), abs(weights.sum())
        scale = min(1., cfg.max_gross_exposure / gross if gross else 1., cfg.max_net_exposure / net if net else 1.)
        return weights * scale

    def rebalance(self, rows, bar):
        cfg, now = self.config, self.actions[bar]
        next_action = self.actions[bar + 1] if bar + 1 < len(self.actions) else now + pd.Timedelta(1, "ns")
        phase = "close" if cfg.execution == "next_close" else "open"
        prices = {t: float(rows[t][phase]) for t in self.tickers}
        equity = self.equity_at(prices)
        if equity <= 0:
            raise ValueError("Portfolio became insolvent; leveraged losses cannot be clipped to zero.")
        current = np.array([self.state.positions[t].quantity * prices[t] / equity for t in self.tickers])
        raw, proposed, reasons, event_ids = [], [], [], []
        atrs = {}
        for ticker in self.tickers:
            pos, row = self.state.positions[ticker], rows[ticker]
            source = self.scheduled[ticker][bar]
            q = float(self.signals[ticker].iloc[source].target_position) if source >= 0 else 0.
            raw.append(q)
            why, ids = [], []
            atr = float(row["atr_" + phase])
            atrs[ticker] = atr if np.isfinite(atr) else None
            snapshot = self.ou_policy.snapshot(ticker, now, bar) if self.ou_policy else {}
            self.ou_snapshots[ticker] = snapshot
            if cfg.enabled:
                if pos.blocked_direction and _sign(q) != pos.blocked_direction:
                    pos.blocked_direction = 0
                waiting = (cfg.reentry == "new_signal" and _sign(q) == pos.blocked_direction and pos.blocked_direction != 0)
                if cfg.ou.mode == "entry_exit" and q > 0:
                    waiting = False  # explicit long acquisition policy replaces the old signal lock
                cooling = cfg.reentry == "cooldown" and pos.last_forced_exit_bar >= 0 and bar <= pos.last_forced_exit_bar + cfg.cooldown_bars
                if pos.last_forced_exit_bar == bar or waiting or cooling:
                    q = 0.
                    why.append("reentry_block")
                if self.state.halted:
                    q = 0.
                    why.append("drawdown_halt")
                if pos.pending_duration_exit:
                    q = 0.
                    why.append("max_holding")
                if cfg.volatility_target is not None and q:
                    vol = float(row["vol_" + phase])
                    if not np.isfinite(vol) or vol <= 0:
                        q = 0.
                        why.append("volatility_warmup")
                    else:
                        q = float(np.clip(q * min(cfg.volatility_scale_cap, cfg.volatility_target / vol), -1, 1))
                        why.append("volatility_size")
                if cfg.event_policy != "none":
                    ids = active_events(self.events, ticker, row.sector, now, next_action, cfg)
                    if ids:
                        why.append("event_" + cfg.event_policy)
                        if cfg.event_policy == "flat":
                            q = 0.
                        elif cfg.event_policy == "reduce":
                            q *= cfg.event_reduce_factor
                        else:
                            old_weight = current[self.tickers.index(ticker)]
                            if _sign(q) != _sign(old_weight):
                                q = 0.
                            else:
                                q = _sign(q) * min(abs(q), abs(old_weight) * len(self.tickers))
                if q and _sign(pos.quantity) != _sign(q) and cfg.needs_atr and atrs[ticker] is None:
                    q = 0.
                    why.append("atr_warmup")
                if self.ou_policy is not None and q > 0 and pos.quantity <= 0:
                    if not snapshot.get("valid"):
                        why.append("ou_fallback_" + snapshot["reason"])
                        if cfg.ou.mode == "entry_exit":
                            q = 0.
                            why.append("ou_entry_block")
                    elif prices[ticker] >= snapshot["sell_price"]:
                        q = 0.
                        why.append("ou_above_sell_block")
                    elif cfg.ou.mode == "entry_exit":
                        # The threshold was computed from completed past bars.
                        # A future open/close may fill this predeclared buy limit;
                        # a gap above it is left unfilled, not bought retroactively.
                        if snapshot["buy_price"] is None or prices[ticker] > snapshot["buy_price"]:
                            q = 0.
                            why.append("ou_entry_block")
                        else:
                            why.append("ou_buy_limit_fill")
                    else:
                        why.append("ou_exit_boundary")
            if bar == len(self.actions) - 1 and phase == "close":
                q = 0.
                why.append("terminal_no_entry")
            proposed.append(q / len(self.tickers))
            reasons.append(why)
            event_ids.append(ids)
        weights = self.capped_weights(np.array(proposed), rows)
        hard_violation = not np.allclose(current, self.capped_weights(current, rows), atol=EPS, rtol=0)
        for i in range(len(weights)):
            if abs(weights[i] - proposed[i]) > EPS:
                reasons[i].append("exposure_limit")
            hard_exit = any(r in reasons[i] for r in ("reentry_block", "drawdown_halt", "max_holding", "event_flat", "event_reduce", "event_block_increases", "ou_entry_block", "ou_above_sell_block"))
            if self.ou_policy and _sign(current[i]) and (raw[i] == 0 or _sign(raw[i]) != _sign(current[i])):
                hard_exit = True
            if cfg.enabled and not hard_violation and not hard_exit and abs(weights[i] - current[i]) < cfg.no_trade_band:
                weights[i] = current[i]
                reasons[i].append("no_trade_band")
        # A held buffer must not prevent another asset's risk reduction.
        capped = self.capped_weights(weights, rows)
        for i in range(len(weights)):
            if abs(capped[i] - weights[i]) > EPS:
                reasons[i].append("exposure_limit")
        weights = capped
        # Reserve an upper bound on fill friction so actual post-fee exposures
        # obey caps. Costs of an unchanged weight are not charged artificially.
        slip = cfg.slippage_bps * 1e-4
        friction = slip + cfg.fees_bps * 1e-4 * (1 + slip)
        unchanged = np.abs(weights - current) < EPS
        changed_gross = np.abs(weights[~unchanged]).sum()
        old_changed = np.abs(current[~unchanged]).sum() * equity
        budget = max(0., (equity - friction * old_changed) / (1 + friction * changed_gross))
        reserved_weights = np.where(unchanged, current * equity / budget, weights) if budget > 0 else current
        # Preserve a no-order band while there is room under the caps. A held
        # weight can only override its band when other fills' costs would put
        # it over a risk limit. Unit gross allocation is shared by both arms.
        reserve_violation = (budget <= 0 or np.abs(reserved_weights).sum() > 1 + EPS or
                             not np.allclose(reserved_weights, self.capped_weights(reserved_weights, rows), atol=EPS, rtol=0))
        if reserve_violation and budget < equity - EPS:
            for i in np.flatnonzero(unchanged & (np.abs(current) > EPS)):
                reasons[i].append("cost_reserve_limit")
            unchanged[:] = False
            budget = max(0., (equity - friction * np.abs(current).sum() * equity) / (1 + friction * np.abs(weights).sum()))
        goals = {t: (self.state.positions[t].quantity if unchanged[i] else weights[i] * budget / prices[t]) for i, t in enumerate(self.tickers)}
        # Validate entry barriers before any fill: an ATR may be too wide for a
        # low-priced instrument. Never create a negative synthetic stop/TP.
        if cfg.enabled:
            for i, ticker in enumerate(self.tickers):
                pos, goal = self.state.positions[ticker], goals[ticker]
                if not _sign(goal) or _sign(pos.quantity) == _sign(goal):
                    continue
                anchor = prices[ticker] * (1 + _sign(goal) * slip)
                for prefix in ("stop_loss", "take_profit", "trailing_stop"):
                    distance = _distance(cfg, prefix, anchor, atrs[ticker])
                    direction = _sign(goal) * (1 if prefix == "take_profit" else -1)
                    if distance is not None and anchor + direction * distance <= 0:
                        goals[ticker] = 0.
                        reasons[i].append("invalid_barrier_distance")
                        break
        # Close/reduce first; then open/add, all sized from one portfolio budget.
        for ticker in self.tickers:
            pos, goal = self.state.positions[ticker], goals[ticker]
            why = reasons[self.tickers.index(ticker)]
            reason = "drawdown_halt" if "drawdown_halt" in why else "max_holding" if "max_holding" in why else "event_flat" if "event_flat" in why else "signal_exit"
            if _sign(pos.quantity) and _sign(goal) != _sign(pos.quantity):
                self.close(ticker, prices[ticker], now, bar,
                           reason if not _sign(goal) else "signal_flip", phase, prices,
                           forced=reason == "max_holding")
                if reason == "max_holding":
                    goals[ticker] = 0.
            elif abs(goal) < abs(pos.quantity) - EPS:
                self.fill(ticker, goal - pos.quantity, prices[ticker], now, bar, "resize_reduce", phase, prices)
        for ticker in self.tickers:
            pos, goal = self.state.positions[ticker], goals[ticker]
            if abs(goal) > abs(pos.quantity) + EPS:
                self.fill(ticker, goal - pos.quantity, prices[ticker], now, bar,
                          "signal_entry" if not _sign(pos.quantity) else "resize_add", phase, prices,
                          atr=atrs[ticker])
        actual_equity = self.equity_at(prices)
        for i, ticker in enumerate(self.tickers):
            source = self.scheduled[ticker][bar]
            self.decisions.append({
                "date": rows[ticker].date, "timestamp": now, "ticker": ticker,
                "signal_available_at": self.signals[ticker].iloc[source].signal_available_at if source >= 0 else pd.NaT,
                "raw_target": raw[i], "requested_weight": proposed[i],
                "target_weight": float(weights[i]),
                "actual_weight": self.state.positions[ticker].quantity * prices[ticker] / actual_equity,
                "reasons": "|".join(reasons[i]) or "model_target",
                "event_ids": "|".join(event_ids[i]),
                "ou_fit_version": self.ou_snapshots[ticker].get("fit_version"),
                "ou_fit_cutoff": self.ou_snapshots[ticker].get("fit_cutoff"),
                "ou_valid": self.ou_snapshots[ticker].get("valid"),
                "ou_reason": self.ou_snapshots[ticker].get("reason"),
                "ou_buy_price": self.ou_snapshots[ticker].get("buy_price"),
                "ou_sell_price": self.ou_snapshots[ticker].get("sell_price"),
            })

    def update_protections(self, rows, bar):
        cfg = self.config
        for ticker, row in rows.items():
            pos = self.state.positions[ticker]
            direction = _sign(pos.quantity)
            if not direction or pos.entry_time >= row.close_time:
                continue
            pos.age += 1
            if not cfg.enabled:
                continue
            if cfg.max_holding_bars is not None and pos.age >= cfg.max_holding_bars:
                pos.pending_duration_exit = True
            favorable = float(row.high if direction > 0 else row.low)
            pos.favorable_price = max(pos.favorable_price, favorable) if direction > 0 else min(pos.favorable_price, favorable)
            gain = direction * (pos.favorable_price / pos.entry_price - 1)
            if gain >= cfg.trailing_activation_pct:
                distance = _distance(cfg, "trailing_stop", pos.favorable_price, pos.atr_entry)
                if distance is not None:
                    self.tighten_stop(pos, pos.favorable_price - direction * distance, "trailing_stop")
            if cfg.break_even_activation_pct is not None and gain >= cfg.break_even_activation_pct:
                self.tighten_stop(pos, pos.entry_price * (1 + direction * cfg.break_even_offset_pct), "break_even")

    def run(self):
        cfg = self.config
        for bar in range(len(self.calendar)):
            self.state.bar_index = bar
            rows = {t: self.groups[t].iloc[bar] for t in self.tickers}
            self.corporate_actions(rows)
            self.protections(rows, bar, gap_only=True)
            if cfg.execution != "next_close":
                self.rebalance(rows, bar)
            self.protections(rows, bar, gap_only=False)
            if cfg.execution == "next_close":
                self.rebalance(rows, bar)
            self.update_protections(rows, bar)
            marks = {t: float(rows[t].close) for t in self.tickers}
            if bar == len(self.calendar) - 1:
                for ticker in self.tickers:
                    self.close(ticker, marks[ticker], rows[ticker].close_time, bar, "terminal_close", "close", marks)
            equity = self.equity_at(marks)
            if equity <= 0:
                raise ValueError("Portfolio became insolvent; leveraged losses cannot be clipped to zero.")
            self.state.equity = equity
            self.state.peak_equity = max(self.state.peak_equity, equity)
            dd = equity / self.state.peak_equity - 1
            if cfg.enabled and cfg.max_drawdown is not None and dd <= -cfg.max_drawdown:
                self.state.halted = True
            weights = [self.state.positions[t].quantity * marks[t] / equity for t in self.tickers]
            self.equity_rows.append({"date": self.calendar.iloc[bar].date,
                                     "timestamp": self.calendar.iloc[bar].close_time,
                                     "cash": self.state.cash, "equity": equity,
                                     "drawdown": dd, "gross_exposure": sum(map(abs, weights)),
                                     "net_exposure": sum(weights), "halted": self.state.halted})
            for ticker, weight in zip(self.tickers, weights):
                pos = self.state.positions[ticker]
                self.position_rows.append({"date": rows[ticker].date, "ticker": ticker,
                                           "quantity": pos.quantity, "weight": weight,
                                           "stop_price": pos.stop_price, "take_profit": pos.take_profit,
                                           "blocked_direction": pos.blocked_direction,
                                           "holding_bars": pos.age, "trade_id": pos.trade_id if pos.quantity else None})
        equity, positions = pd.DataFrame(self.equity_rows), pd.DataFrame(self.position_rows)
        audit_columns = ["ou_fit_version", "ou_fit_cutoff", "ou_buy_price", "ou_sell_price"]
        orders = pd.DataFrame(self.orders, columns=["date", "timestamp", "ticker", "trade_id", "quantity", "quantity_before", "quantity_after", "base_price", "execution_price", "notional", "fee", "slippage_cost", "turnover", "reason", "phase", "earliest_time", "latest_time"] + audit_columns)
        trades = pd.DataFrame(self.trades, columns=["ticker", "trade_id", "side", "entry_time", "exit_time", "entry_price", "exit_price", "entry_notional", "net_pnl", "fees", "dividends", "holding_bars", "exit_reason", "exit_phase"] + audit_columns)
        metrics = portfolio_metrics(equity, orders, trades, cfg)
        metadata = {
            "config": asdict(cfg), "config_sha256": stable_config_hash(cfg),
            "bars_sha256": frame_fingerprint(self.bars),
            "targets_sha256": frame_fingerprint(self.targets),
            "events_sha256": frame_fingerprint(self.events),
            "history_sha256": frame_fingerprint(self.history) if self.history is not None else None,
            "ou_policy": self.ou_policy.metadata() if self.ou_policy else {"mode": "off"},
            "execution_is_proxy": cfg.execution == "open_proxy",
            "session_times_inferred": not self.explicit_times,
            "ambiguous_tp_stop_bars": self.ambiguities,
            "intrabar_time_convention": "unknown within bar; timestamp is upper bound",
            "dividend_convention": "ex-date cash accrual before opening orders",
            "short_borrow_and_financing_included": False,
            "rule_contribution_convention": "episode PnL grouped by final exit reason; not causal attribution",
        }
        return TradingResult(equity, positions, orders, trades, pd.DataFrame(self.decisions), metrics, metadata, self.state)


def run_trading_backtest(bars, targets, config=None, *, events=None, history=None):
    """Replay signed model targets, with optional exits/risk/calendar rules.

    Arrays follow input bar row order. Keyed frames use date/ticker alignment.
    This is a fresh replay starting flat and ending with charged liquidation.
    Turning rules off still uses this OHLC accounting for fair A/B comparisons;
    existing experiment APIs continue to use their original return engines.
    """
    config = config or TradingConfig()
    if isinstance(config, dict):
        config = TradingConfig(**config)
    if not isinstance(config, TradingConfig):
        raise TypeError("config must be TradingConfig or a config dictionary.")
    return _Replay(bars, targets, config, events, history).run()
