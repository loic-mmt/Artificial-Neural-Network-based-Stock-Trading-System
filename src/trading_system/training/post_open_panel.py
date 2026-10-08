"""Complete-calendar post-open financial paths for the label/loss benchmark."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from trading_system.labels.post_open_benchmark import adjusted_open_prices


class PostOpenReturnPanel:
    """Map signal rows to fixed, equal-weight asset slots on a full calendar.

    A missing signal means zero target, not deletion of its price date. Asset
    slots with leading/trailing missing quotes remain cash. Internal quotation
    holes are rejected: filling them or pretending the next quote is tomorrow
    would invent an executable path. Overnight positions liquidate at the last
    available open, and a new position cannot enter on that liquidation day.

    Paths contain every supplied session, including the final overnight day
    (zero price return plus liquidation costs). That actual trading day is part
    of the daily objective and annualization, not an extra synthetic session.
    """

    def __init__(self, frame, *, protocol, tickers=None, calendar=None, signal_frame=None):
        protocol = protocol.replace("-", "_")
        if protocol not in {"intraday", "overnight"}:
            raise ValueError("protocol must be intraday or overnight.")
        missing = [column for column in ("date", "ticker", "open", "close") if column not in frame]
        if missing or frame.empty:
            raise ValueError(f"Nonempty date/ticker/open/close price frame required; missing={missing}.")
        work = frame.reset_index(drop=True).copy()
        work["date"] = pd.to_datetime(work["date"], utc=True, errors="raise").dt.normalize()
        if work["date"].isna().any() or work["ticker"].isna().any():
            raise ValueError("Price dates and tickers cannot be missing.")
        if work.duplicated(["date", "ticker"]).any():
            raise ValueError("Price rows require unique session dates per ticker.")
        self.tickers = tuple(sorted(work["ticker"].unique())) if tickers is None else tuple(tickers)
        if not self.tickers or len(set(self.tickers)) != len(self.tickers) or pd.isna(list(self.tickers)).any():
            raise ValueError("tickers must be a nonempty unique ordered universe.")
        if calendar is None:
            self.dates = pd.DatetimeIndex(work["date"].unique()).sort_values()
        else:
            self.dates = pd.DatetimeIndex(pd.to_datetime(calendar, utc=True)).normalize()
            if not len(self.dates) or self.dates.hasnans or not self.dates.is_unique or not self.dates.is_monotonic_increasing:
                raise ValueError("calendar must contain unique increasing non-missing session dates.")
        signals = (work if signal_frame is None else signal_frame).reset_index(drop=True).copy()
        if not {"date", "ticker"}.issubset(signals):
            raise ValueError("signal_frame must contain date and ticker keys.")
        signals["date"] = pd.to_datetime(signals["date"], utc=True, errors="raise").dt.normalize()
        if signals["date"].isna().any() or signals["ticker"].isna().any() or signals.duplicated(["date", "ticker"]).any():
            raise ValueError("Signals require unique, non-missing session dates per ticker.")
        if not signals["ticker"].isin(self.tickers).all() or not signals["date"].isin(self.dates).all():
            raise ValueError("Signal keys must belong to the supplied ticker universe and calendar.")
        price_keys = pd.MultiIndex.from_frame(work[["date", "ticker"]])
        if not pd.MultiIndex.from_frame(signals[["date", "ticker"]]).isin(price_keys).all():
            raise ValueError("Every signal key must have an original price row.")
        self.rows = len(signals)
        self.signal_frame = signals[["date", "ticker"]].copy()
        shape = (len(self.tickers), len(self.dates))
        self.indices = np.full(shape, -1, dtype=np.int64)
        self.returns = np.zeros(shape, dtype=float)
        self.quote_available = np.zeros(shape, dtype=bool)
        self.available = np.zeros(shape, dtype=bool)
        signals["_signal_row"] = np.arange(self.rows)
        for asset, ticker in enumerate(self.tickers):
            asset_frame = work[work["ticker"] == ticker].set_index("date").reindex(self.dates)
            asset_signals = signals[signals["ticker"] == ticker].set_index("date").reindex(self.dates)
            mapped = asset_signals["_signal_row"].to_numpy()
            mask = np.isfinite(mapped)
            self.indices[asset, mask] = mapped[mask].astype(int)
            if protocol == "intraday":
                opening = pd.to_numeric(asset_frame["open"], errors="coerce").to_numpy(dtype=float)
                closing = pd.to_numeric(asset_frame["close"], errors="coerce").to_numpy(dtype=float)
                if (opening[np.isfinite(opening)] <= 0).any() or (closing[np.isfinite(closing)] <= 0).any():
                    raise ValueError(f"{ticker}: prices must be positive; clean the input first.")
                quoted = np.isfinite(opening) & np.isfinite(closing)
                self.returns[asset, quoted] = closing[quoted] / opening[quoted] - 1
                self.available[asset] = quoted
            else:
                opening = adjusted_open_prices(asset_frame)
                if (opening[np.isfinite(opening)] <= 0).any():
                    raise ValueError(f"{ticker}: adjusted opens must be positive; clean the input first.")
                quoted = np.isfinite(opening)
                adjacent = quoted[:-1] & quoted[1:]
                self.returns[asset, :-1][adjacent] = opening[1:][adjacent] / opening[:-1][adjacent] - 1
                self.available[asset, :-1] = adjacent
            locations = np.flatnonzero(quoted)
            if len(locations) and not quoted[locations[0]:locations[-1] + 1].all():
                raise ValueError(
                    f"{ticker}: internal missing price quotes on the full session calendar; "
                    "no filling or compressed multi-session daily returns are supported."
                )
            self.quote_available[asset] = quoted
        self.protocol = protocol
        self.delay = 0
        self.metadata = {
            "protocol": protocol, "decision": "after_open_J", "execution_price": "open_proxy",
            "allocation": "fixed_equal_slots_q_over_N", "missing_signal": "cash_target",
            "internal_missing_quotes": "rejected", "calendar_sessions": len(self.dates),
            "overnight_terminal_day": "zero_return_liquidation_only" if protocol == "overnight" else None,
            "annualization_calendar": "all_actual_supplied_sessions_including_liquidation_day",
        }

    def _targets(self, positions):
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (self.rows,) or not np.isfinite(positions).all() or (np.abs(positions) > 1 + 1e-6).any():
            raise ValueError("Positions must be finite, signal-row aligned and bounded by [-1, 1].")
        target = np.zeros(self.indices.shape, dtype=float)
        valid = (self.indices >= 0) & self.available
        target[valid] = positions[self.indices[valid]]
        return target

    def path(self, positions, config):
        executed = self._targets(positions)
        if self.protocol == "intraday":
            delta = executed.copy()
            turnover = 2 * np.abs(executed)  # entry and close liquidation daily
        else:
            delta = np.diff(executed, axis=1, prepend=0.0)
            turnover = np.abs(delta)  # last available open already targets zero
        costs = config.cost_bps * 1e-4 * turnover
        net = executed * self.returns - costs
        return net.mean(axis=0), executed, delta, turnover, costs

    def loss_and_gradient(self, positions, config):
        """Exact chronological gradient, including each protocol's costs."""
        if config.objective == "cross_entropy":
            raise ValueError("Cross entropy requires labeled classifier training.")
        net, executed, delta, _, _ = self.path(positions, config)
        count, mean = len(net), net.mean()
        if config.objective == "pnl" or (config.objective == "cara" and config.cara_gamma == 0):
            loss = -mean
            upstream = np.full(count, -1 / count)
        elif config.objective == "cara":
            exponent = -config.cara_gamma * net
            if not np.isfinite(exponent).all() or np.max(exponent) > np.log(np.finfo(float).max) - 1:
                raise FloatingPointError("CARA exponent exceeds finite float64 range.")
            loss = np.expm1(exponent).mean() / config.cara_gamma
            upstream = -np.exp(exponent) / count
        else:
            centered = net - mean
            scale = np.sqrt(np.mean(centered ** 2) + config.sharpe_epsilon ** 2)
            annual = np.sqrt(config.annualization)
            loss = -annual * mean / scale
            upstream = -annual / count * (1 / scale - mean * centered / scale ** 3)
            if config.objective == "combined":
                weight = config.combined_pnl_weight
                loss = (1 - weight) * loss - weight * mean / config.combined_pnl_scale
                upstream = (1 - weight) * upstream - weight / (count * config.combined_pnl_scale)
        upstream /= len(self.tickers)
        rate = config.cost_bps * 1e-4
        if self.protocol == "intraday":
            gradient = upstream * (self.returns - 2 * rate * np.sign(executed))
        else:
            signs = np.sign(delta)
            gradient = upstream * (self.returns - rate * signs)
            gradient[:, :-1] += rate * signs[:, 1:] * upstream[1:]
        aligned = np.zeros(self.rows, dtype=float)
        valid = (self.indices >= 0) & self.available
        aligned[self.indices[valid]] = gradient[valid]
        if not np.isfinite(loss) or not np.isfinite(aligned).all():
            raise FloatingPointError("Non-finite financial objective or gradient.")
        return float(loss), aligned

    def metrics(self, positions, config, initial_capital=10000.0):
        if not math.isfinite(initial_capital) or initial_capital <= 0:
            raise ValueError("initial_capital must be finite and positive.")
        net, executed, _, turnover, costs = self.path(positions, config)
        if (net <= -1).any():
            raise ValueError("Position portfolio became insolvent (net return <= -1).")
        wealth = np.r_[1.0, np.cumprod(1 + net)]
        std = net.std(ddof=0)
        sharpe = np.sqrt(config.annualization) * net.mean() / std if std > 0 else (0.0 if np.all(net == 0) else None)
        gross_exposure = np.abs(executed).mean(axis=0)
        return {
            "net_pnl": float(initial_capital * (wealth[-1] - 1)),
            "net_return": float(wealth[-1] - 1), "mean_net_return": float(net.mean()),
            "net_sharpe": None if sharpe is None else float(sharpe),
            "regularized_sharpe": float(np.sqrt(config.annualization) * net.mean() / np.sqrt(std ** 2 + config.sharpe_epsilon ** 2)),
            "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
            "turnover": float(turnover.mean(axis=0).sum()),
            "cost_return_sum": float(costs.mean(axis=0).sum()),
            "mean_abs_position": float(gross_exposure.mean()),
            "mean_gross_exposure": float(gross_exposure.mean()),
            "mean_net_exposure": float(executed.mean(axis=0).mean()),
            "mean_long_exposure": float(np.maximum(executed, 0).mean()),
            "mean_short_exposure": float(np.maximum(-executed, 0).mean()),
            "mean_cash_allocation": float((1 - gross_exposure).mean()),
            "periods": len(net), "assets": len(self.tickers), "protocol": self.protocol,
        }
