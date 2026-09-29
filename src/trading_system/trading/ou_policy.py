"""Causal completed-bar OU snapshots; frozen boundaries for each long episode."""

import math

import pandas as pd

from .ou import AffineCosts, estimate_ou, solve_ou_boundaries


class OUPolicy:
    def __init__(self, bars, config, history=None):
        self.config = config
        combined = pd.concat([history, bars], ignore_index=True) if history is not None else bars.copy()
        self.groups = {}
        for ticker, group in combined.groupby("ticker", sort=True):
            group = group.sort_values("close_time").reset_index(drop=True)
            # Raw prices are continuous in share units known at each observation.
            group["units"] = group.stock_splits.replace(0, 1).cumprod() if config.price_basis == "raw" else 1.
            group["reference_close"] = group.close * group.units
            self.groups[ticker] = group
        self.cache = {}
        self.records = []
        self.version = 0

    def snapshot(self, ticker, now, bar):
        group = self.groups[ticker]
        past = group.loc[group.close_time + pd.Timedelta(1, "ns") < now]
        previous = self.cache.get(ticker)
        # Warmup can become eligible between normal refit slots.
        warmup_done = previous is not None and previous["reason"] == "warmup" and len(past) >= self.config.ou.window
        if previous is None or bar - previous["fit_bar"] >= self.config.ou.refit_every or warmup_done:
            params, diagnostic = estimate_ou(past.reference_close.tail(self.config.ou.window), self.config.ou,
                                              annualization=self.config.annualization)
            self.version += 1
            snapshot = {**diagnostic, "ticker": ticker, "fit_version": self.version,
                        "fit_bar": bar, "fit_cutoff": past.close_time.iloc[-1] if len(past) else pd.NaT,
                        "fit_available_at": past.close_time.iloc[-1] + pd.Timedelta(1, "ns") if len(past) else pd.NaT,
                        "buy_reference": None, "sell_reference": None, "acquisition_value": None}
            if params is not None:
                try:
                    bounds = solve_ou_boundaries(params, self.config.trailing_stop_pct, self.config.ou.discount_rate,
                                                costs=AffineCosts.from_bps(self.config.fees_bps, self.config.slippage_bps))
                    snapshot.update(buy_reference=bounds.buy_price, sell_reference=bounds.sell_price,
                                    acquisition_value=bounds.acquisition_value, boundary_status=bounds.status,
                                    root_residual=bounds.root_residual)
                except (ValueError, OverflowError, FloatingPointError) as exc:
                    snapshot.update(valid=False, reason="numerical_rejection", numerical_error=str(exc))
            self.cache[ticker] = snapshot
            self.records.append(snapshot.copy())
        snapshot = self.cache[ticker].copy()
        # Only splits already effective at this action can convert share units.
        unit_rows = group.loc[group.open_time <= now]
        units = float(unit_rows.units.iloc[-1]) if len(unit_rows) else 1.
        for name in ("buy", "sell"):
            reference = snapshot[name + "_reference"]
            snapshot[name + "_price"] = reference / units if reference is not None else None
        snapshot["log_mean"] = snapshot.get("log_mean", None)
        if snapshot["log_mean"] is not None:
            snapshot["log_mean"] -= math.log(units)
        return snapshot

    def metadata(self):
        reasons = pd.Series([r["reason"] for r in self.records], dtype="str").value_counts().to_dict()
        return {"mode": self.config.ou.mode, "longs_only": True,
                "fit_count": len(self.records), "fit_reasons": reasons,
                "clock": "observed trading bars; annualization bars/year",
                "fallback": "exit: trailing only; entry_exit: reject new long on invalid fit",
                "entry_order": "predeclared limit checked only at configured execution price; no intrabar entry",
                "long_reentry": "entry_exit replaces same-side signal lock; same-bar exit lock remains",
                "discount_rate": self.config.ou.discount_rate,
                "fit_records": [{k: (v.isoformat() if isinstance(v, pd.Timestamp) else None if v is pd.NaT else v)
                                 for k, v in row.items()} for row in self.records]}
