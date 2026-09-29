"""Explicit execution and optional post-prediction trading rules."""

from dataclasses import dataclass, field
from datetime import time
import math
from zoneinfo import ZoneInfo

from .ou import OUConfig


@dataclass(frozen=True)
class TradingConfig:
    enabled: bool = False
    execution: str = "next_open"
    signal_timing: str = "after_close"
    bar_mode: str = "daily"
    timezone: str = "Europe/Paris"
    session_open: str = "09:00"
    session_close: str = "17:30"
    price_basis: str = "split_adjusted"
    position_mode: str = "long_short"
    initial_capital: float = 10_000.0
    fees_bps: float = 5.0
    slippage_bps: float = 0.0
    annualization: int = 252
    risk_free_rate: float = 0.0
    stop_loss_pct: float | None = None
    stop_loss_atr: float | None = None
    take_profit_pct: float | None = None
    take_profit_atr: float | None = None
    trailing_stop_pct: float | None = None
    trailing_stop_atr: float | None = None
    trailing_activation_pct: float = 0.0
    break_even_activation_pct: float | None = None
    break_even_offset_pct: float = 0.0
    atr_window: int = 14
    max_holding_bars: int | None = None
    reentry: str = "new_signal"
    cooldown_bars: int = 0
    no_trade_band: float = 0.0
    max_asset_weight: float | None = None
    max_sector_weight: float | None = None
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 1.0
    volatility_target: float | None = None
    volatility_window: int = 20
    volatility_scale_cap: float = 1.0
    max_drawdown: float | None = None
    event_policy: str = "none"
    event_types: tuple[str, ...] = ()
    event_pre_hours: float = 0.0
    event_post_hours: float = 0.0
    event_reduce_factor: float = 0.5
    ou: OUConfig = field(default_factory=OUConfig)

    def __post_init__(self):
        if isinstance(self.ou, dict):
            object.__setattr__(self, "ou", OUConfig(**self.ou))
        if not isinstance(self.ou, OUConfig):
            raise TypeError("ou must be OUConfig or a configuration dictionary.")
        choices = {
            "execution": {"next_open", "next_close", "open_proxy"},
            "signal_timing": {"after_open", "after_close"},
            "bar_mode": {"daily", "intraday"},
            "price_basis": {"raw", "split_adjusted"},
            "position_mode": {"long_only", "long_short"},
            "reentry": {"new_signal", "cooldown", "next_bar"},
            "event_policy": {"none", "block_increases", "reduce", "flat"},
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}.")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean.")
        if self.execution == "open_proxy" and self.signal_timing != "after_open":
            raise ValueError("open_proxy requires signal_timing='after_open'.")
        ZoneInfo(self.timezone)
        if time.fromisoformat(self.session_open) >= time.fromisoformat(self.session_close):
            raise ValueError("session_open must precede session_close.")
        for name in ("annualization", "atr_window", "volatility_window", "max_holding_bars"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(self.cooldown_bars, bool) or not isinstance(self.cooldown_bars, int) or self.cooldown_bars < 0:
            raise ValueError("cooldown_bars must be a non-negative integer.")
        if self.volatility_window < 2:
            raise ValueError("volatility_window must be >= 2.")
        for name in (
            "initial_capital", "max_gross_exposure", "max_net_exposure",
            "volatility_scale_cap", "stop_loss_atr", "take_profit_atr", "trailing_stop_atr",
            "max_asset_weight", "max_sector_weight", "volatility_target",
        ):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive.")
        for name in (
            "fees_bps", "slippage_bps", "trailing_activation_pct", "break_even_offset_pct",
            "no_trade_band", "event_pre_hours", "event_post_hours",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if max(self.fees_bps, self.slippage_bps) >= 10_000:
            raise ValueError("Fees and slippage must be less than 10000 bps.")
        for name in ("stop_loss_pct", "take_profit_pct", "trailing_stop_pct", "max_drawdown", "break_even_activation_pct"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1):
                raise ValueError(f"{name} must be in (0, 1).")
        for prefix in ("stop_loss", "take_profit", "trailing_stop"):
            if getattr(self, prefix + "_pct") is not None and getattr(self, prefix + "_atr") is not None:
                raise ValueError(f"Choose percentage OR ATR for {prefix}.")
        if self.break_even_activation_pct is not None and self.break_even_offset_pct >= self.break_even_activation_pct:
            raise ValueError("Break-even offset must be below its activation gain.")
        if not math.isfinite(self.risk_free_rate) or self.risk_free_rate <= -1:
            raise ValueError("risk_free_rate must be finite and > -1.")
        if not math.isfinite(self.event_reduce_factor) or not 0 <= self.event_reduce_factor <= 1:
            raise ValueError("event_reduce_factor must be in [0, 1].")
        if isinstance(self.event_types, str) or any(not isinstance(x, str) or not x.strip() for x in self.event_types):
            raise ValueError("event_types must be a sequence of non-empty strings.")
        object.__setattr__(self, "event_types", tuple(self.event_types))
        if self.ou.mode != "off":
            if self.trailing_stop_pct is None or self.trailing_stop_atr is not None or self.trailing_activation_pct != 0:
                raise ValueError("OU policy requires an immediately active percentage trailing stop.")
            if self.take_profit_pct is not None or self.take_profit_atr is not None:
                raise ValueError("OU policy and a fixed take-profit are mutually exclusive.")
            if self.execution == "open_proxy":
                raise ValueError("OU policy requires causal next_open or next_close execution.")

    @property
    def needs_atr(self):
        return self.enabled and any(x is not None for x in (self.stop_loss_atr, self.take_profit_atr, self.trailing_stop_atr))
