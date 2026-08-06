"""Backtesting API with lightweight core and lazy advanced-report exports."""

from __future__ import annotations

from importlib import import_module

from .engine import evaluate_strategy_vs_buy_hold, run_label_backtest

_LAZY_EXPORTS = {
    "BacktestConfig": ("trading_system.backtest.lib", "BacktestConfig"),
    "execute_first_check_pipeline": (
        "trading_system.backtest.engine",
        "execute_first_check_pipeline",
    ),
    "execute_first_check_pipeline_external": (
        "trading_system.backtest.engine",
        "execute_first_check_pipeline_external",
    ),
    "load_run": ("trading_system.backtest.artifacts", "load_run"),
    "load_run_catalog": ("trading_system.backtest.artifacts", "load_run_catalog"),
    "resolve_artifact_root": (
        "trading_system.backtest.artifacts",
        "resolve_artifact_root",
    ),
    "run_backtest_from_labels": (
        "trading_system.backtest.engine",
        "run_backtest_from_labels",
    ),
    "save_run_artifacts": ("trading_system.backtest.artifacts", "save_run_artifacts"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "BacktestConfig",
    "evaluate_strategy_vs_buy_hold",
    "execute_first_check_pipeline",
    "execute_first_check_pipeline_external",
    "load_run",
    "load_run_catalog",
    "resolve_artifact_root",
    "run_backtest_from_labels",
    "run_label_backtest",
    "save_run_artifacts",
]
