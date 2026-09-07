from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from trading_system.backtest.engine import evaluate_strategy_vs_buy_hold
from trading_system.data.io import read_parquet_dataset
from trading_system.data.splits import chronological_train_val_test_split
from trading_system.labels.config import LabelConfig
from trading_system.labels.registry import LabelContext, create_default_label_registry
from trading_system.paths import default_market_dataset_path

DATA_DIR = default_market_dataset_path()
DEFAULT_BUFFERS = (0.0, 0.001, 0.002, 0.005)


def _grid_values(value: str, *, cast, name: str) -> tuple:
    try:
        values = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid {name} list: {value!r}.") from error
    if not values:
        raise argparse.ArgumentTypeError(f"{name} requires at least one value.")
    return values


def _compatibility_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Keep historical result keys while exposing all canonical metrics."""

    return {
        **metrics,
        "final_capital": metrics["model_final_capital"],
        "pnl": metrics["model_pnl"],
        "n_trades": metrics["transaction_count"],
    }


def _label_gridsearch(
    frame: pd.DataFrame,
    *,
    price_col: str,
    fees: float,
    capital: float,
    position_mode: str,
    windows: Sequence[int],
    buy_buffers: Sequence[float],
    sell_buffers: Sequence[float],
    alternating: bool,
    execution_delay: int,
) -> tuple[dict[str, object], dict[str, float], pd.DataFrame]:
    if frame is None or frame.empty:
        raise ValueError("frame must not be empty.")
    missing = [column for column in ("date", price_col) if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing grid-search columns: {missing}")
    if position_mode not in ("long_only", "long_short"):
        raise ValueError("position_mode must be 'long_only' or 'long_short'.")
    if not windows or not buy_buffers or not sell_buffers:
        raise ValueError("Grid-search parameter collections must not be empty.")

    work = frame.sort_values("date").reset_index(drop=True).copy()
    registry = create_default_label_registry()
    context = LabelContext(price_col=price_col, date_col="date")
    rows: list[dict[str, object]] = []
    best_params: dict[str, object] | None = None
    best_metrics: dict[str, float] | None = None
    best_score = float("-inf")

    for window in windows:
        for buy_buffer in buy_buffers:
            for sell_buffer in sell_buffers:
                config = LabelConfig.breakout(
                    window=window,
                    buy_buffer=buy_buffer,
                    sell_buffer=sell_buffer,
                    alternating=alternating,
                )
                labels = registry.generate(work, config, context)
                metrics = evaluate_strategy_vs_buy_hold(
                    labels.frame,
                    labels.frame["Label_id"].to_numpy(),
                    initial_capital=capital,
                    price_col=price_col,
                    fee_per_trade=fees,
                    position_mode=position_mode,
                    execution_delay=execution_delay,
                )
                score = float(metrics["outperformance"])
                parameters = dict(config.parameters)
                rows.append(
                    {
                        **parameters,
                        "score": score,
                        "final_capital": metrics["model_final_capital"],
                        "pnl": metrics["model_pnl"],
                        "buy_hold_final_capital": metrics["buy_hold_final_capital"],
                        "outperformance": metrics["outperformance"],
                        "n_trades": metrics["transaction_count"],
                        "trade_count": metrics["trade_count"],
                        "total_fees": metrics["total_fees"],
                        "action_rate": labels.metadata["action_rate"],
                    }
                )
                if score > best_score:
                    best_score = score
                    best_params = parameters
                    best_metrics = _compatibility_metrics(metrics)

    if best_params is None or best_metrics is None:
        raise RuntimeError("Grid search produced no result.")
    results = (
        pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    )
    return best_params, best_metrics, results


def label_gridsearch(
    df: pd.DataFrame,
    price_col: str = "adj_close",
    fees: float = 1.0,
    capital: float = 10_000.0,
    *,
    windows: Sequence[int] = tuple(range(5, 61)),
    buy_buffers: Sequence[float] = DEFAULT_BUFFERS,
    sell_buffers: Sequence[float] = DEFAULT_BUFFERS,
    alternating: bool = True,
    execution_delay: int = 1,
):
    """Select breakout parameters on one validation frame using long/flat."""

    return _label_gridsearch(
        df,
        price_col=price_col,
        fees=fees,
        capital=capital,
        position_mode="long_only",
        windows=windows,
        buy_buffers=buy_buffers,
        sell_buffers=sell_buffers,
        alternating=alternating,
        execution_delay=execution_delay,
    )


def label_gridsearch_long_short(
    df: pd.DataFrame,
    price_col: str = "adj_close",
    fees: float = 1.0,
    capital: float = 10_000.0,
    *,
    windows: Sequence[int] = tuple(range(2, 61)),
    buy_buffers: Sequence[float] = DEFAULT_BUFFERS,
    sell_buffers: Sequence[float] = DEFAULT_BUFFERS,
    alternating: bool = True,
    execution_delay: int = 1,
):
    """Select breakout parameters on one validation frame using long/short."""

    return _label_gridsearch(
        df,
        price_col=price_col,
        fees=fees,
        capital=capital,
        position_mode="long_short",
        windows=windows,
        buy_buffers=buy_buffers,
        sell_buffers=sell_buffers,
        alternating=alternating,
        execution_delay=execution_delay,
    )


def load_default_validation_split(
    ticker: str = "EN.PA",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    *,
    data_path: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = read_parquet_dataset(data_path)
    frame = frame[frame["ticker"] == ticker].copy()
    if frame.empty:
        raise ValueError(f"Ticker {ticker!r} is absent from {data_path}.")
    return chronological_train_val_test_split(
        frame,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grid search on breakout labels over the validation split."
    )
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--ticker", default="EN.PA")
    parser.add_argument("--fees", type=float, default=2.0)
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--price-col", default="adj_close")
    parser.add_argument(
        "--mode", choices=("long_only", "long_short"), default="long_only"
    )
    parser.add_argument(
        "--label-method",
        "--label-mode",
        choices=("breakout",),
        default="breakout",
    )
    parser.add_argument(
        "--label-windows",
        "--label-window",
        dest="label_windows",
        type=lambda value: _grid_values(value, cast=int, name="windows"),
    )
    parser.add_argument(
        "--label-buy-buffers",
        "--label-buy-buffer",
        dest="label_buy_buffers",
        type=lambda value: _grid_values(value, cast=float, name="buy buffers"),
        default=DEFAULT_BUFFERS,
    )
    parser.add_argument(
        "--label-sell-buffers",
        "--label-sell-buffer",
        dest="label_sell_buffers",
        type=lambda value: _grid_values(value, cast=float, name="sell buffers"),
        default=DEFAULT_BUFFERS,
    )
    parser.add_argument(
        "--label-alternating",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--execution-delay", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _, validation, _ = load_default_validation_split(
        ticker=args.ticker,
        data_path=args.data,
    )
    windows = args.label_windows
    if windows is None:
        windows = tuple(range(2 if args.mode == "long_short" else 5, 61))
    function = (
        label_gridsearch_long_short if args.mode == "long_short" else label_gridsearch
    )
    best_params, best_metrics, results = function(
        validation,
        price_col=args.price_col,
        fees=args.fees,
        capital=args.capital,
        windows=windows,
        buy_buffers=args.label_buy_buffers,
        sell_buffers=args.label_sell_buffers,
        alternating=args.label_alternating,
        execution_delay=args.execution_delay,
    )
    print(best_params)
    print(best_metrics)
    print(results.head(20))


if __name__ == "__main__":
    main()
