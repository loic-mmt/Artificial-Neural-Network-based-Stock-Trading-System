from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trading_system.backtest.engine import evaluate_strategy_vs_buy_hold
from trading_system.data.io import read_parquet_dataset
from trading_system.data.splits import chronological_train_val_test_split
from trading_system.paths import default_market_dataset_path, derived_data_dir

from .schema import LABEL_ID_TO_NAME, TradeLabel


def _allowed_next_executed_positions(previous_position: int) -> tuple[int, ...]:
    if previous_position == 0:
        return (-1, 0, 1)
    if previous_position in (-1, 1):
        return (-1, 1)
    raise ValueError(f"Invalid position: {previous_position}")


def compute_forward_returns(prices: np.ndarray) -> np.ndarray:
    values = np.asarray(prices, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("prices must be a non-empty 1D array.")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("prices must contain finite positive values.")
    result = np.zeros(len(values), dtype=np.float64)
    if len(values) > 1:
        result[:-1] = (values[1:] / values[:-1]) - 1.0
    return result


def solve_oracle_executed_positions_dp(
    forward_returns: np.ndarray,
    fee_per_trade: float = 0.0,
    initial_capital: float = 10_000.0,
) -> dict[str, object]:
    """Find globally optimal executed positions under current transition rules."""

    if fee_per_trade < 0:
        raise ValueError("fee_per_trade must be non-negative.")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    returns = np.asarray(forward_returns, dtype=np.float64)
    if returns.ndim != 1 or len(returns) < 2 or not np.isfinite(returns).all():
        raise ValueError(
            "forward_returns must be a finite 1D array with at least two values."
        )

    states = np.asarray([-1, 0, 1], dtype=np.int8)
    state_to_index = {int(state): index for index, state in enumerate(states)}
    capital = np.full((len(returns), len(states)), -np.inf, dtype=np.float64)
    parent = np.full((len(returns), len(states)), -1, dtype=np.int8)
    flat_index = state_to_index[0]
    capital[0, flat_index] = float(initial_capital)
    parent[0, flat_index] = flat_index

    for time_index in range(1, len(returns)):
        period_return = float(returns[time_index])
        for previous_index, previous_state in enumerate(states):
            previous_capital = capital[time_index - 1, previous_index]
            if not np.isfinite(previous_capital):
                continue
            for next_state in _allowed_next_executed_positions(int(previous_state)):
                next_index = state_to_index[next_state]
                next_capital = previous_capital * (1.0 + next_state * period_return)
                next_capital -= fee_per_trade * abs(next_state - int(previous_state))
                next_capital = max(next_capital, 0.0)
                if next_capital > capital[time_index, next_index]:
                    capital[time_index, next_index] = next_capital
                    parent[time_index, next_index] = previous_index

    final_index = int(np.argmax(capital[-1]))
    final_capital = float(capital[-1, final_index])
    if not np.isfinite(final_capital):
        raise RuntimeError("Oracle DP found no valid path.")
    state_indices = np.empty(len(returns), dtype=np.int8)
    state_indices[-1] = final_index
    for time_index in range(len(returns) - 1, 0, -1):
        previous_index = int(parent[time_index, int(state_indices[time_index])])
        if previous_index < 0:
            raise RuntimeError(f"Oracle DP backtracking failed at index {time_index}.")
        state_indices[time_index - 1] = previous_index
    executed = states[state_indices].astype(np.int8, copy=False)
    trades = int(np.abs(np.diff(executed.astype(np.int16))).sum())
    return {
        "executed_positions": executed,
        "final_capital": final_capital,
        "pnl": final_capital - float(initial_capital),
        "n_trades": trades,
    }


def executed_to_target_positions(executed_positions: np.ndarray) -> np.ndarray:
    executed = np.asarray(executed_positions, dtype=np.int8)
    if executed.ndim != 1 or len(executed) == 0:
        raise ValueError("executed_positions must be a non-empty 1D array.")
    target = np.empty_like(executed)
    if len(executed) == 1:
        target[0] = 0
    else:
        target[:-1] = executed[1:]
        target[-1] = executed[-1]
    return target


def target_positions_to_labels(
    target_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(target_positions, dtype=np.int8)
    if target.ndim != 1 or len(target) == 0:
        raise ValueError("target_positions must be a non-empty 1D array.")
    names = np.full(len(target), "Hold", dtype=object)
    label_ids = np.full(len(target), TradeLabel.HOLD.value, dtype=np.int64)
    previous = 0
    for index, position in enumerate(target.tolist()):
        if position == previous:
            pass
        elif position == 1:
            names[index] = "Buy"
            label_ids[index] = TradeLabel.BUY.value
        elif position == -1:
            names[index] = "Sell"
            label_ids[index] = TradeLabel.SELL.value
        else:
            raise ValueError(
                f"Position transition cannot be represented: {previous} -> {position}"
            )
        previous = position
    return names, label_ids


def build_oracle_labels_train_only(
    train_df: pd.DataFrame,
    price_col: str = "adj_close",
    initial_capital: float = 10_000.0,
    fee_per_trade: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if train_df is None or train_df.empty:
        raise ValueError("train_df cannot be empty.")
    missing = [
        column for column in ("date", price_col) if column not in train_df.columns
    ]
    if missing:
        raise ValueError(f"Missing oracle columns: {missing}")
    work = train_df.sort_values("date").reset_index(drop=True).copy()
    prices = pd.to_numeric(work[price_col], errors="coerce").to_numpy(dtype=np.float64)
    if len(prices) < 2 or not np.isfinite(prices).all():
        raise ValueError(
            "Oracle training prices must contain at least two finite values."
        )
    forward_returns = compute_forward_returns(prices)
    solution = solve_oracle_executed_positions_dp(
        forward_returns,
        fee_per_trade=fee_per_trade,
        initial_capital=initial_capital,
    )
    executed = np.asarray(solution["executed_positions"], dtype=np.int8)
    target = executed_to_target_positions(executed)
    label_names, label_ids = target_positions_to_labels(target)
    work["Label"] = label_names
    work["Label_id"] = label_ids
    work["oracle_target_position"] = target
    work["oracle_executed_position"] = executed
    work["oracle_forward_return"] = forward_returns

    evaluation = evaluate_strategy_vs_buy_hold(
        work,
        label_ids,
        initial_capital=initial_capital,
        price_col=price_col,
        fee_per_trade=fee_per_trade,
        position_mode="long_short",
        execution_delay=1,
    )
    used_returns = forward_returns[1:]
    report: dict[str, float | int] = {
        "oracle_final_capital_dp": float(solution["final_capital"]),
        "oracle_final_capital_eval": float(evaluation["model_final_capital"]),
        "oracle_pnl": float(evaluation["model_pnl"]),
        "buy_hold_final_capital": float(evaluation["buy_hold_final_capital"]),
        "buy_hold_pnl": float(evaluation["buy_hold_pnl"]),
        "outperformance_vs_buy_hold": float(evaluation["outperformance"]),
        "n_trades": int(solution["n_trades"]),
        "n_rows_train": int(len(work)),
        "mean_abs_return_used": float(np.mean(np.abs(used_returns)))
        if len(used_returns)
        else 0.0,
        "oracle_abs_sign_no_fee_final": float(
            initial_capital * np.prod(1.0 + np.abs(used_returns))
        )
        if len(used_returns)
        else float(initial_capital),
    }
    report["dp_eval_abs_gap"] = abs(
        float(report["oracle_final_capital_dp"])
        - float(report["oracle_final_capital_eval"])
    )
    return work, report


def merge_oracle_labels_on_train_only(
    labeled_df: pd.DataFrame,
    oracle_csv_path: str | Path,
    train_ratio: float,
    val_ratio: float,
) -> pd.DataFrame:
    work = labeled_df.sort_values("date").reset_index(drop=True).copy()
    train, val, test = chronological_train_val_test_split(
        work,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    path = Path(oracle_csv_path)
    if not path.exists():
        return work
    oracle = pd.read_csv(path)
    missing = {"date", "Label_id"} - set(oracle.columns)
    if missing:
        raise ValueError(f"Missing oracle CSV columns: {sorted(missing)}")
    oracle["date"] = pd.to_datetime(oracle["date"], errors="coerce")
    oracle = oracle.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    oracle = oracle[["date", "Label_id"]].rename(
        columns={"Label_id": "oracle_label_id"}
    )
    train["date"] = pd.to_datetime(train["date"], errors="coerce")
    merged = train.merge(oracle, on="date", how="left")
    use_oracle = merged["oracle_label_id"].notna()
    merged.loc[use_oracle, "Label_id"] = merged.loc[
        use_oracle, "oracle_label_id"
    ].astype(int)
    merged["Label_id"] = merged["Label_id"].astype(np.int64)
    merged["Label"] = merged["Label_id"].map(LABEL_ID_TO_NAME)
    merged = merged.drop(columns=["oracle_label_id"])
    return (
        pd.concat([merged, val, test], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )


def apply_oracle_labels_on_all_data(
    df: pd.DataFrame,
    price_col: str = "adj_close",
    capital: float = 10_000.0,
    fee_per_trade: float = 0.0,
) -> pd.DataFrame:
    labels, _ = build_oracle_labels_train_only(
        df.sort_values("date").reset_index(drop=True),
        price_col=price_col,
        initial_capital=capital,
        fee_per_trade=fee_per_trade,
    )
    return labels


def _default_output_path(ticker: str | None) -> Path:
    safe_name = (
        "all_tickers" if not ticker else ticker.replace("/", "_").replace(".", "_")
    )
    return derived_data_dir() / f"oracle_labels_train_{safe_name}.csv"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate oracle-DP labels on training rows only."
    )
    parser.add_argument("--data-dir", type=Path, default=default_market_dataset_path())
    parser.add_argument("--ticker", default="EN.PA")
    parser.add_argument("--price-col", default="adj_close")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--fee-per-trade", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    frame = read_parquet_dataset(args.data_dir)
    if args.ticker:
        if "ticker" not in frame.columns:
            raise ValueError("Dataset has no ticker column.")
        frame = frame[frame["ticker"] == args.ticker].copy()
    if frame.empty:
        raise ValueError("No rows remain after ticker filtering.")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("Dataset contains invalid dates.")
    frame = frame.sort_values("date").reset_index(drop=True)
    train, val, test = chronological_train_val_test_split(
        frame,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    labels, report = build_oracle_labels_train_only(
        train,
        price_col=args.price_col,
        initial_capital=args.capital,
        fee_per_trade=args.fee_per_trade,
    )
    output = args.output or _default_output_path(args.ticker)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output, index=False)
    print(
        f"oracle rows={len(labels)} split={len(train)}/{len(val)}/{len(test)} "
        f"final={report['oracle_final_capital_eval']:.2f} output={output}"
    )


__all__ = [
    "apply_oracle_labels_on_all_data",
    "build_oracle_labels_train_only",
    "compute_forward_returns",
    "executed_to_target_positions",
    "merge_oracle_labels_on_train_only",
    "solve_oracle_executed_positions_dp",
    "target_positions_to_labels",
]


if __name__ == "__main__":
    main()
