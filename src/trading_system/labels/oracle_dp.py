from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import pyarrow.dataset as ds
except Exception:
    ds = None

from trading_system.paths import default_market_dataset_path, derived_data_dir

LABEL_ID_TO_NAME = {0: "Sell", 1: "Hold", 2: "Buy"}
LABEL_NAME_TO_ID = {name: idx for idx, name in LABEL_ID_TO_NAME.items()}


def read_parquet_dataset(base_dir: Path) -> pd.DataFrame:
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(base_dir)

    if ds is not None:
        dataset = ds.dataset(str(base_dir), format="parquet", partitioning="hive")
        return dataset.to_table().to_pandas()

    if not hasattr(pd, "read_parquet"):
        raise RuntimeError(
            "Lecture parquet indisponible: ni pyarrow.dataset ni pandas.read_parquet ne sont disponibles."
        )

    if base_dir.is_file():
        return pd.read_parquet(base_dir)

    parquet_files = sorted(base_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Aucun fichier parquet trouve dans: {base_dir}")
    parts = [pd.read_parquet(path) for path in parquet_files]
    return pd.concat(parts, ignore_index=True)


def chronological_train_val_test_split(df, train_ratio=0.7, val_ratio=0.15):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio doit etre dans l'intervalle ]0, 1[.")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio doit etre dans l'intervalle ]0, 1[.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio doit etre strictement inferieur a 1.")
    if len(df) < 3:
        raise ValueError("Il faut au moins 3 lignes pour faire un split train/val/test.")

    train_end = int(len(df) * train_ratio)
    val_end = int(len(df) * (train_ratio + val_ratio))

    train_end = min(max(train_end, 1), len(df) - 2)
    val_end = min(max(val_end, train_end + 1), len(df) - 1)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def _signals_to_positions(pred_labels: np.ndarray) -> np.ndarray:
    positions = []
    current_position = 0.0
    for label in pred_labels:
        if label == 2:
            current_position = 1.0
        elif label == 0:
            current_position = -1.0
        positions.append(current_position)
    return np.asarray(positions, dtype=np.float64)


def evaluate_strategy_vs_buy_hold(
    test_frame: pd.DataFrame,
    pred_labels: np.ndarray,
    initial_capital: float = 10_000.0,
    price_col: str = "adj_close",
    fee_per_trade: float = 0.0,
) -> dict:
    if len(test_frame) != len(pred_labels):
        raise ValueError("Mismatch entre nombre de predictions et lignes.")
    if len(test_frame) < 2:
        raise ValueError("Il faut au moins 2 lignes pour calculer un PnL.")
    if fee_per_trade < 0:
        raise ValueError("fee_per_trade doit etre >= 0.")

    prices = test_frame[price_col].to_numpy(dtype=np.float64)
    forward_returns = np.zeros(len(prices), dtype=np.float64)
    forward_returns[:-1] = (prices[1:] / prices[:-1]) - 1.0

    target_positions = _signals_to_positions(np.asarray(pred_labels, dtype=np.int64))
    executed_positions = np.zeros_like(target_positions)
    executed_positions[1:] = target_positions[:-1]

    prev_positions = np.zeros_like(executed_positions)
    prev_positions[1:] = executed_positions[:-1]
    turnover = np.abs(executed_positions - prev_positions)

    strategy_returns = executed_positions * forward_returns
    model_curve = np.empty(len(prices), dtype=np.float64)
    capital = float(initial_capital)
    for i in range(len(prices)):
        capital *= (1.0 + strategy_returns[i])
        capital -= float(fee_per_trade) * turnover[i]
        if capital < 0:
            capital = 0.0
        model_curve[i] = capital

    buy_hold_curve = initial_capital * np.cumprod(1.0 + forward_returns)

    model_final = float(model_curve[-1])
    buy_hold_final = float(buy_hold_curve[-1])

    return {
        "initial_capital": float(initial_capital),
        "model_final_capital": model_final,
        "buy_hold_final_capital": buy_hold_final,
        "model_pnl": model_final - float(initial_capital),
        "buy_hold_pnl": buy_hold_final - float(initial_capital),
        "outperformance": model_final - buy_hold_final,
    }


def _allowed_next_executed_positions(prev_pos: int) -> tuple[int, ...]:
    if prev_pos == 0:
        return (-1, 0, 1)
    if prev_pos in (-1, 1):
        return (-1, 1)
    raise ValueError(f"Position invalide: {prev_pos}")


def _compute_forward_returns(prices: np.ndarray) -> np.ndarray:
    if prices.ndim != 1:
        raise ValueError("prices doit etre un vecteur 1D.")
    if len(prices) == 0:
        raise ValueError("prices vide.")

    out = np.zeros(len(prices), dtype=np.float64)
    if len(prices) > 1:
        out[:-1] = (prices[1:] / prices[:-1]) - 1.0
    return out


def solve_oracle_executed_positions_dp(
    forward_returns: np.ndarray,
    fee_per_trade: float = 0.0,
    initial_capital: float = 10_000.0,
) -> dict:
    """Find the globally-optimal executed-position path with DP.

    State e_t is the executed position at bar t in {-1, 0, +1}.
    Transition constraints are aligned with current label semantics:
    - From 0: can stay 0, go +1, or go -1
    - From +/-1: can stay same side or flip side (cannot go back flat)
    """
    if fee_per_trade < 0:
        raise ValueError("fee_per_trade doit etre >= 0.")
    if initial_capital <= 0:
        raise ValueError("initial_capital doit etre > 0.")

    r = np.asarray(forward_returns, dtype=np.float64)
    if r.ndim != 1 or len(r) < 2:
        raise ValueError("forward_returns doit etre un vecteur 1D de longueur >= 2.")

    state_values = np.asarray([-1, 0, 1], dtype=np.int8)
    state_to_idx = {int(s): i for i, s in enumerate(state_values)}

    n = len(r)
    dp_capital = np.full((n, len(state_values)), -np.inf, dtype=np.float64)
    parent_idx = np.full((n, len(state_values)), -1, dtype=np.int8)

    start_idx = state_to_idx[0]
    dp_capital[0, start_idx] = float(initial_capital)
    parent_idx[0, start_idx] = start_idx

    for t in range(1, n):
        rt = float(r[t])
        for prev_i, prev_state in enumerate(state_values):
            prev_cap = dp_capital[t - 1, prev_i]
            if not np.isfinite(prev_cap):
                continue

            for next_state in _allowed_next_executed_positions(int(prev_state)):
                next_i = state_to_idx[int(next_state)]

                next_cap = prev_cap * (1.0 + float(next_state) * rt)
                next_cap -= float(fee_per_trade) * abs(int(next_state) - int(prev_state))
                if next_cap < 0.0:
                    next_cap = 0.0

                if next_cap > dp_capital[t, next_i]:
                    dp_capital[t, next_i] = next_cap
                    parent_idx[t, next_i] = prev_i

    end_i = int(np.argmax(dp_capital[-1]))
    best_final_capital = float(dp_capital[-1, end_i])
    if not np.isfinite(best_final_capital):
        raise RuntimeError("Echec DP: aucun chemin valide trouve.")

    best_state_indices = np.empty(n, dtype=np.int8)
    best_state_indices[-1] = end_i
    for t in range(n - 1, 0, -1):
        prev_i = int(parent_idx[t, int(best_state_indices[t])])
        if prev_i < 0:
            raise RuntimeError(f"Echec backtracking DP a t={t}.")
        best_state_indices[t - 1] = prev_i

    executed_positions = state_values[best_state_indices].astype(np.int8, copy=False)
    turnover = np.abs(np.diff(executed_positions.astype(np.int16)))
    n_trades = int(turnover.sum())

    return {
        "executed_positions": executed_positions,
        "final_capital": best_final_capital,
        "pnl": best_final_capital - float(initial_capital),
        "n_trades": n_trades,
    }


def executed_to_target_positions(executed_positions: np.ndarray) -> np.ndarray:
    """Convert executed positions e_t to target positions inferred from labels."""
    e = np.asarray(executed_positions, dtype=np.int8)
    if e.ndim != 1 or len(e) == 0:
        raise ValueError("executed_positions invalide.")

    target = np.empty_like(e)
    if len(e) == 1:
        target[0] = 0
        return target

    target[:-1] = e[1:]
    target[-1] = e[-1]
    return target


def target_positions_to_labels(target_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build canonical Sell/Hold/Buy labels from target positions."""
    target = np.asarray(target_positions, dtype=np.int8)
    if target.ndim != 1 or len(target) == 0:
        raise ValueError("target_positions invalide.")

    labels = np.full(len(target), "Hold", dtype=object)
    label_ids = np.full(len(target), LABEL_NAME_TO_ID["Hold"], dtype=np.int64)

    prev = 0
    for i, pos in enumerate(target.tolist()):
        if pos == prev:
            labels[i] = "Hold"
            label_ids[i] = LABEL_NAME_TO_ID["Hold"]
        elif pos == 1:
            labels[i] = "Buy"
            label_ids[i] = LABEL_NAME_TO_ID["Buy"]
        elif pos == -1:
            labels[i] = "Sell"
            label_ids[i] = LABEL_NAME_TO_ID["Sell"]
        else:
            raise ValueError(f"Transition non representable vers pos={pos} (prev={prev})")
        prev = pos

    return labels, label_ids


def build_oracle_labels_train_only(
    train_df: pd.DataFrame,
    price_col: str = "adj_close",
    initial_capital: float = 10_000.0,
    fee_per_trade: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    if train_df is None or train_df.empty:
        raise ValueError("train_df vide.")
    if price_col not in train_df.columns:
        raise ValueError(f"Colonne prix manquante: {price_col}")
    if "date" not in train_df.columns:
        raise ValueError("Colonne date manquante.")

    work = train_df.sort_values("date").copy()
    prices = pd.to_numeric(work[price_col], errors="coerce").to_numpy(np.float64)
    if np.isnan(prices).any():
        raise ValueError(f"{price_col} contient des NaN/non numeriques.")
    if len(prices) < 2:
        raise ValueError("Train trop court: besoin d'au moins 2 lignes.")

    forward_returns = _compute_forward_returns(prices)
    dp_out = solve_oracle_executed_positions_dp(
        forward_returns=forward_returns,
        fee_per_trade=fee_per_trade,
        initial_capital=initial_capital,
    )

    target_positions = executed_to_target_positions(dp_out["executed_positions"])
    labels, label_ids = target_positions_to_labels(target_positions)

    out = work.copy()
    out["Label"] = labels
    out["Label_id"] = label_ids
    out["oracle_target_position"] = target_positions
    out["oracle_executed_position"] = dp_out["executed_positions"]
    out["oracle_forward_return"] = forward_returns

    eval_out = evaluate_strategy_vs_buy_hold(
        out,
        label_ids,
        initial_capital=initial_capital,
        price_col=price_col,
        fee_per_trade=fee_per_trade,
    )

    report = {
        "oracle_final_capital_dp": float(dp_out["final_capital"]),
        "oracle_final_capital_eval": float(eval_out["model_final_capital"]),
        "oracle_pnl": float(eval_out["model_pnl"]),
        "buy_hold_final_capital": float(eval_out["buy_hold_final_capital"]),
        "buy_hold_pnl": float(eval_out["buy_hold_pnl"]),
        "outperformance_vs_buy_hold": float(eval_out["outperformance"]),
        "n_trades": int(dp_out["n_trades"]),
        "n_rows_train": int(len(out)),
    }
    used_rets = forward_returns[1:] if len(forward_returns) > 1 else np.asarray([], dtype=np.float64)
    report["mean_abs_return_used"] = float(np.mean(np.abs(used_rets))) if len(used_rets) else 0.0
    report["oracle_abs_sign_no_fee_final"] = float(
        initial_capital * np.prod(1.0 + np.abs(used_rets))
    ) if len(used_rets) else float(initial_capital)
    report["dp_eval_abs_gap"] = abs(
        report["oracle_final_capital_dp"] - report["oracle_final_capital_eval"]
    )

    return out, report


def _default_data_dir() -> Path:
    return default_market_dataset_path()


def _default_output_path(ticker: str | None) -> Path:
    safe = "all_tickers" if not ticker else ticker.replace("/", "_").replace(".", "_")
    return derived_data_dir() / f"oracle_labels_train_{safe}.csv"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Oracle DP labels (Sell/Hold/Buy) computed ONLY on train split."
    )
    parser.add_argument("--data-dir", type=Path, default=_default_data_dir())
    parser.add_argument("--ticker", type=str, default="EN.PA")
    parser.add_argument("--price-col", type=str, default="adj_close")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--fee-per-trade", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()

    df = read_parquet_dataset(data_dir)
    if args.ticker:
        if "ticker" not in df.columns:
            raise ValueError("Le dataset n'a pas de colonne 'ticker' pour filtrer --ticker.")
        df = df[df["ticker"] == args.ticker].copy()
    if df.empty:
        raise ValueError("Aucune ligne apres filtrage ticker.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Dates invalides detectees dans le dataset.")
    df = df.sort_values("date").reset_index(drop=True)

    train_df, val_df, test_df = chronological_train_val_test_split(
        df, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    oracle_train, report = build_oracle_labels_train_only(
        train_df=train_df,
        price_col=args.price_col,
        initial_capital=args.capital,
        fee_per_trade=args.fee_per_trade,
    )

    output_path = args.output if args.output is not None else _default_output_path(args.ticker)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    oracle_train.to_csv(output_path, index=False)

    label_counts = oracle_train["Label"].value_counts().reindex(["Sell", "Hold", "Buy"], fill_value=0)

    train_start = oracle_train["date"].min()
    train_end = oracle_train["date"].max()

    print("Oracle DP labels (train only)")
    print(
        f"| ticker = {args.ticker if args.ticker else 'ALL'}"
        f" | rows(total) = {len(df)}"
        f" | rows(train/val/test) = {len(train_df)}/{len(val_df)}/{len(test_df)}"
    )
    print(f"| train_range = {train_start} -> {train_end}")
    print(
        f"| fee = {args.fee_per_trade:.4f}"
        f" | capital = {args.capital:.2f}"
        f" | dp_eval_gap = {report['dp_eval_abs_gap']:.10f}"
    )
    print(
        f"| oracle_final = {report['oracle_final_capital_eval']:.2f}"
        f" | buy_hold_final = {report['buy_hold_final_capital']:.2f}"
        f" | outperformance = {report['outperformance_vs_buy_hold']:.2f}"
        f" | n_trades = {report['n_trades']}"
    )
    print(
        f"| mean_abs_ret(used) = {report['mean_abs_return_used']:.6f}"
        f" | no_fee_abs_sign_upper = {report['oracle_abs_sign_no_fee_final']:.2f}"
    )
    print(
        f"| labels_train -> Sell={int(label_counts['Sell'])}"
        f" Hold={int(label_counts['Hold'])}"
        f" Buy={int(label_counts['Buy'])}"
    )
    print(f"| output = {output_path}")


if __name__ == "__main__":
    main()
