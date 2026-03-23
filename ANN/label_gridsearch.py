import numpy as np
import pandas as pd

from ANN import (
    read_parquet_dataset,
    chronological_train_val_test_split,
    evaluate_strategy_vs_buy_hold,
    enforce_alternating_signals,
    compute_returns)

DATA_DIR = "ANN/datasets/cac40_daily.parquet"

df = read_parquet_dataset(DATA_DIR)
df = df[df["ticker"] == "EN.PA"].copy()
train, val, test = chronological_train_val_test_split(df, train_ratio=0.7, val_ratio=0.15)


def benchmark(df, price_col="adj_close", capital = 10_000):
    returns = df[price_col].pct_change().fillna(0.0).to_numpy(np.float64)
    buy_hold =  float(capital * np.prod(1.0 + returns))
    return buy_hold

def backtest_long(returns, labels, fees=0.0, capital=10_000.0):
    """
    returns: array-like de rendements simples (ex: pct_change), shape (N,)
    labels:  array-like d'entiers {0:Sell, 1:Hold, 2:Buy}, shape (N,)
    fees: coût fixe par changement de position (en valeur absolue de capital)
    """
    r = np.asarray(returns, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)

    if r.shape[0] != y.shape[0]:
        raise ValueError("returns et labels doivent avoir la même longueur.")
    if r.shape[0] == 0:
        raise ValueError("Séries vides.")

    portfolio = float(capital)
    position = 0  # 0 = flat, 1 = long
    n_trades = 0

    for i in range(r.shape[0]):
        # 1) applique le rendement avec la position courante (signal execute au bar suivant)
        if position == 1:
            portfolio *= (1.0 + r[i])
        if portfolio <= 0:
            return {
                "final_capital": 0.0,
                "pnl": -float(capital),
                "n_trades": n_trades,
                "stopped_early": True,
            }

        # 2) appliquer le signal (changement de position pour le bar suivant)
        prev_position = position
        if y[i] == 2:      # Buy
            position = 1
        elif y[i] == 0:    # Sell
            position = 0
        elif y[i] == 1:    # Hold
            pass
        else:
            raise ValueError(f"Label inconnu: {y[i]}")

        # 3) si changement de position => frais
        trade_actions = abs(position - prev_position)  # 0 ou 1 en long-only
        if trade_actions > 0:
            portfolio -= float(fees) * trade_actions
            n_trades += int(trade_actions)
            if portfolio <= 0:
                return {
                    "final_capital": 0.0,
                    "pnl": -float(capital),
                    "n_trades": n_trades,
                    "stopped_early": True,
                }

    return {
        "final_capital": float(portfolio),
        "pnl": float(portfolio - float(capital)),
        "n_trades": int(n_trades),
        "stopped_early": False,
    }

def backtest_long_short(returns, labels, fees=0.0, capital=10_000.0):
    """
    returns: array-like de rendements simples (ex: pct_change), shape (N,)
    labels:  array-like d'entiers {0:Sell, 1:Hold, 2:Buy}, shape (N,)
    fees: coût fixe par changement de position (en valeur absolue de capital)
    """
    r = np.asarray(returns, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)

    if r.shape[0] != y.shape[0]:
        raise ValueError("returns et labels doivent avoir la même longueur.")
    if r.shape[0] == 0:
        raise ValueError("Séries vides.")

    portfolio = float(capital)
    position = 0  # -1 = short, 0 = flat, 1 = long
    n_trades = 0

    for i in range(r.shape[0]):
        # 1) applique le rendement avec la position courante (signal execute au bar suivant)
        if position == 1:
            portfolio *= (1.0 + r[i])
        elif position == -1:
            portfolio *= (1.0 - r[i])
        if portfolio <= 0:
            return {
                "final_capital": 0.0,
                "pnl": -float(capital),
                "n_trades": n_trades,
                "stopped_early": True,
            }

        # 2) appliquer le signal (changement de position pour le bar suivant)
        prev_position = position
        if y[i] == 2:      # Buy
            position = 1
        elif y[i] == 0:    # Sell
            position = -1
        elif y[i] == 1:    # Hold
            pass
        else:
            raise ValueError(f"Label inconnu: {y[i]}")

        # 3) frais: un flip +1 <-> -1 vaut 2 transactions (close + open)
        trade_actions = abs(position - prev_position)  # 0, 1 ou 2
        if trade_actions > 0:
            portfolio -= float(fees) * trade_actions
            n_trades += int(trade_actions)
            if portfolio <= 0:
                return {
                    "final_capital": 0.0,
                    "pnl": -float(capital),
                    "n_trades": n_trades,
                    "stopped_early": True,
                }

    return {
        "final_capital": float(portfolio),
        "pnl": float(portfolio - float(capital)),
        "n_trades": int(n_trades),
        "stopped_early": False,
    }


def label_gridsearch(df, price_col="adj_close", fees=1.0, capital=10_000.0):
    """
    Grid search simple sur (window, buy_buffer, sell_buffer).
    Retourne best_params, best_metrics, all_results_df
    """
    if df is None or df.empty:
        raise ValueError("df vide")
    if price_col not in df.columns:
        raise ValueError(f"{price_col} manquant")
    if "date" not in df.columns:
        raise ValueError("date manquant")

    work = df.sort_values("date").copy()

    windows = list(range(5, 61))
    buy_buffers = [0.0, 0.001, 0.002, 0.005]
    sell_buffers = [0.0, 0.001, 0.002, 0.005]

    label_map = {"Sell": 0, "Hold": 1, "Buy": 2}
    best_score = -np.inf
    best_params = None
    best_metrics = None
    rows = []

    buy_hold = benchmark(work, price_col=price_col, capital=capital)

    for window in windows:
        prev_min = work[price_col].shift(1).rolling(window).min()
        prev_max = work[price_col].shift(1).rolling(window).max()

        for buy_buffer in buy_buffers:
            for sell_buffer in sell_buffers:
                raw_labels = np.where(
                    work[price_col] <= prev_min * (1.0 - buy_buffer),
                    "Buy",
                    np.where(
                        work[price_col] >= prev_max * (1.0 + sell_buffer),
                        "Sell",
                        "Hold",
                    ),
                )

                raw_labels = pd.Series(raw_labels, index=work.index, dtype="object")
                raw_labels.loc[prev_min.isna() | prev_max.isna()] = "Hold"

                labels = enforce_alternating_signals(raw_labels.tolist())
                label_ids = pd.Series(labels, index=work.index).map(label_map).to_numpy(np.int64)

                rets = work[price_col].pct_change().fillna(0.0).to_numpy(np.float64)
                metrics = backtest_long(rets, label_ids, fees=fees, capital=capital)

                outperformance = float(metrics["final_capital"]) - buy_hold
                score = outperformance

                row = {
                    "window": window,
                    "buy_buffer": buy_buffer,
                    "sell_buffer": sell_buffer,
                    "score": score,
                    "final_capital": metrics["final_capital"],
                    "pnl": metrics["pnl"],
                    "n_trades": metrics["n_trades"],
                }
                rows.append(row)

                if score > best_score:
                    best_score = score
                    best_params = {
                        "window": window,
                        "buy_buffer": buy_buffer,
                        "sell_buffer": sell_buffer,
                    }
                    best_metrics = metrics

    results_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return best_params, best_metrics, results_df


def label_gridsearch_long_short(df, price_col="adj_close", fees=1.0, capital=10_000.0):
    """
    Grid search simple sur (window, buy_buffer, sell_buffer).
    Retourne best_params, best_metrics, all_results_df
    """
    if df is None or df.empty:
        raise ValueError("df vide")
    if price_col not in df.columns:
        raise ValueError(f"{price_col} manquant")
    if "date" not in df.columns:
        raise ValueError("date manquant")

    work = df.sort_values("date").copy()

    windows = list(range(2, 61))
    buy_buffers = [0.0, 0.001, 0.002, 0.005]
    sell_buffers = [0.0, 0.001, 0.002, 0.005]

    label_map = {"Sell": 0, "Hold": 1, "Buy": 2}
    best_score = -np.inf
    best_params = None
    best_metrics = None
    rows = []

    buy_hold = benchmark(work, price_col=price_col, capital=capital)


    for window in windows:
        prev_min = work[price_col].shift(1).rolling(window).min()
        prev_max = work[price_col].shift(1).rolling(window).max()

        for buy_buffer in buy_buffers:
            for sell_buffer in sell_buffers:
                raw_labels = np.where(
                    work[price_col] <= prev_min * (1.0 - buy_buffer),
                    "Buy",
                    np.where(
                        work[price_col] >= prev_max * (1.0 + sell_buffer),
                        "Sell",
                        "Hold",
                    ),
                )

                raw_labels = pd.Series(raw_labels, index=work.index, dtype="object")
                raw_labels.loc[prev_min.isna() | prev_max.isna()] = "Hold"

                labels = enforce_alternating_signals(raw_labels.tolist())
                label_ids = pd.Series(labels, index=work.index).map(label_map).to_numpy(np.int64)

                rets = work[price_col].pct_change().fillna(0.0).to_numpy(np.float64)
                metrics = backtest_long_short(rets, label_ids, fees=fees, capital=capital)
                
                outperformance = float(metrics["final_capital"]) - buy_hold
                score = outperformance

                row = {
                    "window": window,
                    "buy_buffer": buy_buffer,
                    "sell_buffer": sell_buffer,
                    "score": score,
                    "final_capital": metrics["final_capital"],
                    "pnl": metrics["pnl"],
                    "n_trades": metrics["n_trades"],
                }
                rows.append(row)

                if score > best_score:
                    best_score = score
                    best_params = {
                        "window": window,
                        "buy_buffer": buy_buffer,
                        "sell_buffer": sell_buffer,
                    }
                    best_metrics = metrics

    results_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return best_params, best_metrics, results_df


best_params, best_metrics, results_df = label_gridsearch(val, fees=2.0)
print(best_params)
print(best_metrics)
print(results_df)

best_params, best_metrics, results_df = label_gridsearch_long_short(val, fees=2.0)
print(best_params)
print(best_metrics)
print(results_df)
