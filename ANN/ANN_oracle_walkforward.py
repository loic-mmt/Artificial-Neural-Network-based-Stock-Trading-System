from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow.dataset as ds
except Exception:
    ds = None

try:
    from ANN.features import features, compute_market_features
except ModuleNotFoundError:
    from features import features, compute_market_features

LABEL_ID_TO_NAME = {0: "Sell", 1: "Hold", 2: "Buy"}
N_CLASSES = 3


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
    fee_per_trade: float = 1.0,
    initial_capital: float = 10_000.0,
) -> dict:
    if fee_per_trade < 0:
        raise ValueError("fee_per_trade doit etre >= 0.")
    if initial_capital <= 0:
        raise ValueError("initial_capital doit etre > 0.")

    r = np.asarray(forward_returns, dtype=np.float64)
    if r.ndim != 1 or len(r) < 2:
        raise ValueError("forward_returns doit etre 1D de longueur >= 2.")

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
        raise RuntimeError("Echec DP: aucun chemin valide.")

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
    target = np.asarray(target_positions, dtype=np.int8)
    if target.ndim != 1 or len(target) == 0:
        raise ValueError("target_positions invalide.")

    labels = np.full(len(target), "Hold", dtype=object)
    label_ids = np.full(len(target), 1, dtype=np.int64)

    prev = 0
    for i, pos in enumerate(target.tolist()):
        if pos == prev:
            labels[i] = "Hold"
            label_ids[i] = 1
        elif pos == 1:
            labels[i] = "Buy"
            label_ids[i] = 2
        elif pos == -1:
            labels[i] = "Sell"
            label_ids[i] = 0
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
        position_mode="long_short",
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
    report["dp_eval_abs_gap"] = abs(
        report["oracle_final_capital_dp"] - report["oracle_final_capital_eval"]
    )

    return out, report


def build_forward_return_labels(
    df: pd.DataFrame,
    price_col: str = "adj_close",
    horizon: int = 1,
    buy_threshold: float = 0.002,
    sell_threshold: float = 0.002,
) -> tuple[pd.DataFrame, dict]:
    """Build learnable labels from forward return over a fixed horizon."""
    if horizon <= 0:
        raise ValueError("horizon doit etre > 0.")
    if buy_threshold < 0 or sell_threshold < 0:
        raise ValueError("buy_threshold/sell_threshold doivent etre >= 0.")
    if df is None or df.empty:
        raise ValueError("df vide.")
    if price_col not in df.columns:
        raise ValueError(f"Colonne prix manquante: {price_col}")

    out = df.sort_values("date").copy()
    price = pd.to_numeric(out[price_col], errors="coerce").astype(float)
    if price.isna().any():
        raise ValueError(f"{price_col} contient des NaN/non numeriques.")

    fwd_ret = (price.shift(-horizon) / price) - 1.0
    label_id = np.full(len(out), 1, dtype=np.int64)  # Hold by default
    label_id[fwd_ret > float(buy_threshold)] = 2
    label_id[fwd_ret < -float(sell_threshold)] = 0

    out["fwd_ret"] = fwd_ret
    out["Label_id"] = label_id
    out["Label"] = out["Label_id"].map(LABEL_ID_TO_NAME)

    # Tail without future horizon is forced to Hold.
    out.loc[out["fwd_ret"].isna(), "Label_id"] = 1
    out.loc[out["fwd_ret"].isna(), "Label"] = "Hold"

    report = {
        "horizon": int(horizon),
        "buy_threshold": float(buy_threshold),
        "sell_threshold": float(sell_threshold),
        "n_rows": int(len(out)),
        "n_buy": int((out["Label_id"] == 2).sum()),
        "n_hold": int((out["Label_id"] == 1).sum()),
        "n_sell": int((out["Label_id"] == 0).sum()),
    }
    return out, report


def read_parquet_dataset(base_dir: Path) -> pd.DataFrame:
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(base_dir)

    if ds is not None:
        dataset = ds.dataset(str(base_dir), format="parquet", partitioning="hive")
        return dataset.to_table().to_pandas()

    if not hasattr(pd, "read_parquet"):
        raise RuntimeError(
            "Lecture parquet indisponible: ni pyarrow.dataset ni pandas.read_parquet."
        )

    if base_dir.is_file():
        return pd.read_parquet(base_dir)

    parquet_files = sorted(base_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Aucun parquet trouve dans: {base_dir}")
    return pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)


def chronological_train_val_test_split(df, train_ratio=0.7, val_ratio=0.15):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio doit etre dans ]0, 1[.")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio doit etre dans ]0, 1[.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio doit etre < 1.")
    if len(df) < 3:
        raise ValueError("Il faut au moins 3 lignes.")

    train_end = int(len(df) * train_ratio)
    val_end = int(len(df) * (train_ratio + val_ratio))

    train_end = min(max(train_end, 1), len(df) - 2)
    val_end = min(max(val_end, train_end + 1), len(df) - 1)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def chronological_train_val_split(df, val_ratio=0.15):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio doit etre dans ]0, 1[.")
    if len(df) < 3:
        raise ValueError("Il faut au moins 3 lignes.")

    train_end = int(len(df) * (1.0 - val_ratio))
    train_end = min(max(train_end, 2), len(df) - 1)
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:].copy()
    return train_df, val_df


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def dropout_mask(shape, p):
    m = (np.random.rand(*shape) > p).astype(np.float32)
    return m / (1 - p)


def one_hot(y, k=3):
    Y = np.zeros((len(y), k), dtype=np.float32)
    Y[np.arange(len(y)), y] = 1.0
    return Y


def recall_for_label(y_true, y_pred, label):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = ((y_true == label) & (y_pred == label)).sum()
    fn = ((y_true == label) & (y_pred != label)).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def balanced_accuracy(y_true, y_pred, labels=(0, 1, 2)):
    recalls = [recall_for_label(y_true, y_pred, label) for label in labels]
    return float(np.mean(recalls))


def precision_recall_f1_for_label(y_true, y_pred, label):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = ((y_true == label) & (y_pred == label)).sum()
    fp = ((y_true != label) & (y_pred == label)).sum()
    fn = ((y_true == label) & (y_pred != label)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def macro_f1(y_true, y_pred, labels=(0, 1, 2)):
    vals = []
    for label in labels:
        _, _, f1 = precision_recall_f1_for_label(y_true, y_pred, label)
        vals.append(f1)
    return float(np.mean(vals))


def evaluate_predictions(y_true, y_pred):
    precision_sell, recall_sell, _ = precision_recall_f1_for_label(y_true, y_pred, 0)
    precision_hold, recall_hold, _ = precision_recall_f1_for_label(y_true, y_pred, 1)
    precision_buy, recall_buy, _ = precision_recall_f1_for_label(y_true, y_pred, 2)
    return {
        "acc": float((np.asarray(y_pred) == np.asarray(y_true)).mean()),
        "bal_acc": balanced_accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "precision_sell": precision_sell,
        "recall_sell": recall_sell,
        "precision_hold": precision_hold,
        "recall_hold": recall_hold,
        "precision_buy": precision_buy,
        "recall_buy": recall_buy,
    }


def compute_class_weights(y, num_classes=3):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = len(y) / (num_classes * counts)
    return weights.astype(np.float32)


def standardize_features(X, mean=None, std=None):
    X = X.astype(np.float32, copy=False)
    if mean is None:
        mean = X.mean(axis=0, keepdims=True)
    if std is None:
        std = X.std(axis=0, keepdims=True)
    std = std.copy()
    std[std == 0] = 1.0
    X_std = (X - mean) / std
    return X_std, mean, std


def forward_pass(X, W0, b0, W1, b1):
    z1 = X @ W0 + b0
    a1 = relu(z1)
    logits = a1 @ W1 + b1
    probs = softmax(logits)
    return z1, a1, logits, probs


def predict_with_thresholds(probs, buy_threshold=0.75, sell_threshold=0.75):
    preds = np.full(len(probs), 1, dtype=np.int64)  # Hold by default
    best_class = np.argmax(probs, axis=1)
    buy_mask = (best_class == 2) & (probs[:, 2] >= buy_threshold)
    sell_mask = (best_class == 0) & (probs[:, 0] >= sell_threshold)
    preds[buy_mask] = 2
    preds[sell_mask] = 0
    return preds


def predict_from_probs(
    probs: np.ndarray,
    decision_mode: str = "thresholds",
    buy_threshold: float = 0.75,
    sell_threshold: float = 0.75,
) -> np.ndarray:
    if decision_mode == "argmax":
        return np.argmax(probs, axis=1).astype(np.int64)
    if decision_mode == "thresholds":
        return predict_with_thresholds(probs, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    raise ValueError(f"decision_mode inconnu: {decision_mode}")


def threshold_gridsearch(probs, y_val, min_action_rate: float = 0.0):
    buy_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    sell_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    best_score = -np.inf
    best_thresholds = (0.75, 0.75)
    min_action_rate = float(min_action_rate)
    for bthresh in buy_thresholds:
        for sthresh in sell_thresholds:
            val_preds = predict_with_thresholds(probs, bthresh, sthresh)
            action_rate = float((val_preds != 1).mean())
            if action_rate < min_action_rate:
                continue
            score = evaluate_predictions(y_val, val_preds)["macro_f1"]
            if score > best_score:
                best_score = score
                best_thresholds = (bthresh, sthresh)
    if not np.isfinite(best_score):
        return (0.55, 0.35)
    return best_thresholds


def build_context_dataset(df, feature_cols, context_len, target_start=0, return_indices=False):
    if context_len <= 0:
        raise ValueError("context_len doit etre > 0.")

    values = df[feature_cols].to_numpy(dtype=np.float32)
    labels = df["Label_id"].to_numpy(dtype=np.int64)
    feat_dim = len(feature_cols)

    if len(df) < context_len:
        empty_x = np.empty((0, context_len * feat_dim), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.int64)
        empty_idx = np.empty((0,), dtype=np.int64)
        if return_indices:
            return empty_x, empty_y, empty_idx
        return empty_x, empty_y

    X_list, y_list, idx_list = [], [], []
    target_start = max(target_start, context_len - 1)
    for t in range(context_len - 1, len(df)):
        if t < target_start:
            continue
        window = values[t - context_len + 1 : t + 1]
        X_list.append(window.reshape(-1))
        y_list.append(labels[t])
        idx_list.append(t)

    if not X_list:
        empty_x = np.empty((0, context_len * feat_dim), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.int64)
        empty_idx = np.empty((0,), dtype=np.int64)
        if return_indices:
            return empty_x, empty_y, empty_idx
        return empty_x, empty_y

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    idx = np.asarray(idx_list, dtype=np.int64)
    if return_indices:
        return X, y, idx
    return X, y


def build_context_features(df_feat, feature_cols, context_len, target_start=0, return_indices=False):
    if context_len <= 0:
        raise ValueError("context_len doit etre > 0.")

    values = df_feat[feature_cols].to_numpy(dtype=np.float32)
    feat_dim = len(feature_cols)

    if len(df_feat) < context_len:
        empty_x = np.empty((0, context_len * feat_dim), dtype=np.float32)
        empty_idx = np.empty((0,), dtype=np.int64)
        if return_indices:
            return empty_x, empty_idx
        return empty_x

    X_list, idx_list = [], []
    target_start = max(target_start, context_len - 1)
    for t in range(context_len - 1, len(df_feat)):
        if t < target_start:
            continue
        window = values[t - context_len + 1 : t + 1]
        X_list.append(window.reshape(-1))
        idx_list.append(t)

    if not X_list:
        empty_x = np.empty((0, context_len * feat_dim), dtype=np.float32)
        empty_idx = np.empty((0,), dtype=np.int64)
        if return_indices:
            return empty_x, empty_idx
        return empty_x

    X = np.asarray(X_list, dtype=np.float32)
    idx = np.asarray(idx_list, dtype=np.int64)
    if return_indices:
        return X, idx
    return X


def labels_to_positions(pred_labels: np.ndarray, position_mode: str = "long_short"):
    positions = []
    current_position = 0.0
    for label in np.asarray(pred_labels, dtype=np.int64):
        if label == 2:
            current_position = 1.0
        elif label == 0:
            current_position = -1.0 if position_mode == "long_short" else 0.0
        positions.append(current_position)
    return np.asarray(positions, dtype=np.float64)


def evaluate_strategy_vs_buy_hold(
    test_frame,
    pred_labels,
    initial_capital=10_000.0,
    price_col="adj_close",
    fee_per_trade=0.0,
    position_mode: str = "long_short",
):
    if len(test_frame) != len(pred_labels):
        raise ValueError("Mismatch entre predictions et lignes test.")
    if len(test_frame) < 2:
        raise ValueError("Le test doit contenir au moins 2 lignes.")
    if fee_per_trade < 0:
        raise ValueError("fee_per_trade doit etre >= 0.")

    prices = test_frame[price_col].to_numpy(dtype=np.float64)
    forward_returns = np.zeros(len(prices), dtype=np.float64)
    forward_returns[:-1] = (prices[1:] / prices[:-1]) - 1.0

    target_positions = labels_to_positions(pred_labels, position_mode=position_mode)
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


def train_ann_on_labeled_history(
    labeled_history: pd.DataFrame,
    feature_cols: list[str],
    price_col: str = "adj_close",
    val_ratio: float = 0.15,
    context_len: int = 20,
    epochs: int = 200,
    alpha: float = 1e-3,
    hidden: int = 64,
    batch_size: int = 64,
    do_dropout: bool = False,
    dropout_percent: float = 0.1,
    decision_mode: str = "thresholds",
    min_action_rate: float = 0.0,
    early_stopping_patience: int = 30,
    early_stopping_min_delta: float = 1e-4,
):
    work = labeled_history.sort_values("date").reset_index(drop=True).copy()
    work["Label_id"] = pd.to_numeric(work["Label_id"], errors="coerce")
    work = work.dropna(subset=["Label_id"]).copy()
    work["Label_id"] = work["Label_id"].astype(np.int64)

    for col in feature_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
    work = work.dropna(subset=[price_col]).copy()

    train_df, val_df = chronological_train_val_split(work, val_ratio=val_ratio)

    fill_values = train_df[feature_cols].median().fillna(0.0)
    train_df[feature_cols] = train_df[feature_cols].fillna(fill_values)
    val_df[feature_cols] = val_df[feature_cols].fillna(fill_values)
    work[feature_cols] = work[feature_cols].fillna(fill_values)

    train_df = train_df.dropna(subset=feature_cols).copy()
    val_df = val_df.dropna(subset=feature_cols).copy()
    work = work.dropna(subset=feature_cols).copy()

    if len(train_df) < context_len + 1:
        raise ValueError("Historique train insuffisant pour le context_len.")
    if len(val_df) == 0:
        raise ValueError("Validation vide.")

    X_train_raw, y_train = build_context_dataset(train_df, feature_cols, context_len)
    val_prefix_len = min(context_len - 1, len(train_df))
    val_source = pd.concat([train_df.tail(val_prefix_len), val_df], ignore_index=True)
    X_val_raw, y_val = build_context_dataset(
        val_source, feature_cols, context_len, target_start=val_prefix_len
    )

    if len(X_train_raw) == 0:
        raise ValueError("Aucun echantillon train apres fenetrage.")
    if len(X_val_raw) == 0:
        raise ValueError("Aucun echantillon val apres fenetrage.")

    Y_train = one_hot(y_train, N_CLASSES)
    class_weights = compute_class_weights(y_train, num_classes=N_CLASSES)

    X_train, feature_mean, feature_std = standardize_features(X_train_raw)
    X_val, _, _ = standardize_features(X_val_raw, mean=feature_mean, std=feature_std)

    entry = X_train.shape[1]
    W0 = 0.01 * np.random.randn(entry, hidden).astype(np.float32)
    b0 = np.zeros((1, hidden), dtype=np.float32)
    W1 = 0.01 * np.random.randn(hidden, N_CLASSES).astype(np.float32)
    b1 = np.zeros((1, N_CLASSES), dtype=np.float32)

    N = len(X_train)
    best_macro_f1 = -1.0
    best = None
    no_improve_count = 0

    for ep in range(epochs):
        perm = np.random.permutation(N)
        Xp, Yp = X_train[perm], Y_train[perm]

        for start in range(0, N, batch_size):
            xb = Xp[start : start + batch_size]
            yb = Yp[start : start + batch_size]

            z1, a1, _, p = forward_pass(xb, W0, b0, W1, b1)
            dropout_applied = False
            if do_dropout and dropout_percent > 0:
                m1 = dropout_mask(a1.shape, dropout_percent)
                a1 *= m1
                dropout_applied = True

            logits = a1 @ W1 + b1
            p = softmax(logits)
            sample_weights = yb @ class_weights
            weight_sum = sample_weights.sum()

            dz2 = ((p - yb) * sample_weights[:, None]) / weight_sum
            dW1 = a1.T @ dz2
            db1 = dz2.sum(axis=0, keepdims=True)

            da1 = dz2 @ W1.T
            if dropout_applied:
                da1 *= m1

            dz1 = da1 * relu_derivative(z1)
            dW0 = xb.T @ dz1
            db0 = dz1.sum(axis=0, keepdims=True)

            W1 -= alpha * dW1
            b1 -= alpha * db1
            W0 -= alpha * dW0
            b0 -= alpha * db0

        _, _, _, val_probs = forward_pass(X_val, W0, b0, W1, b1)
        if decision_mode == "argmax":
            thresholds = (None, None)
            val_preds = predict_from_probs(val_probs, decision_mode="argmax")
        elif decision_mode == "thresholds":
            thresholds = threshold_gridsearch(val_probs, y_val, min_action_rate=min_action_rate)
            val_preds = predict_from_probs(
                val_probs,
                decision_mode="thresholds",
                buy_threshold=thresholds[0],
                sell_threshold=thresholds[1],
            )
        else:
            raise ValueError(f"decision_mode inconnu: {decision_mode}")
        val_metrics = evaluate_predictions(y_val, val_preds)

        macro_improved = val_metrics["macro_f1"] > (best_macro_f1 + early_stopping_min_delta)
        macro_tie = np.isclose(val_metrics["macro_f1"], best_macro_f1, atol=early_stopping_min_delta)
        bal_acc_improved = (
            best is not None and val_metrics["bal_acc"] > best["val_bal_acc"] + 1e-12
        )

        if best is None or macro_improved or (macro_tie and bal_acc_improved):
            best_macro_f1 = val_metrics["macro_f1"]
            best = {
                "W0": W0.copy(),
                "b0": b0.copy(),
                "W1": W1.copy(),
                "b1": b1.copy(),
                "feature_mean": feature_mean.copy(),
                "feature_std": feature_std.copy(),
                "feature_cols": feature_cols,
                "thresholds": thresholds,
                "decision_mode": decision_mode,
                "val_metrics": val_metrics,
                "val_bal_acc": val_metrics["bal_acc"],
                "best_macro_f1": best_macro_f1,
                "best_epoch": ep + 1,
                "context_len": int(context_len),
                "fill_values": fill_values.copy(),
                "history_for_context": work.copy(),
            }
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stopping_patience:
            break

    if best is None:
        raise RuntimeError("Aucun meilleur modele enregistre.")
    return best


def predict_chunk_with_model(model, chunk_df):
    feature_cols = model["feature_cols"]
    context_len = model["context_len"]
    fill_values = model["fill_values"]
    history_context = model["history_for_context"]

    chunk = chunk_df.copy()
    for col in feature_cols:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
    chunk[feature_cols] = chunk[feature_cols].fillna(fill_values)
    chunk = chunk.dropna(subset=feature_cols).reset_index(drop=True)

    if chunk.empty:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    prefix_len = min(context_len - 1, len(history_context))
    prefix_feat = history_context.tail(prefix_len)[feature_cols].copy()
    source_feat = pd.concat([prefix_feat, chunk[feature_cols]], ignore_index=True)

    X_chunk_raw, source_idx = build_context_features(
        source_feat, feature_cols, context_len, target_start=prefix_len, return_indices=True
    )
    if len(X_chunk_raw) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    X_chunk, _, _ = standardize_features(
        X_chunk_raw, mean=model["feature_mean"], std=model["feature_std"]
    )
    _, _, _, probs = forward_pass(X_chunk, model["W0"], model["b0"], model["W1"], model["b1"])
    if model.get("decision_mode", "thresholds") == "argmax":
        preds = predict_from_probs(probs, decision_mode="argmax")
    else:
        thresholds = model.get("thresholds", (0.75, 0.75))
        if thresholds[0] is None or thresholds[1] is None:
            thresholds = (0.75, 0.75)
        preds = predict_from_probs(
            probs,
            decision_mode="thresholds",
            buy_threshold=thresholds[0],
            sell_threshold=thresholds[1],
        )

    chunk_local_idx = (source_idx - prefix_len).astype(np.int64)
    valid = (chunk_local_idx >= 0) & (chunk_local_idx < len(chunk))
    return preds[valid], chunk_local_idx[valid]


def walk_forward_oracle_ann(
    full_df: pd.DataFrame,
    feature_cols: list[str],
    price_col: str = "adj_close",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    walkforward_step: int = 20,
    oracle_fee_per_trade: float = 2.0,
    label_mode: str = "oracle_dp",
    forward_horizon: int = 1,
    forward_buy_threshold: float = 0.002,
    forward_sell_threshold: float = 0.002,
    decision_mode: str = "thresholds",
    min_action_rate: float = 0.0,
    position_mode: str = "long_short",
    strategy_fee_per_trade: float = 0.0,
    initial_capital: float = 10_000.0,
    context_len: int = 20,
    epochs: int = 150,
    alpha: float = 1e-3,
    hidden: int = 64,
    batch_size: int = 64,
    do_dropout: bool = False,
    dropout_percent: float = 0.1,
    early_stopping_patience: int = 30,
    early_stopping_min_delta: float = 1e-4,
):
    if walkforward_step <= 0:
        raise ValueError("walkforward_step doit etre > 0.")

    data = full_df.sort_values("date").reset_index(drop=True).copy()
    n = len(data)
    if n < context_len + 50:
        raise ValueError("Dataset trop court apres features.")

    _, _, test_df_initial = chronological_train_val_test_split(
        data, train_ratio=train_ratio, val_ratio=val_ratio
    )
    test_start = n - len(test_df_initial)
    if test_start <= context_len:
        raise ValueError("test_start trop court pour le contexte.")

    if label_mode == "oracle_dp":
        eval_labeled_df, eval_label_report = build_oracle_labels_train_only(
            train_df=data,
            price_col=price_col,
            initial_capital=float(initial_capital),
            fee_per_trade=float(oracle_fee_per_trade),
        )
    elif label_mode == "forward_return":
        eval_labeled_df, eval_label_report = build_forward_return_labels(
            data,
            price_col=price_col,
            horizon=forward_horizon,
            buy_threshold=forward_buy_threshold,
            sell_threshold=forward_sell_threshold,
        )
    else:
        raise ValueError(f"label_mode inconnu: {label_mode}")
    y_true_global = eval_labeled_df["Label_id"].to_numpy(dtype=np.int64)

    pred_labels = np.full(n, -1, dtype=np.int64)
    retrain_logs = []

    chunk_id = 0
    for start in range(test_start, n, walkforward_step):
        end = min(start + walkforward_step, n)
        chunk_id += 1

        history = data.iloc[:start].copy()
        if label_mode == "oracle_dp":
            history_labeled, label_hist_report = build_oracle_labels_train_only(
                train_df=history,
                price_col=price_col,
                initial_capital=float(initial_capital),
                fee_per_trade=float(oracle_fee_per_trade),
            )
        else:
            history_labeled, label_hist_report = build_forward_return_labels(
                history,
                price_col=price_col,
                horizon=forward_horizon,
                buy_threshold=forward_buy_threshold,
                sell_threshold=forward_sell_threshold,
            )

        model = train_ann_on_labeled_history(
            labeled_history=history_labeled,
            feature_cols=feature_cols,
            price_col=price_col,
            val_ratio=val_ratio,
            context_len=context_len,
            epochs=epochs,
            alpha=alpha,
            hidden=hidden,
            batch_size=batch_size,
            do_dropout=do_dropout,
            dropout_percent=dropout_percent,
            decision_mode=decision_mode,
            min_action_rate=min_action_rate,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
        )

        chunk = data.iloc[start:end].copy().reset_index(drop=True)
        chunk_preds, chunk_local_idx = predict_chunk_with_model(model, chunk)
        abs_idx = np.arange(start, end, dtype=np.int64)[chunk_local_idx]
        pred_labels[abs_idx] = chunk_preds

        retrain_logs.append(
            {
                "chunk_id": chunk_id,
                "start_idx": int(start),
                "end_idx": int(end),
                "n_hist": int(len(history)),
                "n_pred": int(len(chunk_preds)),
                "best_epoch": int(model["best_epoch"]),
                "val_macro_f1": float(model["best_macro_f1"]),
                "val_bal_acc": float(model["val_bal_acc"]),
                "label_hist_info": label_hist_report,
            }
        )
        print(
            f"[wf {chunk_id:03d}] idx={start}->{end} "
            f"| hist={len(history)} | pred={len(chunk_preds)} "
            f"| best_ep={model['best_epoch']} "
            f"| val_macro_f1={model['best_macro_f1']:.3f}"
        )

    pred_mask = pred_labels != -1
    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_start:] = True
    eval_mask = pred_mask & test_mask

    if not eval_mask.any():
        raise RuntimeError("Aucune prediction disponible sur la periode test.")

    missing_test_preds = int(test_mask.sum() - eval_mask.sum())
    if missing_test_preds > 0:
        print(f"[warn] {missing_test_preds} lignes test sans prediction.")

    y_true = y_true_global[eval_mask]
    y_pred = pred_labels[eval_mask]
    test_metrics = evaluate_predictions(y_true, y_pred)

    aligned_test = data.loc[eval_mask].reset_index(drop=True)
    bench = evaluate_strategy_vs_buy_hold(
        aligned_test,
        y_pred,
        initial_capital=float(initial_capital),
        price_col=price_col,
        fee_per_trade=float(strategy_fee_per_trade),
        position_mode=position_mode,
    )

    return {
        "test_metrics": test_metrics,
        "benchmark_comparison": bench,
        "n_total_rows": int(n),
        "test_start_idx": int(test_start),
        "n_test_rows": int(test_mask.sum()),
        "n_eval_rows": int(eval_mask.sum()),
        "n_missing_test_preds": int(missing_test_preds),
        "label_eval_report": eval_label_report,
        "retrain_logs": retrain_logs,
    }


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parent / "datasets" / "cac40_daily.parquet"


def _build_parser():
    p = argparse.ArgumentParser(
        description=(
            "ANN walk-forward with oracle DP or forward-return labels (no window labelling). "
            "Retrain every N time steps using expanded historical data."
        )
    )
    p.add_argument("--data-dir", type=Path, default=_default_data_dir())
    p.add_argument("--ticker", type=str, default="EN.PA")
    p.add_argument("--price-col", type=str, default="adj_close")
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--context-len", type=int, default=20)
    p.add_argument("--walkforward-step", type=int, default=20)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--alpha", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--do-dropout", action="store_true")
    p.add_argument("--dropout-percent", type=float, default=0.1)
    p.add_argument("--label-mode", type=str, choices=["oracle_dp", "forward_return"], default="forward_return")
    p.add_argument("--forward-horizon", type=int, default=1)
    p.add_argument("--forward-buy-threshold", type=float, default=0.002)
    p.add_argument("--forward-sell-threshold", type=float, default=0.002)
    p.add_argument("--decision-mode", type=str, choices=["thresholds", "argmax"], default="argmax")
    p.add_argument("--min-action-rate", type=float, default=0.02)
    p.add_argument("--position-mode", type=str, choices=["long_short", "long_only"], default="long_only")
    p.add_argument("--oracle-fee-per-trade", type=float, default=2.0)
    p.add_argument("--strategy-fee-per-trade", type=float, default=0.0)
    p.add_argument("--early-stopping-patience", type=int, default=30)
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    return p


def main():
    args = _build_parser().parse_args()

    df = read_parquet_dataset(Path(args.data_dir).expanduser().resolve())
    if "date" not in df.columns:
        raise ValueError("Colonne 'date' manquante.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    if "ticker" in df.columns and args.ticker:
        df = df[df["ticker"] == args.ticker].copy()
    if df.empty:
        raise ValueError("Dataset vide apres filtrage.")

    df = df.sort_values("date").reset_index(drop=True)
    for col in ["open", "high", "low", "close", args.price_col, "volume"]:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante: {col}")

    df_feat = compute_market_features(df)
    feature_cols = list(features)

    for col in [args.price_col] + feature_cols:
        df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")
    df_feat = df_feat.dropna(subset=[args.price_col] + feature_cols).reset_index(drop=True)

    if len(df_feat) < args.context_len + 50:
        raise ValueError("Trop peu de lignes apres feature engineering.")

    print(
        "Walk-forward setup"
        f" | ticker={args.ticker}"
        f" | rows={len(df_feat)}"
        f" | label_mode={args.label_mode}"
        f" | decision_mode={args.decision_mode}"
        f" | position_mode={args.position_mode}"
        f" | context_len={args.context_len}"
        f" | step={args.walkforward_step}"
        f" | epochs={args.epochs}"
    )
    if args.label_mode == "oracle_dp":
        print(
            "[warn] label_mode=oracle_dp utilise une cible clairvoyante "
            "(non apprenable de facon stable hors-echantillon)."
        )

    results = walk_forward_oracle_ann(
        full_df=df_feat,
        feature_cols=feature_cols,
        price_col=args.price_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        walkforward_step=args.walkforward_step,
        oracle_fee_per_trade=args.oracle_fee_per_trade,
        label_mode=args.label_mode,
        forward_horizon=args.forward_horizon,
        forward_buy_threshold=args.forward_buy_threshold,
        forward_sell_threshold=args.forward_sell_threshold,
        decision_mode=args.decision_mode,
        min_action_rate=args.min_action_rate,
        position_mode=args.position_mode,
        strategy_fee_per_trade=args.strategy_fee_per_trade,
        initial_capital=args.capital,
        context_len=args.context_len,
        epochs=args.epochs,
        alpha=args.alpha,
        hidden=args.hidden,
        batch_size=args.batch_size,
        do_dropout=args.do_dropout,
        dropout_percent=args.dropout_percent,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )

    m = results["test_metrics"]
    b = results["benchmark_comparison"]
    label_report = results["label_eval_report"]
    if args.label_mode == "forward_return":
        print(
            "\nLabeling eval"
            f" | mode=forward_return"
            f" | horizon={label_report['horizon']}"
            f" | buy_thr={label_report['buy_threshold']:.4f}"
            f" | sell_thr={label_report['sell_threshold']:.4f}"
            f" | buy/hold/sell={label_report['n_buy']}/{label_report['n_hold']}/{label_report['n_sell']}"
        )
    else:
        print(
            "\nLabeling eval"
            f" | mode=oracle_dp"
            f" | outperf_vs_bh={label_report['outperformance_vs_buy_hold']:.2f}"
            f" | n_trades={label_report['n_trades']}"
        )
    print(
        "\nFinal test \n| "
        f"acc = {m['acc']:.3f} | bal_acc = {m['bal_acc']:.3f} | macro_f1 = {m['macro_f1']:.3f} \n"
        f"| precision_buy = {m['precision_buy']:.3f}  | recall_buy = {m['recall_buy']:.3f} \n"
        f"| precision_sell = {m['precision_sell']:.3f} | recall_sell = {m['recall_sell']:.3f} \n"
        f"| precision_hold = {m['precision_hold']:.3f} | recall_hold = {m['recall_hold']:.3f}"
    )
    print(
        "\nPnL test \n| "
        f"model={b['model_pnl']:.2f} | buy_hold={b['buy_hold_pnl']:.2f} "
        f"| outperformance={b['outperformance']:.2f} | final_model={b['model_final_capital']:.2f} "
        f"| final_buy_hold={b['buy_hold_final_capital']:.2f}"
    )
    print(
        f"\nCoverage | test_rows={results['n_test_rows']} "
        f"| eval_rows={results['n_eval_rows']} "
        f"| missing_preds={results['n_missing_test_preds']}"
    )


if __name__ == "__main__":
    main()
