from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import talib
import matplotlib.pyplot as plt

from trading_system.features.market import compute_market_features, features
from trading_system.paths import default_market_dataset_path

DATA_DIR = default_market_dataset_path()
CAPITAL = 10_000
np.random.seed(1)

def read_parquet_dataset(
    base_dir: Path,
    columns: list[str] | None = None,
    filter_expr: ds.Expression | None = None,
) -> pd.DataFrame:
    """Read a hive-partitioned parquet dataset into a DataFrame.

    Parameters
    ----------
    base_dir : Path
        Dataset root directory.
    columns : list[str] | None
        Optional list of columns to project.
    filter_expr : ds.Expression | None
        Optional Arrow dataset filter expression.

    Returns
    -------
    pd.DataFrame
        Materialized data as a pandas DataFrame.
    """
    dataset = ds.dataset(str(base_dir), format="parquet", partitioning="hive")
    table = dataset.to_table(filter=filter_expr, columns=columns)
    return table.to_pandas()


def compute_returns(df, cols=None):
    """Create log returns for OHLCV columns."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if cols is None:
        cols = ["open", "high", "low", "close", "adj_close", "volume"]

    for col in cols:
        if col not in out.columns:
            continue

        if col == "volume":
            out["volume_ret"] = np.log1p(out["volume"]) - np.log1p(out["volume"].shift(1))
        else:
            safe_prices = out[col].clip(lower=1e-12)
            out[f"{col}_ret"] = np.log(safe_prices / safe_prices.shift(1))

    return out


def normalize_prices(df, cols=None):
    """Add log-normalized price columns."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if cols is None:
        cols = ["open", "high", "low", "close", "adj_close"]

    for col in cols:
        if col not in out.columns:
            continue
        out[f"log_{col}"] = np.log(out[col].clip(lower=1e-12))

    return out


def compute_benchmark(df, capital):
    returns_df = compute_returns(df, cols=["adj_close"])
    if returns_df.empty or "adj_close_ret" not in returns_df.columns:
        return capital
    log_returns = returns_df["adj_close_ret"].dropna()
    return capital * np.exp(log_returns.sum())


def to_train_test(df, split_index):
    split_index = len(df) * split_index
    train = df[:split_index]
    test = df[split_index:]
    return train, test


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


def _split_by_calendar_boundaries(df, val_start, test_start, date_col="date"):
    train_df = df[df[date_col] < val_start].copy()
    val_df = df[(df[date_col] >= val_start) & (df[date_col] < test_start)].copy()
    test_df = df[df[date_col] >= test_start].copy()
    return train_df, val_df, test_df


def _prepare_feature_split_frames(
    labeled_df,
    train_ratio,
    val_ratio,
    feature_cols,
    date_col="date",
):
    """
    Build train/val/test on fixed calendar boundaries from the raw labeled series.

    This keeps the test window stable across feature sets. Features are computed on
    the full series, then val/test NaNs are imputed from train medians so rows are
    not silently dropped from the benchmark period.
    """
    base = labeled_df.sort_values(date_col).reset_index(drop=True).copy()
    raw_train_df, raw_val_df, raw_test_df = chronological_train_val_test_split(
        base,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    if raw_val_df.empty or raw_test_df.empty:
        raise ValueError("Split brut invalide: val/test vides.")

    val_start = raw_val_df[date_col].iloc[0]
    test_start = raw_test_df[date_col].iloc[0]

    feat = compute_market_features(base).sort_values(date_col).reset_index(drop=True).copy()
    feat.loc[:, feature_cols] = feat[feature_cols].apply(pd.to_numeric, errors="coerce")
    feat.loc[:, feature_cols] = feat[feature_cols].replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna(subset=["Label_id"]).copy()

    train_df, val_df, test_df = _split_by_calendar_boundaries(
        feat,
        val_start=val_start,
        test_start=test_start,
        date_col=date_col,
    )
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "Split features invalide apres projection calendrier fixe "
            "(train/val/test vide)."
        )

    # Keep train strictly observed; val/test are imputed from train medians.
    train_df = train_df.dropna(subset=feature_cols + ["Label_id"]).copy()
    if train_df.empty:
        raise ValueError("Aucune ligne train exploitable apres dropna des features.")

    fill_values = train_df[feature_cols].median(numeric_only=True).fillna(0.0)
    val_df.loc[:, feature_cols] = val_df[feature_cols].fillna(fill_values).fillna(0.0)
    test_df.loc[:, feature_cols] = test_df[feature_cols].fillna(fill_values).fillna(0.0)

    train_df = train_df.sort_values(date_col).reset_index(drop=True)
    val_df = val_df.sort_values(date_col).reset_index(drop=True)
    test_df = test_df.sort_values(date_col).reset_index(drop=True)
    raw_test_df = raw_test_df.sort_values(date_col).reset_index(drop=True)
    return train_df, val_df, test_df, raw_test_df


def compute_features(df):
    df = df.sort_values("date").copy()
    df = compute_returns(df, cols=["open", "high", "low", "close", "adj_close", "volume"])
    df = normalize_prices(df, cols=["open", "high", "low", "close", "adj_close"])

    macd, _, _ = talib.MACD(df["log_adj_close"])
    df["rsi"] = talib.RSI(df["log_adj_close"])
    df["macd"] = macd
    df["williams"] = talib.WILLR(df["log_high"], df["log_low"], df["log_close"])
    df["range_log"] = df["log_high"] - df["log_low"]
    df["body_log"] = df["log_close"] - df["log_open"]
    df["upper_wick_log"] = df["log_high"] - np.maximum(df["log_open"], df["log_close"])
    df["lower_wick_log"] = np.minimum(df["log_open"], df["log_close"]) - df["log_low"]
    df["volume_relatif"] = df["volume"] / df["volume"].rolling(10).mean()
    df["volatility_10"] = df["adj_close_ret"].rolling(10).std()
    return df


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(z):
    """
    Convertit des scores (logits) en probabilités qui somment à 1.
    z: (batch, 10)
    Astuce stabilité numérique : soustraire max par ligne avant exp.
    """
    z = z - np.max(z, axis = 1, keepdims = True)
    e = np.exp(z)
    return e / np.sum(e, axis = 1, keepdims = True)


def dropout_mask(shape, p):
    """
    Inverted dropout mask.
    - On garde chaque neurone avec proba (1-p).
    - On divise par (1-p) pour garder la même espérance d'activation.
    shape: (batch, hidden)
    retourne mask de même shape.
    """
    m = (np.random.rand(*shape) > p).astype(np.float32) # 1 si gardé, 0 si drop
    return m / (1-p)


def one_hot(y, k=10):
    """
    y: labels entiers 0..k-1 (shape: (N,))
    retourne une matrice Y (N,k) avec 1 a la classe correcte.
    """
    Y = np.zeros((len(y), k), dtype=np.float32)
    Y[np.arange(len(y)), y] = 1.0
    return Y


def enforce_alternating_signals(labels):
    filtered_labels = []
    last_signal = None

    for label in labels:
        if label == "Hold":
            filtered_labels.append("Hold")
            continue

        if last_signal is None or label != last_signal:
            filtered_labels.append(label)
            last_signal = label
        else:
            filtered_labels.append("Hold")

    return filtered_labels


def add_labels(df, window, price_col="adj_close"):
    df = df.sort_values("date").copy()

    prev_min = df[price_col].shift(1).rolling(window).min()
    prev_max = df[price_col].shift(1).rolling(window).max()

    raw_labels = np.where(
        df[price_col] <= prev_min,
        "Buy",
        np.where(df[price_col] >= prev_max, "Sell", "Hold"),
    )

    raw_labels = pd.Series(raw_labels, index=df.index, dtype="object")
    raw_labels.loc[prev_min.isna()] = "Hold"

    df["Label"] = enforce_alternating_signals(raw_labels.tolist())

    label_map = {"Sell": 0, "Hold": 1, "Buy": 2}
    df["Label_id"] = df["Label"].map(label_map).astype(int)

    return df


def labelling(df, window, price_col="adj_close"):
    labels = add_labels(df, window, price_col=price_col)
    label_stats = {
                "Buy": len(labels[labels["Label"] == "Buy"]),
                "Hold": len(labels[labels["Label"] == "Hold"]),
                "Sell": len(labels[labels["Label"] == "Sell"]),
    }
    return labels, label_stats

            
def labelling_all(df, window, price_col="adj_close"):
    def _one_ticker(group):
        return add_labels(group, window, price_col=price_col)

    if "ticker" in df.columns:
        parts = [_one_ticker(group) for _, group in df.groupby("ticker", sort=False)]
        return pd.concat(parts).sort_index()
    
    return _one_ticker(df)


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
    f1_scores = []
    for label in labels:
        _, _, f1 = precision_recall_f1_for_label(y_true, y_pred, label)
        f1_scores.append(f1)
    return float(np.mean(f1_scores))


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


def predict_with_thresholds(probs, buy_threshold = 0.75, sell_threshold = 0.75):
    # Colonnes de probs : 0=Sell, 1=Hold, 2=Buy
    # On part de Hold partout, puis on n'autorise Buy/Sell
    # que si la proba de la classe depasse un seuil et reste
    # la meilleure classe pour cet echantillon.
    preds = np.full(len(probs), 1, dtype=int)
    best_class = np.argmax(probs, axis=1)

    buy_mask = (best_class == 2) & (probs[:, 2] >= buy_threshold)
    sell_mask = (best_class == 0) & (probs[:, 0] >= sell_threshold)

    preds[buy_mask] = 2
    preds[sell_mask] = 0
    return preds


def threshold_gridsearch(probs, y_val):
    buy_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    sell_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    best_score = -np.inf
    best_thresholds = (0.75, 0.75)

    for bthresh in buy_thresholds:
        for sthresh in sell_thresholds:
            val_preds = predict_with_thresholds(probs, bthresh, sthresh)
            score = evaluate_predictions(y_val, val_preds)["macro_f1"]

            if score > best_score:
                best_score = score
                best_thresholds = (bthresh, sthresh)

    return best_thresholds
        


def compute_confusion_matrix(y_true, y_pred, labels=(0, 1, 2)):
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_index = {label: index for index, label in enumerate(labels)}

    for true_label, pred_label in zip(y_true, y_pred):
        matrix[label_to_index[true_label], label_to_index[pred_label]] += 1

    return matrix


def plot_confusion_matrix(y_true, y_pred, label_names=("Sell", "Hold", "Buy")):
    matrix = compute_confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground truth")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()


def build_context_dataset(df, feature_cols, context_len, target_start=0, return_indices=False):
    if not isinstance(context_len, (int, np.integer)):
        raise TypeError("context_len doit etre un entier.")
    if context_len <= 0:
        raise ValueError("context_len doit etre strictement positif.")

    values = df[feature_cols].to_numpy(dtype=np.float32)
    labels = df["Label_id"].to_numpy(dtype=np.int64)
    feature_dim = len(feature_cols)

    if len(df) < context_len:
        return (
            np.empty((0, context_len * feature_dim), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    X_list, y_list, idx_list = [], [], []
    target_start = max(target_start, context_len - 1)
    for t in range(context_len - 1, len(df)):
        if t < target_start:
            continue
        window = values[t - context_len + 1 : t + 1]
        X_list.append(window.reshape(-1))  # (context_len * F,)
        y_list.append(labels[t])
        idx_list.append(t)

    if not X_list:
        empty_x = np.empty((0, context_len * feature_dim), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.int64)
        empty_idx = np.empty((0,), dtype=np.int64)
        if return_indices:
            return empty_x, empty_y, empty_idx
        return (
            np.empty((0, context_len * feature_dim), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    indices = np.asarray(idx_list, dtype=np.int64)
    if return_indices:
        return X, y, indices
    return X, y


def signals_to_positions(pred_labels):
    """Convert class predictions to a short/flat/long position stream."""
    positions = []
    current_position = 0.0

    for label in pred_labels:
        if label == 2:  # Buy
            current_position = 1.0
        elif label == 0:  # Sell
            current_position = -1.0
        # Hold keeps previous position
        positions.append(current_position)

    return np.asarray(positions, dtype=np.float64)


def pred_labels_to_target_positions(pred_labels):
    """Map ANN class predictions (0/1/2) to persistent target positions (-1/0/1).

    Mapping:
    - 2 (Buy)  -> +1
    - 0 (Sell) -> -1
    - 1 (Hold) -> keep previous position
    """
    positions = []
    current_position = 0

    for label in np.asarray(pred_labels, dtype=np.int64):
        if label == 2:
            current_position = 1
        elif label == 0:
            current_position = -1
        elif label == 1:
            pass
        else:
            raise ValueError(f"Label de prediction inconnu: {label} (attendu: 0,1,2).")
        positions.append(current_position)

    return np.asarray(positions, dtype=np.int8)


def target_positions_to_actions(target_positions):
    """Derive buy/sell/hold actions from target positions transitions."""
    target = pd.Series(np.asarray(target_positions, dtype=np.int64)).clip(-1, 1)
    delta = target.diff().fillna(target)
    actions = np.where(delta > 0, "buy", np.where(delta < 0, "sell", "hold"))
    return np.asarray(actions, dtype=object)


def _coerce_utc_datetime_index(frame, date_col="date"):
    if isinstance(frame.index, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True, errors="coerce"))
    elif date_col in frame.columns:
        idx = pd.DatetimeIndex(pd.to_datetime(frame[date_col], utc=True, errors="coerce"))
    else:
        raise ValueError(
            f"Impossible de construire l'index datetime: index non datetime et colonne '{date_col}' absente."
        )

    if idx.isna().any():
        raise ValueError("Des timestamps invalides ont ete detectes dans le frame de backtest.")
    if idx.has_duplicates:
        raise ValueError("Index datetime du backtest non unique.")

    out = frame.copy()
    out.index = idx
    return out.sort_index()


def prepare_advanced_backtest_inputs(test_frame, pred_labels, date_col="date"):
    """Build market + labels payload compatible with backtest_lib.run_backtest_from_labels."""
    if len(test_frame) != len(pred_labels):
        raise ValueError("Mismatch entre nombre de lignes du frame test et nombre de predictions.")

    market = _coerce_utc_datetime_index(test_frame.copy(), date_col=date_col)

    if "close" not in market.columns and "adj_close" in market.columns:
        market["close"] = market["adj_close"]
    if "close" not in market.columns:
        raise ValueError("Le frame de backtest doit contenir 'close' ou 'adj_close'.")

    close = pd.to_numeric(market["close"], errors="coerce").astype(float)
    if close.isna().any():
        raise ValueError("Colonne close contient des valeurs non numeriques ou NaN.")

    if "open" not in market.columns:
        market["open"] = close
    else:
        market["open"] = pd.to_numeric(market["open"], errors="coerce").astype(float)
    if "high" not in market.columns:
        market["high"] = pd.concat([market["open"], close], axis=1).max(axis=1)
    else:
        market["high"] = pd.to_numeric(market["high"], errors="coerce").astype(float)
    if "low" not in market.columns:
        market["low"] = pd.concat([market["open"], close], axis=1).min(axis=1)
    else:
        market["low"] = pd.to_numeric(market["low"], errors="coerce").astype(float)
    if "volume" not in market.columns:
        market["volume"] = 0.0
    else:
        market["volume"] = pd.to_numeric(market["volume"], errors="coerce").fillna(0.0).astype(float)

    market = market.sort_index()
    target_positions = pred_labels_to_target_positions(pred_labels)
    actions = target_positions_to_actions(target_positions)

    labels_df = pd.DataFrame(
        {
            "target_position": target_positions.astype(np.int8),
            "action": actions,
            "model_label_id": np.asarray(pred_labels, dtype=np.int64),
        },
        index=market.index,
    )

    return market, labels_df


def evaluate_strategy_vs_buy_hold(
    test_frame,
    pred_labels,
    initial_capital=10_000.0,
    price_col="adj_close",
    fee_per_trade=0.0,
):
    """Compute test-period PnL of model signals versus buy-and-hold.

    Notes:
    - `pred_labels[t]` est execute a partir du bar `t+1` (anti look-ahead).
    - `fee_per_trade` est un cout fixe en devise du portefeuille.
      En long/short, un flip (+1 <-> -1) compte pour 2 trades.
    """
    if len(test_frame) != len(pred_labels):
        raise ValueError("Mismatch entre nombre de predictions et lignes test.")
    if len(test_frame) < 2:
        raise ValueError("Le set test doit contenir au moins 2 lignes pour calculer un PnL.")
    if fee_per_trade < 0:
        raise ValueError("fee_per_trade doit etre >= 0.")

    prices = test_frame[price_col].to_numpy(dtype=np.float64)
    forward_returns = np.zeros(len(prices), dtype=np.float64)
    forward_returns[:-1] = (prices[1:] / prices[:-1]) - 1.0

    target_positions = signals_to_positions(pred_labels)
    executed_positions = np.zeros_like(target_positions)
    executed_positions[1:] = target_positions[:-1]  # signal t applique des t+1

    prev_positions = np.zeros_like(executed_positions)
    prev_positions[1:] = executed_positions[:-1]
    turnover = np.abs(executed_positions - prev_positions)  # 0, 1 ou 2

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


def evaluate_buy_hold_only(test_frame, initial_capital=10_000.0, price_col="adj_close"):
    if len(test_frame) < 2:
        raise ValueError("Le set test doit contenir au moins 2 lignes pour calculer un buy&hold.")

    prices = test_frame[price_col].to_numpy(dtype=np.float64)
    forward_returns = np.zeros(len(prices), dtype=np.float64)
    forward_returns[:-1] = (prices[1:] / prices[:-1]) - 1.0
    buy_hold_curve = initial_capital * np.cumprod(1.0 + forward_returns)
    buy_hold_final = float(buy_hold_curve[-1])
    return {
        "buy_hold_final_capital": buy_hold_final,
        "buy_hold_pnl": buy_hold_final - float(initial_capital),
    }


def train_model(
    train,
    epochs=500,
    alpha=1e-3,
    hidden=32,
    do_dropout=False,
    dropout_percent=0.1,
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.15,
    context_len=20,
    early_stopping_patience=50,
    early_stopping_min_delta=1e-4,
):
    if batch_size <= 0:
        raise ValueError("batch_size doit etre strictement positif.")
    if not 0 <= dropout_percent < 1:
        raise ValueError("dropout_percent doit etre dans [0, 1).")
    if not isinstance(context_len, (int, np.integer)):
        raise TypeError("context_len doit etre un entier.")
    if context_len <= 0:
        raise ValueError("context_len doit etre strictement positif.")
    if not isinstance(early_stopping_patience, (int, np.integer)):
        raise TypeError("early_stopping_patience doit etre un entier.")
    if early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience doit etre strictement positif.")
    if early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta doit etre >= 0.")

    feature_cols = list(features)
    train_df, val_df, test_df, raw_test_df = _prepare_feature_split_frames(
        train,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        feature_cols=feature_cols,
    )

    X_train_raw, y_train = build_context_dataset(train_df, feature_cols, context_len)

    # Build validation windows with train history as context
    val_prefix_len = min(context_len - 1, len(train_df))
    val_source = pd.concat([train_df.tail(val_prefix_len), val_df], ignore_index=True)
    X_val_raw, y_val = build_context_dataset(
        val_source,
        feature_cols,
        context_len,
        target_start=val_prefix_len,
    )

    # Build test windows with train+val history as context
    test_history = pd.concat([train_df, val_df], ignore_index=True)
    test_prefix_len = min(context_len - 1, len(test_history))
    test_source = pd.concat([test_history.tail(test_prefix_len), test_df], ignore_index=True)
    X_test_raw, y_test, test_target_indices = build_context_dataset(
        test_source,
        feature_cols,
        context_len,
        target_start=test_prefix_len,
        return_indices=True,
    )

    if len(X_train_raw) == 0:
        raise ValueError("Aucun echantillon train apres fenetrage. Reduis context_len.")
    if len(X_val_raw) == 0:
        raise ValueError("Aucun echantillon validation apres fenetrage. Reduis context_len.")
    if len(X_test_raw) == 0:
        raise ValueError("Aucun echantillon test apres fenetrage. Reduis context_len.")

    Y_train = one_hot(y_train, 3)
    class_weights = compute_class_weights(y_train, num_classes=3)

    X_train, feature_mean, feature_std = standardize_features(X_train_raw)
    X_val, _, _ = standardize_features(
        X_val_raw,
        mean=feature_mean,
        std=feature_std,
    )
    X_test, _, _ = standardize_features(
        X_test_raw,
        mean=feature_mean,
        std=feature_std,
    )

    entry = X_train.shape[1]
    if entry != context_len * len(feature_cols):
        raise ValueError("Entry size miss-match.")
    
    W0 = 0.01 * np.random.randn(entry, hidden).astype(np.float32)
    b0 = np.zeros((1, hidden), dtype=np.float32)
    W1 = 0.01 * np.random.randn(hidden, 3).astype(np.float32)
    b1 = np.zeros((1, 3), dtype=np.float32)

    N = len(X_train)
    best_macro_f1 = -1.0
    best = None

    print("\ntrain rows:", len(train_df), "| val rows:", len(val_df), "| test rows:", len(test_df))
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)
    print("context_len:", context_len, "| feature_dim:", len(feature_cols))
    print("W0:", W0.shape)
    print("b0:", b0.shape)
    print("W1:", W1.shape)
    print("b1:", b1.shape)
    print("class_weights:", class_weights)

    no_improve_count = 0
    best_epoch = 0
    stop_reason = "max_epochs"

    for ep in range(epochs):
        perm = np.random.permutation(N)
        Xp, Yp = X_train[perm], Y_train[perm]
        epoch_loss = 0.0

        for start in range(0, N, batch_size):
            xb = Xp[start:start + batch_size]
            yb = Yp[start:start + batch_size]
            m = len(xb)

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
            batch_loss = -np.sum(yb * np.log(p + 1e-12) * class_weights[None, :]) / weight_sum
            epoch_loss += batch_loss * m

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
        thresholds = threshold_gridsearch(val_probs, y_val)
        val_preds = predict_with_thresholds(val_probs, thresholds[0], thresholds[1])
        val_metrics = evaluate_predictions(y_val, val_preds)

        _, _, _, train_probs = forward_pass(X_train, W0, b0, W1, b1)
        train_preds = predict_with_thresholds(train_probs, thresholds[0], thresholds[1])
        train_metrics = evaluate_predictions(y_train, train_preds)
        avg_loss = epoch_loss / N

        macro_improved = val_metrics["macro_f1"] > (best_macro_f1 + early_stopping_min_delta)
        macro_tie = np.isclose(
            val_metrics["macro_f1"],
            best_macro_f1,
            atol=early_stopping_min_delta,
        )
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
                "feature_cols": feature_cols,
                "feature_mean": feature_mean.copy(),
                "feature_std": feature_std.copy(),
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "thresholds": thresholds,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "val_y_true": y_val.copy(),
                "val_y_pred": val_preds.copy(),
                "val_bal_acc": val_metrics["bal_acc"],
                "best_macro_f1": best_macro_f1,
                "best_epoch": ep + 1,
            }
            best_epoch = ep + 1
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stopping_patience:
            stop_reason = (
                f"early_stopping(patience = {early_stopping_patience}, "
                f"min_delta = {early_stopping_min_delta})"
            )
            print(
                f"\nEarly stopping at epoch {ep + 1}/{epochs} "
                f"(best_epoch = {best_epoch}, best_macro_f1 = {best_macro_f1:.3f}, "
                f"best_bal_acc = {best['val_bal_acc']:.3f})."
            )
            break

        #if ep in {0, 49, 99, 149, 199} or ep == epochs - 1:
        if ep == 0:
            print("\nStarting training ===========================================")
            """print(
                f"\nepoch {ep + 1}/{epochs} \n| loss = {avg_loss:.4f} \n"
                f"| acc(train) = {train_metrics['acc']:.3f}     | acc(val) = {val_metrics['acc']:.3f} \n"
                f"| bal_acc(val) = {val_metrics['bal_acc']:.3f}   | macro_f1(val) = {val_metrics['macro_f1']:.3f} \n"
                f"| precision_buy = {val_metrics['precision_buy']:.3f}  | recall_buy = {val_metrics['recall_buy']:.3f} \n"
                f"| precision_sell = {val_metrics['precision_sell']:.3f} | recall_sell = {val_metrics['recall_sell']:.3f} \n"
                f"| precision_hold = {val_metrics['precision_hold']:.3f} | recall_hold = {val_metrics['recall_hold']:.3f}"
            )"""
        print(
                f"\nepoch {ep + 1}/{epochs} \n| loss = {avg_loss:.4f}          "
                f"| acc(train) = {train_metrics['acc']:.3f}     | acc(val) = {val_metrics['acc']:.3f}       "
                f"| bal_acc(val) = {val_metrics['bal_acc']:.3f}    | macro_f1(val) = {val_metrics['macro_f1']:.3f} \n"
                f"| precision_buy = {val_metrics['precision_buy']:.3f}  | precision_sell = {val_metrics['precision_sell']:.3f} "
                f"| precision_hold = {val_metrics['precision_hold']:.3f} \n| recall_buy = {val_metrics['recall_buy']:.3f}     "
                f"| recall_sell = {val_metrics['recall_sell']:.3f}    | recall_hold = {val_metrics['recall_hold']:.3f}"
            )
        if ep == epochs - 1:
            print("\nEnding training ===========================================")

    if best is None:
        raise RuntimeError("Aucun meilleur modele enregistre pendant l'entrainement.")
    best["stop_reason"] = stop_reason
    print(
        "\nTraining stop \n| "
        f"reason = {best['stop_reason']} | best_epoch = {best['best_epoch']} "
        f"| best_macro_f1(val) = {best['best_macro_f1']:.3f} | best_bal_acc(val) = {best['val_bal_acc']:.3f}"
    )

    _, _, _, test_probs = forward_pass(X_test, best["W0"], best["b0"], best["W1"], best["b1"])
    test_preds = predict_with_thresholds(test_probs, best["thresholds"][0], best["thresholds"][1])
    test_metrics = evaluate_predictions(y_test, test_preds)

    aligned_test_frame = test_source.iloc[test_target_indices].copy()
    benchmark_comparison = evaluate_strategy_vs_buy_hold(
        aligned_test_frame,
        test_preds,
        initial_capital=float(CAPITAL),
        price_col="adj_close",
    )
    fixed_buy_hold = evaluate_buy_hold_only(
        raw_test_df,
        initial_capital=float(CAPITAL),
        price_col="adj_close",
    )
    benchmark_comparison["buy_hold_fixed_final_capital"] = fixed_buy_hold["buy_hold_final_capital"]
    benchmark_comparison["buy_hold_fixed_pnl"] = fixed_buy_hold["buy_hold_pnl"]
    benchmark_comparison["outperformance_vs_fixed_buy_hold"] = (
        benchmark_comparison["model_final_capital"] - fixed_buy_hold["buy_hold_final_capital"]
    )
    advanced_market_df, advanced_labels_df = prepare_advanced_backtest_inputs(
        aligned_test_frame,
        test_preds,
        date_col="date",
    )

    best["test_metrics"] = test_metrics
    best["test_y_true"] = y_test.copy()
    best["test_y_pred"] = test_preds.copy()
    best["benchmark_comparison"] = benchmark_comparison
    best["test_aligned_frame"] = aligned_test_frame.copy()
    best["advanced_backtest_market"] = advanced_market_df
    best["advanced_backtest_labels"] = advanced_labels_df

    print(
        "\nFinal test \n| "
        f"acc = {test_metrics['acc']:.3f} | bal_acc = {test_metrics['bal_acc']:.3f} "
        f"| macro_f1 = {test_metrics['macro_f1']:.3f} \n"
        f"| precision_buy = {test_metrics['precision_buy']:.3f}  | recall_buy = {test_metrics['recall_buy']:.3f} \n"
        f"| precision_sell = {test_metrics['precision_sell']:.3f} | recall_sell = {test_metrics['recall_sell']:.3f} \n"
        f"| precision_hold = {test_metrics['precision_hold']:.3f} | recall_hold = {test_metrics['recall_hold']:.3f} \n"
        f"| buy_threshold = {best['thresholds'][0]:.2f}   | sell_threshold = {best['thresholds'][1]:.2f}"
    )
    print(
        "\nPnL test \n| "
        f"model={benchmark_comparison['model_pnl']:.2f} "
        f"| buy_hold={benchmark_comparison['buy_hold_fixed_pnl']:.2f} "
        f"| outperformance={benchmark_comparison['outperformance_vs_fixed_buy_hold']:.2f} "
        f"| final_model={benchmark_comparison['model_final_capital']:.2f} "
        f"| final_buy_hold={benchmark_comparison['buy_hold_fixed_final_capital']:.2f}\n"
    )

    plot_confusion_matrix(best["test_y_true"], best["test_y_pred"])

    return best


def train_one_trial(
    train,
    epochs=500,
    alpha=1e-3,
    hidden=32,
    do_dropout=False,
    dropout_percent=0.1,
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.15,
    context_len=20,
    early_stopping_patience=50,
    early_stopping_min_delta=1e-4,
):
    if batch_size <= 0:
        raise ValueError("batch_size doit etre strictement positif.")
    if not 0 <= dropout_percent < 1:
        raise ValueError("dropout_percent doit etre dans [0, 1).")
    if not isinstance(context_len, (int, np.integer)):
        raise TypeError("context_len doit etre un entier.")
    if context_len <= 0:
        raise ValueError("context_len doit etre strictement positif.")
    if not isinstance(early_stopping_patience, (int, np.integer)):
        raise TypeError("early_stopping_patience doit etre un entier.")
    if early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience doit etre strictement positif.")
    if early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta doit etre >= 0.")

    feature_cols = list(features)
    train_df, val_df, test_df, raw_test_df = _prepare_feature_split_frames(
        train,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        feature_cols=feature_cols,
    )

    X_train_raw, y_train = build_context_dataset(train_df, feature_cols, context_len)

    # Build validation windows with train history as context
    val_prefix_len = min(context_len - 1, len(train_df))
    val_source = pd.concat([train_df.tail(val_prefix_len), val_df], ignore_index=True)
    X_val_raw, y_val = build_context_dataset(
        val_source,
        feature_cols,
        context_len,
        target_start=val_prefix_len,
    )

    # Build test windows with train+val history as context
    test_history = pd.concat([train_df, val_df], ignore_index=True)
    test_prefix_len = min(context_len - 1, len(test_history))
    test_source = pd.concat([test_history.tail(test_prefix_len), test_df], ignore_index=True)
    X_test_raw, y_test, test_target_indices = build_context_dataset(
        test_source,
        feature_cols,
        context_len,
        target_start=test_prefix_len,
        return_indices=True,
    )

    if len(X_train_raw) == 0:
        raise ValueError("Aucun echantillon train apres fenetrage. Reduis context_len.")
    if len(X_val_raw) == 0:
        raise ValueError("Aucun echantillon validation apres fenetrage. Reduis context_len.")
    if len(X_test_raw) == 0:
        raise ValueError("Aucun echantillon test apres fenetrage. Reduis context_len.")

    Y_train = one_hot(y_train, 3)
    class_weights = compute_class_weights(y_train, num_classes=3)

    X_train, feature_mean, feature_std = standardize_features(X_train_raw)
    X_val, _, _ = standardize_features(
        X_val_raw,
        mean=feature_mean,
        std=feature_std,
    )
    X_test, _, _ = standardize_features(
        X_test_raw,
        mean=feature_mean,
        std=feature_std,
    )

    entry = X_train.shape[1]
    if entry != context_len * len(feature_cols):
        raise ValueError("Entry size miss-match.")
    
    W0 = 0.01 * np.random.randn(entry, hidden).astype(np.float32)
    b0 = np.zeros((1, hidden), dtype=np.float32)
    W1 = 0.01 * np.random.randn(hidden, 3).astype(np.float32)
    b1 = np.zeros((1, 3), dtype=np.float32)

    N = len(X_train)
    best_macro_f1 = -1.0
    best = None

    print("\ntrain rows:", len(train_df), "| val rows:", len(val_df), "| test rows:", len(test_df))
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)
    print("context_len:", context_len, "| feature_dim:", len(feature_cols))
    print("W0:", W0.shape)
    print("b0:", b0.shape)
    print("W1:", W1.shape)
    print("b1:", b1.shape)
    print("class_weights:", class_weights)

    no_improve_count = 0
    best_epoch = 0
    stop_reason = "max_epochs"

    for ep in range(epochs):
        perm = np.random.permutation(N)
        Xp, Yp = X_train[perm], Y_train[perm]
        epoch_loss = 0.0

        for start in range(0, N, batch_size):
            xb = Xp[start:start + batch_size]
            yb = Yp[start:start + batch_size]
            m = len(xb)

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
            batch_loss = -np.sum(yb * np.log(p + 1e-12) * class_weights[None, :]) / weight_sum
            epoch_loss += batch_loss * m

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
        thresholds = threshold_gridsearch(val_probs, y_val)
        val_preds = predict_with_thresholds(val_probs, thresholds[0], thresholds[1])
        val_metrics = evaluate_predictions(y_val, val_preds)

        _, _, _, train_probs = forward_pass(X_train, W0, b0, W1, b1)
        train_preds = predict_with_thresholds(train_probs, thresholds[0], thresholds[1])
        train_metrics = evaluate_predictions(y_train, train_preds)
        avg_loss = epoch_loss / N

        macro_improved = val_metrics["macro_f1"] > (best_macro_f1 + early_stopping_min_delta)
        macro_tie = np.isclose(
            val_metrics["macro_f1"],
            best_macro_f1,
            atol=early_stopping_min_delta,
        )
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
                "feature_cols": feature_cols,
                "feature_mean": feature_mean.copy(),
                "feature_std": feature_std.copy(),
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "thresholds": thresholds,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "val_y_true": y_val.copy(),
                "val_y_pred": val_preds.copy(),
                "val_bal_acc": val_metrics["bal_acc"],
                "best_macro_f1": best_macro_f1,
                "best_epoch": ep + 1,
            }
            best_epoch = ep + 1
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stopping_patience:
            stop_reason = (
                f"early_stopping(patience = {early_stopping_patience}, "
                f"min_delta = {early_stopping_min_delta})"
            )
            print(
                f"\nEarly stopping at epoch {ep + 1}/{epochs} "
                f"(best_epoch = {best_epoch}, best_macro_f1 = {best_macro_f1:.3f}, "
                f"best_bal_acc = {best['val_bal_acc']:.3f})."
            )
            break

        #if ep in {0, 49, 99, 149, 199} or ep == epochs - 1:
        if ep == 0:
            print("\nStarting training ===========================================")
            """print(
                f"\nepoch {ep + 1}/{epochs} \n| loss = {avg_loss:.4f} \n"
                f"| acc(train) = {train_metrics['acc']:.3f}     | acc(val) = {val_metrics['acc']:.3f} \n"
                f"| bal_acc(val) = {val_metrics['bal_acc']:.3f}   | macro_f1(val) = {val_metrics['macro_f1']:.3f} \n"
                f"| precision_buy = {val_metrics['precision_buy']:.3f}  | recall_buy = {val_metrics['recall_buy']:.3f} \n"
                f"| precision_sell = {val_metrics['precision_sell']:.3f} | recall_sell = {val_metrics['recall_sell']:.3f} \n"
                f"| precision_hold = {val_metrics['precision_hold']:.3f} | recall_hold = {val_metrics['recall_hold']:.3f}"
            )"""
        print(
                f"\nepoch {ep + 1}/{epochs} \n| loss = {avg_loss:.4f}          "
                f"| acc(train) = {train_metrics['acc']:.3f}     | acc(val) = {val_metrics['acc']:.3f}       "
                f"| bal_acc(val) = {val_metrics['bal_acc']:.3f}    | macro_f1(val) = {val_metrics['macro_f1']:.3f} \n"
                f"| precision_buy = {val_metrics['precision_buy']:.3f}  | precision_sell = {val_metrics['precision_sell']:.3f} "
                f"| precision_hold = {val_metrics['precision_hold']:.3f} \n| recall_buy = {val_metrics['recall_buy']:.3f}     "
                f"| recall_sell = {val_metrics['recall_sell']:.3f}    | recall_hold = {val_metrics['recall_hold']:.3f}"
            )
        if ep == epochs - 1:
            print("\nEnding training ===========================================")

    if best is None:
        raise RuntimeError("Aucun meilleur modele enregistre pendant l'entrainement.")
    best["stop_reason"] = stop_reason
    print(
        "\nTraining stop \n| "
        f"reason = {best['stop_reason']} | best_epoch = {best['best_epoch']} "
        f"| best_macro_f1(val) = {best['best_macro_f1']:.3f} | best_bal_acc(val) = {best['val_bal_acc']:.3f}"
    )

    _, _, _, test_probs = forward_pass(X_test, best["W0"], best["b0"], best["W1"], best["b1"])
    test_preds = predict_with_thresholds(test_probs, best["thresholds"][0], best["thresholds"][1])
    test_metrics = evaluate_predictions(y_test, test_preds)

    aligned_test_frame = test_source.iloc[test_target_indices].copy()
    benchmark_comparison = evaluate_strategy_vs_buy_hold(
        aligned_test_frame,
        test_preds,
        initial_capital=float(CAPITAL),
        price_col="adj_close",
    )
    fixed_buy_hold = evaluate_buy_hold_only(
        raw_test_df,
        initial_capital=float(CAPITAL),
        price_col="adj_close",
    )
    benchmark_comparison["buy_hold_fixed_final_capital"] = fixed_buy_hold["buy_hold_final_capital"]
    benchmark_comparison["buy_hold_fixed_pnl"] = fixed_buy_hold["buy_hold_pnl"]
    benchmark_comparison["outperformance_vs_fixed_buy_hold"] = (
        benchmark_comparison["model_final_capital"] - fixed_buy_hold["buy_hold_final_capital"]
    )
    advanced_market_df, advanced_labels_df = prepare_advanced_backtest_inputs(
        aligned_test_frame,
        test_preds,
        date_col="date",
    )

    best["test_metrics"] = test_metrics
    best["test_y_true"] = y_test.copy()
    best["test_y_pred"] = test_preds.copy()
    best["benchmark_comparison"] = benchmark_comparison
    best["test_aligned_frame"] = aligned_test_frame.copy()
    best["advanced_backtest_market"] = advanced_market_df
    best["advanced_backtest_labels"] = advanced_labels_df

    print(
        "\nFinal test \n| "
        f"acc = {test_metrics['acc']:.3f} | bal_acc = {test_metrics['bal_acc']:.3f} "
        f"| macro_f1 = {test_metrics['macro_f1']:.3f} \n"
        f"| precision_buy = {test_metrics['precision_buy']:.3f}  | recall_buy = {test_metrics['recall_buy']:.3f} \n"
        f"| precision_sell = {test_metrics['precision_sell']:.3f} | recall_sell = {test_metrics['recall_sell']:.3f} \n"
        f"| precision_hold = {test_metrics['precision_hold']:.3f} | recall_hold = {test_metrics['recall_hold']:.3f} \n"
        f"| buy_threshold = {best['thresholds'][0]:.2f}   | sell_threshold = {best['thresholds'][1]:.2f}"
    )
    print(
        "\nPnL test \n| "
        f"model={benchmark_comparison['model_pnl']:.2f} "
        f"| buy_hold={benchmark_comparison['buy_hold_fixed_pnl']:.2f} "
        f"| outperformance={benchmark_comparison['outperformance_vs_fixed_buy_hold']:.2f} "
        f"| final_model={benchmark_comparison['model_final_capital']:.2f} "
        f"| final_buy_hold={benchmark_comparison['buy_hold_fixed_final_capital']:.2f}\n"
    )

    #plot_confusion_matrix(best["test_y_true"], best["test_y_pred"])

    return best, test_metrics, benchmark_comparison



def plot_signals(df, window=160, price_col="adj_close"):
    plot_df = df.sort_values("date").tail(window).copy()

    buy_points = plot_df[plot_df["Label"] == "Buy"]
    sell_points = plot_df[plot_df["Label"] == "Sell"]

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df["date"], plot_df[price_col], label="Prix", color="steelblue", linewidth=1.8)
    plt.scatter(
        buy_points["date"],
        buy_points[price_col],
        label="Buy",
        color="green",
        marker="^",
        s=90,
        zorder=3,
    )
    plt.scatter(
        sell_points["date"],
        sell_points[price_col],
        label="Sell",
        color="red",
        marker="v",
        s=90,
        zorder=3,
    )

    plt.title(f"Signaux Buy/Sell sur les {len(plot_df)} derniers ticks")
    plt.xlabel("Date")
    plt.ylabel("Prix")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


from itertools import product

def model_grid_search(train_df, fee):
    grid = {
        "alpha": [1e-3, 1e-4],
        "hidden": [16, 32, 64],
        "do_dropout": [False, True],
        "dropout_percent": [0.1],
        "batch_size": [32, 64],
        "context_len": [10, 20],
        "window": [10, 20, 30],
    }

    keys = list(grid.keys())
    combos = list(product(*(grid[k] for k in keys)))

    best_score = -np.inf
    best_params = None
    best_metrics = None
    rows = []

    for vals in combos:
        hp = dict(zip(keys, vals))

        # labels avec la window du trial
        df_labeled, label_stats = labelling(train_df.copy(), hp["window"])

        model, metrics, benchmark = train_one_trial(
            df_labeled,
            alpha=hp["alpha"],
            hidden=hp["hidden"],
            do_dropout=hp["do_dropout"],
            dropout_percent=hp["dropout_percent"],
            batch_size=hp["batch_size"],
            context_len=hp["context_len"],
        )

        # score scalaire (important)
        score = float(benchmark["outperformance"])

        row = {
            **hp,
            "score": score,
            "macro_f1": float(metrics["macro_f1"]),
            "model_pnl": float(benchmark["model_pnl"]),
            "buy_hold_pnl": float(benchmark["buy_hold_pnl"]),
            "outperformance": float(benchmark["outperformance"]),
            "label_buy": int(label_stats["Buy"]),
            "label_hold": int(label_stats["Hold"]),
            "label_sell": int(label_stats["Sell"]),
        }
        rows.append(row)

        if score > best_score:
            best_score = score
            best_params = hp.copy()
            best_metrics = row.copy()

    results_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return best_params, best_metrics, results_df


def main():
    df = read_parquet_dataset(DATA_DIR)
    df = df[df["ticker"] == "EN.PA"].copy()
    df, label_stats = labelling(df, 11)
    print(f"\nLabel stats :{label_stats}")
    model = train_model(df)

    #df = read_parquet_dataset(DATA_DIR)
    #df = df[df["ticker"] == "EN.PA"].copy()
    #best_params, best_metrics, results_df = model_grid_search(df, fee=2.0)
    #print(best_params)
    #print(best_metrics)
    #print(results_df.head(20))


if __name__ == "__main__":
    main()
