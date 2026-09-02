# V3 notebook library extracted and consolidated from research/backtests.ipynb

from __future__ import annotations
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any
import numpy as np
import pandas as pd
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    px = None

    def make_subplots(*args, **kwargs):
        raise ImportError("Plotly is required to build backtest visualizations.")
from trading_system.paths import runs_dir
try:
    import ccxt
except ImportError:
    ccxt = None
try:
    from IPython.display import display
except ImportError:

    def display(*args, **kwargs):
        for arg in args:
            print(arg)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)
np.set_printoptions(suppress=True, precision=6)

@dataclass(frozen=True)
class BacktestConfig:
    schema_version: str = 'v3.0.0'
    exchange_id: str = 'binance'
    symbol: str = 'LTC/USDT'
    timeframe: str = '1h'
    start: str = '2024-01-01T00:00:00Z'
    end: str | None = None
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    ema_fast: int = 20
    ema_slow: int = 50
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    fees_bps: float = 5.0
    slippage_bps: float = 0.0
    initial_capital: float = 10000.0
    timezone: str = 'UTC'
    risk_free_rate: float = 0.0
    annualization_factor: int = 24 * 365
    allow_multiple_positions: bool = False
    run_train_only: bool = True
    artifact_root: str = str(runs_dir())
    notes: str = 'Notebook-first V3 backtest baseline'
    benchmark_enabled: bool = True
    walkforward_windows: tuple[int, int, int] = (600, 200, 100)
    stress_fees_grid: tuple[float, ...] = (0.0, 2.5, 5.0, 10.0)
    stress_slippage_grid: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0)
    hmm_states_grid: tuple[int, ...] = (2, 3, 4, 5)
    scoring_seeds: tuple[int, ...] = (7, 42, 1337)
    bootstrap_iterations: int = 500
    bootstrap_seed: int = 42
    sensitivity_max_combinations: int = 120
    feature_drift_max_features: int = 80
    anomaly_enabled: bool = True
    anomaly_model: str = 'isolation_forest'
    anomaly_contamination: float = 0.03
    anomaly_threshold_quantile: float = 0.97
    anomaly_min_train_rows: int = 180
    anomaly_pre_window_h: int = 6
    anomaly_post_window_h: int = 2
    anomaly_random_state: int = 42
    meta_min_train_trades: int = 30
    recommendations_max_items: int = 12
    purged_cv_folds: int = 5
    purged_cv_embargo_bars: int = 24
    purged_cv_max_variants: int = 40
    pbo_trials: int = 200
    execution_mode: str = 'market'
    execution_latency_bars: int = 1
    execution_limit_offset_bps: float = 2.0
    execution_partial_fill_base: float = 0.9
    execution_dynamic_slippage_k: float = 1.0
    execution_volume_cap_ratio: float = 0.03
    trade_replay_max_trades: int = 300
    portfolio_symbols: tuple[str, ...] = ()
    portfolio_weighting: str = 'equal'
    portfolio_rebalance_freq: str = '1D'

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')

def resolve_config(config: BacktestConfig) -> BacktestConfig:
    end_value = config.end or utc_now_iso()
    return replace(config, end=end_value)

def validate_config(config: BacktestConfig) -> None:
    train_ratio, val_ratio, test_ratio = config.split_ratios
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-09):
        raise ValueError(f'split_ratios must sum to 1.0, got {ratio_sum}')
    if any((r <= 0 for r in config.split_ratios)):
        raise ValueError('split_ratios must all be > 0')
    if config.ema_fast <= 0 or config.ema_slow <= 0:
        raise ValueError('ema_fast and ema_slow must be > 0')
    if config.ema_fast >= config.ema_slow:
        raise ValueError('ema_fast must be strictly lower than ema_slow')
    if config.stop_loss_pct < 0 or config.take_profit_pct < 0:
        raise ValueError('stop_loss_pct and take_profit_pct must be >= 0')
    if config.fees_bps < 0 or config.slippage_bps < 0:
        raise ValueError('fees_bps and slippage_bps must be >= 0')
    if config.initial_capital <= 0:
        raise ValueError('initial_capital must be > 0')
    if config.annualization_factor <= 0:
        raise ValueError('annualization_factor must be > 0')
    if config.timezone != 'UTC':
        raise ValueError("V2 enforces timezone='UTC' for decomposition consistency")
    train_size, test_size, step = [int(x) for x in config.walkforward_windows]
    if min(train_size, test_size, step) <= 0:
        raise ValueError('walkforward_windows must be (train_size, test_size, step), all > 0')
    if len(config.stress_fees_grid) == 0 or any((x < 0 for x in config.stress_fees_grid)):
        raise ValueError('stress_fees_grid must be non-empty and >= 0')
    if len(config.stress_slippage_grid) == 0 or any((x < 0 for x in config.stress_slippage_grid)):
        raise ValueError('stress_slippage_grid must be non-empty and >= 0')
    if len(config.hmm_states_grid) == 0 or any((int(x) < 2 for x in config.hmm_states_grid)):
        raise ValueError('hmm_states_grid must be non-empty with values >= 2')
    if len(config.scoring_seeds) == 0:
        raise ValueError('scoring_seeds must be non-empty')
    if int(config.bootstrap_iterations) <= 0:
        raise ValueError('bootstrap_iterations must be > 0')
    if int(config.sensitivity_max_combinations) <= 0:
        raise ValueError('sensitivity_max_combinations must be > 0')
    if int(config.feature_drift_max_features) <= 0:
        raise ValueError('feature_drift_max_features must be > 0')
    if str(config.anomaly_model) not in {'isolation_forest', 'robust_zscore'}:
        raise ValueError("anomaly_model must be one of {'isolation_forest','robust_zscore'}")
    if not (0.0 < float(config.anomaly_contamination) < 0.50):
        raise ValueError('anomaly_contamination must be in (0, 0.5)')
    if not (0.50 <= float(config.anomaly_threshold_quantile) < 1.0):
        raise ValueError('anomaly_threshold_quantile must be in [0.5, 1.0)')
    if int(config.anomaly_min_train_rows) <= 0:
        raise ValueError('anomaly_min_train_rows must be > 0')
    if int(config.anomaly_pre_window_h) < 0 or int(config.anomaly_post_window_h) < 0:
        raise ValueError('anomaly_pre_window_h and anomaly_post_window_h must be >= 0')
    if int(config.meta_min_train_trades) <= 0:
        raise ValueError('meta_min_train_trades must be > 0')
    if int(config.recommendations_max_items) <= 0:
        raise ValueError('recommendations_max_items must be > 0')
    if int(config.purged_cv_folds) < 3:
        raise ValueError('purged_cv_folds must be >= 3')
    if int(config.purged_cv_embargo_bars) < 0:
        raise ValueError('purged_cv_embargo_bars must be >= 0')
    if int(config.purged_cv_max_variants) <= 0:
        raise ValueError('purged_cv_max_variants must be > 0')
    if int(config.pbo_trials) <= 0:
        raise ValueError('pbo_trials must be > 0')
    if str(config.execution_mode) not in {'market', 'limit'}:
        raise ValueError("execution_mode must be one of {'market','limit'}")
    if int(config.execution_latency_bars) < 0:
        raise ValueError('execution_latency_bars must be >= 0')
    if float(config.execution_limit_offset_bps) < 0:
        raise ValueError('execution_limit_offset_bps must be >= 0')
    if not (0 < float(config.execution_partial_fill_base) <= 1.0):
        raise ValueError('execution_partial_fill_base must be in (0,1]')
    if float(config.execution_dynamic_slippage_k) < 0:
        raise ValueError('execution_dynamic_slippage_k must be >= 0')
    if float(config.execution_volume_cap_ratio) <= 0:
        raise ValueError('execution_volume_cap_ratio must be > 0')
    if int(config.trade_replay_max_trades) <= 0:
        raise ValueError('trade_replay_max_trades must be > 0')
    if str(config.portfolio_weighting) not in {'equal'}:
        raise ValueError("portfolio_weighting must be 'equal' in v3")

def stable_json_dumps(obj: Any) -> str:

    def _default(x: Any) -> Any:
        if isinstance(x, (datetime, pd.Timestamp)):
            return x.isoformat()
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(f'Object not JSON serializable: {type(x)}')
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, default=_default, separators=(',', ':'))

def hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def hash_object(obj: Any) -> str:
    return hash_bytes(stable_json_dumps(obj).encode('utf-8'))

def hash_dataframe(df: pd.DataFrame) -> str:
    if df is None:
        return ''
    h = hashlib.sha256()
    h.update('|'.join([f'{c}:{df[c].dtype}' for c in df.columns]).encode('utf-8'))
    if len(df.index) > 0:
        idx_hash = pd.util.hash_pandas_object(df.index.to_series(), index=True).to_numpy(dtype=np.uint64)
        h.update(idx_hash.tobytes())
    if len(df) > 0:
        val_hash = pd.util.hash_pandas_object(df, index=True).to_numpy(dtype=np.uint64)
        h.update(val_hash.tobytes())
    return h.hexdigest()

def timeframe_to_millis(timeframe: str) -> int:
    units = {'m': 60000, 'h': 3600000, 'd': 86400000}
    if len(timeframe) < 2:
        raise ValueError(f'Unsupported timeframe: {timeframe}')
    qty = int(timeframe[:-1])
    unit = timeframe[-1]
    if unit not in units:
        raise ValueError(f'Unsupported timeframe unit: {unit}')
    return qty * units[unit]

def config_to_dict(config: BacktestConfig) -> dict[str, Any]:
    return asdict(config)

def make_run_id(config: BacktestConfig) -> str:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    symbol_tag = config.symbol.replace('/', '-').lower()
    return f'{stamp}_{symbol_tag}_{config.timeframe}'

def download_market_data(config: BacktestConfig) -> pd.DataFrame:
    """Download OHLCV from CCXT exchange and return normalized UTC DataFrame."""
    validate_config(config)
    config = resolve_config(config)
    if ccxt is None:
        raise ImportError('ccxt is not installed. Install it in your notebook env (pip install ccxt).')
    exchange_cls = getattr(ccxt, config.exchange_id, None)
    if exchange_cls is None:
        raise ValueError(f'Unknown exchange_id: {config.exchange_id}')
    ex = exchange_cls({'enableRateLimit': True})
    since_ms = ex.parse8601(config.start)
    end_ms = ex.parse8601(config.end)
    if since_ms is None or end_ms is None:
        raise ValueError('Invalid start/end timestamps')
    if since_ms >= end_ms:
        raise ValueError('start must be < end')
    all_rows: list[list[float]] = []
    tf_ms = timeframe_to_millis(config.timeframe)
    while since_ms < end_ms:
        batch = ex.fetch_ohlcv(config.symbol, timeframe=config.timeframe, since=since_ms, limit=1000)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = int(batch[-1][0])
        if last_ts >= end_ms:
            break
        next_since = last_ts + tf_ms
        if next_since <= since_ms:
            next_since = last_ts + 1
        since_ms = next_since
        if len(batch) < 1000 and since_ms < end_ms:
            since_ms += tf_ms
    if not all_rows:
        raise ValueError('No OHLCV rows downloaded')
    df = pd.DataFrame(all_rows, columns=['timestamp_ms', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
    df = df.drop(columns=['timestamp_ms'])
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp').sort_index()
    end_ts = pd.Timestamp(config.end, tz='UTC')
    df = df[df.index <= end_ts]
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype('float64')
    df['symbol'] = config.symbol
    df['timeframe'] = config.timeframe
    return df

def assert_market_data_integrity(df: pd.DataFrame) -> None:
    if df.empty:
        raise AssertionError('Market data is empty')
    if not isinstance(df.index, pd.DatetimeIndex):
        raise AssertionError('Market data index must be DatetimeIndex')
    if df.index.tz is None or str(df.index.tz) != 'UTC':
        raise AssertionError('Market data index must be UTC tz-aware')
    if not df.index.is_monotonic_increasing:
        raise AssertionError('Market data index must be sorted ascending')
    if not df.index.is_unique:
        raise AssertionError('Market data index must be unique')
    required = {'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(df.columns):
        raise AssertionError(f'Missing OHLCV columns: {sorted(required - set(df.columns))}')

def split_time_series(df: pd.DataFrame, ratios: tuple[float, float, float]=(0.7, 0.15, 0.15)) -> dict[str, pd.DataFrame]:
    """Chronological contiguous split (train/val/test) without overlap."""
    if df is None or df.empty:
        raise ValueError('Input DataFrame is empty')
    if len(df) < 30:
        raise ValueError('Need at least 30 rows for robust split')
    train_ratio, val_ratio, test_ratio = ratios
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, abs_tol=1e-09):
        raise ValueError(f'Ratios must sum to 1, got {ratio_sum}')
    n = len(df)
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    test_n = n - train_n - val_n
    if min(train_n, val_n, test_n) <= 0:
        raise ValueError(f'Invalid split sizes: train={train_n}, val={val_n}, test={test_n}')
    train = df.iloc[:train_n].copy()
    val = df.iloc[train_n:train_n + val_n].copy()
    test = df.iloc[train_n + val_n:].copy()
    if len(set(train.index) & set(val.index)) > 0:
        raise AssertionError('Train/Val overlap detected')
    if len(set(train.index) & set(test.index)) > 0:
        raise AssertionError('Train/Test overlap detected')
    if len(set(val.index) & set(test.index)) > 0:
        raise AssertionError('Val/Test overlap detected')
    if not train.index.max() < val.index.min() < test.index.min():
        raise AssertionError('Chronological ordering broken across splits')
    return {'train': train, 'val': val, 'test': test}

def summarize_splits(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in splits.items():
        rows.append({'split': name, 'rows': len(frame), 'start': frame.index.min(), 'end': frame.index.max()})
    return pd.DataFrame(rows)

def generate_strategy_labels(train_df: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Generate EMA crossover labels and target positions without look-ahead."""
    if 'close' not in train_df.columns:
        raise ValueError("train_df must include 'close' column")
    close = train_df['close'].astype(float)
    ema_fast = close.ewm(span=config.ema_fast, adjust=False).mean()
    ema_slow = close.ewm(span=config.ema_slow, adjust=False).mean()
    raw = np.sign(ema_fast - ema_slow)
    target_position = pd.Series(raw, index=train_df.index).replace(0, np.nan).ffill().fillna(0).astype(int).clip(-1, 1)
    delta = target_position.diff().fillna(target_position)
    action = np.where(delta > 0, 'buy', np.where(delta < 0, 'sell', 'hold'))
    labels = pd.DataFrame({'close': close, 'ema_fast': ema_fast, 'ema_slow': ema_slow, 'signal_raw': raw, 'target_position': target_position, 'action': action}, index=train_df.index)
    return labels

def labels_to_signals(labels_df: pd.DataFrame) -> pd.DataFrame:
    return labels_df[['action', 'target_position']].copy()

def _coerce_utc_datetime_index(df: pd.DataFrame, timestamp_col: str | None=None) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        idx = pd.to_datetime(out.index, utc=True, errors='coerce')
    else:
        ts_col = timestamp_col
        if ts_col is None:
            for candidate in ['timestamp', 'date', 'datetime', 'time']:
                if candidate in out.columns:
                    ts_col = candidate
                    break
        if ts_col is None or ts_col not in out.columns:
            raise ValueError("Cannot infer timestamps: provide a DatetimeIndex or a timestamp column.")
        idx = pd.to_datetime(out[ts_col], utc=True, errors='coerce')
        out = out.drop(columns=[ts_col])
    if idx.isna().any():
        raise ValueError('Invalid timestamps detected in external data.')
    idx = pd.DatetimeIndex(idx)
    if idx.has_duplicates:
        raise ValueError('External data index must be unique.')
    out.index = idx
    return out.sort_index()

def prepare_external_market_data(market_df: pd.DataFrame, config: BacktestConfig, timestamp_col: str | None=None) -> pd.DataFrame:
    """Normalize external OHLCV data for the event-driven backtester."""
    out = _coerce_utc_datetime_index(market_df, timestamp_col=timestamp_col)
    if 'close' not in out.columns and 'adj_close' in out.columns:
        out['close'] = out['adj_close']
    if 'close' not in out.columns:
        raise ValueError("market_df must include 'close' or 'adj_close'.")
    out['close'] = pd.to_numeric(out['close'], errors='coerce').astype(float)
    if out['close'].isna().any():
        raise ValueError("Column 'close' contains NaN or non-numeric values.")
    if 'open' in out.columns:
        out['open'] = pd.to_numeric(out['open'], errors='coerce').astype(float)
    else:
        out['open'] = out['close']
    if 'high' in out.columns:
        out['high'] = pd.to_numeric(out['high'], errors='coerce').astype(float)
    else:
        out['high'] = pd.concat([out['open'], out['close']], axis=1).max(axis=1)
    if 'low' in out.columns:
        out['low'] = pd.to_numeric(out['low'], errors='coerce').astype(float)
    else:
        out['low'] = pd.concat([out['open'], out['close']], axis=1).min(axis=1)
    if 'volume' in out.columns:
        out['volume'] = pd.to_numeric(out['volume'], errors='coerce').fillna(0.0).astype(float)
    else:
        out['volume'] = 0.0
    if out[['open', 'high', 'low', 'close']].isna().any().any():
        raise ValueError('OHLC columns contain NaN after normalization.')
    if 'symbol' not in out.columns:
        out['symbol'] = config.symbol
    if 'timeframe' not in out.columns:
        out['timeframe'] = config.timeframe
    return out.sort_index()

def class_predictions_to_target_positions(pred_labels: list[int] | np.ndarray | pd.Series) -> np.ndarray:
    """Map ANN labels (0=sell,1=hold,2=buy) to persistent target positions (-1/0/1)."""
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
            raise ValueError(f'Unknown class prediction: {label} (expected 0/1/2).')
        positions.append(current_position)
    return np.asarray(positions, dtype=np.int8)

def _actions_from_target_positions(target_position: pd.Series | np.ndarray) -> np.ndarray:
    target = pd.Series(np.asarray(target_position, dtype=np.int64)).clip(-1, 1)
    delta = target.diff().fillna(target)
    return np.where(delta > 0, 'buy', np.where(delta < 0, 'sell', 'hold'))

def prepare_external_labels(
    market_index: pd.DatetimeIndex,
    labels_df: pd.DataFrame | None=None,
    pred_labels: list[int] | np.ndarray | pd.Series | None=None,
    timestamp_col: str | None=None,
) -> pd.DataFrame:
    """Build/normalize labels for run_backtest_from_labels from external inputs."""
    if labels_df is None and pred_labels is None:
        raise ValueError('Provide either labels_df or pred_labels.')
    if labels_df is None:
        pred_arr = np.asarray(pred_labels, dtype=np.int64)
        if len(pred_arr) != len(market_index):
            raise ValueError('pred_labels length must match market rows.')
        target_pos = class_predictions_to_target_positions(pred_arr)
        action = _actions_from_target_positions(target_pos)
        return pd.DataFrame(
            {'target_position': target_pos.astype(np.int8), 'action': action, 'model_label_id': pred_arr},
            index=market_index,
        )
    out = labels_df.copy()
    if not out.index.equals(market_index):
        if isinstance(out.index, pd.DatetimeIndex):
            out = _coerce_utc_datetime_index(out, timestamp_col=None)
        else:
            out = _coerce_utc_datetime_index(out, timestamp_col=timestamp_col)
        if out.index.equals(market_index):
            pass
        elif len(out) == len(market_index):
            out = out.copy()
            out.index = market_index
        else:
            raise ValueError('labels_df index does not match market index and cannot be aligned safely.')
    out = out.reindex(market_index)
    if 'target_position' not in out.columns:
        pred_col = None
        for c in ['model_label_id', 'pred_label', 'prediction', 'Label_id', 'label_id']:
            if c in out.columns:
                pred_col = c
                break
        if pred_col is None:
            raise ValueError("labels_df must contain 'target_position' or one class-prediction column.")
        out['target_position'] = class_predictions_to_target_positions(out[pred_col].to_numpy())
    out['target_position'] = pd.to_numeric(out['target_position'], errors='coerce')
    if out['target_position'].isna().any():
        raise ValueError("labels_df['target_position'] contains NaN values.")
    out['target_position'] = out['target_position'].astype(int).clip(-1, 1)
    # Canonical action derivation from target_position transitions (flip long<->short stays one signal,
    # executed as close+open by run_backtest_from_labels).
    out['action'] = _actions_from_target_positions(out['target_position'].to_numpy())
    return out

def assert_label_integrity(train_df: pd.DataFrame, labels_df: pd.DataFrame) -> None:
    if not labels_df.index.equals(train_df.index):
        raise AssertionError('labels index must match train index')
    if not set(labels_df['target_position'].unique()).issubset({-1, 0, 1}):
        raise AssertionError('target_position must be in {-1, 0, 1}')
    delta = labels_df['target_position'].diff().fillna(labels_df['target_position'])
    expected_action = np.where(delta > 0, 'buy', np.where(delta < 0, 'sell', 'hold'))
    if not np.array_equal(expected_action, labels_df['action'].to_numpy()):
        raise AssertionError('action is not coherent with target_position transitions')

def apply_slippage(price: float, transaction_side: str, slippage_bps: float) -> float:
    """Apply adverse slippage in basis points to an execution price."""
    slip = slippage_bps / 10000.0
    if transaction_side == 'buy':
        return float(price) * (1.0 + slip)
    if transaction_side == 'sell':
        return float(price) * (1.0 - slip)
    raise ValueError(f'Unknown transaction_side: {transaction_side}')

def _mtm_equity_from_notional(notional: float, direction: int, entry_price: float, mark_price: float) -> float:
    ratio = float(mark_price) / float(entry_price)
    if direction == 1:
        eq = notional * ratio
    elif direction == -1:
        eq = notional * (2.0 - ratio)
    else:
        eq = notional
    return max(float(eq), 0.0)

def run_backtest_from_labels(train_df: pd.DataFrame, labels_df: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Event-driven backtest on train set from labels.

    Rules:
    - Signal at bar t is executed at bar t+1 open.
    - Single-position V1 engine (long/short/flat).
    - A long<->short flip at open is executed as close + open (2 transactions on that bar).
    - Intrabar SL/TP checked on high/low.
    - If both SL and TP are touched in same bar, SL wins (conservative).
    - Fees and slippage are applied per transaction.
    """
    if config.allow_multiple_positions:
        raise ValueError(
            "allow_multiple_positions is incompatible with scalar target_position "
            "labels; use the portfolio backtest for concurrent asset positions."
        )
    required = {'open', 'high', 'low', 'close'}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f'train_df missing columns: {sorted(missing)}')
    if not labels_df.index.equals(train_df.index):
        raise ValueError('labels_df index must match train_df index')
    fee_rate = config.fees_bps / 10000.0
    desired_raw = labels_df['target_position'].astype(int).clip(-1, 1)
    desired_exec = desired_raw.shift(1).fillna(0).astype(int)
    equity = float(config.initial_capital)
    open_trade: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    trade_counter = 0

    def open_position(ts: pd.Timestamp, bar_idx: int, direction: int, base_price: float, reason: str) -> dict[str, Any]:
        nonlocal equity, trade_counter, open_trade
        tx_side = 'buy' if direction == 1 else 'sell'
        exec_price = apply_slippage(base_price, tx_side, config.slippage_bps)
        equity_before_entry = float(equity)
        entry_fee = equity_before_entry * fee_rate
        equity_after_fee = max(equity_before_entry - entry_fee, 0.0)
        notional = equity_after_fee
        trade_counter += 1
        open_trade = {'trade_id': trade_counter, 'direction': int(direction), 'side': 'long' if direction == 1 else 'short', 'entry_time': ts, 'entry_bar_idx': int(bar_idx), 'entry_price': float(exec_price), 'entry_price_base': float(base_price), 'entry_reason': reason, 'equity_before_entry': equity_before_entry, 'entry_equity': equity_after_fee, 'notional': notional, 'entry_fee': entry_fee}
        equity = equity_after_fee
        return {'entry_fee': entry_fee, 'entry_exec_price': exec_price, 'trade_id': trade_counter}

    def close_position(ts: pd.Timestamp, bar_idx: int, base_price: float, reason: str) -> dict[str, Any] | None:
        nonlocal equity, open_trade
        if open_trade is None:
            return None
        direction = int(open_trade['direction'])
        tx_side = 'sell' if direction == 1 else 'buy'
        exec_price = apply_slippage(base_price, tx_side, config.slippage_bps)
        gross_exit_equity = _mtm_equity_from_notional(notional=float(open_trade['notional']), direction=direction, entry_price=float(open_trade['entry_price']), mark_price=float(exec_price))
        exit_fee = gross_exit_equity * fee_rate
        net_exit_equity = max(gross_exit_equity - exit_fee, 0.0)
        duration_hours = (ts - open_trade['entry_time']).total_seconds() / 3600.0
        duration_bars = int(bar_idx - int(open_trade['entry_bar_idx']) + 1)
        net_pnl = net_exit_equity - float(open_trade['equity_before_entry'])
        return_pct = net_pnl / float(open_trade['equity_before_entry']) if float(open_trade['equity_before_entry']) > 0 else np.nan
        trade_row = {'trade_id': int(open_trade['trade_id']), 'side': open_trade['side'], 'direction': direction, 'entry_time': open_trade['entry_time'], 'exit_time': ts, 'entry_price': float(open_trade['entry_price']), 'exit_price': float(exec_price), 'entry_price_base': float(open_trade['entry_price_base']), 'exit_price_base': float(base_price), 'entry_reason': open_trade['entry_reason'], 'exit_reason': reason, 'equity_before_entry': float(open_trade['equity_before_entry']), 'entry_equity': float(open_trade['entry_equity']), 'gross_exit_equity': float(gross_exit_equity), 'exit_equity': float(net_exit_equity), 'entry_fee': float(open_trade['entry_fee']), 'exit_fee': float(exit_fee), 'total_fees': float(open_trade['entry_fee'] + exit_fee), 'net_pnl': float(net_pnl), 'return_pct': float(return_pct), 'duration_bars': duration_bars, 'duration_hours': float(duration_hours), 'is_winner': bool(net_pnl > 0)}
        trades.append(trade_row)
        equity = net_exit_equity
        open_trade = None
        return {'exit_fee': exit_fee, 'exit_exec_price': exec_price, 'trade_id': trade_row['trade_id'], 'exit_reason': reason}
    for bar_idx, (ts, row) in enumerate(train_df.iterrows()):
        bar_open = float(row['open'])
        bar_high = float(row['high'])
        bar_low = float(row['low'])
        bar_close = float(row['close'])
        desired_position = int(desired_exec.loc[ts])
        position_before_open = 0 if open_trade is None else int(open_trade['direction'])
        fees_paid_bar = 0.0
        transactions_bar = 0
        intrabar_exit_reason = None
        closed_trade_id = np.nan
        opened_trade_id = np.nan
        open_exec_price = np.nan
        close_exec_price = np.nan
        if desired_position != position_before_open:
            if open_trade is not None:
                reason = 'signal_flip' if desired_position != 0 else 'signal_exit'
                close_info = close_position(ts, bar_idx, bar_open, reason)
                if close_info is not None:
                    fees_paid_bar += float(close_info['exit_fee'])
                    transactions_bar += 1
                    close_exec_price = float(close_info['exit_exec_price'])
                    closed_trade_id = int(close_info['trade_id'])
            if desired_position != 0:
                reason = 'signal_flip_entry' if position_before_open != 0 else 'signal_entry'
                open_info = open_position(ts, bar_idx, desired_position, bar_open, reason)
                fees_paid_bar += float(open_info['entry_fee'])
                transactions_bar += 1
                open_exec_price = float(open_info['entry_exec_price'])
                opened_trade_id = int(open_info['trade_id'])
        position_after_open = 0 if open_trade is None else int(open_trade['direction'])
        if open_trade is not None and (config.stop_loss_pct > 0 or config.take_profit_pct > 0):
            direction = int(open_trade['direction'])
            entry_px = float(open_trade['entry_price'])
            sl_px = None
            tp_px = None
            hit_sl = False
            hit_tp = False
            if config.stop_loss_pct > 0:
                sl_px = entry_px * (1.0 - config.stop_loss_pct) if direction == 1 else entry_px * (1.0 + config.stop_loss_pct)
                hit_sl = bar_low <= sl_px if direction == 1 else bar_high >= sl_px
            if config.take_profit_pct > 0:
                tp_px = entry_px * (1.0 + config.take_profit_pct) if direction == 1 else entry_px * (1.0 - config.take_profit_pct)
                hit_tp = bar_high >= tp_px if direction == 1 else bar_low <= tp_px
            if hit_sl or hit_tp:
                if hit_sl and hit_tp:
                    exit_base = float(sl_px)
                    intrabar_exit_reason = 'stop_loss_conflict'
                elif hit_sl:
                    exit_base = float(sl_px)
                    intrabar_exit_reason = 'stop_loss'
                else:
                    exit_base = float(tp_px)
                    intrabar_exit_reason = 'take_profit'
                close_info = close_position(ts, bar_idx, exit_base, intrabar_exit_reason)
                if close_info is not None:
                    fees_paid_bar += float(close_info['exit_fee'])
                    transactions_bar += 1
                    close_exec_price = float(close_info['exit_exec_price'])
                    closed_trade_id = int(close_info['trade_id'])
        if open_trade is None:
            equity_close = float(equity)
            active_trade_id = np.nan
            position_close = 0
        else:
            equity_close = _mtm_equity_from_notional(notional=float(open_trade['notional']), direction=int(open_trade['direction']), entry_price=float(open_trade['entry_price']), mark_price=bar_close)
            active_trade_id = int(open_trade['trade_id'])
            position_close = int(open_trade['direction'])
        ledger_rows.append({'timestamp': ts, 'open': bar_open, 'high': bar_high, 'low': bar_low, 'close': bar_close, 'signal_target_position': int(desired_raw.loc[ts]), 'desired_position': desired_position, 'position_before_open': position_before_open, 'position_after_open': position_after_open, 'position_close': position_close, 'transactions_bar': int(transactions_bar), 'fees_paid_bar': float(fees_paid_bar), 'opened_trade_id': opened_trade_id, 'closed_trade_id': closed_trade_id, 'active_trade_id': active_trade_id, 'open_exec_price': open_exec_price, 'close_exec_price': close_exec_price, 'intrabar_exit_reason': intrabar_exit_reason, 'equity_close': float(equity_close), 'realized_equity': float(equity)})
    if open_trade is not None:
        last_ts = train_df.index[-1]
        last_close = float(train_df.iloc[-1]['close'])
        close_info = close_position(last_ts, len(train_df) - 1, last_close, 'eod_close')
        if close_info is not None:
            ledger_rows[-1]['transactions_bar'] += 1
            ledger_rows[-1]['fees_paid_bar'] += float(close_info['exit_fee'])
            ledger_rows[-1]['closed_trade_id'] = int(close_info['trade_id'])
            ledger_rows[-1]['close_exec_price'] = float(close_info['exit_exec_price'])
            if ledger_rows[-1]['intrabar_exit_reason'] is None:
                ledger_rows[-1]['intrabar_exit_reason'] = 'eod_close'
            ledger_rows[-1]['position_close'] = 0
            ledger_rows[-1]['active_trade_id'] = np.nan
            ledger_rows[-1]['equity_close'] = float(equity)
            ledger_rows[-1]['realized_equity'] = float(equity)
    bar_ledger = pd.DataFrame(ledger_rows).set_index('timestamp')
    bar_ledger.index = pd.DatetimeIndex(bar_ledger.index, tz='UTC')
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values('entry_time').reset_index(drop=True)
    equity_curve = bar_ledger[['equity_close']].rename(columns={'equity_close': 'equity'})
    positions_df = bar_ledger[['signal_target_position', 'desired_position', 'position_before_open', 'position_after_open', 'position_close', 'active_trade_id']].copy()
    return (bar_ledger, trades_df, equity_curve, positions_df)

def compute_drawdown_series(equity_curve: pd.DataFrame) -> pd.DataFrame:
    eq = equity_curve['equity'].astype(float)
    rolling_peak = eq.cummax()
    drawdown = eq / rolling_peak - 1.0
    out = pd.DataFrame({'equity': eq, 'rolling_peak': rolling_peak, 'drawdown': drawdown}, index=equity_curve.index)
    return out

def compute_core_metrics(equity_curve: pd.DataFrame, trades_df: pd.DataFrame, config: BacktestConfig, bar_ledger: pd.DataFrame | None=None) -> dict[str, float | int | None]:
    if equity_curve.empty:
        raise ValueError('equity_curve is empty')
    eq = equity_curve['equity'].astype(float)
    ret = eq.pct_change().fillna(0.0)
    drawdown_df = compute_drawdown_series(equity_curve)
    total_pnl = float(eq.iloc[-1] - eq.iloc[0])
    cumulative_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    excess_ret = ret - config.risk_free_rate / config.annualization_factor
    ret_std = float(excess_ret.std(ddof=0))
    sharpe = float(np.sqrt(config.annualization_factor) * excess_ret.mean() / ret_std) if ret_std > 0 else np.nan
    max_drawdown = float(drawdown_df['drawdown'].min())
    total_trades = int(len(trades_df))
    if total_trades > 0:
        pnl_series = trades_df['net_pnl'].astype(float)
        wins = pnl_series[pnl_series > 0]
        losses = pnl_series[pnl_series < 0]
        win_rate = float((pnl_series > 0).mean())
        gross_profit = float(wins.sum())
        gross_loss = float(losses.sum())
        profit_factor = float(gross_profit / abs(gross_loss)) if gross_loss < 0 else np.inf if gross_profit > 0 else np.nan
        expectancy = float(pnl_series.mean())
        avg_win = float(wins.mean()) if len(wins) else np.nan
        avg_loss = float(losses.mean()) if len(losses) else np.nan
        gain_loss_ratio = float(avg_win / abs(avg_loss)) if not np.isnan(avg_win) and (not np.isnan(avg_loss)) and (avg_loss != 0) else np.nan
        avg_trade_duration_hours = float(trades_df['duration_hours'].mean())
    else:
        win_rate = np.nan
        profit_factor = np.nan
        expectancy = np.nan
        gain_loss_ratio = np.nan
        avg_trade_duration_hours = np.nan
    if bar_ledger is not None and (not bar_ledger.empty):
        exposure = float((bar_ledger['position_close'].abs() > 0).mean())
    else:
        exposure = np.nan
    return {'initial_capital': float(eq.iloc[0]), 'final_capital': float(eq.iloc[-1]), 'total_pnl': total_pnl, 'cumulative_return': cumulative_return, 'sharpe_ratio': sharpe, 'max_drawdown': max_drawdown, 'win_rate': win_rate, 'profit_factor': profit_factor, 'expectancy': expectancy, 'total_trades': total_trades, 'gain_loss_ratio': gain_loss_ratio, 'avg_trade_duration_hours': avg_trade_duration_hours, 'exposure': exposure}

def _aggregate_return_groups(df: pd.DataFrame, group_key: str, sort_by: str | None=None) -> pd.DataFrame:
    grouped = df.groupby(group_key)['return']
    out = grouped.agg(['sum', 'mean', 'median', 'std', 'count']).rename(columns={'sum': 'sum_return', 'mean': 'mean_return', 'median': 'median_return', 'std': 'std_return', 'count': 'count_bars'})
    out['positive_rate'] = grouped.apply(lambda s: float((s > 0).mean()))
    out['cumulative_return'] = grouped.apply(lambda s: float((1.0 + s).prod() - 1.0))
    out = out.reset_index().rename(columns={group_key: 'bucket'})
    if sort_by is not None and sort_by in out.columns:
        out = out.sort_values(sort_by).reset_index(drop=True)
    return out

def compute_time_decomposition(equity_curve: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if equity_curve.empty:
        raise ValueError('equity_curve is empty')
    base = pd.DataFrame(index=equity_curve.index.copy())
    base['return'] = equity_curve['equity'].pct_change().fillna(0.0)
    base['hour'] = base.index.hour
    base['day_name'] = base.index.day_name()
    base['day_num'] = base.index.dayofweek
    iso = base.index.isocalendar()
    base['week'] = iso['year'].astype(str) + '-W' + iso['week'].astype(str).str.zfill(2)
    index_naive = base.index.tz_localize(None) if base.index.tz is not None else base.index
    base['month'] = index_naive.to_period('M').astype(str)
    base['quarter'] = index_naive.to_period('Q').astype(str)
    base['year'] = base.index.year.astype(str)
    hour_df = _aggregate_return_groups(base, 'hour', sort_by='bucket')
    day_df = _aggregate_return_groups(base, 'day_name')
    day_order = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    day_df['day_num'] = day_df['bucket'].map(day_order)
    day_df = day_df.sort_values('day_num').drop(columns=['day_num']).reset_index(drop=True)
    week_df = _aggregate_return_groups(base, 'week', sort_by='bucket')
    month_df = _aggregate_return_groups(base, 'month', sort_by='bucket')
    quarter_df = _aggregate_return_groups(base, 'quarter', sort_by='bucket')
    year_df = _aggregate_return_groups(base, 'year', sort_by='bucket')
    duration_hours = (base.index.max() - base.index.min()).total_seconds() / 3600.0
    total_duration_df = pd.DataFrame([{'bucket': 'full_backtest', 'start': base.index.min(), 'end': base.index.max(), 'duration_hours': float(duration_hours), 'count_bars': int(len(base)), 'cumulative_return': float((1.0 + base['return']).prod() - 1.0)}])
    return {'hour': hour_df, 'day': day_df, 'week': week_df, 'month': month_df, 'quarter': quarter_df, 'year': year_df, 'total_duration': total_duration_df, 'base_returns': base}

def flatten_time_decomposition(time_decomp: dict[str, pd.DataFrame]) -> pd.DataFrame:
    chunks = []
    for key, frame in time_decomp.items():
        if key == 'base_returns':
            continue
        tmp = frame.copy()
        tmp['bucket_type'] = str(key)
        chunks.append(tmp)
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True, sort=False)
    if 'bucket' in out.columns:
        out['bucket'] = out['bucket'].astype('string')
    out['bucket_type'] = out['bucket_type'].astype('string')
    cols = ['bucket_type'] + [c for c in out.columns if c != 'bucket_type']
    return out[cols]

def _safe_metric_display(metrics: dict[str, Any], key: str, fmt: str='.4f') -> str:
    value = metrics.get(key, np.nan)
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return 'n/a'
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return format(float(value), fmt)

def build_viz_bundle(core_metrics: dict[str, Any], equity_curve: pd.DataFrame, drawdown_df: pd.DataFrame, trades_df: pd.DataFrame, time_decomp: dict[str, pd.DataFrame]) -> dict[str, go.Figure]:
    kpi_rows = [('PnL total', _safe_metric_display(core_metrics, 'total_pnl', '.2f')), ('Rendement cumule', _safe_metric_display(core_metrics, 'cumulative_return', '.4%')), ('Sharpe ratio', _safe_metric_display(core_metrics, 'sharpe_ratio', '.3f')), ('Drawdown maximal', _safe_metric_display(core_metrics, 'max_drawdown', '.4%')), ('Win rate', _safe_metric_display(core_metrics, 'win_rate', '.2%')), ('Profit factor', _safe_metric_display(core_metrics, 'profit_factor', '.3f')), ('Expectancy', _safe_metric_display(core_metrics, 'expectancy', '.3f')), ('Nombre total de trades', _safe_metric_display(core_metrics, 'total_trades')), ('Ratio gains/pertes', _safe_metric_display(core_metrics, 'gain_loss_ratio', '.3f')), ('Duree moyenne trades (h)', _safe_metric_display(core_metrics, 'avg_trade_duration_hours', '.2f')), ('Exposition', _safe_metric_display(core_metrics, 'exposure', '.2%'))]
    kpi_fig = go.Figure(data=[go.Table(header=dict(values=['Metric', 'Value'], fill_color='#111827', font=dict(color='white')), cells=dict(values=[[r[0] for r in kpi_rows], [r[1] for r in kpi_rows]], fill_color='#F8FAFC'))])
    kpi_fig.update_layout(title='Synthese Performance')
    equity_fig = go.Figure(data=[go.Scatter(x=equity_curve.index, y=equity_curve['equity'], mode='lines', name='Equity')])
    equity_fig.update_layout(title='Evolution du Capital', xaxis_title='Time', yaxis_title='Equity')
    dd_fig = go.Figure(data=[go.Scatter(x=drawdown_df.index, y=drawdown_df['drawdown'] * 100, mode='lines', fill='tozeroy', name='Drawdown %')])
    dd_fig.update_layout(title='Drawdown', xaxis_title='Time', yaxis_title='Drawdown (%)')
    if trades_df.empty:
        dist_fig = go.Figure()
        dist_fig.add_annotation(text='No trades', showarrow=False, x=0.5, y=0.5, xref='paper', yref='paper')
    else:
        dist_fig = go.Figure(data=[go.Histogram(x=trades_df['net_pnl'], nbinsx=min(50, max(10, len(trades_df) // 2)), name='Trade PnL')])
    dist_fig.update_layout(title='Repartition Gains / Pertes', xaxis_title='Trade PnL', yaxis_title='Count')
    base = time_decomp['base_returns'].copy()
    heat = base.pivot_table(index=base.index.dayofweek, columns=base.index.hour, values='return', aggfunc='mean')
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    heat_fig = go.Figure(data=[go.Heatmap(z=heat.values * 100 if heat.size else [[0]], x=list(heat.columns) if heat.size else [0], y=[day_labels[i] for i in heat.index] if heat.size else ['n/a'], colorscale='RdYlGn', zmid=0, colorbar=dict(title='Mean return %'))])
    heat_fig.update_layout(title='Heatmap Performance (Jour x Heure)', xaxis_title='Hour', yaxis_title='Day')
    ret = base['return']
    month_base = pd.DataFrame({'year': ret.index.year.astype(int), 'month': ret.index.month.astype(int), 'return': ret.values})
    month_ret = month_base.groupby(['year', 'month'])['return'].apply(lambda s: (1.0 + s).prod() - 1.0).unstack('month')
    month_ret = month_ret.reindex(columns=list(range(1, 13)))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_fig = go.Figure(data=[go.Heatmap(z=month_ret.values * 100 if not month_ret.empty else [[0]], x=month_names if not month_ret.empty else ['n/a'], y=[str(y) for y in month_ret.index] if not month_ret.empty else ['n/a'], colorscale='RdYlGn', zmid=0, colorbar=dict(title='Return %'), hovertemplate='Year %{y}<br>Month %{x}<br>Return %{z:.2f}%<extra></extra>')])
    month_fig.update_layout(title='Heatmap Performance (Annee x Mois)')
    return {'kpi_table': kpi_fig, 'equity_curve': equity_fig, 'drawdown': dd_fig, 'pnl_distribution': dist_fig, 'heatmap_hour_day': heat_fig, 'heatmap_month_year': month_fig}

def render_viz_bundle(viz_bundle: dict[str, go.Figure]) -> None:
    for name, fig in viz_bundle.items():
        print(f'[viz] {name}')
        try:
            fig.show()
        except Exception as exc:
            msg = str(exc)
            is_no_nbformat = 'nbformat>=4.2.0' in msg or 'Mime type rendering' in msg
            is_headless_renderer = 'Operation not permitted' in msg or exc.__class__.__name__ == 'PermissionError'
            if is_no_nbformat or is_headless_renderer:
                print(f'[viz] Inline render skipped for {name}: renderer unavailable in this environment.')
            else:
                raise

def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding='utf-8')

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))

def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert pandas dtypes that are known to be problematic with some pyarrow versions.
    In particular: Interval / Categorical[Interval].
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        s = out[col]
        dtype = s.dtype
        if isinstance(dtype, pd.IntervalDtype):
            out[col] = s.astype(str)
            continue
        if isinstance(dtype, pd.CategoricalDtype):
            cat_dtype = getattr(dtype, 'categories', None)
            cat_inner_dtype = getattr(cat_dtype, 'dtype', None)
            if isinstance(cat_inner_dtype, pd.IntervalDtype):
                out[col] = s.astype(str)
                continue
        if dtype == object:
            sample = s.dropna().head(32)
            if (not sample.empty) and sample.map(lambda x: isinstance(x, pd.Interval)).any():
                out[col] = s.map(lambda x: str(x) if isinstance(x, pd.Interval) else x)
    return out

def _write_parquet_safe(df: pd.DataFrame, path: Path) -> None:
    """Write parquet with a fallback for pyarrow pandas.period extension collisions."""
    df_to_write = _sanitize_for_parquet(df)
    try:
        df_to_write.to_parquet(path, engine='pyarrow')
        return
    except Exception as exc:
        msg = str(exc)
        is_period_collision = 'pandas.period' in msg and 'already defined' in msg or exc.__class__.__name__ == 'ArrowKeyError'
        if not is_period_collision:
            raise
    import pyarrow as pa
    import pyarrow.parquet as pq
    try:
        if hasattr(pa, 'unregister_extension_type'):
            pa.unregister_extension_type('pandas.period')
    except Exception:
        pass
    table = pa.Table.from_pandas(df_to_write, preserve_index=True)
    pq.write_table(table, path)

def _read_parquet_safe(path: Path) -> pd.DataFrame:
    """Read parquet with fallback when pandas<->pyarrow extension registration collides."""
    try:
        return pd.read_parquet(path, engine='pyarrow')
    except Exception as exc:
        msg = str(exc)
        is_period_collision = 'pandas.period' in msg and 'already defined' in msg or exc.__class__.__name__ == 'ArrowKeyError'
        if not is_period_collision:
            raise
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    return table.to_pandas()
REQUIRED_ARTIFACTS = ['manifest.json', 'config.json', 'environment.json', 'data/raw_market.parquet', 'data/train.parquet', 'data/val.parquet', 'data/test.parquet', 'strategy/labels.parquet', 'strategy/signals.parquet', 'backtest/bar_ledger.parquet', 'backtest/trades.parquet', 'backtest/equity_curve.parquet', 'backtest/positions.parquet', 'metrics/core_metrics.json', 'metrics/time_decomposition.parquet', 'metrics/drawdown_series.parquet', 'future/events.parquet', 'future/regimes.parquet', 'future/features_ml.parquet', 'future/scoring.parquet']

def validate_required_artifacts(run_path: Path) -> list[str]:
    missing = []
    for rel in REQUIRED_ARTIFACTS:
        if not (run_path / rel).exists():
            missing.append(rel)
    viz_dir = run_path / 'viz'
    if not viz_dir.exists() or not any(viz_dir.glob('*.html')):
        missing.append('viz/*.html')
    return missing

def execute_first_check_pipeline(config: BacktestConfig, persist: bool=True, render: bool=False) -> tuple[dict[str, Any], Path | None]:
    config = resolve_config(config)
    validate_config(config)
    raw_market = download_market_data(config)
    assert_market_data_integrity(raw_market)
    splits = split_time_series(raw_market, ratios=config.split_ratios)
    train_df = splits['train']
    labels = generate_strategy_labels(train_df, config)
    assert_label_integrity(train_df, labels)
    signals = labels_to_signals(labels)
    bar_ledger, trades, equity_curve, positions = run_backtest_from_labels(train_df, labels, config)
    drawdown_df = compute_drawdown_series(equity_curve)
    core_metrics = compute_core_metrics(equity_curve, trades, config, bar_ledger=bar_ledger)
    time_decomp = compute_time_decomposition(equity_curve)
    time_decomp_flat = flatten_time_decomposition(time_decomp)
    viz_bundle = build_viz_bundle(core_metrics, equity_curve, drawdown_df, trades, time_decomp)
    run_obj = {'raw_market': raw_market, 'splits': splits, 'labels': labels, 'signals': signals, 'bar_ledger': bar_ledger, 'trades': trades, 'equity_curve': equity_curve, 'positions': positions, 'core_metrics': core_metrics, 'time_decomposition': time_decomp, 'time_decomposition_flat': time_decomp_flat, 'drawdown_series': drawdown_df, 'viz_bundle': viz_bundle}
    run_path = save_run_artifacts(run_obj, config) if persist else None
    if render:
        display(summarize_splits(splits))
        display(pd.DataFrame([core_metrics]))
        render_viz_bundle(viz_bundle)
    return (run_obj, run_path)

def execute_first_check_pipeline_external(
    config: BacktestConfig,
    market_df: pd.DataFrame,
    labels_df: pd.DataFrame | None=None,
    pred_labels: list[int] | np.ndarray | pd.Series | None=None,
    market_timestamp_col: str | None=None,
    labels_timestamp_col: str | None=None,
    persist: bool=True,
    render: bool=False,
) -> tuple[dict[str, Any], Path | None]:
    """
    First-check pipeline variant for external market data and external ANN labels.

    Notes:
    - Uses provided data (no CCXT download).
    - Backtest runs on the full external frame and is stored under splits['train'].
    - labels_df can provide target_position/action directly, or class predictions (0/1/2).
    """
    cfg = resolve_config(config)
    validate_config(cfg)
    raw_market = prepare_external_market_data(market_df, cfg, timestamp_col=market_timestamp_col)
    labels = prepare_external_labels(
        market_index=raw_market.index,
        labels_df=labels_df,
        pred_labels=pred_labels,
        timestamp_col=labels_timestamp_col,
    )
    assert_label_integrity(raw_market, labels)
    signals = labels_to_signals(labels)
    bar_ledger, trades, equity_curve, positions = run_backtest_from_labels(raw_market, labels, cfg)
    drawdown_df = compute_drawdown_series(equity_curve)
    core_metrics = compute_core_metrics(equity_curve, trades, cfg, bar_ledger=bar_ledger)
    time_decomp = compute_time_decomposition(equity_curve)
    time_decomp_flat = flatten_time_decomposition(time_decomp)
    viz_bundle = build_viz_bundle(core_metrics, equity_curve, drawdown_df, trades, time_decomp)
    empty_split = raw_market.iloc[0:0].copy()
    splits = {'train': raw_market.copy(), 'val': empty_split.copy(), 'test': empty_split.copy()}
    run_obj = {
        'raw_market': raw_market,
        'splits': splits,
        'labels': labels,
        'signals': signals,
        'bar_ledger': bar_ledger,
        'trades': trades,
        'equity_curve': equity_curve,
        'positions': positions,
        'core_metrics': core_metrics,
        'time_decomposition': time_decomp,
        'time_decomposition_flat': time_decomp_flat,
        'drawdown_series': drawdown_df,
        'viz_bundle': viz_bundle,
    }
    run_path = save_run_artifacts(run_obj, cfg) if persist else None
    if render:
        display(summarize_splits(splits))
        display(pd.DataFrame([core_metrics]))
        render_viz_bundle(viz_bundle)
    return (run_obj, run_path)

def _build_synthetic_ohlcv(index: pd.DatetimeIndex, close_values: list[float]) -> pd.DataFrame:
    close = pd.Series(close_values, index=index, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) * 1.002
    low = pd.concat([open_, close], axis=1).min(axis=1) * 0.998
    volume = pd.Series(1000.0, index=index)
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=index)
    return df

def run_acceptance_checks(run_obj: dict[str, Any], config: BacktestConfig, run_path: Path | None=None) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []
    raw = run_obj['raw_market']
    splits = run_obj['splits']
    labels = run_obj['labels']
    ledger = run_obj['bar_ledger']
    trades = run_obj['trades']
    equity = run_obj['equity_curve']
    core = run_obj['core_metrics']
    time_decomp = run_obj['time_decomposition']
    try:
        assert_market_data_integrity(raw)
        assert splits['train'].index.max() < splits['val'].index.min() < splits['test'].index.min()
        checks.append({'check': '1.data_integrity', 'passed': True, 'details': 'OK'})
    except Exception as exc:
        checks.append({'check': '1.data_integrity', 'passed': False, 'details': str(exc)})
    try:
        assert_label_integrity(splits['train'], labels)
        checks.append({'check': '2.label_validity', 'passed': True, 'details': 'OK'})
    except Exception as exc:
        checks.append({'check': '2.label_validity', 'passed': False, 'details': str(exc)})
    try:
        assert set(ledger['position_close'].unique()).issubset({-1, 0, 1})
        assert (ledger['fees_paid_bar'] >= 0).all()
        checks.append({'check': '3.backtest_invariants', 'passed': True, 'details': 'OK'})
    except Exception as exc:
        checks.append({'check': '3.backtest_invariants', 'passed': False, 'details': str(exc)})
    try:
        idx = pd.date_range('2025-01-01', periods=4, freq='1h', tz='UTC')
        df = pd.DataFrame({'open': [100, 100, 100, 100], 'high': [100, 103, 101, 101], 'low': [100, 97, 99, 99], 'close': [100, 101, 100, 100], 'volume': [1000, 1000, 1000, 1000]}, index=idx)
        lbl = pd.DataFrame({'target_position': [1, 1, 1, 1], 'action': ['buy', 'hold', 'hold', 'hold']}, index=idx)
        conf = replace(config, stop_loss_pct=0.01, take_profit_pct=0.01, fees_bps=0.0, slippage_bps=0.0)
        _, tr, _, _ = run_backtest_from_labels(df, lbl, conf)
        assert not tr.empty
        assert 'stop_loss_conflict' in set(tr['exit_reason'])
        checks.append({'check': '4.sltp_edge_cases', 'passed': True, 'details': 'OK'})
    except Exception as exc:
        checks.append({'check': '4.sltp_edge_cases', 'passed': False, 'details': str(exc)})
    try:
        idx = pd.date_range('2025-02-01', periods=10, freq='1h', tz='UTC')
        close_vals = [100 + i for i in range(10)]
        toy_df = _build_synthetic_ohlcv(idx, close_vals)
        toy_lbl = pd.DataFrame({'target_position': [1] * len(idx), 'action': ['buy'] + ['hold'] * (len(idx) - 1)}, index=idx)
        conf = replace(config, stop_loss_pct=0.0, take_profit_pct=0.0, fees_bps=0.0, slippage_bps=0.0)
        _, toy_trades, toy_eq, _ = run_backtest_from_labels(toy_df, toy_lbl, conf)
        toy_metrics = compute_core_metrics(toy_eq, toy_trades, conf)
        assert toy_metrics['final_capital'] > toy_metrics['initial_capital']
        checks.append({'check': '5.metrics_sanity', 'passed': True, 'details': 'OK'})
    except Exception as exc:
        checks.append({'check': '5.metrics_sanity', 'passed': False, 'details': str(exc)})
    try:
        idx = pd.date_range('2025-02-02', periods=4, freq='1h', tz='UTC')
        flip_df = _build_synthetic_ohlcv(idx, [100, 101, 100, 102])
        flip_lbl = pd.DataFrame({'target_position': [1, -1, -1, -1], 'action': ['buy', 'sell', 'hold', 'hold']}, index=idx)
        conf = replace(config, stop_loss_pct=0.0, take_profit_pct=0.0, fees_bps=0.0, slippage_bps=0.0)
        flip_ledger, _, _, _ = run_backtest_from_labels(flip_df, flip_lbl, conf)
        assert int(flip_ledger.iloc[1]['transactions_bar']) == 2
        checks.append({'check': '5b.flip_long_short_two_transactions', 'passed': True, 'details': 'OK'})
    except Exception as exc:
        checks.append({'check': '5b.flip_long_short_two_transactions', 'passed': False, 'details': str(exc)})
    try:
        total_from_equity = equity['equity'].iloc[-1] / equity['equity'].iloc[0] - 1.0
        assert abs(total_from_equity - core['cumulative_return']) < 1e-08
        month_df = time_decomp['month']
        assert month_df['count_bars'].sum() == len(time_decomp['base_returns'])
        checks.append({'check': '6.time_decomposition', 'passed': True, 'details': 'OK'})
    except Exception as exc:
        checks.append({'check': '6.time_decomposition', 'passed': False, 'details': str(exc)})
    try:
        cfg_hash_1 = hash_object(config_to_dict(resolve_config(config)))
        cfg_hash_2 = hash_object(config_to_dict(resolve_config(config)))
        assert cfg_hash_1 == cfg_hash_2
        raw_hash = hash_dataframe(raw)
        assert isinstance(raw_hash, str) and len(raw_hash) == 64
        checks.append({'check': '7.reproducibility_hashes', 'passed': True, 'details': 'OK'})
    except Exception as exc:
        checks.append({'check': '7.reproducibility_hashes', 'passed': False, 'details': str(exc)})
    try:
        if run_path is None:
            checks.append({'check': '8.artifact_completeness', 'passed': True, 'details': 'Skipped (persist=False)'})
        else:
            missing = validate_required_artifacts(run_path)
            assert not missing, f'Missing artifacts: {missing}'
            checks.append({'check': '8.artifact_completeness', 'passed': True, 'details': 'OK'})
    except Exception as exc:
        checks.append({'check': '8.artifact_completeness', 'passed': False, 'details': str(exc)})
    report = pd.DataFrame(checks)
    return report

def assert_acceptance_passed(report: pd.DataFrame) -> None:
    failed = report[~report['passed']]
    if not failed.empty:
        raise AssertionError(f'Acceptance checks failed:\n{failed}')
from dataclasses import field
import warnings
try:
    from scipy.stats import mannwhitneyu, ks_2samp
except Exception:
    mannwhitneyu = None
    ks_2samp = None
try:
    import statsmodels.api as sm
except Exception:
    sm = None
try:
    from hmmlearn.hmm import GaussianHMM
except Exception:
    GaussianHMM = None
try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import IsolationForest
    from sklearn.inspection import permutation_importance as sk_permutation_importance
except Exception:
    average_precision_score = None
    roc_auc_score = None
    StandardScaler = None
    DecisionTreeClassifier = None
    IsolationForest = None
    sk_permutation_importance = None

@dataclass(frozen=True)
class EventSchemaV1:
    required_columns: tuple[str, ...] = ('timestamp', 'event_type', 'event_name', 'asset', 'importance', 'source', 'notes')
    timezone: str = 'UTC'

@dataclass(frozen=True)
class RegimeModelConfig:
    n_states: int = 4
    return_lookback: int = 1
    vol_window: int = 24
    trend_fast: int = 20
    trend_slow: int = 100
    volume_window: int = 48
    random_state: int = 42
    covariance_type: str = 'diag'
    n_iter: int = 300

@dataclass(frozen=True)
class ScoringConfig:
    target_col: str = 'is_winner'
    train_ratio: float = 0.7
    random_state: int = 42
    max_tree_depth: int = 3
    min_samples_leaf: int = 10
    min_train_rows: int = 30

@dataclass(frozen=True)
class VariantConfig:
    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

def _find_project_root(start: Path | None=None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start] + list(start.parents):
        if (candidate / 'artifacts').exists() and (candidate / 'requirements.txt').exists():
            return candidate
        if (candidate / '.git').exists():
            return candidate
    return start

def resolve_artifact_root(config: BacktestConfig, notebook_dir: Path | None=None) -> Path:
    """
    Resolve a stable absolute artifact root independent from the notebook CWD.
    This prevents nested roots like artifacts/artifacts/runs.
    """
    raw = Path(config.artifact_root).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    project_root = _find_project_root(notebook_dir or Path.cwd())
    resolved = (project_root / raw).resolve()
    return resolved

def _empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)

def _to_datetime_utc_ns(values: pd.Series | pd.Index | list[Any] | np.ndarray) -> pd.Series:
    """Parse datetime-like values as UTC and canonicalize to ns resolution."""
    parsed = pd.to_datetime(values, utc=True, errors='coerce')
    # Re-parse string representation to force a consistent datetime64[ns, UTC] dtype.
    return pd.to_datetime(pd.Series(parsed).astype('string'), utc=True, errors='coerce')

def _ensure_datetime_utc(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = _to_datetime_utc_ns(out[c])
    return out

def _load_optional_parquet(path: Path, default_columns: list[str] | None=None) -> pd.DataFrame:
    if path.exists():
        return _read_parquet_safe(path)
    if default_columns is None:
        return pd.DataFrame()
    return _empty_df(default_columns)

def _run_catalog_columns() -> list[str]:
    return ['run_id', 'created_at_utc', 'run_path', 'symbol', 'timeframe', 'start', 'end', 'strategy_tag', 'config_hash', 'raw_market_hash', 'train_hash', 'total_pnl', 'cumulative_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor', 'total_trades', 'notes']

def load_run_catalog(config: BacktestConfig, notebook_dir: Path | None=None) -> pd.DataFrame:
    catalog_path = resolve_artifact_root(config, notebook_dir=notebook_dir) / 'run_catalog.parquet'
    if not catalog_path.exists():
        return pd.DataFrame(columns=_run_catalog_columns())
    catalog = _read_parquet_safe(catalog_path)
    for c in _run_catalog_columns():
        if c not in catalog.columns:
            catalog[c] = pd.NA
    catalog = catalog[_run_catalog_columns()].copy()
    if 'created_at_utc' in catalog.columns:
        catalog['created_at_utc'] = pd.to_datetime(catalog['created_at_utc'], utc=True, errors='coerce')
    return catalog.sort_values('created_at_utc').reset_index(drop=True)

def _upsert_run_catalog(catalog_path: Path, row: dict[str, Any]) -> None:
    if catalog_path.exists():
        catalog = _read_parquet_safe(catalog_path)
    else:
        catalog = pd.DataFrame(columns=_run_catalog_columns())
    for c in _run_catalog_columns():
        if c not in catalog.columns:
            catalog[c] = pd.NA
    run_id = str(row['run_id'])
    catalog = catalog[catalog['run_id'].astype(str) != run_id].copy()
    catalog = pd.concat([catalog, pd.DataFrame([row])], ignore_index=True)
    catalog['created_at_utc'] = pd.to_datetime(catalog['created_at_utc'], utc=True, errors='coerce')
    catalog = catalog.sort_values('created_at_utc').reset_index(drop=True)
    _write_parquet_safe(catalog[_run_catalog_columns()], catalog_path)

def create_future_placeholders() -> dict[str, pd.DataFrame]:
    event_schema = EventSchemaV1()
    return {'events': _empty_df(list(event_schema.required_columns)), 'regimes': _empty_df(['timestamp', 'regime_state', 'regime_label', 'confidence', 'ret_1', 'vol_rolling', 'trend_proxy', 'volume_zscore']), 'features_ml': _empty_df(['trade_id', 'entry_time', 'is_winner', 'feature_set_version']), 'scoring': _empty_df(['trade_id', 'entry_time', 'target', 'score', 'prob_logit', 'prob_tree', 'split', 'explanation'])}
REQUIRED_ARTIFACTS_EXTENDED = ['manifest.json', 'config.json', 'environment.json', 'data/raw_market.parquet', 'data/train.parquet', 'data/val.parquet', 'data/test.parquet', 'strategy/labels.parquet', 'strategy/signals.parquet', 'backtest/bar_ledger.parquet', 'backtest/trades.parquet', 'backtest/equity_curve.parquet', 'backtest/positions.parquet', 'metrics/core_metrics.json', 'metrics/time_decomposition.parquet', 'metrics/drawdown_series.parquet', 'metrics/benchmark_metrics.json', 'metrics/split_metrics.parquet', 'metrics/event_impact.parquet', 'metrics/regime_performance.parquet', 'metrics/hmm_model_selection.parquet', 'metrics/scoring_calibration.parquet', 'metrics/stress_scenarios.parquet', 'metrics/walkforward_summary.parquet', 'metrics/feature_stability.parquet', 'metrics/robustness_report.parquet', 'metrics/failure_diagnosis.parquet', 'metrics/stat_robustness_summary.parquet', 'metrics/stat_bootstrap_samples.parquet', 'metrics/parameter_sensitivity.parquet', 'metrics/feature_drift.parquet', 'metrics/meta_labeling_scores.parquet', 'metrics/meta_labeling_thresholds.parquet', 'metrics/meta_labeling_report.json', 'metrics/auto_recommendations.parquet', 'metrics/purged_cv.parquet', 'metrics/purged_cv_summary.parquet', 'metrics/overfit_report.json', 'metrics/execution_impact.parquet', 'metrics/execution_trade_impact.parquet', 'metrics/execution_report.json', 'metrics/trade_root_cause.parquet', 'metrics/trade_replay_paths.parquet', 'metrics/trade_root_cause_summary.parquet', 'metrics/portfolio_attribution.parquet', 'metrics/portfolio_asset_metrics.parquet', 'metrics/portfolio_correlation.parquet', 'metrics/portfolio_report.json', 'metrics/anomaly_scores.parquet', 'metrics/trade_anomalies.parquet', 'metrics/anomaly_impact.parquet', 'metrics/anomaly_report.json', 'metrics/model_report.json', 'future/events.parquet', 'future/regimes.parquet', 'future/features_ml.parquet', 'future/scoring.parquet', 'viz/dashboard_main.html', 'viz/report_v2.html', 'viz/report_v3.html']

def validate_required_artifacts_extended(run_path: Path) -> list[str]:
    missing = []
    for rel in REQUIRED_ARTIFACTS_EXTENDED:
        if not (run_path / rel).exists():
            missing.append(rel)
    return missing

def save_run_artifacts(run_obj: dict[str, Any], config: BacktestConfig) -> Path:
    config = resolve_config(config)
    artifact_root = resolve_artifact_root(config, notebook_dir=Path.cwd())
    _ensure_dir(artifact_root)
    run_id = make_run_id(config)
    run_path = artifact_root / run_id
    data_dir = run_path / 'data'
    strategy_dir = run_path / 'strategy'
    backtest_dir = run_path / 'backtest'
    metrics_dir = run_path / 'metrics'
    viz_dir = run_path / 'viz'
    future_dir = run_path / 'future'
    for d in [run_path, data_dir, strategy_dir, backtest_dir, metrics_dir, viz_dir, future_dir]:
        _ensure_dir(d)
    raw_market = run_obj['raw_market'].copy()
    splits = run_obj['splits']
    labels = run_obj['labels'].copy()
    signals = run_obj['signals'].copy()
    bar_ledger = run_obj['bar_ledger'].copy()
    trades = run_obj['trades'].copy()
    equity_curve = run_obj['equity_curve'].copy()
    positions = run_obj['positions'].copy()
    drawdown = run_obj.get('drawdown_series', compute_drawdown_series(equity_curve)).copy()
    core_metrics = dict(run_obj.get('core_metrics', {}))
    if 'time_decomposition_flat' in run_obj:
        time_decomp_flat = run_obj['time_decomposition_flat'].copy()
    elif 'time_decomposition' in run_obj:
        time_decomp_flat = flatten_time_decomposition(run_obj['time_decomposition'])
    else:
        time_decomp_flat = pd.DataFrame()
    benchmark_metrics = dict(run_obj.get('benchmark_metrics', {}))
    split_metrics = run_obj.get('split_metrics', pd.DataFrame()).copy()
    event_impact = run_obj.get('event_impact', pd.DataFrame()).copy()
    regime_performance = run_obj.get('regime_performance', pd.DataFrame()).copy()
    hmm_model_selection = run_obj.get('hmm_model_selection', pd.DataFrame()).copy()
    scoring_calibration = run_obj.get('scoring_calibration', pd.DataFrame()).copy()
    stress_scenarios = run_obj.get('stress_scenarios', pd.DataFrame()).copy()
    walkforward_summary = run_obj.get('walkforward_summary', pd.DataFrame()).copy()
    feature_stability = run_obj.get('feature_stability', pd.DataFrame()).copy()
    robustness_df = run_obj.get('robustness_report', pd.DataFrame()).copy()
    failure_diagnosis = run_obj.get('failure_diagnosis', pd.DataFrame()).copy()
    stat_robustness_summary = run_obj.get('stat_robustness_summary', pd.DataFrame()).copy()
    stat_bootstrap_samples = run_obj.get('stat_bootstrap_samples', pd.DataFrame()).copy()
    parameter_sensitivity = run_obj.get('parameter_sensitivity', pd.DataFrame()).copy()
    feature_drift = run_obj.get('feature_drift', pd.DataFrame()).copy()
    meta_labeling_scores = run_obj.get('meta_labeling_scores', pd.DataFrame()).copy()
    meta_labeling_thresholds = run_obj.get('meta_labeling_thresholds', pd.DataFrame()).copy()
    meta_labeling_report = dict(run_obj.get('meta_labeling_report', {}))
    auto_recommendations = run_obj.get('auto_recommendations', pd.DataFrame()).copy()
    purged_cv = run_obj.get('purged_cv', pd.DataFrame()).copy()
    purged_cv_summary = run_obj.get('purged_cv_summary', pd.DataFrame()).copy()
    overfit_report = dict(run_obj.get('overfit_report', {}))
    execution_impact = run_obj.get('execution_impact', pd.DataFrame()).copy()
    execution_trade_impact = run_obj.get('execution_trade_impact', pd.DataFrame()).copy()
    execution_report = dict(run_obj.get('execution_report', {}))
    trade_root_cause = run_obj.get('trade_root_cause', pd.DataFrame()).copy()
    trade_replay_paths = run_obj.get('trade_replay_paths', pd.DataFrame()).copy()
    trade_root_cause_summary = run_obj.get('trade_root_cause_summary', pd.DataFrame()).copy()
    portfolio_attribution = run_obj.get('portfolio_attribution', pd.DataFrame()).copy()
    portfolio_asset_metrics = run_obj.get('portfolio_asset_metrics', pd.DataFrame()).copy()
    portfolio_correlation = run_obj.get('portfolio_correlation', pd.DataFrame()).copy()
    portfolio_report = dict(run_obj.get('portfolio_report', {}))
    anomaly_scores = run_obj.get('anomaly_scores', pd.DataFrame()).copy()
    trade_anomalies = run_obj.get('trade_anomalies', pd.DataFrame()).copy()
    anomaly_impact = run_obj.get('anomaly_impact', pd.DataFrame()).copy()
    anomaly_report = dict(run_obj.get('anomaly_report', {}))
    model_report = dict(run_obj.get('model_report', {}))
    future_placeholders = create_future_placeholders()
    events_df = run_obj.get('events', future_placeholders['events']).copy()
    regimes_df = run_obj.get('regimes', future_placeholders['regimes']).copy()
    features_ml_df = run_obj.get('features_ml', future_placeholders['features_ml']).copy()
    scoring_df = run_obj.get('scoring', future_placeholders['scoring']).copy()
    _write_parquet_safe(raw_market, data_dir / 'raw_market.parquet')
    _write_parquet_safe(splits['train'], data_dir / 'train.parquet')
    _write_parquet_safe(splits['val'], data_dir / 'val.parquet')
    _write_parquet_safe(splits['test'], data_dir / 'test.parquet')
    _write_parquet_safe(labels, strategy_dir / 'labels.parquet')
    _write_parquet_safe(signals, strategy_dir / 'signals.parquet')
    _write_parquet_safe(bar_ledger, backtest_dir / 'bar_ledger.parquet')
    _write_parquet_safe(trades, backtest_dir / 'trades.parquet')
    _write_parquet_safe(equity_curve, backtest_dir / 'equity_curve.parquet')
    _write_parquet_safe(positions, backtest_dir / 'positions.parquet')
    _write_json(metrics_dir / 'core_metrics.json', core_metrics)
    _write_json(metrics_dir / 'benchmark_metrics.json', benchmark_metrics)
    _write_parquet_safe(time_decomp_flat, metrics_dir / 'time_decomposition.parquet')
    _write_parquet_safe(drawdown, metrics_dir / 'drawdown_series.parquet')
    _write_parquet_safe(split_metrics, metrics_dir / 'split_metrics.parquet')
    _write_parquet_safe(event_impact, metrics_dir / 'event_impact.parquet')
    _write_parquet_safe(regime_performance, metrics_dir / 'regime_performance.parquet')
    _write_parquet_safe(hmm_model_selection, metrics_dir / 'hmm_model_selection.parquet')
    _write_parquet_safe(scoring_calibration, metrics_dir / 'scoring_calibration.parquet')
    _write_parquet_safe(stress_scenarios, metrics_dir / 'stress_scenarios.parquet')
    _write_parquet_safe(walkforward_summary, metrics_dir / 'walkforward_summary.parquet')
    _write_parquet_safe(feature_stability, metrics_dir / 'feature_stability.parquet')
    _write_parquet_safe(robustness_df, metrics_dir / 'robustness_report.parquet')
    _write_parquet_safe(failure_diagnosis, metrics_dir / 'failure_diagnosis.parquet')
    _write_parquet_safe(stat_robustness_summary, metrics_dir / 'stat_robustness_summary.parquet')
    _write_parquet_safe(stat_bootstrap_samples, metrics_dir / 'stat_bootstrap_samples.parquet')
    _write_parquet_safe(parameter_sensitivity, metrics_dir / 'parameter_sensitivity.parquet')
    _write_parquet_safe(feature_drift, metrics_dir / 'feature_drift.parquet')
    _write_parquet_safe(meta_labeling_scores, metrics_dir / 'meta_labeling_scores.parquet')
    _write_parquet_safe(meta_labeling_thresholds, metrics_dir / 'meta_labeling_thresholds.parquet')
    _write_json(metrics_dir / 'meta_labeling_report.json', meta_labeling_report)
    _write_parquet_safe(auto_recommendations, metrics_dir / 'auto_recommendations.parquet')
    _write_parquet_safe(purged_cv, metrics_dir / 'purged_cv.parquet')
    _write_parquet_safe(purged_cv_summary, metrics_dir / 'purged_cv_summary.parquet')
    _write_json(metrics_dir / 'overfit_report.json', overfit_report)
    _write_parquet_safe(execution_impact, metrics_dir / 'execution_impact.parquet')
    _write_parquet_safe(execution_trade_impact, metrics_dir / 'execution_trade_impact.parquet')
    _write_json(metrics_dir / 'execution_report.json', execution_report)
    _write_parquet_safe(trade_root_cause, metrics_dir / 'trade_root_cause.parquet')
    _write_parquet_safe(trade_replay_paths, metrics_dir / 'trade_replay_paths.parquet')
    _write_parquet_safe(trade_root_cause_summary, metrics_dir / 'trade_root_cause_summary.parquet')
    _write_parquet_safe(portfolio_attribution, metrics_dir / 'portfolio_attribution.parquet')
    _write_parquet_safe(portfolio_asset_metrics, metrics_dir / 'portfolio_asset_metrics.parquet')
    _write_parquet_safe(portfolio_correlation, metrics_dir / 'portfolio_correlation.parquet')
    _write_json(metrics_dir / 'portfolio_report.json', portfolio_report)
    _write_parquet_safe(anomaly_scores, metrics_dir / 'anomaly_scores.parquet')
    _write_parquet_safe(trade_anomalies, metrics_dir / 'trade_anomalies.parquet')
    _write_parquet_safe(anomaly_impact, metrics_dir / 'anomaly_impact.parquet')
    _write_json(metrics_dir / 'anomaly_report.json', anomaly_report)
    _write_json(metrics_dir / 'model_report.json', model_report)
    _write_parquet_safe(events_df, future_dir / 'events.parquet')
    _write_parquet_safe(regimes_df, future_dir / 'regimes.parquet')
    _write_parquet_safe(features_ml_df, future_dir / 'features_ml.parquet')
    _write_parquet_safe(scoring_df, future_dir / 'scoring.parquet')
    viz_bundle_all: dict[str, go.Figure] = {}
    viz_bundle_all.update(run_obj.get('viz_bundle', {}))
    viz_bundle_all.update(run_obj.get('viz_bundle_extended', {}))
    if 'report_v2' not in viz_bundle_all:
        viz_bundle_all['report_v2'] = build_v2_consolidated_report(run_obj)
    if 'report_v3' not in viz_bundle_all:
        viz_bundle_all['report_v3'] = build_v3_consolidated_report(run_obj)
    if 'dashboard_main' not in viz_bundle_all:
        fallback = go.Figure()
        fallback.add_annotation(text='Dashboard unavailable', showarrow=False, x=0.5, y=0.5, xref='paper', yref='paper')
        viz_bundle_all['dashboard_main'] = fallback
    for name, fig in viz_bundle_all.items():
        try:
            fig.write_html(viz_dir / f'{name}.html', include_plotlyjs='cdn')
        except Exception:
            pass
    cfg_dict = config_to_dict(config)
    _write_json(run_path / 'config.json', cfg_dict)
    env_payload = {'python': sys.version, 'platform': platform.platform(), 'cwd': str(Path.cwd()), 'artifact_root_resolved': str(artifact_root), 'packages': {'numpy': _package_version('numpy'), 'pandas': _package_version('pandas'), 'plotly': _package_version('plotly'), 'ccxt': _package_version('ccxt'), 'pyarrow': _package_version('pyarrow'), 'hmmlearn': _package_version('hmmlearn'), 'statsmodels': _package_version('statsmodels'), 'scikit-learn': _package_version('scikit-learn'), 'scipy': _package_version('scipy')}}
    _write_json(run_path / 'environment.json', env_payload)
    hashes = {
        'config_hash': hash_object(cfg_dict),
        'raw_market_hash': hash_dataframe(raw_market),
        'train_hash': hash_dataframe(splits['train']),
        'val_hash': hash_dataframe(splits['val']),
        'test_hash': hash_dataframe(splits['test']),
        'labels_hash': hash_dataframe(labels),
        'signals_hash': hash_dataframe(signals),
        'bar_ledger_hash': hash_dataframe(bar_ledger),
        'trades_hash': hash_dataframe(trades),
        'equity_curve_hash': hash_dataframe(equity_curve),
        'positions_hash': hash_dataframe(positions),
        'drawdown_hash': hash_dataframe(drawdown),
        'time_decomposition_hash': hash_dataframe(time_decomp_flat),
        'split_metrics_hash': hash_dataframe(split_metrics),
        'event_impact_hash': hash_dataframe(event_impact),
        'regime_performance_hash': hash_dataframe(regime_performance),
        'hmm_model_selection_hash': hash_dataframe(hmm_model_selection),
        'scoring_calibration_hash': hash_dataframe(scoring_calibration),
        'stress_scenarios_hash': hash_dataframe(stress_scenarios),
        'walkforward_summary_hash': hash_dataframe(walkforward_summary),
        'feature_stability_hash': hash_dataframe(feature_stability),
        'robustness_hash': hash_dataframe(robustness_df),
        'failure_diagnosis_hash': hash_dataframe(failure_diagnosis),
        'stat_robustness_summary_hash': hash_dataframe(stat_robustness_summary),
        'stat_bootstrap_samples_hash': hash_dataframe(stat_bootstrap_samples),
        'parameter_sensitivity_hash': hash_dataframe(parameter_sensitivity),
        'feature_drift_hash': hash_dataframe(feature_drift),
        'meta_labeling_scores_hash': hash_dataframe(meta_labeling_scores),
        'meta_labeling_thresholds_hash': hash_dataframe(meta_labeling_thresholds),
        'meta_labeling_report_hash': hash_object(meta_labeling_report),
        'auto_recommendations_hash': hash_dataframe(auto_recommendations),
        'purged_cv_hash': hash_dataframe(purged_cv),
        'purged_cv_summary_hash': hash_dataframe(purged_cv_summary),
        'overfit_report_hash': hash_object(overfit_report),
        'execution_impact_hash': hash_dataframe(execution_impact),
        'execution_trade_impact_hash': hash_dataframe(execution_trade_impact),
        'execution_report_hash': hash_object(execution_report),
        'trade_root_cause_hash': hash_dataframe(trade_root_cause),
        'trade_replay_paths_hash': hash_dataframe(trade_replay_paths),
        'trade_root_cause_summary_hash': hash_dataframe(trade_root_cause_summary),
        'portfolio_attribution_hash': hash_dataframe(portfolio_attribution),
        'portfolio_asset_metrics_hash': hash_dataframe(portfolio_asset_metrics),
        'portfolio_correlation_hash': hash_dataframe(portfolio_correlation),
        'portfolio_report_hash': hash_object(portfolio_report),
        'anomaly_scores_hash': hash_dataframe(anomaly_scores),
        'trade_anomalies_hash': hash_dataframe(trade_anomalies),
        'anomaly_impact_hash': hash_dataframe(anomaly_impact),
        'anomaly_report_hash': hash_object(anomaly_report),
        'events_hash': hash_dataframe(events_df),
        'regimes_hash': hash_dataframe(regimes_df),
        'features_ml_hash': hash_dataframe(features_ml_df),
        'scoring_hash': hash_dataframe(scoring_df),
        'benchmark_metrics_hash': hash_object(benchmark_metrics),
        'model_report_hash': hash_object(model_report),
        'core_metrics_hash': hash_object(core_metrics),
    }
    row_counts = {
        'raw_market': int(len(raw_market)),
        'train': int(len(splits['train'])),
        'val': int(len(splits['val'])),
        'test': int(len(splits['test'])),
        'labels': int(len(labels)),
        'signals': int(len(signals)),
        'bar_ledger': int(len(bar_ledger)),
        'trades': int(len(trades)),
        'equity_curve': int(len(equity_curve)),
        'positions': int(len(positions)),
        'time_decomposition': int(len(time_decomp_flat)),
        'split_metrics': int(len(split_metrics)),
        'drawdown': int(len(drawdown)),
        'event_impact': int(len(event_impact)),
        'regime_performance': int(len(regime_performance)),
        'hmm_model_selection': int(len(hmm_model_selection)),
        'scoring_calibration': int(len(scoring_calibration)),
        'stress_scenarios': int(len(stress_scenarios)),
        'walkforward_summary': int(len(walkforward_summary)),
        'feature_stability': int(len(feature_stability)),
        'robustness': int(len(robustness_df)),
        'failure_diagnosis': int(len(failure_diagnosis)),
        'stat_robustness_summary': int(len(stat_robustness_summary)),
        'stat_bootstrap_samples': int(len(stat_bootstrap_samples)),
        'parameter_sensitivity': int(len(parameter_sensitivity)),
        'feature_drift': int(len(feature_drift)),
        'meta_labeling_scores': int(len(meta_labeling_scores)),
        'meta_labeling_thresholds': int(len(meta_labeling_thresholds)),
        'auto_recommendations': int(len(auto_recommendations)),
        'purged_cv': int(len(purged_cv)),
        'purged_cv_summary': int(len(purged_cv_summary)),
        'execution_impact': int(len(execution_impact)),
        'execution_trade_impact': int(len(execution_trade_impact)),
        'trade_root_cause': int(len(trade_root_cause)),
        'trade_replay_paths': int(len(trade_replay_paths)),
        'trade_root_cause_summary': int(len(trade_root_cause_summary)),
        'portfolio_attribution': int(len(portfolio_attribution)),
        'portfolio_asset_metrics': int(len(portfolio_asset_metrics)),
        'portfolio_correlation': int(len(portfolio_correlation)),
        'anomaly_scores': int(len(anomaly_scores)),
        'trade_anomalies': int(len(trade_anomalies)),
        'anomaly_impact': int(len(anomaly_impact)),
        'events': int(len(events_df)),
        'regimes': int(len(regimes_df)),
        'features_ml': int(len(features_ml_df)),
        'scoring': int(len(scoring_df)),
    }
    manifest = {'run_id': run_id, 'schema_version': 'v3.x', 'created_at_utc': utc_now_iso(), 'artifact_root': str(artifact_root), 'config': {'symbol': config.symbol, 'timeframe': config.timeframe, 'start': config.start, 'end': config.end, 'split_ratios': list(config.split_ratios), 'portfolio_symbols': list(config.portfolio_symbols)}, 'hashes': hashes, 'row_counts': row_counts, 'required_artifacts': REQUIRED_ARTIFACTS_EXTENDED}
    _write_json(run_path / 'manifest.json', manifest)
    strategy_tag = f'ema{config.ema_fast}_{config.ema_slow}_sl{config.stop_loss_pct}_tp{config.take_profit_pct}'
    catalog_row = {'run_id': run_id, 'created_at_utc': manifest['created_at_utc'], 'run_path': str(run_path), 'symbol': config.symbol, 'timeframe': config.timeframe, 'start': config.start, 'end': config.end, 'strategy_tag': strategy_tag, 'config_hash': hashes['config_hash'], 'raw_market_hash': hashes['raw_market_hash'], 'train_hash': hashes['train_hash'], 'total_pnl': core_metrics.get('total_pnl', np.nan), 'cumulative_return': core_metrics.get('cumulative_return', np.nan), 'sharpe_ratio': core_metrics.get('sharpe_ratio', np.nan), 'max_drawdown': core_metrics.get('max_drawdown', np.nan), 'win_rate': core_metrics.get('win_rate', np.nan), 'profit_factor': core_metrics.get('profit_factor', np.nan), 'total_trades': core_metrics.get('total_trades', np.nan), 'notes': config.notes}
    _upsert_run_catalog(artifact_root / 'run_catalog.parquet', catalog_row)
    return run_path

def load_run(run_path: str | Path) -> dict[str, Any]:
    run_path = Path(run_path)
    if not run_path.exists():
        raise FileNotFoundError(run_path)
    future_placeholders = create_future_placeholders()
    out = {'manifest': _read_json(run_path / 'manifest.json'), 'config': _read_json(run_path / 'config.json'), 'environment': _read_json(run_path / 'environment.json'), 'raw_market': _read_parquet_safe(run_path / 'data/raw_market.parquet'), 'splits': {'train': _read_parquet_safe(run_path / 'data/train.parquet'), 'val': _read_parquet_safe(run_path / 'data/val.parquet'), 'test': _read_parquet_safe(run_path / 'data/test.parquet')}, 'labels': _read_parquet_safe(run_path / 'strategy/labels.parquet'), 'signals': _read_parquet_safe(run_path / 'strategy/signals.parquet'), 'bar_ledger': _read_parquet_safe(run_path / 'backtest/bar_ledger.parquet'), 'trades': _read_parquet_safe(run_path / 'backtest/trades.parquet'), 'equity_curve': _read_parquet_safe(run_path / 'backtest/equity_curve.parquet'), 'positions': _read_parquet_safe(run_path / 'backtest/positions.parquet'), 'core_metrics': _read_json(run_path / 'metrics/core_metrics.json'), 'benchmark_metrics': _read_json(run_path / 'metrics/benchmark_metrics.json') if (run_path / 'metrics/benchmark_metrics.json').exists() else {}, 'time_decomposition_flat': _read_parquet_safe(run_path / 'metrics/time_decomposition.parquet'), 'drawdown_series': _read_parquet_safe(run_path / 'metrics/drawdown_series.parquet'), 'split_metrics': _load_optional_parquet(run_path / 'metrics/split_metrics.parquet'), 'event_impact': _load_optional_parquet(run_path / 'metrics/event_impact.parquet'), 'regime_performance': _load_optional_parquet(run_path / 'metrics/regime_performance.parquet'), 'hmm_model_selection': _load_optional_parquet(run_path / 'metrics/hmm_model_selection.parquet'), 'scoring_calibration': _load_optional_parquet(run_path / 'metrics/scoring_calibration.parquet'), 'stress_scenarios': _load_optional_parquet(run_path / 'metrics/stress_scenarios.parquet'), 'walkforward_summary': _load_optional_parquet(run_path / 'metrics/walkforward_summary.parquet'), 'feature_stability': _load_optional_parquet(run_path / 'metrics/feature_stability.parquet'), 'robustness_report': _load_optional_parquet(run_path / 'metrics/robustness_report.parquet'), 'failure_diagnosis': _load_optional_parquet(run_path / 'metrics/failure_diagnosis.parquet'), 'stat_robustness_summary': _load_optional_parquet(run_path / 'metrics/stat_robustness_summary.parquet'), 'stat_bootstrap_samples': _load_optional_parquet(run_path / 'metrics/stat_bootstrap_samples.parquet'), 'parameter_sensitivity': _load_optional_parquet(run_path / 'metrics/parameter_sensitivity.parquet'), 'feature_drift': _load_optional_parquet(run_path / 'metrics/feature_drift.parquet'), 'meta_labeling_scores': _load_optional_parquet(run_path / 'metrics/meta_labeling_scores.parquet'), 'meta_labeling_thresholds': _load_optional_parquet(run_path / 'metrics/meta_labeling_thresholds.parquet'), 'meta_labeling_report': _read_json(run_path / 'metrics/meta_labeling_report.json') if (run_path / 'metrics/meta_labeling_report.json').exists() else {}, 'auto_recommendations': _load_optional_parquet(run_path / 'metrics/auto_recommendations.parquet'), 'purged_cv': _load_optional_parquet(run_path / 'metrics/purged_cv.parquet'), 'purged_cv_summary': _load_optional_parquet(run_path / 'metrics/purged_cv_summary.parquet'), 'overfit_report': _read_json(run_path / 'metrics/overfit_report.json') if (run_path / 'metrics/overfit_report.json').exists() else {}, 'execution_impact': _load_optional_parquet(run_path / 'metrics/execution_impact.parquet'), 'execution_trade_impact': _load_optional_parquet(run_path / 'metrics/execution_trade_impact.parquet'), 'execution_report': _read_json(run_path / 'metrics/execution_report.json') if (run_path / 'metrics/execution_report.json').exists() else {}, 'trade_root_cause': _load_optional_parquet(run_path / 'metrics/trade_root_cause.parquet'), 'trade_replay_paths': _load_optional_parquet(run_path / 'metrics/trade_replay_paths.parquet'), 'trade_root_cause_summary': _load_optional_parquet(run_path / 'metrics/trade_root_cause_summary.parquet'), 'portfolio_attribution': _load_optional_parquet(run_path / 'metrics/portfolio_attribution.parquet'), 'portfolio_asset_metrics': _load_optional_parquet(run_path / 'metrics/portfolio_asset_metrics.parquet'), 'portfolio_correlation': _load_optional_parquet(run_path / 'metrics/portfolio_correlation.parquet'), 'portfolio_report': _read_json(run_path / 'metrics/portfolio_report.json') if (run_path / 'metrics/portfolio_report.json').exists() else {}, 'anomaly_scores': _load_optional_parquet(run_path / 'metrics/anomaly_scores.parquet'), 'trade_anomalies': _load_optional_parquet(run_path / 'metrics/trade_anomalies.parquet'), 'anomaly_impact': _load_optional_parquet(run_path / 'metrics/anomaly_impact.parquet'), 'anomaly_report': _read_json(run_path / 'metrics/anomaly_report.json') if (run_path / 'metrics/anomaly_report.json').exists() else {}, 'model_report': _read_json(run_path / 'metrics/model_report.json') if (run_path / 'metrics/model_report.json').exists() else {}, 'future': {'events': _load_optional_parquet(run_path / 'future/events.parquet', list(future_placeholders['events'].columns)), 'regimes': _load_optional_parquet(run_path / 'future/regimes.parquet', list(future_placeholders['regimes'].columns)), 'features_ml': _load_optional_parquet(run_path / 'future/features_ml.parquet', list(future_placeholders['features_ml'].columns)), 'scoring': _load_optional_parquet(run_path / 'future/scoring.parquet', list(future_placeholders['scoring'].columns))}}
    return out

def _make_empty_figure(title: str, text: str='No data') -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=text, showarrow=False, x=0.5, y=0.5, xref='paper', yref='paper')
    fig.update_layout(title=title)
    return fig

def build_main_dashboard(run_obj: dict[str, Any], compare_df: pd.DataFrame | None=None) -> dict[str, go.Figure]:
    core = run_obj.get('core_metrics', {})
    equity = run_obj.get('equity_curve', pd.DataFrame())
    drawdown = run_obj.get('drawdown_series', pd.DataFrame())
    trades = run_obj.get('trades', pd.DataFrame())
    time_decomp = run_obj.get('time_decomposition', {})
    kpi_rows = [('PnL total', _safe_metric_display(core, 'total_pnl', '.2f')), ('Rendement cumule', _safe_metric_display(core, 'cumulative_return', '.2%')), ('Sharpe ratio', _safe_metric_display(core, 'sharpe_ratio', '.3f')), ('Max drawdown', _safe_metric_display(core, 'max_drawdown', '.2%')), ('Win rate', _safe_metric_display(core, 'win_rate', '.2%')), ('Profit factor', _safe_metric_display(core, 'profit_factor', '.3f')), ('Expectancy', _safe_metric_display(core, 'expectancy', '.3f')), ('Trades', _safe_metric_display(core, 'total_trades')), ('Ratio gains/pertes', _safe_metric_display(core, 'gain_loss_ratio', '.3f')), ('Duree moyenne (h)', _safe_metric_display(core, 'avg_trade_duration_hours', '.2f')), ('Exposition', _safe_metric_display(core, 'exposure', '.2%'))]
    kpi_fig = go.Figure(data=[go.Table(header=dict(values=['Metric', 'Valeur'], fill_color='#0f172a', font=dict(color='white')), cells=dict(values=[[x[0] for x in kpi_rows], [x[1] for x in kpi_rows]], fill_color='#f8fafc'))])
    kpi_fig.update_layout(title='Cartes KPI')
    dashboard = make_subplots(rows=2, cols=2, subplot_titles=('Equity', 'Drawdown', 'Distribution PnL', 'Heatmap Jour/Heure'), specs=[[{'type': 'xy'}, {'type': 'xy'}], [{'type': 'xy'}, {'type': 'heatmap'}]])
    if not equity.empty:
        dashboard.add_trace(go.Scatter(x=equity.index, y=equity['equity'], mode='lines', name='Equity'), row=1, col=1)
    else:
        dashboard.add_annotation(text='No equity data', showarrow=False, x=0.2, y=0.8, xref='paper', yref='paper')
    if not drawdown.empty:
        dashboard.add_trace(go.Scatter(x=drawdown.index, y=drawdown['drawdown'] * 100.0, mode='lines', fill='tozeroy', name='Drawdown %'), row=1, col=2)
    if not trades.empty and 'net_pnl' in trades.columns:
        dashboard.add_trace(go.Histogram(x=trades['net_pnl'], nbinsx=40, name='Trade PnL'), row=2, col=1)
    if isinstance(time_decomp, dict) and 'base_returns' in time_decomp and (not time_decomp['base_returns'].empty):
        base = time_decomp['base_returns']
        heat = base.pivot_table(index=base.index.dayofweek, columns=base.index.hour, values='return', aggfunc='mean')
        heat = heat.reindex(index=list(range(7)), columns=list(range(24)))
        day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dashboard.add_trace(go.Heatmap(z=heat.values * 100.0, x=list(heat.columns), y=[day_labels[d] for d in heat.index], colorscale='RdYlGn', zmid=0, colorbar=dict(title='Mean %'), showscale=True), row=2, col=2)
    dashboard.update_layout(height=900, title='Dashboard Principal')
    comparison_fig = _make_empty_figure('Comparaison Runs', 'Aucune comparaison fournie')
    if compare_df is not None and (not compare_df.empty):
        cols = [c for c in ['run_id', 'symbol', 'timeframe', 'cumulative_return', 'sharpe_ratio', 'max_drawdown', 'total_trades'] if c in compare_df.columns]
        if cols:
            tmp = compare_df[cols].copy()
            comparison_fig = go.Figure(data=[go.Table(header=dict(values=cols, fill_color='#111827', font=dict(color='white')), cells=dict(values=[tmp[c] for c in cols], fill_color='#f8fafc'))])
            comparison_fig.update_layout(title='Comparaison Structuree des Runs')
    return {'dashboard_main': dashboard, 'dashboard_kpis': kpi_fig, 'comparison_runs': comparison_fig}

def build_comparison_views(run_catalog: pd.DataFrame, run_ids: list[str]) -> dict[str, go.Figure]:
    if run_catalog is None or run_catalog.empty:
        return {'comparison_runs': _make_empty_figure('Comparaison Runs', 'Run catalog vide'), 'comparison_scatter': _make_empty_figure('Comparaison Scatter', 'Run catalog vide'), 'comparison_radar': _make_empty_figure('Comparaison Radar', 'Run catalog vide')}
    subset = run_catalog[run_catalog['run_id'].astype(str).isin([str(x) for x in run_ids])].copy()
    if subset.empty:
        subset = run_catalog.sort_values('created_at_utc').tail(min(5, len(run_catalog))).copy()
    table_cols = [c for c in ['run_id', 'symbol', 'timeframe', 'strategy_tag', 'cumulative_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades'] if c in subset.columns]
    table_fig = go.Figure(data=[go.Table(header=dict(values=table_cols, fill_color='#0f172a', font=dict(color='white')), cells=dict(values=[subset[c] for c in table_cols], fill_color='#f8fafc'))])
    table_fig.update_layout(title='Comparaison Runs')
    if {'max_drawdown', 'cumulative_return', 'total_trades'}.issubset(subset.columns):
        scatter_fig = px.scatter(subset, x='max_drawdown', y='cumulative_return', size='total_trades', color='symbol' if 'symbol' in subset.columns else None, hover_name='run_id', title='Runs: risque vs rendement')
    else:
        scatter_fig = _make_empty_figure('Comparaison Scatter', 'Colonnes insuffisantes')
    radar_metrics = [m for m in ['cumulative_return', 'sharpe_ratio', 'win_rate', 'profit_factor'] if m in subset.columns]
    if len(radar_metrics) >= 3 and len(subset) > 0:
        radar_fig = go.Figure()
        norm = subset[radar_metrics].copy()
        for c in radar_metrics:
            col = pd.to_numeric(norm[c], errors='coerce')
            span = col.max() - col.min()
            if pd.notna(span) and span > 0:
                norm[c] = (col - col.min()) / span
            else:
                norm[c] = 0.0
        for _, row in norm.iterrows():
            radar_fig.add_trace(go.Scatterpolar(r=[float(row[m]) for m in radar_metrics], theta=radar_metrics, fill='toself', name=str(row.get('run_id', 'run'))))
        radar_fig.update_layout(title='Radar Metriques (normalisees)', polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    else:
        radar_fig = _make_empty_figure('Comparaison Radar', 'Pas assez de metriques comparables')
    return {'comparison_runs': table_fig, 'comparison_scatter': scatter_fig, 'comparison_radar': radar_fig}

def build_trade_enriched_table(trades: pd.DataFrame, bar_ledger: pd.DataFrame, labels: pd.DataFrame, market_df: pd.DataFrame, regimes: pd.DataFrame | None=None, trade_events: pd.DataFrame | None=None) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    out = trades.copy()
    out = _ensure_datetime_utc(out, ['entry_time', 'exit_time'])
    mkt = market_df[['close', 'volume']].copy()
    mkt['ret_1h'] = mkt['close'].pct_change().fillna(0.0)
    mkt['vol_24h'] = mkt['ret_1h'].rolling(24, min_periods=5).std()
    mkt['ret_24h'] = mkt['close'].pct_change(24)
    if {'ema_fast', 'ema_slow', 'close'}.issubset(labels.columns):
        ema_spread = (labels['ema_fast'] - labels['ema_slow']) / labels['close'].replace(0.0, np.nan)
    else:
        ema_spread = pd.Series(np.nan, index=labels.index)
    entry_idx = out['entry_time']
    exit_idx = out['exit_time']
    out['entry_hour'] = entry_idx.dt.hour
    out['entry_day_name'] = entry_idx.dt.day_name()
    out['entry_day_of_week'] = entry_idx.dt.dayofweek
    out['entry_ema_spread'] = ema_spread.reindex(entry_idx).to_numpy()
    out['entry_vol_24h'] = mkt['vol_24h'].reindex(entry_idx).to_numpy()
    out['entry_ret_24h'] = mkt['ret_24h'].reindex(entry_idx).to_numpy()
    out['entry_close'] = mkt['close'].reindex(entry_idx).to_numpy()
    out['exit_close'] = mkt['close'].reindex(exit_idx).to_numpy()
    out['market_move_pct'] = out['exit_close'] / out['entry_close'] - 1.0
    if bar_ledger is not None and (not bar_ledger.empty):
        pos = bar_ledger[['position_close']].copy()
        out['entry_position_context'] = pos['position_close'].reindex(entry_idx).to_numpy()
    else:
        out['entry_position_context'] = np.nan
    out['entry_regime'] = pd.NA
    if regimes is not None and (not regimes.empty):
        reg = regimes.copy().reset_index(drop=False)
        if 'timestamp' not in reg.columns:
            reg = reg.rename(columns={reg.columns[0]: 'timestamp'})
        reg['timestamp'] = _to_datetime_utc_ns(reg['timestamp'])
        reg = reg.sort_values('timestamp').dropna(subset=['timestamp'])
        entry_key = out[['trade_id', 'entry_time']].copy()
        entry_key['entry_time'] = _to_datetime_utc_ns(entry_key['entry_time'])
        entry_map = pd.merge_asof(
            entry_key.sort_values('entry_time'),
            reg[['timestamp', 'regime_label']].sort_values('timestamp'),
            left_on='entry_time',
            right_on='timestamp',
            direction='backward',
        )
        entry_map = entry_map.set_index('trade_id')['regime_label']
        out['entry_regime'] = out['trade_id'].map(entry_map)
    out['event_count_near_trade'] = 0
    out['event_types_near_trade'] = ''
    if trade_events is not None and (not trade_events.empty) and ('trade_id' in trade_events.columns):
        g = trade_events.groupby('trade_id')
        out['event_count_near_trade'] = out['trade_id'].map(g.size()).fillna(0).astype(int)
        if 'event_type' in trade_events.columns:
            types_map = g['event_type'].apply(lambda s: ','.join(sorted(set((str(x) for x in s if pd.notna(x))))))
            out['event_types_near_trade'] = out['trade_id'].map(types_map).fillna('')
    out['winner_loser'] = np.where(out['is_winner'].astype(bool), 'winner', 'loser')
    return out

def analyze_winners_vs_losers(trade_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if trade_df is None or trade_df.empty:
        return {'summary': pd.DataFrame(), 'tests': pd.DataFrame(), 'patterns': pd.DataFrame()}
    metrics = [c for c in ['net_pnl', 'return_pct', 'duration_hours', 'entry_ema_spread', 'entry_vol_24h', 'entry_ret_24h', 'market_move_pct'] if c in trade_df.columns]
    summary = trade_df.groupby('winner_loser')[metrics].agg(['mean', 'median', 'std', 'count']).reset_index()
    tests_rows = []
    if mannwhitneyu is not None and ks_2samp is not None:
        for col in metrics:
            wins = pd.to_numeric(trade_df.loc[trade_df['winner_loser'] == 'winner', col], errors='coerce').dropna()
            losses = pd.to_numeric(trade_df.loc[trade_df['winner_loser'] == 'loser', col], errors='coerce').dropna()
            if len(wins) < 3 or len(losses) < 3:
                continue
            mw = mannwhitneyu(wins, losses, alternative='two-sided')
            ks = ks_2samp(wins, losses)
            tests_rows.append({'feature': col, 'wins_mean': float(wins.mean()), 'losses_mean': float(losses.mean()), 'mw_stat': float(mw.statistic), 'mw_pvalue': float(mw.pvalue), 'ks_stat': float(ks.statistic), 'ks_pvalue': float(ks.pvalue)})
    tests = pd.DataFrame(tests_rows)
    baseline_wr = float((trade_df['winner_loser'] == 'winner').mean())
    pattern_rows = []
    for col in [c for c in ['entry_hour', 'entry_day_name', 'entry_regime', 'event_count_near_trade', 'anomaly_count_near_trade', 'has_anomaly_near_trade', 'anomaly_top_feature'] if c in trade_df.columns]:
        grp = trade_df.groupby(col).agg(trades=('trade_id', 'count'), win_rate=('is_winner', 'mean'), expectancy=('net_pnl', 'mean'), median_pnl=('net_pnl', 'median')).reset_index()
        grp['lift_vs_baseline'] = grp['win_rate'] - baseline_wr
        grp['feature'] = col
        pattern_rows.append(grp)
    patterns = pd.concat(pattern_rows, ignore_index=True) if pattern_rows else pd.DataFrame()
    if not patterns.empty:
        patterns = patterns.sort_values(['trades', 'lift_vs_baseline'], ascending=[False, False]).reset_index(drop=True)
    return {'summary': summary, 'tests': tests, 'patterns': patterns}

def plot_trade_explorer(trade_df: pd.DataFrame) -> dict[str, go.Figure]:
    if trade_df is None or trade_df.empty:
        return {'trade_timeline': _make_empty_figure('Timeline Trades', 'No trades'), 'trade_scatter_duration_return': _make_empty_figure('Duree vs Rendement', 'No trades'), 'trade_box_winners_losers': _make_empty_figure('Winners vs Losers', 'No trades')}
    timeline = px.scatter(trade_df, x='entry_time', y='net_pnl', color='winner_loser', hover_data=['trade_id', 'duration_hours', 'return_pct', 'entry_regime'] if 'entry_regime' in trade_df.columns else ['trade_id', 'duration_hours', 'return_pct'], title='Timeline des trades')
    plot_df = trade_df.copy()
    plot_df['abs_net_pnl'] = plot_df['net_pnl'].abs()
    scatter = px.scatter(plot_df, x='duration_hours', y='return_pct', color='winner_loser', size='abs_net_pnl', hover_data=['trade_id', 'entry_hour', 'entry_day_name'], title='Duree vs Rendement trade')
    box = px.box(trade_df, x='winner_loser', y='net_pnl', color='winner_loser', points='all', title='Distribution net PnL: winners vs losers')
    return {'trade_timeline': timeline, 'trade_scatter_duration_return': scatter, 'trade_box_winners_losers': box}

def load_events_csv(path: str | Path) -> pd.DataFrame:
    schema = EventSchemaV1()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    events = pd.read_csv(path)
    missing = [c for c in schema.required_columns if c not in events.columns]
    if missing:
        raise ValueError(f'Events CSV missing columns: {missing}')
    events = events[list(schema.required_columns)].copy()
    events['timestamp'] = pd.to_datetime(events['timestamp'], utc=True, errors='coerce')
    if events['timestamp'].isna().any():
        bad_rows = events[events['timestamp'].isna()]
        raise ValueError(f'Invalid timestamp rows in events CSV: {len(bad_rows)}')
    events['event_type'] = events['event_type'].astype('string').str.strip()
    events['event_name'] = events['event_name'].astype('string').str.strip()
    events['asset'] = events['asset'].astype('string').str.strip().str.upper()
    events['importance'] = pd.to_numeric(events['importance'], errors='coerce').fillna(0.0)
    events['source'] = events['source'].astype('string').str.strip()
    events['notes'] = events['notes'].astype('string')
    events = events.sort_values('timestamp').reset_index(drop=True)
    return events

def attach_events_to_trades(trades: pd.DataFrame, events: pd.DataFrame, pre_window_h: int=6, post_window_h: int=6, asset: str | None=None) -> pd.DataFrame:
    if trades is None or trades.empty or events is None or events.empty:
        return pd.DataFrame(columns=['trade_id', 'entry_time', 'exit_time', 'event_timestamp', 'event_type', 'event_name', 'asset', 'importance', 'hours_from_entry', 'hours_to_exit', 'in_trade_window'])
    tdf = trades.copy()
    tdf = _ensure_datetime_utc(tdf, ['entry_time', 'exit_time'])
    edf = events.copy()
    edf = _ensure_datetime_utc(edf, ['timestamp'])
    if asset is not None and 'asset' in edf.columns:
        normalized_asset = str(asset).replace('/', '').upper()
        mask = edf['asset'].astype(str).str.replace('/', '', regex=False).str.upper().isin([normalized_asset, 'ALL', 'CRYPTO'])
        edf = edf[mask].copy()
    links = []
    pre_delta = pd.Timedelta(hours=int(pre_window_h))
    post_delta = pd.Timedelta(hours=int(post_window_h))
    for _, tr in tdf.iterrows():
        start = tr['entry_time'] - pre_delta
        end = tr['exit_time'] + post_delta
        local_events = edf[(edf['timestamp'] >= start) & (edf['timestamp'] <= end)]
        if local_events.empty:
            continue
        for _, ev in local_events.iterrows():
            entry_gap_h = (ev['timestamp'] - tr['entry_time']).total_seconds() / 3600.0
            exit_gap_h = (tr['exit_time'] - ev['timestamp']).total_seconds() / 3600.0
            links.append({'trade_id': int(tr['trade_id']), 'entry_time': tr['entry_time'], 'exit_time': tr['exit_time'], 'event_timestamp': ev['timestamp'], 'event_type': ev.get('event_type', pd.NA), 'event_name': ev.get('event_name', pd.NA), 'asset': ev.get('asset', pd.NA), 'importance': ev.get('importance', np.nan), 'hours_from_entry': float(entry_gap_h), 'hours_to_exit': float(exit_gap_h), 'in_trade_window': bool(ev['timestamp'] >= tr['entry_time'] and ev['timestamp'] <= tr['exit_time'])})
    return pd.DataFrame(links)

def event_impact_report(trades: pd.DataFrame, trade_events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if trades is None or trades.empty:
        empty = pd.DataFrame()
        return {'with_vs_without': empty, 'by_event_type': empty}
    tdf = trades.copy()
    tdf = _ensure_datetime_utc(tdf, ['entry_time', 'exit_time'])
    if trade_events is None or trade_events.empty:
        tdf['has_event'] = False
    else:
        ids_with_event = set(trade_events['trade_id'].astype(int).tolist())
        tdf['has_event'] = tdf['trade_id'].astype(int).isin(ids_with_event)
    with_vs_without = tdf.groupby('has_event').agg(trades=('trade_id', 'count'), win_rate=('is_winner', 'mean'), avg_pnl=('net_pnl', 'mean'), median_pnl=('net_pnl', 'median'), expectancy=('net_pnl', 'mean')).reset_index()
    by_event_type = pd.DataFrame()
    if trade_events is not None and (not trade_events.empty) and ('event_type' in trade_events.columns):
        merged = tdf[['trade_id', 'is_winner', 'net_pnl']].merge(trade_events[['trade_id', 'event_type', 'importance']], on='trade_id', how='inner')
        by_event_type = merged.groupby('event_type').agg(linked_trades=('trade_id', 'nunique'), win_rate=('is_winner', 'mean'), avg_pnl=('net_pnl', 'mean'), median_pnl=('net_pnl', 'median'), avg_importance=('importance', 'mean')).reset_index().sort_values('linked_trades', ascending=False).reset_index(drop=True)
    return {'with_vs_without': with_vs_without, 'by_event_type': by_event_type}

def _build_event_impact_viz(report: dict[str, pd.DataFrame]) -> go.Figure:
    by_type = report.get('by_event_type', pd.DataFrame())
    if by_type.empty:
        return _make_empty_figure('Impact Events', 'No event links')
    fig = px.bar(by_type, x='event_type', y='avg_pnl', color='win_rate', title="Impact moyen par type d'evenement")
    return fig

def compute_regime_features(market_df: pd.DataFrame, regime_config: RegimeModelConfig | None=None) -> pd.DataFrame:
    rcfg = regime_config or RegimeModelConfig()
    if market_df is None or market_df.empty:
        return pd.DataFrame()
    close = market_df['close'].astype(float)
    volume = market_df['volume'].astype(float) if 'volume' in market_df.columns else pd.Series(1.0, index=market_df.index)
    ret_1 = np.log(close).diff(rcfg.return_lookback)
    vol_roll = ret_1.rolling(rcfg.vol_window, min_periods=max(5, rcfg.vol_window // 2)).std()
    ema_fast = close.ewm(span=rcfg.trend_fast, adjust=False).mean()
    ema_slow = close.ewm(span=rcfg.trend_slow, adjust=False).mean()
    trend_proxy = (ema_fast - ema_slow) / close.replace(0.0, np.nan)
    vol_mean = volume.rolling(rcfg.volume_window, min_periods=max(5, rcfg.volume_window // 2)).mean()
    vol_std = volume.rolling(rcfg.volume_window, min_periods=max(5, rcfg.volume_window // 2)).std()
    volume_z = (volume - vol_mean) / vol_std.replace(0.0, np.nan)
    features = pd.DataFrame({'ret_1': ret_1, 'vol_rolling': vol_roll, 'trend_proxy': trend_proxy, 'volume_zscore': volume_z}, index=market_df.index)
    return features.replace([np.inf, -np.inf], np.nan)

def _label_hmm_states(state_stats: pd.DataFrame) -> dict[int, str]:
    if state_stats.empty:
        return {}
    mapping: dict[int, str] = {}
    stats = state_stats.copy()
    vol_order = stats['vol_rolling'].sort_values(ascending=False).index.tolist()
    trend_order = stats['trend_proxy'].sort_values(ascending=False).index.tolist()
    ret_order = stats['ret_1'].sort_values(ascending=True).index.tolist()
    if vol_order:
        mapping[int(vol_order[0])] = 'high_vol'
    if trend_order:
        mapping[int(trend_order[0])] = 'trend'
    if ret_order:
        mapping.setdefault(int(ret_order[0]), 'risk_off')
    for st in stats.index.tolist():
        mapping.setdefault(int(st), 'range')
    return mapping

def fit_hmm_regimes(train_features: pd.DataFrame, n_states: int=4, regime_config: RegimeModelConfig | None=None) -> dict[str, Any]:
    rcfg = regime_config or RegimeModelConfig(n_states=n_states)
    fcols = ['ret_1', 'vol_rolling', 'trend_proxy', 'volume_zscore']
    feat = train_features[fcols].copy().replace([np.inf, -np.inf], np.nan)
    feat_clean = feat.dropna().copy()
    if feat_clean.empty or GaussianHMM is None or len(feat_clean) < max(30, rcfg.n_states * 10):
        proxy = feat['vol_rolling'].copy()
        if proxy.notna().sum() == 0:
            proxy = pd.Series(0.0, index=feat.index)
        else:
            proxy = proxy.fillna(proxy.median())
        max_bins = max(2, min(rcfg.n_states, int(proxy.nunique()) if proxy.nunique() > 0 else 2))
        quantiles = np.linspace(0.0, 1.0, max_bins + 1)
        edges = np.unique(np.quantile(proxy.to_numpy(dtype=float), quantiles))
        if len(edges) <= 2:
            edges = np.array([float(proxy.min()) - 1e-12, float(proxy.max()) + 1e-12])
        states = np.digitize(proxy.to_numpy(dtype=float), edges[1:-1], right=True).astype(int)
        feat_state = feat.copy()
        feat_state['regime_state'] = states
        state_stats = feat_state.groupby('regime_state')[fcols].mean().replace([np.inf, -np.inf], np.nan).fillna(0.0).sort_index()
        state_mapping = _label_hmm_states(state_stats)
        return {'model': None, 'scaler': None, 'state_stats': state_stats, 'state_mapping': state_mapping, 'feature_columns': fcols, 'config': rcfg, 'fallback_edges': edges, 'fallback_mode': 'vol_quantile'}
    X = feat_clean.to_numpy(dtype=float)
    scaler = None
    if StandardScaler is not None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    model = GaussianHMM(n_components=rcfg.n_states, covariance_type=rcfg.covariance_type, n_iter=rcfg.n_iter, random_state=rcfg.random_state)
    model.fit(X)
    states = model.predict(X)
    feat_state = feat_clean.copy()
    feat_state['regime_state'] = states
    state_stats = feat_state.groupby('regime_state')[fcols].mean().replace([np.inf, -np.inf], np.nan).fillna(0.0).sort_index()
    state_mapping = _label_hmm_states(state_stats)
    return {'model': model, 'scaler': scaler, 'state_stats': state_stats, 'state_mapping': state_mapping, 'feature_columns': fcols, 'config': rcfg, 'fallback_edges': None, 'fallback_mode': None}

def predict_regimes(features: pd.DataFrame, regime_model: dict[str, Any]) -> pd.DataFrame:
    model = regime_model.get('model')
    scaler = regime_model.get('scaler')
    fcols = regime_model.get('feature_columns', ['ret_1', 'vol_rolling', 'trend_proxy', 'volume_zscore'])
    state_mapping = regime_model.get('state_mapping', {})
    feat = features[fcols].copy().replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame(index=features.index)
    out['regime_state'] = pd.NA
    out['confidence'] = np.nan
    out['regime_label'] = pd.NA
    if model is None:
        proxy = feat['vol_rolling'].copy() if 'vol_rolling' in feat.columns else pd.Series(0.0, index=feat.index)
        if proxy.notna().sum() == 0:
            proxy = pd.Series(0.0, index=feat.index)
        else:
            proxy = proxy.fillna(proxy.median())
        edges = regime_model.get('fallback_edges')
        if edges is None:
            states = np.zeros(len(proxy), dtype=int)
        else:
            edges_arr = np.asarray(edges, dtype=float)
            states = np.digitize(proxy.to_numpy(dtype=float), edges_arr[1:-1], right=True).astype(int)
        out['regime_state'] = states
        out['confidence'] = 1.0
        out['regime_label'] = out['regime_state'].map(lambda x: state_mapping.get(int(x), 'range'))
        merged = out.join(features, how='left').reset_index()
        if 'timestamp' not in merged.columns and len(merged.columns) > 0:
            merged = merged.rename(columns={merged.columns[0]: 'timestamp'})
        merged['timestamp'] = _to_datetime_utc_ns(merged['timestamp'])
        return merged
    valid = feat.dropna().copy()
    if valid.empty:
        out['regime_state'] = 0
        out['confidence'] = 0.0
        out['regime_label'] = 'range'
        merged = out.join(features, how='left').reset_index()
        if 'timestamp' not in merged.columns and len(merged.columns) > 0:
            merged = merged.rename(columns={merged.columns[0]: 'timestamp'})
        merged['timestamp'] = _to_datetime_utc_ns(merged['timestamp'])
        return merged
    X = valid.to_numpy(dtype=float)
    if scaler is not None:
        X = scaler.transform(X)
    states = model.predict(X)
    try:
        probs = model.predict_proba(X)
        conf = probs.max(axis=1)
    except Exception:
        conf = np.full(shape=len(states), fill_value=np.nan)
    out.loc[valid.index, 'regime_state'] = states.astype(int)
    out.loc[valid.index, 'confidence'] = conf
    out['regime_state'] = out['regime_state'].ffill().bfill().fillna(0).astype(int)
    out['confidence'] = out['confidence'].fillna(out['confidence'].median() if out['confidence'].notna().any() else 0.0)
    out['regime_label'] = out['regime_state'].map(lambda x: state_mapping.get(int(x), 'range'))
    merged = out.join(features, how='left').reset_index()
    if 'timestamp' not in merged.columns and len(merged.columns) > 0:
        merged = merged.rename(columns={merged.columns[0]: 'timestamp'})
    merged['timestamp'] = _to_datetime_utc_ns(merged['timestamp'])
    return merged

def regime_performance_report(trades: pd.DataFrame, equity: pd.DataFrame, regimes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if regimes is None or regimes.empty:
        return {'bars_by_regime': pd.DataFrame(), 'trades_by_regime': pd.DataFrame()}
    reg = regimes.copy()
    if 'timestamp' not in reg.columns:
        reg = reg.reset_index(drop=False)
        if 'timestamp' not in reg.columns and len(reg.columns) > 0:
            reg = reg.rename(columns={reg.columns[0]: 'timestamp'})
    reg['timestamp'] = _to_datetime_utc_ns(reg['timestamp'])
    reg = reg.dropna(subset=['timestamp']).sort_values('timestamp')
    bars_by_regime = pd.DataFrame()
    if equity is not None and (not equity.empty):
        eq = equity.copy()
        eq['bar_return'] = eq['equity'].pct_change().fillna(0.0)
        eq = eq.reset_index().rename(columns={'index': 'timestamp'})
        eq['timestamp'] = _to_datetime_utc_ns(eq['timestamp'])
        eq = pd.merge_asof(eq.sort_values('timestamp'), reg[['timestamp', 'regime_label']], on='timestamp', direction='backward')
        bars_by_regime = eq.groupby('regime_label').agg(bars=('timestamp', 'count'), mean_bar_return=('bar_return', 'mean'), cumulative_return=('bar_return', lambda s: float((1.0 + s).prod() - 1.0))).reset_index().sort_values('bars', ascending=False).reset_index(drop=True)
    trades_by_regime = pd.DataFrame()
    if trades is not None and (not trades.empty):
        tdf = trades.copy()
        tdf = _ensure_datetime_utc(tdf, ['entry_time'])
        tdf['entry_time'] = _to_datetime_utc_ns(tdf['entry_time'])
        tmp = pd.merge_asof(tdf.sort_values('entry_time'), reg[['timestamp', 'regime_label']].sort_values('timestamp'), left_on='entry_time', right_on='timestamp', direction='backward')
        trades_by_regime = tmp.groupby('regime_label').agg(trades=('trade_id', 'count'), win_rate=('is_winner', 'mean'), avg_pnl=('net_pnl', 'mean'), median_pnl=('net_pnl', 'median'), expectancy=('net_pnl', 'mean')).reset_index().sort_values('trades', ascending=False).reset_index(drop=True)
    return {'bars_by_regime': bars_by_regime, 'trades_by_regime': trades_by_regime}

def _build_regime_viz(report: dict[str, pd.DataFrame]) -> go.Figure:
    df = report.get('trades_by_regime', pd.DataFrame())
    if df.empty:
        return _make_empty_figure('Analyse Regimes', 'No regime trade data')
    fig = px.bar(df, x='regime_label', y='avg_pnl', color='win_rate', title='Performance des trades par regime')
    return fig

def _safe_score_metric(metric_fn, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if metric_fn is None:
        return np.nan
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(metric_fn(y_true, y_pred))
    except Exception:
        return np.nan

def explain_signal_score(row: pd.Series, coef_map: dict[str, float] | None=None, top_n: int=3) -> str:
    if coef_map is None:
        return 'No explanation model available'
    contributions = []
    for feature, coef in coef_map.items():
        if feature not in row.index:
            continue
        value = row[feature]
        if pd.isna(value):
            continue
        contributions.append((feature, float(value) * float(coef)))
    if not contributions:
        return 'No feature contribution available'
    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:top_n]
    return '; '.join([f'{f}={c:+.4f}' for f, c in contributions])

def train_interpretable_scoring_models(trade_df: pd.DataFrame, scoring_config: ScoringConfig | None=None) -> dict[str, Any]:
    scfg = scoring_config or ScoringConfig()
    if trade_df is None or trade_df.empty:
        return {'features_ml': pd.DataFrame(), 'scoring': pd.DataFrame(), 'model_report': {'status': 'empty_input'}, 'models': {}}
    base_cols = ['trade_id', 'entry_time', 'is_winner', 'direction', 'duration_hours', 'return_pct', 'net_pnl', 'entry_hour', 'entry_day_of_week', 'entry_ema_spread', 'entry_vol_24h', 'entry_ret_24h', 'market_move_pct', 'event_count_near_trade', 'entry_regime']
    cols = [c for c in base_cols if c in trade_df.columns]
    feats = trade_df[cols].copy()
    feats['entry_time'] = pd.to_datetime(feats['entry_time'], utc=True, errors='coerce')
    feats = feats.sort_values('entry_time').dropna(subset=['entry_time']).reset_index(drop=True)
    feats['target'] = feats['is_winner'].astype(int)
    if len(feats) < scfg.min_train_rows:
        out_scoring = feats[['trade_id', 'entry_time', 'target']].copy()
        out_scoring['score'] = np.nan
        out_scoring['prob_logit'] = np.nan
        out_scoring['prob_tree'] = np.nan
        out_scoring['split'] = 'train'
        out_scoring['explanation'] = 'Insufficient rows for training'
        return {'features_ml': feats, 'scoring': out_scoring, 'model_report': {'status': 'insufficient_rows', 'rows': int(len(feats))}, 'models': {}}
    model_cols = [c for c in feats.columns if c not in ['target', 'is_winner', 'trade_id', 'entry_time', 'net_pnl', 'return_pct']]
    X_raw = feats[model_cols].copy()
    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        X_raw[c] = pd.to_numeric(X_raw[c], errors='coerce')
        X_raw[c] = X_raw[c].fillna(X_raw[c].median())
    categorical_cols = [c for c in X_raw.columns if c not in numeric_cols]
    X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False, dtype=float)
    y = feats['target'].astype(int)
    split_idx = max(1, min(len(feats) - 1, int(len(feats) * scfg.train_ratio)))
    train_mask = np.zeros(len(feats), dtype=bool)
    train_mask[:split_idx] = True
    X_train = X.loc[train_mask].copy()
    y_train = y.loc[train_mask].copy()
    X_test = X.loc[~train_mask].copy()
    y_test = y.loc[~train_mask].copy()
    prob_logit = np.full(len(feats), np.nan)
    logit_coef_map: dict[str, float] | None = None
    logit_result = None
    if sm is not None and len(np.unique(y_train)) >= 2:
        X_train_sm = sm.add_constant(X_train, has_constant='add')
        X_all_sm = sm.add_constant(X, has_constant='add')
        try:
            logit_result = sm.Logit(y_train, X_train_sm).fit(disp=False, maxiter=200)
            prob_logit = np.asarray(logit_result.predict(X_all_sm), dtype=float)
            params = logit_result.params.drop(labels=['const'], errors='ignore')
            logit_coef_map = {str(k): float(v) for k, v in params.items()}
        except Exception as exc:
            warnings.warn(f'Logit fit failed: {exc}')
    prob_tree = np.full(len(feats), np.nan)
    tree_model = None
    tree_importance_df = pd.DataFrame()
    if DecisionTreeClassifier is not None and len(np.unique(y_train)) >= 2:
        tree_model = DecisionTreeClassifier(max_depth=scfg.max_tree_depth, min_samples_leaf=scfg.min_samples_leaf, random_state=scfg.random_state)
        tree_model.fit(X_train, y_train)
        try:
            prob_tree = tree_model.predict_proba(X)[:, 1]
        except Exception:
            pass
        if sk_permutation_importance is not None and len(X_test) >= 10 and (len(np.unique(y_test)) >= 2):
            try:
                perm = sk_permutation_importance(tree_model, X_test, y_test, n_repeats=10, random_state=scfg.random_state, scoring='roc_auc')
                tree_importance_df = pd.DataFrame({'feature': X.columns, 'importance': perm.importances_mean}).sort_values('importance', ascending=False).reset_index(drop=True)
            except Exception:
                tree_importance_df = pd.DataFrame()
    score = np.nanmean(np.vstack([prob_logit, prob_tree]), axis=0)
    score[np.isinf(score)] = np.nan
    scoring = feats[['trade_id', 'entry_time', 'target']].copy()
    scoring['prob_logit'] = prob_logit
    scoring['prob_tree'] = prob_tree
    scoring['score'] = score
    scoring['split'] = np.where(train_mask, 'train', 'test')
    if logit_coef_map is not None:
        contrib_base = pd.concat([scoring, X], axis=1)
        scoring['explanation'] = contrib_base.apply(lambda row: explain_signal_score(row, coef_map=logit_coef_map), axis=1)
    else:
        scoring['explanation'] = 'Model explanations unavailable'
    test_slice = scoring['split'] == 'test'
    y_true_test = scoring.loc[test_slice, 'target'].astype(int).to_numpy()
    metrics = {'auc_logit': _safe_score_metric(roc_auc_score, y_true_test, scoring.loc[test_slice, 'prob_logit'].fillna(0.5).to_numpy()), 'auc_tree': _safe_score_metric(roc_auc_score, y_true_test, scoring.loc[test_slice, 'prob_tree'].fillna(0.5).to_numpy()), 'auc_score': _safe_score_metric(roc_auc_score, y_true_test, scoring.loc[test_slice, 'score'].fillna(0.5).to_numpy()), 'pr_logit': _safe_score_metric(average_precision_score, y_true_test, scoring.loc[test_slice, 'prob_logit'].fillna(0.5).to_numpy()), 'pr_tree': _safe_score_metric(average_precision_score, y_true_test, scoring.loc[test_slice, 'prob_tree'].fillna(0.5).to_numpy()), 'pr_score': _safe_score_metric(average_precision_score, y_true_test, scoring.loc[test_slice, 'score'].fillna(0.5).to_numpy())}
    feature_importance = []
    if logit_coef_map is not None:
        top_logit = pd.DataFrame({'feature': list(logit_coef_map.keys()), 'importance': [abs(v) for v in logit_coef_map.values()]}).sort_values('importance', ascending=False).head(10)
        for _, row in top_logit.iterrows():
            feature_importance.append({'model': 'logit', 'feature': str(row['feature']), 'importance': float(row['importance'])})
    if not tree_importance_df.empty:
        for _, row in tree_importance_df.head(10).iterrows():
            feature_importance.append({'model': 'tree_perm', 'feature': str(row['feature']), 'importance': float(row['importance'])})
    model_report = {'status': 'ok', 'rows': int(len(feats)), 'train_rows': int(train_mask.sum()), 'test_rows': int((~train_mask).sum()), 'target_positive_rate': float(feats['target'].mean()), 'metrics': metrics, 'feature_importance': feature_importance}
    features_ml = pd.concat([feats[['trade_id', 'entry_time', 'target']], X], axis=1)
    features_ml = features_ml.rename(columns={'target': 'is_winner'})
    features_ml['feature_set_version'] = 'v1_interpretable'
    return {'features_ml': features_ml, 'scoring': scoring, 'model_report': model_report, 'models': {'logit': logit_result, 'tree': tree_model}}

def _build_scoring_viz(scoring_df: pd.DataFrame) -> go.Figure:
    if scoring_df is None or scoring_df.empty:
        return _make_empty_figure('Signal Scoring', 'No scoring data')
    fig = px.scatter(scoring_df, x='entry_time', y='score', color='target', symbol='split', title='Signal scoring dans le temps')
    return fig

def _compute_psi(train_values: pd.Series, test_values: pd.Series, bins: int=10) -> float:
    train = pd.to_numeric(train_values, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    test = pd.to_numeric(test_values, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(train) < 10 or len(test) < 10:
        return np.nan
    q = np.linspace(0.0, 1.0, max(3, int(bins)) + 1)
    edges = np.unique(np.quantile(train, q))
    if len(edges) <= 2:
        lo = float(np.min(train))
        hi = float(np.max(train))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return np.nan
        if hi <= lo:
            hi = lo + 1e-9
        edges = np.linspace(lo, hi, num=max(3, int(bins) + 1))
    train_hist = np.histogram(train, bins=edges)[0].astype(float)
    test_hist = np.histogram(test, bins=edges)[0].astype(float)
    eps = 1e-9
    train_pct = np.clip(train_hist / max(1.0, train_hist.sum()), eps, 1.0)
    test_pct = np.clip(test_hist / max(1.0, test_hist.sum()), eps, 1.0)
    psi = float(np.sum((test_pct - train_pct) * np.log(test_pct / train_pct)))
    return psi

def compute_feature_drift_report(features_ml: pd.DataFrame, scoring_df: pd.DataFrame | None=None, max_features: int=80, min_non_null: int=25) -> pd.DataFrame:
    cols = ['rank', 'feature', 'train_n', 'test_n', 'train_mean', 'test_mean', 'mean_shift', 'mean_shift_z', 'std_ratio', 'psi', 'ks_stat', 'ks_pvalue', 'severity', 'drift_flag']
    if features_ml is None or features_ml.empty:
        return pd.DataFrame(columns=cols)
    df = features_ml.copy()
    if 'trade_id' not in df.columns:
        return pd.DataFrame(columns=cols)
    if 'entry_time' in df.columns:
        df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True, errors='coerce')
    if ('split' not in df.columns) and (scoring_df is not None) and (not scoring_df.empty) and {'trade_id', 'split'}.issubset(scoring_df.columns):
        split_map = scoring_df[['trade_id', 'split']].dropna().drop_duplicates(subset=['trade_id'], keep='last')
        df = df.merge(split_map, on='trade_id', how='left')
    if 'split' not in df.columns:
        if 'entry_time' in df.columns:
            df = df.sort_values('entry_time').reset_index(drop=True)
            cut = max(1, min(len(df) - 1, int(len(df) * 0.7)))
            df['split'] = np.where(np.arange(len(df)) < cut, 'train', 'test')
        else:
            return pd.DataFrame(columns=cols)
    numeric_candidates = []
    for c in df.columns:
        if c in {'trade_id', 'entry_time', 'split', 'feature_set_version'}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_candidates.append(c)
    if not numeric_candidates:
        return pd.DataFrame(columns=cols)
    train = df[df['split'].astype(str) == 'train'].copy()
    test = df[df['split'].astype(str) == 'test'].copy()
    if train.empty or test.empty:
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, Any]] = []
    for c in numeric_candidates:
        tr = pd.to_numeric(train[c], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        te = pd.to_numeric(test[c], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if len(tr) < int(min_non_null) or len(te) < int(min_non_null):
            continue
        tr_mean = float(tr.mean())
        te_mean = float(te.mean())
        tr_std = float(tr.std(ddof=0))
        te_std = float(te.std(ddof=0))
        mean_shift = float(te_mean - tr_mean)
        mean_shift_z = float(mean_shift / tr_std) if tr_std > 0 else np.nan
        std_ratio = float(te_std / tr_std) if tr_std > 0 else np.nan
        psi = _compute_psi(tr, te, bins=10)
        if ks_2samp is not None:
            try:
                ks_out = ks_2samp(tr.to_numpy(dtype=float), te.to_numpy(dtype=float))
                ks_stat = float(ks_out.statistic)
                ks_pvalue = float(ks_out.pvalue)
            except Exception:
                ks_stat = np.nan
                ks_pvalue = np.nan
        else:
            ks_stat = np.nan
            ks_pvalue = np.nan
        severity = float(
            0.45 * min(5.0, abs(mean_shift_z) if np.isfinite(mean_shift_z) else 0.0)
            + 0.35 * min(5.0, psi if np.isfinite(psi) else 0.0)
            + 0.20 * min(5.0, (ks_stat * 5.0) if np.isfinite(ks_stat) else 0.0)
        )
        drift_flag = bool(
            ((np.isfinite(psi) and psi >= 0.20))
            or ((np.isfinite(mean_shift_z) and abs(mean_shift_z) >= 0.75))
            or ((np.isfinite(ks_stat) and ks_stat >= 0.25))
        )
        rows.append({'feature': str(c), 'train_n': int(len(tr)), 'test_n': int(len(te)), 'train_mean': tr_mean, 'test_mean': te_mean, 'mean_shift': mean_shift, 'mean_shift_z': mean_shift_z, 'std_ratio': std_ratio, 'psi': psi, 'ks_stat': ks_stat, 'ks_pvalue': ks_pvalue, 'severity': severity, 'drift_flag': drift_flag})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=cols)
    out = out.sort_values(['severity', 'psi', 'mean_shift_z'], ascending=[False, False, False]).head(int(max_features)).reset_index(drop=True)
    out.insert(0, 'rank', np.arange(1, len(out) + 1))
    return out[cols]

def _build_feature_drift_viz(feature_drift_df: pd.DataFrame) -> go.Figure:
    if feature_drift_df is None or feature_drift_df.empty:
        return _make_empty_figure('Feature Drift', 'No drift data')
    top = feature_drift_df.head(20).copy()
    top = top.sort_values('severity', ascending=True)
    fig = px.bar(top, x='severity', y='feature', orientation='h', color='psi', title='Feature drift (train vs test)', hover_data=['mean_shift_z', 'ks_stat', 'drift_flag'])
    fig.update_layout(yaxis_title='Feature', xaxis_title='Drift severity')
    return fig


def compute_market_anomaly_features(market_df: pd.DataFrame) -> pd.DataFrame:
    cols = ['open', 'high', 'low', 'close', 'volume']
    if market_df is None or market_df.empty or (not set(cols).issubset(market_df.columns)):
        return pd.DataFrame()
    close = pd.to_numeric(market_df['close'], errors='coerce')
    open_ = pd.to_numeric(market_df['open'], errors='coerce')
    high = pd.to_numeric(market_df['high'], errors='coerce')
    low = pd.to_numeric(market_df['low'], errors='coerce')
    volume = pd.to_numeric(market_df['volume'], errors='coerce')
    ret_1h = close.pct_change()
    abs_ret_1h = ret_1h.abs()
    gap_pct = open_ / close.shift(1).replace(0.0, np.nan) - 1.0
    range_pct = (high - low) / close.replace(0.0, np.nan)
    body_pct = (close - open_).abs() / open_.replace(0.0, np.nan)
    vol_24h = ret_1h.rolling(24, min_periods=8).std()
    vol_72h = ret_1h.rolling(72, min_periods=24).std()
    vol_mean_48 = volume.rolling(48, min_periods=12).mean()
    vol_std_48 = volume.rolling(48, min_periods=12).std()
    volume_zscore = (volume - vol_mean_48) / vol_std_48.replace(0.0, np.nan)
    close_mean_48 = close.rolling(48, min_periods=12).mean()
    close_std_48 = close.rolling(48, min_periods=12).std()
    close_zscore = (close - close_mean_48) / close_std_48.replace(0.0, np.nan)
    feat = pd.DataFrame(
        {
            'ret_1h': ret_1h,
            'abs_ret_1h': abs_ret_1h,
            'gap_pct': gap_pct,
            'range_pct': range_pct,
            'body_pct': body_pct,
            'vol_24h': vol_24h,
            'vol_72h': vol_72h,
            'volume_zscore': volume_zscore,
            'close_zscore': close_zscore,
        },
        index=market_df.index,
    )
    return feat.replace([np.inf, -np.inf], np.nan)


def _robust_center_scale(train_series: pd.Series) -> tuple[float, float]:
    tr = pd.to_numeric(train_series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if tr.empty:
        return (0.0, 1.0)
    center = float(tr.median())
    mad = float((tr - center).abs().median())
    scale = 1.4826 * mad
    if (not np.isfinite(scale)) or scale <= 1e-12:
        std = float(tr.std(ddof=0))
        if np.isfinite(std) and std > 1e-12:
            scale = std
        else:
            scale = 1.0
    return (center, float(scale))


def detect_market_anomalies(market_df: pd.DataFrame, config: BacktestConfig, fit_end_time: pd.Timestamp | None=None) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = ['anomaly_score', 'anomaly_flag', 'model_flag', 'robust_score', 'top_feature', 'top_feature_z', 'threshold_score', 'fit_split', 'model_name']
    if (not bool(config.anomaly_enabled)) or market_df is None or market_df.empty:
        return (pd.DataFrame(columns=cols), {'status': 'disabled'})
    features = compute_market_anomaly_features(market_df)
    if features.empty:
        return (pd.DataFrame(columns=cols), {'status': 'empty_features'})
    valid = features.dropna().copy()
    if valid.empty:
        return (pd.DataFrame(columns=cols), {'status': 'no_valid_rows'})
    if fit_end_time is not None:
        fit_ts = pd.to_datetime(fit_end_time, utc=True, errors='coerce')
        train_idx = valid.index[valid.index <= fit_ts] if pd.notna(fit_ts) else valid.index[:0]
        train_valid = valid.loc[train_idx].copy()
    else:
        cut = max(1, min(len(valid) - 1, int(len(valid) * float(config.split_ratios[0]))))
        train_valid = valid.iloc[:cut].copy()
    if train_valid.empty:
        cut = max(1, min(len(valid) - 1, int(len(valid) * 0.7)))
        train_valid = valid.iloc[:cut].copy()
    z_abs = pd.DataFrame(index=valid.index)
    for c in valid.columns:
        center, scale = _robust_center_scale(train_valid[c])
        z_abs[c] = (pd.to_numeric(valid[c], errors='coerce') - center).abs() / scale
    robust_score = z_abs.mean(axis=1).astype(float)
    top_feature = z_abs.idxmax(axis=1).astype('string')
    top_feature_z = z_abs.max(axis=1).astype(float)
    model_name = 'robust_zscore'
    model_flag = pd.Series(False, index=valid.index, dtype=bool)
    anomaly_score = robust_score.copy()
    model_status = 'fallback_robust_zscore'
    use_iforest = (
        str(config.anomaly_model) == 'isolation_forest'
        and IsolationForest is not None
        and len(train_valid) >= max(int(config.anomaly_min_train_rows), len(valid.columns) * 8)
    )
    if use_iforest:
        try:
            if_model = IsolationForest(
                n_estimators=300,
                contamination=float(config.anomaly_contamination),
                random_state=int(config.anomaly_random_state),
            )
            if_model.fit(train_valid.to_numpy(dtype=float))
            scores = -if_model.decision_function(valid.to_numpy(dtype=float))
            flags = if_model.predict(valid.to_numpy(dtype=float)) == -1
            anomaly_score = pd.Series(scores, index=valid.index, dtype=float)
            model_flag = pd.Series(flags, index=valid.index, dtype=bool)
            model_name = 'isolation_forest'
            model_status = 'ok'
        except Exception as exc:
            model_status = f'iforest_fallback:{exc.__class__.__name__}'
    train_scores = anomaly_score.reindex(train_valid.index).dropna()
    if train_scores.empty:
        train_scores = anomaly_score.dropna()
    q_target = float(config.anomaly_threshold_quantile)
    q_contam = max(0.0, min(0.999, 1.0 - float(config.anomaly_contamination)))
    thr_q = float(train_scores.quantile(q_target)) if not train_scores.empty else np.nan
    thr_c = float(train_scores.quantile(q_contam)) if not train_scores.empty else np.nan
    if np.isfinite(thr_q) and np.isfinite(thr_c):
        threshold = max(thr_q, thr_c)
    elif np.isfinite(thr_q):
        threshold = thr_q
    elif np.isfinite(thr_c):
        threshold = thr_c
    else:
        threshold = np.nan
    if np.isfinite(threshold):
        anomaly_flag = (anomaly_score >= float(threshold)) | model_flag
    else:
        anomaly_flag = model_flag.copy()
    out = pd.DataFrame(index=valid.index)
    out['anomaly_score'] = anomaly_score.astype(float)
    out['anomaly_flag'] = anomaly_flag.astype(bool)
    out['model_flag'] = model_flag.astype(bool)
    out['robust_score'] = robust_score.astype(float)
    out['top_feature'] = top_feature.astype('string')
    out['top_feature_z'] = top_feature_z.astype(float)
    out['threshold_score'] = float(threshold) if np.isfinite(threshold) else np.nan
    fit_last = train_valid.index.max()
    out['fit_split'] = pd.Series(np.where(out.index <= fit_last, 'train_fit', 'post_fit'), index=out.index, dtype='string')
    out['model_name'] = str(model_name)
    for c in valid.columns:
        out[f'feat__{c}'] = pd.to_numeric(valid[c], errors='coerce')
    out_full = out.reindex(market_df.index)
    out_full['anomaly_flag'] = out_full['anomaly_flag'].fillna(False).astype(bool)
    out_full['model_flag'] = out_full['model_flag'].fillna(False).astype(bool)
    out_full['top_feature'] = out_full['top_feature'].astype('string').fillna('')
    out_full['fit_split'] = out_full['fit_split'].astype('string').fillna('unknown')
    out_full['model_name'] = out_full['model_name'].astype('string').fillna(str(model_name))
    report = {
        'status': 'ok',
        'model_status': str(model_status),
        'model_name': str(model_name),
        'rows_total': int(len(out_full)),
        'rows_scored': int(len(valid)),
        'train_rows': int(len(train_valid)),
        'anomaly_rows': int(out_full['anomaly_flag'].sum()),
        'anomaly_rate': float(out_full['anomaly_flag'].mean()),
        'threshold_score': float(threshold) if np.isfinite(threshold) else np.nan,
        'contamination_target': float(config.anomaly_contamination),
    }
    return (out_full, report)


def attach_anomalies_to_trades(trades: pd.DataFrame, anomaly_scores: pd.DataFrame, pre_window_h: int=6, post_window_h: int=2) -> pd.DataFrame:
    cols = ['trade_id', 'entry_time', 'exit_time', 'anomaly_timestamp', 'anomaly_score', 'top_feature', 'top_feature_z', 'hours_from_entry', 'hours_to_exit', 'in_trade_window']
    if trades is None or trades.empty or anomaly_scores is None or anomaly_scores.empty:
        return pd.DataFrame(columns=cols)
    tdf = _ensure_datetime_utc(trades, ['entry_time', 'exit_time'])
    adf = anomaly_scores.copy()
    adf = adf[adf['anomaly_flag'].astype(bool)].copy() if 'anomaly_flag' in adf.columns else adf.copy()
    if adf.empty:
        return pd.DataFrame(columns=cols)
    adf = adf.reset_index()
    if 'anomaly_timestamp' not in adf.columns:
        if 'index' in adf.columns:
            adf = adf.rename(columns={'index': 'anomaly_timestamp'})
        elif 'timestamp' in adf.columns:
            adf = adf.rename(columns={'timestamp': 'anomaly_timestamp'})
        elif len(adf.columns) > 0:
            adf = adf.rename(columns={adf.columns[0]: 'anomaly_timestamp'})
    adf['anomaly_timestamp'] = pd.to_datetime(adf['anomaly_timestamp'], utc=True, errors='coerce')
    adf = adf.dropna(subset=['anomaly_timestamp']).sort_values('anomaly_timestamp')
    links = []
    pre_delta = pd.Timedelta(hours=int(pre_window_h))
    post_delta = pd.Timedelta(hours=int(post_window_h))
    for _, tr in tdf.iterrows():
        if pd.isna(tr['entry_time']) or pd.isna(tr['exit_time']):
            continue
        start = tr['entry_time'] - pre_delta
        end = tr['exit_time'] + post_delta
        local = adf[(adf['anomaly_timestamp'] >= start) & (adf['anomaly_timestamp'] <= end)]
        if local.empty:
            continue
        for _, row in local.iterrows():
            h_entry = (row['anomaly_timestamp'] - tr['entry_time']).total_seconds() / 3600.0
            h_exit = (tr['exit_time'] - row['anomaly_timestamp']).total_seconds() / 3600.0
            links.append(
                {
                    'trade_id': int(pd.to_numeric(tr['trade_id'], errors='coerce')),
                    'entry_time': tr['entry_time'],
                    'exit_time': tr['exit_time'],
                    'anomaly_timestamp': row['anomaly_timestamp'],
                    'anomaly_score': float(pd.to_numeric(row.get('anomaly_score', np.nan), errors='coerce')),
                    'top_feature': str(row.get('top_feature', '')),
                    'top_feature_z': float(pd.to_numeric(row.get('top_feature_z', np.nan), errors='coerce')),
                    'hours_from_entry': float(h_entry),
                    'hours_to_exit': float(h_exit),
                    'in_trade_window': bool(row['anomaly_timestamp'] >= tr['entry_time'] and row['anomaly_timestamp'] <= tr['exit_time']),
                }
            )
    return pd.DataFrame(links, columns=cols)


def anomaly_impact_report(trades: pd.DataFrame, trade_anomalies: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if trades is None or trades.empty:
        empty = pd.DataFrame()
        return {'with_vs_without': empty, 'by_top_feature': empty}
    tdf = trades.copy()
    if trade_anomalies is None or trade_anomalies.empty:
        tdf['has_anomaly'] = False
        with_vs_without = tdf.groupby('has_anomaly').agg(trades=('trade_id', 'count'), win_rate=('is_winner', 'mean'), avg_pnl=('net_pnl', 'mean'), median_pnl=('net_pnl', 'median'), expectancy=('net_pnl', 'mean')).reset_index()
        return {'with_vs_without': with_vs_without, 'by_top_feature': pd.DataFrame()}
    ids = set(pd.to_numeric(trade_anomalies['trade_id'], errors='coerce').dropna().astype(int).tolist())
    tdf['has_anomaly'] = pd.to_numeric(tdf['trade_id'], errors='coerce').fillna(-1).astype(int).isin(ids)
    with_vs_without = tdf.groupby('has_anomaly').agg(trades=('trade_id', 'count'), win_rate=('is_winner', 'mean'), avg_pnl=('net_pnl', 'mean'), median_pnl=('net_pnl', 'median'), expectancy=('net_pnl', 'mean')).reset_index()
    tmp = trade_anomalies.copy()
    tmp['trade_id'] = pd.to_numeric(tmp['trade_id'], errors='coerce').astype('Int64')
    tmp = tmp.dropna(subset=['trade_id'])
    tmp['trade_id'] = tmp['trade_id'].astype(int)
    tmp['top_feature'] = tmp['top_feature'].astype('string').fillna('unknown')
    per_trade_feature = tmp.sort_values(['trade_id', 'anomaly_score'], ascending=[True, False]).drop_duplicates(subset=['trade_id', 'top_feature'], keep='first')
    merged = per_trade_feature.merge(tdf[['trade_id', 'is_winner', 'net_pnl']], on='trade_id', how='left')
    by_top_feature = merged.groupby('top_feature').agg(linked_trades=('trade_id', 'nunique'), win_rate=('is_winner', 'mean'), avg_pnl=('net_pnl', 'mean'), median_pnl=('net_pnl', 'median'), expectancy=('net_pnl', 'mean'), avg_anomaly_score=('anomaly_score', 'mean')).reset_index()
    base_no = with_vs_without[with_vs_without['has_anomaly'] == False]
    base_exp = float(pd.to_numeric(base_no['expectancy'].iloc[0], errors='coerce')) if not base_no.empty else np.nan
    by_top_feature['delta_expectancy'] = pd.to_numeric(by_top_feature['expectancy'], errors='coerce') - base_exp
    by_top_feature['n_with_anomaly'] = by_top_feature['linked_trades'].astype(int)
    by_top_feature = by_top_feature.sort_values('linked_trades', ascending=False).reset_index(drop=True)
    return {'with_vs_without': with_vs_without, 'by_top_feature': by_top_feature}


def _build_anomaly_viz(anomaly_scores: pd.DataFrame, anomaly_report_dict: dict[str, pd.DataFrame]) -> go.Figure:
    if anomaly_scores is None or anomaly_scores.empty:
        return _make_empty_figure('Anomaly Analysis', 'No anomaly scores')
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Anomaly score timeline', 'Impact by anomaly feature'))
    local = anomaly_scores.copy()
    fig.add_trace(go.Scatter(x=local.index, y=local['anomaly_score'], mode='lines', name='Anomaly score'), row=1, col=1)
    flagged = local[local['anomaly_flag'].astype(bool)] if 'anomaly_flag' in local.columns else pd.DataFrame()
    if not flagged.empty:
        fig.add_trace(
            go.Scatter(
                x=flagged.index,
                y=flagged['anomaly_score'],
                mode='markers',
                marker=dict(color='#dc2626', size=5),
                name='Flagged anomalies',
                hovertext=flagged['top_feature'] if 'top_feature' in flagged.columns else None,
            ),
            row=1,
            col=1,
        )
    by_feature = anomaly_report_dict.get('by_top_feature', pd.DataFrame()) if isinstance(anomaly_report_dict, dict) else pd.DataFrame()
    if by_feature is not None and (not by_feature.empty):
        top = by_feature.head(12).copy()
        fig.add_trace(
            go.Bar(
                x=top['top_feature'].astype(str),
                y=top['delta_expectancy'],
                marker_color=np.where(pd.to_numeric(top['delta_expectancy'], errors='coerce') >= 0, '#16a34a', '#dc2626'),
                text=top['linked_trades'].astype(int).astype(str),
                name='Delta expectancy',
            ),
            row=1,
            col=2,
        )
    fig.update_yaxes(title_text='Anomaly score', row=1, col=1)
    fig.update_yaxes(title_text='Delta expectancy', row=1, col=2)
    fig.update_layout(title='Anomaly detection (market context + trade impact)')
    return fig

def _trade_outcome_metrics(trades_df: pd.DataFrame) -> dict[str, float]:
    if trades_df is None or trades_df.empty:
        return {'trades': 0.0, 'total_pnl': 0.0, 'expectancy': np.nan, 'win_rate': np.nan, 'profit_factor': np.nan}
    pnl = pd.to_numeric(trades_df['net_pnl'], errors='coerce').dropna()
    if pnl.empty:
        return {'trades': 0.0, 'total_pnl': 0.0, 'expectancy': np.nan, 'win_rate': np.nan, 'profit_factor': np.nan}
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(losses.sum())
    profit_factor = float(gross_profit / abs(gross_loss)) if gross_loss < 0 else (np.inf if gross_profit > 0 else np.nan)
    return {'trades': float(len(pnl)), 'total_pnl': float(pnl.sum()), 'expectancy': float(pnl.mean()), 'win_rate': float((pnl > 0).mean()), 'profit_factor': profit_factor}

def train_meta_labeling_model(trade_enriched: pd.DataFrame, scoring_df: pd.DataFrame, config: BacktestConfig, scoring_config: ScoringConfig | None=None) -> dict[str, Any]:
    empty_result = {'meta_scoring': pd.DataFrame(), 'threshold_scan': pd.DataFrame(), 'meta_report': {'status': 'empty_input'}}
    if trade_enriched is None or trade_enriched.empty or scoring_df is None or scoring_df.empty:
        return empty_result
    scfg = scoring_config or ScoringConfig()
    req_trade_cols = ['trade_id', 'entry_time', 'is_winner', 'net_pnl']
    if not set(req_trade_cols).issubset(trade_enriched.columns) or 'trade_id' not in scoring_df.columns:
        return empty_result
    base_cols = ['trade_id', 'entry_time', 'is_winner', 'net_pnl', 'direction', 'entry_hour', 'entry_day_of_week', 'entry_ema_spread', 'entry_vol_24h', 'entry_ret_24h', 'event_count_near_trade', 'entry_regime']
    trade_part = trade_enriched[[c for c in base_cols if c in trade_enriched.columns]].copy()
    score_part_cols = [c for c in ['trade_id', 'score', 'prob_logit', 'prob_tree', 'split'] if c in scoring_df.columns]
    score_part = scoring_df[score_part_cols].copy().drop_duplicates(subset=['trade_id'], keep='last')
    df = trade_part.merge(score_part, on='trade_id', how='left')
    df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True, errors='coerce')
    df = df.dropna(subset=['entry_time']).sort_values('entry_time').reset_index(drop=True)
    if len(df) < int(config.meta_min_train_trades):
        return {'meta_scoring': pd.DataFrame(), 'threshold_scan': pd.DataFrame(), 'meta_report': {'status': 'insufficient_rows', 'rows': int(len(df))}}
    if 'split' not in df.columns:
        df['split'] = pd.NA
    split_series = df['split'].astype(str).str.lower()
    if (split_series == 'train').sum() < int(config.meta_min_train_trades) or (split_series == 'test').sum() < 5:
        cut = max(1, min(len(df) - 1, int(len(df) * scfg.train_ratio)))
        df['split'] = np.where(np.arange(len(df)) < cut, 'train', 'test')
    else:
        df['split'] = np.where(split_series == 'train', 'train', 'test')
    df['target'] = pd.to_numeric(df['is_winner'], errors='coerce').fillna(0).astype(int)
    model_cols = [c for c in df.columns if c not in {'trade_id', 'entry_time', 'target', 'is_winner', 'net_pnl', 'split'}]
    X_raw = df[model_cols].copy()
    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        X_raw[c] = pd.to_numeric(X_raw[c], errors='coerce')
        X_raw[c] = X_raw[c].fillna(X_raw[c].median())
    categorical_cols = [c for c in X_raw.columns if c not in numeric_cols]
    for c in categorical_cols:
        X_raw[c] = X_raw[c].astype('string').fillna('missing')
    X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False, dtype=float)
    y = df['target'].astype(int)
    train_mask = (df['split'].astype(str) == 'train').to_numpy(dtype=bool)
    test_mask = ~train_mask
    X_train = X.loc[train_mask].copy()
    y_train = y.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()
    y_test = y.loc[test_mask].copy()
    prob_logit = np.full(len(df), np.nan)
    prob_tree = np.full(len(df), np.nan)
    logit_coef_map: dict[str, float] | None = None
    if sm is not None and len(X_train) >= int(config.meta_min_train_trades) and len(np.unique(y_train)) >= 2:
        X_train_sm = sm.add_constant(X_train, has_constant='add')
        X_all_sm = sm.add_constant(X, has_constant='add')
        try:
            logit_res = sm.Logit(y_train, X_train_sm).fit(disp=False, maxiter=200)
            prob_logit = np.asarray(logit_res.predict(X_all_sm), dtype=float)
            params = logit_res.params.drop(labels=['const'], errors='ignore')
            logit_coef_map = {str(k): float(v) for k, v in params.items()}
        except Exception:
            pass
    if DecisionTreeClassifier is not None and len(X_train) >= int(config.meta_min_train_trades) and len(np.unique(y_train)) >= 2:
        try:
            tree_model = DecisionTreeClassifier(max_depth=scfg.max_tree_depth, min_samples_leaf=scfg.min_samples_leaf, random_state=scfg.random_state)
            tree_model.fit(X_train, y_train)
            prob_tree = tree_model.predict_proba(X)[:, 1]
        except Exception:
            pass
    base_score = pd.to_numeric(df['score'], errors='coerce').to_numpy(dtype=float) if 'score' in df.columns else np.full(len(df), np.nan)
    stack = np.vstack([prob_logit, prob_tree, base_score])
    meta_score = np.nanmean(stack, axis=0)
    meta_score[np.isinf(meta_score)] = np.nan
    out = df[['trade_id', 'entry_time', 'target', 'is_winner', 'net_pnl', 'split']].copy()
    out['base_score'] = base_score
    out['meta_prob_logit'] = prob_logit
    out['meta_prob_tree'] = prob_tree
    out['meta_score'] = meta_score
    if logit_coef_map is not None:
        contrib_base = pd.concat([out[['trade_id']], X], axis=1)
        out['meta_explanation'] = contrib_base.apply(lambda row: explain_signal_score(row, coef_map=logit_coef_map), axis=1)
    else:
        out['meta_explanation'] = 'Meta explanation unavailable'
    train_scores = pd.to_numeric(out.loc[train_mask, 'meta_score'], errors='coerce').dropna()
    threshold_rows = []
    min_sel = max(5, int(0.10 * max(1, train_mask.sum())))
    if not train_scores.empty:
        cand = np.unique(np.concatenate([np.linspace(0.45, 0.90, num=10), np.quantile(train_scores, [0.50, 0.60, 0.70, 0.80, 0.90])]))
        for thr in cand:
            sel = out.loc[train_mask & (out['meta_score'] >= float(thr))].copy()
            if len(sel) < min_sel:
                continue
            m = _trade_outcome_metrics(sel)
            objective = float(pd.to_numeric(m['expectancy'], errors='coerce')) * math.sqrt(max(1.0, float(m['trades'])))
            threshold_rows.append({'threshold': float(thr), 'trades': int(m['trades']), 'expectancy': float(m['expectancy']), 'win_rate': float(m['win_rate']), 'total_pnl': float(m['total_pnl']), 'objective': objective})
    threshold_scan = pd.DataFrame(threshold_rows)
    if threshold_scan.empty:
        best_threshold = 0.5
    else:
        threshold_scan = threshold_scan.sort_values(['objective', 'expectancy', 'trades'], ascending=[False, False, False]).reset_index(drop=True)
        best_threshold = float(threshold_scan.iloc[0]['threshold'])
    out['meta_take'] = pd.to_numeric(out['meta_score'], errors='coerce') >= float(best_threshold)
    test_all = out.loc[test_mask].copy()
    test_take = out.loc[test_mask & out['meta_take']].copy()
    base_metrics = _trade_outcome_metrics(test_all)
    filt_metrics = _trade_outcome_metrics(test_take)
    meta_metrics = {
        'status': 'ok',
        'rows': int(len(out)),
        'train_rows': int(train_mask.sum()),
        'test_rows': int(test_mask.sum()),
        'recommended_threshold': float(best_threshold),
        'test_coverage': float((test_all['meta_take'].mean()) if (not test_all.empty) else np.nan),
        'baseline_test': base_metrics,
        'filtered_test': filt_metrics,
        'delta_expectancy_test': float(pd.to_numeric(filt_metrics['expectancy'], errors='coerce') - pd.to_numeric(base_metrics['expectancy'], errors='coerce')) if (test_all is not None) else np.nan,
        'delta_total_pnl_test': float(pd.to_numeric(filt_metrics['total_pnl'], errors='coerce') - pd.to_numeric(base_metrics['total_pnl'], errors='coerce')) if (test_all is not None) else np.nan,
        'delta_win_rate_test': float(pd.to_numeric(filt_metrics['win_rate'], errors='coerce') - pd.to_numeric(base_metrics['win_rate'], errors='coerce')) if (test_all is not None) else np.nan,
        'auc_meta': _safe_score_metric(roc_auc_score, y_test.to_numpy(), pd.to_numeric(out.loc[test_mask, 'meta_score'], errors='coerce').fillna(0.5).to_numpy()) if len(y_test) else np.nan,
        'pr_meta': _safe_score_metric(average_precision_score, y_test.to_numpy(), pd.to_numeric(out.loc[test_mask, 'meta_score'], errors='coerce').fillna(0.5).to_numpy()) if len(y_test) else np.nan,
    }
    return {'meta_scoring': out, 'threshold_scan': threshold_scan, 'meta_report': meta_metrics}

def _build_meta_labeling_viz(meta_scoring: pd.DataFrame, threshold_scan: pd.DataFrame) -> go.Figure:
    if meta_scoring is None or meta_scoring.empty:
        return _make_empty_figure('Meta Labeling', 'No meta-labeling data')
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Threshold search (train)', 'Test cumulative PnL: baseline vs filtered'))
    if threshold_scan is not None and (not threshold_scan.empty):
        fig.add_trace(go.Scatter(x=threshold_scan['threshold'], y=threshold_scan['objective'], mode='lines+markers', name='Objective'), row=1, col=1)
    test_df = meta_scoring[meta_scoring['split'].astype(str) == 'test'].copy()
    if not test_df.empty:
        test_df = test_df.sort_values('entry_time')
        pnl = pd.to_numeric(test_df['net_pnl'], errors='coerce').fillna(0.0)
        pnl_f = pnl.where(test_df['meta_take'].astype(bool), 0.0)
        fig.add_trace(go.Scatter(x=test_df['entry_time'], y=pnl.cumsum(), mode='lines', name='Baseline cum pnl'), row=1, col=2)
        fig.add_trace(go.Scatter(x=test_df['entry_time'], y=pnl_f.cumsum(), mode='lines', name='Filtered cum pnl'), row=1, col=2)
    fig.update_layout(title='Meta-labeling: take/skip signals')
    return fig

def build_auto_recommendations(core_metrics: dict[str, Any] | None, failure_diagnosis: pd.DataFrame | None, feature_drift: pd.DataFrame | None, meta_report: dict[str, Any] | None, parameter_sensitivity: pd.DataFrame | None, stat_robustness_summary: pd.DataFrame | None, event_impact: pd.DataFrame | None, regime_performance: pd.DataFrame | None, anomaly_impact: pd.DataFrame | None=None, max_items: int=12) -> pd.DataFrame:
    cols = ['rank', 'category', 'priority', 'confidence', 'recommendation', 'rationale', 'expected_impact', 'evidence']
    rows: list[dict[str, Any]] = []
    core = core_metrics or {}
    if failure_diagnosis is not None and (not failure_diagnosis.empty):
        for _, row in failure_diagnosis.head(4).iterrows():
            sev = float(pd.to_numeric(row.get('severity', np.nan), errors='coerce'))
            rows.append({'category': 'failure_context', 'priority': sev + 2.0, 'confidence': min(0.95, 0.40 + float(pd.to_numeric(row.get('trades', 0), errors='coerce')) / 200.0), 'recommendation': f"Filtrer contexte: {row.get('dimension')}={row.get('context')}", 'rationale': str(row.get('action_hint', 'Apply contextual filter')), 'expected_impact': f"delta_expectancy={float(pd.to_numeric(row.get('delta_expectancy', np.nan), errors='coerce')):.4f}", 'evidence': str(row.get('evidence', 'failure_diagnosis'))})
    if feature_drift is not None and (not feature_drift.empty):
        flagged = feature_drift[feature_drift['drift_flag'] == True].copy()
        if not flagged.empty:
            top = flagged.head(3)
            for _, row in top.iterrows():
                psi = float(pd.to_numeric(row.get('psi', np.nan), errors='coerce'))
                rows.append({'category': 'feature_drift', 'priority': 3.0 + float(pd.to_numeric(row.get('severity', np.nan), errors='coerce')), 'confidence': 0.75, 'recommendation': f"Recalibrer/re-entrainer scoring (drift sur {row.get('feature')})", 'rationale': 'Distribution train/test instable sur feature cle.', 'expected_impact': f"psi={psi:.3f}", 'evidence': f"mean_shift_z={float(pd.to_numeric(row.get('mean_shift_z', np.nan), errors='coerce')):.3f}, ks={float(pd.to_numeric(row.get('ks_stat', np.nan), errors='coerce')):.3f}"})
    if meta_report:
        status = str(meta_report.get('status', ''))
        if status == 'ok':
            d_exp = float(pd.to_numeric(meta_report.get('delta_expectancy_test', np.nan), errors='coerce'))
            d_pnl = float(pd.to_numeric(meta_report.get('delta_total_pnl_test', np.nan), errors='coerce'))
            thr = float(pd.to_numeric(meta_report.get('recommended_threshold', np.nan), errors='coerce'))
            cov = float(pd.to_numeric(meta_report.get('test_coverage', np.nan), errors='coerce'))
            if np.isfinite(d_exp) and (d_exp > 0):
                rows.append({'category': 'meta_labeling', 'priority': 7.0 + min(5.0, d_exp * 100.0), 'confidence': 0.80, 'recommendation': f"Activer filtre meta: prendre signal si score >= {thr:.3f}", 'rationale': 'Le filtre meta augmente expectancy sur le split test.', 'expected_impact': f"delta_expectancy_test={d_exp:.4f}, delta_total_pnl_test={d_pnl:.2f}", 'evidence': f"coverage_test={cov:.2%}, auc_meta={float(pd.to_numeric(meta_report.get('auc_meta', np.nan), errors='coerce')):.3f}"})
            elif np.isfinite(d_exp):
                rows.append({'category': 'meta_labeling', 'priority': 3.0, 'confidence': 0.60, 'recommendation': 'Ne pas activer meta-filter en production pour l instant', 'rationale': 'Le filtre meta degrade ou n ameliore pas expectancy sur test.', 'expected_impact': f"delta_expectancy_test={d_exp:.4f}", 'evidence': f"coverage_test={cov:.2%}"})
    if parameter_sensitivity is not None and (not parameter_sensitivity.empty):
        top = parameter_sensitivity.iloc[0]
        d_ret = float(pd.to_numeric(top.get('delta_vs_baseline_return', np.nan), errors='coerce'))
        d_shp = float(pd.to_numeric(top.get('delta_vs_baseline_sharpe', np.nan), errors='coerce'))
        if np.isfinite(d_ret) and d_ret > 0:
            rec = f"Tester variante params: ema_fast={int(top['ema_fast'])}, ema_slow={int(top['ema_slow'])}, sl={float(top['stop_loss_pct']):.4f}, tp={float(top['take_profit_pct']):.4f}"
            rows.append({'category': 'parameter_tuning', 'priority': 6.0 + min(4.0, d_ret * 100.0), 'confidence': 0.70, 'recommendation': rec, 'rationale': 'Top combinaison sensitivity superieure au baseline.', 'expected_impact': f"delta_return={d_ret:.2%}, delta_sharpe={d_shp:.3f}", 'evidence': f"composite_score={float(pd.to_numeric(top.get('composite_score', np.nan), errors='coerce')):.3f}"})
    if stat_robustness_summary is not None and (not stat_robustness_summary.empty):
        s0 = stat_robustness_summary.iloc[0]
        dsr = float(pd.to_numeric(s0.get('deflated_sharpe_ratio', np.nan), errors='coerce'))
        p_pos = float(pd.to_numeric(s0.get('prob_cum_return_positive', np.nan), errors='coerce'))
        if np.isfinite(dsr) and dsr < 0.60:
            rows.append({'category': 'robustness', 'priority': 7.5, 'confidence': 0.85, 'recommendation': 'Reduire risque ou ajouter filtres avant scale-up', 'rationale': 'Robustesse statistique insuffisante.', 'expected_impact': f"DSR={dsr:.3f}, P(CumRet>0)={p_pos:.2%}", 'evidence': 'bootstrap+deflated_sharpe'})
    if event_impact is not None and (not event_impact.empty):
        bad_evt = event_impact[pd.to_numeric(event_impact.get('delta_expectancy', np.nan), errors='coerce') < 0]
        if not bad_evt.empty:
            row = bad_evt.sort_values('delta_expectancy').iloc[0]
            rows.append({'category': 'events', 'priority': 5.0, 'confidence': 0.65, 'recommendation': f"Eviter fenetres proches des evenements type={row.get('event_type', 'unknown')}", 'rationale': 'Impact evenementiel negatif observe.', 'expected_impact': f"delta_expectancy={float(pd.to_numeric(row.get('delta_expectancy', np.nan), errors='coerce')):.4f}", 'evidence': f"n_with_event={int(pd.to_numeric(row.get('n_with_event', 0), errors='coerce'))}"})
    if anomaly_impact is not None and (not anomaly_impact.empty):
        bad_anom = anomaly_impact[pd.to_numeric(anomaly_impact.get('delta_expectancy', np.nan), errors='coerce') < 0].copy()
        if not bad_anom.empty:
            row = bad_anom.sort_values('delta_expectancy').iloc[0]
            rows.append({'category': 'anomaly_filter', 'priority': 5.8, 'confidence': 0.70, 'recommendation': f"Appliquer filtre soft sur anomalies liees a {row.get('top_feature', 'unknown')}", 'rationale': 'Certaines anomalies de marche sont associees a une expectancy negative.', 'expected_impact': f"delta_expectancy={float(pd.to_numeric(row.get('delta_expectancy', np.nan), errors='coerce')):.4f}", 'evidence': f"linked_trades={int(pd.to_numeric(row.get('linked_trades', 0), errors='coerce'))}, avg_score={float(pd.to_numeric(row.get('avg_anomaly_score', np.nan), errors='coerce')):.4f}"})
    if regime_performance is not None and (not regime_performance.empty):
        if {'regime_label', 'expectancy'}.issubset(regime_performance.columns):
            bad_reg = regime_performance[pd.to_numeric(regime_performance['expectancy'], errors='coerce') < 0]
            if not bad_reg.empty:
                row = bad_reg.sort_values('expectancy').iloc[0]
                rows.append({'category': 'regime_filter', 'priority': 5.5, 'confidence': 0.70, 'recommendation': f"Filtrer regime {row.get('regime_label', 'unknown')} ou reduire taille", 'rationale': 'Regime structurellement defavorable.', 'expected_impact': f"expectancy={float(pd.to_numeric(row.get('expectancy', np.nan), errors='coerce')):.4f}", 'evidence': f"trades={int(pd.to_numeric(row.get('trades', 0), errors='coerce'))}"})
    if core:
        max_dd = float(pd.to_numeric(core.get('max_drawdown', np.nan), errors='coerce'))
        if np.isfinite(max_dd) and max_dd < -0.25:
            rows.append({'category': 'risk_control', 'priority': 6.5, 'confidence': 0.80, 'recommendation': 'Ajouter hard risk cap (max DD guard) et couper en mode degrade', 'rationale': 'Drawdown observe eleve pour une phase de test.', 'expected_impact': f"max_drawdown={max_dd:.2%}", 'evidence': 'core_metrics'})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=cols)
    out = out.sort_values(['priority', 'confidence'], ascending=[False, False]).head(int(max_items)).reset_index(drop=True)
    out.insert(0, 'rank', np.arange(1, len(out) + 1))
    return out[cols]

def _build_auto_recommendations_viz(recommendations: pd.DataFrame) -> go.Figure:
    if recommendations is None or recommendations.empty:
        return _make_empty_figure('Auto Recommendations', 'No recommendation generated')
    fig = go.Figure(data=[go.Table(header=dict(values=['Rank', 'Category', 'Priority', 'Confidence', 'Recommendation', 'Expected Impact']), cells=dict(values=[recommendations['rank'], recommendations['category'], recommendations['priority'].round(2), recommendations['confidence'].round(2), recommendations['recommendation'], recommendations['expected_impact']]))])
    fig.update_layout(title='Recommendations Priorisees (auto)')
    return fig

def _build_variant_grid_for_cv(base_config: BacktestConfig, max_variants: int, param_grid: dict[str, list[Any]] | None=None) -> list[dict[str, Any]]:
    grid = param_grid or _default_parameter_grid(base_config)
    keys = [k for k in ['ema_fast', 'ema_slow', 'stop_loss_pct', 'take_profit_pct'] if k in grid]
    if len(keys) < 2:
        return [{'variant_name': 'baseline', 'overrides': {}}]
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
    valid = []
    for c in combos:
        if int(c.get('ema_fast', base_config.ema_fast)) >= int(c.get('ema_slow', base_config.ema_slow)):
            continue
        valid.append(c)
    if not valid:
        return [{'variant_name': 'baseline', 'overrides': {}}]
    keep = valid
    if len(keep) > int(max_variants):
        pick = np.linspace(0, len(keep) - 1, num=int(max_variants), dtype=int)
        keep = [keep[int(i)] for i in pick]
    out = []
    for i, params in enumerate(keep):
        name = f"v{i:03d}_ema{int(params['ema_fast'])}_{int(params['ema_slow'])}_sl{float(params['stop_loss_pct']):.4f}_tp{float(params['take_profit_pct']):.4f}"
        out.append({'variant_name': name, 'overrides': params})
    return out

def _strategy_objective(core: dict[str, Any]) -> float:
    shp = float(pd.to_numeric(core.get('sharpe_ratio', np.nan), errors='coerce'))
    ret = float(pd.to_numeric(core.get('cumulative_return', np.nan), errors='coerce'))
    if np.isfinite(shp):
        return shp
    if np.isfinite(ret):
        return ret
    return -1e9

def run_purged_walkforward_cv(base_config: BacktestConfig, market_df: pd.DataFrame, param_grid: dict[str, list[Any]] | None=None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    detail_cols = ['fold_id', 'test_start', 'test_end', 'train_end_idx', 'variant_name', 'train_objective', 'test_objective', 'train_cumulative_return', 'test_cumulative_return', 'train_sharpe', 'test_sharpe', 'train_trades', 'test_trades']
    if market_df is None or market_df.empty:
        return (pd.DataFrame(columns=detail_cols), pd.DataFrame(), {'status': 'empty_input'})
    n = len(market_df)
    n_folds = max(3, int(base_config.purged_cv_folds))
    fold_size = max(40, n // n_folds)
    embargo = max(0, int(base_config.purged_cv_embargo_bars))
    variants = _build_variant_grid_for_cv(base_config, max_variants=int(base_config.purged_cv_max_variants), param_grid=param_grid)
    rows = []
    fold_id = 0
    for test_start in range(0, n, fold_size):
        test_end = min(n, test_start + fold_size)
        if (test_end - test_start) < 30:
            continue
        train_end = max(0, test_start - embargo)
        if train_end < max(60, base_config.ema_slow + 20):
            continue
        train_slice = market_df.iloc[:train_end].copy()
        test_slice = market_df.iloc[test_start:test_end].copy()
        if train_slice.empty or test_slice.empty:
            continue
        for variant in variants:
            cfg = resolve_config(replace(base_config, **variant['overrides']))
            lbl_tr = generate_strategy_labels(train_slice, cfg)
            led_tr, tr_tr, eq_tr, _ = run_backtest_from_labels(train_slice, lbl_tr, cfg)
            core_tr = compute_core_metrics(eq_tr, tr_tr, cfg, bar_ledger=led_tr)
            lbl_te = generate_strategy_labels(test_slice, cfg)
            led_te, tr_te, eq_te, _ = run_backtest_from_labels(test_slice, lbl_te, cfg)
            core_te = compute_core_metrics(eq_te, tr_te, cfg, bar_ledger=led_te)
            rows.append({
                'fold_id': int(fold_id),
                'test_start': test_slice.index.min(),
                'test_end': test_slice.index.max(),
                'train_end_idx': int(train_end),
                'variant_name': str(variant['variant_name']),
                'train_objective': _strategy_objective(core_tr),
                'test_objective': _strategy_objective(core_te),
                'train_cumulative_return': core_tr.get('cumulative_return', np.nan),
                'test_cumulative_return': core_te.get('cumulative_return', np.nan),
                'train_sharpe': core_tr.get('sharpe_ratio', np.nan),
                'test_sharpe': core_te.get('sharpe_ratio', np.nan),
                'train_trades': core_tr.get('total_trades', np.nan),
                'test_trades': core_te.get('total_trades', np.nan),
            })
        fold_id += 1
    detail = pd.DataFrame(rows)
    if detail.empty:
        return (pd.DataFrame(columns=detail_cols), pd.DataFrame(), {'status': 'no_valid_fold'})
    summary_rows = []
    pbo_flags = []
    rank_corrs = []
    for fid, g in detail.groupby('fold_id'):
        gg = g.copy().sort_values('train_objective', ascending=False).reset_index(drop=True)
        gg['train_rank'] = np.arange(1, len(gg) + 1)
        gg['test_rank'] = gg['test_objective'].rank(method='first', ascending=False)
        train_top = gg.iloc[0]
        top_test_rank = int(train_top['test_rank'])
        nvar = int(len(gg))
        pct = float(top_test_rank / max(1, nvar))
        overfit = bool(pct > 0.5)
        pbo_flags.append(float(overfit))
        rank_corr = float(gg['train_rank'].corr(gg['test_rank'], method='spearman')) if nvar >= 3 else np.nan
        rank_corrs.append(rank_corr)
        summary_rows.append({
            'fold_id': int(fid),
            'n_variants': nvar,
            'best_train_variant': str(train_top['variant_name']),
            'best_train_objective': float(train_top['train_objective']),
            'best_train_test_objective': float(train_top['test_objective']),
            'best_train_test_rank': top_test_rank,
            'best_train_test_rank_pct': pct,
            'overfit_flag': overfit,
            'rank_spearman': rank_corr,
        })
    summary = pd.DataFrame(summary_rows).sort_values('fold_id').reset_index(drop=True)
    pbo = float(np.mean(pbo_flags)) if pbo_flags else np.nan
    mean_rank_corr = float(np.nanmean(rank_corrs)) if rank_corrs else np.nan
    robust_score = float((1.0 - pbo) * 0.6 + (max(-1.0, min(1.0, mean_rank_corr)) + 1.0) * 0.2) if np.isfinite(pbo) else np.nan
    verdict = 'go' if (np.isfinite(pbo) and pbo <= 0.45 and np.isfinite(mean_rank_corr) and mean_rank_corr >= 0.0) else 'caution'
    report = {
        'status': 'ok',
        'folds': int(summary['fold_id'].nunique()),
        'variants': int(detail['variant_name'].nunique()),
        'pbo': pbo,
        'mean_rank_spearman': mean_rank_corr,
        'robustness_score': robust_score,
        'verdict': verdict,
        'embargo_bars': int(embargo),
    }
    return (detail, summary, report)

def _build_overfit_guardrails_viz(detail_df: pd.DataFrame, fold_summary_df: pd.DataFrame) -> go.Figure:
    if detail_df is None or detail_df.empty:
        return _make_empty_figure('Overfit Guardrails', 'No purged CV data')
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Train vs Test objective', 'Fold overfit risk'))
    fig.add_trace(go.Scatter(x=detail_df['train_objective'], y=detail_df['test_objective'], mode='markers', marker=dict(size=7, color=detail_df['fold_id'], colorscale='Viridis', showscale=True), text=detail_df['variant_name'], hovertemplate='fold=%{marker.color}<br>train=%{x:.3f}<br>test=%{y:.3f}<br>%{text}<extra></extra>'), row=1, col=1)
    if fold_summary_df is not None and (not fold_summary_df.empty):
        fig.add_trace(go.Bar(x=fold_summary_df['fold_id'].astype(str), y=fold_summary_df['best_train_test_rank_pct'] * 100, marker_color=np.where(fold_summary_df['overfit_flag'], '#dc2626', '#16a34a'), name='Top-IS variant rank pct on OOS'), row=1, col=2)
    fig.update_xaxes(title_text='Train objective', row=1, col=1)
    fig.update_yaxes(title_text='Test objective', row=1, col=1)
    fig.update_yaxes(title_text='Rank percentile (%)', row=1, col=2)
    fig.update_layout(title='Overfitting Guardrails (Purged CV + PBO proxy)')
    return fig

def _execution_fill_model(tx_side: str, row: pd.Series, ref_price: float, mode: str, limit_offset_bps: float) -> tuple[float, bool]:
    if mode == 'market':
        return (float(ref_price), True)
    off = float(limit_offset_bps) / 10000.0
    high = float(pd.to_numeric(row.get('high', ref_price), errors='coerce'))
    low = float(pd.to_numeric(row.get('low', ref_price), errors='coerce'))
    if tx_side == 'buy':
        limit_px = float(ref_price * (1.0 - off))
        filled = bool(low <= limit_px <= high)
        if filled:
            return (limit_px, True)
        return (float(ref_price * (1.0 + off * 0.5)), False)
    limit_px = float(ref_price * (1.0 + off))
    filled = bool(low <= limit_px <= high)
    if filled:
        return (limit_px, True)
    return (float(ref_price * (1.0 - off * 0.5)), False)

def simulate_execution_realism_for_trades(trades: pd.DataFrame, market_df: pd.DataFrame, config: BacktestConfig, mode: str='market', latency_bars: int=1) -> pd.DataFrame:
    cols = ['trade_id', 'entry_time', 'exit_time', 'side', 'direction', 'baseline_net_pnl', 'exec_net_pnl', 'delta_net_pnl', 'fill_ratio', 'entry_latency_bars', 'exit_latency_bars', 'dynamic_slippage_bps_entry', 'dynamic_slippage_bps_exit', 'participation_ratio', 'entry_fill_ok', 'exit_fill_ok', 'execution_mode']
    if trades is None or trades.empty or market_df is None or market_df.empty:
        return pd.DataFrame(columns=cols)
    df = trades.copy()
    mkt = market_df.copy()
    mkt = mkt.sort_index()
    close = pd.to_numeric(mkt['close'], errors='coerce')
    vol = pd.to_numeric(mkt['volume'], errors='coerce') if 'volume' in mkt.columns else pd.Series(1.0, index=mkt.index)
    ret = close.pct_change()
    vol24 = ret.rolling(24, min_periods=6).std().fillna(ret.std(ddof=0) if ret.notna().any() else 0.0)
    idx = mkt.index
    rows = []
    for _, tr in df.iterrows():
        try:
            et = pd.to_datetime(tr['entry_time'], utc=True)
            xt = pd.to_datetime(tr['exit_time'], utc=True)
        except Exception:
            continue
        if pd.isna(et) or pd.isna(xt):
            continue
        epos = int(idx.searchsorted(et))
        xpos = int(idx.searchsorted(xt))
        if epos >= len(idx):
            epos = len(idx) - 1
        if xpos >= len(idx):
            xpos = len(idx) - 1
        efill = min(len(idx) - 1, max(0, epos + int(latency_bars)))
        xfill = min(len(idx) - 1, max(0, xpos + int(latency_bars)))
        erow = mkt.iloc[efill]
        xrow = mkt.iloc[xfill]
        ref_entry = float(pd.to_numeric(erow.get('open', erow.get('close', np.nan)), errors='coerce'))
        ref_exit = float(pd.to_numeric(xrow.get('open', xrow.get('close', np.nan)), errors='coerce'))
        direction = int(pd.to_numeric(tr.get('direction', 0), errors='coerce'))
        entry_side = 'buy' if direction >= 0 else 'sell'
        exit_side = 'sell' if direction >= 0 else 'buy'
        entry_px, entry_ok = _execution_fill_model(entry_side, erow, ref_entry, mode, config.execution_limit_offset_bps)
        exit_px, exit_ok = _execution_fill_model(exit_side, xrow, ref_exit, mode, config.execution_limit_offset_bps)
        capital = float(pd.to_numeric(tr.get('equity_before_entry', np.nan), errors='coerce'))
        if not np.isfinite(capital) or capital <= 0:
            capital = max(1.0, float(pd.to_numeric(tr.get('entry_equity', 1.0), errors='coerce')))
        bar_vol = float(pd.to_numeric(vol.iloc[efill], errors='coerce')) if efill < len(vol) else 1.0
        bar_close = float(pd.to_numeric(close.iloc[efill], errors='coerce')) if efill < len(close) else ref_entry
        participation = capital / max(1.0, bar_vol * max(1e-9, bar_close))
        vol_component_bps = float(pd.to_numeric(vol24.iloc[efill], errors='coerce')) * 10000.0 * float(config.execution_dynamic_slippage_k)
        part_penalty_bps = max(0.0, (participation / float(config.execution_volume_cap_ratio) - 1.0) * 5.0)
        dyn_bps = max(0.0, float(config.slippage_bps) + vol_component_bps + part_penalty_bps)
        entry_exec = apply_slippage(float(entry_px), entry_side, dyn_bps)
        exit_exec = apply_slippage(float(exit_px), exit_side, dyn_bps)
        fill_ratio = min(1.0, float(config.execution_partial_fill_base) / max(1.0, participation / float(config.execution_volume_cap_ratio)))
        fill_ratio = max(0.05, fill_ratio)
        engaged = capital * fill_ratio
        if direction >= 0:
            ret_ratio = exit_exec / max(entry_exec, 1e-12) - 1.0
        else:
            ret_ratio = entry_exec / max(exit_exec, 1e-12) - 1.0
        fee_rate = float(config.fees_bps) / 10000.0
        exec_net_pnl = engaged * ret_ratio - engaged * fee_rate * 2.0
        baseline = float(pd.to_numeric(tr.get('net_pnl', np.nan), errors='coerce'))
        rows.append({
            'trade_id': int(pd.to_numeric(tr.get('trade_id', -1), errors='coerce')),
            'entry_time': et,
            'exit_time': xt,
            'side': tr.get('side', pd.NA),
            'direction': direction,
            'baseline_net_pnl': baseline,
            'exec_net_pnl': float(exec_net_pnl),
            'delta_net_pnl': float(exec_net_pnl - baseline) if np.isfinite(baseline) else np.nan,
            'fill_ratio': float(fill_ratio),
            'entry_latency_bars': int(latency_bars),
            'exit_latency_bars': int(latency_bars),
            'dynamic_slippage_bps_entry': float(dyn_bps),
            'dynamic_slippage_bps_exit': float(dyn_bps),
            'participation_ratio': float(participation),
            'entry_fill_ok': bool(entry_ok),
            'exit_fill_ok': bool(exit_ok),
            'execution_mode': str(mode),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=cols)
    return out

def run_execution_impact_analysis(trades: pd.DataFrame, market_df: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    scenario_rows = []
    detail_frames = []
    baseline_total = float(pd.to_numeric(trades.get('net_pnl', pd.Series(dtype=float)), errors='coerce').sum()) if (trades is not None and not trades.empty and 'net_pnl' in trades.columns) else np.nan
    for mode in ['market', 'limit']:
        for lat in [0, 1, 2]:
            det = simulate_execution_realism_for_trades(trades, market_df, config, mode=mode, latency_bars=int(lat))
            if det.empty:
                continue
            total_exec = float(det['exec_net_pnl'].sum())
            total_delta = float(det['delta_net_pnl'].sum())
            scenario_rows.append({
                'execution_mode': mode,
                'latency_bars': int(lat),
                'trades': int(len(det)),
                'baseline_total_pnl': baseline_total,
                'exec_total_pnl': total_exec,
                'delta_total_pnl': total_delta,
                'baseline_expectancy': float(det['baseline_net_pnl'].mean()),
                'exec_expectancy': float(det['exec_net_pnl'].mean()),
                'exec_win_rate': float((det['exec_net_pnl'] > 0).mean()),
                'mean_fill_ratio': float(det['fill_ratio'].mean()),
                'mean_dynamic_slippage_bps': float(det[['dynamic_slippage_bps_entry', 'dynamic_slippage_bps_exit']].mean().mean()),
                'fill_success_rate': float((det['entry_fill_ok'] & det['exit_fill_ok']).mean()),
            })
            detail_frames.append(det.assign(scenario=f'{mode}_lat{lat}'))
    scen = pd.DataFrame(scenario_rows).sort_values(['exec_total_pnl', 'mean_fill_ratio'], ascending=[False, False]).reset_index(drop=True) if scenario_rows else pd.DataFrame()
    detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    report = {'status': 'ok' if not scen.empty else 'empty', 'scenarios': int(len(scen)), 'baseline_total_pnl': baseline_total}
    if not scen.empty:
        best = scen.iloc[0]
        report.update({
            'best_scenario': f"{best['execution_mode']}_lat{int(best['latency_bars'])}",
            'best_exec_total_pnl': float(best['exec_total_pnl']),
            'edge_retention_ratio': float(best['exec_total_pnl'] / baseline_total) if np.isfinite(baseline_total) and baseline_total != 0 else np.nan,
            'execution_drag_total_pnl': float(baseline_total - best['exec_total_pnl']) if np.isfinite(baseline_total) else np.nan,
        })
    return (scen, detail, report)

def _build_execution_waterfall_viz(execution_summary: pd.DataFrame) -> go.Figure:
    if execution_summary is None or execution_summary.empty:
        return _make_empty_figure('Execution Waterfall', 'No execution impact data')
    best = execution_summary.iloc[0]
    baseline = float(pd.to_numeric(best.get('baseline_total_pnl', np.nan), errors='coerce'))
    exec_total = float(pd.to_numeric(best.get('exec_total_pnl', np.nan), errors='coerce'))
    drag = baseline - exec_total if np.isfinite(baseline) and np.isfinite(exec_total) else np.nan
    fig = go.Figure()
    fig.add_trace(go.Bar(x=['Baseline edge', 'Execution drag', 'Net edge'], y=[baseline, -drag if np.isfinite(drag) else np.nan, exec_total], marker_color=['#16a34a', '#dc2626', '#2563eb']))
    fig.update_layout(title='Execution Impact Waterfall (best scenario)', yaxis_title='PnL')
    return fig

def build_trade_replay_root_cause(trade_df: pd.DataFrame, market_df: pd.DataFrame, max_trades: int=300) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = ['trade_id', 'entry_time', 'exit_time', 'side', 'direction', 'duration_bars', 'duration_hours', 'return_pct', 'net_pnl', 'entry_price', 'exit_price', 'total_fees', 'mfe_pct', 'mae_pct', 'vol_during_trade', 'volume_mean', 'exit_reason', 'root_cause']
    if trade_df is None or trade_df.empty or market_df is None or market_df.empty:
        return (pd.DataFrame(columns=cols), pd.DataFrame(), pd.DataFrame())
    mkt = market_df.sort_index()
    close = pd.to_numeric(mkt['close'], errors='coerce')
    ret = close.pct_change().fillna(0.0)
    tdf = trade_df.copy()
    tdf['entry_time'] = pd.to_datetime(tdf['entry_time'], utc=True, errors='coerce')
    tdf['exit_time'] = pd.to_datetime(tdf['exit_time'], utc=True, errors='coerce')
    tdf = tdf.dropna(subset=['entry_time', 'exit_time']).sort_values('entry_time').tail(int(max_trades)).reset_index(drop=True)
    detail_rows = []
    path_rows = []
    idx = mkt.index
    for _, tr in tdf.iterrows():
        tid = int(pd.to_numeric(tr.get('trade_id', -1), errors='coerce'))
        et = tr['entry_time']
        xt = tr['exit_time']
        s = int(idx.searchsorted(et))
        e = int(idx.searchsorted(xt))
        s = min(max(0, s), len(idx) - 1)
        e = min(max(0, e), len(idx) - 1)
        if e <= s:
            continue
        seg = mkt.iloc[s:e + 1].copy()
        direction = int(pd.to_numeric(tr.get('direction', 1), errors='coerce'))
        entry_px = float(pd.to_numeric(tr.get('entry_price', seg.iloc[0]['close']), errors='coerce'))
        if direction >= 0:
            rel_close = seg['close'] / max(entry_px, 1e-12) - 1.0
            mfe = float((seg['high'] / max(entry_px, 1e-12) - 1.0).max())
            mae = float((seg['low'] / max(entry_px, 1e-12) - 1.0).min())
        else:
            rel_close = entry_px / seg['close'].replace(0, np.nan) - 1.0
            mfe = float((entry_px / seg['low'].replace(0, np.nan) - 1.0).max())
            mae = float((entry_px / seg['high'].replace(0, np.nan) - 1.0).min())
        seg_ret = ret.iloc[s:e + 1]
        vol_trade = float(seg_ret.std(ddof=0))
        vol_mean = float(pd.to_numeric(seg.get('volume', pd.Series(dtype=float)), errors='coerce').mean()) if 'volume' in seg.columns else np.nan
        pnl = float(pd.to_numeric(tr.get('net_pnl', np.nan), errors='coerce'))
        win = bool(pnl > 0)
        exit_reason = str(tr.get('exit_reason', 'unknown'))
        if (not win) and ('stop_loss' in exit_reason):
            cause = 'stop_loss_hit'
        elif (not win) and (mae < -abs(mfe) * 0.8):
            cause = 'adverse_move'
        elif (not win) and (int(pd.to_numeric(tr.get('duration_bars', 0), errors='coerce')) <= 2):
            cause = 'fast_reversal'
        elif win and ('eod' in exit_reason):
            cause = 'late_or_time_exit'
        elif win:
            cause = 'trend_followthrough'
        else:
            cause = 'mixed'
        detail_rows.append({
            'trade_id': tid,
            'entry_time': et,
            'exit_time': xt,
            'side': tr.get('side', pd.NA),
            'direction': direction,
            'duration_bars': int(pd.to_numeric(tr.get('duration_bars', np.nan), errors='coerce')) if pd.notna(tr.get('duration_bars', np.nan)) else np.nan,
            'duration_hours': float(pd.to_numeric(tr.get('duration_hours', np.nan), errors='coerce')),
            'return_pct': float(pd.to_numeric(tr.get('return_pct', np.nan), errors='coerce')),
            'net_pnl': pnl,
            'entry_price': entry_px,
            'exit_price': float(pd.to_numeric(tr.get('exit_price', np.nan), errors='coerce')),
            'total_fees': float(pd.to_numeric(tr.get('total_fees', np.nan), errors='coerce')),
            'mfe_pct': mfe,
            'mae_pct': mae,
            'vol_during_trade': vol_trade,
            'volume_mean': vol_mean,
            'exit_reason': exit_reason,
            'root_cause': cause,
        })
        for ts, rv in rel_close.items():
            path_rows.append({'trade_id': tid, 'timestamp': ts, 'path_return_pct': float(rv), 'root_cause': cause, 'is_winner': win})
    detail = pd.DataFrame(detail_rows)
    path = pd.DataFrame(path_rows)
    summary = pd.DataFrame()
    if not detail.empty:
        summary = detail.groupby('root_cause').agg(trades=('trade_id', 'count'), win_rate=('net_pnl', lambda s: float((s > 0).mean())), expectancy=('net_pnl', 'mean'), avg_mfe=('mfe_pct', 'mean'), avg_mae=('mae_pct', 'mean')).reset_index().sort_values('trades', ascending=False).reset_index(drop=True)
    return (detail, path, summary)

def _build_trade_replay_viz(trade_root_cause: pd.DataFrame, trade_replay_paths: pd.DataFrame) -> go.Figure:
    if trade_root_cause is None or trade_root_cause.empty:
        return _make_empty_figure('Trade Replay', 'No trade replay data')
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Root cause summary', 'Replay paths (sample)'))
    rc = trade_root_cause.groupby('root_cause').agg(expectancy=('net_pnl', 'mean'), trades=('trade_id', 'count')).reset_index()
    fig.add_trace(go.Bar(x=rc['root_cause'], y=rc['expectancy'], marker_color=np.where(rc['expectancy'] >= 0, '#16a34a', '#dc2626')), row=1, col=1)
    if trade_replay_paths is not None and (not trade_replay_paths.empty):
        sample_ids = trade_replay_paths['trade_id'].dropna().astype(int).drop_duplicates().head(6).tolist()
        samp = trade_replay_paths[trade_replay_paths['trade_id'].astype(int).isin(sample_ids)].copy()
        for tid, g in samp.groupby('trade_id'):
            fig.add_trace(go.Scatter(x=g['timestamp'], y=g['path_return_pct'] * 100, mode='lines', name=f'trade {tid}', line=dict(width=1)), row=1, col=2)
    fig.update_yaxes(title_text='Expectancy', row=1, col=1)
    fig.update_yaxes(title_text='Path return %', row=1, col=2)
    fig.update_layout(title='Trade Replay / Root-Cause Explorer')
    return fig

def run_multi_asset_portfolio_pipeline(base_config: BacktestConfig, symbols: tuple[str, ...] | list[str], market_data_map: dict[str, pd.DataFrame] | None=None) -> dict[str, Any]:
    syms = [str(s) for s in symbols if str(s)]
    if not syms:
        return {'portfolio_equity': pd.DataFrame(), 'portfolio_returns': pd.DataFrame(), 'portfolio_attribution': pd.DataFrame(), 'portfolio_asset_metrics': pd.DataFrame(), 'portfolio_correlation': pd.DataFrame(), 'portfolio_report': {'status': 'no_symbols'}, 'asset_runs': {}}
    asset_runs: dict[str, dict[str, Any]] = {}
    ret_cols = []
    for sym in syms[: max(1, len(syms))]:
        cfg_sym = resolve_config(replace(base_config, symbol=sym, portfolio_symbols=()))
        if market_data_map is not None and sym in market_data_map:
            raw = market_data_map[sym].copy()
            assert_market_data_integrity(raw)
            splits = split_time_series(raw, ratios=cfg_sym.split_ratios)
            train_df = splits['train']
            labels = generate_strategy_labels(train_df, cfg_sym)
            signals = labels_to_signals(labels)
            bar_ledger, trades, equity_curve, positions = run_backtest_from_labels(train_df, labels, cfg_sym)
            drawdown = compute_drawdown_series(equity_curve)
            core = compute_core_metrics(equity_curve, trades, cfg_sym, bar_ledger=bar_ledger)
            time_decomp = compute_time_decomposition(equity_curve)
            run_obj = {'raw_market': raw, 'splits': splits, 'labels': labels, 'signals': signals, 'bar_ledger': bar_ledger, 'trades': trades, 'equity_curve': equity_curve, 'positions': positions, 'core_metrics': core, 'time_decomposition': time_decomp, 'time_decomposition_flat': flatten_time_decomposition(time_decomp), 'drawdown_series': drawdown, 'viz_bundle': {}}
        else:
            run_obj, _ = execute_first_check_pipeline(cfg_sym, persist=False, render=False)
        asset_runs[sym] = run_obj
        eq = run_obj['equity_curve'].copy()
        ret = eq['equity'].pct_change().fillna(0.0).rename(sym)
        ret_cols.append(ret)
    if not ret_cols:
        return {'portfolio_equity': pd.DataFrame(), 'portfolio_returns': pd.DataFrame(), 'portfolio_attribution': pd.DataFrame(), 'portfolio_asset_metrics': pd.DataFrame(), 'portfolio_correlation': pd.DataFrame(), 'portfolio_report': {'status': 'no_equity'}, 'asset_runs': asset_runs}
    rets = pd.concat(ret_cols, axis=1, join='outer').sort_index()
    avail = rets.notna().astype(float)
    weights = avail.div(avail.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    weighted = rets.fillna(0.0) * weights
    p_ret = weighted.sum(axis=1)
    p_eq = (1.0 + p_ret).cumprod() * float(base_config.initial_capital)
    portfolio_equity = pd.DataFrame({'equity': p_eq}, index=rets.index)
    portfolio_returns = pd.DataFrame({'portfolio_return': p_ret}, index=rets.index)
    attrib = weighted.copy()
    attrib.columns = [f'contrib_{c}' for c in attrib.columns]
    attrib['portfolio_return'] = p_ret
    asset_metrics_rows = []
    for sym in rets.columns:
        s = rets[sym].dropna()
        vol = float(s.std(ddof=0))
        sharpe = float(np.sqrt(base_config.annualization_factor) * s.mean() / vol) if vol > 0 else np.nan
        asset_metrics_rows.append({'symbol': sym, 'bars': int(len(s)), 'cumulative_return': float((1.0 + s).prod() - 1.0), 'mean_return': float(s.mean()), 'vol': vol, 'sharpe': sharpe, 'contribution_sum': float(weighted[sym].sum())})
    asset_metrics = pd.DataFrame(asset_metrics_rows).sort_values('contribution_sum', ascending=False).reset_index(drop=True)
    corr = rets.corr()
    report = {'status': 'ok', 'symbols': syms, 'bars': int(len(rets)), 'portfolio_cumulative_return': float(portfolio_equity['equity'].iloc[-1] / portfolio_equity['equity'].iloc[0] - 1.0), 'mean_correlation': float(corr.where(~np.eye(len(corr), dtype=bool)).stack().mean()) if (corr is not None and not corr.empty) else np.nan}
    return {'portfolio_equity': portfolio_equity, 'portfolio_returns': portfolio_returns, 'portfolio_attribution': attrib.reset_index().rename(columns={'index': 'timestamp'}), 'portfolio_asset_metrics': asset_metrics, 'portfolio_correlation': corr, 'portfolio_report': report, 'asset_runs': asset_runs}

def _build_portfolio_dashboard(portfolio_equity: pd.DataFrame, portfolio_asset_metrics: pd.DataFrame, portfolio_correlation: pd.DataFrame) -> go.Figure:
    if portfolio_equity is None or portfolio_equity.empty:
        return _make_empty_figure('Portfolio Dashboard', 'No portfolio data')
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Portfolio equity', 'Asset contribution', 'Correlation matrix', 'Asset sharpe'))
    fig.add_trace(go.Scatter(x=portfolio_equity.index, y=portfolio_equity['equity'], mode='lines', name='Portfolio'), row=1, col=1)
    if portfolio_asset_metrics is not None and (not portfolio_asset_metrics.empty):
        fig.add_trace(go.Bar(x=portfolio_asset_metrics['symbol'], y=portfolio_asset_metrics['contribution_sum'], name='Contribution'), row=1, col=2)
        fig.add_trace(go.Bar(x=portfolio_asset_metrics['symbol'], y=portfolio_asset_metrics['sharpe'], name='Sharpe'), row=2, col=2)
    if portfolio_correlation is not None and (not portfolio_correlation.empty):
        fig.add_trace(go.Heatmap(z=portfolio_correlation.values, x=list(portfolio_correlation.columns), y=list(portfolio_correlation.index), colorscale='RdBu', zmid=0), row=2, col=1)
    fig.update_layout(height=1100, title='Multi-Asset Portfolio Dashboard')
    return fig

def _safe_standardize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    mu = float(s.mean()) if len(s) else np.nan
    sigma = float(s.std(ddof=0)) if len(s) else np.nan
    if (not np.isfinite(sigma)) or sigma <= 0:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma

def _context_action_hint(rule_type: str) -> str:
    mapping = {
        'entry_hour': 'Ajouter un filtre horaire ou baisser la taille de position sur ces heures.',
        'entry_day_name': 'Limiter le trading ce jour ou renforcer la confirmation de signal.',
        'entry_regime': 'Activer un filtre regime-aware avant l entree.',
        'event_count_near_trade': 'Neutraliser les entrees proches des evenements.',
        'anomaly_count_near_trade': 'Reduire la taille ou sauter les signaux pendant anomalies de marche.',
        'has_anomaly_near_trade': 'Appliquer un gate take/skip conditionne par le score d anomalie.',
        'anomaly_top_feature': 'Diagnostiquer la feature d anomalie dominante et ajouter un filtre dedie.',
        'exit_reason': 'Reviser la logique de sortie et le couple SL/TP.',
    }
    return mapping.get(rule_type, 'Appliquer un filtre conditionnel et re-evaluer en walk-forward.')

def build_failure_diagnosis_report(trade_df: pd.DataFrame, core_metrics: dict[str, Any] | None=None, avoidance_rules: pd.DataFrame | None=None, regime_performance: pd.DataFrame | None=None, event_impact: pd.DataFrame | None=None, min_trades: int=8, top_n: int=25) -> pd.DataFrame:
    cols = ['rank', 'dimension', 'context', 'trades', 'win_rate', 'expectancy', 'loss_rate', 'delta_expectancy', 'severity', 'action_hint', 'evidence']
    if trade_df is None or trade_df.empty:
        return pd.DataFrame(columns=cols)
    df = trade_df.copy()
    if 'is_winner' not in df.columns and 'net_pnl' in df.columns:
        df['is_winner'] = (pd.to_numeric(df['net_pnl'], errors='coerce') > 0).astype(int)
    if 'is_winner' not in df.columns or 'net_pnl' not in df.columns:
        return pd.DataFrame(columns=cols)
    baseline_wr = float(pd.to_numeric(df['is_winner'], errors='coerce').mean())
    baseline_exp = float(pd.to_numeric(df['net_pnl'], errors='coerce').mean())
    rows: list[dict[str, Any]] = []
    candidate_cols = [c for c in ['entry_hour', 'entry_day_name', 'entry_regime', 'event_count_near_trade', 'exit_reason'] if c in df.columns]
    for col in candidate_cols:
        grp = df.groupby(col).agg(trades=('trade_id', 'count'), win_rate=('is_winner', 'mean'), expectancy=('net_pnl', 'mean'), loss_rate=('is_winner', lambda s: float((1 - s).mean()))).reset_index()
        grp = grp[grp['trades'] >= int(min_trades)].copy()
        if grp.empty:
            continue
        for _, row in grp.iterrows():
            expectancy = float(row['expectancy'])
            delta_exp = float(expectancy - baseline_exp)
            if not np.isfinite(delta_exp) or delta_exp >= 0:
                continue
            trades = int(row['trades'])
            severity = float(abs(delta_exp) * math.sqrt(max(1, trades)))
            context = str(row[col])
            evidence = f"baseline_exp={baseline_exp:.4f}, baseline_wr={baseline_wr:.2%}"
            rows.append({'dimension': str(col), 'context': context, 'trades': trades, 'win_rate': float(row['win_rate']), 'expectancy': expectancy, 'loss_rate': float(row['loss_rate']), 'delta_expectancy': delta_exp, 'severity': severity, 'action_hint': _context_action_hint(str(col)), 'evidence': evidence})
    if avoidance_rules is not None and (not avoidance_rules.empty):
        tmp = avoidance_rules.copy()
        for _, row in tmp.head(top_n).iterrows():
            delta_exp = float(pd.to_numeric(row.get('delta_expectancy', np.nan), errors='coerce'))
            if not np.isfinite(delta_exp):
                continue
            trades = int(pd.to_numeric(row.get('trades', 0), errors='coerce'))
            severity = float(abs(min(0.0, delta_exp)) * math.sqrt(max(1, trades)))
            rows.append({'dimension': str(row.get('rule_type', 'rule')), 'context': str(row.get('rule_value', 'unknown')), 'trades': trades, 'win_rate': float(pd.to_numeric(row.get('win_rate', np.nan), errors='coerce')), 'expectancy': float(pd.to_numeric(row.get('expectancy', np.nan), errors='coerce')), 'loss_rate': float(1.0 - pd.to_numeric(row.get('win_rate', np.nan), errors='coerce')) if pd.notna(row.get('win_rate', np.nan)) else np.nan, 'delta_expectancy': delta_exp, 'severity': severity, 'action_hint': _context_action_hint(str(row.get('rule_type', 'rule'))), 'evidence': 'derive_avoidance_rules'})
    if regime_performance is not None and (not regime_performance.empty):
        rp = regime_performance.copy()
        if {'regime_label', 'expectancy', 'trades'}.issubset(rp.columns):
            bad_rp = rp[pd.to_numeric(rp['expectancy'], errors='coerce') < baseline_exp].copy()
            for _, row in bad_rp.iterrows():
                trades = int(pd.to_numeric(row.get('trades', 0), errors='coerce'))
                delta_exp = float(pd.to_numeric(row.get('expectancy', np.nan), errors='coerce') - baseline_exp)
                severity = float(abs(min(0.0, delta_exp)) * math.sqrt(max(1, trades)))
                rows.append({'dimension': 'entry_regime', 'context': str(row.get('regime_label', 'unknown')), 'trades': trades, 'win_rate': float(pd.to_numeric(row.get('win_rate', np.nan), errors='coerce')), 'expectancy': float(pd.to_numeric(row.get('expectancy', np.nan), errors='coerce')), 'loss_rate': np.nan, 'delta_expectancy': delta_exp, 'severity': severity, 'action_hint': _context_action_hint('entry_regime'), 'evidence': 'regime_performance'})
    if event_impact is not None and (not event_impact.empty):
        ei = event_impact.copy()
        if {'event_type', 'delta_expectancy', 'n_with_event'}.issubset(ei.columns):
            bad_ei = ei[pd.to_numeric(ei['delta_expectancy'], errors='coerce') < 0].copy()
            for _, row in bad_ei.iterrows():
                trades = int(pd.to_numeric(row.get('n_with_event', 0), errors='coerce'))
                delta_exp = float(pd.to_numeric(row.get('delta_expectancy', np.nan), errors='coerce'))
                severity = float(abs(min(0.0, delta_exp)) * math.sqrt(max(1, trades)))
                rows.append({'dimension': 'event_type', 'context': str(row.get('event_type', 'unknown')), 'trades': trades, 'win_rate': np.nan, 'expectancy': float(pd.to_numeric(row.get('expectancy_with_event', np.nan), errors='coerce')), 'loss_rate': np.nan, 'delta_expectancy': delta_exp, 'severity': severity, 'action_hint': _context_action_hint('event_count_near_trade'), 'evidence': 'event_impact'})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=cols)
    out = out.sort_values(['severity', 'delta_expectancy', 'trades'], ascending=[False, True, False]).drop_duplicates(subset=['dimension', 'context'], keep='first').head(int(top_n)).reset_index(drop=True)
    out.insert(0, 'rank', np.arange(1, len(out) + 1))
    if core_metrics:
        global_note = f"global: sharpe={core_metrics.get('sharpe_ratio', np.nan)}, max_dd={core_metrics.get('max_drawdown', np.nan)}, win_rate={core_metrics.get('win_rate', np.nan)}"
        out['evidence'] = out['evidence'].astype(str) + ' | ' + global_note
    return out[cols]

def _build_failure_diagnosis_viz(failure_df: pd.DataFrame) -> go.Figure:
    if failure_df is None or failure_df.empty:
        return _make_empty_figure('Failure Diagnosis', 'No failure contexts identified')
    top = failure_df.head(15).copy()
    top['label'] = top['dimension'].astype(str) + ' :: ' + top['context'].astype(str)
    fig = px.bar(top.sort_values('severity', ascending=True), x='severity', y='label', orientation='h', color='delta_expectancy', color_continuous_scale='Reds', title='Top contextes a risque (diagnostic auto)', hover_data=['trades', 'win_rate', 'expectancy', 'action_hint'])
    fig.update_layout(yaxis_title='Contexte', xaxis_title='Severity score')
    return fig

def compute_deflated_sharpe_ratio(equity_curve: pd.DataFrame, config: BacktestConfig, n_trials: int=1) -> dict[str, float]:
    out = {'sharpe_ratio': np.nan, 'sigma_sr': np.nan, 'skew': np.nan, 'kurtosis': np.nan, 'n_returns': 0.0, 'n_trials': float(max(1, int(n_trials))), 'sr_star': np.nan, 'psr_vs_zero': np.nan, 'deflated_sharpe_ratio': np.nan}
    if equity_curve is None or equity_curve.empty:
        return out
    eq = pd.to_numeric(equity_curve['equity'], errors='coerce').dropna()
    ret = eq.pct_change().dropna()
    n = int(len(ret))
    out['n_returns'] = float(n)
    if n < 5:
        return out
    excess = ret - config.risk_free_rate / config.annualization_factor
    sr_std = float(np.std(excess, ddof=0))
    sr = float(np.sqrt(config.annualization_factor) * np.mean(excess) / sr_std) if sr_std > 0 else np.nan
    skew = float(pd.Series(excess).skew()) if n > 2 else 0.0
    kurt_excess = float(pd.Series(excess).kurt()) if n > 3 else 0.0
    kurtosis = float(kurt_excess + 3.0)
    if not np.isfinite(skew):
        skew = 0.0
    if (not np.isfinite(kurtosis)) or kurtosis <= 0:
        kurtosis = 3.0
    denom = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * (sr**2) if np.isfinite(sr) else np.nan
    sigma_sr = float(np.sqrt(denom / max(1, n - 1))) if (np.isfinite(denom) and denom > 0) else np.nan
    out.update({'sharpe_ratio': sr, 'sigma_sr': sigma_sr, 'skew': skew, 'kurtosis': kurtosis})
    if not np.isfinite(sigma_sr) or sigma_sr <= 0:
        return out
    nd = NormalDist()
    psr_vs_zero = float(nd.cdf(sr / sigma_sr))
    k = max(1, int(n_trials))
    if k <= 1:
        sr_star = 0.0
    else:
        gamma = 0.5772156649
        q1 = min(max(1e-12, 1.0 - 1.0 / k), 1.0 - 1e-12)
        q2 = min(max(1e-12, 1.0 - 1.0 / (k * math.e)), 1.0 - 1e-12)
        sr_star = float(sigma_sr * ((1.0 - gamma) * nd.inv_cdf(q1) + gamma * nd.inv_cdf(q2)))
    dsr = float(nd.cdf((sr - sr_star) / sigma_sr))
    out.update({'sr_star': sr_star, 'psr_vs_zero': psr_vs_zero, 'deflated_sharpe_ratio': dsr})
    return out

def run_bootstrap_robustness(equity_curve: pd.DataFrame, config: BacktestConfig, observed_metrics: dict[str, Any] | None=None, n_trials: int=1, n_bootstrap: int | None=None, random_state: int | None=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_cols = ['bootstrap_runs', 'observed_cumulative_return', 'observed_sharpe_ratio', 'observed_max_drawdown', 'cum_return_p05', 'cum_return_p50', 'cum_return_p95', 'sharpe_p05', 'sharpe_p50', 'sharpe_p95', 'max_dd_p05', 'max_dd_p50', 'max_dd_p95', 'prob_cum_return_positive', 'prob_sharpe_positive', 'deflated_sharpe_ratio', 'psr_vs_zero', 'sr_star', 'n_trials']
    if equity_curve is None or equity_curve.empty:
        return (pd.DataFrame(columns=summary_cols), pd.DataFrame())
    obs = observed_metrics or {}
    eq = pd.to_numeric(equity_curve['equity'], errors='coerce').dropna()
    ret = eq.pct_change().dropna()
    if len(ret) < 5:
        return (pd.DataFrame(columns=summary_cols), pd.DataFrame())
    vals = ret.to_numpy(dtype=float)
    n = int(len(vals))
    n_boot = int(n_bootstrap if n_bootstrap is not None else config.bootstrap_iterations)
    seed = int(random_state if random_state is not None else config.bootstrap_seed)
    rng = np.random.default_rng(seed)
    initial_capital = float(eq.iloc[0])
    rows = []
    for idx in range(n_boot):
        pick = rng.integers(0, n, size=n)
        sample_ret = vals[pick]
        equity_path = initial_capital * np.cumprod(1.0 + sample_ret)
        cum_ret = float(equity_path[-1] / initial_capital - 1.0)
        ex = sample_ret - config.risk_free_rate / config.annualization_factor
        ex_std = float(np.std(ex, ddof=0))
        sharpe = float(np.sqrt(config.annualization_factor) * np.mean(ex) / ex_std) if ex_std > 0 else np.nan
        running_peak = np.maximum.accumulate(equity_path)
        dd = equity_path / running_peak - 1.0
        max_dd = float(np.min(dd))
        rows.append({'bootstrap_id': int(idx), 'cumulative_return': cum_ret, 'sharpe_ratio': sharpe, 'max_drawdown': max_dd, 'win_rate_bars': float(np.mean(sample_ret > 0))})
    samples = pd.DataFrame(rows)
    dsr_info = compute_deflated_sharpe_ratio(equity_curve, config, n_trials=max(1, int(n_trials)))
    summary = pd.DataFrame([{'bootstrap_runs': int(len(samples)), 'observed_cumulative_return': float(pd.to_numeric(obs.get('cumulative_return', np.nan), errors='coerce')), 'observed_sharpe_ratio': float(pd.to_numeric(obs.get('sharpe_ratio', np.nan), errors='coerce')), 'observed_max_drawdown': float(pd.to_numeric(obs.get('max_drawdown', np.nan), errors='coerce')), 'cum_return_p05': float(samples['cumulative_return'].quantile(0.05)), 'cum_return_p50': float(samples['cumulative_return'].quantile(0.50)), 'cum_return_p95': float(samples['cumulative_return'].quantile(0.95)), 'sharpe_p05': float(samples['sharpe_ratio'].quantile(0.05)), 'sharpe_p50': float(samples['sharpe_ratio'].quantile(0.50)), 'sharpe_p95': float(samples['sharpe_ratio'].quantile(0.95)), 'max_dd_p05': float(samples['max_drawdown'].quantile(0.05)), 'max_dd_p50': float(samples['max_drawdown'].quantile(0.50)), 'max_dd_p95': float(samples['max_drawdown'].quantile(0.95)), 'prob_cum_return_positive': float((samples['cumulative_return'] > 0).mean()), 'prob_sharpe_positive': float((samples['sharpe_ratio'] > 0).mean()), 'deflated_sharpe_ratio': float(pd.to_numeric(dsr_info.get('deflated_sharpe_ratio', np.nan), errors='coerce')), 'psr_vs_zero': float(pd.to_numeric(dsr_info.get('psr_vs_zero', np.nan), errors='coerce')), 'sr_star': float(pd.to_numeric(dsr_info.get('sr_star', np.nan), errors='coerce')), 'n_trials': float(pd.to_numeric(dsr_info.get('n_trials', np.nan), errors='coerce'))}])
    return (summary, samples)

def _build_stat_robustness_viz(summary_df: pd.DataFrame, samples_df: pd.DataFrame) -> go.Figure:
    if samples_df is None or samples_df.empty:
        return _make_empty_figure('Statistical Robustness', 'No bootstrap samples')
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Bootstrap cumulative return', 'Bootstrap max drawdown'))
    fig.add_trace(go.Histogram(x=samples_df['cumulative_return'] * 100, nbinsx=40, name='CumRet %'), row=1, col=1)
    fig.add_trace(go.Histogram(x=samples_df['max_drawdown'] * 100, nbinsx=40, name='MaxDD %'), row=1, col=2)
    if summary_df is not None and (not summary_df.empty):
        s0 = summary_df.iloc[0]
        txt = f"DSR={s0.get('deflated_sharpe_ratio', np.nan):.3f}<br>Prob(CumRet>0)={s0.get('prob_cum_return_positive', np.nan):.2%}<br>Prob(Sharpe>0)={s0.get('prob_sharpe_positive', np.nan):.2%}"
        fig.add_annotation(text=txt, showarrow=False, x=0.5, y=1.13, xref='paper', yref='paper')
    fig.update_layout(title='Robustesse Statistique (Bootstrap + DSR)')
    return fig

def _default_parameter_grid(config: BacktestConfig) -> dict[str, list[Any]]:
    ema_fast = sorted({max(2, int(config.ema_fast * 0.7)), int(config.ema_fast), int(config.ema_fast * 1.3)})
    ema_slow = sorted({max(max(ema_fast) + 2, int(config.ema_slow * 0.7)), int(config.ema_slow), int(config.ema_slow * 1.3)})
    stop_loss = sorted({max(0.001, round(float(config.stop_loss_pct) * 0.5, 4)), round(float(config.stop_loss_pct), 4), max(0.001, round(float(config.stop_loss_pct) * 1.5, 4))})
    take_profit = sorted({max(0.001, round(float(config.take_profit_pct) * 0.5, 4)), round(float(config.take_profit_pct), 4), max(0.001, round(float(config.take_profit_pct) * 1.5, 4))})
    return {'ema_fast': ema_fast, 'ema_slow': ema_slow, 'stop_loss_pct': stop_loss, 'take_profit_pct': take_profit}

def run_parameter_sensitivity(base_config: BacktestConfig, market_df: pd.DataFrame, param_grid: dict[str, list[Any]] | None=None, max_combinations: int | None=None) -> pd.DataFrame:
    if market_df is None or market_df.empty:
        return pd.DataFrame()
    grid = param_grid or _default_parameter_grid(base_config)
    keys = [k for k in ['ema_fast', 'ema_slow', 'stop_loss_pct', 'take_profit_pct'] if k in grid]
    if len(keys) < 2:
        return pd.DataFrame()
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
    max_n = int(max_combinations if max_combinations is not None else base_config.sensitivity_max_combinations)
    if len(combos) > max_n:
        keep_idx = np.linspace(0, len(combos) - 1, num=max_n, dtype=int)
        combos = [combos[int(i)] for i in keep_idx]
    splits = split_time_series(market_df, ratios=base_config.split_ratios)
    train_df = splits['train']
    rows = []
    for cid, params in enumerate(combos):
        ema_fast = int(params.get('ema_fast', base_config.ema_fast))
        ema_slow = int(params.get('ema_slow', base_config.ema_slow))
        if ema_fast >= ema_slow:
            continue
        cfg = resolve_config(replace(base_config, ema_fast=ema_fast, ema_slow=ema_slow, stop_loss_pct=float(params.get('stop_loss_pct', base_config.stop_loss_pct)), take_profit_pct=float(params.get('take_profit_pct', base_config.take_profit_pct))))
        labels = generate_strategy_labels(train_df, cfg)
        ledger, trades, eq, _ = run_backtest_from_labels(train_df, labels, cfg)
        core = compute_core_metrics(eq, trades, cfg, bar_ledger=ledger)
        rows.append({'combo_id': int(cid), 'ema_fast': int(cfg.ema_fast), 'ema_slow': int(cfg.ema_slow), 'stop_loss_pct': float(cfg.stop_loss_pct), 'take_profit_pct': float(cfg.take_profit_pct), 'total_pnl': core.get('total_pnl', np.nan), 'cumulative_return': core.get('cumulative_return', np.nan), 'sharpe_ratio': core.get('sharpe_ratio', np.nan), 'max_drawdown': core.get('max_drawdown', np.nan), 'win_rate': core.get('win_rate', np.nan), 'profit_factor': core.get('profit_factor', np.nan), 'total_trades': core.get('total_trades', np.nan)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out['ret_z'] = _safe_standardize(out['cumulative_return'])
    out['sharpe_z'] = _safe_standardize(out['sharpe_ratio'])
    out['winrate_z'] = _safe_standardize(out['win_rate'])
    out['dd_z'] = _safe_standardize(-pd.to_numeric(out['max_drawdown'], errors='coerce'))
    out['composite_score'] = 0.35 * out['ret_z'] + 0.35 * out['sharpe_z'] + 0.20 * out['winrate_z'] + 0.10 * out['dd_z']
    out = out.sort_values(['composite_score', 'cumulative_return', 'sharpe_ratio'], ascending=[False, False, False]).reset_index(drop=True)
    out['rank'] = np.arange(1, len(out) + 1)
    baseline_mask = (out['ema_fast'] == int(base_config.ema_fast)) & (out['ema_slow'] == int(base_config.ema_slow)) & (np.isclose(out['stop_loss_pct'], float(base_config.stop_loss_pct))) & (np.isclose(out['take_profit_pct'], float(base_config.take_profit_pct)))
    if baseline_mask.any():
        baseline = out.loc[baseline_mask].iloc[0]
        out['delta_vs_baseline_return'] = pd.to_numeric(out['cumulative_return'], errors='coerce') - float(pd.to_numeric(baseline['cumulative_return'], errors='coerce'))
        out['delta_vs_baseline_sharpe'] = pd.to_numeric(out['sharpe_ratio'], errors='coerce') - float(pd.to_numeric(baseline['sharpe_ratio'], errors='coerce'))
    else:
        out['delta_vs_baseline_return'] = np.nan
        out['delta_vs_baseline_sharpe'] = np.nan
    return out

def _build_parameter_sensitivity_viz(sensitivity_df: pd.DataFrame) -> go.Figure:
    if sensitivity_df is None or sensitivity_df.empty:
        return _make_empty_figure('Parameter Sensitivity', 'No sensitivity data')
    fig = make_subplots(rows=1, cols=2, subplot_titles=('SL/TP heatmap (mean return)', 'EMA map (composite score)'), specs=[[{'type': 'heatmap'}, {'type': 'xy'}]])
    piv = sensitivity_df.pivot_table(index='stop_loss_pct', columns='take_profit_pct', values='cumulative_return', aggfunc='mean')
    fig.add_trace(go.Heatmap(z=piv.values * 100 if piv.size else [[0]], x=list(piv.columns) if piv.size else [0], y=list(piv.index) if piv.size else [0], colorscale='RdYlGn', zmid=0, colorbar=dict(title='Return %')), row=1, col=1)
    fig.add_trace(go.Scatter(x=sensitivity_df['ema_fast'], y=sensitivity_df['ema_slow'], mode='markers', marker=dict(size=10, color=sensitivity_df['composite_score'], colorscale='Viridis', showscale=True, colorbar=dict(title='Score')), text=sensitivity_df.apply(lambda r: f"rank={int(r['rank'])}, ret={r['cumulative_return']:.2%}, sharpe={r['sharpe_ratio']:.2f}", axis=1), hovertemplate='EMA fast=%{x}<br>EMA slow=%{y}<br>%{text}<extra></extra>'), row=1, col=2)
    fig.update_xaxes(title_text='Take profit %', row=1, col=1)
    fig.update_yaxes(title_text='Stop loss %', row=1, col=1)
    fig.update_xaxes(title_text='EMA fast', row=1, col=2)
    fig.update_yaxes(title_text='EMA slow', row=1, col=2)
    fig.update_layout(title='Sensitivity Analysis')
    return fig

def derive_avoidance_rules(trade_df: pd.DataFrame, min_trades: int=8) -> pd.DataFrame:
    if trade_df is None or trade_df.empty:
        return pd.DataFrame(columns=['rule_type', 'rule_value', 'trades', 'win_rate', 'expectancy', 'baseline_win_rate', 'baseline_expectancy', 'delta_expectancy'])
    baseline_wr = float(trade_df['is_winner'].mean())
    baseline_exp = float(trade_df['net_pnl'].mean())
    rules = []
    candidate_cols = [c for c in ['entry_hour', 'entry_day_name', 'entry_regime', 'event_count_near_trade', 'anomaly_count_near_trade', 'has_anomaly_near_trade', 'anomaly_top_feature'] if c in trade_df.columns]
    for col in candidate_cols:
        grp = trade_df.groupby(col).agg(trades=('trade_id', 'count'), win_rate=('is_winner', 'mean'), expectancy=('net_pnl', 'mean')).reset_index()
        grp = grp[grp['trades'] >= int(min_trades)].copy()
        if grp.empty:
            continue
        bad = grp[(grp['expectancy'] < baseline_exp) & (grp['win_rate'] < baseline_wr)]
        for _, row in bad.iterrows():
            rules.append({'rule_type': col, 'rule_value': str(row[col]), 'trades': int(row['trades']), 'win_rate': float(row['win_rate']), 'expectancy': float(row['expectancy']), 'baseline_win_rate': baseline_wr, 'baseline_expectancy': baseline_exp, 'delta_expectancy': float(row['expectancy'] - baseline_exp)})
    out = pd.DataFrame(rules)
    if not out.empty:
        out = out.sort_values(['delta_expectancy', 'trades'], ascending=[True, False]).reset_index(drop=True)
    return out

def run_strategy_variants(base_config: BacktestConfig, variants: list[VariantConfig], market_df: pd.DataFrame | None=None) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    if market_df is None:
        market_df = download_market_data(base_config)
    summaries = []
    run_map: dict[str, dict[str, Any]] = {}
    for variant in variants:
        cfg = replace(base_config, **variant.overrides)
        cfg = resolve_config(cfg)
        splits = split_time_series(market_df, ratios=cfg.split_ratios)
        labels = generate_strategy_labels(splits['train'], cfg)
        bar_ledger, trades, equity_curve, positions = run_backtest_from_labels(splits['train'], labels, cfg)
        dd = compute_drawdown_series(equity_curve)
        core = compute_core_metrics(equity_curve, trades, cfg, bar_ledger=bar_ledger)
        run_map[variant.name] = {'config': cfg, 'splits': splits, 'labels': labels, 'bar_ledger': bar_ledger, 'trades': trades, 'equity_curve': equity_curve, 'positions': positions, 'drawdown_series': dd, 'core_metrics': core}
        summaries.append({'variant_name': variant.name, 'overrides': stable_json_dumps(variant.overrides), 'total_pnl': core.get('total_pnl', np.nan), 'cumulative_return': core.get('cumulative_return', np.nan), 'sharpe_ratio': core.get('sharpe_ratio', np.nan), 'max_drawdown': core.get('max_drawdown', np.nan), 'win_rate': core.get('win_rate', np.nan), 'profit_factor': core.get('profit_factor', np.nan), 'total_trades': core.get('total_trades', np.nan), 'gain_loss_ratio': core.get('gain_loss_ratio', np.nan), 'avg_trade_duration_hours': core.get('avg_trade_duration_hours', np.nan), 'exposure': core.get('exposure', np.nan)})
    variant_df = pd.DataFrame(summaries)
    return (variant_df, run_map)

def robustness_report(variant_df: pd.DataFrame, avoidance_rules: pd.DataFrame) -> pd.DataFrame:
    if variant_df is None or variant_df.empty:
        return pd.DataFrame()
    out = variant_df.copy()
    baseline = out.iloc[0]
    for col in ['total_pnl', 'cumulative_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']:
        if col in out.columns:
            out[f'delta_vs_baseline__{col}'] = pd.to_numeric(out[col], errors='coerce') - float(pd.to_numeric(baseline[col], errors='coerce'))
    avoid_count = int(len(avoidance_rules)) if avoidance_rules is not None else 0
    out['avoidance_rules_count'] = avoid_count
    return out

def _build_robustness_viz(variant_df: pd.DataFrame) -> go.Figure:
    if variant_df is None or variant_df.empty:
        return _make_empty_figure('Robustesse', 'No variants')
    fig = px.bar(variant_df, x='variant_name', y='cumulative_return', color='sharpe_ratio' if 'sharpe_ratio' in variant_df.columns else None, title='Comparaison variantes (rendement cumule)')
    return fig

def compute_benchmark_metrics(split_df: pd.DataFrame, equity_curve: pd.DataFrame, config: BacktestConfig, split_name: str='train') -> dict[str, Any]:
    if split_df is None or split_df.empty or equity_curve is None or equity_curve.empty:
        return {'split': split_name, 'enabled': False}
    close = split_df['close'].astype(float)
    bh_equity = config.initial_capital * (close / close.iloc[0])
    bh_curve = pd.DataFrame({'equity': bh_equity}, index=split_df.index)
    bh_dd = compute_drawdown_series(bh_curve)
    bh_ret = bh_equity.pct_change().fillna(0.0)
    bh_std = float(bh_ret.std(ddof=0))
    bh_sharpe = float(np.sqrt(config.annualization_factor) * bh_ret.mean() / bh_std) if bh_std > 0 else np.nan
    strat_eq = equity_curve['equity'].astype(float)
    out = {'split': split_name, 'enabled': bool(config.benchmark_enabled), 'strategy_final_capital': float(strat_eq.iloc[-1]), 'strategy_cumulative_return': float(strat_eq.iloc[-1] / strat_eq.iloc[0] - 1.0), 'benchmark_final_capital': float(bh_equity.iloc[-1]), 'benchmark_cumulative_return': float(bh_equity.iloc[-1] / bh_equity.iloc[0] - 1.0), 'benchmark_sharpe_ratio': bh_sharpe, 'benchmark_max_drawdown': float(bh_dd['drawdown'].min())}
    out['alpha_vs_benchmark'] = float(out['strategy_cumulative_return'] - out['benchmark_cumulative_return'])
    return out

def _compute_split_one(split_df: pd.DataFrame, config: BacktestConfig, split_name: str) -> dict[str, Any]:
    labels = generate_strategy_labels(split_df, config)
    bar_ledger, trades, equity_curve, _ = run_backtest_from_labels(split_df, labels, config)
    core = compute_core_metrics(equity_curve, trades, config, bar_ledger=bar_ledger)
    bench = compute_benchmark_metrics(split_df, equity_curve, config, split_name=split_name)
    row = {'split': split_name}
    row.update(core)
    row.update({f'benchmark__{k}': v for k, v in bench.items() if k not in {'split', 'enabled'}})
    return row

def compute_split_metrics(splits: dict[str, pd.DataFrame], config: BacktestConfig) -> pd.DataFrame:
    rows = []
    for split_name in ['train', 'val', 'test']:
        if split_name not in splits:
            continue
        frame = splits[split_name]
        if frame is None or frame.empty:
            continue
        rows.append(_compute_split_one(frame, config, split_name))
    return pd.DataFrame(rows)

def evaluate_hmm_model_order(train_features: pd.DataFrame, states_grid: tuple[int, ...], regime_config: RegimeModelConfig | None=None) -> tuple[pd.DataFrame, int]:
    rcfg = regime_config or RegimeModelConfig()
    rows = []
    best_state = int(states_grid[0]) if len(states_grid) else int(rcfg.n_states)
    for n_states in states_grid:
        cfg = replace(rcfg, n_states=int(n_states))
        try:
            model_obj = fit_hmm_regimes(train_features, n_states=int(n_states), regime_config=cfg)
            model = model_obj.get('model')
            fcols = model_obj.get('feature_columns', ['ret_1', 'vol_rolling', 'trend_proxy', 'volume_zscore'])
            feat = train_features[fcols].replace([np.inf, -np.inf], np.nan).dropna()
            if model is None or feat.empty:
                ll = np.nan
                aic = np.nan
                bic = np.nan
                fallback = True
            else:
                X = feat.to_numpy(dtype=float)
                scaler = model_obj.get('scaler')
                if scaler is not None:
                    X = scaler.transform(X)
                ll = float(model.score(X))
                n = int(len(X))
                p = int(n_states * n_states + n_states * len(fcols) * 2)
                aic = float(2 * p - 2 * ll)
                bic = float(np.log(max(n, 1)) * p - 2 * ll)
                fallback = False
        except Exception:
            ll = np.nan
            aic = np.nan
            bic = np.nan
            fallback = True
        rows.append({'n_states': int(n_states), 'log_likelihood': ll, 'aic': aic, 'bic': bic, 'fallback_mode': fallback})
    result = pd.DataFrame(rows).sort_values('n_states').reset_index(drop=True)
    valid = result[result['bic'].notna()]
    if not valid.empty:
        best_state = int(valid.sort_values('bic').iloc[0]['n_states'])
    return (result, best_state)

def compute_regime_transition_matrix(regimes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if regimes is None or regimes.empty or 'regime_state' not in regimes.columns:
        return {'counts': pd.DataFrame(), 'probabilities': pd.DataFrame()}
    s = pd.to_numeric(regimes['regime_state'], errors='coerce').dropna().astype(int)
    if len(s) < 2:
        return {'counts': pd.DataFrame(), 'probabilities': pd.DataFrame()}
    prev = s.shift(1).dropna().astype(int)
    curr = s.loc[prev.index].astype(int)
    counts = pd.crosstab(prev, curr)
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    counts.index.name = 'from_state'
    counts.columns.name = 'to_state'
    probs.index.name = 'from_state'
    probs.columns.name = 'to_state'
    return {'counts': counts, 'probabilities': probs}

def calibrate_signal_scores(scoring_df: pd.DataFrame, n_bins: int=10) -> pd.DataFrame:
    if scoring_df is None or scoring_df.empty:
        return pd.DataFrame()
    df = scoring_df.copy()
    if 'target' not in df.columns or 'score' not in df.columns:
        return pd.DataFrame()
    if 'split' in df.columns:
        df = df[df['split'].astype(str) == 'test'].copy()
    df = df[pd.to_numeric(df['score'], errors='coerce').notna()].copy()
    if df.empty:
        return pd.DataFrame()
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['target'] = pd.to_numeric(df['target'], errors='coerce').fillna(0).astype(int)
    n_bins_eff = max(2, min(n_bins, int(df['score'].nunique())))
    if n_bins_eff < 2:
        return pd.DataFrame()
    try:
        df['_bin'] = pd.qcut(df['score'], q=n_bins_eff, duplicates='drop')
    except Exception:
        df['_bin'] = pd.cut(df['score'], bins=n_bins_eff)
    bins_obj = df['_bin'].astype(object)
    df['bin_left'] = bins_obj.map(lambda x: float(x.left) if isinstance(x, pd.Interval) else np.nan)
    df['bin_right'] = bins_obj.map(lambda x: float(x.right) if isinstance(x, pd.Interval) else np.nan)
    df['bin_label'] = bins_obj.map(lambda x: str(x) if isinstance(x, pd.Interval) else str(x))
    calib = df.groupby(['bin_left', 'bin_right', 'bin_label'], observed=False, dropna=False).agg(n=('target', 'count'), avg_score=('score', 'mean'), observed_win_rate=('target', 'mean')).reset_index()
    calib = calib.sort_values(['bin_left', 'bin_right', 'bin_label'], kind='stable').reset_index(drop=True)
    calib['bin'] = calib['bin_label']
    calib['calibration_gap'] = calib['observed_win_rate'] - calib['avg_score']
    return calib

def compute_feature_stability(trade_df: pd.DataFrame, scoring_config: ScoringConfig, seeds: tuple[int, ...]) -> pd.DataFrame:
    if trade_df is None or trade_df.empty:
        return pd.DataFrame()
    rows = []
    for seed in seeds:
        cfg = replace(scoring_config, random_state=int(seed))
        out = train_interpretable_scoring_models(trade_df, scoring_config=cfg)
        report = out.get('model_report', {})
        for rec in report.get('feature_importance', []):
            rows.append({'seed': int(seed), 'model': rec.get('model', 'unknown'), 'feature': rec.get('feature', 'unknown'), 'importance': float(rec.get('importance', np.nan))})
    if not rows:
        return pd.DataFrame()
    imp = pd.DataFrame(rows)
    agg = imp.groupby(['model', 'feature']).agg(runs=('importance', 'count'), mean_importance=('importance', 'mean'), std_importance=('importance', 'std'), min_importance=('importance', 'min'), max_importance=('importance', 'max')).reset_index()
    agg = agg.sort_values(['mean_importance', 'runs'], ascending=[False, False]).reset_index(drop=True)
    return agg

def run_walkforward_variants(base_config: BacktestConfig, variants: list[VariantConfig], market_df: pd.DataFrame, walkforward_windows: tuple[int, int, int]) -> pd.DataFrame:
    if market_df is None or market_df.empty:
        return pd.DataFrame()
    n = len(market_df)
    train_size, test_size, step = [int(x) for x in walkforward_windows]
    if n <= train_size + test_size:
        train_size = max(40, int(n * 0.6))
        test_size = max(20, n - train_size)
        step = max(1, int(test_size * 0.5))
    rows = []
    widx = 0
    upper = n - train_size - test_size + 1
    if upper <= 0:
        return pd.DataFrame()
    for start in range(0, upper, step):
        train_end = start + train_size
        test_end = train_end + test_size
        if test_end > n:
            break
        test_slice = market_df.iloc[train_end:test_end].copy()
        if test_slice.empty:
            continue
        for variant in variants:
            cfg = resolve_config(replace(base_config, **variant.overrides))
            labels = generate_strategy_labels(test_slice, cfg)
            ledger, trades, eq, _ = run_backtest_from_labels(test_slice, labels, cfg)
            core = compute_core_metrics(eq, trades, cfg, bar_ledger=ledger)
            rows.append({'window_id': int(widx), 'variant_name': variant.name, 'start': test_slice.index.min(), 'end': test_slice.index.max(), 'rows': int(len(test_slice)), 'total_pnl': core.get('total_pnl', np.nan), 'cumulative_return': core.get('cumulative_return', np.nan), 'sharpe_ratio': core.get('sharpe_ratio', np.nan), 'max_drawdown': core.get('max_drawdown', np.nan), 'win_rate': core.get('win_rate', np.nan), 'profit_factor': core.get('profit_factor', np.nan), 'total_trades': core.get('total_trades', np.nan)})
        widx += 1
    return pd.DataFrame(rows)

def run_stress_scenarios(base_config: BacktestConfig, market_df: pd.DataFrame, fees_grid: tuple[float, ...], slippage_grid: tuple[float, ...]) -> pd.DataFrame:
    if market_df is None or market_df.empty:
        return pd.DataFrame()
    base_splits = split_time_series(market_df, ratios=base_config.split_ratios)
    train = base_splits['train']
    rows = []
    for fees in fees_grid:
        for slippage in slippage_grid:
            cfg = resolve_config(replace(base_config, fees_bps=float(fees), slippage_bps=float(slippage)))
            labels = generate_strategy_labels(train, cfg)
            ledger, trades, eq, _ = run_backtest_from_labels(train, labels, cfg)
            core = compute_core_metrics(eq, trades, cfg, bar_ledger=ledger)
            rows.append({'fees_bps': float(fees), 'slippage_bps': float(slippage), 'total_pnl': core.get('total_pnl', np.nan), 'cumulative_return': core.get('cumulative_return', np.nan), 'sharpe_ratio': core.get('sharpe_ratio', np.nan), 'max_drawdown': core.get('max_drawdown', np.nan), 'win_rate': core.get('win_rate', np.nan), 'profit_factor': core.get('profit_factor', np.nan), 'total_trades': core.get('total_trades', np.nan)})
    return pd.DataFrame(rows)

def build_v2_consolidated_report(run_obj: dict[str, Any]) -> go.Figure:
    fig = make_subplots(rows=4, cols=2, subplot_titles=('Equity', 'Drawdown', 'Split Returns', 'Stress (Return by Costs)', 'Walkforward Variants', 'Calibration', 'Failure Contexts', 'Parameter Sensitivity (SL/TP)'))
    eq = run_obj.get('equity_curve', pd.DataFrame())
    if not eq.empty:
        fig.add_trace(go.Scatter(x=eq.index, y=eq['equity'], mode='lines', name='Equity'), row=1, col=1)
    dd = run_obj.get('drawdown_series', pd.DataFrame())
    if not dd.empty:
        fig.add_trace(go.Scatter(x=dd.index, y=dd['drawdown'] * 100, mode='lines', fill='tozeroy', name='DD %'), row=1, col=2)
    split_metrics = run_obj.get('split_metrics', pd.DataFrame())
    if split_metrics is not None and (not split_metrics.empty) and ('cumulative_return' in split_metrics.columns):
        fig.add_trace(go.Bar(x=split_metrics['split'], y=split_metrics['cumulative_return'] * 100, name='Split Return %'), row=2, col=1)
    stress = run_obj.get('stress_scenarios', pd.DataFrame())
    if stress is not None and (not stress.empty):
        piv = stress.pivot_table(index='fees_bps', columns='slippage_bps', values='cumulative_return', aggfunc='mean')
        fig.add_trace(go.Heatmap(z=piv.values * 100, x=list(piv.columns), y=list(piv.index), colorscale='RdYlGn', zmid=0), row=2, col=2)
    wf = run_obj.get('walkforward_summary', pd.DataFrame())
    if wf is not None and (not wf.empty):
        agg = wf.groupby('variant_name')['cumulative_return'].mean().reset_index()
        fig.add_trace(go.Bar(x=agg['variant_name'], y=agg['cumulative_return'] * 100, name='WF Avg Return %'), row=3, col=1)
    calib = run_obj.get('scoring_calibration', pd.DataFrame())
    if calib is not None and (not calib.empty):
        fig.add_trace(go.Scatter(x=calib['avg_score'], y=calib['observed_win_rate'], mode='lines+markers', name='Calibration'), row=3, col=2)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect', line=dict(dash='dash')), row=3, col=2)
    failure_df = run_obj.get('failure_diagnosis', pd.DataFrame())
    if failure_df is not None and (not failure_df.empty):
        top_fail = failure_df.head(10).copy()
        top_fail['label'] = top_fail['dimension'].astype(str) + '::' + top_fail['context'].astype(str)
        fig.add_trace(go.Bar(x=top_fail['severity'], y=top_fail['label'], orientation='h', name='Failure severity'), row=4, col=1)
    sensitivity = run_obj.get('parameter_sensitivity', pd.DataFrame())
    if sensitivity is not None and (not sensitivity.empty):
        piv = sensitivity.pivot_table(index='stop_loss_pct', columns='take_profit_pct', values='composite_score', aggfunc='mean')
        fig.add_trace(go.Heatmap(z=piv.values if piv.size else [[0]], x=list(piv.columns) if piv.size else [0], y=list(piv.index) if piv.size else [0], colorscale='RdYlGn', zmid=0, colorbar=dict(title='Score')), row=4, col=2)
    stat_summary = run_obj.get('stat_robustness_summary', pd.DataFrame())
    if stat_summary is not None and (not stat_summary.empty):
        ss = stat_summary.iloc[0]
        ann = f"DSR={ss.get('deflated_sharpe_ratio', np.nan):.3f}, P(CumRet>0)={ss.get('prob_cum_return_positive', np.nan):.2%}"
        fig.add_annotation(text=ann, showarrow=False, x=0.5, y=0.515, xref='paper', yref='paper')
    meta_report = run_obj.get('meta_labeling_report', {})
    if isinstance(meta_report, dict) and meta_report:
        d_exp = float(pd.to_numeric(meta_report.get('delta_expectancy_test', np.nan), errors='coerce'))
        thr = float(pd.to_numeric(meta_report.get('recommended_threshold', np.nan), errors='coerce'))
        if np.isfinite(d_exp):
            fig.add_annotation(text=f"Meta filter thr={thr:.3f}, delta_exp_test={d_exp:.4f}", showarrow=False, x=0.5, y=1.05, xref='paper', yref='paper')
    fig.update_layout(height=1500, title='V2 Consolidated Report')
    return fig

def build_v3_consolidated_report(run_obj: dict[str, Any]) -> go.Figure:
    fig = make_subplots(
        rows=5,
        cols=2,
        subplot_titles=(
            'Equity',
            'Drawdown',
            'Purged CV (Train vs Test)',
            'Execution Impact',
            'Trade Root-Cause',
            'Portfolio Equity',
            'Feature Drift',
            'Meta Labeling (Threshold)',
            'Auto Recommendations',
            'Parameter Sensitivity',
        ),
    )
    eq = run_obj.get('equity_curve', pd.DataFrame())
    if eq is not None and (not eq.empty):
        fig.add_trace(go.Scatter(x=eq.index, y=eq['equity'], mode='lines', name='Equity'), row=1, col=1)
    dd = run_obj.get('drawdown_series', pd.DataFrame())
    if dd is not None and (not dd.empty):
        fig.add_trace(go.Scatter(x=dd.index, y=dd['drawdown'] * 100, mode='lines', fill='tozeroy', name='Drawdown %'), row=1, col=2)
    purged = run_obj.get('purged_cv', pd.DataFrame())
    if purged is not None and (not purged.empty):
        fig.add_trace(go.Scatter(x=purged['train_objective'], y=purged['test_objective'], mode='markers', marker=dict(size=6), name='Purged CV'), row=2, col=1)
    exec_imp = run_obj.get('execution_impact', pd.DataFrame())
    if exec_imp is not None and (not exec_imp.empty):
        x = exec_imp['execution_mode'].astype(str) + '_lat' + exec_imp['latency_bars'].astype(int).astype(str)
        fig.add_trace(go.Bar(x=x, y=exec_imp['exec_total_pnl'], name='Exec total pnl'), row=2, col=2)
    rc = run_obj.get('trade_root_cause_summary', pd.DataFrame())
    if rc is not None and (not rc.empty):
        fig.add_trace(go.Bar(x=rc['root_cause'], y=rc['expectancy'], name='Root cause expectancy'), row=3, col=1)
    peq = run_obj.get('portfolio_equity', pd.DataFrame())
    if peq is not None and (not peq.empty):
        fig.add_trace(go.Scatter(x=peq.index, y=peq['equity'], mode='lines', name='Portfolio'), row=3, col=2)
    drift = run_obj.get('feature_drift', pd.DataFrame())
    if drift is not None and (not drift.empty):
        top = drift.head(12).copy()
        fig.add_trace(go.Bar(x=top['feature'], y=top['severity'], name='Drift severity'), row=4, col=1)
    meta_thr = run_obj.get('meta_labeling_thresholds', pd.DataFrame())
    if meta_thr is not None and (not meta_thr.empty):
        fig.add_trace(go.Scatter(x=meta_thr['threshold'], y=meta_thr['objective'], mode='lines+markers', name='Meta objective'), row=4, col=2)
    rec = run_obj.get('auto_recommendations', pd.DataFrame())
    if rec is not None and (not rec.empty):
        top_rec = rec.head(8).copy()
        fig.add_trace(go.Bar(x=top_rec['rank'].astype(str), y=top_rec['priority'], name='Recommendation priority'), row=5, col=1)
    sens = run_obj.get('parameter_sensitivity', pd.DataFrame())
    if sens is not None and (not sens.empty):
        piv = sens.pivot_table(index='stop_loss_pct', columns='take_profit_pct', values='composite_score', aggfunc='mean')
        fig.add_trace(go.Heatmap(z=piv.values if piv.size else [[0]], x=list(piv.columns) if piv.size else [0], y=list(piv.index) if piv.size else [0], colorscale='RdYlGn', zmid=0), row=5, col=2)
    anom_rep = run_obj.get('anomaly_report', {})
    if isinstance(anom_rep, dict) and anom_rep:
        if np.isfinite(pd.to_numeric(anom_rep.get('anomaly_rate', np.nan), errors='coerce')):
            fig.add_annotation(text=f"Anomaly rate={float(pd.to_numeric(anom_rep.get('anomaly_rate', np.nan), errors='coerce')):.2%} ({anom_rep.get('model_name', 'na')})", showarrow=False, x=0.02, y=0.01, xref='paper', yref='paper')
    fig.update_layout(height=1900, title='V3 Consolidated Report')
    return fig

def execute_complementary_pipeline(run_obj: dict[str, Any], config: BacktestConfig, events_csv_path: str | Path | None=None, regime_config: RegimeModelConfig | None=None, scoring_config: ScoringConfig | None=None, variants: list[VariantConfig] | None=None, persist: bool=True, render: bool=False) -> tuple[dict[str, Any], Path | None]:
    cfg = resolve_config(config)
    validate_config(cfg)
    compare_catalog = load_run_catalog(cfg, notebook_dir=Path.cwd())
    dashboard_bundle = build_main_dashboard(run_obj, compare_df=compare_catalog.tail(10))
    comparison_bundle = build_comparison_views(compare_catalog, run_ids=compare_catalog.tail(5)['run_id'].astype(str).tolist())
    splits = run_obj['splits']
    train_df = splits['train']
    split_metrics = compute_split_metrics(splits, cfg)
    benchmark_metrics = compute_benchmark_metrics(train_df, run_obj['equity_curve'], cfg, split_name='train')
    events_df = create_future_placeholders()['events']
    if events_csv_path is not None:
        p = Path(events_csv_path)
        if p.exists():
            events_df = load_events_csv(p)
    trade_events = attach_events_to_trades(trades=run_obj['trades'], events=events_df, pre_window_h=6, post_window_h=6, asset=cfg.symbol)
    regime_cfg = regime_config or RegimeModelConfig()
    regime_features = compute_regime_features(train_df, regime_cfg)
    hmm_model_selection, best_states = evaluate_hmm_model_order(regime_features, cfg.hmm_states_grid, regime_config=regime_cfg)
    regime_cfg_best = replace(regime_cfg, n_states=int(best_states))
    regime_model = fit_hmm_regimes(regime_features, n_states=regime_cfg_best.n_states, regime_config=regime_cfg_best)
    regimes_df = predict_regimes(regime_features, regime_model)
    regime_transition = compute_regime_transition_matrix(regimes_df)
    trade_enriched = build_trade_enriched_table(trades=run_obj['trades'], bar_ledger=run_obj['bar_ledger'], labels=run_obj['labels'], market_df=train_df, regimes=regimes_df, trade_events=trade_events)
    anomaly_scores = pd.DataFrame()
    trade_anomalies = pd.DataFrame()
    anomaly_impact_df = pd.DataFrame()
    anomaly_impact_detail = pd.DataFrame()
    anomaly_report: dict[str, Any] = {'status': 'disabled'}
    anomaly_viz = _make_empty_figure('Anomaly Analysis', 'Anomaly detection disabled')
    if bool(cfg.anomaly_enabled):
        fit_end_time = splits['train'].index.max()
        anomaly_scores, anomaly_report = detect_market_anomalies(run_obj['raw_market'], cfg, fit_end_time=fit_end_time)
        trade_anomalies = attach_anomalies_to_trades(run_obj['trades'], anomaly_scores, pre_window_h=int(cfg.anomaly_pre_window_h), post_window_h=int(cfg.anomaly_post_window_h))
        anomaly_report_dict = anomaly_impact_report(run_obj['trades'], trade_anomalies)
        anomaly_impact_df = anomaly_report_dict.get('by_top_feature', pd.DataFrame())
        anomaly_impact_detail = anomaly_report_dict.get('with_vs_without', pd.DataFrame())
        anomaly_viz = _build_anomaly_viz(anomaly_scores, anomaly_report_dict)
        if not trade_enriched.empty:
            aggs = trade_anomalies.groupby('trade_id').agg(anomaly_count_near_trade=('anomaly_timestamp', 'count'), anomaly_max_score=('anomaly_score', 'max')).reset_index() if (trade_anomalies is not None and not trade_anomalies.empty) else pd.DataFrame(columns=['trade_id', 'anomaly_count_near_trade', 'anomaly_max_score'])
            if not aggs.empty:
                topf = trade_anomalies.sort_values(['trade_id', 'anomaly_score'], ascending=[True, False]).drop_duplicates(subset=['trade_id'], keep='first')[['trade_id', 'top_feature']].rename(columns={'top_feature': 'anomaly_top_feature'})
                aggs = aggs.merge(topf, on='trade_id', how='left')
                trade_enriched = trade_enriched.merge(aggs, on='trade_id', how='left')
            if 'anomaly_count_near_trade' not in trade_enriched.columns:
                trade_enriched['anomaly_count_near_trade'] = 0
            if 'anomaly_top_feature' not in trade_enriched.columns:
                trade_enriched['anomaly_top_feature'] = ''
            if 'anomaly_max_score' not in trade_enriched.columns:
                trade_enriched['anomaly_max_score'] = np.nan
            trade_enriched['anomaly_count_near_trade'] = pd.to_numeric(trade_enriched['anomaly_count_near_trade'], errors='coerce').fillna(0).astype(int)
            trade_enriched['has_anomaly_near_trade'] = trade_enriched['anomaly_count_near_trade'] > 0
            trade_enriched['anomaly_top_feature'] = trade_enriched['anomaly_top_feature'].astype('string').fillna('')
            trade_enriched['anomaly_max_score'] = pd.to_numeric(trade_enriched['anomaly_max_score'], errors='coerce')
    winner_loser_report = analyze_winners_vs_losers(trade_enriched)
    trade_viz = plot_trade_explorer(trade_enriched)
    event_report = event_impact_report(trade_enriched, trade_events)
    event_impact_df = event_report.get('by_event_type', pd.DataFrame())
    event_viz = _build_event_impact_viz(event_report)
    regime_report = regime_performance_report(trade_enriched, run_obj['equity_curve'], regimes_df)
    regime_perf_df = regime_report.get('trades_by_regime', pd.DataFrame())
    regime_viz = _build_regime_viz(regime_report)
    scfg = scoring_config or ScoringConfig()
    scoring_out = train_interpretable_scoring_models(trade_enriched, scoring_config=scfg)
    features_ml_df = scoring_out.get('features_ml', pd.DataFrame())
    scoring_df = scoring_out.get('scoring', pd.DataFrame())
    model_report = scoring_out.get('model_report', {})
    scoring_calibration = calibrate_signal_scores(scoring_df, n_bins=10)
    feature_drift = compute_feature_drift_report(features_ml_df, scoring_df, max_features=cfg.feature_drift_max_features)
    feature_drift_viz = _build_feature_drift_viz(feature_drift)
    meta_out = train_meta_labeling_model(trade_enriched, scoring_df, cfg, scoring_config=scfg)
    meta_labeling_scores = meta_out.get('meta_scoring', pd.DataFrame())
    meta_labeling_thresholds = meta_out.get('threshold_scan', pd.DataFrame())
    meta_labeling_report = meta_out.get('meta_report', {})
    meta_labeling_viz = _build_meta_labeling_viz(meta_labeling_scores, meta_labeling_thresholds)
    feature_stability = compute_feature_stability(trade_enriched, scfg, cfg.scoring_seeds)
    scoring_viz = _build_scoring_viz(scoring_df)
    avoidance_rules = derive_avoidance_rules(trade_enriched)
    failure_diagnosis = build_failure_diagnosis_report(trade_enriched, core_metrics=run_obj.get('core_metrics', {}), avoidance_rules=avoidance_rules, regime_performance=regime_perf_df, event_impact=event_impact_df)
    failure_viz = _build_failure_diagnosis_viz(failure_diagnosis)
    if variants is None:
        variants = [VariantConfig(name='baseline', overrides={}), VariantConfig(name='lower_fees', overrides={'fees_bps': max(0.0, cfg.fees_bps * 0.5)}), VariantConfig(name='tighter_sl', overrides={'stop_loss_pct': max(0.001, cfg.stop_loss_pct * 0.75)})]
    variant_df, _variant_runs = run_strategy_variants(cfg, variants=variants, market_df=run_obj['raw_market'])
    walkforward_summary = run_walkforward_variants(cfg, variants=variants, market_df=run_obj['raw_market'], walkforward_windows=cfg.walkforward_windows)
    stress_scenarios = run_stress_scenarios(cfg, market_df=run_obj['raw_market'], fees_grid=cfg.stress_fees_grid, slippage_grid=cfg.stress_slippage_grid)
    n_trials = max(1, int(len(variants) * max(1, len(cfg.stress_fees_grid)) * max(1, len(cfg.stress_slippage_grid))))
    stat_robustness_summary, stat_bootstrap_samples = run_bootstrap_robustness(run_obj['equity_curve'], cfg, observed_metrics=run_obj.get('core_metrics', {}), n_trials=n_trials, n_bootstrap=cfg.bootstrap_iterations, random_state=cfg.bootstrap_seed)
    stat_robustness_viz = _build_stat_robustness_viz(stat_robustness_summary, stat_bootstrap_samples)
    parameter_sensitivity = run_parameter_sensitivity(cfg, market_df=run_obj['raw_market'], max_combinations=cfg.sensitivity_max_combinations)
    parameter_sensitivity_viz = _build_parameter_sensitivity_viz(parameter_sensitivity)
    purged_cv, purged_cv_summary, overfit_report = run_purged_walkforward_cv(cfg, run_obj['raw_market'])
    overfit_viz = _build_overfit_guardrails_viz(purged_cv, purged_cv_summary)
    execution_impact, execution_trade_impact, execution_report = run_execution_impact_analysis(run_obj['trades'], train_df, cfg)
    execution_viz = _build_execution_waterfall_viz(execution_impact)
    trade_root_cause, trade_replay_paths, trade_root_cause_summary = build_trade_replay_root_cause(run_obj['trades'], train_df, max_trades=cfg.trade_replay_max_trades)
    trade_replay_viz = _build_trade_replay_viz(trade_root_cause, trade_replay_paths)
    portfolio_equity = pd.DataFrame()
    portfolio_attribution = pd.DataFrame()
    portfolio_asset_metrics = pd.DataFrame()
    portfolio_correlation = pd.DataFrame()
    portfolio_report: dict[str, Any] = {'status': 'disabled'}
    portfolio_dashboard_viz = _make_empty_figure('Portfolio Dashboard', 'Portfolio analysis disabled (set portfolio_symbols)')
    if len(cfg.portfolio_symbols) > 0:
        market_map = {}
        if 'raw_market' in run_obj and run_obj['raw_market'] is not None and (not run_obj['raw_market'].empty):
            market_map[str(cfg.symbol)] = run_obj['raw_market'].copy()
        portfolio_obj = run_multi_asset_portfolio_pipeline(cfg, symbols=cfg.portfolio_symbols, market_data_map=market_map)
        portfolio_equity = portfolio_obj.get('portfolio_equity', pd.DataFrame())
        portfolio_attribution = portfolio_obj.get('portfolio_attribution', pd.DataFrame())
        portfolio_asset_metrics = portfolio_obj.get('portfolio_asset_metrics', pd.DataFrame())
        portfolio_correlation = portfolio_obj.get('portfolio_correlation', pd.DataFrame())
        portfolio_report = portfolio_obj.get('portfolio_report', {'status': 'unknown'})
        portfolio_dashboard_viz = _build_portfolio_dashboard(portfolio_equity, portfolio_asset_metrics, portfolio_correlation)
    robustness_df = robustness_report(variant_df, avoidance_rules)
    auto_recommendations = build_auto_recommendations(core_metrics=run_obj.get('core_metrics', {}), failure_diagnosis=failure_diagnosis, feature_drift=feature_drift, meta_report=meta_labeling_report, parameter_sensitivity=parameter_sensitivity, stat_robustness_summary=stat_robustness_summary, event_impact=event_impact_df, regime_performance=regime_perf_df, anomaly_impact=anomaly_impact_df, max_items=cfg.recommendations_max_items)
    auto_recommendations_viz = _build_auto_recommendations_viz(auto_recommendations)
    robustness_viz = _build_robustness_viz(robustness_df)
    out_obj = dict(run_obj)
    out_obj.update({'benchmark_metrics': benchmark_metrics, 'split_metrics': split_metrics, 'events': events_df, 'trade_events': trade_events, 'regimes': regimes_df, 'regime_transition': regime_transition, 'trade_enriched': trade_enriched, 'winner_loser_report': winner_loser_report, 'event_impact': event_impact_df, 'event_impact_detail': event_report.get('with_vs_without', pd.DataFrame()), 'regime_performance': regime_perf_df, 'regime_performance_detail': regime_report.get('bars_by_regime', pd.DataFrame()), 'hmm_model_selection': hmm_model_selection, 'features_ml': features_ml_df, 'scoring': scoring_df, 'scoring_calibration': scoring_calibration, 'feature_drift': feature_drift, 'meta_labeling_scores': meta_labeling_scores, 'meta_labeling_thresholds': meta_labeling_thresholds, 'meta_labeling_report': meta_labeling_report, 'feature_stability': feature_stability, 'model_report': model_report, 'avoidance_rules': avoidance_rules, 'failure_diagnosis': failure_diagnosis, 'variant_summary': variant_df, 'walkforward_summary': walkforward_summary, 'stress_scenarios': stress_scenarios, 'stat_robustness_summary': stat_robustness_summary, 'stat_bootstrap_samples': stat_bootstrap_samples, 'parameter_sensitivity': parameter_sensitivity, 'auto_recommendations': auto_recommendations, 'purged_cv': purged_cv, 'purged_cv_summary': purged_cv_summary, 'overfit_report': overfit_report, 'execution_impact': execution_impact, 'execution_trade_impact': execution_trade_impact, 'execution_report': execution_report, 'trade_root_cause': trade_root_cause, 'trade_replay_paths': trade_replay_paths, 'trade_root_cause_summary': trade_root_cause_summary, 'portfolio_equity': portfolio_equity, 'portfolio_attribution': portfolio_attribution, 'portfolio_asset_metrics': portfolio_asset_metrics, 'portfolio_correlation': portfolio_correlation, 'portfolio_report': portfolio_report, 'anomaly_scores': anomaly_scores, 'trade_anomalies': trade_anomalies, 'anomaly_impact': anomaly_impact_df, 'anomaly_impact_detail': anomaly_impact_detail, 'anomaly_report': anomaly_report, 'robustness_report': robustness_df})
    report_v2 = build_v2_consolidated_report(out_obj)
    report_v3 = build_v3_consolidated_report(out_obj)
    viz_bundle_extended = {}
    viz_bundle_extended.update(dashboard_bundle)
    viz_bundle_extended.update(comparison_bundle)
    viz_bundle_extended.update(trade_viz)
    viz_bundle_extended['event_impact'] = event_viz
    viz_bundle_extended['regime_analysis'] = regime_viz
    viz_bundle_extended['signal_scoring'] = scoring_viz
    viz_bundle_extended['feature_drift'] = feature_drift_viz
    viz_bundle_extended['meta_labeling'] = meta_labeling_viz
    viz_bundle_extended['failure_diagnosis'] = failure_viz
    viz_bundle_extended['stat_robustness'] = stat_robustness_viz
    viz_bundle_extended['parameter_sensitivity'] = parameter_sensitivity_viz
    viz_bundle_extended['auto_recommendations'] = auto_recommendations_viz
    viz_bundle_extended['overfit_guardrails'] = overfit_viz
    viz_bundle_extended['execution_waterfall'] = execution_viz
    viz_bundle_extended['trade_replay'] = trade_replay_viz
    viz_bundle_extended['portfolio_dashboard'] = portfolio_dashboard_viz
    viz_bundle_extended['anomaly_analysis'] = anomaly_viz
    viz_bundle_extended['robustness_comparison'] = robustness_viz
    viz_bundle_extended['report_v2'] = report_v2
    viz_bundle_extended['report_v3'] = report_v3
    out_obj['viz_bundle_extended'] = viz_bundle_extended
    run_path = save_run_artifacts(out_obj, cfg) if persist else None
    if render:
        display(pd.DataFrame([run_obj['core_metrics']]))
        display(pd.DataFrame([benchmark_metrics]))
        display(split_metrics)
        display(trade_enriched.head(10))
        display(avoidance_rules.head(10))
        display(feature_drift.head(10))
        display(pd.DataFrame([meta_labeling_report]))
        display(meta_labeling_thresholds.head(10))
        display(pd.DataFrame([overfit_report]))
        display(execution_impact.head(10))
        display(trade_root_cause.head(10))
        display(pd.DataFrame([portfolio_report]))
        display(failure_diagnosis.head(10))
        display(stat_robustness_summary)
        display(parameter_sensitivity.head(10))
        display(pd.DataFrame([anomaly_report]))
        display(anomaly_impact_df.head(10))
        display(auto_recommendations.head(10))
        render_viz_bundle(viz_bundle_extended)
    return (out_obj, run_path)

def run_extended_acceptance_checks(run_obj: dict[str, Any], config: BacktestConfig, run_path: Path | None=None) -> pd.DataFrame:
    checks = []
    try:
        base_report = run_acceptance_checks(run_obj, config, run_path=None)
        assert bool(base_report['passed'].all())
        checks.append({'check': '1.base_regression', 'passed': True, 'details': 'Legacy acceptance checks pass'})
    except Exception as exc:
        checks.append({'check': '1.base_regression', 'passed': False, 'details': str(exc)})
    try:
        root_a = resolve_artifact_root(config, notebook_dir=Path('/Users/loic/Documents/Code/backtester-dash'))
        root_b = resolve_artifact_root(config, notebook_dir=Path('/Users/loic/Documents/Code/backtester-dash/research'))
        assert root_a == root_b
        checks.append({'check': '2.artifact_root_stable', 'passed': True, 'details': str(root_a)})
    except Exception as exc:
        checks.append({'check': '2.artifact_root_stable', 'passed': False, 'details': str(exc)})
    try:
        viz = run_obj.get('viz_bundle_extended', {})
        assert 'dashboard_main' in viz and 'comparison_runs' in viz and ('report_v2' in viz)
        checks.append({'check': '3.dashboard_comparison', 'passed': True, 'details': 'dashboard/comparison/report figs present'})
    except Exception as exc:
        checks.append({'check': '3.dashboard_comparison', 'passed': False, 'details': str(exc)})
    try:
        trade_enriched = run_obj.get('trade_enriched', pd.DataFrame())
        assert not trade_enriched.empty
        required_cols = {'trade_id', 'entry_time', 'exit_time', 'entry_hour', 'entry_day_name', 'entry_ema_spread'}
        assert required_cols.issubset(trade_enriched.columns)
        checks.append({'check': '4.trade_enriched', 'passed': True, 'details': 'trade enrichment valid'})
    except Exception as exc:
        checks.append({'check': '4.trade_enriched', 'passed': False, 'details': str(exc)})
    try:
        split_metrics = run_obj.get('split_metrics', pd.DataFrame())
        assert not split_metrics.empty
        assert {'split', 'cumulative_return'}.issubset(split_metrics.columns)
        checks.append({'check': '5.split_metrics', 'passed': True, 'details': 'train/val/test metrics present'})
    except Exception as exc:
        checks.append({'check': '5.split_metrics', 'passed': False, 'details': str(exc)})
    try:
        hmm_sel = run_obj.get('hmm_model_selection', pd.DataFrame())
        regimes = run_obj.get('regimes', pd.DataFrame())
        assert not hmm_sel.empty and (not regimes.empty)
        assert 'n_states' in hmm_sel.columns and 'regime_label' in regimes.columns
        checks.append({'check': '6.hmm_selection_regimes', 'passed': True, 'details': 'hmm selection + regimes present'})
    except Exception as exc:
        checks.append({'check': '6.hmm_selection_regimes', 'passed': False, 'details': str(exc)})
    try:
        calib = run_obj.get('scoring_calibration', pd.DataFrame())
        feat_stab = run_obj.get('feature_stability', pd.DataFrame())
        status = 'ok'
        if calib.empty:
            status = 'calibration_empty'
        checks.append({'check': '7.scoring_quality', 'passed': True, 'details': status + f', feature_stability_rows={len(feat_stab)}'})
    except Exception as exc:
        checks.append({'check': '7.scoring_quality', 'passed': False, 'details': str(exc)})
    try:
        wf = run_obj.get('walkforward_summary', pd.DataFrame())
        stress = run_obj.get('stress_scenarios', pd.DataFrame())
        assert not wf.empty and (not stress.empty)
        checks.append({'check': '8.robustness_scenarios', 'passed': True, 'details': f'walkforward={len(wf)}, stress={len(stress)}'})
    except Exception as exc:
        checks.append({'check': '8.robustness_scenarios', 'passed': False, 'details': str(exc)})
    try:
        fail_df = run_obj.get('failure_diagnosis', pd.DataFrame())
        required = {'dimension', 'context', 'severity', 'action_hint'}
        assert isinstance(fail_df, pd.DataFrame)
        if not fail_df.empty:
            assert required.issubset(fail_df.columns)
        checks.append({'check': '9.failure_diagnosis', 'passed': True, 'details': f'rows={len(fail_df)}'})
    except Exception as exc:
        checks.append({'check': '9.failure_diagnosis', 'passed': False, 'details': str(exc)})
    try:
        stat_summary = run_obj.get('stat_robustness_summary', pd.DataFrame())
        stat_samples = run_obj.get('stat_bootstrap_samples', pd.DataFrame())
        assert isinstance(stat_summary, pd.DataFrame) and isinstance(stat_samples, pd.DataFrame)
        assert not stat_summary.empty and (not stat_samples.empty)
        checks.append({'check': '10.stat_robustness', 'passed': True, 'details': f'summary_rows={len(stat_summary)}, samples={len(stat_samples)}'})
    except Exception as exc:
        checks.append({'check': '10.stat_robustness', 'passed': False, 'details': str(exc)})
    try:
        sens = run_obj.get('parameter_sensitivity', pd.DataFrame())
        assert isinstance(sens, pd.DataFrame) and (not sens.empty)
        assert {'composite_score', 'rank', 'ema_fast', 'ema_slow', 'stop_loss_pct', 'take_profit_pct'}.issubset(sens.columns)
        checks.append({'check': '11.parameter_sensitivity', 'passed': True, 'details': f'rows={len(sens)}'})
    except Exception as exc:
        checks.append({'check': '11.parameter_sensitivity', 'passed': False, 'details': str(exc)})
    try:
        drift = run_obj.get('feature_drift', pd.DataFrame())
        assert isinstance(drift, pd.DataFrame)
        if not drift.empty:
            assert {'feature', 'severity', 'drift_flag', 'psi', 'mean_shift_z'}.issubset(drift.columns)
        checks.append({'check': '12.feature_drift', 'passed': True, 'details': f'rows={len(drift)}'})
    except Exception as exc:
        checks.append({'check': '12.feature_drift', 'passed': False, 'details': str(exc)})
    try:
        meta_scores = run_obj.get('meta_labeling_scores', pd.DataFrame())
        meta_thr = run_obj.get('meta_labeling_thresholds', pd.DataFrame())
        meta_report = run_obj.get('meta_labeling_report', {})
        assert isinstance(meta_scores, pd.DataFrame) and isinstance(meta_thr, pd.DataFrame) and isinstance(meta_report, dict)
        if not meta_scores.empty:
            assert {'trade_id', 'meta_score', 'meta_take', 'split'}.issubset(meta_scores.columns)
        checks.append({'check': '13.meta_labeling', 'passed': True, 'details': f"status={meta_report.get('status', 'na')}, scores_rows={len(meta_scores)}"})
    except Exception as exc:
        checks.append({'check': '13.meta_labeling', 'passed': False, 'details': str(exc)})
    try:
        rec = run_obj.get('auto_recommendations', pd.DataFrame())
        assert isinstance(rec, pd.DataFrame)
        if not rec.empty:
            assert {'rank', 'category', 'recommendation', 'priority'}.issubset(rec.columns)
        checks.append({'check': '14.auto_recommendations', 'passed': True, 'details': f'rows={len(rec)}'})
    except Exception as exc:
        checks.append({'check': '14.auto_recommendations', 'passed': False, 'details': str(exc)})
    try:
        purged = run_obj.get('purged_cv', pd.DataFrame())
        purged_sum = run_obj.get('purged_cv_summary', pd.DataFrame())
        overfit = run_obj.get('overfit_report', {})
        assert isinstance(purged, pd.DataFrame) and isinstance(purged_sum, pd.DataFrame) and isinstance(overfit, dict)
        if not purged.empty:
            assert {'fold_id', 'train_objective', 'test_objective'}.issubset(purged.columns)
        checks.append({'check': '15.overfit_guardrails', 'passed': True, 'details': f"status={overfit.get('status', 'na')}, folds={overfit.get('folds', 'na')}"})
    except Exception as exc:
        checks.append({'check': '15.overfit_guardrails', 'passed': False, 'details': str(exc)})
    try:
        ex_sum = run_obj.get('execution_impact', pd.DataFrame())
        ex_det = run_obj.get('execution_trade_impact', pd.DataFrame())
        ex_rep = run_obj.get('execution_report', {})
        assert isinstance(ex_sum, pd.DataFrame) and isinstance(ex_det, pd.DataFrame) and isinstance(ex_rep, dict)
        if not ex_sum.empty:
            assert {'execution_mode', 'latency_bars', 'exec_total_pnl'}.issubset(ex_sum.columns)
        checks.append({'check': '16.execution_realism', 'passed': True, 'details': f"scenarios={len(ex_sum)}, status={ex_rep.get('status', 'na')}"})
    except Exception as exc:
        checks.append({'check': '16.execution_realism', 'passed': False, 'details': str(exc)})
    try:
        trc = run_obj.get('trade_root_cause', pd.DataFrame())
        trp = run_obj.get('trade_replay_paths', pd.DataFrame())
        trs = run_obj.get('trade_root_cause_summary', pd.DataFrame())
        assert isinstance(trc, pd.DataFrame) and isinstance(trp, pd.DataFrame) and isinstance(trs, pd.DataFrame)
        if not trc.empty:
            assert {'trade_id', 'root_cause', 'mfe_pct', 'mae_pct', 'net_pnl'}.issubset(trc.columns)
        checks.append({'check': '17.trade_replay_root_cause', 'passed': True, 'details': f'root_rows={len(trc)}, path_rows={len(trp)}'})
    except Exception as exc:
        checks.append({'check': '17.trade_replay_root_cause', 'passed': False, 'details': str(exc)})
    try:
        pm = run_obj.get('portfolio_asset_metrics', pd.DataFrame())
        pc = run_obj.get('portfolio_correlation', pd.DataFrame())
        pr = run_obj.get('portfolio_report', {})
        enabled = len(getattr(config, 'portfolio_symbols', ())) > 0
        if enabled:
            assert isinstance(pm, pd.DataFrame) and (not pm.empty)
        checks.append({'check': '18.portfolio_multi_asset', 'passed': True, 'details': f"enabled={enabled}, status={pr.get('status', 'na') if isinstance(pr, dict) else 'na'}"})
    except Exception as exc:
        checks.append({'check': '18.portfolio_multi_asset', 'passed': False, 'details': str(exc)})
    try:
        anom = run_obj.get('anomaly_scores', pd.DataFrame())
        tr_anom = run_obj.get('trade_anomalies', pd.DataFrame())
        anom_imp = run_obj.get('anomaly_impact', pd.DataFrame())
        anom_rep = run_obj.get('anomaly_report', {})
        assert isinstance(anom, pd.DataFrame) and isinstance(tr_anom, pd.DataFrame) and isinstance(anom_imp, pd.DataFrame) and isinstance(anom_rep, dict)
        if not anom.empty:
            assert {'anomaly_score', 'anomaly_flag', 'top_feature'}.issubset(anom.columns)
        checks.append({'check': '19.anomaly_detection', 'passed': True, 'details': f"scores={len(anom)}, links={len(tr_anom)}, status={anom_rep.get('status', 'na')}"})
    except Exception as exc:
        checks.append({'check': '19.anomaly_detection', 'passed': False, 'details': str(exc)})
    try:
        if run_path is None:
            checks.append({'check': '20.persist_load', 'passed': True, 'details': 'Skipped (persist=False)'})
        else:
            missing = validate_required_artifacts_extended(run_path)
            assert not missing, f'missing={missing}'
            loaded = load_run(run_path)
            assert 'future' in loaded and 'events' in loaded['future'] and ('benchmark_metrics' in loaded) and ('failure_diagnosis' in loaded) and ('stat_robustness_summary' in loaded) and ('parameter_sensitivity' in loaded) and ('feature_drift' in loaded) and ('meta_labeling_report' in loaded) and ('auto_recommendations' in loaded) and ('overfit_report' in loaded) and ('execution_report' in loaded) and ('trade_root_cause' in loaded) and ('portfolio_report' in loaded) and ('anomaly_report' in loaded)
            checks.append({'check': '20.persist_load', 'passed': True, 'details': 'extended artifacts loadable'})
    except Exception as exc:
        checks.append({'check': '20.persist_load', 'passed': False, 'details': str(exc)})
    return pd.DataFrame(checks)
