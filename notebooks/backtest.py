# %%
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / 'src').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from trading_system.pipelines import single_ticker_long_short as ann
from trading_system.pipelines import single_ticker_long_short_features as annF
from trading_system.backtest import lib as v3

pd.set_option('display.max_columns', 240)
pd.set_option('display.width', 260)
np.random.seed(1)

# %%
# ------------------------------
# Inputs
# ------------------------------
DATASET_PATH = PROJECT_ROOT / 'data' / 'processed' / 'cac40_daily.parquet'

TICKER = 'SAF.PA'
PRICE_COL = 'adj_close'

# Mets ici ton DataFrame custom si besoin (sinon laisse None)
CUSTOM_DF = None
# Mets ici ton DataFrame déjà labélisé (doit contenir Label_id). Sinon laisse None
CUSTOM_LABELS_DF = None

# Si CUSTOM_LABELS_DF est None, on applique la labelisation ANN
LABEL_WINDOW = 30

# ------------------------------
# ANN hyperparams
# ------------------------------
ANN_PARAMS = dict(
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
)

# ------------------------------
# Backtest config
# ------------------------------
RUN_EXTENDED = True
PERSIST_RUN = True

BT_OVERRIDES = dict(
    symbol=TICKER,
    timeframe='1d',
    fees_bps=5.0,
    slippage_bps=0.0,
    stop_loss_pct=0.12,
    take_profit_pct=0.04,
    notes='ANN_long_short -> backtest_lib external pipeline',
)

# %%
def ensure_training_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if 'date' not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={'index': 'date'})
        else:
            raise ValueError("Le DataFrame doit avoir une colonne 'date' ou un DatetimeIndex.")

    out['date'] = pd.to_datetime(out['date'], utc=True, errors='coerce')
    if out['date'].isna().any():
        raise ValueError("Timestamps invalides dans la colonne 'date'.")

    if 'adj_close' not in out.columns and 'close' in out.columns:
        out['adj_close'] = out['close']

    required = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour ANN: {missing}")

    for c in required:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    if out[required].isna().any().any():
        raise ValueError("NaN detectes dans les colonnes OHLCV requises.")

    return out.sort_values('date').reset_index(drop=True)

if CUSTOM_DF is not None:
    df_raw = CUSTOM_DF.copy()
else:
    df_raw = ann.read_parquet_dataset(DATASET_PATH)
    if 'ticker' in df_raw.columns:
        df_raw = df_raw[df_raw['ticker'] == TICKER].copy()

df_raw = ensure_training_schema(df_raw)

if CUSTOM_LABELS_DF is not None:
    train_df = ensure_training_schema(CUSTOM_LABELS_DF.copy())
    if 'Label_id' not in train_df.columns:
        raise ValueError("CUSTOM_LABELS_DF doit contenir la colonne 'Label_id'.")
    train_df['Label_id'] = pd.to_numeric(train_df['Label_id'], errors='coerce').astype(int)
    if 'Label' not in train_df.columns:
        label_map = {0: 'Sell', 1: 'Hold', 2: 'Buy'}
        train_df['Label'] = train_df['Label_id'].map(label_map)
    label_stats = train_df['Label'].value_counts(dropna=False).to_dict()
else:
    train_df, label_stats = ann.labelling(df_raw.copy(), LABEL_WINDOW, price_col=PRICE_COL)

print('Data rows:', len(train_df))
print('Label stats:', label_stats)
display(train_df.head(5))

# %%
best, test_metrics, benchmark = ann.train_one_trial(train_df, **ANN_PARAMS)

print('Test metrics:')
display(pd.DataFrame([test_metrics]))
print('Simple ANN benchmark comparison:')
display(pd.DataFrame([benchmark]))

market_df = best['advanced_backtest_market']
labels_df = best['advanced_backtest_labels']

print('Advanced market rows:', len(market_df))
display(market_df.head(3))
display(labels_df.head(3))

# %%
start_iso = market_df.index.min().isoformat().replace('+00:00', 'Z')
end_iso = market_df.index.max().isoformat().replace('+00:00', 'Z')

cfg = v3.BacktestConfig(start=start_iso, end=end_iso)
cfg = replace(cfg, **BT_OVERRIDES)
v3.validate_config(v3.resolve_config(cfg))

run_obj, run_path = v3.execute_first_check_pipeline_external(
    cfg,
    market_df=market_df,
    labels_df=labels_df,
    persist=PERSIST_RUN,
    render=False,
)

print('Run path:', run_path)
display(pd.DataFrame([run_obj['core_metrics']]))
display(run_obj['trades'].head(20))
v3.render_viz_bundle(run_obj['viz_bundle'])

# %%
if RUN_EXTENDED:
    events_csv = Path('events_demo.csv')
    events_arg = events_csv if events_csv.exists() else None

    ext_obj, ext_path = v3.execute_complementary_pipeline(
        run_obj,
        cfg,
        events_csv_path=events_arg,
        persist=PERSIST_RUN,
        render=True,
    )

    ext_report = v3.run_extended_acceptance_checks(ext_obj, cfg, run_path=ext_path)
    display(ext_report)
    print('Extended path:', ext_path)

catalog = v3.load_run_catalog(cfg)
display(catalog.tail(10))
