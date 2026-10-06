#!/usr/bin/env bash
set -euo pipefail

# Requires a verified historical export with point-in-time coverage.
# A backfill collected today is a technical pilot, not an eligible benchmark.
.venv/bin/python scripts/run_news_sentiment_comparison.py \
  --data data/processed/mt5_stocks_us_daily_clean.parquet \
  --ticker-selection configs/benchmark/stocks_us_gnn_complete_2005.json \
  --preset multi_ticker_long_short \
  --models gru \
  --model-parameter-sets configs/benchmark/gru_market_context.json \
  --losses combined \
  --combined-weights 0.25 \
  --loss-cost-bps 5 \
  --selection-metric regularized_sharpe \
  --context-len 60 \
  --position-mode long_short \
  --execution-delay 1 \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --label-method triple-barrier \
  --label-max-holding 10 \
  --label-vol-window 20 \
  --label-volatility-estimator atr \
  --label-profit-barrier 0.75 \
  --label-stop-barrier 0.75 \
  --label-event-filter cusum \
  --label-cusum-threshold 0.5 \
  --label-between-events hold \
  --label-cost-bps 5 \
  --feature-set expanded \
  --feature-groups technical,market,sector \
  --no-external-features \
  --overfitting-control \
  --overfitting-max-features 32 \
  --overfitting-max-feature-correlation 0.95 \
  --news-sentiment-export data/derived/us_finbert_company.parquet \
  --sentiment-candidates gru,gru_activity,gru_features,sentiment,gru_sentiment_mean \
  --date-batch-size 16 \
  --cv-folds 3 \
  --cv-gap-bars 5 \
  --cv-score regularized_sharpe \
  --seeds 1,7,19 \
  --device auto \
  --output-dir artifacts/comparisons/us-news-sentiment \
  --fail-fast \
  "$@"
