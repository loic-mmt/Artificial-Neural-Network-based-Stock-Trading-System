"""Standalone offline comparison of price GRU and audited ticker news."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

from trading_system.data.news_sentiment import load_news_sentiment_export
from trading_system.experiments.news_sentiment_ablation import (
    CANDIDATES, NewsSentimentAblationConfig, run_news_sentiment_ablation,
)
from trading_system.paths import comparisons_dir
from trading_system.pipelines.compare_gnn_graphs import (
    build_parser as graph_parser, prepare_graph_run,
)


def build_parser():
    parser = graph_parser()
    parser.description = ("Matched price GRU, count/coverage placebo and audited FinBERT news "
                          "controls. Offline only; final holdout remains sealed.")
    parser.set_defaults(graph_candidates="gru", graph_lookback=3)
    parser.add_argument("--news-sentiment-export", type=Path, required=True,
                        help="Offline daily Parquet and same-stem .manifest.json; no collection or text scoring.")
    parser.add_argument("--sentiment-candidates", default=",".join(CANDIDATES),
                        help="Comma-separated: " + ", ".join(CANDIDATES))
    parser.add_argument("--sentiment-hidden-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify export, eligible TRAIN coverage, task signatures and optional resume without writing files.")
    return parser


def prepare_news_sentiment_run(argv=None):
    """Reuse graph CLI's frozen price/CV configuration, never its graph model."""
    tokens = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(tokens)
    if args.sentiment is not None:
        raise ValueError("Use --news-sentiment-export; legacy --sentiment is not a matched news route.")
    if args.market_context_data or args.market_cross_section:
        raise ValueError("Macro/market context belongs to its separate market-domain comparison.")
    if args.graph_candidates != "gru" or args.graph_lookback != 3:
        raise ValueError("News comparison has no graph; retain graph-candidates=gru and graph-lookback=3.")
    candidates = tuple(item.strip() for item in args.sentiment_candidates.split(",") if item.strip())
    ablation = NewsSentimentAblationConfig(
        candidates=candidates, sentiment_hidden_size=args.sentiment_hidden_size,
        date_batch_size=args.date_batch_size)
    # Only pass the original graph parser's options. Its preparation function
    # supplies the shared label, train-only selector and calendar protocol.
    own_flags = {"--news-sentiment-export", "--sentiment-candidates", "--sentiment-hidden-size"}
    forwarded, index = [], 0
    while index < len(tokens):
        token = tokens[index]
        name = token.split("=", 1)[0]
        if name in own_flags:
            index += 1 if "=" in token else 2
        elif name == "--dry-run":
            index += 1
        else:
            forwarded.append(token)
            index += 1
    if not any(token.split("=", 1)[0] == "--graph-candidates" for token in forwarded):
        forwarded.extend(("--graph-candidates", "gru"))
    if not any(token.split("=", 1)[0] == "--graph-lookback" for token in forwarded):
        forwarded.extend(("--graph-lookback", "3"))
    shared = prepare_graph_run(forwarded)
    exported = load_news_sentiment_export(args.news_sentiment_export)
    target = args.output_dir or comparisons_dir() / (
        "news-sentiment-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    return {key: shared[key] for key in (
        "frame", "config", "loss", "gru_parameters", "seeds", "n_splits",
        "initial_train_fraction", "inner_val_fraction", "gap_bars", "embargo_bars", "dataset_path",
    )} | {"destination": target, "sentiment_export": exported, "ablation": ablation}


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run_news_sentiment_ablation(**prepare_news_sentiment_run(argv),
                                          resume=args.resume, dry_run=args.dry_run)
    if args.dry_run:
        covered = [task["spec"]["preprocessing"]["sentiment_scaler"]["covered_fit_rows"]
                   for task in result["task_specs"]]
        print(f"news_dry_run tasks={len(result['task_specs'])} completed={result['completed_tasks']} "
              f"minimum_covered_train_rows={min(covered)} final_holdout_opened=False")
    return result


__all__ = ["build_parser", "prepare_news_sentiment_run", "main"]
