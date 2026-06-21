from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_system.paths import default_market_dataset_path, gridsearch_dir
from trading_system.pipelines import walkforward as wf


@dataclass(frozen=True)
class TrialConfig:
    forward_horizon: int
    forward_buy_threshold: float
    forward_sell_threshold: float
    context_len: int
    walkforward_step: int
    hidden: int
    epochs: int
    alpha: float
    batch_size: int
    decision_mode: str
    min_action_rate: float

def parse_int_list(raw: str) -> list[int]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return [int(v) for v in vals]


def parse_float_list(raw: str) -> list[float]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return [float(v) for v in vals]


def parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def make_trial_grid(args) -> list[TrialConfig]:
    horizons = parse_int_list(args.forward_horizons)
    buy_thrs = parse_float_list(args.forward_buy_thresholds)
    sell_thrs = parse_float_list(args.forward_sell_thresholds)
    context_lens = parse_int_list(args.context_lens)
    wf_steps = parse_int_list(args.walkforward_steps)
    hiddens = parse_int_list(args.hiddens)
    epochs = parse_int_list(args.epochs_grid)
    alphas = parse_float_list(args.alphas)
    batch_sizes = parse_int_list(args.batch_sizes)
    decision_modes = parse_str_list(args.decision_modes)
    min_action_rates = parse_float_list(args.min_action_rates)

    grid = [
        TrialConfig(
            forward_horizon=h,
            forward_buy_threshold=bt,
            forward_sell_threshold=st,
            context_len=cl,
            walkforward_step=ws,
            hidden=hd,
            epochs=ep,
            alpha=al,
            batch_size=bs,
            decision_mode=dm,
            min_action_rate=mar,
        )
        for h, bt, st, cl, ws, hd, ep, al, bs, dm, mar in itertools.product(
            horizons,
            buy_thrs,
            sell_thrs,
            context_lens,
            wf_steps,
            hiddens,
            epochs,
            alphas,
            batch_sizes,
            decision_modes,
            min_action_rates,
        )
    ]
    if not grid:
        raise ValueError("La grille est vide.")
    return grid


def pick_trials(grid: list[TrialConfig], max_trials: int | None, seed: int) -> list[TrialConfig]:
    if max_trials is None or max_trials <= 0 or max_trials >= len(grid):
        return grid

    rng = np.random.default_rng(seed)
    idx = np.arange(len(grid))
    pick = rng.choice(idx, size=max_trials, replace=False)
    pick = np.sort(pick)
    return [grid[i] for i in pick.tolist()]


def objective_value(row: dict[str, Any], objective: str) -> float:
    if objective == "outperformance":
        return float(row["outperformance"])
    if objective == "model_pnl":
        return float(row["model_pnl"])
    if objective == "final_model":
        return float(row["final_model"])
    if objective == "macro_f1":
        return float(row["macro_f1"])
    if objective == "bal_acc":
        return float(row["bal_acc"])
    if objective == "acc":
        return float(row["acc"])
    raise ValueError(f"Objective inconnu: {objective}")


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Grid search walk-forward pour pipeline walkforward.py "
            "(objectif principal: outperformance)."
        )
    )
    p.add_argument("--data-dir", type=Path, default=default_market_dataset_path())
    p.add_argument("--ticker", type=str, default="EN.PA")
    p.add_argument("--price-col", type=str, default="adj_close")
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--label-mode", type=str, choices=["forward_return", "oracle_dp"], default="forward_return")
    p.add_argument("--position-mode", type=str, choices=["long_only", "long_short"], default="long_only")
    p.add_argument("--oracle-fee-per-trade", type=float, default=2.0)
    p.add_argument("--strategy-fee-per-trade", type=float, default=0.0)
    p.add_argument("--do-dropout", action="store_true")
    p.add_argument("--dropout-percent", type=float, default=0.1)
    p.add_argument("--early-stopping-patience", type=int, default=30)
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    p.add_argument("--objective", type=str, choices=["outperformance", "model_pnl", "final_model", "macro_f1", "bal_acc", "acc"], default="outperformance")

    # Grid dimensions
    p.add_argument("--forward-horizons", type=str, default="3,5")
    p.add_argument("--forward-buy-thresholds", type=str, default="0.003,0.005")
    p.add_argument("--forward-sell-thresholds", type=str, default="0.003,0.005")
    p.add_argument("--context-lens", type=str, default="60,90")
    p.add_argument("--walkforward-steps", type=str, default="30,60")
    p.add_argument("--hiddens", type=str, default="64,128")
    p.add_argument("--epochs-grid", type=str, default="300")
    p.add_argument("--alphas", type=str, default="0.001")
    p.add_argument("--batch-sizes", type=str, default="64")
    p.add_argument("--decision-modes", type=str, default="argmax")
    p.add_argument("--min-action-rates", type=str, default="0.02")

    p.add_argument("--max-trials", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--suppress-inner-logs", action="store_true", default=True)
    p.add_argument("--show-inner-logs", action="store_true")
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--output-json", type=Path, default=None)
    return p


def main():
    args = build_parser().parse_args()
    if args.show_inner_logs:
        args.suppress_inner_logs = False

    data_dir = Path(args.data_dir).expanduser().resolve()

    df = wf.read_parquet_dataset(data_dir)
    if "date" not in df.columns:
        raise ValueError("Colonne 'date' manquante.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    if "ticker" in df.columns and args.ticker:
        df = df[df["ticker"] == args.ticker].copy()
    if df.empty:
        raise ValueError("Dataset vide apres filtrage ticker.")

    df = df.sort_values("date").reset_index(drop=True)
    needed = {"open", "high", "low", "close", args.price_col, "volume"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {sorted(missing)}")

    df_feat = wf.compute_market_features(df)
    feature_cols = list(wf.features)

    for col in [args.price_col] + feature_cols:
        df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")
    df_feat = df_feat.dropna(subset=[args.price_col] + feature_cols).reset_index(drop=True)
    if df_feat.empty:
        raise ValueError("Dataset vide apres feature engineering.")

    full_grid = make_trial_grid(args)
    trials = pick_trials(full_grid, args.max_trials, args.seed)

    print(
        f"Grid search setup | ticker={args.ticker} | rows={len(df_feat)} "
        f"| objective={args.objective} | label_mode={args.label_mode} "
        f"| trials={len(trials)} (full_grid={len(full_grid)})"
    )

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for i, cfg in enumerate(trials, start=1):
        np.random.seed(args.seed + i)
        t0 = time.perf_counter()

        base_row = {
            "trial_id": i,
            "label_mode": args.label_mode,
            "position_mode": args.position_mode,
            "forward_horizon": cfg.forward_horizon,
            "forward_buy_threshold": cfg.forward_buy_threshold,
            "forward_sell_threshold": cfg.forward_sell_threshold,
            "context_len": cfg.context_len,
            "walkforward_step": cfg.walkforward_step,
            "hidden": cfg.hidden,
            "epochs": cfg.epochs,
            "alpha": cfg.alpha,
            "batch_size": cfg.batch_size,
            "decision_mode": cfg.decision_mode,
            "min_action_rate": cfg.min_action_rate,
        }

        try:
            if args.suppress_inner_logs:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = wf.walk_forward_oracle_ann(
                        full_df=df_feat,
                        feature_cols=feature_cols,
                        price_col=args.price_col,
                        train_ratio=args.train_ratio,
                        val_ratio=args.val_ratio,
                        walkforward_step=cfg.walkforward_step,
                        oracle_fee_per_trade=args.oracle_fee_per_trade,
                        label_mode=args.label_mode,
                        forward_horizon=cfg.forward_horizon,
                        forward_buy_threshold=cfg.forward_buy_threshold,
                        forward_sell_threshold=cfg.forward_sell_threshold,
                        decision_mode=cfg.decision_mode,
                        min_action_rate=cfg.min_action_rate,
                        position_mode=args.position_mode,
                        strategy_fee_per_trade=args.strategy_fee_per_trade,
                        initial_capital=args.capital,
                        context_len=cfg.context_len,
                        epochs=cfg.epochs,
                        alpha=cfg.alpha,
                        hidden=cfg.hidden,
                        batch_size=cfg.batch_size,
                        do_dropout=args.do_dropout,
                        dropout_percent=args.dropout_percent,
                        early_stopping_patience=args.early_stopping_patience,
                        early_stopping_min_delta=args.early_stopping_min_delta,
                    )
            else:
                result = wf.walk_forward_oracle_ann(
                    full_df=df_feat,
                    feature_cols=feature_cols,
                    price_col=args.price_col,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    walkforward_step=cfg.walkforward_step,
                    oracle_fee_per_trade=args.oracle_fee_per_trade,
                    label_mode=args.label_mode,
                    forward_horizon=cfg.forward_horizon,
                    forward_buy_threshold=cfg.forward_buy_threshold,
                    forward_sell_threshold=cfg.forward_sell_threshold,
                    decision_mode=cfg.decision_mode,
                    min_action_rate=cfg.min_action_rate,
                    position_mode=args.position_mode,
                    strategy_fee_per_trade=args.strategy_fee_per_trade,
                    initial_capital=args.capital,
                    context_len=cfg.context_len,
                    epochs=cfg.epochs,
                    alpha=cfg.alpha,
                    hidden=cfg.hidden,
                    batch_size=cfg.batch_size,
                    do_dropout=args.do_dropout,
                    dropout_percent=args.dropout_percent,
                    early_stopping_patience=args.early_stopping_patience,
                    early_stopping_min_delta=args.early_stopping_min_delta,
                )

            m = result["test_metrics"]
            b = result["benchmark_comparison"]
            label_eval = result.get("label_eval_report", {})

            row = {
                **base_row,
                "status": "ok",
                "acc": float(m["acc"]),
                "bal_acc": float(m["bal_acc"]),
                "macro_f1": float(m["macro_f1"]),
                "precision_buy": float(m["precision_buy"]),
                "recall_buy": float(m["recall_buy"]),
                "precision_sell": float(m["precision_sell"]),
                "recall_sell": float(m["recall_sell"]),
                "precision_hold": float(m["precision_hold"]),
                "recall_hold": float(m["recall_hold"]),
                "model_pnl": float(b["model_pnl"]),
                "buy_hold_pnl": float(b["buy_hold_pnl"]),
                "outperformance": float(b["outperformance"]),
                "final_model": float(b["model_final_capital"]),
                "final_buy_hold": float(b["buy_hold_final_capital"]),
                "n_test_rows": int(result["n_test_rows"]),
                "n_eval_rows": int(result["n_eval_rows"]),
                "n_missing_test_preds": int(result["n_missing_test_preds"]),
                "label_n_buy": int(label_eval.get("n_buy", -1)),
                "label_n_hold": int(label_eval.get("n_hold", -1)),
                "label_n_sell": int(label_eval.get("n_sell", -1)),
            }
            row["objective_score"] = objective_value(row, args.objective)
        except Exception as exc:
            row = {
                **base_row,
                "status": "error",
                "error": str(exc),
                "objective_score": -np.inf,
            }

        row["runtime_sec"] = time.perf_counter() - t0
        rows.append(row)

        if row["status"] == "ok":
            print(
                f"[{i:03d}/{len(trials)}] ok "
                f"| score={row['objective_score']:.4f} "
                f"| outperf={row['outperformance']:.2f} "
                f"| macro_f1={row['macro_f1']:.3f} "
                f"| cfg=({cfg.forward_horizon},{cfg.forward_buy_threshold},{cfg.forward_sell_threshold},"
                f"{cfg.context_len},{cfg.walkforward_step},{cfg.hidden},{cfg.epochs},{cfg.decision_mode})"
            )
        else:
            print(
                f"[{i:03d}/{len(trials)}] error "
                f"| cfg=({cfg.forward_horizon},{cfg.forward_buy_threshold},{cfg.forward_sell_threshold},"
                f"{cfg.context_len},{cfg.walkforward_step},{cfg.hidden},{cfg.epochs},{cfg.decision_mode}) "
                f"| {row['error']}"
            )

    elapsed = time.perf_counter() - started
    results_df = pd.DataFrame(rows)
    ok_df = results_df[results_df["status"] == "ok"].copy()
    if not ok_df.empty:
        ok_df = ok_df.sort_values(
            by=["objective_score", "outperformance", "macro_f1"],
            ascending=False,
        ).reset_index(drop=True)
    top_df = ok_df.head(max(1, int(args.top_k))) if not ok_df.empty else ok_df

    now_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_csv = gridsearch_dir() / f"gridsearch_walkforward_{now_tag}.csv"
    default_json = gridsearch_dir() / f"gridsearch_walkforward_{now_tag}.json"
    out_csv = Path(args.output_csv) if args.output_csv is not None else default_csv
    out_json = Path(args.output_json) if args.output_json is not None else default_json
    out_csv = out_csv.expanduser().resolve()
    out_json = out_json.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(out_csv, index=False)
    payload = {
        "meta": {
            "ticker": args.ticker,
            "rows": int(len(df_feat)),
            "objective": args.objective,
            "label_mode": args.label_mode,
            "position_mode": args.position_mode,
            "n_trials": int(len(trials)),
            "n_ok": int((results_df["status"] == "ok").sum()),
            "elapsed_sec": float(elapsed),
            "seed": int(args.seed),
        },
        "top": top_df.to_dict(orient="records"),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2))

    print(
        f"\nGrid search termine | elapsed={elapsed:.1f}s | ok={payload['meta']['n_ok']}/{len(trials)}"
    )
    print(f"CSV: {out_csv}")
    print(f"JSON: {out_json}")

    if top_df.empty:
        print("Aucun trial valide.")
        return

    print("\nTop results:")
    show_cols = [
        "trial_id",
        "objective_score",
        "outperformance",
        "final_model",
        "macro_f1",
        "bal_acc",
        "forward_horizon",
        "forward_buy_threshold",
        "forward_sell_threshold",
        "context_len",
        "walkforward_step",
        "hidden",
        "epochs",
        "decision_mode",
    ]
    print(top_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
