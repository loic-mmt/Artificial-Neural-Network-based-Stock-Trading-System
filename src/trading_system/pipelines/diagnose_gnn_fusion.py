"""Fixed-position GRU/GNN fusion diagnostic using saved point-7 predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading_system.artifacts.experiment import _nullable_metadata
from trading_system.experiments.graph_fusion_diagnostic import (
    FUSION_GRU_WEIGHTS, GRAPH_CANDIDATES, evaluate_fixed_fusions,
)
from trading_system.training.financial_loss import FinancialLossConfig


def _matched_positions(frame, *, fold, seed, expected_dates, expected_assets):
    pieces = {}
    for name in ("gru", *GRAPH_CANDIDATES):
        part = frame.loc[frame.candidate.eq(name)].sort_values(["date", "ticker"])
        if len(part) != expected_dates * expected_assets:
            raise ValueError(f"Incomplete {name} predictions for fold={fold} seed={seed}.")
        pieces[name] = part.reset_index(drop=True)
    base = pieces["gru"]
    result = base[["date", "ticker", "adj_close"]].copy()
    for name, part in pieces.items():
        if not (base[["date", "ticker"]].equals(part[["date", "ticker"]])
                and np.array_equal(base.adj_close.to_numpy(), part.adj_close.to_numpy())):
            raise ValueError(f"Unmatched prices or ticker/date rows: {name} fold={fold} seed={seed}.")
        result[name] = part.position.to_numpy(dtype=np.float64)
    return result


def run_fusion_diagnostic(information_dir, run_dir, output_dir, *,
                          block_length=20, bootstrap_samples=1000):
    """Evaluate fixed blends only; the ex-post exposure control is descriptive."""
    information_dir, run_dir, output_dir = (
        Path(path).resolve() for path in (information_dir, run_dir, output_dir)
    )
    if output_dir.exists():
        raise FileExistsError(f"Fusion output already exists: {output_dir}")
    information = json.loads((information_dir / "statistics.json").read_text())
    source = json.loads((run_dir / "report.json").read_text())
    metadata = source["metadata"]
    if (source.get("final_test") != [] or metadata.get("final_holdout_opened") is not False
            or information.get("final_holdout_opened") is not False):
        raise ValueError("The final holdout must remain sealed.")
    if (information["source_dataset_sha256"] != metadata["dataset_sha256"]
            or information["replay_dataset_sha256"] != metadata["dataset_sha256"]
            or information.get("hash_mismatch_override")):
        raise ValueError("Information replay must match the benchmark data exactly.")
    predictions = pd.read_parquet(information_dir / "predictions.parquet")
    required = {"date", "ticker", "adj_close", "position", "candidate", "fold", "seed"}
    if required - set(predictions):
        raise ValueError(f"Missing prediction columns: {sorted(required - set(predictions))}")
    predictions["date"] = pd.to_datetime(predictions.date, utc=True, errors="raise")
    if (predictions.empty or predictions.date.isna().any()
            or predictions.duplicated(["candidate", "fold", "seed", "date", "ticker"]).any()
            or predictions.date.max() >= pd.Timestamp(metadata["final_split"]["test_start"])):
        raise ValueError("Predictions are duplicate, empty, undated, or reach the final holdout.")
    expected_pairs = {(fold, seed) for fold in range(metadata["n_splits"])
                      for seed in metadata["seeds"]}
    observed_pairs = set(zip(predictions.fold, predictions.seed))
    if observed_pairs != expected_pairs:
        raise ValueError("Information predictions must cover every benchmark fold and seed.")
    loss = FinancialLossConfig(**metadata["loss_config"])
    config = metadata["config"]
    saved = {(row["candidate"], row["fold"], row["seed"]): row for row in source["folds"]}
    results, daily_paths = [], []
    for fold, seed in sorted(expected_pairs):
        group = predictions.loc[predictions.fold.eq(fold) & predictions.seed.eq(seed)]
        row = saved.get(("gru", fold, seed))
        if row is None or row["status"] != "ok":
            raise ValueError(f"Missing benchmark GRU metrics: fold={fold} seed={seed}.")
        paired = _matched_positions(
            group, fold=fold, seed=seed,
            expected_dates=row["outer_dates"], expected_assets=row["outer_metrics"]["assets"],
        )
        result, daily = evaluate_fixed_fusions(
            paired, loss, execution_delay=config["execution_delay"],
            initial_capital=config["initial_capital"],
            block_length=block_length, samples=bootstrap_samples,
            seed=seed + 1000 * fold,
        )
        if any(not np.isclose(result["gru"][key], row["outer_metrics"][key],
                              rtol=5e-4, atol=5e-4)
               for key in ("regularized_sharpe", "net_return", "max_drawdown")):
            raise ValueError(f"Saved GRU metrics do not match predictions: fold={fold} seed={seed}.")
        results.append({"fold": fold, "seed": seed, **result})
        daily.insert(0, "fold", fold)
        daily.insert(1, "seed", seed)
        daily_paths.append(daily)
    summary = []
    for graph in GRAPH_CANDIDATES:
        for weight in FUSION_GRU_WEIGHTS:
            comparisons = [
                (row["gru"], next(item for item in row["fusions"]
                                  if item["gnn"] == graph and item["gru_weight"] == weight))
                for row in results
            ]
            summary.append({
                "gnn": graph, "gru_weight": weight,
                "mean_gru_regularized_sharpe": float(np.mean([
                    base["regularized_sharpe"] for base, _ in comparisons])),
                "mean_gru_net_return": float(np.mean([
                    base["net_return"] for base, _ in comparisons])),
                "mean_gru_max_drawdown": float(np.mean([
                    base["max_drawdown"] for base, _ in comparisons])),
                "mean_fusion_regularized_sharpe": float(np.mean([
                    item["fusion"]["regularized_sharpe"] for _, item in comparisons])),
                "mean_fusion_net_return": float(np.mean([
                    item["fusion"]["net_return"] for _, item in comparisons])),
                "mean_fusion_max_drawdown": float(np.mean([
                    item["fusion"]["max_drawdown"] for _, item in comparisons])),
                "mean_exposure_matched_gru_sharpe": float(np.mean([
                    item["exposure_matched_gru"]["regularized_sharpe"] for _, item in comparisons])),
                "mean_exposure_matched_gru_net_return": float(np.mean([
                    item["exposure_matched_gru"]["net_return"] for _, item in comparisons])),
                "mean_exposure_matched_gru_max_drawdown": float(np.mean([
                    item["exposure_matched_gru"]["max_drawdown"] for _, item in comparisons])),
                "mean_daily_delta_vs_gru_bps": float(np.mean([
                    item["fusion_minus_gru"]["mean_bps_per_day"] for _, item in comparisons])),
                "mean_daily_delta_vs_exposure_control_bps": float(np.mean([
                    item["fusion_minus_exposure_matched_gru"]["mean_bps_per_day"]
                    for _, item in comparisons])),
                "net_positive_vs_gru": sum(item["fusion"]["net_return"] > base["net_return"]
                                           for base, item in comparisons),
                "net_positive_vs_exposure_control": sum(
                    item["fusion"]["net_return"] > item["exposure_matched_gru"]["net_return"]
                    for _, item in comparisons),
                "drawdown_better_vs_exposure_control": sum(
                    item["fusion"]["max_drawdown"] > item["exposure_matched_gru"]["max_drawdown"]
                    for _, item in comparisons),
                "paired_intervals_above_zero_vs_exposure_control": sum(
                    item["fusion_minus_exposure_matched_gru"]["ci_95_bps_per_day"][0] > 0
                    for _, item in comparisons),
            })
    output = {
        "source_run": str(run_dir), "information_dir": str(information_dir),
        "dataset_sha256": metadata["dataset_sha256"], "final_holdout_opened": False,
        "exploratory": True, "weights_predeclared": list(FUSION_GRU_WEIGHTS),
        "exposure_control": "ex_post_outer_positions_only_not_executable",
        "bootstrap_block_length": block_length, "bootstrap_samples": bootstrap_samples,
        "summary": summary, "folds": results,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(json.dumps(_nullable_metadata(output), indent=2, allow_nan=False))
    pd.concat(daily_paths, ignore_index=True).to_parquet(output_dir / "daily_returns.parquet", index=False)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--information-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args(argv)
    return run_fusion_diagnostic(
        args.information_dir, args.run_dir, args.output_dir,
        block_length=args.block_length, bootstrap_samples=args.bootstrap_samples,
    )


__all__ = ["main", "run_fusion_diagnostic"]
