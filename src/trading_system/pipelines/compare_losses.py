"""Validation-first, matched objective comparison with optional frozen final test."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from trading_system.artifacts.experiment import _nullable_metadata, hash_dataframe
from trading_system.artifacts.serialization import stable_config_hash
from trading_system.data.io import read_parquet_dataset
from trading_system.experiments.position_objectives import (
    run_position_validation, save_position_artifact, load_position_artifact, evaluate_position_test,
)
from trading_system.models.specs import ModelSelection
from trading_system.paths import comparisons_dir
from trading_system.reporting.warnings import current_universe_warning
from trading_system.training.financial_loss import FinancialLossConfig
from .compare_models import build_parser as model_parser, PRESETS, _parameter_sets, load_ticker_selection
from .feature_arguments import apply_feature_arguments, apply_feature_sources
from .label_arguments import apply_label_arguments
from .training_arguments import apply_weight_arguments
from .cv_arguments import validate_cv_arguments, execute_cv
from .overfitting_arguments import overfitting_config_from_args


def build_parser():
    parser = model_parser()
    parser.description = "Compare cross-entropy, net P&L and Sharpe on validation; final test stays sealed by default."
    parser.add_argument("--losses", nargs="+", choices=("cross_entropy", "pnl", "sharpe"), default=["cross_entropy", "pnl", "sharpe"])
    parser.add_argument("--loss-cost-bps", type=float, default=5.)
    parser.add_argument("--loss-sharpe-epsilon", type=float, default=1e-4)
    parser.add_argument("--loss-annualization", type=int, default=252)
    parser.add_argument("--selection-metric", choices=("regularized_sharpe", "net_return"), default="regularized_sharpe")
    parser.add_argument("--final-test", action="store_true", help="After validation selection, evaluate the frozen CE control and financial winner across their predeclared seeds.")
    parser.add_argument("--context-len", type=int)
    parser.add_argument("--train-ratio", type=float)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--position-mode", choices=("long_only", "long_short"))
    parser.add_argument("--execution-delay", type=int)
    return parser


def select_candidates(rows, seeds, metric):
    """Require every predeclared seed; select CE and financial controls separately."""
    summary = []
    for candidate, group in pd.DataFrame(rows).groupby("candidate", sort=True):
        complete = (len(group) == len(seeds) and set(group.seed) == set(seeds)
                    and group.status.eq("ok").all())
        score = float(group[f"val_{metric}"].mean()) if complete else None
        summary.append({"candidate": candidate, "objective": group.objective.iloc[0],
                        "model_name": group.model_name.iloc[0], "complete": bool(complete),
                        "validation_score": score if score is not None and np.isfinite(score) else None})
    selected = []
    for baseline in (True, False):
        eligible = [row for row in summary if (row["objective"] == "cross_entropy") == baseline
                    and row["complete"] and row["validation_score"] is not None]
        if eligible:
            winner = max(eligible, key=lambda row: row["validation_score"])
            selected.append(winner["candidate"])
    return summary, selected


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_cv_arguments(args)
    if len(set(args.losses)) != len(args.losses):
        raise ValueError("--losses must be unique.")
    if args.final_test and args.no_run_artifacts:
        raise ValueError("--final-test requires persisted run artifacts.")
    loss_configs = [FinancialLossConfig(name, args.loss_cost_bps, args.loss_annualization, args.loss_sharpe_epsilon) for name in args.losses]
    config = replace(PRESETS[args.preset], device=args.device)
    config = apply_weight_arguments(apply_feature_arguments(apply_label_arguments(config, args), args), args)
    config = replace(config, overfitting_control=overfitting_config_from_args(args))
    config = replace(config, **{name: getattr(args, name) for name in
                               ("context_len", "train_ratio", "val_ratio", "position_mode", "execution_delay")
                               if getattr(args, name) is not None})
    if config.sample_weighting is not None and any(loss.objective != "cross_entropy" for loss in loss_configs):
        raise ValueError("Sample weighting is supported only by the cross_entropy control.")
    parameter_sets = _parameter_sets(args.models, args.model_parameter_sets)
    from trading_system.models.factory import create_default_model_registry
    if set(args.models) - set(create_default_model_registry().names()):
        raise ValueError("Unknown comparison model.")
    frame = read_parquet_dataset(args.data)
    if args.ticker_selection:
        if config.universe != "multi":
            raise ValueError("Ticker selection requires a multi-ticker preset.")
        selected = load_ticker_selection(args.ticker_selection)
        missing = set(selected) - set(frame[config.group_col])
        if missing:
            raise ValueError(f"Selected tickers missing from dataset: {sorted(missing)}")
        frame = frame[frame[config.group_col].isin(selected)].copy()
    frame, sources = apply_feature_sources(frame, args, config)
    warning = current_universe_warning(args.data)
    if warning:
        print(warning)
    target = args.output_dir or comparisons_dir() / ("losses-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    target = target.expanduser().resolve()
    if args.cv_folds is not None:
        return execute_cv(frame, config, parameter_sets, args, target, loss_configs=loss_configs)
    if target.exists():
        raise FileExistsError(f"Objective comparison output already exists: {target}")
    target.mkdir(parents=True)
    rows = []
    for model_name, parameters_list in parameter_sets.items():
        for parameters in parameters_list:
            for loss_config in loss_configs:
                candidate = stable_config_hash({"model": model_name, "parameters": parameters, "loss": asdict(loss_config)})
                for seed in args.seeds:
                    configured = replace(config, model=ModelSelection(model_name, parameters), seed=seed)
                    row = {"candidate": candidate, "objective": loss_config.objective, "model_name": model_name,
                           "model_parameters": json.dumps(parameters, sort_keys=True), "seed": seed,
                           "config_hash": stable_config_hash({"experiment": asdict(configured), "loss": asdict(loss_config)})}
                    print(f"loss={loss_config.objective} model={model_name} seed={seed}", flush=True)
                    try:
                        validation = run_position_validation(frame, configured, loss_config)
                        row.update(status="ok", best_epoch=validation.bundle.fit_result.best_epoch,
                                   **{f"val_{key}": value for key, value in validation.validation_metrics.items()})
                        if validation.legacy_validation_metrics is not None:
                            row.update({f"legacy_val_{key}": value for key, value in validation.legacy_validation_metrics.items()})
                        if not args.no_run_artifacts:
                            artifact = target / "runs" / f"{len(rows) + 1:04d}-{loss_config.objective}-{model_name}-{seed}"
                            save_position_artifact(artifact, frame, validation)
                            row["artifact_path"] = str(artifact)
                        del validation
                    except Exception as error:
                        if args.fail_fast:
                            raise
                        row.update(status="error", error_type=type(error).__name__, error=str(error))
                    rows.append(row)
    summary, selected = select_candidates(rows, args.seeds, args.selection_metric)
    metadata = {"experiment": asdict(config), "losses": [asdict(item) for item in loss_configs],
                "data_path": str(args.data.resolve()), "dataset_sha256": hash_dataframe(frame),
                "feature_sources": sources, "survivor_bias_warning": warning,
                "selection_metric": args.selection_metric, "seeds": args.seeds,
                "selection": selected, "final_test_requested": args.final_test,
                "evaluation": "continuous probability expectation; proportional turnover costs; flat split boundaries; terminal liquidation"}
    # Persist the selection before accessing any final-test prices or outcomes.
    (target / "selection.json").write_text(json.dumps(_nullable_metadata({"metadata": metadata, "summary": summary}), indent=2, allow_nan=False))
    pd.DataFrame(rows).to_csv(target / "validation.csv", index=False)
    final = []
    if args.final_test:
        if not selected:
            raise ValueError("No complete finite validation candidate; final test remains sealed.")
        for row in rows:
            if row["candidate"] in selected:
                validation = load_position_artifact(row["artifact_path"])
                final.append({"candidate": row["candidate"], "objective": row["objective"], "model_name": row["model_name"],
                              "seed": row["seed"], "metrics": evaluate_position_test(frame, validation)})
                del validation
    report = _nullable_metadata({"metadata": metadata, "validation": rows, "summary": summary, "final_test": final})
    (target / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"saved={target} runs={len(rows)} failures={sum(row['status'] != 'ok' for row in rows)} final_tests={len(final)}")
    return report
