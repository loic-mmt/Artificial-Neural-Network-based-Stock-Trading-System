"""Sequential, nested expanding-fold selection with a sealed final holdout."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading_system.artifacts.experiment import _nullable_metadata, hash_dataframe, save_experiment_artifact
from trading_system.artifacts.serialization import stable_config_hash
from trading_system.data.purged_cv import expanding_calendar_folds
from trading_system.models.specs import ModelSelection
from trading_system.reporting.warnings import current_universe_warning
from .runner import _filter_universe, run_validation_experiment, evaluate_experiment_test
from .position_objectives import run_position_validation, evaluate_position_test, save_position_artifact


def run_purged_cv(frame, config, parameter_sets, seeds, destination, *, n_splits=3,
                  initial_train_fraction=.5, inner_val_fraction=.2, gap_bars=0,
                  embargo_bars=0, loss_configs=None, selection_metric="macro_f1",
                  final_test=False, save_artifacts=True, fail_fast=False, dataset_path=None):
    """Score each candidate on every outer fold/seed; refit only the selected one.

    Fold 'test' fields from shared runners denote outer CV validation here. The
    true final-test rows are absent from every fold frame and first opened after
    selection is written. Memory holds one fitted model at a time.
    """
    if config.purged_split is not None:
        raise ValueError("CV owns calendar boundaries; do not supply purged_split.")
    if config.label_mode.startswith("oracle"):
        raise ValueError("Oracle labels cannot select a CV model.")
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("CV seeds must be unique non-negative integers.")
    if not parameter_sets or any(not values for values in parameter_sets.values()):
        raise ValueError("CV requires non-empty model parameter sets.")
    allowed_metrics = ({"net_return", "net_pnl", "regularized_sharpe"} if loss_configs is not None
                       else {"acc", "bal_acc", "macro_f1", "model_pnl", "outperformance"})
    if selection_metric not in allowed_metrics:
        raise ValueError(f"Unsupported CV score {selection_metric!r}; choose {sorted(allowed_metrics)}.")
    if loss_configs is not None and not loss_configs:
        raise ValueError("loss_configs cannot be empty.")
    frame = _filter_universe(frame, config)
    folds, final_split = expanding_calendar_folds(
        frame, n_splits=n_splits, initial_train_fraction=initial_train_fraction,
        inner_val_fraction=inner_val_fraction,
        final_test_fraction=1 - config.train_ratio - config.val_ratio,
        gap_bars=gap_bars, embargo_bars=embargo_bars, date_col=config.date_col,
    )
    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"CV output already exists: {target}")
    target.mkdir(parents=True)
    dates = pd.to_datetime(frame[config.date_col], utc=True)
    candidates, rows = {}, []
    for name, choices in parameter_sets.items():
        for parameters in choices:
            for loss in loss_configs if loss_configs is not None else [None]:
                description = {"model": name, "parameters": parameters, "loss": asdict(loss) if loss else None}
                candidate = stable_config_hash(description)
                if candidate in candidates:
                    raise ValueError("Duplicate CV candidate configuration.")
                candidates[candidate] = (ModelSelection(name, parameters), loss)
                for seed in seeds:
                    for fold in folds:
                        fold_frame = frame.loc[dates <= pd.Timestamp(fold["end"])].copy()
                        configured = replace(config, model=candidates[candidate][0], seed=seed, purged_split=fold["split"])
                        row = {"candidate": candidate, "model": name, "parameters": parameters,
                               "objective": loss.objective if loss else "cross_entropy",
                               "seed": seed, "fold": fold["fold"],
                               "split": asdict(fold["split"]), "outer_end": fold["end"]}
                        print(f"cv model={name} loss={row['objective']} seed={seed} fold={fold['fold']}", flush=True)
                        try:
                            if loss is None:
                                fitted = run_validation_experiment(fold_frame, configured)
                                evaluated = evaluate_experiment_test(fold_frame, fitted)
                                metrics = {**evaluated.test_metrics, **evaluated.backtest}
                            else:
                                fitted = run_position_validation(fold_frame, configured, loss)
                                evaluated = evaluate_position_test(fold_frame, fitted)
                                metrics = evaluated["continuous"]
                            score = float(metrics[selection_metric])
                            if not np.isfinite(score):
                                raise ValueError("CV selection score must be finite.")
                            row.update(status="ok", score=score, outer_metrics=metrics,
                                       purging=fitted.bundle.purging_state, best_epoch=fitted.bundle.fit_result.best_epoch)
                            if save_artifacts:
                                artifact = target / "folds" / f"{len(rows):05d}"
                                if loss is None:
                                    save_experiment_artifact(artifact, fold_frame, evaluated, dataset_path=dataset_path)
                                else:
                                    save_position_artifact(artifact, fold_frame, fitted)
                                row["artifact_path"] = str(artifact)
                            del fitted, evaluated
                        except Exception as error:
                            if fail_fast:
                                raise
                            row.update(status="error", error_type=type(error).__name__, error=str(error))
                        finally:
                            fitted = evaluated = None
                        rows.append(row)
                        # Incremental progress survives interruption; never mixes final outcomes.
                        (target / "folds.json").write_text(json.dumps(_nullable_metadata(rows), indent=2, allow_nan=False))
    summary = []
    for candidate in candidates:
        runs = [row for row in rows if row["candidate"] == candidate]
        complete = len(runs) == len(seeds) * n_splits and all(row["status"] == "ok" for row in runs)
        scores = [row["score"] for row in runs] if complete else []
        summary.append({"candidate": candidate, "complete": complete,
                        "mean": float(np.mean(scores)) if scores else None,
                        "std": float(np.std(scores)) if scores else None,
                        "min": float(np.min(scores)) if scores else None})
    eligible = [row for row in summary if row["complete"]]
    winner = max(eligible, key=lambda row: row["mean"])["candidate"] if eligible else None
    metadata = {"config": asdict(config), "n_splits": n_splits, "seeds": list(seeds),
                "initial_train_fraction": initial_train_fraction, "inner_val_fraction": inner_val_fraction,
                "final_split": asdict(final_split), "selection_metric": selection_metric,
                "dataset_sha256": hash_dataframe(frame), "feature_sources": frame.attrs.get("feature_sources"),
                "dataset_path": str(Path(dataset_path).resolve()) if dataset_path is not None else None,
                "survivor_bias_warning": current_universe_warning(dataset_path),
                "protocol": "nested_expanding_purged_cv", "selected": winner,
                "embargo_note": "Post-validation embargo has no additional exclusions when training is past-only."}
    (target / "selection.json").write_text(json.dumps(_nullable_metadata({"metadata": metadata, "summary": summary}), indent=2, allow_nan=False))
    final = []
    if final_test:
        if winner is None:
            raise ValueError("No complete CV candidate; final test remains sealed.")
        selection, loss = candidates[winner]
        for seed in seeds:
            configured = replace(config, model=selection, seed=seed, purged_split=final_split)
            if loss is None:
                fitted = run_validation_experiment(frame, configured)
                evaluated = evaluate_experiment_test(frame, fitted)
                metrics = {"classification": evaluated.test_metrics, "backtest": evaluated.backtest}
            else:
                fitted = run_position_validation(frame, configured, loss)
                metrics = evaluate_position_test(frame, fitted)
            if save_artifacts:
                artifact = target / "final" / f"seed-{seed}"
                if loss is None:
                    save_experiment_artifact(artifact, frame, evaluated, dataset_path=dataset_path)
                else:
                    save_position_artifact(artifact, frame, fitted)
            final.append({"candidate": winner, "seed": seed, "metrics": metrics,
                          "purging": fitted.bundle.purging_state})
            del fitted
            if loss is None:
                del evaluated
    report = _nullable_metadata({"metadata": metadata, "folds": rows, "summary": summary, "final_test": final})
    (target / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    pd.DataFrame([{key: value for key, value in row.items() if key not in ("outer_metrics", "purging", "split", "parameters")}
                  for row in rows]).to_csv(target / "folds.csv", index=False)
    print(f"cv_saved={target} selected={winner} final_tests={len(final)}")
    return report
