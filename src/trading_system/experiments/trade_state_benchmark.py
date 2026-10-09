"""S0-S3 post-open overnight study, isolated from historical runners."""

from dataclasses import asdict, dataclass
from pathlib import Path
import gc
import json
import os

import numpy as np
import pandas as pd

from trading_system.artifacts.experiment import hash_dataframe
from trading_system.artifacts.multimodal_study import (
    atomic_torch_save, atomic_write_json, file_record, runtime_provenance, stable_digest, validate_files,
)
from trading_system.analysis.label_loss_evaluation import classification_metrics, evaluate_positions
from trading_system.data.post_open import build_post_open_frame
from trading_system.experiments.label_loss_benchmark import (
    LabelLossBenchmarkConfig, _comparisons, _json_safe, _supervision, _write_table,
    calendar_folds, evaluate_oracle, prepare_common_fold,
)
from trading_system.models.neural.trade_state_gru import create_trade_state_gru_classifier
from trading_system.models.specs import ModelBuildContext
from trading_system.training.post_open_panel import PostOpenReturnPanel
from trading_system.training.trade_state import STATE_COLUMNS, STATE_VARIANTS, TradeStateConfig
from trading_system.training.trade_state_trainer import fit_trade_state_model, rollout_trade_state_model


@dataclass(frozen=True)
class TradeStateBenchmarkConfig(LabelLossBenchmarkConfig):
    label_methods: tuple[str, ...] = ("volatility_position",)
    families: tuple[str, ...] = ("cross_entropy", "financial")
    protocols: tuple[str, ...] = ("overnight",)
    decoders: tuple[str, ...] = ("continuous", "sign")
    state_variants: tuple[str, ...] = STATE_VARIANTS
    state_return_scale: float = .1
    state_age_scale: float = 252.
    state_gradient_mode: str = "detached"

    def __post_init__(self):
        super().__post_init__()
        if self.label_methods != ("volatility_position",) or self.protocols != ("overnight",):
            raise ValueError("S0-S3 freezes volatility_position labels and overnight execution.")
        if set(self.families) - {"cross_entropy", "financial"} or set(self.decoders) - {"continuous", "sign"}:
            raise ValueError("S0-S3 supports CE/financial and continuous/sign only.")
        if not self.state_variants or len(set(self.state_variants)) != len(self.state_variants):
            raise ValueError("State variants must be nonempty and unique.")
        for variant in self.state_variants:
            self.state_config(variant)

    def state_config(self, variant):
        return TradeStateConfig(variant, self.state_return_scale, self.state_age_scale, self.state_gradient_mode)


def candidate_grid(config):
    return [{"candidate": f"{family}-{variant}", "family": family, "state_variant": variant,
             "label_method": "volatility_position" if family == "cross_entropy" else None,
             "training_protocol": "overnight"}
            for family in config.families for variant in config.state_variants]


def benchmark_counts(config):
    candidates = len(candidate_grid(config))
    repeats = len(config.seeds) * len(config.folds if config.folds is not None else range(config.n_folds))
    return {"unique_configurations": candidates, "fits": candidates * repeats,
            "evaluation_paths": candidates * repeats * len(config.decoders)}


def _evaluate_fit(model, parts, panels, supervision, candidate, fold, seed, config, directory, plots):
    classification, evaluations = [], []
    model.module.eval()
    for phase in ("train", "inner", "outer"):
        decoders = config.decoders if phase == "outer" else ("continuous",)
        for decoder in decoders:
            rolled = rollout_trade_state_model(model, parts[phase]["X"], panels[phase],
                state_config=config.state_config(candidate["state_variant"]),
                loss_config=config.financial_config(), decoder=decoder)
            identity = {**candidate, "fold": fold, "seed": seed, "protocol": "overnight", "decoder": decoder}
            target = supervision[phase]
            classification.append({**identity, "phase": phase, "diagnostic_label_method": "volatility_position",
                **classification_metrics(target.Label_id.to_numpy(), rolled.probabilities,
                    target._label_known.to_numpy(dtype=bool), majority_class=target.attrs["train_majority_class"])})
            if phase != "outer":
                continue
            name = f"overnight-{decoder}"
            predictions = parts[phase]["aligned"][["date", "ticker"]].copy()
            predictions[["p_short", "p_flat", "p_long"]] = rolled.probabilities
            predictions["signal_position"] = rolled.positions
            _write_table(directory / f"{name}-probabilities.parquet", predictions)
            states = parts[phase]["aligned"][["date", "ticker"]].copy()
            states[list(STATE_COLUMNS)] = rolled.raw_states
            states[[f"input_{col}" for col in STATE_COLUMNS]] = rolled.states
            _write_table(directory / f"{name}-state-inputs.parquet", states)
            result = evaluate_positions(panels[phase], rolled.positions, config.financial_config(),
                                       initial_capital=config.initial_capital, keys=identity)
            for key, suffix in (("daily_paths", "portfolio"), ("position_records", "positions"),
                                ("per_ticker", "tickers"), ("trades", "trades")):
                _write_table(directory / f"{name}-{suffix}.parquet", pd.DataFrame(result[key]))
            atomic_write_json(directory / f"{name}-metrics.json", _json_safe({**identity,
                "metrics": result["metrics"], "exposure_controls": result["exposure_controls"]}))
            evaluations.append({**identity, **result["metrics"], "artifact": name})
            if plots:
                from trading_system.analysis.label_loss_evaluation import plot_evaluation
                plot_evaluation(directory / f"{name}-plots", result["daily_paths"],
                    position_frame=pd.DataFrame(result["position_records"]), prices=parts[phase]["prices"])
    _write_table(directory / "classification.csv", pd.DataFrame(classification))
    atomic_write_json(directory / "classification.json", _json_safe(classification))
    atomic_write_json(directory / "evaluations.json", _json_safe(evaluations))
    return evaluations


def run_trade_state_benchmark(frame, tickers, destination, *, config=None, resume=False, plots=True, progress=True):
    from tqdm.auto import tqdm

    config = config or TradeStateBenchmarkConfig()
    if config.device in ("auto", "cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("An explicit unique ordered ticker universe is required.")
    work = frame.loc[frame.ticker.isin(tickers)].copy()
    work["date"] = pd.to_datetime(work.date, utc=True, errors="raise")
    if work.date.isna().any() or work.duplicated(["date", "ticker"]).any():
        raise ValueError("Unique non-missing ticker/date keys required.")
    work = work.loc[(work.date < pd.to_datetime(config.holdout_start, utc=True)) &
                    (work.date >= pd.to_datetime(config.start, utc=True))].copy()
    if config.end is not None:
        work = work.loc[work.date <= pd.to_datetime(config.end, utc=True)].copy()
    if work.empty or set(tickers) - set(work.ticker):
        raise ValueError("Development range must contain every requested ticker.")
    work = work.sort_values(["ticker", "date"]).reset_index(drop=True)
    if set(config.feature_groups) - {"technical", "market", "sector"}:
        raise ValueError("Only causal technical, market and sector groups are supported.")
    featured, _ = build_post_open_frame(work, groups=config.feature_groups)
    featured["date"] = pd.to_datetime(featured.date, utc=True)
    folds = calendar_folds(work, config)
    selected = [f for f in folds if config.folds is None or f["fold"] in config.folds]
    provenance = runtime_provenance()
    identity = {"schema_version": "trade_state_v1", "config": asdict(config), "tickers": list(tickers),
        "development_sha256": hash_dataframe(work), "source_sha256": provenance["source_sha256"],
        "packages": provenance["runtime"]["packages"], "folds": folds,
        "cuda_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG") if config.device in ("auto", "cuda") else None}
    digest, root = stable_digest(identity), Path(destination).resolve()
    metadata_path = root / "metadata.json"
    if root.exists() and any(root.iterdir()):
        if not resume or not metadata_path.is_file():
            raise FileExistsError("Nonempty output requires --resume and matching metadata.")
        if json.loads(metadata_path.read_text()).get("identity_sha256") != digest:
            raise ValueError("Resume inputs, code, package versions, folds or recipe differ.")
    else:
        root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(metadata_path, _json_safe({**identity, "identity_sha256": digest,
            "runtime": provenance["runtime"], "counts": benchmark_counts(config),
            "state_columns": STATE_COLUMNS, "state_gradient_mode": "detached",
            "state_reset": "cash_at_each_train_inner_outer_partition_and_each_decoder",
            "feature_contract": "J-1 market window + gap_open J; current state outside temporal window",
            "execution": "open_J_to_open_J_plus_1_proxy; five bps per side by default",
            "allocation": "fixed_equal_slots_q_over_N; cash_fraction_is_not_broker_margin",
            "training": "continuous_policy; one global AdamW step per epoch",
            "checkpoint_selection": "inner_validation_only", "final_holdout_opened": False,
            "decoder_contract": "independent_closed_loop_rollout_not_cached_redecoding"}))
    results = []
    with tqdm(total=benchmark_counts(config)["fits"], desc="Trade-state GRU", unit="fit", disable=not progress) as bar:
        for fold in selected:
            parts, preprocessing = prepare_common_fold(featured, fold, config, price_frame=work)
            fold_directory = root / f"fold-{fold['fold']}"
            atomic_write_json(fold_directory / "preprocessing.json", _json_safe(preprocessing))
            supervision = {phase: _supervision(work, part, "volatility_position", fold, phase, config)
                           for phase, part in parts.items()}
            train_target = supervision["train"]
            majority = int(np.bincount(train_target.loc[train_target._label_known, "Label_id"], minlength=3).argmax())
            for target in supervision.values():
                target.attrs["train_majority_class"] = majority
            panels = {phase: PostOpenReturnPanel(part["prices"], protocol="overnight", tickers=tickers,
                      calendar=part["calendar"], signal_frame=part["aligned"]) for phase, part in parts.items()}
            atomic_write_json(fold_directory / "execution_contracts.json", {phase: p.metadata for phase, p in panels.items()})
            target = supervision["outer"]
            _write_table(fold_directory / "volatility_position-labels.parquet", target)
            oracle = evaluate_oracle(panels["outer"], np.where(target._label_known, target.Label_id - 1, 0), config)
            atomic_write_json(fold_directory / "label_oracle.json", _json_safe({"retrospective_not_realizable": True, **oracle}))
            for seed in config.seeds:
                for candidate in candidate_grid(config):
                    directory = fold_directory / f"seed-{seed}" / candidate["candidate"]
                    completion = directory / "complete.json"
                    bar.set_postfix(fold=fold["fold"], seed=seed, candidate=candidate["candidate"], refresh=False)
                    if completion.is_file():
                        completed = json.loads(completion.read_text())
                        if completed.get("identity_sha256") != digest:
                            raise ValueError(f"Incompatible completed task {directory}.")
                        validate_files(completed["files"], root)
                        evaluations = json.loads((directory / "evaluations.json").read_text())
                    else:
                        context = ModelBuildContext(input_size=len(preprocessing["columns"]),
                            context_len=config.context_len, seed=seed, device=config.device)
                        model = create_trade_state_gru_classifier(context, config.model_parameters())
                        epoch_elapsed = [0.]

                        def epoch_progress(record):
                            epoch_elapsed[0] += record["duration_seconds"]
                            cap = model.config.epochs
                            remaining = epoch_elapsed[0] / record["epoch"] * (cap - record["epoch"])
                            fields = {"epoch": f"{record['epoch']}/{cap}",
                                      "val": f"{record['validation']['total_loss']:.4f}",
                                      "epoch_cap_eta": f"{remaining / 60:.1f}m"}
                            if str(model.device).startswith("cuda"):
                                fields["VRAM"] = f"{model.torch.cuda.memory_allocated(model.device) / 1024**3:.2f}GiB"
                            bar.set_postfix(fold=fold["fold"], seed=seed, candidate=candidate["candidate"], **fields, refresh=True)

                        train_target, inner_target = supervision["train"], supervision["inner"]
                        fit = fit_trade_state_model(model, parts["train"]["X"], train_target.Label_id.to_numpy(),
                            train_target._label_known.to_numpy(dtype=bool), parts["inner"]["X"],
                            inner_target.Label_id.to_numpy(), inner_target._label_known.to_numpy(dtype=bool),
                            objective=candidate["family"], train_panel=panels["train"], val_panel=panels["inner"],
                            loss_config=config.financial_config(), state_config=config.state_config(candidate["state_variant"]),
                            progress_callback=epoch_progress)
                        # Persist selected weights before any OUTER reporting.
                        atomic_torch_save(directory / "checkpoint.pt", model.state_dict())
                        atomic_write_json(directory / "learning_diagnostics.json", _json_safe(model.learning_diagnostics_))
                        if plots:
                            from trading_system.analysis.label_loss_evaluation import plot_learning_trace
                            plot_learning_trace(directory / "learning-plots", model.learning_trace_)
                        evaluations = _evaluate_fit(model, parts, panels, supervision, candidate,
                            fold["fold"], seed, config, directory, plots)
                        atomic_write_json(directory / "fit.json", _json_safe({**candidate,
                            "fold": fold["fold"], "seed": seed, "state_config": asdict(config.state_config(candidate["state_variant"])),
                            "best_epoch": fit.best_epoch, "stop_reason": fit.stop_reason,
                            "epochs_ran": len(fit.history.train_loss), "training_duration_seconds": fit.training_duration_seconds,
                            "parameter_count": fit.parameter_count, "device": fit.device,
                            "budget_insufficient": model.learning_diagnostics_["budget_insufficient"],
                            "preprocessing_sha256": stable_digest(preprocessing)}))
                        files = [file_record(p, root) for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "complete.json"]
                        atomic_write_json(completion, {"identity_sha256": digest, "files": files})
                        del model
                    fit_metadata = json.loads((directory / "fit.json").read_text())
                    run_fields = {key: fit_metadata[key] for key in ("best_epoch", "stop_reason", "epochs_ran",
                                  "training_duration_seconds", "parameter_count", "budget_insufficient")}
                    results.extend({**row, **run_fields, "fit_directory": directory.relative_to(root).as_posix()} for row in evaluations)
                    _write_table(root / "results.csv", pd.DataFrame(results))
                    bar.update(1)
            del parts, supervision, panels
            gc.collect()
    opposition = _comparisons(root, results, config, prices=featured, plots=plots)
    table = pd.DataFrame(results)
    metrics = ["net_return", "net_pnl", "net_sharpe", "regularized_sharpe", "max_drawdown",
               "mean_abs_position", "turnover", "outperformance_vs_always_long", "outperformance_vs_exposure_matched_long"]
    aggregate = table.groupby(["candidate", "family", "state_variant", "protocol", "decoder"])[metrics].agg(["count", "mean", "std", "min", "max"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    _write_table(root / "summary.csv", aggregate.reset_index())
    classification = pd.concat([pd.read_csv(root / directory / "classification.csv").assign(fit_directory=directory)
                                for directory in sorted(table.fit_directory.unique())], ignore_index=True)
    _write_table(root / "classification.csv", classification)
    report = {"identity_sha256": digest, "counts": benchmark_counts(config),
        "completed_fits": int(table.fit_directory.nunique()), "completed_evaluation_paths": len(results),
        "opposition_comparisons": len(opposition), "final_holdout_opened": False,
        "selection_performed_on_outer": False, "state_gradient_mode": "detached"}
    atomic_write_json(root / "report.json", report)
    return report
