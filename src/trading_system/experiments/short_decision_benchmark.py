"""Matched short specialization with explicit execution and alert counterfactuals."""

from dataclasses import asdict, dataclass
from pathlib import Path
import gc
import json
import os

import numpy as np
import pandas as pd

from trading_system.analysis.label_loss_evaluation import evaluate_positions, compare_opposite_positions
from trading_system.analysis.short_decisions import binary_head_metrics, causal_asset_trends, short_alert_diagnostics, trend_attribution
from trading_system.artifacts.experiment import hash_dataframe
from trading_system.artifacts.multimodal_study import atomic_torch_save, atomic_write_json, file_record, runtime_provenance, stable_digest, validate_files
from trading_system.data.post_open import build_post_open_frame
from trading_system.experiments.label_loss_benchmark import _json_safe, _supervision, _write_table, calendar_folds, prepare_common_fold, common_exposure_comparison
from trading_system.experiments.trade_state_benchmark import TradeStateBenchmarkConfig
from trading_system.models.neural.short_decision_gru import SHORT_ARCHITECTURES, create_short_decision_gru
from trading_system.models.neural.trade_state_gru import create_trade_state_gru_classifier
from trading_system.models.specs import ModelBuildContext
from trading_system.training.post_open_panel import PostOpenReturnPanel
from trading_system.training.short_decision import SHORT_POLICIES, ShortPolicyConfig, fit_short_decision_model, project_policy, rollout_short_decisions
from trading_system.training.trade_state import STATE_COLUMNS, TradeStateConfig
from trading_system.training.trade_state_trainer import fit_trade_state_model


@dataclass(frozen=True)
class ShortDecisionBenchmarkConfig(TradeStateBenchmarkConfig):
    state_variants: tuple[str, ...] = ("S0", "S3")
    architectures: tuple[str, ...] = SHORT_ARCHITECTURES
    policies: tuple[str, ...] = SHORT_POLICIES
    include_controls: bool = True
    long_activation_threshold: float = .5
    short_activation_threshold: float = .5
    conflict_policy: str = "flat"
    regime_window: int = 200

    def __post_init__(self):
        super().__post_init__()
        if self.state_variants != ("S0", "S3"):
            raise ValueError("Short study fixes S3 states and CE-S0 control.")
        for name, choices in (("architectures", SHORT_ARCHITECTURES), ("policies", SHORT_POLICIES)):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values) or set(values) - set(choices):
                raise ValueError(f"{name} must contain unique choices from {choices}.")
        if not isinstance(self.include_controls, bool): raise ValueError("include_controls must be boolean.")
        if isinstance(self.regime_window, bool) or not isinstance(self.regime_window, int) or self.regime_window < 0 or self.regime_window == 1:
            raise ValueError("Regime window is zero (disabled) or at least two sessions.")
        self.policy_config()

    def policy_config(self):
        return ShortPolicyConfig(self.long_activation_threshold, self.short_activation_threshold, self.conflict_policy)

    @property
    def schema_version(self):
        return "short_decisions_v1"

    def candidates(self):
        grid = [{"candidate": f"{family}-{architecture}", "family": family, "architecture": architecture,
            "state_variant": "S3", "label_method": "volatility_position" if family == "cross_entropy" else None,
            "training_protocol": "overnight", "training_policy": "long_short"}
            for family in self.families for architecture in self.architectures]
        if self.include_controls:
            for family in self.families:
                for state in (("S3", "S0") if family == "cross_entropy" else ("S3",)):
                    grid.append({"candidate": f"{family}-{state}-reference", "family": family,
                        "architecture": "three_class_reference", "state_variant": state,
                        "label_method": "volatility_position" if family == "cross_entropy" else None,
                        "training_protocol": "overnight", "training_policy": "long_short"})
        return grid


def benchmark_counts(config):
    grid = config.candidates(); repeats = len(config.seeds) * len(config.folds if config.folds is not None else range(config.n_folds))
    fits = len(grid) * repeats
    return {"unique_configurations": len(grid), "fits": fits,
        "gru_encoders_trained": repeats * sum(2 if c["architecture"] == "separate_grus" else 1 for c in grid),
        "evaluation_paths": fits * len(config.decoders) * len(config.policies),
        "frozen_alert_counterfactual_paths": fits * len(config.decoders) * 2 if "long_short" in config.policies else 0}


def _evaluate(model, parts, panels, supervision, candidate, fold, seed, config, directory, trends, plots):
    metrics, heads, alerts, regimes, counterfactuals = [], [], [], [], []
    state = config.state_config_for_candidate(candidate); loss = config.financial_config()
    model.module.eval()
    for phase in ("train", "inner", "outer"):
        for decoder in config.decoders if phase == "outer" else ("continuous",):
            for policy in config.policies if phase == "outer" else ("long_short",):
                rolled = rollout_short_decisions(model, parts[phase]["X"], panels[phase],
                    state_config=state, loss_config=loss, decoder=decoder, policy=policy, policy_config=config.policy_config())
                identity = {**candidate, "fold": fold, "seed": seed, "protocol": "overnight", "decoder": decoder, "policy": policy}
                target = supervision[phase]
                reference = candidate["architecture"] == "three_class_reference"
                names = ("long", "short") if reference else model.head_names
                scores = rolled.scores[:, [2, 0]] if reference else rolled.scores
                thresholds = [config.short_activation_threshold if name == "short" else config.long_activation_threshold for name in names]
                heads.extend({**identity, "phase": phase, "score_semantics": "binary_activation_uncalibrated" if candidate["family"] == "financial" else "weighted_supervised_score_uncalibrated",
                    **row} for row in binary_head_metrics(target.Label_id.to_numpy(), target._label_known.to_numpy(dtype=bool), scores, names, thresholds))
                if phase != "outer": continue
                name = f"overnight-{decoder}"
                destination = directory / policy
                predictions = parts[phase]["aligned"][["date", "ticker"]].copy()
                for i, head in enumerate(names): predictions[f"score_{head}"] = scores[:, i]
                predictions["signal_position"] = rolled.positions
                _write_table(destination / f"{name}-scores.parquet", predictions)
                states = parts[phase]["aligned"][["date", "ticker"]].copy()
                states[list(STATE_COLUMNS)] = rolled.raw_states
                states[[f"input_{column}" for column in STATE_COLUMNS]] = rolled.states
                _write_table(destination / f"{name}-state-inputs.parquet", states)
                result = evaluate_positions(panels[phase], rolled.positions, loss, initial_capital=config.initial_capital, keys=identity)
                for key, suffix in (("daily_paths", "portfolio"), ("position_records", "positions"), ("per_ticker", "tickers"), ("trades", "trades")):
                    _write_table(destination / f"{name}-{suffix}.parquet", pd.DataFrame(result[key]))
                atomic_write_json(destination / f"{name}-metrics.json", _json_safe({**identity, "metrics": result["metrics"], "exposure_controls": result["exposure_controls"]}))
                metrics.append({**identity, **result["metrics"], "artifact": f"{policy}/{name}"})
                available = np.zeros(len(rolled.positions), bool); index = panels[phase].indices >= 0
                available[panels[phase].indices[index]] = panels[phase].available[index]
                executed = np.where(available, rolled.positions, 0.)
                alerts.append({**identity, **short_alert_diagnostics(result["position_records"], result["trades"], rolled, executed)})
                if config.regime_window:
                    regimes.extend({**identity, **row} for row in trend_attribution(result["position_records"], trends).to_dict("records"))
                if policy == "long_short":
                    for projection in ("long_flat", "short_flat"):
                        counter = evaluate_positions(panels[phase], project_policy(rolled.positions, projection), loss,
                            initial_capital=config.initial_capital, keys={**identity, "policy": projection})
                        counterfactuals.append({**identity, "counterfactual_policy": projection,
                            "scope": "frozen_long_short_signals_no_state_feedback_no_refit", **counter["metrics"],
                            "delta_net_return_vs_source": counter["metrics"]["net_return"] - result["metrics"]["net_return"],
                            "delta_net_sharpe_vs_source": counter["metrics"]["net_sharpe"] - result["metrics"]["net_sharpe"]})
                if plots:
                    from trading_system.analysis.label_loss_evaluation import plot_evaluation
                    plot_evaluation(destination / f"{name}-plots", result["daily_paths"],
                                    position_frame=pd.DataFrame(result["position_records"]), prices=parts[phase]["prices"])
    for name, rows in (("evaluations", metrics), ("binary-heads", heads), ("short-alerts", alerts), ("regimes", regimes), ("frozen-alert-counterfactuals", counterfactuals)):
        _write_table(directory / f"{name}.csv", pd.DataFrame(rows))
        atomic_write_json(directory / f"{name}.json", _json_safe(rows))
    return metrics


def _comparisons(root, table, config, featured, plots):
    exposure, opposition = [], []
    for keys, group in table.groupby(["fold", "seed", "decoder", "policy"]):
        fold, seed, decoder, policy = keys
        positions = pd.concat([pd.read_parquet(root / row.fit_directory / policy / f"overnight-{decoder}-positions.parquet") for row in group.itertuples()], ignore_index=True)
        identity = {"fold": int(fold), "seed": int(seed), "decoder": decoder, "policy": policy, "protocol": "overnight"}
        exposure.extend({**identity, **row} for row in common_exposure_comparison(positions, config))
        compared = compare_opposite_positions(positions)
        opposition.extend({**identity, **row} for row in compared["summary"])
        destination = root / "oppositions" / f"fold-{fold}-seed-{seed}" / f"{decoder}-{policy}"
        _write_table(destination / "episodes.parquet", pd.DataFrame(compared["episodes"]))
        if plots:
            from trading_system.analysis.label_loss_evaluation import plot_evaluation, plot_oppositions
            paths = pd.concat([pd.read_parquet(root / row.fit_directory / policy / f"overnight-{decoder}-portfolio.parquet") for row in group.itertuples()], ignore_index=True)
            plot_evaluation(destination / "plots", paths)
            if policy == "long_short": plot_oppositions(destination / "plots", positions, featured)
    _write_table(root / "common-exposure.csv", pd.DataFrame(exposure))
    _write_table(root / "oppositions.csv", pd.DataFrame(opposition))
    comparisons = []
    for (fold, seed, decoder, policy, family), group in table.groupby(["fold", "seed", "decoder", "policy", "family"]):
        baseline = group[group.architecture.eq("three_class_reference") & group.state_variant.eq("S3")]
        if baseline.empty: continue
        base = baseline.iloc[0]
        for row in group[group.architecture.ne("three_class_reference")].to_dict("records"):
            item = {k: row[k] for k in ("fold", "seed", "decoder", "policy", "family", "candidate")}
            item["reference"] = base.candidate
            for metric in ("net_return", "net_sharpe", "max_drawdown", "mean_abs_position", "mean_short_exposure", "turnover", "short_pnl_contribution", "parameter_count", "training_duration_seconds"):
                item[f"delta_{metric}"] = row[metric] - base[metric]
            comparisons.append(item)
    _write_table(root / "short-paired.csv", pd.DataFrame(comparisons))


def run_short_decision_benchmark(frame, tickers, destination, *, config=None, resume=False, plots=True, progress=True):
    from tqdm.auto import tqdm
    config = config or ShortDecisionBenchmarkConfig()
    if not isinstance(config, ShortDecisionBenchmarkConfig): raise TypeError("ShortDecisionBenchmarkConfig required.")
    if config.device in ("auto", "cuda"): os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not tickers or len(set(tickers)) != len(tickers): raise ValueError("Explicit unique ticker universe required.")
    work = frame.loc[frame.ticker.isin(tickers)].copy(); work["date"] = pd.to_datetime(work.date, utc=True, errors="raise")
    if work.date.isna().any() or work.duplicated(["date", "ticker"]).any(): raise ValueError("Unique nonmissing ticker/date keys required.")
    work = work.loc[(work.date < pd.to_datetime(config.holdout_start, utc=True)) & (work.date >= pd.to_datetime(config.start, utc=True))]
    if config.end is not None: work = work.loc[work.date <= pd.to_datetime(config.end, utc=True)]
    if work.empty or set(tickers) - set(work.ticker): raise ValueError("Every ticker must have development data.")
    work = work.sort_values(["ticker", "date"]).reset_index(drop=True)
    if set(config.feature_groups) - {"technical", "market", "sector"}: raise ValueError("Only causal technical, market and sector features supported.")
    featured, _ = build_post_open_frame(work, groups=config.feature_groups)
    featured["date"] = pd.to_datetime(featured.date, utc=True)
    folds = calendar_folds(work, config); selected = [f for f in folds if config.folds is None or f["fold"] in config.folds]
    provenance = runtime_provenance(); root = Path(destination).resolve()
    identity = {"schema_version": config.schema_version, "config": asdict(config), "tickers": list(tickers),
        "development_sha256": hash_dataframe(work), "source_sha256": provenance["source_sha256"],
        "packages": provenance["runtime"]["packages"], "folds": folds}
    digest = stable_digest(identity); metadata = root / "metadata.json"
    if root.exists() and any(root.iterdir()):
        if not resume or not metadata.is_file(): raise FileExistsError("Nonempty output requires --resume and matching metadata.")
        if json.loads(metadata.read_text()).get("identity_sha256") != digest: raise ValueError("Resume inputs, sources, runtime or recipe differ.")
    else:
        root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(metadata, _json_safe({**identity, "identity_sha256": digest, "counts": benchmark_counts(config),
            "runtime": provenance["runtime"], "candidates": config.candidates(), "final_holdout_opened": False,
            "execution": "open_J_to_open_J_plus_1_proxy", "allocation": "fixed_equal_ticker_slots_q_over_N",
            "feature_contract": "completed_J_minus_1_plus_gap_open_J", "state_gradient_mode": "detached",
            "training": "continuous_long_short; one_global_AdamW_step_per_epoch; inner_checkpoint",
            "binary_targets": "Short vs not-Short; Long vs not-Long; known TRAIN-only class weights",
            "threshold_selection": "fixed_a_priori_no_outer_optimization_no_calibration_claim",
            "separate_grus": "independent_parameters_joint_loss_and_checkpoint_same_executed_book",
            "continuous_conflicts": "q=long_activation-short_activation; detector_q=1-2*short_activation",
            "closed_loop_controls": "independent_book_each_policy; alerts_can_change",
            "frozen_alert_controls": "same_long_short_signals_projected_without_feedback",
            "regime_definition": f"asset_close_J_minus_1_vs_SMA_{config.regime_window}; reporting_only",
            "short_borrow_and_financing_costs": "not_modeled; no_broker_short_availability_validation"}))
    records = []; trends = causal_asset_trends(work, config.regime_window)
    with tqdm(total=benchmark_counts(config)["fits"], desc="Short-decision GRU", unit="fit", disable=not progress) as bar:
        for fold in selected:
            parts, preprocessing = prepare_common_fold(featured, fold, config, price_frame=work)
            folder = root / f"fold-{fold['fold']}"; atomic_write_json(folder / "preprocessing.json", _json_safe(preprocessing))
            targets = {phase: _supervision(work, part, "volatility_position", fold, phase, config) for phase, part in parts.items()}
            _write_table(folder / "volatility-position-labels.parquet", targets["outer"])
            panels = {phase: PostOpenReturnPanel(part["prices"], protocol="overnight", tickers=tickers,
                calendar=part["calendar"], signal_frame=part["aligned"]) for phase, part in parts.items()}
            atomic_write_json(folder / "execution_contracts.json", {phase: p.metadata for phase, p in panels.items()})
            for seed in config.seeds:
                for candidate in config.candidates():
                    directory = folder / f"seed-{seed}" / candidate["candidate"]; completion = directory / "complete.json"
                    bar.set_postfix(fold=fold["fold"], seed=seed, candidate=candidate["candidate"], refresh=False)
                    if completion.is_file():
                        manifest = json.loads(completion.read_text())
                        if manifest.get("identity_sha256") != digest: raise ValueError("Incompatible completed fit.")
                        validate_files(manifest["files"], root)
                        evaluations = json.loads((directory / "evaluations.json").read_text())
                    else:
                        context = ModelBuildContext(input_size=len(preprocessing["columns"]), context_len=config.context_len, seed=seed, device=config.device)
                        reference = candidate["architecture"] == "three_class_reference"
                        model = create_trade_state_gru_classifier(context, config.model_parameters()) if reference else create_short_decision_gru(context, config.model_parameters(), candidate["architecture"])
                        elapsed = [0.]
                        def update(record):
                            elapsed[0] += record["duration_seconds"]
                            fields = {"epoch": f"{record['epoch']}/{model.config.epochs}", "val": f"{record['validation']['total_loss']:.4f}",
                                "epoch_cap_eta": f"{elapsed[0] / record['epoch'] * (model.config.epochs - record['epoch']) / 60:.1f}m"}
                            if str(model.device).startswith("cuda"): fields["VRAM"] = f"{model.torch.cuda.memory_allocated(model.device) / 1024**3:.2f}GiB"
                            bar.set_postfix(fold=fold["fold"], seed=seed, candidate=candidate["candidate"], **fields, refresh=True)
                        fitter = fit_trade_state_model if reference else fit_short_decision_model
                        fit = fitter(model, parts["train"]["X"], targets["train"].Label_id.to_numpy(), targets["train"]._label_known.to_numpy(dtype=bool),
                            parts["inner"]["X"], targets["inner"].Label_id.to_numpy(), targets["inner"]._label_known.to_numpy(dtype=bool),
                            objective=candidate["family"], train_panel=panels["train"], val_panel=panels["inner"],
                            loss_config=config.financial_config(), state_config=config.state_config_for_candidate(candidate), progress_callback=update)
                        atomic_torch_save(directory / "checkpoint.pt", model.state_dict())
                        atomic_write_json(directory / "learning_diagnostics.json", _json_safe(model.learning_diagnostics_))
                        if plots:
                            from trading_system.analysis.label_loss_evaluation import plot_learning_trace
                            plot_learning_trace(directory / "learning-plots", model.learning_trace_)
                        evaluations = _evaluate(model, parts, panels, targets, candidate, fold["fold"], seed, config, directory, trends, plots)
                        atomic_write_json(directory / "fit.json", _json_safe({**candidate, "fold": fold["fold"], "seed": seed,
                            "best_epoch": fit.best_epoch, "stop_reason": fit.stop_reason, "epochs_ran": len(fit.history.train_loss),
                            "training_duration_seconds": fit.training_duration_seconds, "parameter_count": fit.parameter_count,
                            "device": fit.device, "budget_insufficient": model.learning_diagnostics_["budget_insufficient"],
                            "preprocessing_sha256": stable_digest(preprocessing)}))
                        files = [file_record(p, root) for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "complete.json"]
                        atomic_write_json(completion, {"identity_sha256": digest, "files": files}); del model
                    fit = json.loads((directory / "fit.json").read_text())
                    fields = {key: fit[key] for key in ("best_epoch", "stop_reason", "epochs_ran", "training_duration_seconds", "parameter_count", "budget_insufficient")}
                    records.extend({**row, **fields, "fit_directory": directory.relative_to(root).as_posix()} for row in evaluations)
                    _write_table(root / "results.csv", pd.DataFrame(records)); bar.update(1)
            del parts, panels, targets; gc.collect()
    table = pd.DataFrame(records)
    _comparisons(root, table, config, featured, plots)
    metrics = ["net_return", "net_sharpe", "max_drawdown", "mean_abs_position", "mean_short_exposure", "turnover", "short_pnl_contribution", "outperformance_vs_always_long"]
    aggregate = table.groupby(["candidate", "family", "architecture", "decoder", "policy"])[metrics].agg(["count", "mean", "std", "min", "max"])
    aggregate.columns = [f"{m}_{stat}" for m, stat in aggregate.columns]; _write_table(root / "summary.csv", aggregate.reset_index())
    for name in ("binary-heads", "short-alerts", "regimes", "frozen-alert-counterfactuals"):
        rows = [row for directory in sorted(table.fit_directory.unique()) for row in json.loads((root / directory / f"{name}.json").read_text())]
        _write_table(root / f"{name}.csv", pd.DataFrame(rows))
    counterfactual_count = sum(len(json.loads((root / directory / "frozen-alert-counterfactuals.json").read_text()))
                               for directory in table.fit_directory.unique())
    report = {"counts": benchmark_counts(config), "completed_fits": int(table.fit_directory.nunique()),
        "completed_evaluation_paths": len(records), "final_holdout_opened": False, "selection_performed_on_outer": False,
        "completed_frozen_alert_counterfactual_paths": counterfactual_count,
        "identity_sha256": digest, "state_gradient_mode": "detached"}
    atomic_write_json(root / "report.json", report)
    return report
