"""Opt-in position experiments. Legacy classifier runners remain unchanged."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trading_system.artifacts.experiment import hash_dataframe, _runtime_metadata, _nullable_metadata
from trading_system.artifacts.serialization import ArtifactManifest, stable_config_hash, save_model_artifact, load_model_artifact
from trading_system.data.scaling import SequenceStandardizer
from trading_system.evaluation.position_gate import (
    GateSearchConfig, GateSelection, PositionGate, fit_position_gate, signal_coverage,
)
from trading_system.evaluation.thresholds import DecisionPolicy
from trading_system.features.expanded import ExpandedFeatureSelector
from trading_system.features.fracdiff import FracDiffTransformer
from trading_system.models.base import FitResult, TrainingHistory
from trading_system.models.specs import ModelSelection
from trading_system.training.financial_loss import FinancialLossConfig, ReturnPanel, probabilities_to_positions
from trading_system.training.position_trainer import fit_position_model, predict_positions
from trading_system.training.overfitting import TrainOnlyFeatureSelector
from .config import ExperimentConfig
from .runner import (
    TrainedModelBundle, _filter_universe, _prepare_splits, _build_split_windows,
    _resolve_sequence_estimator, _select_label_rows, run_validation_experiment,
    _report_sequence_memory,
)


@dataclass
class PositionValidation:
    bundle: TrainedModelBundle
    config: ExperimentConfig
    loss_config: FinancialLossConfig
    validation_metrics: dict
    legacy_validation_metrics: dict | None = None
    test_evaluated: bool = False
    position_gate: PositionGate | None = None
    gate_selection: GateSelection | None = None
    raw_validation_metrics: dict | None = None

    def predict_positions(self, windows):
        positions = probabilities_to_positions(
            self.bundle.predict_proba(windows), self.config.resolved_backtest_position_mode()
        )
        return self.position_gate.apply(positions) if self.position_gate else positions


def _panel(frame, config):
    return ReturnPanel(frame, price_col=config.price_col, date_col=config.date_col,
                       group_col=config.group_col if config.universe == "multi" else None,
                       execution_delay=config.execution_delay)


def _align_position_calendar(frame, config):
    """Keep only dates observed for every asset in a multi-asset portfolio.

    Financial objectives aggregate simultaneous asset returns, so incomplete
    dates cannot be retained without either inventing prices or changing the
    portfolio composition.  The intersection is deterministic and is applied
    before splitting, feature fitting, or label generation.
    """
    if config.universe != "multi":
        return frame
    work = frame.copy()
    dates = pd.to_datetime(work[config.date_col], utc=True, errors="raise")
    keys = pd.DataFrame({"asset": work[config.group_col].to_numpy(),
                         "date": dates.to_numpy()}, index=work.index)
    if keys.duplicated(["asset", "date"]).any():
        raise ValueError("Position comparisons require unique dates per asset.")
    asset_count = work[config.group_col].nunique()
    if asset_count == 0:
        raise ValueError("Position comparisons require at least one asset.")
    coverage = keys.groupby("date", sort=False)["asset"].nunique()
    common_dates = coverage.index[coverage.eq(asset_count)]
    aligned = work.loc[dates.isin(common_dates)].copy()
    if aligned.empty:
        raise ValueError("Position comparisons have no common asset calendar.")
    aligned.attrs.update(frame.attrs)
    return aligned


def _validate_config(frame, config, loss_config):
    if config.evaluation_mode != "static":
        raise ValueError("Position objectives currently support static chronological experiments only.")
    if config.label_mode.startswith("oracle"):
        raise ValueError("Oracle labels are not permitted in objective comparisons.")
    if loss_config.objective != "cross_entropy" and config.sample_weighting is not None:
        raise ValueError("Event sample weights cannot weight a chronological portfolio objective.")
    if config.execution_delay < 1:
        raise ValueError("Position objectives require execution_delay >= 1.")
    if config.universe == "multi":
        dates = None
        for _, part in frame.groupby(config.group_col, sort=False):
            current = pd.to_datetime(part[config.date_col], utc=True).sort_values().to_numpy()
            if dates is not None and not np.array_equal(current, dates):
                raise ValueError("Position comparisons require identical asset calendars before splitting.")
            dates = current


def run_position_validation(frame, config, loss_config, *, progress_callback=None,
                            gate_search: GateSearchConfig | None = None):
    """Fit on train, select checkpoint on validation; never construct test returns."""
    work = _filter_universe(frame, config)
    work = _align_position_calendar(work, config)
    _validate_config(work, config, loss_config)
    if gate_search is not None and loss_config.objective == "cross_entropy":
        raise ValueError("Position gate search requires a financial objective.")
    if loss_config.objective == "cross_entropy":
        # Preserve the previous trainer, class/sample weights, calibration and
        # legacy cash-fee backtest exactly; add a common continuous evaluation.
        legacy = run_validation_experiment(
            work, config, progress_callback=progress_callback
        )
        panel = _panel(legacy.aligned_val_frame, config)
        positions = probabilities_to_positions(legacy.val_probabilities, config.resolved_backtest_position_mode())
        return PositionValidation(legacy.bundle, config, loss_config,
                                  panel.metrics(positions, loss_config, config.initial_capital), legacy.val_backtest)

    selector = ExpandedFeatureSelector(config.expanded_min_coverage) if config.feature_set == "expanded" else None
    overfitting_selector = (
        TrainOnlyFeatureSelector(config.overfitting_control)
        if config.overfitting_control is not None
        else None
    )
    fracdiff = FracDiffTransformer(config.fracdiff, price_col=config.price_col, date_col=config.date_col,
                                  group_col=config.group_col if config.universe == "multi" else None) if config.fracdiff else None
    train, val, _, fills, columns = _prepare_splits(
        work, config, fracdiff_transformer=fracdiff, feature_selector=selector,
        overfitting_selector=overfitting_selector,
        overfitting_supervised=False,
    )
    del work
    X_train, _, aligned_train = _build_split_windows(train, columns, config)
    X_val, _, aligned_val = _build_split_windows(val, columns, config, train)
    purging_state = train.attrs.get("purging")
    del train, val
    if config.purged_split is not None:
        # Direct return losses have one-step price targets. Exclude the gap from
        # their paths, retaining its feature rows only as later causal context.
        train_mask = ~aligned_train["_cv_gap"].to_numpy(dtype=bool)
        val_mask = ~aligned_val["_cv_gap"].to_numpy(dtype=bool)
        X_train, aligned_train = X_train[train_mask], aligned_train.loc[train_mask].copy()
        X_val, aligned_val = X_val[val_mask], aligned_val.loc[val_mask].copy()
    if progress_callback is not None:
        _report_sequence_memory(
            (("train", X_train), ("val", X_val)), progress_callback
        )
    train_panel, val_panel = _panel(aligned_train, config), _panel(aligned_val, config)
    known = aligned_train["_label_known"].to_numpy(dtype=bool)
    if not known.any():
        raise ValueError("Common baseline preprocessing requires observed training labels.")
    # Identical fitted scaler to the CE control, even though direct losses do not
    # need label horizons and can use all observed within-split price returns.
    scaler = SequenceStandardizer()
    scaler.fit(_select_label_rows(X_train, known))
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    model, selection = _resolve_sequence_estimator(None, config, len(columns))
    fit = fit_position_model(model, X_train, train_panel, X_val, val_panel, loss_config, config.resolved_backtest_position_mode())
    del X_train, aligned_train
    positions = predict_positions(model, X_val, config.resolved_backtest_position_mode())
    raw_metrics = val_panel.metrics(positions, loss_config, config.initial_capital)
    gate_selection = None
    if gate_search is not None:
        gate_selection = fit_position_gate(
            positions, val_panel, loss_config, config.initial_capital, gate_search
        )
        filtered = gate_selection.gate.apply(positions)
        metrics = val_panel.metrics(filtered, loss_config, config.initial_capital)
        metrics["signal_coverage"] = signal_coverage(filtered)
        raw_metrics["signal_coverage"] = signal_coverage(positions)
    else:
        metrics = raw_metrics
    bundle = TrainedModelBundle(model, scaler, columns, config.context_len, DecisionPolicy(mode="argmax"),
                                fills.copy(), fit, selection, fracdiff, None, selector,
                                overfitting_selector, purging_state)
    return PositionValidation(
        bundle, config, loss_config, metrics,
        position_gate=gate_selection.gate if gate_selection else None,
        gate_selection=gate_selection,
        raw_validation_metrics=raw_metrics if gate_selection else None,
    )


def evaluate_position_test(frame, validation):
    """Open final test for a frozen candidate without refitting or calibration."""
    if validation.test_evaluated:
        raise ValueError("Final test already evaluated for this position result.")
    config, bundle = validation.config, validation.bundle
    work = _align_position_calendar(_filter_universe(frame, config), config)
    train, val, test, _, columns = _prepare_splits(
        work, config, include_test=True,
        fill_values=bundle.feature_fill_values, fracdiff_transformer=bundle.fracdiff_transformer,
        feature_selector=bundle.feature_selector,
        overfitting_selector=bundle.overfitting_selector,
        overfitting_supervised=validation.loss_config.objective == "cross_entropy",
    )
    if columns != bundle.feature_columns:
        raise ValueError("Final-test features differ from frozen training columns.")
    X, _, aligned = _build_split_windows(test, columns, config, pd.concat([train, val], ignore_index=True))
    del train, val, test
    positions = validation.predict_positions(X)
    panel = _panel(aligned, config)
    metrics = panel.metrics(positions, validation.loss_config, config.initial_capital)
    if validation.gate_selection is not None or validation.position_gate is not None:
        metrics["signal_coverage"] = signal_coverage(positions)
    output = {"continuous": metrics}
    if validation.loss_config.objective == "cross_entropy":
        from trading_system.backtest.engine import evaluate_strategy_vs_buy_hold
        output["legacy"] = evaluate_strategy_vs_buy_hold(
            aligned, bundle.predict(X), initial_capital=config.initial_capital,
            price_col=config.price_col, fee_per_trade=config.fee_per_trade,
            position_mode=config.resolved_backtest_position_mode(), execution_delay=config.execution_delay,
            label_semantics=config.resolved_label_semantics(),
            group_col=config.group_col if config.universe == "multi" else None, date_col=config.date_col,
        )
    validation.test_evaluated = True
    return output


def save_position_artifact(destination, frame, validation):
    config, bundle = validation.config, validation.bundle
    parameters = {
        "kind": "position_objective_v1", "config": asdict(config),
        "loss_config": asdict(validation.loss_config), "dataset_sha256": hash_dataframe(frame),
        "feature_sources": frame.attrs.get("feature_sources"),
        "feature_selection": bundle.feature_selector.state_dict() if bundle.feature_selector else None,
        "overfitting_feature_selection": (
            bundle.overfitting_selector.state_dict()
            if bundle.overfitting_selector else None
        ),
        "fracdiff_state": bundle.fracdiff_transformer.state_dict() if bundle.fracdiff_transformer else None,
        "sample_weight_state": bundle.sample_weight_state,
        "purging": bundle.purging_state,
    }
    parameters = _nullable_metadata(parameters)
    decision = {"position_decoder": "probability_expectation", "position_mode": config.resolved_backtest_position_mode(),
                "legacy_policy": asdict(bundle.decision_policy),
                "position_gate": asdict(validation.position_gate) if validation.position_gate else None}
    canonical = {"model_name": bundle.model_selection.name, "model_parameters": bundle.model_selection.parameters,
                 "experiment_parameters": parameters, "decision_parameters": decision}
    manifest = ArtifactManifest(1, bundle.model_selection.name, bundle.model_selection.parameters, parameters,
                                stable_config_hash(canonical), bundle.feature_columns, bundle.context_len,
                                config.resolved_class_names(), decision, _runtime_metadata())
    state = bundle.scaler.state_dict()
    state["fill_values"] = bundle.feature_fill_values.reindex(bundle.feature_columns).to_numpy(dtype=np.float64)
    return save_model_artifact(destination, manifest=manifest, model_state=bundle.estimator.state_dict(),
                               scaler_state=state, training_history=_nullable_metadata(asdict(bundle.fit_result)),
                               metrics=_nullable_metadata({"validation": validation.validation_metrics,
                                                           "legacy_validation": validation.legacy_validation_metrics}))


def load_position_artifact(source):
    manifest, model_state, scaler_state, diagnostics = load_model_artifact(source)
    parameters = manifest.experiment_parameters
    if parameters.get("kind") != "position_objective_v1":
        raise ValueError("Not a position objective artifact.")
    values = dict(parameters["config"])
    values["model"] = ModelSelection(**values["model"])
    config = ExperimentConfig(**values)
    model, selection = _resolve_sequence_estimator(None, config, len(manifest.feature_columns))
    model.load_state_dict(model_state)
    fills = pd.Series(scaler_state.pop("fill_values"), index=manifest.feature_columns)
    scaler = SequenceStandardizer.from_state_dict(scaler_state)
    selector = None
    if parameters["feature_selection"] is not None:
        selector = ExpandedFeatureSelector(config.expanded_min_coverage)
        selector.state = parameters["feature_selection"]
    fracdiff = FracDiffTransformer.from_state_dict(parameters["fracdiff_state"]) if parameters["fracdiff_state"] else None
    fit_state = dict(diagnostics["training_history"])
    fit_state["history"] = TrainingHistory(**fit_state["history"])
    fit = FitResult(**fit_state)
    overfitting_selector = None
    if parameters.get("overfitting_feature_selection") is not None:
        overfitting_selector = TrainOnlyFeatureSelector(config.overfitting_control)
        overfitting_selector.state = parameters["overfitting_feature_selection"]
    bundle = TrainedModelBundle(model, scaler, manifest.feature_columns, manifest.context_len,
                                DecisionPolicy(**manifest.decision_parameters["legacy_policy"]), fills, fit,
                                selection, fracdiff, parameters["sample_weight_state"], selector,
                                overfitting_selector, parameters.get("purging"))
    gate_state = manifest.decision_parameters.get("position_gate")
    return PositionValidation(bundle, config, FinancialLossConfig(**parameters["loss_config"]),
                              diagnostics["metrics"]["validation"], diagnostics["metrics"]["legacy_validation"],
                              position_gate=PositionGate(**gate_state) if gate_state is not None else None)
