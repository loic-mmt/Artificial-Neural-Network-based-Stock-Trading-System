"""Prospective comparability of feature-cap and market-gate interventions."""

from pathlib import Path

import pytest

from trading_system.artifacts.multimodal_study import stable_digest
from trading_system.experiments.multimodal_study import (
    audit_feature_grid, execute_study, load_study_config, plan_study,
)


def _run(cap, columns):
    return {
        "variant_id": f"features-{cap}", "feature_cap": cap,
        "metadata": {
            "config": {"overfitting_control": {"max_features": cap}, "expanded_feature_groups": ["technical", "market", "sector"]},
            "dataset_sha256": "frozen-prices", "calendar": {"sessions": ["day1", "day2"]},
            "cv_spec": {"n_splits": 3}, "final_holdout_opened": False,
        },
        "feature_preprocessing": [{
            "fold": 0, "feature_columns": columns, "input_columns": ["a", "b", "c", "d"],
            "fill_values": {name: 0 for name in columns},
            "scaler": {"mean": [0] * len(columns), "scale": [1] * len(columns)},
            "selector_config": {"max_features": cap, "max_feature_correlation": .95},
            "train_only": True, "train_calendar_sha256": "train", "inner_calendar_sha256": "inner",
            "tickers": ["A", "B"], "fracdiff": None, "purging": {"gap": 5},
        }],
    }


def _runs():
    return [_run(2, ["a", "b"]), _run(4, ["a", "b", "c", "d"])]


def test_feature_grid_is_nested_train_only_and_has_actual_counts():
    audit = audit_feature_grid(_runs())
    assert audit["available"] and audit["complete"] and not audit["errors"]
    row = audit["folds"][0]
    assert row["prefix_nested"] and row["comparable"]
    assert row["additional_columns"] == ["c", "d"]
    assert row["actual_low_count"] == 2 and row["actual_high_count"] == 4


def test_feature_grid_handles_standardizer_keepdims_arrays():
    runs = _runs()
    for run in runs:
        state = run["feature_preprocessing"][0]
        state["scaler"] = {field: [values] for field, values in state["scaler"].items()}
    assert audit_feature_grid(runs)["complete"]


@pytest.mark.parametrize("change", [
    "columns", "pool", "train", "inner", "scaler", "fills", "supervised", "cv", "prices", "groups", "holdout",
])
def test_feature_grid_refuses_confounded_or_leaking_interventions(change):
    runs = _runs()
    high = runs[1]["feature_preprocessing"][0]
    if change == "columns":
        high["feature_columns"] = ["b", "a", "c", "d"]
    elif change == "pool":
        high["input_columns"] = ["a", "b", "c", "extra"]
    elif change in ("train", "inner"):
        high[f"{change}_calendar_sha256"] = "different"
    elif change == "scaler":
        high["scaler"]["mean"][0] = 1
    elif change == "fills":
        high["fill_values"]["a"] = 1
    elif change == "supervised":
        high["train_only"] = False
    elif change == "cv":
        runs[1]["metadata"]["cv_spec"]["n_splits"] = 5
    elif change == "prices":
        runs[1]["metadata"]["dataset_sha256"] = "different"
    elif change == "groups":
        runs[1]["metadata"]["config"]["expanded_feature_groups"] = ["technical", "sector"]
    elif change == "holdout":
        runs[1]["metadata"]["final_holdout_opened"] = True
    audit = audit_feature_grid(runs)
    assert audit["available"] and not audit["complete"] and audit["errors"]


def test_cap_is_a_maximum_and_no_growth_is_explicit():
    runs = [_run(2, ["a", "b"]), _run(4, ["a", "b", "c"])]
    audit = audit_feature_grid(runs)
    assert audit["complete"] and audit["folds"][0]["actual_high_count"] == 3
    assert "retains 3 actual" in audit["warnings"][0]
    runs[1] = _run(4, ["a", "b"])
    audit = audit_feature_grid(runs)
    assert audit["complete"] and not audit["folds"][0]["actual_growth"]
    assert any("no additional" in value for value in audit["warnings"])


def test_failed_feature_audit_blocks_before_any_training_or_writes(tmp_path):
    config = {"common_arguments": ["--overfitting-control"], "feature_caps": [2, 4]}

    def prepare(args):
        cap = int(args[args.index("--overfitting-max-features") + 1])
        return {"cap": cap}

    def planner(cap):
        run = _run(cap, ["a", "b"] if cap == 2 else ["b", "a", "c", "d"])
        state = run["feature_preprocessing"][0]
        spec = {"preprocessing": {
            "feature_columns": state["feature_columns"], "tickers": state["tickers"],
            "fill_values": state["fill_values"], "scaler": state["scaler"],
            "overfitting_selector": {"fit_scope": "train_only", "supervised": False,
                                    "input_columns": state["input_columns"], "config": state["selector_config"]},
        }, "eligible_sessions": {"train": ["one"], "inner": ["two"]}}
        return {"metadata": run["metadata"], "task_specs": [
            {"candidate": "gru", "fold": 0, "seed": 1, "signature": stable_digest(spec), "spec": spec},
        ]}

    plan = plan_study(config, "features", tmp_path / "study", graph_choice="rolling_residual_topk",
                      prepare=prepare, planner=planner)
    assert plan["blocked"]
    with pytest.raises(ValueError, match="not isolated"):
        execute_study(plan, prepare=prepare, runner=lambda **kwargs: pytest.fail("Must not train"))
    assert not (tmp_path / "study").exists()


def test_frozen_cv_and_portable_exposure_cli():
    root = Path(__file__).resolve().parents[1]
    config = load_study_config(root / "configs/benchmark/us_multimodal_optimization.json")
    args = config["common_arguments"]
    for flag, value in (("--cv-initial-train-fraction", "0.5"), ("--cv-inner-val-fraction", "0.2"), ("--cv-embargo-bars", "0")):
        assert args[args.index(flag) + 1] == value
    from trading_system.pipelines.us_multimodal_study import build_compare_parser, build_parser, main
    parsed = build_parser().parse_args([
        "--stage", "features", "--graph-choice", "rolling_residual_topk",
        "--output-dir", "study", "--compare-exposure", "--dry-run",
    ])
    assert parsed.compare_exposure and parsed.dry_run
    assert build_compare_parser().parse_args(["--study-dir", "study", "--exposure-comparison"]).exposure_comparison
    with pytest.raises(ValueError, match="only for"):
        main(["--stage", "gnn-depth", "--output-dir", "study", "--compare-exposure"])


def test_exposure_report_requires_its_supported_grid_before_preparing_or_training(tmp_path):
    from trading_system.artifacts.multimodal_study import atomic_write_json
    from trading_system.pipelines.us_multimodal_study import main
    config = {"schema_version": 1, "common_arguments": ["--overfitting-control"], "feature_caps": [16, 32]}
    path = tmp_path / "study-config.json"
    atomic_write_json(path, config)
    with pytest.raises(ValueError, match="caps 32 and 64"):
        main(["--study-config", str(path), "--stage", "features", "--compare-exposure",
              "--output-dir", str(tmp_path / "study")])
    assert not (tmp_path / "study").exists()
