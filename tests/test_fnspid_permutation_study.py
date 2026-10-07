"""Corpus seeds are paired controls, with one shared reference training run."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_fnspid_polarity_study import _inputs, _fingerprints
from trading_system.artifacts.multimodal_study import atomic_write_json, completion_manifest, file_record, stable_digest
from trading_system.experiments.fnspid_permutation_study import (
    DEFAULT_PERMUTATION_SEEDS, REFERENCE_CANDIDATES, run_fnspid_permutation_study,
)
from trading_system.experiments.news_sentiment_ablation import NewsSentimentAblationConfig
from trading_system.pipelines.multi_ticker_long_short import DEFAULT_CONFIG
from trading_system.training.financial_loss import FinancialLossConfig


def _stub_inputs():
    return {"config": replace(DEFAULT_CONFIG, context_len=5, device="cpu"),
            "loss": FinancialLossConfig("sharpe"), "seeds": [1, 7, 19], "n_splits": 3,
            "ablation": NewsSentimentAblationConfig(news_protocol="fnspid-exploratory", scored_articles_path="verified.parquet")}


def _stub_runner(root, calls, *, fail_preflight_seed=None, different_calendar_seed=None,
                 fail_training_seed=None):
    training_started = False
    def run(**kwargs):
        nonlocal training_started
        assert "_study_cache" not in kwargs, "Injected adapters retain their existing interface."
        ablation, target = kwargs["ablation"], Path(kwargs["destination"])
        calls.append((kwargs["dry_run"], ablation.candidates, ablation.shuffle_seed, kwargs["resume"]))
        if kwargs["dry_run"] and not kwargs["resume"] and not training_started:
            assert not root.exists(), "All child plans must precede any output or training."
        if kwargs["dry_run"] and ablation.candidates == ("gru_features_shuffled",) and ablation.shuffle_seed == fail_preflight_seed:
            raise ValueError("Failed late permutation preflight")
        dates = pd.date_range("2020-01-01", periods=8, tz="UTC")
        metadata = {"config": {"context_len": kwargs["config"].context_len}, "loss_config": {}, "gru_parameters": {},
                    "seeds": kwargs["seeds"], "cv_spec": {}, "cv_folds": [{"fold": fold} for fold in range(kwargs["n_splits"])],
                    "calendar": {"sessions": dates.astype(str).tolist()}, "dataset_sha256": "prices",
                    "news_export_sha256": "daily-data", "news_manifest_sha256": "daily-manifest",
                    "final_split": {"test_start": "2025-01-01"}, "final_holdout_opened": False,
                    "news_manifest": {"inputs": {"scored": {"sha256": "scored"}}, "parquet_sha256": "daily-file"},
                    "provenance": {"code": "test-source"}, "ablation": {"shuffle_seed": ablation.shuffle_seed,
                                                                                "candidates": list(ablation.candidates)}}
        if ablation.candidates == ("gru_features_shuffled",) and ablation.shuffle_seed == different_calendar_seed:
            metadata["calendar"] = {"sessions": ["changed"]}
        tasks = []
        for candidate in ablation.candidates:
            for fold in range(kwargs["n_splits"]):
                for seed in kwargs["seeds"]:
                    tasks.append({"candidate": candidate, "fold": fold, "seed": seed,
                                  "signature": stable_digest((candidate, fold, seed, ablation.shuffle_seed))})
        if kwargs["dry_run"]:
            completed = len(json.loads((target / "folds.json").read_text())) if kwargs["resume"] else 0
            return {"metadata": metadata, "task_specs": tasks, "completed_tasks": completed}
        training_started = True
        if ablation.candidates == ("gru_features_shuffled",) and ablation.shuffle_seed == fail_training_seed:
            raise RuntimeError("Interrupted before next permutation training")
        target.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target / "metadata.json", metadata)
        rows = []
        for task in tasks:
            prediction_path = target / f"{task['candidate']}-{task['fold']}-{task['seed']}.parquet"
            pd.DataFrame({"date": dates, "ticker": "AAPL", "adj_close": 100 + np.arange(len(dates)),
                          "position": .2 if task["candidate"] == "gru_features_shuffled" else .3,
                          "available": True}).to_parquet(prediction_path, index=False)
            score = 1. if task["candidate"] == "gru_features" else .5
            row = {"candidate": task["candidate"], "fold": task["fold"], "seed": task["seed"],
                   "task_signature": task["signature"], "score": score,
                   "outer_metrics": {"net_return": score / 100, "mean_abs_position": .2},
                   "eligible_sessions": {"train": ["2020-01-01"], "inner": ["2020-01-02"], "outer": ["2020-01-03"]},
                   "effective_temporal_columns": ["price", "news"], "fit": {"parameter_count": 10},
                   "prediction_artifacts": {"outer": prediction_path.name},
                   "completion_manifest": completion_manifest(task["signature"], [file_record(prediction_path, target)])}
            rows.append(row)
        atomic_write_json(target / "folds.json", rows)
        return {"metadata": metadata, "folds": rows, "final_test": []}
    return run


def test_dry_run_has_81_fits_and_does_not_write(tmp_path):
    target, calls = tmp_path / "study", []
    result = run_fnspid_permutation_study(_stub_inputs(), target, dry_run=True,
                                           run_callback=_stub_runner(target, calls))
    assert result["planned_tasks"] == 81
    assert result["completed_tasks"] == 0
    assert len(calls) == 6 and all(call[0] for call in calls)
    assert result["runs"][0]["planned_tasks"] == 36
    assert [run["planned_tasks"] for run in result["runs"][1:]] == [9] * 5
    assert result["metadata"]["permutation_seeds"] == list(DEFAULT_PERMUTATION_SEEDS)
    assert result["final_holdout_opened"] is False
    assert not target.exists()


def test_preflight_all_permutations_before_writing_or_fitting(tmp_path):
    target, calls = tmp_path / "study", []
    with pytest.raises(ValueError, match="late permutation"):
        run_fnspid_permutation_study(_stub_inputs(), target, permutation_seeds=(11, 23),
                                    run_callback=_stub_runner(target, calls, fail_preflight_seed=23))
    assert len(calls) == 3 and all(call[0] for call in calls)
    assert not target.exists()


def test_different_calendar_rejected_before_output(tmp_path):
    target, calls = tmp_path / "study", []
    with pytest.raises(ValueError, match="identical"):
        run_fnspid_permutation_study(_stub_inputs(), target, permutation_seeds=(11, 23),
                                    run_callback=_stub_runner(target, calls, different_calendar_seed=23))
    assert not target.exists()


def test_shared_baselines_and_permutation_level_distribution(tmp_path):
    target, calls = tmp_path / "study", []
    report = run_fnspid_permutation_study(_stub_inputs(), target, permutation_seeds=(11, 23),
                                          run_callback=_stub_runner(target, calls))
    assert calls[:3] == [(True, REFERENCE_CANDIDATES, 314159, False),
                         (True, ("gru_features_shuffled",), 11, False),
                         (True, ("gru_features_shuffled",), 23, False)]
    fitted = [call for call in calls if not call[0]]
    assert len(fitted) == 3 and fitted[0][1] == REFERENCE_CANDIDATES
    assert report["metadata"]["planned_tasks"] == 54
    assert [row["tasks"] for row in report["permutations"]] == [9, 9]
    assert report["permutation_distribution"]["mean_score"]["mean"] == .5
    assert report["permutation_distribution"]["mean_score"]["fraction_below_original"] == 1.
    assert len(report["paired_vs_original_features"]) == 27
    assert {row["permutation_seed"] for row in report["paired_vs_original_features"]} == {None, 11, 23}
    assert all(row["score_delta"] == -.5 for row in report["paired_vs_original_features"])
    exposure = report["exposure_controlled"]["tasks"]
    assert len(exposure) == 63  # 4 references + 2 shuffles + buy-and-hold, 9 pairs.
    for fold in (0, 1, 2):
        group = [row for row in exposure if row["fold"] == fold and row["seed"] == 1]
        assert len({row["target_mean_exposure"] for row in group}) == 1
        assert all(row["factor"] <= 1. for row in group)
    assert (target / "report.json").is_file()


@pytest.mark.parametrize("seeds", [(), (1,), (1, 1), (1, -1), (1, 2**32), (1, True), (1, "2"), "1,2"])
def test_bad_permutation_seeds_rejected_without_output(tmp_path, seeds):
    with pytest.raises(ValueError, match="permutation_seeds"):
        run_fnspid_permutation_study(_stub_inputs(), tmp_path / "study", permutation_seeds=seeds)
    assert not (tmp_path / "study").exists()


def test_resume_rejects_changed_seeds_config_or_unregistered_artifacts(tmp_path):
    target, calls = tmp_path / "study", []
    runner = _stub_runner(target, calls)
    inputs = _stub_inputs()
    run_fnspid_permutation_study(inputs, target, permutation_seeds=(11, 23), run_callback=runner)
    before = _fingerprints(target)
    for changed, seeds in ((inputs, (11, 29)), ({**inputs, "config": replace(inputs["config"], context_len=6)}, (11, 23))):
        with pytest.raises(ValueError, match="Resume permutation plan|Unregistered"):
            run_fnspid_permutation_study(changed, target, permutation_seeds=seeds,
                                        run_callback=runner, resume=True, dry_run=True)
        assert _fingerprints(target) == before
    atomic_write_json(target / "reference" / "unregistered.pt", {})
    with pytest.raises(ValueError, match="Unregistered"):
        run_fnspid_permutation_study(inputs, target, permutation_seeds=(11, 23),
                                    run_callback=runner, resume=True, dry_run=True)


def test_interrupted_study_validates_existing_runs_and_missing_future_permutation(tmp_path):
    target, calls = tmp_path / "study", []
    runner = _stub_runner(target, calls, fail_training_seed=23)
    with pytest.raises(RuntimeError, match="Interrupted"):
        run_fnspid_permutation_study(_stub_inputs(), target, permutation_seeds=(11, 23), run_callback=runner)
    before = _fingerprints(target)
    plan = run_fnspid_permutation_study(_stub_inputs(), target, permutation_seeds=(11, 23),
                                       run_callback=runner, resume=True, dry_run=True)
    assert plan["planned_tasks"] == 54 and plan["completed_tasks"] == 45
    assert plan["runs"][-1]["completed_tasks"] == 0
    assert not (target / "shuffle-seed-23").exists()
    assert _fingerprints(target) == before


def test_tiny_cpu_study_matches_originals_and_resumes_without_retraining(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import trading_system.experiments.news_sentiment_ablation as news
    import trading_system.data.fnspid_polarity as polarity

    counters = {"prepare": 0, "scored_articles": 0, "shuffled_prefix": 0}
    original_prepare, original_articles = news._prepare, news._scored_articles
    original_control = polarity.prepare_fold_news_control

    def prepare(*args, **kwargs):
        counters["prepare"] += 1
        return original_prepare(*args, **kwargs)

    def articles(*args, **kwargs):
        counters["scored_articles"] += 1
        return original_articles(*args, **kwargs)

    def control(*args, **kwargs):
        counters["shuffled_prefix"] += int(kwargs.get("mode") == "shuffled" and not kwargs.get("include_outer", False))
        return original_control(*args, **kwargs)

    monkeypatch.setattr(news, "_prepare", prepare)
    monkeypatch.setattr(news, "_scored_articles", articles)
    monkeypatch.setattr(polarity, "prepare_fold_news_control", control)
    market, exported, config, loss, parameters, ablation, daily, scored = _inputs(tmp_path)
    source_before = _fingerprints(tmp_path)
    inputs = dict(frame=market, config=config, loss=loss, gru_parameters=parameters, seeds=[1],
                  sentiment_export=exported, ablation=ablation, n_splits=2, gap_bars=2)
    target = tmp_path / "trained-study"
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        report = run_fnspid_permutation_study(inputs, target, permutation_seeds=(11, 23))
        assert counters == {"prepare": 2, "scored_articles": 1, "shuffled_prefix": 4}
        assert report["metadata"]["planned_tasks"] == 12
        assert report["final_holdout_opened"] is False and report["final_test"] == []
        assert len(report["paired_vs_original_features"]) == 6
        checkpoints = {path: path.stat().st_mtime_ns for path in target.rglob("*.pt")}
        assert len(checkpoints) == 12
        resumed = run_fnspid_permutation_study(inputs, target, permutation_seeds=(11, 23), resume=True)
        assert resumed["permutations"] == report["permutations"]
        assert {path: path.stat().st_mtime_ns for path in checkpoints} == checkpoints
        dry = run_fnspid_permutation_study(inputs, target, permutation_seeds=(11, 23), resume=True, dry_run=True)
        assert dry["completed_tasks"] == 12
    finally:
        torch.set_num_threads(previous)
    after = _fingerprints(tmp_path)
    assert all(after[path] == fingerprint for path, fingerprint in source_before.items())
