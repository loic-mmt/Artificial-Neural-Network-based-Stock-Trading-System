"""Frozen FNSPID corpus permutations with shared, once-trained references.

Model seeds, folds and corpus permutations are different sources of variation.
Reports average matched fold/model-seed pairs within each permutation first;
they never count the resulting fits as independent market observations.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import statistics

import numpy as np
import pandas as pd

from trading_system.artifacts.experiment import _nullable_metadata
from trading_system.artifacts.multimodal_study import (
    artifact_path, atomic_write_json, stable_digest,
)
from trading_system.data.fnspid_export import FNSPID_PROTOCOL
from trading_system.experiments.graph_ablation import _resume_metadata
from trading_system.experiments.news_sentiment_ablation import run_news_sentiment_ablation
from trading_system.training.financial_loss import ReturnPanel


DEFAULT_PERMUTATION_SEEDS = (314159, 271828, 161803, 57721, 141421)
REFERENCE_CANDIDATES = ("gru", "gru_activity", "gru_features", "gru_features_neutralized")
_MANIFEST = "study.manifest.json"
_SHARED_FIELDS = (
    "config", "loss_config", "gru_parameters", "seeds", "cv_spec", "cv_folds",
    "calendar", "dataset_sha256", "news_export_sha256", "news_manifest_sha256",
    "final_split", "final_holdout_opened",
)


def _seeds(values):
    if isinstance(values, (str, bytes)):
        raise ValueError("permutation_seeds must be a sequence of distinct integer seeds.")
    try:
        result = tuple(values)
    except TypeError as error:
        raise ValueError("permutation_seeds must be a sequence of distinct integer seeds.") from error
    if (len(result) < 2 or any(isinstance(value, bool) or not isinstance(value, int)
                             or not 0 <= value < 2**32 for value in result)
            or len(set(result)) != len(result)):
        raise ValueError("Use at least two distinct permutation_seeds in [0, 2**32).")
    return result


def _check_directory(target, names, *, resume):
    if not target.exists():
        if resume:
            raise FileNotFoundError(f"Cannot resume missing permutation study: {target}")
        return
    if not resume:
        raise FileExistsError(f"Permutation study output already exists: {target}")
    if not target.is_dir() or not (target / _MANIFEST).is_file():
        raise ValueError("Cannot resume unregistered permutation study; choose a new output directory.")
    allowed = {_MANIFEST, "report.json", *names}
    unexpected = sorted(path.name for path in target.iterdir() if path.name not in allowed)
    if unexpected:
        raise ValueError(f"Unregistered permutation study artifacts: {unexpected}")
    for name in names:
        child = target / name
        if not child.exists():
            continue
        if not child.is_dir() or not (child / "metadata.json").is_file():
            raise ValueError(f"Unregistered run directory: {name}")
        rows = json.loads((child / "folds.json").read_text(encoding="utf-8")) if (child / "folds.json").exists() else []
        registered = {"metadata.json", "folds.json", "report.json", "exposure-mean-min-daily.parquet"}
        for row in rows:
            for record in row.get("completion_manifest", {}).get("files", []):
                registered.add(record["path"])
        leftovers = sorted(path.relative_to(child).as_posix() for path in child.rglob("*")
                           if path.is_dir() or path.relative_to(child).as_posix() not in registered)
        if leftovers:
            raise ValueError(f"Unregistered run artifacts in {name}: {leftovers}")


def _locked_plan(plans, seeds):
    reference = plans[0]["plan"]["metadata"]
    shared = {key: reference[key] for key in _SHARED_FIELDS}
    if shared["final_holdout_opened"] is not False:
        raise ValueError("Permutation study requires a sealed final holdout.")
    runs = []
    for item in plans:
        plan = item["plan"]
        if {key: plan["metadata"][key] for key in _SHARED_FIELDS} != shared:
            raise ValueError("Permutation runs must have identical price/news inputs, folds and model seeds.")
        tasks = [{key: task[key] for key in ("candidate", "fold", "seed", "signature")}
                 for task in plan["task_specs"]]
        keys = [(task["candidate"], task["fold"], task["seed"]) for task in tasks]
        expected = {(candidate, fold["fold"], seed)
                    for candidate in item["arguments"]["ablation"].candidates
                    for fold in reference["cv_folds"] for seed in reference["seeds"]}
        if len(set(keys)) != len(keys) or set(keys) != expected:
            raise ValueError("Permutation plan contains missing, duplicate or unexpected tasks.")
        runs.append({"name": item["name"], "permutation_seed": item["permutation_seed"],
                     "metadata_sha256": stable_digest(_resume_metadata(plan["metadata"])),
                     "tasks": tasks})
    news = reference["news_manifest"]
    return {"schema_version": 1, "protocol": "fnspid_multiple_permutation_study",
            "permutation_seeds": list(seeds), "reference_shuffle_seed": 314159,
            "reference_candidates": list(REFERENCE_CANDIDATES), "shared": shared,
            "scored_articles_sha256": news["inputs"]["scored"]["sha256"],
            "daily_parquet_sha256": news["parquet_sha256"],
            "daily_manifest_sha256": reference["news_manifest_sha256"],
            "provenance": _resume_metadata({"provenance": reference["provenance"]})["provenance"],
            "runs": runs, "planned_tasks": sum(len(item["tasks"]) for item in runs),
            "final_holdout_opened": False}


def _mean(rows):
    return {"tasks": len(rows), "mean_score": statistics.fmean(row["score"] for row in rows),
            "mean_net_return": statistics.fmean(row["outer_metrics"]["net_return"] for row in rows),
            "mean_gross_exposure": statistics.fmean(row["outer_metrics"]["mean_abs_position"] for row in rows)}


def _pair(row, original, permutation_seed):
    if row["eligible_sessions"] != original["eligible_sessions"]:
        raise ValueError("Compared controls do not have identical eligible sessions.")
    if row["effective_temporal_columns"] != original["effective_temporal_columns"]:
        raise ValueError("Compared polarity controls do not have identical feature dimensions.")
    if row["fit"]["parameter_count"] != original["fit"]["parameter_count"]:
        raise ValueError("Compared polarity controls do not have identical model capacity.")
    return {"candidate": row["candidate"], "reference": "gru_features",
            "permutation_seed": permutation_seed, "fold": row["fold"], "seed": row["seed"],
            "score_delta": row["score"] - original["score"],
            "net_return_delta": row["outer_metrics"]["net_return"] - original["outer_metrics"]["net_return"],
            "gross_exposure_delta": row["outer_metrics"]["mean_abs_position"] - original["outer_metrics"]["mean_abs_position"]}


def _exposure(reports, target, config, loss):
    """Replay all once-trained references and permutations at one common scale."""
    from trading_system.experiments.exposure_comparison import normalize_positions

    groups = {}
    for item in reports:
        for row in item["report"]["folds"]:
            key = row["fold"], row["seed"]
            groups.setdefault(key, []).append((item, row))
    tasks = []
    for (fold, seed), rows in sorted(groups.items()):
        positions, labels, common = {}, {}, None
        for item, row in rows:
            path = artifact_path(row["prediction_artifacts"]["outer"], target / item["name"])
            predictions = pd.read_parquet(path).sort_values(["date", "ticker"]).reset_index(drop=True)
            current = predictions[["date", "ticker", "adj_close"]]
            if (predictions.duplicated(["date", "ticker"]).any() or not predictions.available.eq(True).all()
                    or (common is not None and not current.equals(common))):
                raise ValueError("Exposure comparison requires identical complete available prediction keys/prices.")
            common = current
            label = row["candidate"] if item["permutation_seed"] is None else item["name"]
            positions[label] = predictions.position.to_numpy(dtype=float)
            labels[label] = (row["candidate"], item["permutation_seed"])
        positions["buy_hold"] = np.ones(len(common))
        labels["buy_hold"] = ("buy_hold", None)
        panel = ReturnPanel(common, group_col="ticker", execution_delay=config.execution_delay)
        adjusted, information = normalize_positions(panel, positions, "mean_min")
        for label in positions:
            candidate, permutation_seed = labels[label]
            tasks.append({"candidate": candidate, "permutation_seed": permutation_seed,
                          "fold": fold, "seed": seed, "factor": information["scales"][label],
                          "target_mean_exposure": information["target_mean_exposure"],
                          "raw_metrics": panel.metrics(positions[label], loss, config.initial_capital),
                          "metrics": panel.metrics(adjusted[label], loss, config.initial_capital)})
    summary, pairs = [], []
    labels = sorted({(row["candidate"], row["permutation_seed"]) for row in tasks},
                    key=lambda item: (item[0], -1 if item[1] is None else item[1]))
    originals = {(row["fold"], row["seed"]): row for row in tasks if row["candidate"] == "gru_features"}
    for candidate, permutation_seed in labels:
        selected = [row for row in tasks if (row["candidate"], row["permutation_seed"]) == (candidate, permutation_seed)]
        summary.append({"candidate": candidate, "permutation_seed": permutation_seed, "tasks": len(selected),
                        "mean_net_return": statistics.fmean(row["metrics"]["net_return"] for row in selected),
                        "mean_score": statistics.fmean(row["metrics"]["regularized_sharpe"] for row in selected),
                        "mean_gross_exposure": statistics.fmean(row["metrics"]["mean_abs_position"] for row in selected)})
        if candidate in ("gru_features_shuffled", "gru_features_neutralized", "buy_hold"):
            for row in selected:
                original = originals[row["fold"], row["seed"]]
                pairs.append({"candidate": candidate, "permutation_seed": permutation_seed,
                              "reference": "gru_features", "fold": row["fold"], "seed": row["seed"],
                              "net_return_delta": row["metrics"]["net_return"] - original["metrics"]["net_return"],
                              "score_delta": row["metrics"]["regularized_sharpe"] - original["metrics"]["regularized_sharpe"]})
    return {"method": "mean_min", "descriptive_only": True, "ex_post": True,
            "selection_uses_adjusted_metrics": False, "tasks": tasks, "summary": summary,
            "paired_vs_original_features": pairs,
            "notes": ["One common mean gross exposure across all references, all permutations and delayed equal-weight buy-and-hold per fold/model seed.",
                      "Fresh costs and final liquidation are computed from the adjusted positions.",
                      "Whole-fold normalization is ex post; beta, volatility and net exposure are not equalized."]}


def _aggregate(reports, manifest, target, config, loss):
    reference_rows = reports[0]["report"]["folds"]
    originals = {(row["fold"], row["seed"]): row for row in reference_rows if row["candidate"] == "gru_features"}
    pairs = [_pair(row, originals[row["fold"], row["seed"]], None)
             for row in reference_rows if row["candidate"] == "gru_features_neutralized"]
    permutations = []
    for item in reports[1:]:
        rows = item["report"]["folds"]
        matched = [_pair(row, originals[row["fold"], row["seed"]], item["permutation_seed"]) for row in rows]
        pairs.extend(matched)
        permutations.append({"permutation_seed": item["permutation_seed"], **_mean(rows),
                             "mean_score_delta": statistics.fmean(row["score_delta"] for row in matched),
                             "mean_net_return_delta": statistics.fmean(row["net_return_delta"] for row in matched),
                             "by_fold": [{"fold": fold, **_mean([row for row in rows if row["fold"] == fold]),
                                          "mean_score_delta": statistics.fmean(row["score_delta"] for row in matched if row["fold"] == fold),
                                          "mean_net_return_delta": statistics.fmean(row["net_return_delta"] for row in matched if row["fold"] == fold)}
                                         for fold in sorted({row["fold"] for row in rows})]})
    original_mean = _mean(list(originals.values()))
    distribution = {}
    for field in ("mean_score", "mean_net_return"):
        values = [row[field] for row in permutations]
        distribution[field] = {"mean": statistics.fmean(values), "median": statistics.median(values),
                               "min": min(values), "max": max(values), "std": statistics.pstdev(values),
                               "fraction_below_original": sum(value < original_mean[field] for value in values) / len(values)}
    return _nullable_metadata({"metadata": manifest, "final_test": [], "final_holdout_opened": False,
        "reference_summary": [{"candidate": candidate, **_mean([row for row in reference_rows if row["candidate"] == candidate])}
                              for candidate in REFERENCE_CANDIDATES],
        "original_features": original_mean, "permutations": permutations,
        "permutation_distribution": distribution, "paired_vs_original_features": pairs,
        "exposure_controlled": _exposure(reports, target, config, loss),
        "limitations": ["FNSPID publication-delay availability remains exploratory, not verified point-in-time coverage.",
                        "Permutation means summarize matched folds and model seeds; fits and overlapping calendar folds are not independent observations.",
                        "Reported returns are arithmetic per-run averages, not a concatenated or compounded portfolio.",
                        "No permutation or checkpoint is selected using the closed final holdout."]})


def run_fnspid_permutation_study(inputs, destination, *, permutation_seeds=DEFAULT_PERMUTATION_SEEDS,
                                resume=False, dry_run=False, run_callback=None, compare_callback=None):
    """Validate every child plan before writing, then train shared references once.

    ``inputs`` is the dictionary returned by ``prepare_news_sentiment_run``.
    Optional callbacks have the same keyword interface as the news runner and
    permit small local verification without fitting the full study.
    """
    seeds = _seeds(permutation_seeds)
    if run_callback is not None and compare_callback is not None:
        raise ValueError("Supply only one run_callback or compare_callback.")
    runner = run_callback or compare_callback or run_news_sentiment_ablation
    options = dict(inputs)
    options.pop("destination", None)
    options.pop("resume", None)
    options.pop("dry_run", None)
    ablation = options["ablation"]
    if ablation.news_protocol != FNSPID_PROTOCOL or not ablation.scored_articles_path:
        raise ValueError("Multiple permutations require verified FNSPID scored articles and the exploratory protocol.")
    # This invocation owns the cache. The runner checks exact data/config/fold
    # fingerprints before sharing TRAIN price preparation and scored articles;
    # permutation exports/scalers retain their separate cache identities.
    # Adapters keep their existing public keyword interface.
    if runner is run_news_sentiment_ablation:
        options["_study_cache"] = {}
    target = Path(destination).expanduser().resolve()
    runs = [("reference", None, replace(ablation, candidates=REFERENCE_CANDIDATES, shuffle_seed=314159))]
    runs.extend((f"shuffle-seed-{seed}", seed, replace(ablation, candidates=("gru_features_shuffled",), shuffle_seed=seed))
                for seed in seeds)
    _check_directory(target, [name for name, _, _ in runs], resume=resume)
    plans = []
    for name, seed, settings in runs:
        arguments = {**options, "ablation": settings, "destination": target / name}
        child_resume = resume and (target / name).exists()
        plan = runner(**arguments, resume=child_resume, dry_run=True)
        plans.append({"name": name, "permutation_seed": seed, "arguments": arguments,
                      "resume": child_resume, "plan": plan})
    manifest = _locked_plan(plans, seeds)
    if resume:
        saved = json.loads((target / _MANIFEST).read_text(encoding="utf-8"))
        if saved != manifest:
            raise ValueError("Resume permutation plan differs: seeds, inputs, signatures, code or runtime changed.")
    if dry_run:
        return {"dry_run": True, "destination": str(target), "permutation_seeds": list(seeds),
                "planned_tasks": manifest["planned_tasks"],
                "completed_tasks": sum(item["plan"]["completed_tasks"] for item in plans),
                "runs": [{"name": item["name"], "permutation_seed": item["permutation_seed"],
                          "planned_tasks": len(item["plan"]["task_specs"]),
                          "completed_tasks": item["plan"]["completed_tasks"]} for item in plans],
                "metadata": manifest, "final_holdout_opened": False}
    if not resume:
        target.mkdir(parents=True)
        atomic_write_json(target / _MANIFEST, manifest)
    reports = []
    for item in plans:
        report = runner(**item["arguments"], resume=item["resume"], dry_run=False)
        actual = {(row["candidate"], row["fold"], row["seed"]): row["task_signature"] for row in report["folds"]}
        expected = {(row["candidate"], row["fold"], row["seed"]): row["signature"] for row in item["plan"]["task_specs"]}
        if (actual != expected or len(report["folds"]) != len(expected)
                or _resume_metadata(report["metadata"]) != _resume_metadata(item["plan"]["metadata"])
                or report["metadata"]["final_holdout_opened"] is not False or report.get("final_test")):
            raise ValueError("Completed child report differs from the validated plan or opens the final holdout.")
        reports.append({"name": item["name"], "permutation_seed": item["permutation_seed"], "report": report})
    result = _aggregate(reports, manifest, target, options["config"], options["loss"])
    atomic_write_json(target / "report.json", result)
    return result


__all__ = ["DEFAULT_PERMUTATION_SEEDS", "REFERENCE_CANDIDATES", "run_fnspid_permutation_study"]
