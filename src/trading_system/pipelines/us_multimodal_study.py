"""Portable CLI for planning and executing the US study's small stage grids."""

from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import statistics

from trading_system.artifacts.multimodal_study import atomic_write_json
from trading_system.experiments.multimodal_study import (
    GRAPH_CHOICES, STAGES, compare_study, execute_study, load_study_config, plan_study,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-config", type=Path,
                        default=Path("configs/benchmark/us_multimodal_optimization.json"))
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--reference-run", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph-choice", choices=GRAPH_CHOICES,
                        help="Explicit G*; never inferred from the unfinished reference run.")
    parser.add_argument("--feature-choice", type=int,
                        help="Predeclared cap selected after features; required for depth stages.")
    parser.add_argument("--data", type=Path, help="Override local frozen price path (e.g. on Windows).")
    parser.add_argument("--market-context-data", type=Path)
    parser.add_argument("--ticker-selection", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--learning-diagnostics", action="store_true",
                        help="Opt-in epoch losses, raw positions and gradient observations; use a new run directory.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute signatures, compatibility and counts; never train or write outputs.")
    parser.add_argument("--compare-exposure", action="store_true",
                        help="After the features stage, report 32/64 x gate interactions, raw and equal-gross-exposure.")
    return parser


def _override(arguments, flag, value):
    if value is None:
        return
    try:
        index = arguments.index(flag)
    except ValueError:
        arguments.extend([flag, str(value)])
    else:
        arguments[index + 1] = str(value)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.compare_exposure and args.stage != "features":
        raise ValueError("--compare-exposure is available only for --stage features.")
    config = load_study_config(args.study_config)
    if args.compare_exposure and sorted(config["feature_caps"]) != [32, 64]:
        raise ValueError("--compare-exposure requires the predeclared feature caps 32 and 64.")
    for name in ("data", "market_context_data", "ticker_selection", "device"):
        _override(config["common_arguments"], "--" + name.replace("_", "-"), getattr(args, name))
    if args.learning_diagnostics and "--learning-diagnostics" not in config["common_arguments"]:
        config["common_arguments"].append("--learning-diagnostics")
    from trading_system.pipelines.compare_gnn_graphs import prepare_graph_run
    from trading_system.experiments.graph_ablation import plan_run_graph_ablation, run_graph_ablation
    plan = plan_study(config, args.stage, args.output_dir,
                      graph_choice=args.graph_choice, feature_choice=args.feature_choice,
                      reference_runs=args.reference_run, prepare=prepare_graph_run,
                      planner=plan_run_graph_ablation)
    if args.dry_run:
        compact = {key: plan[key] for key in ("stage", "graph_choice", "feature_choice", "counts",
                                             "blocked", "incompatibilities")}
        compact["runs"] = [{key: run[key] for key in ("variant_id", "feature_cap", "gnn_layers",
                                                     "market_layers", "tasks")} for run in plan["runs"]]
        compact["feature_audit"] = plan["feature_audit"]
        print(json.dumps(compact, indent=2, allow_nan=False))
        return plan
    result = execute_study(plan, prepare=prepare_graph_run, runner=run_graph_ablation,
                           resume=args.resume)
    if args.compare_exposure:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = args.output_dir / "reports" / f"features-interaction-{stamp}"
        result["comparison"] = compare_main([
            "--study-dir", str(args.output_dir), "--output-dir", str(output), "--exposure-comparison",
        ])
    return result


def build_compare_parser():
    parser = argparse.ArgumentParser(description="Report paired stage comparisons; no automatic winner or final-test access.")
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        help="New report directory; default is <study-dir>/reports.")
    parser.add_argument("--exposure-comparison", action="store_true",
                        help="Recompute costs and feature/gate interactions at equal mean and daily gross exposure.")
    return parser


def _write_csv(path, rows):
    """Flatten metric dictionaries while keeping per-fold details readable."""
    from trading_system.artifacts.multimodal_study import atomic_write_text

    def flatten(row, prefix=""):
        result = {}
        for name, value in row.items():
            key = prefix + name
            if isinstance(value, dict):
                result.update(flatten(value, key + "_"))
            else:
                result[key] = json.dumps(value, allow_nan=False) if isinstance(value, list) else value
        return result

    flattened = [flatten(row) for row in rows]
    fields = list(dict.fromkeys(key for row in flattened for key in row)) or ["complete"]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(flattened)
    atomic_write_text(path, stream.getvalue())


def _interaction_markdown(report):
    details = report.get("feature_interaction")
    if details is None:
        return []
    state = "complete" if details["complete"] else "incomplete or incompatible; no interaction claim"
    lines = ["", "## Feature cap x market gate", "", f"State: {state}.", "",
             "Interaction = (high gate - low gate) - (high plain - low plain).",
             "Positive interaction alone does not establish that the high cap or the gate wins.", "",
             "| Branch | Method | Pairs | Gain features plain | Gain features gated | Gate gain low cap | Gate gain high cap | Interaction Sharpe | Interaction return (pp) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]

    def fmt(value, factor=1):
        return "unavailable" if value is None else f"{value * factor:+.6f}"

    for aggregate in details["aggregates"]:
        rows = [row for row in details["rows"] if row["method"] == aggregate["method"]
                and row["branch"] == aggregate["branch"]]
        def effect(name):
            values = [row[name]["regularized_sharpe"] for row in rows
                      if row[name]["regularized_sharpe"] is not None]
            return statistics.fmean(values) if values else None

        means = aggregate["mean_interaction"]
        lines.append(f"| {aggregate['branch']} | {aggregate['method']} | {aggregate['paired_count']}/{aggregate['expected_count']} | "
                     f"{fmt(effect('gain_high_plain'))} | {fmt(effect('gain_high_gate'))} | "
                     f"{fmt(effect('gate_gain_low'))} | {fmt(effect('gate_gain_high'))} | "
                     f"{fmt(means['regularized_sharpe'])} | {fmt(means['net_return'], 100)} |")
    lines += ["", "Sharpe in this table is regularized net Sharpe. Returns are not annualized.",
              "See feature_interactions.csv for all simple effects and financial metrics by fold/seed.", "",
              "### Actual feature counts", "",
              "| Fold | Low cap count | High cap count | Ordered prefix | Added features | Comparable |",
              "| --- | ---: | ---: | --- | ---: | --- |"]
    for row in details["feature_audit"]:
        lines.append(f"| {row['fold']} | {row.get('actual_low_count', 'unavailable')} | "
                     f"{row.get('actual_high_count', 'unavailable')} | {row.get('prefix_nested', False)} | "
                     f"{len(row.get('additional_columns', []))} | {row['comparable']} |")
    exposure = details["exposure"]
    if exposure["requested"]:
        lines += ["", "### Gross-exposure controls", "",
                  "| Method | Variant | Tasks | Mean gross | Net return | Net Sharpe | Mean max drawdown |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in exposure["aggregates"]:
            metrics = row["mean_metrics"]
            lines.append(f"| {row['method']} | {row['variant_id']} | {row['tasks']} | "
                         f"{fmt(metrics['mean_abs_position'], 100)}% | {fmt(metrics['net_return'], 100)}% | "
                         f"{fmt(metrics['net_sharpe'])} | {fmt(metrics['max_drawdown'], 100)}% |")
    lines += ["", *[f"- {note}" for note in details["notes"]]]
    if details["unavailable"]:
        lines += ["", "### Unavailable or incompatible inputs", "",
                  *[f"- {row['reason']}" for row in details["unavailable"]]]
    return lines


def compare_main(argv=None):
    args = build_compare_parser().parse_args(argv)
    report = compare_study(args.study_dir, exposure_comparison=args.exposure_comparison)
    if args.exposure_comparison:
        report["training_complete"] = report["complete"]
        report["complete"] = report["complete"] and report["feature_interaction"]["complete"]
    target = (args.output_dir or args.study_dir / "reports").expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Report output exists: {target}")
    target.mkdir(parents=True)
    atomic_write_json(target / "report.json", report)
    for name, rows in (("summary", report["variants"]), ("paired_deltas", report["paired_deltas"]),
                       ("paired_aggregates", report["paired_aggregates"])):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else
                                (["variant_id", "complete"] if name == "summary" else
                                 ["baseline", "variant", "fold", "seed", "score_delta", "complete_pair"]))
        writer.writeheader()
        writer.writerows(rows)
        # Atomic publication applies to text/CSV too, via the artifacts helper.
        from trading_system.artifacts.multimodal_study import atomic_write_text
        atomic_write_text(target / f"{name}.csv", stream.getvalue())
    complete = "complete" if report["complete"] else "incomplete; no winning claim is valid"
    lines = ["# US multimodal study", "", f"State: {complete}.", "",
             "Exploratory, paired by (fold, seed); final holdout remains sealed.", "",
             "| Variant | Completed/expected | Mean regularized Sharpe |", "| --- | ---: | ---: |"]
    for row in report["variants"]:
        score = "pending" if row["mean_score"] is None else f"{row['mean_score']:.6f}"
        lines.append(f"| {row['variant_id']} | {row['completed']}/{row['expected']} | {score} |")
    lines += ["", report["promotion"], "", "See paired_deltas.csv for every paired fold/seed, including losses."]
    details = report.get("feature_interaction")
    if details is not None:
        for name, rows in (("feature_interactions", details["rows"]),
                           ("feature_interaction_aggregates", details["aggregates"]),
                           ("feature_audit", details["feature_audit"]),
                           ("exposure_interactions", details["exposure"].get("interaction_rows", [])),
                           ("exposure_metrics", details["exposure"]["rows"]),
                           ("exposure_summary", details["exposure"]["aggregates"])):
            _write_csv(target / f"{name}.csv", rows)
        lines.extend(_interaction_markdown(report))
    from trading_system.artifacts.multimodal_study import atomic_write_text
    atomic_write_text(target / "report.md", "\n".join(lines) + "\n")
    print(json.dumps({"complete": report["complete"], "report_dir": str(target)}, indent=2))
    return report


__all__ = ["build_parser", "main", "build_compare_parser", "compare_main"]
