"""Model-free, retrospective comparison of labeling quality."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from tqdm.auto import tqdm

from trading_system.analysis.label_diagnostics import analyze_label_result
from trading_system.data.io import read_parquet_dataset
from trading_system.labels.config import LabelConfig, normalize_label_method
from trading_system.labels.registry import LabelContext, create_default_label_registry

TABLE_NAMES = ("label_runs", "position_trades", "native_events", "transitions", "horizon_probes")
METHODS = ("breakout", "forward_return", "volatility_position", "triple_barrier", "intraday_return")


def _stats(values) -> dict:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return {"count": int(len(array)), **{
        key: float(function(array)) if len(array) else None
        for key, function in (
            ("min", np.min), ("mean", np.mean), ("median", np.median),
            ("p90", lambda a: np.quantile(a, 0.9)), ("max", np.max),
        )
    }}


def summarize_methods(by_ticker: list[dict], tables: dict[str, pd.DataFrame]) -> list[dict]:
    """Pool observations without crossing ticker boundaries or compounding events."""
    summaries = []
    for method in dict.fromkeys(row["method"] for row in by_ticker):
        rows = [row for row in by_ticker if row["method"] == method]
        runs, trades, events, transitions = [
            tables[name].loc[tables[name]["method"].eq(method)]
            for name in TABLE_NAMES[:4]
        ]
        n_known = sum(row["n_known"] for row in rows)
        counts: dict[str, int] = {}
        for row in rows:
            for name, count in row["class_counts"].items():
                counts[name] = counts.get(name, 0) + count
        proportions = {name: count / n_known if n_known else 0.0 for name, count in counts.items()}
        directed = events.loc[events["directed"].eq(True)] if len(events) else events
        active_runs = runs.loc[runs["label_id"].ne(1)]
        total_pairs = int(transitions["count"].sum()) if len(transitions) else 0
        changed = transitions.loc[transitions["from_label"].ne(transitions["to_label"]), "count"].sum()
        summary = {
            "method": method, "semantics": rows[0]["semantics"], "n_tickers": len(rows),
            "n_rows": sum(row["n_rows"] for row in rows), "n_known": n_known,
            "n_unknown": sum(row["n_unknown"] for row in rows),
            "class_counts": counts, "class_proportions": proportions,
            "label_entropy_bits": -sum(p * np.log2(p) for p in proportions.values() if p),
            "label_switch_rate": float(changed / total_pairs) if total_pairs else None,
            "label_run_duration": _stats(runs["length_sessions"]),
            "directional_label_run_duration": _stats(active_runs["length_sessions"]),
            "label_run_duration_by_class": {
                name: _stats(group["length_sessions"])
                for name, group in runs.groupby("label", sort=False)
            },
            "label_run_singleton_fraction": float(runs["length_sessions"].eq(1).mean()) if len(runs) else None,
            "position_trade_duration": _stats(trades["duration_sessions"]),
            "position_trade_gross_return": _stats(trades["gross_return"]),
            "position_trade_net_return": _stats(trades["net_return"]),
            "position_trade_net_win_rate": float(trades["net_return"].gt(0).mean()) if len(trades) else None,
            "position_trade_left_censored": int(trades["left_censored"].sum()),
            "position_trade_right_censored": int(trades["right_censored"].sum()),
            "native_event_duration": _stats(directed["duration_sessions"]),
            "native_event_gross_return": _stats(directed["gross_return"]),
            "native_event_net_return": _stats(directed["net_return"]),
            "turnover_units": sum(row["turnover_units"] for row in rows),
            "protocol": rows[0]["protocol"],
            "fees_bps_one_way": rows[0]["fees_bps"],
            "noise": {
                str(threshold): {
                    "native_small_move_fraction": float(directed["event_return"].abs().lt(threshold).mean()) if len(directed) else None,
                    "position_small_move_fraction": float(trades["gross_return"].abs().lt(threshold).mean()) if len(trades) else None,
                }
                for threshold in rows[0]["noise_thresholds"]
            },
        }
        n_exposure = sum(row["n_exposure_sessions"] for row in rows)
        summary["n_exposure_sessions"] = n_exposure
        summary["turnover_per_session"] = summary["turnover_units"] / n_exposure if n_exposure else None
        for key in ("long_exposure", "short_exposure", "flat_exposure", "unknown_exposure"):
            summary[key] = sum((row[key] or 0.0) * row["n_exposure_sessions"] for row in rows) / n_exposure if n_exposure else None
        summaries.append(summary)
    return summaries


def benchmark_labels(
    frame: pd.DataFrame,
    configs: list[LabelConfig],
    *,
    start: str | None = None,
    end: str | None = None,
    fee_bps: float = 5.0,
    noise_thresholds=(0.002, 0.005, 0.01, 0.015),
    probe_horizons=(1, 5, 10),
    progress: bool = True,
) -> dict:
    """Keep history for warmup, but never read target prices past the end boundary."""
    required = {"date", "ticker", "adj_close"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing benchmark columns: {sorted(missing)}")
    if not configs or len({config.method for config in configs}) != len(configs):
        raise ValueError("Choose at least one method, with no duplicate methods.")
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], utc=True, errors="raise").dt.tz_localize(None)
    if work["date"].isna().any() or work["ticker"].isna().any():
        raise ValueError("Session dates and tickers must be non-missing.")
    if work.duplicated(["ticker", "date"]).any():
        raise ValueError("Session dates must be unique per ticker.")
    lower = pd.to_datetime(start, utc=True).tz_localize(None) if start else None
    upper = pd.to_datetime(end, utc=True).tz_localize(None) if end else None
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("--start must not be later than --end.")
    if upper is not None:
        work = work.loc[work["date"] <= upper]
    if lower is not None:
        eligible = work.loc[work["date"] >= lower, "ticker"].unique()
        work = work.loc[work["ticker"].isin(eligible)]
    if work.empty:
        raise ValueError("No ticker sessions in the requested interval.")
    registry = create_default_label_registry()
    by_ticker, collected = [], {name: [] for name in TABLE_NAMES}
    groups = work.groupby("ticker", sort=True)
    for ticker, group in tqdm(groups, total=groups.ngroups, desc="Label diagnostics", disable=not progress):
        group = group.sort_values("date").reset_index(drop=True)
        results = [registry.generate(group, config, LabelContext(split_col=None)) for config in configs]
        for result in results:
            if not result.frame["date"].reset_index(drop=True).equals(group["date"]):
                raise ValueError(f"Label rows are misaligned for {ticker}.")
        common = np.logical_and.reduce([result.known_mask for result in results])
        for config, result in zip(configs, results):
            analysis = analyze_label_result(
                result, config, ticker=str(ticker), start=start, fee_bps=fee_bps,
                noise_thresholds=tuple(noise_thresholds), probe_horizons=tuple(probe_horizons),
                common_mask=common,
            )
            summary = analysis["summary"]
            summary["ticker"] = str(ticker)
            by_ticker.append(summary)
            for name in TABLE_NAMES:
                collected[name].append(analysis[name].assign(method=config.method))
    tables = {}
    for name, parts in collected.items():
        nonempty = [part for part in parts if len(part)]
        tables[name] = pd.concat(nonempty, ignore_index=True) if nonempty else parts[0].iloc[:0].copy()
    return {
        "summary": summarize_methods(by_ticker, tables), "by_ticker": by_ticker, "tables": tables,
        "configs": [asdict(config) for config in configs],
        "actual_start": work.loc[work["date"] >= lower, "date"].min() if lower is not None else work["date"].min(),
        "actual_end": work["date"].max(),
    }


def _clean_json(value):
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def report_markdown(result: dict) -> str:
    def fmt(value, percent=False):
        return "n/a" if value is None else f"{value * 100:.2f}%" if percent else f"{value:.2f}"

    lines = [
        "# Analyse comparative des labels", "",
        f"Période : {pd.Timestamp(result['actual_start']).date()} au {pd.Timestamp(result['actual_end']).date()}.", "",
        "Aucun entraînement. Labels rétrospectifs, pas résultats d'une stratégie déployable.", "",
        "| Méthode | Tickers | Labels connus | Changement label | Durée couleur moy. | Durée position moy. | Turnover / séance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summary"]:
        lines.append(f"| {row['method']} | {row['n_tickers']} | {row['n_known']} | {fmt(row['label_switch_rate'], True)} | {fmt(row['label_run_duration']['mean'])} | {fmt(row['position_trade_duration']['mean'])} | {fmt(row['turnover_per_session'])} |")
    lines += [
        "", "## Durées des couleurs directionnelles et positions", "",
        "Les couleurs directionnelles excluent Hold/Flat. Les positions décodées conservent Hold pour les méthodes d'action.", "",
        "| Méthode | Couleurs directionnelles min / moy. / max | Positions min / moy. / max | Rendement net position min / moy. / max | Positions gagnantes nettes |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for row in result["summary"]:
        runs, positions, returns = row["directional_label_run_duration"], row["position_trade_duration"], row["position_trade_net_return"]
        lines.append(f"| {row['method']} | {' / '.join(fmt(runs[k]) for k in ('min', 'mean', 'max'))} | {' / '.join(fmt(positions[k]) for k in ('min', 'mean', 'max'))} | {' / '.join(fmt(returns[k], True) for k in ('min', 'mean', 'max'))} | {fmt(row['position_trade_net_win_rate'], True)} |")
    lines += [
        "", "## Rendements natifs des événements directionnels", "",
        "| Méthode | Événements | Durée min / moy. / max | Rendement brut min / moy. / max | Rendement net moy. | Mouvement < 0,2% |",
        "| --- | ---: | --- | --- | ---: | ---: |",
    ]
    for row in result["summary"]:
        duration, returns = row["native_event_duration"], row["native_event_gross_return"]
        noise = row["noise"].get("0.002", {}).get("native_small_move_fraction")
        lines.append(f"| {row['method']} | {returns['count']} | {' / '.join(fmt(duration[k]) for k in ('min', 'mean', 'max'))} | {' / '.join(fmt(returns[k], True) for k in ('min', 'mean', 'max'))} | {fmt(row['native_event_net_return']['mean'], True)} | {fmt(noise, True)} |")
    lines += ["", "## Petits mouvements natifs", "",
              "| Méthode | < 0,2% | < 0,5% | < 1% | < 1,5% |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in result["summary"]:
        fractions = [row["noise"].get(str(threshold), {}).get("native_small_move_fraction") for threshold in (0.002, 0.005, 0.01, 0.015)]
        lines.append(f"| {row['method']} | {' | '.join(fmt(value, True) for value in fractions)} |")
    lines += [
        "", "## Interprétation", "",
        "- Une couleur Buy/Sell est une action pour M0/M1/M3-hold. Hold conserve la position décodée, contrairement à Flat.",
        "- Les épisodes de position close-to-close sont un diagnostic instantané, sans délai, avec liquidation aux limites. Ils ne reproduisent aucun benchmark du modèle.",
        "- Intraday ferme chaque position à la clôture du même jour, même si le label suivant a le même sens. Ses rendements open-to-close ne sont pas directement comparables aux événements close-to-close.",
        "- Les événements triple-barrier et les horizons futurs peuvent se chevaucher. Leurs rendements ne sont jamais composés en PnL portefeuille.",
        "- Les coûts du diagnostic sont en bps par sens : une entrée et une sortie coûtent deux fois le coût indiqué. Les paramètres internes des labelers sont enregistrés séparément.",
        "- Les fractions de petits mouvements sont des indicateurs de faible amplitude, pas une preuve que les labels sont aléatoires ou inapprenables.",
        "- horizon_probes.csv compare le sens des labels sur les mêmes lignes connues par toutes les méthodes, à horizons close-to-close 1/5/10. Un neutre avec mouvement > seuil est compté comme opportunité neutre, pas comme trade réellement raté.",
        "- Les limites censurées sont conservées dans les tables détaillées. Les jours inconnus interrompent les épisodes et ne deviennent jamais Hold/Flat connus.",
        "", "## Fichiers", "",
        "summary.json / summary.csv : agrégats ; by_ticker.csv : détail par actif ; label_runs.parquet : couleurs ; position_trades.parquet : positions décodées ; native_events.parquet : événements ; transitions.csv : transitions ; horizon_probes.csv et horizon_summary.csv : bruit et alignement directionnel.", "",
    ]
    return "\n".join(lines)


def save_benchmark(result: dict, output: Path, metadata: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    report = _clean_json({**metadata, **{key: value for key, value in result.items() if key != "tables"}})
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pd.json_normalize(report["summary"]).to_csv(output / "summary.csv", index=False)
    pd.json_normalize(report["by_ticker"]).to_csv(output / "by_ticker.csv", index=False)
    for name, table in result["tables"].items():
        if name in ("transitions", "horizon_probes"):
            table.to_csv(output / f"{name}.csv", index=False)
        else:
            table.to_parquet(output / f"{name}.parquet", index=False)
    summarize_probes(result["tables"]["horizon_probes"]).to_csv(output / "horizon_summary.csv", index=False)
    (output / "report.md").write_text(report_markdown(result))


def refresh_reports(output: Path) -> None:
    """Rebuild descriptive aggregates from saved tables, without generating labels."""
    report = json.loads((output / "summary.json").read_text())
    tables = {
        name: pd.read_csv(output / f"{name}.csv") if name in ("transitions", "horizon_probes")
        else pd.read_parquet(output / f"{name}.parquet")
        for name in TABLE_NAMES
    }
    report["summary"] = summarize_methods(report["by_ticker"], tables)
    (output / "summary.json").write_text(json.dumps(_clean_json(report), indent=2, allow_nan=False) + "\n")
    pd.json_normalize(report["summary"]).to_csv(output / "summary.csv", index=False)
    summarize_probes(tables["horizon_probes"]).to_csv(output / "horizon_summary.csv", index=False)
    (output / "report.md").write_text(report_markdown(report))


def summarize_probes(probes: pd.DataFrame) -> pd.DataFrame:
    """Pool common-row probes using their observation counts, not ticker means."""
    records = []
    for (method, horizon, threshold), group in probes.groupby(["method", "horizon", "threshold"], sort=False):
        active, opportunities = int(group["n_active"].sum()), int(group["n_opportunities"].sum())
        row = {"method": method, "horizon": int(horizon), "threshold": float(threshold),
               "n_rows": int(group["n_rows"].sum()), "n_active": active,
               "n_neutral": int(group["n_neutral"].sum()), "n_opportunities": opportunities,
               "missed_neutral_count": int(group["missed_neutral_count"].sum()),
               "wrong_way_count": int(group["wrong_way_count"].sum())}
        for key in ("side_accuracy", "small_move_fraction_active", "mean_directed_return"):
            row[key] = float((group[key].fillna(0) * group["n_active"]).sum() / active) if active else None
        row["missed_neutral_rate"] = row["missed_neutral_count"] / opportunities if opportunities else None
        captured = opportunities - row["missed_neutral_count"] - row["wrong_way_count"]
        relevant = opportunities - row["missed_neutral_count"]
        row["opportunity_capture_rate"] = captured / opportunities if opportunities else None
        row["opportunity_direction_accuracy"] = captured / relevant if relevant else None
        records.append(row)
    return pd.DataFrame(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tickers", nargs="+", help="Default: every ticker in the dataset.")
    parser.add_argument("--start")
    parser.add_argument("--end", help="Cap source prices before labeling, including future target paths.")
    parser.add_argument("--methods", nargs="+", type=normalize_label_method, choices=METHODS, default=list(METHODS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Diagnostic cost per one-way turnover unit.")
    parser.add_argument("--noise-thresholds", nargs="+", type=float, default=[0.002, 0.005, 0.01, 0.015])
    parser.add_argument("--probe-horizons", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--label-horizon", type=int, default=10)
    parser.add_argument("--label-window", type=int, default=20)
    parser.add_argument("--label-vol-window", type=int, default=20)
    parser.add_argument("--label-volatility-estimator", choices=("rolling_std", "atr", "bollinger"), default="atr")
    parser.add_argument("--label-profit-barrier", type=float, default=0.75)
    parser.add_argument("--label-stop-barrier", type=float, default=0.75)
    parser.add_argument("--label-event-filter", choices=("all", "cusum"), default="cusum")
    parser.add_argument("--label-cusum-threshold", type=float, default=0.5)
    parser.add_argument("--label-between-events", choices=("hold", "flat", "carry"), default="hold")
    parser.add_argument("--forward-threshold", type=float, default=0.002)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or Path("artifacts/diagnostics") / f"label-quality-{stamp}"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output directory is not empty; choose a new directory to preserve existing results.")
    configs = {
        "breakout": LabelConfig.breakout(window=args.label_window),
        "forward_return": LabelConfig.forward_return(horizon=args.label_horizon, buy_threshold=args.forward_threshold, sell_threshold=args.forward_threshold),
        "volatility_position": LabelConfig.volatility_position(horizon=args.label_horizon, volatility_window=args.label_vol_window, position_mode="long_short", cost_bps=args.cost_bps),
        "triple_barrier": LabelConfig.triple_barrier(max_holding=args.label_horizon, volatility_window=args.label_vol_window, volatility_estimator=args.label_volatility_estimator, profit_barrier=args.label_profit_barrier, stop_barrier=args.label_stop_barrier, event_filter=args.label_event_filter, cusum_threshold=args.label_cusum_threshold, between_event_policy=args.label_between_events, cost_bps=args.cost_bps),
        "intraday_return": LabelConfig.intraday_return(),
    }
    selected = [configs[method] for method in args.methods]
    columns = ["date", "ticker", "adj_close"]
    if "intraday_return" in args.methods:
        columns += ["open", "close"]
    if "triple_barrier" in args.methods and args.label_volatility_estimator == "atr":
        columns += ["high", "low", "close"]
    filter_expr = ds.field("ticker").isin(args.tickers) if args.tickers else None
    frame = read_parquet_dataset(args.data, columns=list(dict.fromkeys(columns)), filter_expr=filter_expr)
    if args.tickers:
        missing = set(args.tickers) - set(frame["ticker"])
        if missing:
            parser.error(f"Tickers not found: {sorted(missing)}")
    try:
        result = benchmark_labels(frame, selected, start=args.start, end=args.end, fee_bps=args.cost_bps,
                                  noise_thresholds=args.noise_thresholds, probe_horizons=args.probe_horizons,
                                  progress=not args.no_progress)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    save_benchmark(result, output, {"schema_version": 1, "created_at": stamp, "dataset_path": str(args.data.resolve()),
                                  "requested_start": args.start, "requested_end": args.end, "training_performed": False})
    print(f"saved={output.resolve()} methods={len(selected)} tickers={result['summary'][0]['n_tickers']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
