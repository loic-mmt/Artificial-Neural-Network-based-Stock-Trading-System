"""Fixed OU exit ablations on frozen validation predictions; no tuning/holdout."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
from pathlib import Path
import re

import pandas as pd

from .config import TradingConfig
from .engine import run_trading_backtest
from .ou import AffineCosts, OUConfig, OUParameters, solve_ou_boundaries
from .replay import _json, file_sha256, save_result


def variants(fees=5., slippage=2.):
    """Predeclared experimental values; these are not recommended settings."""
    base = TradingConfig(enabled=True, execution="next_open", signal_timing="after_open",
                         fees_bps=fees, slippage_bps=slippage, trailing_stop_pct=.04,
                         trailing_activation_pct=0., reentry="next_bar")
    return {
        "without_rules": replace(base, enabled=False),
        "trailing_only": base,
        "trailing_fixed_tp": replace(base, take_profit_pct=.10),
        "trailing_ou_exit": replace(base, ou=OUConfig(mode="exit")),
        "trailing_ou_entry_exit": replace(base, ou=OUConfig(mode="entry_exit")),
    }


def _evaluate(job):
    signal_path, data_path, folder, identity = job
    signals = pd.read_parquet(signal_path)
    signals["date"] = pd.to_datetime(signals.date, utc=True)
    if signals.date.ge(pd.Timestamp("2022-05-10T00:00Z")).any():
        raise ValueError("Validation inputs enter the sealed final holdout.")
    data = pd.read_parquet(data_path, columns=["date", "ticker", "open", "high", "low", "close", "dividends", "stock_splits", "sector"])
    data["date"] = pd.to_datetime(data.date, utc=True)
    data = data.loc[data.ticker.isin(signals.ticker.unique()) & (data.date <= signals.date.max())]
    bars = signals[["date", "ticker"]].merge(data, on=["date", "ticker"], how="left", validate="one_to_one")
    history = data.loc[data.date < signals.date.min()].copy()
    # Only window bars are used; retain synchronized last window + one bar.
    dates = sorted(history.date.unique())[-253:]
    history = history.loc[history.date.isin(dates)]
    folder = Path(folder)
    folder.mkdir()
    rows = []
    for scenario, multiplier in (("base_costs", 1.), ("costs_x3", 3.)):
        scenario_folder = folder / scenario
        scenario_folder.mkdir()
        for name, config in variants(5*multiplier, 2*multiplier).items():
            result = run_trading_backtest(bars, signals, config, history=history)
            save_result(result, scenario_folder / name)
            decisions = result.decisions
            positive = decisions.raw_target.gt(0)
            row = {**identity, "scenario": scenario, "variant": name,
                   **{k: v for k, v in result.metrics.items() if not isinstance(v, dict)},
                   "valid_ou_fraction": float(decisions.ou_valid.fillna(False).astype(bool).mean()),
                   "long_signal_count": int(positive.sum()),
                   "long_entry_block_fraction": float(decisions.loc[positive, "reasons"].str.contains("ou_entry_block|ou_above_sell_block").mean()) if positive.any() else 0.,
                   "reentry_block_fraction": float(decisions.reasons.str.contains("reentry_block").mean()),
                   "computed_tp_exits": int(result.trades.exit_reason.eq("ou_take_profit").sum()),
                   "ambiguous_bars": result.metadata["ambiguous_tp_stop_bars"]}
            rows.append(row)
    _json(folder / "inputs.json", {"signals_sha256": file_sha256(signal_path),
                                  "first_date": signals.date.min().isoformat(),
                                  "last_date": signals.date.max().isoformat(),
                                  "history_last_date": history.date.max().isoformat(),
                                  "final_holdout_opened": False})
    return rows


def validate_directory(signals_dir, data, output, *, families=("open_gap", "lagged"), workers=3):
    output, signals_dir, data = Path(output), Path(signals_dir), Path(data)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not families or any(f not in {"open_gap", "lagged"} for f in families):
        raise ValueError("Specify open_gap and/or lagged families.")
    if workers < 1:
        raise ValueError("workers must be positive.")
    jobs = []
    for family in families:
        for path in sorted(signals_dir.glob(f"fold-*-seed-*-{family}-positions.parquet")):
            match = re.fullmatch(rf"fold-(\d+)-seed-(\d+)-{family}-positions.parquet", path.name)
            if match:
                identity = {"family": family, "fold": int(match[1]), "seed": int(match[2])}
                jobs.append((path, data, output / path.stem, identity))
    if not jobs:
        raise ValueError("No matching frozen validation prediction files.")
    # Manifest is frozen before inspecting scores. The output remains visibly
    # incomplete until every independent replay has succeeded.
    output.mkdir(parents=True)
    benchmark = solve_ou_boundaries(OUParameters(.6, 1., .2), .3,
                                    costs=AffineCosts(buy_fixed=.02, sell_fixed=.02))
    manifest = {"status": "running", "final_holdout_opened": False,
                "variants": {scenario: {name: asdict(c) for name, c in variants(5*m, 2*m).items()}
                             for scenario, m in (("base_costs", 1), ("costs_x3", 3))},
                "groups": [j[3] for j in jobs], "inputs_sha256": {str(j[0]): file_sha256(j[0]) for j in jobs},
                "data_sha256": file_sha256(data), "paper_benchmark": asdict(benchmark),
                "code_sha256": {str(p): file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                "protocol": "fixed ablations, same causal next-open engine/costs; no search or holdout evaluation",
                "model_horizon_limit": "post-open pilot was trained with current-open proxy; next-open replay delays the signal"}
    _json(output / "manifest.json", manifest)
    rows = []
    try:
        if workers == 1:
            for job in jobs:
                rows.extend(_evaluate(job))
                print(json.dumps({"completed": job[3]}, ensure_ascii=False), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_evaluate, j): j for j in jobs}
                for future in as_completed(futures):
                    rows.extend(future.result())
                    print(json.dumps({"completed": futures[future][3]}, ensure_ascii=False), flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        _json(output / "manifest.json", manifest)
        raise
    frame = pd.DataFrame(rows).sort_values(["family", "fold", "seed", "scenario", "variant"])
    frame.to_csv(output / "comparison.csv", index=False)
    keys = ["family", "scenario", "variant"]
    summary = frame.groupby(keys).agg(groups=("net_return", "size"),
        median_return=("net_return", "median"), min_return=("net_return", "min"), max_return=("net_return", "max"),
        median_sharpe=("sharpe", "median"), median_drawdown=("max_drawdown", "median"),
        median_exposure=("mean_gross_exposure", "median"), median_turnover=("turnover", "median"),
        median_fees=("fees", "median"), median_valid_ou=("valid_ou_fraction", "median"),
        median_entry_block=("long_entry_block_fraction", "median"), computed_tp_exits=("computed_tp_exits", "sum")).reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    baseline = frame.query("variant == 'without_rules'")[["family", "fold", "seed", "scenario", "net_return"]].rename(columns={"net_return": "baseline_return"})
    paired = frame.merge(baseline, on=["family", "fold", "seed", "scenario"], validate="many_to_one")
    paired["return_delta"] = paired.net_return-paired.baseline_return
    paired.to_csv(output / "paired.csv", index=False)
    manifest.update(status="complete", replay_count=len(frame))
    _json(output / "manifest.json", manifest)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families", nargs="+", default=["open_gap", "lagged"])
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    summary = validate_directory(args.signals_dir, args.data, args.output_dir, families=tuple(args.families), workers=args.workers)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
