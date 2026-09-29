"""CLI replay/reporting over frozen signals, without fitting any model."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import TradingConfig
from .engine import run_trading_backtest
from .events import load_events


def read_table(path):
    path = Path(path)
    if path.suffix.lower() not in (".parquet", ".csv"):
        raise ValueError("Input table must be CSV or Parquet.")
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def save_result(result, directory):
    directory.mkdir()
    for name in ("equity", "positions", "orders", "trades", "decisions"):
        getattr(result, name).to_parquet(directory / f"{name}.parquet", index=False)
    _json(directory / "metrics.json", result.metrics)
    _json(directory / "metadata.json", result.metadata)


def replay_signals(signals, config, output, *, data=None, events=None, history=None,
                   final_test_start="2022-05-10", allow_final_test=False):
    """Compare rule-enabled/disabled paths, independently for each fold/seed.

    final_test_start is the repository's sealed holdout by default. Projects with
    other split boundaries must explicitly supply their own boundary.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    source = read_table(signals)
    if not {"date", "ticker"}.issubset(source):
        raise ValueError("Signals require date and ticker.")
    source["date"] = pd.to_datetime(source.date, utc=True, errors="raise")
    if source.empty or source[["date", "ticker"]].isna().any().any():
        raise ValueError("Signals must be non-empty with valid date/ticker keys.")
    boundary = pd.Timestamp(final_test_start)
    boundary = boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")
    if not allow_final_test and source.date.ge(boundary).any():
        raise ValueError("Signals enter the sealed final holdout. Use --allow-final-test only for an explicit final evaluation.")
    price_source = read_table(data) if data else None
    if price_source is not None:
        if not {"date", "ticker"}.issubset(price_source):
            raise ValueError("Price input requires date and ticker.")
        price_source["date"] = pd.to_datetime(price_source.date, utc=True, errors="raise")
        if price_source.duplicated(["date", "ticker"]).any():
            raise ValueError("Price input contains duplicate ticker/date rows.")
    calendar = load_events(events) if events else None
    history_source = read_table(history) if history else None
    if history_source is not None:
        history_source["date"] = pd.to_datetime(history_source.date, utc=True, errors="raise")
    group_columns = [name for name in ("source_candidate", "candidate", "fold", "seed") if name in source]
    groups = source.groupby(group_columns, sort=True, dropna=False) if group_columns else [((), source)]
    comparisons = []
    # Validate and compute all groups before creating outputs, so a bad later
    # group cannot leave a misleading completed comparison on disk.
    prepared = []
    for number, (key, group) in enumerate(groups):
        keys = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(group_columns, [x.item() if hasattr(x, "item") else x for x in keys]))
        if any(pd.isna(value) for value in identity.values()):
            raise ValueError("Replay group identifiers cannot be missing.")
        if group.duplicated(["date", "ticker"]).any():
            raise ValueError("Signals contain duplicate ticker/date rows within a replay group.")
        # Read only price rows matching these predictions; no holdout prices are
        # returned to the replay engine even when a source file spans all dates.
        bars = (group[["date", "ticker"]].merge(price_source, on=["date", "ticker"], how="left", validate="one_to_one")
                if price_source is not None else group.copy())
        past = (history_source.loc[(history_source.date < group.date.min()) & history_source.ticker.isin(group.ticker.unique())]
                if history_source is not None else None)
        off = run_trading_backtest(bars, group, replace(config, enabled=False), events=calendar, history=past)
        on = run_trading_backtest(bars, group, replace(config, enabled=True), events=calendar, history=past)
        deltas = {name: on.metrics[name] - off.metrics[name] if on.metrics[name] is not None and off.metrics[name] is not None else None
                  for name in ("net_return", "net_pnl", "sharpe", "sortino", "max_drawdown", "turnover", "fees")}
        directory = f"group-{number:03d}"
        comparisons.append({"group": identity, "directory": directory,
                            "without_rules": off.metrics, "with_rules": on.metrics, "delta": deltas})
        prepared.append((directory, off, on))
    output.mkdir(parents=True)
    for directory, off, on in prepared:
        folder = output / directory
        folder.mkdir()
        save_result(off, folder / "without_rules")
        save_result(on, folder / "with_rules")
    report = {"config": asdict(config), "comparisons": comparisons,
              "final_test_start": boundary.isoformat(),
              "final_holdout_opened": bool(source.date.ge(boundary).any()),
              "inputs_sha256": {"signals": file_sha256(signals),
                                  "data": file_sha256(data) if data else None,
                                  "history": file_sha256(history) if history else None,
                                  "events": file_sha256(events) if events else None},
              "comparison_contract": "same frozen predictions, timing, costs, price basis and accounting; only rules differ"}
    _json(output / "report.json", report)
    rows = []
    for comparison in comparisons:
        row = {**comparison["group"], "directory": comparison["directory"]}
        for arm in ("without_rules", "with_rules", "delta"):
            row.update({f"{arm}_{k}": v for k, v in comparison[arm].items() if not isinstance(v, dict)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="OHLC/actions/sector table; defaults to signals table.")
    parser.add_argument("--trading-config", type=Path, required=True, help="JSON TradingConfig fields.")
    parser.add_argument("--events", type=Path)
    parser.add_argument("--ou-history", type=Path, help="Price history; only rows before each group's first prediction reach the OU fit.")
    parser.add_argument("--execution", choices=("next_open", "next_close", "open_proxy"))
    parser.add_argument("--signal-timing", choices=("after_open", "after_close"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-test-start", default="2022-05-10")
    parser.add_argument("--allow-final-test", action="store_true")
    args = parser.parse_args(argv)
    values = json.loads(args.trading_config.read_text())
    for name in ("execution", "signal_timing"):
        if getattr(args, name) is not None:
            values[name] = getattr(args, name)
    config = TradingConfig(**values)
    report = replay_signals(args.signals, config, args.output_dir, data=args.data,
                            events=args.events, history=args.ou_history, final_test_start=args.final_test_start,
                            allow_final_test=args.allow_final_test)
    print(json.dumps({"output": str(args.output_dir.resolve()), "groups": len(report["comparisons"]),
                      "final_holdout_opened": report["final_holdout_opened"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
