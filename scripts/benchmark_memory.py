"""Compare memory/time and numerical outputs against a local Git revision.

Runs isolated, single-threaded workers on synthetic data; no market download.
Peak allocations exclude input construction and are measured separately from
runtime. tracemalloc includes NumPy buffers but not all native BLAS allocations.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
import types

import _bootstrap

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "trading_system.data.windows",
    "trading_system.data.scaling",
    "trading_system.models.manual_ann.manual_nn",
    "trading_system.models.manual_ann.sequence_adapter",
    "trading_system.experiments.config",
    "trading_system.experiments.runner",
    "trading_system.experiments.walkforward",
)
CASES = (
    "windows", "flat_windows", "scaler_fit", "scaler_transform", "predict",
    "fit", "pipeline", "multi_pipeline", "walkforward",
)


def load_reference(revision: str) -> None:
    """Load only the relevant local source modules, not data or Git worktrees."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    # Import current dependencies first, then replace the selected graph in
    # dependency order; never import a current module against a half-old graph.
    for name in MODULES:
        importlib.import_module(name)
    for name in MODULES:
        path = "src/" + name.replace(".", "/") + ".py"
        source = subprocess.check_output(
            ["git", "show", f"{commit}:{path}"], cwd=ROOT, text=True
        )
        module = types.ModuleType(name)
        module.__file__ = str(ROOT / path)
        module.__package__ = name.rpartition(".")[0]
        sys.modules[name] = module
        setattr(sys.modules[module.__package__], name.rpartition(".")[2], module)
        exec(compile(source, module.__file__, "exec"), module.__dict__)


def workload(case: str, rows: int):
    import numpy as np
    import pandas as pd
    from trading_system.data.scaling import SequenceStandardizer
    from trading_system.data.windows import build_context_dataset, build_sequence_dataset
    from trading_system.experiments.config import ExperimentConfig
    from trading_system.experiments.runner import run_experiment
    from trading_system.models.manual_ann.manual_nn import ManualANNClassifier, ManualANNConfig

    rng = np.random.default_rng(17)
    columns = [f"f{i}" for i in range(15)]
    frame = pd.DataFrame({name: rng.normal(size=rows).astype(np.float32) for name in columns})
    frame["Label_id"] = rng.integers(0, 3, size=rows)
    if case in ("windows", "flat_windows"):
        build = build_sequence_dataset if case == "windows" else build_context_dataset
        return lambda: {"windows": build(frame, columns, 20)[0]}
    if case.startswith("scaler"):
        X = rng.normal(size=(rows, 20, 15)).astype(np.float32)
        if case == "scaler_fit":
            def run_fit():
                scaler = SequenceStandardizer().fit(X)
                return scaler.state_dict()
            return run_fit
        scaler = SequenceStandardizer().fit(X)
        return lambda: {"transformed": scaler.transform(X)}
    settings = ManualANNConfig(hidden_size=64, epochs=3, batch_size=512, seed=9)
    if case in ("predict", "fit"):
        X = rng.normal(size=(rows, 300)).astype(np.float32)
        y = rng.integers(0, 3, size=rows)
        if case == "predict":
            model = ManualANNClassifier(settings)
            model.fit(X[:512], y[:512])
            return lambda: {"probabilities": model.predict_proba(X)}
        def run_fit():
            model = ManualANNClassifier(settings)
            result = model.fit(X, y, X_val=X[:1024], y_val=y[:1024])
            return {
                **model.state_dict(),
                "train_loss": np.asarray(result.history.train_loss),
                "val_loss": np.asarray(result.history.val_loss),
                "best_epoch": np.asarray(result.best_epoch),
                "probabilities": model.predict_proba(X[:1024]),
            }
        return run_fit
    x = np.arange(rows, dtype=float)
    close = 100.0 + 0.001 * x + 4 * np.sin(x / 5)
    market = pd.DataFrame({
        "date": pd.date_range("2000-01-01", periods=rows, freq="h"),
        "open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
        "close": close, "adj_close": close, "volume": 1e6 + 1e4 * np.cos(x / 7),
    })
    config = ExperimentConfig(
        context_len=20, label_mode="forward_return", forward_horizon=3,
        decision_mode="argmax", manual_ann=settings,
    )
    if case == "walkforward":
        from trading_system.experiments.walkforward import walk_forward_oracle_ann
        from trading_system.features.technical import (
            TECHNICAL_FEATURE_COLUMNS, compute_technical_features,
        )
        featured = compute_technical_features(market, group_col=None)

        def run_walkforward():
            result = walk_forward_oracle_ann(
                featured, TECHNICAL_FEATURE_COLUMNS,
                context_len=20, forward_horizon=3, walkforward_step=rows // 10,
                epochs=settings.epochs, hidden=settings.hidden_size,
                batch_size=settings.batch_size, seed=settings.seed,
            )
            return {
                "predictions": result["predictions"],
                "metrics": np.array(list(result["test_metrics"].values())),
                "backtest": np.array(list(result["benchmark_comparison"].values())),
                "best_epoch": np.array([log["best_epoch"] for log in result["retrain_logs"]]),
            }

        return run_walkforward
    if case == "multi_pipeline":
        from dataclasses import replace

        market = pd.concat([
            market.iloc[:rows // 4].assign(ticker=f"TICKER_{index}")
            for index in range(4)
        ], ignore_index=True)
        config = replace(config, universe="multi")

    def run_pipeline():
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_experiment(market, config)
        return {
            **result.bundle.estimator.estimator.state_dict(),
            "probabilities": result.test_probabilities,
            "predictions": result.test_predictions,
            "metrics": np.array(list(result.test_metrics.values())),
            "backtest": np.array(list(result.backtest.values())),
            "train_loss": np.asarray(result.bundle.fit_result.history.train_loss),
            "val_loss": np.asarray(result.bundle.fit_result.history.val_loss),
        }
    return run_pipeline


def worker(args):
    import numpy as np

    if args.reference:
        load_reference(args.reference)
    run = workload(args.worker, args.rows)
    run()  # Warm up imports/BLAS before measuring either memory or time.
    times = []
    for _ in range(args.repeat):
        gc.collect()
        start = time.perf_counter()
        output = run()
        times.append(time.perf_counter() - start)
        del output
    gc.collect()
    tracemalloc.start()
    output = run()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    np.savez(args.arrays, **output)
    print(json.dumps({
        "median_seconds": statistics.median(times),
        "peak_alloc_mib": peak / 1024**2,
        "numpy": np.__version__,
    }))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="HEAD")
    parser.add_argument("--rows", type=int, default=30_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--reference", help=argparse.SUPPRESS)
    parser.add_argument("--arrays", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rows < 1024 or args.repeat < 1:
        parser.error("rows must be >= 1024 and repeat must be positive")
    if args.worker:
        worker(args)
        return
    import numpy as np

    env = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"):
        env[name] = "1"
    reports = []
    with tempfile.TemporaryDirectory(prefix="ann-memory-") as temporary:
        for case in args.cases:
            row = {"case": case, "rows": args.rows, "baseline_ref": args.baseline_ref}
            for variant in ("baseline", "patched"):
                arrays = Path(temporary) / f"{variant}.npz"
                command = [
                    sys.executable, str(Path(__file__).resolve()), "--worker", case,
                    "--rows", str(args.rows), "--repeat", str(args.repeat),
                    "--arrays", str(arrays),
                ]
                if variant == "baseline":
                    command += ["--reference", args.baseline_ref]
                row[variant] = json.loads(subprocess.check_output(command, env=env, text=True))
            with np.load(Path(temporary) / "baseline.npz") as before, np.load(Path(temporary) / "patched.npz") as after:
                row["exact_outputs"] = all(np.array_equal(before[key], after[key]) for key in before.files)
                for key in before.files:
                    if key in ("predictions", "backtest", "metrics", "best_epoch"):
                        np.testing.assert_array_equal(after[key], before[key])
                    else:
                        np.testing.assert_allclose(after[key], before[key], rtol=1e-6, atol=1e-7)
            reports.append(row)
            print(json.dumps(row), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
