# %% [markdown]
# # Model benchmark comparison
#
# Load the latest saved benchmark, compare predictive and trading metrics,
# inspect seed stability, and identify the strongest model without hiding failures.

# %%
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 220)
plt.style.use("seaborn-v0_8-whitegrid")

# Set a specific result directory here. Leave as None to use the latest one.
RESULT_DIR: Path | None = None


# %%
def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Project root not found.")


def latest_complete_benchmark(root: Path) -> Path:
    comparison_root = root / "artifacts" / "comparisons"
    candidates = [
        path
        for path in comparison_root.iterdir()
        if path.is_dir()
        and (path / "runs.csv").exists()
        and (path / "summary.csv").exists()
        and (path / "report.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No complete benchmark found under {comparison_root}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


PROJECT_ROOT = find_project_root(Path.cwd().resolve())
BENCHMARK_DIR = (
    RESULT_DIR.expanduser().resolve()
    if RESULT_DIR is not None
    else latest_complete_benchmark(PROJECT_ROOT)
)

runs = pd.read_csv(BENCHMARK_DIR / "runs.csv")
summary_raw = pd.read_csv(BENCHMARK_DIR / "summary.csv")
failures = read_csv_or_empty(BENCHMARK_DIR / "failures.csv")
report = json.loads((BENCHMARK_DIR / "report.json").read_text(encoding="utf-8"))

print(f"Benchmark: {BENCHMARK_DIR}")
display(pd.Series(report.get("metadata", {}), name="value").to_frame())
display(runs.groupby(["model_name", "status"]).size().rename("runs").unstack(fill_value=0))

if not failures.empty:
    display(failures[["model_name", "seed", "error_type", "error"]])


# %% [markdown]
# ## Perfect-label reference versus buy & hold
#
# This diagnostic assumes 100% correct reproduction of the configured labels on
# the test split, with the same execution delay, fees, and position mode as the
# models. It measures the economic behaviour of the labels, not a market oracle
# or the maximum theoretically achievable PnL.

# %%
successful = runs[runs["status"] == "ok"].copy()
if successful.empty:
    raise RuntimeError("Every benchmark run failed; inspect failures.csv.")

artifact_paths = successful.get("artifact_path", pd.Series(dtype="object")).dropna()
if artifact_paths.empty:
    raise FileNotFoundError(
        "Perfect-label analysis requires run artifacts; rerun without --no-run-artifacts."
    )

manifest = json.loads(
    (Path(artifact_paths.iloc[0]) / "manifest.json").read_text(encoding="utf-8")
)
experiment_config = manifest["experiment_parameters"]["config"]
dataset_metadata = manifest["experiment_parameters"]["dataset"]

src_path = str(PROJECT_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from trading_system.backtest.engine import evaluate_strategy_vs_buy_hold
from trading_system.data.splits import chronological_train_val_test_split
from trading_system.labels.breakout import (
    generate_breakout_labels,
    generate_breakout_labels_by_ticker,
)
from trading_system.labels.forward_return import build_forward_return_labels


def build_test_labels(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    group_col = config["group_col"] if config["universe"] == "multi" else None
    raw_splits = chronological_train_val_test_split(
        frame,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        group_col=group_col,
        date_col=config["date_col"],
    )
    parts = []
    for name, split in zip(("train", "val", "test"), raw_splits):
        split = split.copy()
        split["_experiment_split"] = name
        parts.append(split)
    source = pd.concat(parts, ignore_index=True)

    if config["label_mode"] in ("breakout", "oracle_train_only"):
        if group_col is None:
            labeled = generate_breakout_labels(
                source,
                config["label_window"],
                price_col=config["price_col"],
                date_col=config["date_col"],
            )
        else:
            labeled = generate_breakout_labels_by_ticker(
                source,
                config["label_window"],
                price_col=config["price_col"],
                group_col=group_col,
                date_col=config["date_col"],
            )
    elif config["label_mode"] == "forward_return":
        groups = (
            source.groupby(group_col, sort=False, dropna=False)
            if group_col is not None
            else [(None, source)]
        )
        labeled = pd.concat(
            [
                build_forward_return_labels(
                    group,
                    price_col=config["price_col"],
                    horizon=config["forward_horizon"],
                    buy_threshold=config["forward_buy_threshold"],
                    sell_threshold=config["forward_sell_threshold"],
                    date_col=config["date_col"],
                )[0]
                for _, group in groups
            ],
            ignore_index=True,
        )
    else:
        raise ValueError(
            f"Perfect-label diagnostic does not support {config['label_mode']!r}."
        )

    sort_columns = (
        [group_col, config["date_col"]]
        if group_col is not None
        else [config["date_col"]]
    )
    return (
        labeled[labeled["_experiment_split"] == "test"]
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )


dataset_path = Path(dataset_metadata["path"])
market = pd.read_parquet(dataset_path)
selected_tickers = dataset_metadata.get("tickers", [])
if selected_tickers and experiment_config["group_col"] in market:
    market = market[
        market[experiment_config["group_col"]].isin(selected_tickers)
    ].copy()

perfect_test = build_test_labels(market, experiment_config)
perfect_label_metrics = evaluate_strategy_vs_buy_hold(
    perfect_test,
    perfect_test["Label_id"].to_numpy(dtype=np.int64),
    initial_capital=experiment_config["initial_capital"],
    price_col=experiment_config["price_col"],
    fee_per_trade=experiment_config["fee_per_trade"],
    position_mode=experiment_config["position_mode"],
    execution_delay=experiment_config["execution_delay"],
    group_col=(
        experiment_config["group_col"]
        if experiment_config["universe"] == "multi"
        else None
    ),
    date_col=experiment_config["date_col"],
)

perfect_label_pnl = float(perfect_label_metrics["model_pnl"])
buy_hold_pnl = float(perfect_label_metrics["buy_hold_pnl"])
perfect_label_outperformance = float(perfect_label_metrics["outperformance"])

label_reference = pd.DataFrame(
    [
        {
            "strategy": "Buy & hold",
            "pnl": buy_hold_pnl,
            "outperformance_vs_buy_hold": 0.0,
        },
        {
            "strategy": "Perfect configured labels",
            "pnl": perfect_label_pnl,
            "outperformance_vs_buy_hold": perfect_label_outperformance,
        },
    ]
).set_index("strategy")

print(f"Label mode: {experiment_config['label_mode']}")
display(perfect_test["Label"].value_counts().rename("test_labels").to_frame())
display(label_reference.round(2))


# %% [markdown]
# ## Aggregated ranking
#
# Ranking uses mean test outperformance across seeds. Predictive metrics remain
# secondary: high classification accuracy alone does not imply profitable trading.

# %%
METRICS = [
    "backtest_outperformance",
    "backtest_model_pnl",
    "backtest_model_return",
    "backtest_sharpe_ratio",
    "backtest_sortino_ratio",
    "backtest_max_drawdown",
    "test_macro_f1",
    "test_bal_acc",
    "duration_seconds",
]
missing = [column for column in METRICS if column not in successful]
if missing:
    raise KeyError(f"Benchmark is missing expected metrics: {missing}")

comparison = successful.groupby("model_name")[METRICS].agg(["mean", "std"])
comparison.columns = [f"{metric}_{stat}" for metric, stat in comparison.columns]
robustness = successful.groupby("model_name").agg(
    successful_runs=("status", "size"),
    profitable_seed_rate=("backtest_model_pnl", lambda values: (values > 0).mean()),
    outperforming_seed_rate=("backtest_outperformance", lambda values: (values > 0).mean()),
)
comparison = comparison.join(robustness).sort_values(
    ["backtest_outperformance_mean", "backtest_sharpe_ratio_mean"],
    ascending=False,
)

DISPLAY_COLUMNS = [
    "backtest_outperformance_mean",
    "backtest_outperformance_std",
    "backtest_model_pnl_mean",
    "backtest_sharpe_ratio_mean",
    "backtest_max_drawdown_mean",
    "test_macro_f1_mean",
    "test_bal_acc_mean",
    "profitable_seed_rate",
    "outperforming_seed_rate",
    "duration_seconds_mean",
]
display(comparison[DISPLAY_COLUMNS].round(4))


# %% [markdown]
# ## PnL and outperformance stability

# %%
models = comparison.index.tolist()
pnl = successful.groupby("model_name")["backtest_model_pnl"].agg(["mean", "std"]).reindex(models)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(models, pnl["mean"], yerr=pnl["std"], capsize=4, color="#3B82F6")
axes[0].axhline(buy_hold_pnl, color="#DC2626", linestyle="--", label="Buy & hold")
axes[0].axhline(
    perfect_label_pnl,
    color="#16A34A",
    linestyle=":",
    linewidth=2,
    label="Perfect labels",
)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set(title="Test PnL: mean ± std across seeds", ylabel="PnL")
axes[0].tick_params(axis="x", rotation=30)
axes[0].legend()

seed_matrix = successful.pivot(index="seed", columns="model_name", values="backtest_outperformance")
seed_matrix = seed_matrix.reindex(columns=models)
seed_matrix.plot(marker="o", ax=axes[1])
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set(title="Outperformance by seed", ylabel="Model PnL − buy & hold PnL")
axes[1].legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")

fig.tight_layout()
plt.show()


# %% [markdown]
# ## Predictive score versus trading result

# %%
points = successful.groupby("model_name")[["test_macro_f1", "backtest_outperformance"]].mean()
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(points["test_macro_f1"], points["backtest_outperformance"], s=80)
for model_name, row in points.iterrows():
    ax.annotate(model_name, (row["test_macro_f1"], row["backtest_outperformance"]), xytext=(5, 5), textcoords="offset points")
ax.axhline(0, color="black", linewidth=0.8)
ax.set(
    title="Classification quality does not guarantee trading performance",
    xlabel="Test macro-F1",
    ylabel="Mean test outperformance",
)
fig.tight_layout()
plt.show()


# %% [markdown]
# ## Automatic conclusion

# %%
leader_name = comparison.index[0]
leader = comparison.iloc[0]
leader_outperformance = float(leader["backtest_outperformance_mean"])
leader_stability = float(leader["outperforming_seed_rate"])

print(f"Leader by mean test outperformance: {leader_name}")
print(f"Mean outperformance: {leader_outperformance:,.2f}")
print(f"Positive outperformance seeds: {leader_stability:.0%}")
print(f"Perfect-label outperformance: {perfect_label_outperformance:,.2f}")

if perfect_label_outperformance <= 0:
    print(
        "Label diagnosis: perfectly reproduced labels underperform buy & hold; "
        "the labeling objective is economically misaligned on this test period."
    )
elif leader_outperformance <= 0:
    print(
        "Label diagnosis: perfect labels beat buy & hold, but trained models do not; "
        "the main gap is learning or decision calibration."
    )

if leader_outperformance <= 0:
    print("Conclusion: no model beats buy & hold on average. Do not declare a winner.")
elif leader_stability < 0.60:
    print("Conclusion: apparent winner, but unstable across seeds.")
else:
    print("Conclusion: promising winner; confirm with walk-forward and untouched data.")
