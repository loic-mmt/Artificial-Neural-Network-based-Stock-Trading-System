"""Plot retrospective labels, not predictions or executed positions."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from trading_system.data.io import read_parquet_dataset
from trading_system.labels.config import LabelConfig, normalize_label_method
from trading_system.labels.registry import LabelContext, create_default_label_registry

LABEL_COLORS = {0: "#d62728", 1: "#888888", 2: "#2ca02c"}
UNKNOWN_COLOR = "#e69f00"
DEFAULT_LABEL_PARAMETERS = {
    "max_holding": 10,
    "volatility_window": 20,
    "volatility_estimator": "atr",
    "profit_barrier": 0.75,
    "stop_barrier": 0.75,
    "event_filter": "cusum",
    "cusum_threshold": 0.5,
    "cost_bps": 5.0,
    "between_event_policy": "hold",
}
LABEL_FACTORIES = {
    "breakout": LabelConfig.breakout,
    "forward_return": LabelConfig.forward_return,
    "volatility_position": LabelConfig.volatility_position,
    "triple_barrier": LabelConfig.triple_barrier,
    "intraday_return": LabelConfig.intraday_return,
}
DEFAULT_METHOD_PARAMETERS = {
    "breakout": LabelConfig.breakout().parameters,
    "forward_return": LabelConfig.forward_return(horizon=10).parameters,
    "volatility_position": LabelConfig.volatility_position(position_mode="long_short").parameters,
    "triple_barrier": DEFAULT_LABEL_PARAMETERS,
    "intraday_return": {},
}
METHOD_ARGUMENTS = {
    "breakout": {
        "label_window": "window", "label_buy_buffer": "buy_buffer",
        "label_sell_buffer": "sell_buffer", "label_alternating": "alternating",
    },
    "forward_return": {
        "label_horizon": "horizon", "label_buy_threshold": "buy_threshold",
        "label_sell_threshold": "sell_threshold",
    },
    "volatility_position": {
        "label_horizon": "horizon", "label_vol_window": "volatility_window",
        "label_long_threshold": "long_threshold", "label_short_threshold": "short_threshold",
        "label_exit_threshold": "exit_threshold", "label_min_holding": "min_holding_period",
        "label_cooldown": "cooldown", "label_cost_bps": "cost_bps",
        "label_position_mode": "position_mode",
    },
    "triple_barrier": {
        "label_horizon": "max_holding", "label_vol_window": "volatility_window",
        "label_volatility_estimator": "volatility_estimator",
        "label_profit_barrier": "profit_barrier", "label_stop_barrier": "stop_barrier",
        "label_event_filter": "event_filter", "label_cusum_threshold": "cusum_threshold",
        "label_cost_bps": "cost_bps", "label_between_events": "between_event_policy",
    },
    "intraday_return": {},
}


def label_config(label_method: str, **parameters: object) -> LabelConfig:
    """Use the analysis benchmark defaults and reject parameters of other methods."""
    method = normalize_label_method(label_method)
    if method not in LABEL_FACTORIES:
        raise ValueError(f"Unknown label method {method!r}; choose from {tuple(LABEL_FACTORIES)}.")
    defaults = DEFAULT_METHOD_PARAMETERS[method]
    extras = set(parameters) - set(defaults)
    if extras:
        raise ValueError(f"{method} labels do not accept parameters: {sorted(extras)}.")
    return LABEL_FACTORIES[method](**(defaults | parameters))


def prepare_labels(
    frame: pd.DataFrame,
    ticker: str,
    *,
    start: str | None = None,
    end: str | None = None,
    price_col: str = "adj_close",
    label_method: str = "triple_barrier",
    **label_parameters: object,
) -> pd.DataFrame:
    """Label the complete ticker history before cropping the displayed interval."""
    work = frame.loc[frame["ticker"].eq(ticker)].copy()
    if work.empty:
        raise ValueError(f"No rows found for ticker {ticker!r}.")
    work["date"] = pd.to_datetime(work["date"], utc=True, errors="raise")
    if work["date"].isna().any() or work["date"].duplicated().any():
        raise ValueError("Dates must be non-missing and unique for the ticker.")
    work = work.sort_values("date").reset_index(drop=True)
    method = normalize_label_method(label_method)
    config = label_config(method, **label_parameters)
    result = create_default_label_registry().generate(
        work, config, LabelContext(price_col=price_col, split_col=None)
    )
    labeled = result.frame
    labeled["_label_known"] = result.known_mask
    labeled.attrs.update(
        label_method=method, label_class_names=result.class_names,
        label_semantics=result.semantics,
    )
    lower = pd.to_datetime(start, utc=True) if start is not None else None
    upper = pd.to_datetime(end, utc=True) if end is not None else None
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("--start must not be later than --end.")
    if lower is not None:
        labeled = labeled.loc[labeled["date"] >= lower]
    if upper is not None:
        labeled = labeled.loc[labeled["date"] <= upper]
    if len(labeled) < 2:
        raise ValueError("At least two sessions are required in the displayed interval.")
    return labeled.reset_index(drop=True)


def build_figure(labeled: pd.DataFrame, ticker: str, price_col: str = "adj_close"):
    """Color each segment J to J+1 with the label at J; never carry actions forward."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    x = mdates.date2num(labeled["date"].to_numpy())
    points = np.column_stack((x, labeled[price_col].to_numpy(dtype=float)))
    segments = np.stack((points[:-1], points[1:]), axis=1)
    colors = [
        LABEL_COLORS[int(label)] if known else UNKNOWN_COLOR
        for label, known in zip(labeled["Label_id"].iloc[:-1], labeled["_label_known"].iloc[:-1])
    ]
    fig, ax = plt.subplots(figsize=(14, 5), layout="constrained")
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=2))
    ax.autoscale_view()
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    method = labeled.attrs.get("label_method", "triple_barrier")
    class_names = labeled.attrs.get("label_class_names", ("Sell", "Hold", "Buy"))
    title = str(method).replace("_", " ").capitalize()
    ax.set(title=f"{ticker} | {title} | Retrospective labels", ylabel=price_col)
    ax.grid(alpha=0.2)
    ax.legend(handles=[
        Line2D([], [], color=color, linewidth=2, label=name)
        for name, color in (
            (class_names[2], LABEL_COLORS[2]), (class_names[1], LABEL_COLORS[1]),
            (class_names[0], LABEL_COLORS[0]), ("Unknown", UNKNOWN_COLOR),
        )
    ])
    caption = (
        f"Open J to close J targets, displayed on visual J to J+1 {price_col} segments. "
        "Not overnight returns or executed positions."
        if method == "intraday_return" else
        "Segment J to J+1: label at J. Not an executed or held position."
    )
    fig.supxlabel(caption, fontsize=9)
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Parquet file or directory.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", help="First displayed session, YYYY-MM-DD.")
    parser.add_argument("--end", help="Last displayed session, YYYY-MM-DD (inclusive).")
    parser.add_argument("--price-col", default="adj_close")
    parser.add_argument("--label-method", type=normalize_label_method, choices=tuple(LABEL_FACTORIES),
                        default="triple_barrier", help="Hyphenated names also accepted. Default: triple-barrier.")
    parser.add_argument("--output", type=Path, help="Defaults to artifacts/plots/<ticker>[-<method>]-labels.png.")
    parser.add_argument("--no-show", action="store_true", help="Save without opening a window.")
    parser.add_argument("--label-horizon", "--label-max-holding", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--label-vol-window", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--label-volatility-estimator", choices=("rolling_std", "atr", "bollinger"), default=argparse.SUPPRESS)
    parser.add_argument("--label-profit-barrier", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--label-stop-barrier", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--label-event-filter", choices=("all", "cusum"), default=argparse.SUPPRESS)
    parser.add_argument("--label-cusum-threshold", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--label-cost-bps", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--label-window", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--label-buy-buffer", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--label-sell-buffer", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--label-alternating", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--label-buy-threshold", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--label-sell-threshold", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--label-long-threshold", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--label-short-threshold", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--label-exit-threshold", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--label-min-holding", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--label-cooldown", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--label-position-mode", choices=("long_flat", "long_short"), default=argparse.SUPPRESS)
    parser.add_argument("--label-between-events", choices=("hold", "flat", "carry"), default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    method = normalize_label_method(args.label_method)
    destinations = METHOD_ARGUMENTS[method]
    label_options = set().union(*(set(options) for options in METHOD_ARGUMENTS.values()))
    irrelevant = (set(vars(args)) & label_options) - set(destinations)
    if irrelevant:
        flags = ["--" + name.replace("_", "-") for name in sorted(irrelevant)]
        parser.error(f"{method} does not accept options: {', '.join(flags)}")
    parameters = {
        target: vars(args)[source] for source, target in destinations.items() if source in vars(args)
    }
    config = label_config(method, **parameters)

    # Project only the required columns and filter the ticker at Parquet read time.
    columns = ["date", "ticker", args.price_col]
    if method == "intraday_return":
        columns += ["open", "close"]
    elif method == "triple_barrier" and config.parameters["volatility_estimator"] == "atr":
        columns += ["high", "low", "close"]
    frame = read_parquet_dataset(
        args.data, columns=list(dict.fromkeys(columns)),
        filter_expr=ds.field("ticker") == args.ticker,
    )
    try:
        labeled = prepare_labels(
            frame, args.ticker, start=args.start, end=args.end, price_col=args.price_col,
            label_method=method, **parameters,
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = build_figure(labeled, args.ticker, args.price_col)
    safe_ticker = args.ticker.replace("/", "_").replace("\\", "_")
    suffix = "" if method == "triple_barrier" else "-" + method.replace("_", "-")
    output = args.output or Path("artifacts/plots") / f"{safe_ticker}{suffix}-labels.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    names = labeled["Label"].where(labeled["_label_known"], "Unknown")
    print(f"labels={names.value_counts().to_dict()} saved={output.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
