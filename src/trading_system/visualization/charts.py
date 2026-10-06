"""Pure Plotly views over saved run data; no training or UI side effects."""

from __future__ import annotations

import hashlib
import html
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .theme import PALETTE, SYSTEM_FONT


_COLORS = (PALETTE["blue"], PALETTE["blue-dark"], "#3973df", "#547dbb", PALETTE["ink"], "#30496e")
_AGGREGATE_SUFFIXES = ("_mean", "_std", "_median", "_min", "_max", "_sem")
_FAILURE_STATUSES = ("error", "failed", "failure")
_PERCENT_METRICS = frozenset({
    "net_return", "gross_return", "model_return", "buy_hold_return", "total_return",
    "cumulative_return", "outperformance_return", "backtest_model_return", "backtest_buy_hold_return",
    "observed_return", "drawdown", "max_drawdown", "window_drawdown", "backtest_max_drawdown",
    "net_max_drawdown",
})


def _color(key: str) -> str:
    return _COLORS[int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % len(_COLORS)]


def _figure(title: str, *, y_title: str | None = None) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        template="plotly_white", title=dict(text=title, font=dict(size=16), x=0), height=420,
        margin=dict(l=55, r=20, t=48, b=45),
        font=dict(family=SYSTEM_FONT, size=12, color=PALETTE["ink"]),
        paper_bgcolor=PALETTE["white"], plot_bgcolor=PALETTE["white"],
        colorway=list(_COLORS), hovermode="x unified", dragmode="zoom",
        legend=dict(orientation="h", y=-0.2, font=dict(size=11)), yaxis_title=y_title,
        hoverlabel=dict(bgcolor=PALETTE["white"], bordercolor=PALETTE["line"], font=dict(size=12)),
        uirevision=title, transition=dict(duration=0),
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor=PALETTE["line"], zerolinecolor=PALETTE["line"])
    return figure


def _empty(figure: go.Figure, message: str) -> go.Figure:
    if not figure.data:
        figure.add_annotation(
            text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
            font=dict(color=PALETTE["muted"], size=13),
        )
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
    return figure


def _value(data: Any, key: str, default: Any = None) -> Any:
    return data.get(key, default) if isinstance(data, Mapping) else getattr(data, key, default)


def _identity(data: Any) -> tuple[str, str]:
    record = _value(data, "record", {})
    run_id = str(_value(record, "run_id", "run"))
    return run_id, str(_value(record, "label", run_id))


def _run_styles(runs: Sequence[Any]) -> dict[str, tuple[str, str]]:
    """Give selected identities distinct styles, independent of selection order."""
    identities = sorted({_identity(data)[0] for data in runs})
    if len(identities) == 1:
        return {identities[0]: (PALETTE["blue"], "solid")}
    result: dict[str, tuple[str, str]] = {}
    used: set[tuple[str, str]] = set()
    for run_id in identities:
        preferred = int(hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8], 16) % len(_COLORS)
        candidates = [(_COLORS[(preferred + offset) % len(_COLORS)], dash)
                      for dash in ("solid", "dot", "dashdot", "longdash") for offset in range(len(_COLORS))]
        style = next((candidate for candidate in candidates if candidate not in used), candidates[0])
        used.add(style)
        result[run_id] = style
    return result


def _legend_labels(runs: Sequence[Any]) -> dict[str, str]:
    labels = {run_id: label if len(label) <= 32 else label[:21] + "…" + label[-10:]
              for run_id, label in (_identity(data) for data in runs)}
    originals = list(labels.values())
    for ordinal, run_id in enumerate(sorted(labels), 1):
        label = labels[run_id]
        if originals.count(label) > 1:
            labels[run_id] = label[:26] + f" · {ordinal}"
    return labels


def _table(data: Any, name: str) -> pd.DataFrame:
    frame = _value(data, name)
    return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _dated(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    date_column = next((name for name in ("date", "timestamp", "time") if name in frame), None)
    if date_column is not None:
        dates = pd.to_datetime(frame[date_column], utc=True, errors="coerce")
    elif isinstance(frame.index, pd.DatetimeIndex):
        dates = pd.Series(pd.to_datetime(frame.index, utc=True), index=frame.index)
    else:
        return pd.DataFrame()
    result = frame.copy()
    result["date"] = np.asarray(dates)
    result["date"] = pd.to_datetime(result["date"], utc=True, errors="coerce")
    return result.dropna(subset=["date"]).sort_values("date", kind="stable").reset_index(drop=True)


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)


def _sample(values: np.ndarray, max_points: int) -> np.ndarray:
    """Select display points only, retaining endpoints and bucket extrema."""
    if isinstance(max_points, bool) or not isinstance(max_points, (int, np.integer)) or max_points < 2:
        raise ValueError("max_points must be an integer of at least 2.")
    size = len(values)
    if size <= max_points:
        return np.arange(size)
    selected = {0, size - 1}
    if max_points == 3:
        finite = np.flatnonzero(np.isfinite(values[1:-1])) + 1
        if len(finite):
            baseline = np.linspace(values[0], values[-1], size)
            distance = np.abs(values[finite] - baseline[finite])
            selected.add(int(finite[np.nanargmax(distance)]) if np.isfinite(distance).any() else int(finite[0]))
    elif max_points > 3:
        for bucket in np.array_split(np.arange(1, size - 1), (max_points - 2) // 2):
            finite = bucket[np.isfinite(values[bucket])]
            if len(finite):
                selected.add(int(finite[np.argmin(values[finite])]))
                selected.add(int(finite[np.argmax(values[finite])]))
            else:
                selected.add(int(bucket[0]))
    return np.asarray(sorted(selected), dtype=int)


def _line(figure: go.Figure, frame: pd.DataFrame, values: np.ndarray, *, name: str,
          color: str, max_points: int, dash: str = "solid", suffix: str = "",
          original: np.ndarray | None = None, fill: str | None = None, group: str | None = None) -> None:
    if not len(values) or not np.isfinite(values).any():
        return
    sample = _sample(values, max_points)
    hover = "%{x|%Y-%m-%d %H:%M UTC}<br>%{y:,.4f}" + suffix
    if original is not None:
        hover += "<br>Capital observé : %{customdata:,.4f}"
    figure.add_trace(go.Scatter(
        x=[stamp.to_pydatetime() for stamp in frame["date"].iloc[sample]], y=values[sample], name=name, mode="lines",
        line=dict(color=color, width=2, dash=dash), fill=fill,
        fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.09)" if fill else None,
        connectgaps=False,
        customdata=original[sample] if original is not None else None,
        legendgroup=group,
        hovertemplate=hover + "<extra>%{fullData.name}</extra>",
    ))


def equity_figure(runs: Sequence[Any], *, normalize: bool = True, max_points: int = 3000) -> go.Figure:
    figure = _figure("Capital et référence", y_title="Base 100" if normalize else "Capital observé")
    styles = _run_styles(runs)
    for data in runs:
        frame = _dated(_table(data, "equity"))
        run_id, label = _identity(data)
        strategy_color, strategy_dash = styles[run_id]
        baseline_name = "Buy & Hold" if len(runs) == 1 else f"Buy & Hold · {label}"
        for column, name, dash in (("equity", label, strategy_dash), ("benchmark_equity", baseline_name, "dash")):
            if frame.empty or column not in frame:
                continue
            observed = _numeric(frame, column)
            finite = np.flatnonzero(np.isfinite(observed))
            if not len(finite):
                continue
            if normalize:
                initial = observed[finite[0]]
                if initial <= 0:
                    continue
                values = observed / initial * 100.0
            else:
                values = observed
            _line(figure, frame, values, name=name,
                  color=PALETTE["muted"] if column == "benchmark_equity" else strategy_color, max_points=max_points,
                  dash=dash, original=observed if normalize else None,
                  group=f"reference:{run_id}" if column == "benchmark_equity" else run_id)
    return _empty(figure, "Courbes de capital absentes ou non normalisables.")


def drawdown_figure(runs: Sequence[Any], *, max_points: int = 3000) -> go.Figure:
    figure = _figure("Drawdown stratégie", y_title="Drawdown (%)")
    styles = _run_styles(runs)
    for data in runs:
        frame = _dated(_table(data, "equity"))
        if frame.empty:
            continue
        if "drawdown" in frame:
            values = _numeric(frame, "drawdown")
        elif "equity" in frame:
            equity = pd.Series(_numeric(frame, "equity"))
            peak = equity.cummax()
            values = ((equity / peak.where(peak > 0)) - 1.0).to_numpy()
        else:
            continue
        run_id, label = _identity(data)
        color, dash = styles[run_id]
        _line(figure, frame, values * 100.0, name=label, color=color, dash=dash,
              max_points=max_points, suffix=" %", fill="tozeroy", group=run_id)
    return _empty(figure, "Drawdown absent ; capital requis pour le calculer.")


def capital_drawdown_figure(runs: Sequence[Any], *, normalize: bool = True, max_points: int = 3000) -> go.Figure:
    """Keep capital, reference and strategy drawdown on the same date axis."""
    capital = equity_figure(runs, normalize=normalize, max_points=max_points)
    drawdown = drawdown_figure(runs, max_points=max_points)
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06)
    figure.update_layout(_figure("Capital et drawdown").layout)
    legend_rows = (len(capital.data) + 1) // 2
    bottom_margin = max(55, 34 + legend_rows * 24)
    labels = _legend_labels(runs)
    figure.update_layout(
        height=550 + max(0, bottom_margin - 60), margin=dict(b=bottom_margin),
        legend=dict(orientation="h", y=-0.07, yanchor="top", x=0, entrywidth=.5, entrywidthmode="fraction"),
    )
    for trace in capital.data:
        trace.meta = trace.name
        group = str(trace.legendgroup)
        reference = group.startswith("reference:")
        run_id = group.removeprefix("reference:") if reference else group
        trace.name = ("Buy & Hold" if len(runs) == 1 else f"B&H · {labels[run_id]}") if reference else labels[run_id]
        trace.hovertemplate = trace.hovertemplate.replace("%{fullData.name}", "%{meta}")
        figure.add_trace(trace, row=1, col=1)
    for trace in drawdown.data:
        trace.meta = trace.name
        trace.name = labels[str(trace.legendgroup)]
        trace.hovertemplate = trace.hovertemplate.replace("%{fullData.name}", "%{meta}")
        trace.showlegend = False
        figure.add_trace(trace, row=2, col=1)
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor=PALETTE["line"], zerolinecolor=PALETTE["line"])
    figure.update_yaxes(title="Base 100" if normalize else "Capital observé", row=1, col=1)
    figure.update_yaxes(title="Drawdown (%)", row=2, col=1)
    if not capital.data and drawdown.data:
        figure.add_annotation(text="Capital absent", x=.5, y=.75, xref="paper", yref="paper", showarrow=False)
    elif capital.data and not drawdown.data:
        figure.add_annotation(text="Drawdown absent", x=.5, y=.15, xref="paper", yref="paper", showarrow=False)
    return _empty(figure, "Courbes de capital et drawdown absentes.")


def kpi_bar_figure(frame: pd.DataFrame, *, metric: str) -> go.Figure:
    """Compare each recorded run, retaining the metric's exact definition."""
    percent = metric in _PERCENT_METRICS
    figure = _figure(metric, y_title=None)
    figure.update_xaxes(title=f"{metric} (%)" if percent else metric)
    observations = _observations(frame, metric)
    if observations.empty:
        return _empty(figure, "Métrique enregistrée indisponible.")
    labels = (observations["label"] if "label" in observations else observations.get("run_id", observations.index.to_series())).astype(str).copy()
    repeated = labels.duplicated(keep=False)
    if repeated.any():
        identities = observations.get("run_id", observations.index.to_series()).astype(str)
        labels.loc[repeated] = labels.loc[repeated] + " · " + identities.loc[repeated]
    full_labels = labels.copy()
    labels = labels.map(lambda value: value if len(value) <= 22 else value[:13] + "…" + value[-8:])
    repeated_short = labels.duplicated(keep=False)
    for ordinal, index in enumerate(labels[repeated_short].index, 1):
        labels.loc[index] = labels.loc[index][:17] + f" · {ordinal}"
    values = observations[metric].to_numpy(dtype=float) * (100.0 if percent else 1.0)
    fields = [name for name in ("run_id", "model", "seed", "fold", "partition", "status") if name in observations]
    suffix = " %" if percent else ""
    hover = html.escape(metric) + " : %{x:+,.4f}" + suffix
    for index, field in enumerate(fields, 1):
        hover += f"<br>{html.escape(field)} : %{{customdata[{index}]}}"
    details = np.column_stack([full_labels.astype(str).to_numpy(), observations[fields].astype(str).to_numpy()])
    figure.add_trace(go.Bar(
        x=values, y=labels, orientation="h", showlegend=False,
        marker=dict(color=[PALETTE["sell-ink"] if value < 0 else PALETTE["blue"] for value in values]),
        text=[f"{value:+,.2f}{suffix}" for value in values], textposition="auto",
        customdata=details,
        hovertemplate=hover + "<extra>%{customdata[0]}</extra>",
    ))
    figure.update_layout(height=max(260, min(800, 110 + len(observations) * 34)), hovermode="closest", margin=dict(l=20, r=20))
    figure.update_yaxes(autorange="reversed", showgrid=False, automargin=True, tickfont=dict(size=11))
    return figure


def _window(frame: pd.DataFrame, start: Any, end: Any) -> pd.DataFrame:
    if frame.empty:
        return frame
    keep = np.ones(len(frame), dtype=bool)
    for boundary, is_start in ((start, True), (end, False)):
        if boundary is None:
            continue
        timestamp = pd.Timestamp(boundary)
        timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
        if is_start:
            keep &= np.asarray(frame["date"] >= timestamp)
        elif timestamp == timestamp.normalize():
            keep &= np.asarray(frame["date"] < timestamp + pd.Timedelta(days=1))
        else:
            keep &= np.asarray(frame["date"] <= timestamp)
    return frame.loc[keep].copy()


def position_figure(data: Any, ticker: str, *, start: Any = None, end: Any = None,
                    max_points: int = 3000) -> go.Figure:
    """Derive events on the complete run before narrowing the visible window."""
    from .positions import asset_frame, position_events

    figure = _figure(str(ticker), y_title="Prix observé")
    market = _window(_dated(asset_frame(data, ticker)), start, end)
    events = _window(_dated(position_events(data, ticker)), start, end)
    has_price = not market.empty and "price" in market and np.isfinite(_numeric(market, "price")).any()
    if has_price:
        _line(figure, market, _numeric(market, "price"), name="Prix", color=PALETTE["ink"], max_points=max_points)
    else:
        figure.add_annotation(
            text="Série de prix absente ; seuls les prix d’événements enregistrés peuvent apparaître.",
            x=.5, y=.95, xref="paper", yref="paper", showarrow=False,
            font=dict(color=PALETTE["muted"], size=12),
        )
    if not events.empty and {"side", "price"}.issubset(events.columns):
        events["price"] = pd.to_numeric(events["price"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        for side, name, symbol, color in (
            ("buy", "Achat", "triangle-up", PALETTE["buy"]),
            ("sell", "Vente", "triangle-down", PALETTE["sell-ink"]),
        ):
            selected = events.loc[events.side.astype(str).str.lower().eq(side)].dropna(subset=["price"])
            if selected.empty:
                continue
            sources = set(selected["source"].astype(str)) if "source" in selected else set()
            if sources == {"position_signals"}:
                name = "Signal " + name.lower()
            elif sources in ({"observed_positions"}, {"executed_positions"}):
                name = "Variation " + name.lower()
            fields = [field for field in ("source", "kind", "quantity", "quantity_delta") if field in selected]
            hover = "%{x|%Y-%m-%d %H:%M:%S UTC}<br>Prix : %{y:,.4f}"
            for index, field in enumerate(fields):
                hover += f"<br>{html.escape(field)} : %{{customdata[{index}]}}"
            figure.add_trace(go.Scatter(
                x=[stamp.to_pydatetime() for stamp in selected["date"]], y=selected["price"], name=name, mode="markers",
                marker=dict(symbol=symbol, size=11, color=color, line=dict(width=1, color=PALETTE["white"])),
                customdata=selected[fields].astype(str).to_numpy() if fields else None,
                hovertemplate=hover + "<extra>%{fullData.name}</extra>",
            ))
    if not any(trace.mode == "markers" for trace in figure.data):
        figure.add_annotation(text="Aucun achat/vente enregistré dans cette fenêtre.", x=0, y=1.03,
                              xref="paper", yref="paper", showarrow=False, xanchor="left",
                              font=dict(color=PALETTE["muted"], size=12))
    return _empty(figure, "Prix et événements de position absents.")


def _observations(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    if frame.empty or metric not in frame or metric.endswith(_AGGREGATE_SUFFIXES):
        return pd.DataFrame()
    result = frame.copy()
    if "status" in result:
        result = result[~result["status"].astype(str).str.lower().isin(_FAILURE_STATUSES)]
    for column in ("row_kind", "kind", "level", "record_type"):
        if column in result:
            result = result[~result[column].astype(str).str.lower().isin(("summary", "aggregate", "aggregated"))]
    if "is_aggregate" in result:
        result = result[~result["is_aggregate"].fillna(False).astype(bool)]
    result[metric] = pd.to_numeric(result[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return result.dropna(subset=[metric])


def benchmark_figure(frame: pd.DataFrame, *, x: str, y: str, color: str = "family") -> go.Figure:
    figure = _figure("Comparaison des résultats", y_title=y)
    figure.update_xaxes(title=x)
    if frame.empty or x not in frame or y not in frame:
        return _empty(figure, "Métriques choisies indisponibles.")
    points = frame.copy()
    if "status" in points:
        points = points[~points["status"].astype(str).str.lower().isin(_FAILURE_STATUSES)]
    for metric in (x, y):
        points[metric] = pd.to_numeric(points[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
    points = points.dropna(subset=[x, y])
    if color not in points:
        points[color] = "Résultats"
    labels = points["label"] if "label" in points else points.get("run_id", pd.Series("Run", index=points.index))
    points["_label"] = labels.astype(str)
    for group, subset in points.groupby(color, dropna=False, sort=True):
        fields = [name for name in ("run_id", "model", "model_name", "seed", "fold", "partition", "status") if name in subset]
        custom = subset[["_label", *fields]].astype(str).to_numpy()
        hover = "%{customdata[0]}<br>" + html.escape(x) + ": %{x:,.4f}<br>" + html.escape(y) + ": %{y:,.4f}"
        for index, field in enumerate(fields, 1):
            hover += f"<br>{html.escape(field)} : %{{customdata[{index}]}}"
        figure.add_trace(go.Scatter(
            x=subset[x], y=subset[y], mode="markers", name=str(group), customdata=custom,
            marker=dict(color=_color(str(group)), size=10, opacity=0.85),
            hovertemplate=hover + "<extra></extra>",
        ))
    return _empty(figure, "Aucun résultat fini pour ces métriques.")


def metric_distribution_figure(frame: pd.DataFrame, *, metric: str, group: str = "model") -> go.Figure:
    figure = _figure("Distribution des observations", y_title=metric)
    observations = _observations(frame, metric)
    if observations.empty:
        return _empty(figure, "Observations individuelles absentes ; statistiques agrégées exclues.")
    if group not in observations and group == "model" and "model_name" in observations:
        group = "model_name"
    if group not in observations:
        observations[group] = "Résultats"
    for name, subset in observations.groupby(group, dropna=False, sort=True):
        fields = [column for column in ("run_id", "seed", "fold", "partition", "status") if column in subset]
        hover = "%{y:,.4f}"
        for index, field in enumerate(fields):
            hover += f"<br>{html.escape(field)} : %{{customdata[{index}]}}"
        figure.add_trace(go.Box(
            y=subset[metric], name=str(name), boxpoints="all", jitter=0.25,
            marker=dict(color=_color(str(name)), size=6), line=dict(color=_color(str(name))),
            customdata=subset[fields].astype(str).to_numpy() if fields else None,
            hovertemplate=hover + "<extra>%{fullData.name}</extra>",
        ))
    figure.update_xaxes(title=group)
    return figure


def exposure_figure(data: Any, *, max_points: int = 3000) -> go.Figure:
    figure = _figure("Exposition", y_title="Exposition (%)")
    frame = _dated(_table(data, "equity"))
    if frame.empty or not {"gross_exposure", "net_exposure"}.intersection(frame.columns):
        frame = _dated(_table(data, "positions"))
    run_id, _ = _identity(data)
    for column, label, dash in (("gross_exposure", "Brute", "solid"), ("net_exposure", "Nette", "dash")):
        if not frame.empty and column in frame:
            _line(figure, frame, _numeric(frame, column) * 100.0, name=label,
                  color=PALETTE["blue"], max_points=max_points, dash=dash, suffix=" %")
    return _empty(figure, "Séries d’exposition brute/nette absentes.")


def trades_figure(data: Any) -> go.Figure:
    figure = _figure("Résultat des trades", y_title="Nombre de trades")
    trades = _table(data, "trades")
    column = next((name for name in ("net_pnl", "pnl", "realized_pnl") if name in trades), None)
    if column is not None:
        values = _numeric(trades, column)
        values = values[np.isfinite(values)]
        if len(values):
            run_id, label = _identity(data)
            figure.add_trace(go.Histogram(
                x=values, name=label, nbinsx=min(40, max(5, int(np.sqrt(len(values))))),
                marker_color=PALETTE["blue"], hovertemplate="PnL : %{x:,.4f}<br>Trades : %{y}<extra></extra>",
            ))
            figure.update_xaxes(title="PnL net" if column == "net_pnl" else "PnL enregistré")
    return _empty(figure, "Trades ou PnL individuels absents.")


def history_figure(data: Any) -> go.Figure:
    figure = _figure("Historique d’apprentissage", y_title="Valeur enregistrée")
    history = _value(data, "history", {})
    if not isinstance(history, Mapping):
        return _empty(figure, "Historique d’apprentissage absent.")
    # FitResult artifacts contain a nested history mapping; old exports are flat.
    history = history.get("history", history)
    if not isinstance(history, Mapping):
        return _empty(figure, "Historique d’apprentissage absent.")
    run_id, _ = _identity(data)
    for name, raw in history.items():
        if name in ("epoch", "epochs") or not isinstance(raw, (list, tuple, np.ndarray, pd.Series)):
            continue
        if len(raw) == 0 or any(isinstance(value, (dict, list, tuple)) for value in raw):
            continue
        values = pd.to_numeric(pd.Series(raw), errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        if not np.isfinite(values).any():
            continue
        epochs = history.get("epoch", history.get("epochs"))
        x = epochs if isinstance(epochs, (list, tuple, np.ndarray)) and len(epochs) == len(values) else np.arange(1, len(values) + 1)
        figure.add_trace(go.Scatter(
            x=x, y=values, name=str(name), mode="lines", connectgaps=False,
            line=dict(color=_color(run_id + str(name)), width=2),
            hovertemplate="Epoch : %{x}<br>%{y:,.6f}<extra>%{fullData.name}</extra>",
        ))
    figure.update_xaxes(title="Epoch")
    return _empty(figure, "Historique d’apprentissage absent.")


def monthly_returns_figure(data: Any) -> go.Figure:
    figure = _figure("Rendements mensuels", y_title="Année")
    frame = _dated(_table(data, "equity"))
    if frame.empty:
        return _empty(figure, "Série de rendements absente.")
    derived = False
    if "net_return" in frame:
        returns = pd.Series(_numeric(frame, "net_return"), index=frame["date"])
    elif "equity" in frame:
        returns = pd.Series(_numeric(frame, "equity"), index=frame["date"]).pct_change(fill_method=None)
        # The first recorded capital has no preceding observed transition.
        returns = returns.iloc[1:]
        derived = True
    else:
        return _empty(figure, "Série de rendements absente.")
    returns = returns.replace([np.inf, -np.inf], np.nan)
    # Missing observations stay missing, rather than inventing flat returns.
    if not returns.notna().any():
        return _empty(figure, "Série de rendements absente.")
    work = pd.DataFrame({"year": returns.index.year, "month": returns.index.month, "return": returns.to_numpy()})
    monthly = work.groupby(["year", "month"])["return"].agg(
        lambda values: float(np.prod(1.0 + values) - 1.0) if values.notna().all() else np.nan
    ).unstack("month").reindex(columns=range(1, 13))
    if not np.isfinite(monthly.to_numpy()).any():
        return _empty(figure, "Rendements mensuels incomplets : observations manquantes.")
    figure.add_trace(go.Heatmap(
        z=monthly.to_numpy() * 100.0,
        x=["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"],
        y=monthly.index.astype(str).tolist(),
        colorscale=[[0, PALETTE["sell"]], [0.5, PALETTE["paper"]], [1, PALETTE["blue"]]], zmid=0,
        colorbar=dict(title="%"), hoverongaps=False,
        hovertemplate="%{x} %{y}<br>Rendement : %{z:.2f} %<extra></extra>",
    ))
    figure.update_xaxes(title="Mois")
    if derived:
        figure.update_layout(title="Rendements mensuels · transitions de capital observées")
    return figure


def export_html(figures: Mapping[str, go.Figure]) -> str:
    """Produce a standalone document with Plotly's runtime embedded once."""
    sections = []
    for index, (name, figure) in enumerate(figures.items()):
        plot = pio.to_html(
            figure, full_html=False, include_plotlyjs=True if index == 0 else False,
            config={"responsive": True, "displaylogo": False, "scrollZoom": True},
            div_id=f"run-chart-{index}",
        )
        sections.append(f"<section><h2>{html.escape(str(name))}</h2>{plot}</section>")
    body = "\n".join(sections) or "<p>Aucune figure à exporter.</p>"
    return (
        '<!doctype html><html lang="fr"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Visualisation des résultats</title>'
        f'<style>body{{font:14px {SYSTEM_FONT};color:{PALETTE["ink"]};background:{PALETTE["paper"]};'
        'max-width:1280px;margin:24px auto;padding:0 20px}section{background:white;'
        f'padding:16px;margin:24px 0;border:1px solid {PALETTE["line"]};border-radius:8px}}'
        'h1{font-size:24px}h2{font-size:18px}</style></head>'
        f'<body><h1>Visualisation des résultats</h1>{body}</body></html>'
    )


__all__ = [
    "equity_figure", "drawdown_figure", "benchmark_figure", "metric_distribution_figure",
    "exposure_figure", "trades_figure", "history_figure", "monthly_returns_figure", "export_html",
    "capital_drawdown_figure", "kpi_bar_figure", "position_figure",
]
