"""Capital, asset entries/exits and run comparisons over saved artifacts."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
import re

import pandas as pd

from trading_system.paths import artifacts_root
from trading_system.visualization.adapters import load_run
from trading_system.visualization.analytics import slice_run, window_statistics
from trading_system.visualization.catalog import catalog_frame, catalog_signature, comparison_issues, discover_runs
from trading_system.visualization.charts import capital_drawdown_figure, export_html, kpi_bar_figure, position_figure
from trading_system.visualization.positions import asset_frame, asset_names, position_events
from trading_system.visualization.theme import dashboard_css


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"seed", "fold", "retrain_every_sessions"}
    names = [
        name for name in frame.select_dtypes(include="number").columns
        if name not in excluded and not name.startswith("has_") and frame[name].notna().any()
        and not name.endswith(("_mean", "_std", "_median", "_min", "_max", "_sem"))
    ]
    preferred = ["net_return", "model_return", "cumulative_return", "max_drawdown", "net_sharpe", "sharpe", "sharpe_ratio", "regularized_sharpe", "turnover", "trade_count"]
    return [name for name in preferred if name in names] + sorted(set(names) - set(preferred))


def _filter_column(st, frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    if column not in frame:
        return frame
    options = sorted(set(frame[column].dropna().astype(str)) - {"", "None", "nan"})
    if not options:
        return frame
    selected = st.multiselect(label, options, key=f"filter-{column}")
    return frame.loc[frame[column].astype(str).isin(selected)] if selected else frame


def _display_records(records):
    """Keep identities intact while using readable legends and controls."""
    result = {}
    for record in records:
        directory = record.path
        while re.fullmatch(r"(?:seed|fold|every|trial)[-_]\d+", directory.name) and directory.parent != directory:
            directory = directory.parent
        name = directory.name
        name = name if len(name) <= 44 else name[:28] + "…" + name[-15:]
        parts = [name]
        for key in ("model", "candidate", "seed", "fold", "partition", "decoder", "method", "retrain_every_sessions"):
            value = record.selectors.get(key, record.metadata.get(key))
            if value is None or isinstance(value, (list, dict)):
                continue
            value = str(value)
            if key == "candidate" and re.fullmatch(r"[a-fA-F0-9]{32,}", value):
                value = value[:8]
            if value and value not in name:
                parts.append(f"{key} {value}")
        result[record.run_id] = replace(record, label=" · ".join(parts))
    labels = pd.Series({key: record.label for key, record in result.items()})
    for key in labels[labels.duplicated(keep=False)].index:
        result[key] = replace(result[key], label=f"{result[key].label} · {key[-8:]}")
    return result


def _default_run(options, records):
    available = [value for value in options if "equity" in records[value].tables and records[value].status not in ("error", "failed", "failure")]
    preferred = [value for value in available if records[value].family == "mt5" and "positions" in records[value].tables]
    described = [value for value in preferred if isinstance(records[value].metadata.get("tickers"), (list, tuple))]
    choices = described or preferred or available or options
    def modified(value):
        path = records[value].tables.get("equity")
        try:
            return path.stat().st_mtime_ns if path is not None else 0
        except OSError:
            return 0
    return [max(choices, key=modified)] if choices else []


def main(argv: list[str] | None = None) -> None:
    import streamlit as st

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", type=Path, default=artifacts_root())
    args = parser.parse_args(argv)
    st.set_page_config(page_title="Backtests · SIGNAL/API", page_icon="↗", layout="wide", initial_sidebar_state="collapsed")
    st.html(dashboard_css())
    st.caption("SIGNAL/API · RECHERCHE")
    st.title("Backtests")
    st.caption("Capital, risque et prises de position.")

    @st.cache_data(show_spinner=False, max_entries=8)
    def cached_catalog(directory: str, signature):
        return discover_runs(Path(directory))

    @st.cache_data(show_spinner=False, max_entries=32)
    def cached_run(record, signature, details: bool = False, tickers: tuple[str, ...] = ()):
        return load_run(record, include_details=details, tickers=tickers or None)

    with st.sidebar:
        st.header("Résultats")
        root = Path(st.text_input("Dossier artifacts", str(args.artifacts_root))).expanduser().resolve()
        if st.button("Actualiser", width="stretch"):
            cached_catalog.clear()
            cached_run.clear()
        if not root.is_dir():
            st.error(f"Dossier introuvable : {root}")
            return
    try:
        signature = catalog_signature(root)
        with st.spinner("Lecture des résultats…"):
            catalog = cached_catalog(str(root), signature)
    except (OSError, ValueError) as exc:
        st.error(f"Catalogue indisponible : {exc}")
        return
    if not catalog.records:
        st.info("Aucun résultat reconnu. Choisir un dossier contenant des résultats exportés par le projet.")
        return
    originals = {record.run_id: record for record in catalog.records}
    records = _display_records(catalog.records)
    frame = catalog_frame(list(records.values()))
    with st.sidebar:
        frame = _filter_column(st, frame, "family", "Source")
        with st.expander("Filtres avancés"):
            for column, label in (("status", "État"), ("model", "Modèle"), ("candidate", "Candidat"), ("partition", "Partition"), ("seed", "Seed"), ("fold", "Fold"), ("decoder", "Décodeur")):
                frame = _filter_column(st, frame, column, label)
            if "tickers" in frame:
                options = sorted({ticker.strip() for value in frame.tickers.dropna().astype(str) for ticker in value.split(",") if ticker.strip()})
                tickers = st.multiselect("Univers contient", options, key="filter-tickers") if options else []
                if tickers:
                    frame = frame.loc[frame.tickers.fillna("").astype(str).map(lambda value: bool({token.strip() for token in value.split(",")} & set(tickers)))]
                st.caption("Filtre univers : capital du portefeuille entier.")
        st.caption(f"{len(frame):,} runs disponibles")
        if catalog.issues:
            with st.expander(f"Artifacts · {len(catalog.issues)} observations"):
                for issue in catalog.issues[:100]:
                    st.write(issue)
    if frame.empty:
        st.info("Aucun run ne correspond aux filtres.")
        return

    options = frame.run_id.astype(str).tolist()
    selection_key = hashlib.sha256("|".join(options).encode()).hexdigest()[:12]
    selected_ids = st.multiselect("Runs", options, default=_default_run(options, records), format_func=lambda value: records[value].label, max_selections=8, key=f"runs-{selection_key}")
    selected_records = [records[value] for value in selected_ids]
    if not selected_records:
        st.info("Sélectionner un ou plusieurs runs.")
        return
    issues = comparison_issues(selected_records) if len(selected_records) > 1 else []
    if issues:
        st.warning("Comparaison descriptive : protocoles différents ou incomplets.")
        with st.expander("Vérifier la comparabilité"):
            for issue in issues:
                st.write(issue)
    loaded = []
    for record in selected_records:
        try:
            loaded.append(cached_run(record, signature))
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"{record.label} : {exc}")
    dates = [pd.to_datetime(data.equity.date, utc=True) for data in loaded if not data.equity.empty and "date" in data.equity]
    start = end = None
    if dates:
        first, last = min(values.min() for values in dates).date(), max(values.max() for values in dates).date()
        date_key = hashlib.sha256(("|".join(selected_ids) + repr(signature)).encode()).hexdigest()[:12]
        window = st.date_input("Période affichée (UTC)", value=(first, last), min_value=first, max_value=last, key=f"dates-{date_key}")
        if isinstance(window, (tuple, list)) and len(window) == 2:
            start, end = window
        else:
            st.info("Sélectionner début et fin de période.")
    visible = [slice_run(data, start, end) for data in loaded]
    selected_frame = frame.loc[frame.run_id.isin(selected_ids)]
    portfolio_tab, comparison_tab = st.tabs(["Portefeuille", "Comparer les runs"])

    with portfolio_tab:
        detail_id = st.selectbox("Run observé", selected_ids, format_func=lambda value: records[value].label) if len(selected_ids) > 1 else selected_ids[0]
        current = next((data for data in visible if data.record.run_id == detail_id), None)
        stats = window_statistics(current) if current is not None else {}
        if stats:
            cards = st.columns(4)
            cards[0].metric("Capital final", f"{stats['last_equity']:,.2f}")
            cards[1].metric("Stratégie", f"{stats['observed_return']:+.2%}")
            cards[2].metric("Buy & Hold", f"{stats['benchmark_return']:+.2%}" if "benchmark_return" in stats else "—")
            cards[3].metric("Drawdown max.", f"{stats['window_drawdown']:.2%}")
        available = [data for data in visible if not data.equity.empty]
        figures = {}
        if available:
            normalize = st.toggle("Base 100", value=len(available) > 1)
            st.caption("Fenêtre : rendement et drawdown depuis le premier capital observé. Survol pour les valeurs ; zoom pour explorer.")
            figures["Capital et drawdown"] = capital_drawdown_figure(available, normalize=normalize)
            st.plotly_chart(figures["Capital et drawdown"], width="stretch", theme=None)
            without_reference = [data.record.label for data in available if "benchmark_equity" not in data.equity or data.equity.benchmark_equity.notna().sum() == 0]
            if without_reference:
                st.info("Buy & Hold absent des exports : " + ", ".join(without_reference))
        missing = [data.record.label for data in visible if data.equity.empty]
        if missing:
            st.info("Courbes absentes dans la fenêtre affichée : " + ", ".join(missing))

        st.subheader("Prises de position")
        st.caption("Un graphique par actif : prix, achats et ventes. Le run observé détermine les positions affichées.")
        baseline = next((data for data in loaded if data.record.run_id == detail_id), None)
        record = records[detail_id]
        known = asset_names(baseline) if baseline is not None else []
        asset_key = f"asset-list-{detail_id}"
        if not known:
            if st.button("Charger les actifs disponibles"):
                try:
                    st.session_state[asset_key] = asset_names(cached_run(record, signature, True))
                except (OSError, ValueError, KeyError) as exc:
                    st.error(str(exc))
            known = st.session_state.get(asset_key, [])
        details = None
        if known:
            assets = st.multiselect("Actifs", known, default=known[:1], key=f"assets-{detail_id}")
            if assets:
                try:
                    with st.spinner("Lecture des prix et positions…"):
                        details = cached_run(record, signature, True, tuple(assets))
                    for ticker in assets:
                        figures[ticker] = position_figure(details, ticker, start=start, end=end)
                        st.plotly_chart(figures[ticker], width="stretch", theme=None)
                        events = position_events(details, ticker)
                        sources = set(events.source) if not events.empty else set()
                        if "position_signals" in sources:
                            st.caption(f"{ticker} · Marqueurs = changements de cible du modèle ; prix observé, exécution non confirmée.")
                        elif sources & {"observed_positions", "executed_positions"}:
                            st.caption(f"{ticker} · Marqueurs = variations de position observées ; prix d’exécution absent.")
                        elif sources:
                            st.caption(f"{ticker} · Achats/ventes issus des exécutions enregistrées.")
                        prices = asset_frame(details, ticker)
                        if prices.attrs.get("price_column"):
                            st.caption(f"Prix : {prices.attrs['price_column']} · {prices.attrs.get('price_basis', '')}")
                except (OSError, ValueError, KeyError) as exc:
                    st.error(f"Prix et positions indisponibles : {exc}")
        else:
            st.info("Aucun actif identifié dans les métadonnées chargées.")

        if figures:
            with st.expander("Exporter"):
                st.download_button("Graphiques HTML", export_html(figures), "backtests.html", "text/html", key="equity-html")
                if available:
                    csv = pd.concat([data.equity.assign(run_id=data.record.run_id) for data in available], ignore_index=True)
                    st.download_button("Capital CSV", csv.to_csv(index=False).encode(), "equity.csv", "text/csv", key="equity-csv")
        with st.expander("Données et provenance"):
            original = originals[detail_id]
            st.json({"run": original.label, "run_id": detail_id, "source": record.family, "path": str(record.path), "metadata": (details or baseline).record.metadata if (details or baseline) else record.metadata, "selectors": record.selectors})
            if available:
                st.dataframe(pd.DataFrame([{"run": data.record.label, **window_statistics(data)} for data in available]), width="stretch", hide_index=True)
            for warning in dict.fromkeys([*record.warnings, *(baseline.warnings if baseline else ()), *(details.warnings if details else ())]):
                st.write(warning)
            if details is not None:
                displayed = slice_run(details, start, end)
                for table, label in ((displayed.positions, "Positions"), (displayed.orders, "Ordres exécutés"), (displayed.trades, "Trades")):
                    if not table.empty:
                        st.write(label)
                        preview = table.head(1000).copy()
                        preview.attrs = {}
                        st.dataframe(preview, width="stretch", hide_index=True)
                        st.download_button(f"{label} CSV", table.to_csv(index=False).encode(), f"{label.lower().replace(' ', '-')}.csv", "text/csv")

    with comparison_tab:
        st.subheader("KPI par run")
        st.caption("Métriques enregistrées sur l’intervalle complet de chaque run, indépendantes du filtre dates.")
        if len(selected_records) < 2:
            st.caption("Ajouter d’autres runs dans la sélection pour comparer leurs barres.")
        metrics = _metric_columns(selected_frame)
        if metrics:
            chosen = st.multiselect("KPI à comparer", metrics, default=metrics[:3], key=f"kpis-{selection_key}")
            for metric in chosen:
                st.plotly_chart(kpi_bar_figure(selected_frame, metric=metric), width="stretch", theme=None)
        else:
            st.info("Aucune métrique numérique enregistrée pour cette sélection.")
        with st.expander("Valeurs enregistrées"):
            leading = [name for name in ("label", "family", "status", "model", "seed", "fold", "partition", *metrics) if name in selected_frame]
            st.dataframe(selected_frame, width="stretch", hide_index=True, column_order=leading,
                         column_config={"label": st.column_config.TextColumn("Run", width="medium"), "run_id": None})
            st.download_button("KPI CSV", selected_frame.to_csv(index=False).encode(), "metrics.csv", "text/csv", key="metrics-csv")


if __name__ == "__main__":
    main()
