#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_BASE = SCRIPT_DIR / "datasets" / "cac40_daily"

# Composition courante du CAC 40 au 22 decembre 2025.
# Source de reference utilisee pour figer une liste stable a date.
CAC40_CONSTITUENTS = [
    {"company": "Accor", "ticker": "AC.PA"},
    {"company": "Air Liquide", "ticker": "AI.PA"},
    {"company": "Airbus", "ticker": "AIR.PA"},
    {"company": "ArcelorMittal", "ticker": "MT.AS"},
    {"company": "AXA", "ticker": "CS.PA"},
    {"company": "BNP Paribas", "ticker": "BNP.PA"},
    {"company": "Bouygues", "ticker": "EN.PA"},
    {"company": "Bureau Veritas", "ticker": "BVI.PA"},
    {"company": "Capgemini", "ticker": "CAP.PA"},
    {"company": "Carrefour", "ticker": "CA.PA"},
    {"company": "Credit Agricole", "ticker": "ACA.PA"},
    {"company": "Danone", "ticker": "BN.PA"},
    {"company": "Dassault Systemes", "ticker": "DSY.PA"},
    {"company": "Eiffage", "ticker": "FGR.PA"},
    {"company": "Engie", "ticker": "ENGI.PA"},
    {"company": "EssilorLuxottica", "ticker": "EL.PA"},
    {"company": "Eurofins Scientific", "ticker": "ERF.PA"},
    {"company": "Euronext", "ticker": "ENX.PA"},
    {"company": "Hermes", "ticker": "RMS.PA"},
    {"company": "Kering", "ticker": "KER.PA"},
    {"company": "Legrand", "ticker": "LR.PA"},
    {"company": "L'Oreal", "ticker": "OR.PA"},
    {"company": "LVMH", "ticker": "MC.PA"},
    {"company": "Michelin", "ticker": "ML.PA"},
    {"company": "Orange", "ticker": "ORA.PA"},
    {"company": "Pernod Ricard", "ticker": "RI.PA"},
    {"company": "Publicis Groupe", "ticker": "PUB.PA"},
    {"company": "Renault", "ticker": "RNO.PA"},
    {"company": "Safran", "ticker": "SAF.PA"},
    {"company": "Saint-Gobain", "ticker": "SGO.PA"},
    {"company": "Sanofi", "ticker": "SAN.PA"},
    {"company": "Schneider Electric", "ticker": "SU.PA"},
    {"company": "Societe Generale", "ticker": "GLE.PA"},
    {"company": "Stellantis", "ticker": "STLAP.PA"},
    {"company": "STMicroelectronics", "ticker": "STMPA.PA"},
    {"company": "Thales", "ticker": "HO.PA"},
    {"company": "TotalEnergies", "ticker": "TTE.PA"},
    {"company": "Unibail-Rodamco-Westfield", "ticker": "URW.PA"},
    {"company": "Veolia Environnement", "ticker": "VIE.PA"},
    {"company": "Vinci", "ticker": "DG.PA"},
]

PRICE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
]

RENAMED_COLUMNS = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "stock_splits",
}

YAHOO_MACRO_TICKERS = {
    "^FCHI": "market_close",
    "^VIX": "vix_close",
    "BZ=F": "oil_close",
    "DX-Y.NYB": "dxy_close",
    "GC=F": "gold_close",
}

FRED_SERIES = {
    "DGS2": "ust2y",
    "DGS10": "ust10y",
    "IR3TIB01FRM156N": "frt2y",
    "IRLTLT01FRM156N": "frt10y",
    "BAMLH0A0HYM2": "credit_spread",
}

SECTOR_BUCKETS = [
    "energy",
    "banks",
    "insurance",
    "materials",
    "industrials",
    "consumer_defensive",
    "consumer_cyclical",
    "healthcare",
    "technology",
    "telecom",
    "utilities",
    "real_estate",
    "financial_services",
    "unknown",
]


class FredDownloadError(RuntimeError):
    pass


def load_dependencies() -> tuple[Any, Any]:
    missing: list[str] = []

    try:
        pandas = importlib.import_module("pandas")
    except ModuleNotFoundError:
        missing.append("pandas")
        pandas = None

    try:
        yfinance = importlib.import_module("yfinance")
    except ModuleNotFoundError:
        missing.append("yfinance")
        yfinance = None

    if missing:
        packages = " ".join(missing)
        raise SystemExit(
            "Modules manquants: "
            f"{packages}. Installe-les avec `python3 -m pip install {packages}`."
        )

    return pandas, yfinance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Telecharge les donnees journalieres OHLCV du CAC 40 depuis Yahoo Finance, "
            "enrichit avec macro/fondamentaux/secteurs, puis conserve automatiquement "
            "le plus petit format entre csv et parquet."
        )
    )
    parser.add_argument(
        "--start",
        default="2000-01-01",
        help="Date de debut inclusive au format YYYY-MM-DD. Defaut: 2000-01-01.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help=(
            "Date de fin inclusive au format YYYY-MM-DD. "
            "Par defaut: aujourd'hui."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_BASE),
        help=(
            "Chemin de base du fichier de sortie, sans extension de preference. "
            f"Defaut: {DEFAULT_OUTPUT_BASE}"
        ),
    )
    parser.add_argument(
        "--format",
        choices=("auto", "csv", "parquet"),
        default="auto",
        help="Mode de sauvegarde. Defaut: auto.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Nombre de tickers telecharges par lot. Defaut: 10.",
    )
    parser.add_argument(
        "--fred-timeout",
        type=int,
        default=120,
        help="Timeout reseau en secondes pour chaque serie FRED. Defaut: 120.",
    )
    parser.add_argument(
        "--fred-retries",
        type=int,
        default=3,
        help="Nombre de tentatives par serie FRED. Defaut: 3.",
    )
    parser.add_argument(
        "--allow-missing-fred",
        action="store_true",
        help=(
            "Sauvegarde quand meme le dataset si FRED est inaccessible, "
            "avec les colonnes FRED remplies par des valeurs manquantes."
        ),
    )
    return parser.parse_args()


def validate_date(date_str: str) -> str:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise SystemExit(
            f"Date invalide `{date_str}`. Format attendu: YYYY-MM-DD."
        ) from exc


def resolve_end_date_for_yfinance(end: str | None) -> str | None:
    if end is None:
        return None

    validated = datetime.strptime(validate_date(end), "%Y-%m-%d").date()
    return (validated + timedelta(days=1)).isoformat()


def resolve_end_date_inclusive(end: str | None) -> str:
    if end is None:
        return datetime.now(timezone.utc).date().isoformat()
    return validate_date(end)


def output_base_path(output: str) -> Path:
    path = Path(output).expanduser()
    if path.suffix.lower() in {".csv", ".parquet"}:
        return path.with_suffix("")
    return path


def get_constituents_frame(pandas_module: Any) -> Any:
    return pandas_module.DataFrame(CAC40_CONSTITUENTS).sort_values("ticker").reset_index(
        drop=True
    )


def chunked(values: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        raise SystemExit("--batch-size doit etre superieur a 0.")
    return [values[index : index + size] for index in range(0, len(values), size)]


def extract_ticker_frame(raw_data: Any, ticker: str, pandas_module: Any) -> Any:
    if raw_data.empty:
        return pandas_module.DataFrame(columns=PRICE_COLUMNS)

    if not hasattr(raw_data.columns, "nlevels") or raw_data.columns.nlevels == 1:
        frame = raw_data.copy()
    elif ticker in raw_data.columns.get_level_values(0):
        frame = raw_data[ticker].copy()
    else:
        try:
            frame = raw_data.xs(ticker, axis=1, level=-1).copy()
        except KeyError:
            return pandas_module.DataFrame(columns=PRICE_COLUMNS)

    for column in PRICE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pandas_module.NA

    frame = frame[PRICE_COLUMNS]
    frame = frame.dropna(how="all", subset=PRICE_COLUMNS)
    return frame


def normalize_frame(frame: Any, ticker: str, company: str) -> Any:
    normalized = frame.reset_index().rename(columns=RENAMED_COLUMNS)
    normalized["ticker"] = ticker
    normalized["company"] = company

    ordered_columns = [
        "date",
        "ticker",
        "company",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits",
    ]

    return normalized[ordered_columns]


def download_history(
    pandas_module: Any,
    yfinance_module: Any,
    constituents: Any,
    start: str,
    end: str | None,
    batch_size: int,
) -> Any:
    batches = chunked(constituents["ticker"].tolist(), batch_size)
    company_by_ticker = dict(zip(constituents["ticker"], constituents["company"]))
    frames = []

    for batch_index, batch in enumerate(batches, start=1):
        batch_label = ", ".join(batch)
        print(f"[{batch_index}/{len(batches)}] Telechargement CAC40: {batch_label}")

        raw_data = yfinance_module.download(
            tickers=batch,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            actions=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )

        for ticker in batch:
            frame = extract_ticker_frame(raw_data, ticker, pandas_module)
            if frame.empty:
                print(f"  - Aucun resultat pour {ticker}")
                continue

            frames.append(normalize_frame(frame, ticker, company_by_ticker[ticker]))

    if not frames:
        raise SystemExit("Aucune donnee CAC40 n'a ete telechargee.")

    dataset = pandas_module.concat(frames, ignore_index=True)
    dataset["date"] = pandas_module.to_datetime(dataset["date"], errors="coerce")
    dataset = dataset.dropna(subset=["date"]).copy()
    dataset = dataset.sort_values(["date", "ticker"]).reset_index(drop=True)
    return dataset


def _download_single_yahoo_close_series(
    pandas_module: Any,
    yfinance_module: Any,
    ticker: str,
    start: str,
    end: str | None,
) -> Any:
    raw = yfinance_module.download(
        tickers=ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    if raw is None or raw.empty:
        return pandas_module.DataFrame(columns=["date", "value"])

    if hasattr(raw.columns, "nlevels") and raw.columns.nlevels > 1:
        if ticker in raw.columns.get_level_values(0):
            frame = raw[ticker].copy()
        else:
            frame = raw.droplevel(0, axis=1)
    else:
        frame = raw.copy()

    source_col = "Adj Close" if "Adj Close" in frame.columns else "Close"
    if source_col not in frame.columns:
        return pandas_module.DataFrame(columns=["date", "value"])

    out = frame[[source_col]].reset_index()
    out = out.rename(columns={"Date": "date", source_col: "value"})
    out = out[["date", "value"]].dropna(subset=["value"])
    return out


def download_yahoo_macro_series(
    pandas_module: Any,
    yfinance_module: Any,
    start: str,
    end: str | None,
) -> Any:
    merged = None

    for ticker, target_col in YAHOO_MACRO_TICKERS.items():
        print(f"[macro-yahoo] Telechargement {ticker} -> {target_col}")
        frame = _download_single_yahoo_close_series(
            pandas_module=pandas_module,
            yfinance_module=yfinance_module,
            ticker=ticker,
            start=start,
            end=end,
        )
        if frame.empty:
            raise SystemExit(f"Aucune donnee Yahoo pour la serie macro `{ticker}`.")
        frame = frame.rename(columns={"value": target_col})

        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="date", how="outer")

    if merged is None or merged.empty:
        raise SystemExit("Aucune serie macro Yahoo n'a ete telechargee.")

    merged["date"] = pandas_module.to_datetime(merged["date"], errors="coerce")
    merged = merged.dropna(subset=["date"]).copy()
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def _download_fred_series(
    pandas_module: Any,
    series_id: str,
    start: str,
    end_inclusive: str,
    timeout: int,
    retries: int,
) -> Any:
    if timeout <= 0:
        raise SystemExit("--fred-timeout doit etre superieur a 0.")
    if retries <= 0:
        raise SystemExit("--fred-retries doit etre superieur a 0.")

    query = urlencode({"id": series_id, "cosd": start, "coed": end_inclusive})
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{query}"
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        request = Request(
            url,
            headers={
                "Accept": "text/csv,*/*;q=0.8",
                "User-Agent": "Mozilla/5.0",
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                frame = pandas_module.read_csv(response)
            break
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                raise FredDownloadError(
                    f"Impossible de telecharger la serie FRED `{series_id}` "
                    f"apres {retries} tentative(s): {exc}"
                ) from exc
            wait_seconds = min(2 ** (attempt - 1), 10)
            print(
                f"  - FRED {series_id}: tentative {attempt}/{retries} echouee "
                f"({exc}); nouvelle tentative dans {wait_seconds}s"
            )
            time.sleep(wait_seconds)
    else:
        raise FredDownloadError(
            f"Impossible de telecharger la serie FRED `{series_id}`: {last_error}"
        )

    date_col = next(
        (col for col in ("DATE", "date", "observation_date") if col in frame.columns),
        None,
    )
    value_col = next(
        (
            col
            for col in frame.columns
            if str(col).strip().upper() == series_id.upper()
        ),
        None,
    )
    if date_col is None or value_col is None:
        received = ", ".join(str(col) for col in frame.columns[:8]) or "aucune"
        raise FredDownloadError(
            f"Reponse FRED invalide pour `{series_id}`. "
            f"Colonnes recues: {received}."
        )

    out = frame.rename(columns={date_col: "date", value_col: "value"})
    out["date"] = pandas_module.to_datetime(out["date"], errors="coerce")
    out["value"] = pandas_module.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date"]).copy()
    start_ts = pandas_module.to_datetime(start, errors="coerce")
    end_ts = pandas_module.to_datetime(end_inclusive, errors="coerce")
    out = out[(out["date"] >= start_ts) & (out["date"] <= end_ts)].copy()
    out = out.sort_values("date").reset_index(drop=True)
    return out


def download_fred_macro_series(
    pandas_module: Any,
    start: str,
    end_inclusive: str,
    timeout: int,
    retries: int,
) -> Any:
    merged = None

    for series_id, target_col in FRED_SERIES.items():
        print(f"[macro-fred] Telechargement {series_id} -> {target_col}")
        frame = _download_fred_series(
            pandas_module=pandas_module,
            series_id=series_id,
            start=start,
            end_inclusive=end_inclusive,
            timeout=timeout,
            retries=retries,
        )
        if frame.empty:
            raise FredDownloadError(
                f"Aucune donnee FRED pour la serie `{series_id}`."
            )
        frame = frame.rename(columns={"value": target_col})

        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="date", how="outer")

    if merged is None or merged.empty:
        raise FredDownloadError("Aucune serie FRED n'a ete telechargee.")

    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def build_missing_fred_macro_frame(pandas_module: Any, dataset: Any) -> Any:
    frame = (
        dataset[["date"]]
        .drop_duplicates()
        .sort_values("date")
        .reset_index(drop=True)
    )
    for target_col in FRED_SERIES.values():
        frame[target_col] = pandas_module.NA
    return frame


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def normalize_sector_bucket(sector: Any, industry: Any) -> str:
    sector_text = str(sector or "").strip().lower()
    industry_text = str(industry or "").strip().lower()
    both = f"{sector_text} {industry_text}".strip()

    if any(token in both for token in ("bank", "banque")):
        return "banks"
    if "insurance" in both or "assurance" in both:
        return "insurance"
    if "energy" in both or "oil" in both or "gas" in both:
        return "energy"
    if "material" in both or "metals" in both or "steel" in both:
        return "materials"
    if "health" in both or "pharma" in both or "biotech" in both:
        return "healthcare"
    if "technology" in both or "software" in both or "semiconductor" in both:
        return "technology"
    if "telecom" in both or "communication" in both:
        return "telecom"
    if "utility" in both:
        return "utilities"
    if "real estate" in both or "reit" in both:
        return "real_estate"
    if "consumer defensive" in both or "household" in both:
        return "consumer_defensive"
    if "consumer cyclical" in both or "luxury" in both or "retail" in both:
        return "consumer_cyclical"
    if "industrial" in both or "aerospace" in both or "transport" in both:
        return "industrials"
    if "financial services" in both or "asset management" in both:
        return "financial_services"
    return "unknown"


def download_yahoo_snapshot_metadata(
    pandas_module: Any,
    yfinance_module: Any,
    constituents: Any,
) -> Any:
    rows = []

    for ticker in constituents["ticker"].tolist():
        print(f"[metadata] Telechargement snapshot Yahoo: {ticker}")
        info: dict[str, Any] = {}
        fast_info: dict[str, Any] = {}

        instrument = yfinance_module.Ticker(ticker)
        try:
            info = instrument.get_info()
            if info is None:
                info = {}
        except Exception:
            try:
                info = instrument.info or {}
            except Exception:
                info = {}

        try:
            fast_info = dict(instrument.fast_info)
        except Exception:
            fast_info = {}

        sector_value = info.get("sector")
        industry_value = info.get("industry")

        market_cap = _safe_float(info.get("marketCap"))
        if not pandas_module.notna(market_cap):
            market_cap = _safe_float(fast_info.get("market_cap"))

        shares_outstanding = _safe_float(info.get("sharesOutstanding"))
        if not pandas_module.notna(shares_outstanding):
            shares_outstanding = _safe_float(fast_info.get("shares"))

        short_percent_float = _safe_float(info.get("shortPercentOfFloat"))
        short_ratio = _safe_float(info.get("shortRatio"))

        rows.append(
            {
                "ticker": ticker,
                "sector": str(sector_value).strip() if sector_value else "unknown",
                "industry": str(industry_value).strip() if industry_value else "unknown",
                "sector_bucket": normalize_sector_bucket(sector_value, industry_value),
                "market_cap": market_cap,
                "book_value": _safe_float(info.get("bookValue")),
                "trailing_eps": _safe_float(info.get("trailingEps")),
                "shares_outstanding": shares_outstanding,
                "short_percent_float": short_percent_float,
                "short_ratio": short_ratio,
            }
        )

    metadata = pandas_module.DataFrame(rows)
    if metadata.empty:
        raise SystemExit("Impossible de recuperer les metadata Yahoo des tickers.")

    missing_tickers = sorted(set(constituents["ticker"]) - set(metadata["ticker"]))
    if missing_tickers:
        missing_label = ", ".join(missing_tickers)
        raise SystemExit(f"Tickers manquants dans les metadata Yahoo: {missing_label}")

    metadata = metadata.drop_duplicates(subset=["ticker"]).copy()
    metadata["sector"] = metadata["sector"].replace("", "unknown").fillna("unknown")
    metadata["industry"] = metadata["industry"].replace("", "unknown").fillna("unknown")
    metadata["sector_bucket"] = metadata["sector_bucket"].replace("", "unknown").fillna("unknown")
    metadata.loc[~metadata["sector_bucket"].isin(SECTOR_BUCKETS), "sector_bucket"] = "unknown"
    return metadata


def build_exogenous_daily_panel(
    pandas_module: Any,
    dataset: Any,
    yahoo_macro: Any,
    fred_macro: Any,
    allow_missing_fred: bool = False,
) -> Any:
    calendar = dataset[["date"]].copy()
    calendar["date"] = pandas_module.to_datetime(calendar["date"], errors="coerce")
    calendar = calendar.dropna(subset=["date"]).drop_duplicates().sort_values("date").reset_index(drop=True)

    yahoo_macro = yahoo_macro.copy()
    yahoo_macro["date"] = pandas_module.to_datetime(yahoo_macro["date"], errors="coerce")
    yahoo_macro = yahoo_macro.dropna(subset=["date"]).copy()

    fred_macro = fred_macro.copy()
    fred_macro["date"] = pandas_module.to_datetime(fred_macro["date"], errors="coerce")
    fred_macro = fred_macro.dropna(subset=["date"]).copy()

    exo = calendar.merge(yahoo_macro, on="date", how="left")
    exo = exo.merge(fred_macro, on="date", how="left")
    exo = exo.sort_values("date").reset_index(drop=True)

    numeric_cols = list(YAHOO_MACRO_TICKERS.values()) + list(FRED_SERIES.values())
    for col in numeric_cols:
        if col not in exo.columns:
            exo[col] = pandas_module.NA
        exo[col] = pandas_module.to_numeric(exo[col], errors="coerce")

    # Policy locked: forward-fill only, no backward-fill.
    exo[numeric_cols] = exo[numeric_cols].ffill()

    required_numeric_cols = list(YAHOO_MACRO_TICKERS.values())
    if not allow_missing_fred:
        required_numeric_cols += list(FRED_SERIES.values())

    fully_missing = [
        col for col in required_numeric_cols if exo[col].notna().sum() == 0
    ]
    if fully_missing:
        cols_label = ", ".join(fully_missing)
        raise SystemExit(
            f"Colonnes exogenes completement vides apres merge/ffill: {cols_label}."
        )
    return exo


def enrich_dataset(
    pandas_module: Any,
    dataset: Any,
    exogenous_daily: Any,
    metadata: Any,
) -> Any:
    expected_rows = len(dataset)
    out = dataset.merge(exogenous_daily, on="date", how="left", validate="many_to_one")
    out = out.merge(metadata, on="ticker", how="left", validate="many_to_one")

    if len(out) != expected_rows:
        raise SystemExit(
            "Le nombre de lignes a change apres enrichissement "
            "(integrite many_to_one rompue)."
        )

    required_columns = [
        "market_close",
        "vix_close",
        "oil_close",
        "dxy_close",
        "gold_close",
        "ust2y",
        "ust10y",
        "frt2y",
        "frt10y",
        "credit_spread",
        "sector",
        "industry",
        "sector_bucket",
        "market_cap",
        "book_value",
        "trailing_eps",
        "shares_outstanding",
        "short_percent_float",
        "short_ratio",
    ]
    missing = [col for col in required_columns if col not in out.columns]
    if missing:
        missing_label = ", ".join(missing)
        raise SystemExit(f"Colonnes manquantes apres enrichissement: {missing_label}")

    for col in required_columns:
        if col in {"sector", "industry", "sector_bucket"}:
            continue
        out[col] = pandas_module.to_numeric(out[col], errors="coerce")

    out["sector"] = out["sector"].replace("", "unknown").fillna("unknown")
    out["industry"] = out["industry"].replace("", "unknown").fillna("unknown")
    out["sector_bucket"] = out["sector_bucket"].replace("", "unknown").fillna("unknown")
    out.loc[~out["sector_bucket"].isin(SECTOR_BUCKETS), "sector_bucket"] = "unknown"

    if out["sector"].eq("unknown").all():
        raise SystemExit("Tous les secteurs sont `unknown`: metadata Yahoo inexploitable.")

    if out.duplicated(subset=["date", "ticker"]).any():
        raise SystemExit("Cle logique non unique apres enrichissement: (`date`,`ticker`).")

    ordered_columns = [
        "date",
        "ticker",
        "company",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits",
        "market_close",
        "vix_close",
        "oil_close",
        "dxy_close",
        "gold_close",
        "ust2y",
        "ust10y",
        "frt2y",
        "frt10y",
        "credit_spread",
        "sector",
        "industry",
        "sector_bucket",
        "market_cap",
        "book_value",
        "trailing_eps",
        "shares_outstanding",
        "short_percent_float",
        "short_ratio",
    ]

    return out[ordered_columns].sort_values(["date", "ticker"]).reset_index(drop=True)


def write_csv(dataset: Any, path: Path) -> int:
    dataset.to_csv(path, index=False)
    return path.stat().st_size


def write_parquet(dataset: Any, path: Path) -> int:
    dataset.to_parquet(path, index=False)
    return path.stat().st_size


def save_dataset(dataset: Any, output_base: Path, preferred_format: str) -> Path:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_base.with_suffix(".csv")
    parquet_path = output_base.with_suffix(".parquet")

    if preferred_format == "csv":
        write_csv(dataset, csv_path)
        if parquet_path.exists():
            parquet_path.unlink()
        return csv_path

    if preferred_format == "parquet":
        try:
            write_parquet(dataset, parquet_path)
        except Exception as exc:
            raise SystemExit(
                "Impossible d'ecrire en parquet. "
                "Installe `pyarrow` ou `fastparquet`."
            ) from exc

        if csv_path.exists():
            csv_path.unlink()
        return parquet_path

    csv_size = write_csv(dataset, csv_path)

    try:
        parquet_size = write_parquet(dataset, parquet_path)
    except Exception as exc:
        if parquet_path.exists():
            parquet_path.unlink()
        print(
            "Parquet indisponible, conservation du CSV. "
            f"Raison: {exc}"
        )
        return csv_path

    if parquet_size < csv_size:
        csv_path.unlink()
        return parquet_path

    parquet_path.unlink()
    return csv_path


def main() -> None:
    args = parse_args()
    pandas_module, yfinance_module = load_dependencies()

    start = validate_date(args.start)
    end_for_yf = resolve_end_date_for_yfinance(args.end)
    end_inclusive = resolve_end_date_inclusive(args.end)
    if args.end is not None:
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_inclusive, "%Y-%m-%d").date()
        if end_date < start_date:
            raise SystemExit("`--end` doit etre posterieure ou egale a `--start`.")

    output_base = output_base_path(args.output)
    constituents = get_constituents_frame(pandas_module)

    dataset = download_history(
        pandas_module=pandas_module,
        yfinance_module=yfinance_module,
        constituents=constituents,
        start=start,
        end=end_for_yf,
        batch_size=args.batch_size,
    )

    yahoo_macro = download_yahoo_macro_series(
        pandas_module=pandas_module,
        yfinance_module=yfinance_module,
        start=start,
        end=end_for_yf,
    )
    try:
        fred_macro = download_fred_macro_series(
            pandas_module=pandas_module,
            start=start,
            end_inclusive=end_inclusive,
            timeout=args.fred_timeout,
            retries=args.fred_retries,
        )
    except FredDownloadError as exc:
        if not args.allow_missing_fred:
            raise SystemExit(str(exc)) from exc
        print(f"[macro-fred] Avertissement: {exc}")
        print("[macro-fred] Colonnes FRED remplies avec des valeurs manquantes.")
        fred_macro = build_missing_fred_macro_frame(
            pandas_module=pandas_module,
            dataset=dataset,
        )
    metadata = download_yahoo_snapshot_metadata(
        pandas_module=pandas_module,
        yfinance_module=yfinance_module,
        constituents=constituents,
    )

    exogenous_daily = build_exogenous_daily_panel(
        pandas_module=pandas_module,
        dataset=dataset,
        yahoo_macro=yahoo_macro,
        fred_macro=fred_macro,
        allow_missing_fred=args.allow_missing_fred,
    )
    dataset = enrich_dataset(
        pandas_module=pandas_module,
        dataset=dataset,
        exogenous_daily=exogenous_daily,
        metadata=metadata,
    )

    saved_path = save_dataset(dataset, output_base, args.format)
    ticker_count = dataset["ticker"].nunique()
    print(
        f"Dataset enregistre dans {saved_path} "
        f"({len(dataset):,} lignes, {ticker_count} tickers)."
    )


if __name__ == "__main__":
    main()
