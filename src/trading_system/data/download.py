#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import time
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from trading_system.paths import processed_data_dir

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_BASE = processed_data_dir() / "cac40_daily"

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

TREASURY_YIELD_COLUMNS = {
    "2 Yr": "ust2y",
    "10 Yr": "ust10y",
}

OECD_FINMARK_SERIES = {
    "IR3TIB": "frt2y",
    "IRLT": "frt10y",
}

CREDIT_SPREAD_FRED_SERIES_ID = "BAMLH0A0HYM2"
CREDIT_SPREAD_COLUMN = "credit_spread"
RATE_MACRO_COLUMNS = list(TREASURY_YIELD_COLUMNS.values()) + list(OECD_FINMARK_SERIES.values())
OPTIONAL_MACRO_COLUMNS = [CREDIT_SPREAD_COLUMN]

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


class MacroDownloadError(RuntimeError):
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
        help=(
            "Timeout reseau en secondes pour les sources macro externes "
            "(Treasury/OECD/FRED optionnel). Defaut: 120."
        ),
    )
    parser.add_argument(
        "--fred-retries",
        type=int,
        default=3,
        help="Nombre de tentatives par source macro externe. Defaut: 3.",
    )
    parser.add_argument(
        "--allow-missing-fred",
        action="store_true",
        help=(
            "Compatibilite: credit_spread est desormais optionnel et rempli "
            "avec des valeurs manquantes si FRED est inaccessible."
        ),
    )
    parser.add_argument(
        "--include-credit-spread",
        action="store_true",
        help=(
            "Tente d'ajouter credit_spread depuis FRED/BAMLH0A0HYM2. "
            "Par defaut, cette colonne optionnelle est creee vide."
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


def _read_csv_url_with_retries(
    pandas_module: Any,
    url: str,
    label: str,
    timeout: int,
    retries: int,
    headers: dict[str, str] | None = None,
) -> Any:
    if timeout <= 0:
        raise SystemExit("--fred-timeout doit etre superieur a 0.")
    if retries <= 0:
        raise SystemExit("--fred-retries doit etre superieur a 0.")

    last_error: Exception | None = None
    request_headers = {
        "Accept": "text/csv,*/*;q=0.8",
        "User-Agent": "Mozilla/5.0",
    }
    if headers:
        request_headers.update(headers)

    for attempt in range(1, retries + 1):
        request = Request(url, headers=request_headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                return pandas_module.read_csv(response)
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                raise MacroDownloadError(
                    f"Impossible de telecharger `{label}` "
                    f"apres {retries} tentative(s): {exc}"
                ) from exc
            wait_seconds = min(2 ** (attempt - 1), 10)
            print(
                f"  - {label}: tentative {attempt}/{retries} echouee "
                f"({exc}); nouvelle tentative dans {wait_seconds}s"
            )
            time.sleep(wait_seconds)

    raise MacroDownloadError(f"Impossible de telecharger `{label}`: {last_error}")


def _read_text_url_with_retries(
    url: str,
    label: str,
    timeout: int,
    retries: int,
    headers: dict[str, str] | None = None,
) -> str:
    if timeout <= 0:
        raise SystemExit("--fred-timeout doit etre superieur a 0.")
    if retries <= 0:
        raise SystemExit("--fred-retries doit etre superieur a 0.")

    last_error: Exception | None = None
    request_headers = {
        "Accept": "text/html,*/*;q=0.8",
        "User-Agent": "Mozilla/5.0",
    }
    if headers:
        request_headers.update(headers)

    for attempt in range(1, retries + 1):
        request = Request(url, headers=request_headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                raise MacroDownloadError(
                    f"Impossible de telecharger `{label}` "
                    f"apres {retries} tentative(s): {exc}"
                ) from exc
            wait_seconds = min(2 ** (attempt - 1), 10)
            print(
                f"  - {label}: tentative {attempt}/{retries} echouee "
                f"({exc}); nouvelle tentative dans {wait_seconds}s"
            )
            time.sleep(wait_seconds)

    raise MacroDownloadError(f"Impossible de telecharger `{label}`: {last_error}")


class _FirstHtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._in_table = False
        self._finished = False
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._finished:
            return
        if tag == "table" and not self._in_table:
            self._in_table = True
            return
        if not self._in_table:
            return
        if tag == "tr":
            self._current_row = []
        elif tag in {"th", "td"}:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._in_table and self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self._in_table or self._finished:
            return
        if tag in {"th", "td"} and self._current_row is not None:
            text = " ".join("".join(self._current_cell or []).split())
            self._current_row.append(text)
            self._current_cell = None
        elif tag == "tr":
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = None
        elif tag == "table":
            self._in_table = False
            self._finished = True


def _parse_first_html_table(pandas_module: Any, html_text: str) -> Any:
    # Treasury.gov exposes the data as a public HTML table. The CSV endpoint is
    # easier to parse, but can answer 403 to non-browser clients, so we parse the
    # first table to keep this source free and usable without an API key.
    parser = _FirstHtmlTableParser()
    parser.feed(html_text)
    rows = parser.rows
    if len(rows) < 2:
        raise MacroDownloadError("Aucun tableau exploitable dans la reponse HTML.")

    header = rows[0]
    body = []
    for row in rows[1:]:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        body.append(row[: len(header)])

    return pandas_module.DataFrame(body, columns=header)


def _estimate_treasury_start_page(start: str) -> int:
    try:
        start_year = datetime.strptime(start, "%Y-%m-%d").year
    except ValueError:
        return 0
    estimated_rows_before_start = max(0, start_year - 1990) * 252
    return max(0, estimated_rows_before_start // 300 - 2)


def _shift_month_period(date_str: str, month_delta: int) -> str:
    date_value = datetime.strptime(validate_date(date_str), "%Y-%m-%d").date()
    month_index = date_value.year * 12 + date_value.month - 1 + month_delta
    year = month_index // 12
    month = month_index % 12 + 1
    return f"{year:04d}-{month:02d}"


def download_treasury_yield_series(
    pandas_module: Any,
    start: str,
    end_inclusive: str,
    timeout: int,
    retries: int,
) -> Any:
    print("[macro-treasury] Telechargement U.S. Treasury -> ust2y, ust10y")
    base_url = (
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
        "TextView?type=daily_treasury_yield_curve"
    )
    start_ts = pandas_module.to_datetime(start, errors="coerce")
    end_ts = pandas_module.to_datetime(end_inclusive, errors="coerce")
    page = _estimate_treasury_start_page(start)
    frames = []

    while True:
        url = f"{base_url}&page={page}"
        try:
            html_text = _read_text_url_with_retries(
                url=url,
                label=f"U.S. Treasury yield curve page {page}",
                timeout=timeout,
                retries=retries,
                headers={"Referer": "https://home.treasury.gov/"},
            )
            page_frame = _parse_first_html_table(pandas_module, html_text)
        except MacroDownloadError:
            if frames:
                break
            raise

        if page_frame.empty:
            break

        page_frame = page_frame.rename(
            columns={col: str(col).strip() for col in page_frame.columns}
        )
        date_col = next(
            (col for col in page_frame.columns if col.lower() == "date"),
            None,
        )
        if date_col is None:
            raise MacroDownloadError("Reponse U.S. Treasury sans colonne Date.")

        page_dates = pandas_module.to_datetime(page_frame[date_col], errors="coerce")
        if page_dates.notna().any() and page_dates.max() >= start_ts:
            frames.append(page_frame)
        if page_dates.notna().any() and page_dates.max() >= end_ts:
            break
        if len(page_frame) < 300:
            break

        page += 1
        if page > 200:
            raise MacroDownloadError("Pagination U.S. Treasury anormalement longue.")

    if not frames:
        raise MacroDownloadError("Aucune page U.S. Treasury exploitable.")

    frame = pandas_module.concat(frames, ignore_index=True)

    frame = frame.rename(columns={col: str(col).strip() for col in frame.columns})
    date_col = next((col for col in frame.columns if col.lower() == "date"), None)
    normalized_columns = {str(col).strip().lower(): col for col in frame.columns}

    missing_source_cols = [
        source_col
        for source_col in TREASURY_YIELD_COLUMNS
        if source_col.lower() not in normalized_columns
    ]
    if date_col is None or missing_source_cols:
        received = ", ".join(str(col) for col in frame.columns[:12]) or "aucune"
        raise MacroDownloadError(
            "Reponse U.S. Treasury invalide. "
            f"Colonnes manquantes: {missing_source_cols}. "
            f"Colonnes recues: {received}."
        )

    out = pandas_module.DataFrame()
    out["date"] = pandas_module.to_datetime(frame[date_col], errors="coerce")
    for source_col, target_col in TREASURY_YIELD_COLUMNS.items():
        actual_col = normalized_columns[source_col.lower()]
        out[target_col] = pandas_module.to_numeric(frame[actual_col], errors="coerce")

    out = out.dropna(subset=["date"]).copy()
    out = out[(out["date"] >= start_ts) & (out["date"] <= end_ts)].copy()
    out = out.sort_values("date").reset_index(drop=True)
    if out.empty:
        raise MacroDownloadError("Aucune donnee U.S. Treasury apres filtrage date.")
    return out


def _download_oecd_finmark_series(
    pandas_module: Any,
    measure: str,
    target_col: str,
    start: str,
    end_inclusive: str,
    timeout: int,
    retries: int,
) -> Any:
    # OECD can return HTTP 404 instead of an empty CSV when the requested start
    # month has no published observation yet, especially for IRLT. Query a few
    # months before --start; build_exogenous_daily_panel then forward-fills onto
    # the CAC40 trading calendar and filters back to the requested window.
    start_period = _shift_month_period(start, -3)
    end_period = end_inclusive[:7]
    path = f"@DF_FINMARK,4.0/FRA.M.{measure}.PA....."
    query = urlencode(
        {
            "startPeriod": start_period,
            "endPeriod": end_period,
            "dimensionAtObservation": "AllDimensions",
        }
    )
    url = f"https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES{path}?{query}"
    label = f"OECD {measure}"
    print(f"[macro-oecd] Telechargement {measure} -> {target_col}")
    frame = _read_csv_url_with_retries(
        pandas_module=pandas_module,
        url=url,
        label=label,
        timeout=timeout,
        retries=retries,
        headers={"Accept": "application/vnd.sdmx.data+csv; charset=utf-8,*/*;q=0.8"},
    )

    required_cols = {"TIME_PERIOD", "OBS_VALUE"}
    if not required_cols.issubset(frame.columns):
        received = ", ".join(str(col) for col in frame.columns[:12]) or "aucune"
        raise MacroDownloadError(
            f"Reponse OECD invalide pour `{measure}`. Colonnes recues: {received}."
        )

    if "MEASURE" in frame.columns:
        frame = frame[frame["MEASURE"].astype(str).eq(measure)].copy()
    if "REF_AREA" in frame.columns:
        frame = frame[frame["REF_AREA"].astype(str).eq("FRA")].copy()
    if "FREQ" in frame.columns:
        frame = frame[frame["FREQ"].astype(str).eq("M")].copy()

    out = pandas_module.DataFrame()
    out["date"] = pandas_module.to_datetime(
        frame["TIME_PERIOD"].astype(str) + "-01",
        errors="coerce",
    )
    out[target_col] = pandas_module.to_numeric(frame["OBS_VALUE"], errors="coerce")
    out = out.dropna(subset=["date"]).copy()
    out = out.sort_values("date").reset_index(drop=True)
    if out.empty:
        raise MacroDownloadError(f"Aucune donnee OECD pour `{measure}`.")
    return out


def download_oecd_rate_series(
    pandas_module: Any,
    start: str,
    end_inclusive: str,
    timeout: int,
    retries: int,
) -> Any:
    merged = None

    for measure, target_col in OECD_FINMARK_SERIES.items():
        frame = _download_oecd_finmark_series(
            pandas_module=pandas_module,
            measure=measure,
            target_col=target_col,
            start=start,
            end_inclusive=end_inclusive,
            timeout=timeout,
            retries=retries,
        )

        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="date", how="outer")

    if merged is None or merged.empty:
        raise MacroDownloadError("Aucune serie OECD n'a ete telechargee.")

    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def download_rate_macro_series(
    pandas_module: Any,
    start: str,
    end_inclusive: str,
    timeout: int,
    retries: int,
) -> Any:
    treasury = download_treasury_yield_series(
        pandas_module=pandas_module,
        start=start,
        end_inclusive=end_inclusive,
        timeout=timeout,
        retries=retries,
    )
    oecd = download_oecd_rate_series(
        pandas_module=pandas_module,
        start=start,
        end_inclusive=end_inclusive,
        timeout=timeout,
        retries=retries,
    )
    return (
        treasury.merge(oecd, on="date", how="outer")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _download_fred_series(
    pandas_module: Any,
    series_id: str,
    start: str,
    end_inclusive: str,
    timeout: int,
    retries: int,
) -> Any:
    query = urlencode({"id": series_id, "cosd": start, "coed": end_inclusive})
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{query}"
    try:
        frame = _read_csv_url_with_retries(
            pandas_module=pandas_module,
            url=url,
            label=f"FRED {series_id}",
            timeout=timeout,
            retries=retries,
        )
    except MacroDownloadError as exc:
        raise FredDownloadError(str(exc)) from exc

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


def download_credit_spread_series(
    pandas_module: Any,
    start: str,
    end_inclusive: str,
    timeout: int,
    retries: int,
) -> Any:
    print(
        f"[macro-fred] Telechargement optionnel "
        f"{CREDIT_SPREAD_FRED_SERIES_ID} -> {CREDIT_SPREAD_COLUMN}"
    )
    frame = _download_fred_series(
        pandas_module=pandas_module,
        series_id=CREDIT_SPREAD_FRED_SERIES_ID,
        start=start,
        end_inclusive=end_inclusive,
        timeout=timeout,
        retries=retries,
    )
    if frame.empty:
        raise FredDownloadError(
            f"Aucune donnee FRED pour la serie `{CREDIT_SPREAD_FRED_SERIES_ID}`."
        )
    return frame.rename(columns={"value": CREDIT_SPREAD_COLUMN})


def build_missing_credit_spread_frame(pandas_module: Any, dataset: Any) -> Any:
    # FRED was the unstable dependency in the original pipeline. Keep the column
    # for dataset schema compatibility, but make it all-missing unless the user
    # explicitly opts into --include-credit-spread.
    frame = (
        dataset[["date"]]
        .drop_duplicates()
        .sort_values("date")
        .reset_index(drop=True)
    )
    frame[CREDIT_SPREAD_COLUMN] = pandas_module.NA
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
    rate_macro: Any,
    credit_macro: Any,
) -> Any:
    calendar = dataset[["date"]].copy()
    calendar["date"] = pandas_module.to_datetime(calendar["date"], errors="coerce")
    calendar = (
        calendar.dropna(subset=["date"])
        .drop_duplicates()
        .sort_values("date")
        .reset_index(drop=True)
    )
    calendar_dates = calendar["date"]

    yahoo_macro = yahoo_macro.copy()
    yahoo_macro["date"] = pandas_module.to_datetime(yahoo_macro["date"], errors="coerce")
    yahoo_macro = yahoo_macro.dropna(subset=["date"]).copy()

    rate_macro = rate_macro.copy()
    rate_macro["date"] = pandas_module.to_datetime(rate_macro["date"], errors="coerce")
    rate_macro = rate_macro.dropna(subset=["date"]).copy()

    credit_macro = credit_macro.copy()
    credit_macro["date"] = pandas_module.to_datetime(credit_macro["date"], errors="coerce")
    credit_macro = credit_macro.dropna(subset=["date"]).copy()

    # Rates have mixed calendars: Yahoo/Treasury are daily, OECD is monthly and
    # often lands on a non-trading day. Merge on the union of dates first so the
    # monthly OECD observations can be carried forward, then return to the CAC40
    # trading calendar.
    exo = calendar.merge(yahoo_macro, on="date", how="outer")
    exo = exo.merge(rate_macro, on="date", how="outer")
    exo = exo.merge(credit_macro, on="date", how="outer")
    exo = exo.sort_values("date").reset_index(drop=True)

    numeric_cols = (
        list(YAHOO_MACRO_TICKERS.values())
        + RATE_MACRO_COLUMNS
        + OPTIONAL_MACRO_COLUMNS
    )
    for col in numeric_cols:
        if col not in exo.columns:
            exo[col] = pandas_module.NA
        exo[col] = pandas_module.to_numeric(exo[col], errors="coerce")

    # Policy locked: forward-fill only, no backward-fill.
    exo[numeric_cols] = exo[numeric_cols].ffill()
    exo = exo[exo["date"].isin(calendar_dates)].copy()

    required_numeric_cols = list(YAHOO_MACRO_TICKERS.values()) + RATE_MACRO_COLUMNS

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

    for col in OPTIONAL_MACRO_COLUMNS:
        if col not in out.columns:
            out[col] = pandas_module.NA

    for col in required_columns + OPTIONAL_MACRO_COLUMNS:
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
    rate_macro = download_rate_macro_series(
        pandas_module=pandas_module,
        start=start,
        end_inclusive=end_inclusive,
        timeout=args.fred_timeout,
        retries=args.fred_retries,
    )

    if args.include_credit_spread:
        try:
            credit_macro = download_credit_spread_series(
                pandas_module=pandas_module,
                start=start,
                end_inclusive=end_inclusive,
                timeout=args.fred_timeout,
                retries=args.fred_retries,
            )
        except FredDownloadError as exc:
            print(f"[macro-fred] Avertissement: {exc}")
            print("[macro-fred] credit_spread rempli avec des valeurs manquantes.")
            credit_macro = build_missing_credit_spread_frame(
                pandas_module=pandas_module,
                dataset=dataset,
            )
    else:
        credit_macro = build_missing_credit_spread_frame(
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
        rate_macro=rate_macro,
        credit_macro=credit_macro,
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
