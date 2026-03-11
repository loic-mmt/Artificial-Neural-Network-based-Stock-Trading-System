#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


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
            "Telecharge les donnees journalieres OHLCV du CAC 40 depuis Yahoo Finance "
            "et conserve automatiquement le plus petit format entre csv et parquet."
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
        print(f"[{batch_index}/{len(batches)}] Telechargement: {batch_label}")

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
        raise SystemExit("Aucune donnee n'a ete telechargee.")

    dataset = pandas_module.concat(frames, ignore_index=True)
    dataset = dataset.sort_values(["date", "ticker"]).reset_index(drop=True)
    return dataset


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
    end = resolve_end_date_for_yfinance(args.end)
    if end is not None:
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date() - timedelta(days=1)
        if end_date < start_date:
            raise SystemExit("`--end` doit etre posterieure ou egale a `--start`.")
    output_base = output_base_path(args.output)
    constituents = get_constituents_frame(pandas_module)

    dataset = download_history(
        pandas_module=pandas_module,
        yfinance_module=yfinance_module,
        constituents=constituents,
        start=start,
        end=end,
        batch_size=args.batch_size,
    )

    saved_path = save_dataset(dataset, output_base, args.format)
    print(
        f"Dataset enregistre dans {saved_path} "
        f"({len(dataset):,} lignes, {constituents['ticker'].nunique()} tickers)."
    )


if __name__ == "__main__":
    main()
