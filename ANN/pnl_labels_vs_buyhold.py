from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd


WINDOW = 20
CAPITAL = 10_000.0


def load_multi_ticker_module():
    ann_dir = Path(__file__).resolve().parent
    module_path = ann_dir / "ANN_multi-ticker.py"

    if not module_path.exists():
        raise FileNotFoundError(f"Module introuvable: {module_path}")

    spec = importlib.util.spec_from_file_location("ann_multi_ticker_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger le module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_no_ticker_mixing(df):
    required_cols = {"ticker", "date", "adj_close", "Label_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour l'evaluation: {sorted(missing)}")

    out = df.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    if out.empty:
        raise ValueError("Dataset vide apres chargement/labelling.")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("Des dates invalides ont ete detectees.")

    # Garde-fou: chaque ticker doit etre dans un seul bloc contigu.
    block_starts = out["ticker"].ne(out["ticker"].shift(fill_value=out["ticker"].iloc[0]))
    ticker_starts = out.loc[block_starts, "ticker"]
    if ticker_starts.duplicated().any():
        raise ValueError("Melange inter-ticker detecte: blocs ticker non contigus.")

    bad_tickers = []
    for ticker, group in out.groupby("ticker", sort=False):
        if not group["date"].is_monotonic_increasing:
            bad_tickers.append(ticker)
    if bad_tickers:
        raise ValueError(f"Dates non monotones pour certains tickers: {bad_tickers[:5]}")

    return out


def main():
    mod = load_multi_ticker_module()
    read_parquet_dataset = mod.read_parquet_dataset
    labelling_all = mod.labelling_all
    evaluate_strategy_vs_buy_hold = mod.evaluate_strategy_vs_buy_hold

    data_dir = Path(__file__).resolve().parent / "datasets" / "cac40_daily.parquet"
    df = read_parquet_dataset(data_dir)
    df = labelling_all(df, WINDOW)
    df = ensure_no_ticker_mixing(df)

    pred_labels = df["Label_id"].to_numpy(dtype=np.int64)
    results = evaluate_strategy_vs_buy_hold(
        df,
        pred_labels,
        initial_capital=CAPITAL,
        price_col="adj_close",
    )
    print(results)


if __name__ == "__main__":
    main()

