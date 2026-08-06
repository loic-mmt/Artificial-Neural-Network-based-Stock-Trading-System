from __future__ import annotations

import numpy as np
import pandas as pd


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
SECTOR_ONE_HOT_FEATURES = [f"sector_{bucket}" for bucket in SECTOR_BUCKETS]

# Feature list for model import.
MARKET_FEATURE_COLUMNS = [
    # Multi-horizon returns
    "ret_2",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "ret_120",
    # Trend / distance
    "dist_ma_5",
    "dist_ma_20",
    "dist_ma_60",
    "ma_ratio_5_20",
    "ma_ratio_20_60",
    "bollinger_z_20",
    "dist_high_20",
    "dist_low_20",
    "drawdown_20",
    "drawdown_60",
    # Volatility
    "vol_5",
    "vol_20",
    "vol_60",
    "vol_ratio_5_20",
    "vol_ratio_20_60",
    "atr_norm",
    "parkinson_vol_20",
    "garman_klass_vol_20",
    "rogers_satchell_vol_20",
    # Liquidity / volume
    "dollar_volume",
    "amihud_illiq",
    "volume_z_20",
    "ret_x_volume_shock",
    "ret_x_illiq",
    # Serial dependence / regime
    "autocorr_5",
    "autocorr_20",
    "variance_ratio_20",
    "trend_regime",
    "vol_regime",
    # Market
    "market_ret_1",
    "market_ret_5",
    "market_ret_20",
    "market_vol_20",
    "market_drawdown_20",
    "beta_market_60",
    "corr_market_60",
    "vix_level",
    "vix_ret_5",
    "market_breadth",
    # Sector
    "sector_ret_1",
    "sector_ret_5",
    "sector_ret_20",
    "sector_vol_20",
    "sector_drawdown_20",
    "stock_minus_sector_ret_5",
    "stock_minus_sector_ret_20",
    "beta_sector_60",
    "corr_sector_60",
    "sector_breadth_up",
    # Rates / credit
    "ust2y",
    "ust10y",
    "frt2y",
    "frt10y",
    "spread_2s10s",
    "delta_10y_5d",
    # credit_spread depends on optional FRED access in data.py, so model
    # feature engineering must not require it to complete.
    # Commodities / FX
    "oil_ret_5",
    "dxy_ret_5",
    "gold_ret_5",
    # Firm / micro
    "market_cap_log",
    "book_to_market",
    "earnings_yield",
    "turnover_ratio",
    "short_interest",
    # Relative ranks
    "ret_20_rank_sector",
    "vol_20_rank_sector",
    "rsi_rank_sector",
    "drawdown_rank_sector",
    # Situational
    "day_of_week",
    "week",
    "month",
    *SECTOR_ONE_HOT_FEATURES,
]


FEATURE_COLUMNS = MARKET_FEATURE_COLUMNS
# Temporary compatibility alias. New code should use MARKET_FEATURE_COLUMNS.
features = MARKET_FEATURE_COLUMNS
EPS = 1e-12

EXTERNAL_REQUIRED_COLUMNS = [
    "market_close",
    "vix_close",
    "oil_close",
    "dxy_close",
    "gold_close",
    "ust2y",
    "ust10y",
    "frt2y",
    "frt10y",
    "market_cap",
    "book_value",
    "trailing_eps",
    "shares_outstanding",
]


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = denominator.where(denominator.abs() > EPS, np.nan)
    return numerator / den


def _require_columns(frame: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns for {context}: {missing}")


def _normalize_sector_bucket(sector: pd.Series, industry: pd.Series | None = None) -> pd.Series:
    sector_text = sector.fillna("unknown").astype(str).str.strip().str.lower()
    if industry is not None:
        industry_text = industry.fillna("unknown").astype(str).str.strip().str.lower()
    else:
        industry_text = pd.Series("unknown", index=sector.index, dtype="object")
    both = (sector_text + " " + industry_text).str.strip()

    out = pd.Series("unknown", index=sector.index, dtype="object")
    out.loc[both.str.contains("bank|banque", regex=True)] = "banks"
    out.loc[both.str.contains("insurance|assurance", regex=True)] = "insurance"
    out.loc[both.str.contains("energy|oil|gas", regex=True)] = "energy"
    out.loc[both.str.contains("material|metals|steel", regex=True)] = "materials"
    out.loc[both.str.contains("health|pharma|biotech", regex=True)] = "healthcare"
    out.loc[both.str.contains("technology|software|semiconductor", regex=True)] = "technology"
    out.loc[both.str.contains("telecom|communication", regex=True)] = "telecom"
    out.loc[both.str.contains("utility", regex=True)] = "utilities"
    out.loc[both.str.contains("real estate|reit", regex=True)] = "real_estate"
    out.loc[both.str.contains("consumer defensive|household", regex=True)] = "consumer_defensive"
    out.loc[both.str.contains("consumer cyclical|luxury|retail", regex=True)] = "consumer_cyclical"
    out.loc[both.str.contains("industrial|aerospace|transport", regex=True)] = "industrials"
    out.loc[both.str.contains("financial services|asset management", regex=True)] = "financial_services"
    out.loc[~out.isin(SECTOR_BUCKETS)] = "unknown"
    return out


def _compute_rsi(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = _safe_div(avg_gain, avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_one_symbol_base(
    frame: pd.DataFrame,
    date_col: str,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    adj_close_col: str,
    volume_col: str,
) -> pd.DataFrame:
    out = frame.copy()
    out = out.sort_values(date_col).copy()

    price_col = adj_close_col if adj_close_col in out.columns else close_col
    required = [open_col, high_col, low_col, close_col, volume_col, price_col]
    _require_columns(out, required, "base feature engineering")

    open_p = out[open_col].astype(float).clip(lower=EPS)
    high_p = out[high_col].astype(float).clip(lower=EPS)
    low_p = out[low_col].astype(float).clip(lower=EPS)
    close_p = out[close_col].astype(float).clip(lower=EPS)
    price = out[price_col].astype(float).clip(lower=EPS)
    volume = out[volume_col].astype(float).clip(lower=0.0)

    ret_1 = price.pct_change()
    log_ret_1 = np.log(price).diff()

    out["_ret_1"] = ret_1
    out["_log_ret_1"] = log_ret_1
    out["_rsi_14"] = _compute_rsi(price, window=14)

    # Multi-horizon returns
    for horizon in (2, 3, 5, 10, 20, 60, 120):
        out[f"ret_{horizon}"] = price.pct_change(horizon)

    # Trend / distance
    ma_5 = price.rolling(5).mean()
    ma_20 = price.rolling(20).mean()
    ma_60 = price.rolling(60).mean()
    std_20 = price.rolling(20).std(ddof=0)

    out["dist_ma_5"] = _safe_div(price - ma_5, ma_5)
    out["dist_ma_20"] = _safe_div(price - ma_20, ma_20)
    out["dist_ma_60"] = _safe_div(price - ma_60, ma_60)
    out["ma_ratio_5_20"] = _safe_div(ma_5, ma_20) - 1.0
    out["ma_ratio_20_60"] = _safe_div(ma_20, ma_60) - 1.0
    out["bollinger_z_20"] = _safe_div(price - ma_20, std_20)

    rolling_high_20 = high_p.rolling(20).max()
    rolling_low_20 = low_p.rolling(20).min()
    out["dist_high_20"] = _safe_div(price - rolling_high_20, rolling_high_20)
    out["dist_low_20"] = _safe_div(price - rolling_low_20, rolling_low_20)
    out["drawdown_20"] = _safe_div(price, price.rolling(20).max()) - 1.0
    out["drawdown_60"] = _safe_div(price, price.rolling(60).max()) - 1.0

    # Volatility
    out["vol_5"] = log_ret_1.rolling(5).std(ddof=0)
    out["vol_20"] = log_ret_1.rolling(20).std(ddof=0)
    out["vol_60"] = log_ret_1.rolling(60).std(ddof=0)
    out["vol_ratio_5_20"] = _safe_div(out["vol_5"], out["vol_20"])
    out["vol_ratio_20_60"] = _safe_div(out["vol_20"], out["vol_60"])

    prev_close = price.shift(1)
    true_range = pd.concat(
        [
            (high_p - low_p).abs(),
            (high_p - prev_close).abs(),
            (low_p - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_14 = true_range.rolling(14).mean()
    out["atr_norm"] = _safe_div(atr_14, price)

    log_hl = np.log(_safe_div(high_p, low_p))
    parkinson_var = log_hl.pow(2).rolling(20).mean() / (4.0 * np.log(2.0))
    out["parkinson_vol_20"] = np.sqrt(parkinson_var.clip(lower=0.0))

    log_co = np.log(_safe_div(close_p, open_p))
    gk_var = (
        0.5 * log_hl.pow(2)
        - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)
    ).rolling(20).mean()
    out["garman_klass_vol_20"] = np.sqrt(gk_var.clip(lower=0.0))

    rs_var = (
        np.log(_safe_div(high_p, close_p)) * np.log(_safe_div(high_p, open_p))
        + np.log(_safe_div(low_p, close_p)) * np.log(_safe_div(low_p, open_p))
    ).rolling(20).mean()
    out["rogers_satchell_vol_20"] = np.sqrt(rs_var.clip(lower=0.0))

    # Liquidity / volume
    out["dollar_volume"] = price * volume
    amihud_raw = _safe_div(ret_1.abs(), out["dollar_volume"])
    out["amihud_illiq"] = amihud_raw.rolling(20).mean()
    vol_mean_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std(ddof=0)
    out["volume_z_20"] = _safe_div(volume - vol_mean_20, vol_std_20)
    out["ret_x_volume_shock"] = ret_1 * out["volume_z_20"]
    out["ret_x_illiq"] = ret_1 * out["amihud_illiq"]

    # Serial dependence / regime
    out["autocorr_5"] = ret_1.rolling(5).corr(ret_1.shift(1))
    out["autocorr_20"] = ret_1.rolling(20).corr(ret_1.shift(1))

    k = 20
    var_1 = log_ret_1.rolling(k).var(ddof=0)
    var_k = np.log(price).diff(k).rolling(k).var(ddof=0)
    out["variance_ratio_20"] = _safe_div(var_k, k * var_1)

    ma_spread = _safe_div(ma_20, ma_60) - 1.0
    out["trend_regime"] = np.where(
        ma_spread > 0.002, 1.0, np.where(ma_spread < -0.002, -1.0, 0.0)
    )

    vol_ratio = _safe_div(out["vol_20"], out["vol_60"])
    out["vol_regime"] = np.where(
        vol_ratio > 1.2, 1.0, np.where(vol_ratio < 0.8, -1.0, 0.0)
    )

    return out


def _add_market_level_features(out: pd.DataFrame, date_col: str) -> pd.DataFrame:
    market_cols = [
        date_col,
        "market_close",
        "vix_close",
        "oil_close",
        "dxy_close",
        "gold_close",
        "ust2y",
        "ust10y",
        "frt2y",
        "frt10y",
    ]
    daily = (
        out[market_cols]
        .drop_duplicates(subset=[date_col])
        .sort_values(date_col)
        .copy()
    )

    market_close = daily["market_close"].astype(float).clip(lower=EPS)
    market_ret_1 = market_close.pct_change()
    market_log_ret_1 = np.log(market_close).diff()
    market_index = (1.0 + market_ret_1.fillna(0.0)).cumprod()

    daily["market_ret_1"] = market_ret_1
    daily["market_ret_5"] = market_close.pct_change(5)
    daily["market_ret_20"] = market_close.pct_change(20)
    daily["market_vol_20"] = market_log_ret_1.rolling(20).std(ddof=0)
    daily["market_drawdown_20"] = _safe_div(market_index, market_index.rolling(20).max()) - 1.0
    daily["vix_level"] = daily["vix_close"].astype(float)
    daily["vix_ret_5"] = daily["vix_close"].astype(float).pct_change(5)

    daily["ust2y"] = daily["ust2y"].astype(float)
    daily["ust10y"] = daily["ust10y"].astype(float)
    daily["frt2y"] = daily["frt2y"].astype(float)
    daily["frt10y"] = daily["frt10y"].astype(float)
    daily["spread_2s10s"] = daily["ust10y"] - daily["ust2y"]
    daily["delta_10y_5d"] = daily["ust10y"].diff(5)

    daily["oil_ret_5"] = daily["oil_close"].astype(float).pct_change(5)
    daily["dxy_ret_5"] = daily["dxy_close"].astype(float).pct_change(5)
    daily["gold_ret_5"] = daily["gold_close"].astype(float).pct_change(5)

    keep_cols = [
        date_col,
        "market_ret_1",
        "market_ret_5",
        "market_ret_20",
        "market_vol_20",
        "market_drawdown_20",
        "vix_level",
        "vix_ret_5",
        "ust2y",
        "ust10y",
        "frt2y",
        "frt10y",
        "spread_2s10s",
        "delta_10y_5d",
        "oil_ret_5",
        "dxy_ret_5",
        "gold_ret_5",
    ]
    daily = daily[keep_cols]
    overlapping_feature_cols = [
        col for col in keep_cols if col != date_col and col in out.columns
    ]
    if overlapping_feature_cols:
        out = out.drop(columns=overlapping_feature_cols)
    return out.merge(daily, on=date_col, how="left")


def _add_sector_features(out: pd.DataFrame, date_col: str) -> pd.DataFrame:
    sector_daily = (
        out[[date_col, "sector_bucket", "_ret_1"]]
        .groupby([date_col, "sector_bucket"], as_index=False)["_ret_1"]
        .mean()
        .rename(columns={"_ret_1": "sector_ret_1"})
        .sort_values(["sector_bucket", date_col])
    )
    sector_daily["sector_ret_1"] = sector_daily["sector_ret_1"].astype(float)
    sector_daily["sector_index"] = (
        1.0 + sector_daily["sector_ret_1"].fillna(0.0)
    ).groupby(sector_daily["sector_bucket"]).cumprod()
    sector_daily["sector_ret_5"] = sector_daily.groupby("sector_bucket")["sector_index"].pct_change(5)
    sector_daily["sector_ret_20"] = sector_daily.groupby("sector_bucket")["sector_index"].pct_change(20)
    sector_log_ret = np.log1p(sector_daily["sector_ret_1"].clip(lower=-0.999999))
    sector_daily["sector_vol_20"] = (
        sector_log_ret.groupby(sector_daily["sector_bucket"]).rolling(20).std(ddof=0).reset_index(level=0, drop=True)
    )
    sector_daily["sector_drawdown_20"] = (
        _safe_div(
            sector_daily["sector_index"],
            sector_daily.groupby("sector_bucket")["sector_index"].rolling(20).max().reset_index(level=0, drop=True),
        )
        - 1.0
    )

    up = np.where(out["_ret_1"].isna(), np.nan, (out["_ret_1"] > 0).astype(float))
    breadth_df = out[[date_col, "sector_bucket"]].copy()
    breadth_df["up_flag"] = up
    sector_breadth = (
        breadth_df.groupby([date_col, "sector_bucket"], as_index=False)["up_flag"]
        .mean()
        .rename(columns={"up_flag": "sector_breadth_up"})
    )
    sector_daily = sector_daily.merge(sector_breadth, on=[date_col, "sector_bucket"], how="left")

    keep_cols = [
        date_col,
        "sector_bucket",
        "sector_ret_1",
        "sector_ret_5",
        "sector_ret_20",
        "sector_vol_20",
        "sector_drawdown_20",
        "sector_breadth_up",
    ]
    return out.merge(sector_daily[keep_cols], on=[date_col, "sector_bucket"], how="left")


def _add_beta_corr_features(out: pd.DataFrame, group_col: str, date_col: str) -> pd.DataFrame:
    def _one(group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values(date_col).copy()
        g["beta_market_60"] = _safe_div(
            g["_ret_1"].rolling(60).cov(g["market_ret_1"]),
            g["market_ret_1"].rolling(60).var(ddof=0),
        )
        g["corr_market_60"] = g["_ret_1"].rolling(60).corr(g["market_ret_1"])

        g["beta_sector_60"] = _safe_div(
            g["_ret_1"].rolling(60).cov(g["sector_ret_1"]),
            g["sector_ret_1"].rolling(60).var(ddof=0),
        )
        g["corr_sector_60"] = g["_ret_1"].rolling(60).corr(g["sector_ret_1"])
        return g

    parts = [_one(group) for _, group in out.groupby(group_col, sort=False)]
    return pd.concat(parts, ignore_index=True)


def _add_firm_micro_features(
    out: pd.DataFrame,
    volume_col: str,
    close_col: str,
    adj_close_col: str,
) -> pd.DataFrame:
    def _numeric_or_nan(col_name: str) -> pd.Series:
        if col_name in out.columns:
            return pd.to_numeric(out[col_name], errors="coerce")
        return pd.Series(np.nan, index=out.index, dtype=float)

    for col in [
        "market_cap",
        "book_value",
        "trailing_eps",
        "shares_outstanding",
        "short_percent_float",
        "short_ratio",
        volume_col,
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    price_col = adj_close_col if adj_close_col in out.columns else close_col
    price = pd.to_numeric(out[price_col], errors="coerce").clip(lower=EPS)

    out["market_cap_log"] = np.log(pd.to_numeric(out["market_cap"], errors="coerce").clip(lower=EPS))
    equity_value = pd.to_numeric(out["book_value"], errors="coerce") * pd.to_numeric(
        out["shares_outstanding"], errors="coerce"
    )
    out["book_to_market"] = _safe_div(equity_value, pd.to_numeric(out["market_cap"], errors="coerce"))
    out["earnings_yield"] = _safe_div(pd.to_numeric(out["trailing_eps"], errors="coerce"), price)
    out["turnover_ratio"] = _safe_div(
        pd.to_numeric(out[volume_col], errors="coerce"),
        pd.to_numeric(out["shares_outstanding"], errors="coerce"),
    )
    short_pct = _numeric_or_nan("short_percent_float")
    short_ratio = _numeric_or_nan("short_ratio")
    # Yahoo often exposes the short-interest fields for European tickers but
    # leaves every value empty. Treat that as neutral information; otherwise a
    # strict model-side dropna removes the whole ticker history.
    out["short_interest"] = short_pct.where(short_pct.notna(), short_ratio).fillna(0.0)
    return out


def _add_relative_ranks(out: pd.DataFrame, date_col: str) -> pd.DataFrame:
    rank_group = [date_col, "sector_bucket"]
    out["ret_20_rank_sector"] = out.groupby(rank_group)["ret_20"].rank(pct=True)
    out["vol_20_rank_sector"] = out.groupby(rank_group)["vol_20"].rank(pct=True)
    out["rsi_rank_sector"] = out.groupby(rank_group)["_rsi_14"].rank(pct=True)
    out["drawdown_rank_sector"] = out.groupby(rank_group)["drawdown_20"].rank(pct=True)
    return out


def _add_situational_features(out: pd.DataFrame, date_col: str) -> pd.DataFrame:
    dates = pd.to_datetime(out[date_col], errors="coerce")
    out["day_of_week"] = dates.dt.dayofweek.astype(float)
    out["week"] = dates.dt.isocalendar().week.astype(float)
    out["month"] = dates.dt.month.astype(float)

    for bucket in SECTOR_BUCKETS:
        feature_name = f"sector_{bucket}"
        out[feature_name] = (out["sector_bucket"] == bucket).astype(float)
    return out


def compute_market_features(
    df: pd.DataFrame,
    group_col: str = "ticker",
    date_col: str = "date",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    adj_close_col: str = "adj_close",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """
    Compute the requested feature set and append them to the input DataFrame.

    If `group_col` exists, features are computed independently per symbol for
    intra-symbol transforms, then merged with market/sector cross-sectional
    features on (`date`, `ticker`).
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    work = df.copy()
    if date_col not in work.columns:
        raise ValueError(f"Missing `{date_col}` column.")

    _require_columns(work, EXTERNAL_REQUIRED_COLUMNS, "external feature engineering")
    if "sector_bucket" not in work.columns and "sector" not in work.columns:
        raise ValueError("Missing sector information: expected `sector_bucket` or `sector`.")
    if "short_percent_float" not in work.columns:
        work["short_percent_float"] = np.nan
    if "short_ratio" not in work.columns:
        work["short_ratio"] = np.nan

    if "sector_bucket" not in work.columns:
        industry = work["industry"] if "industry" in work.columns else None
        work["sector_bucket"] = _normalize_sector_bucket(work["sector"], industry)
    else:
        work["sector_bucket"] = (
            work["sector_bucket"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        work.loc[~work["sector_bucket"].isin(SECTOR_BUCKETS), "sector_bucket"] = "unknown"

    created_group_col = False
    if group_col not in work.columns:
        work[group_col] = "__single_symbol__"
        created_group_col = True

    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).copy()
    work = work.sort_values([group_col, date_col]).reset_index(drop=True)

    parts = []
    for _, group in work.groupby(group_col, sort=False):
        parts.append(
            _compute_one_symbol_base(
                frame=group,
                date_col=date_col,
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
                adj_close_col=adj_close_col,
                volume_col=volume_col,
            )
        )
    out = pd.concat(parts, ignore_index=True)

    out = _add_market_level_features(out, date_col=date_col)
    out = _add_sector_features(out, date_col=date_col)

    up = np.where(out["_ret_1"].isna(), np.nan, (out["_ret_1"] > 0).astype(float))
    out["market_breadth"] = out[[date_col]].assign(up_flag=up).groupby(date_col)["up_flag"].transform("mean")

    out = _add_beta_corr_features(out, group_col=group_col, date_col=date_col)
    out["stock_minus_sector_ret_5"] = out["ret_5"] - out["sector_ret_5"]
    out["stock_minus_sector_ret_20"] = out["ret_20"] - out["sector_ret_20"]

    out = _add_firm_micro_features(
        out,
        volume_col=volume_col,
        close_col=close_col,
        adj_close_col=adj_close_col,
    )
    out = _add_relative_ranks(out, date_col=date_col)
    out = _add_situational_features(out, date_col=date_col)

    missing_features = [col for col in FEATURE_COLUMNS if col not in out.columns]
    if missing_features:
        raise ValueError(f"Missing computed feature columns: {missing_features}")

    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    out = out.sort_values([group_col, date_col]).reset_index(drop=True)

    if created_group_col:
        out = out.drop(columns=[group_col])

    internal_cols = [col for col in ["_ret_1", "_log_ret_1", "_rsi_14"] if col in out.columns]
    if internal_cols:
        out = out.drop(columns=internal_cols)
    return out
