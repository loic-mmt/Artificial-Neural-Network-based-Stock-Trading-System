"""Optional broad feature set with explicit missing-data and publication safeguards."""

from copy import deepcopy

import numpy as np
import pandas as pd

from trading_system.data.feature_sources import FUNDAMENTAL_FIELDS, SENTIMENT_FEATURES
from .market import MARKET_FEATURE_COLUMNS, EXTERNAL_REQUIRED_COLUMNS, compute_market_features
from .technical import TECHNICAL_FEATURE_COLUMNS, compute_technical_features

FUNDAMENTAL_FEATURES = (
    "pe_ratio", "pb_ratio", "ev_ebitda", "profit_margin", "return_on_equity",
    "return_on_assets", "revenue_growth_yoy", "eps_growth_yoy", "debt_to_equity",
    "fundamental_age_days", "market_cap_log", "book_to_market", "earnings_yield", "turnover_ratio",
)
SECTOR_EXTRA_FEATURES = ("sector_momentum_60", "sector_surge_z_20", "sector_surge_breadth")
UNSAFE_LEGACY_FIRM = ("market_cap", "book_value", "trailing_eps", "shares_outstanding", "short_percent_float", "short_ratio")
SECTOR_COLUMNS = tuple(c for c in MARKET_FEATURE_COLUMNS if "sector" in c or c == "rsi_rank_sector")
GROUP_COLUMNS = {
    "technical": tuple(TECHNICAL_FEATURE_COLUMNS),
    "market": tuple(c for c in MARKET_FEATURE_COLUMNS if c not in SECTOR_COLUMNS and c not in FUNDAMENTAL_FEATURES and c != "short_interest"),
    "sector": (*SECTOR_COLUMNS, *SECTOR_EXTRA_FEATURES),
    "fundamentals": FUNDAMENTAL_FEATURES,
    "sentiment": SENTIMENT_FEATURES,
}
DEFAULT_GROUPS = tuple(GROUP_COLUMNS)


def feature_columns(groups=DEFAULT_GROUPS):
    if not groups or any(group not in GROUP_COLUMNS for group in groups) or len(set(groups)) != len(groups):
        raise ValueError(f"Feature groups must be unique members of {DEFAULT_GROUPS}.")
    return tuple(dict.fromkeys(column for group in groups for column in GROUP_COLUMNS[group]))


def _ratio(numerator, denominator, *, positive=True):
    valid = denominator > 1e-12 if positive else denominator.abs() > 1e-12
    return numerator / denominator.where(valid)


def compute_expanded_features(frame, *, group_col="ticker", date_col="date"):
    work = frame.copy()
    if work.duplicated([c for c in (group_col, date_col) if c in work]).any():
        raise ValueError("Expanded features require unique ticker/date rows.")
    # Legacy Yahoo .info snapshots are not historical publication-dated values.
    work[list(UNSAFE_LEGACY_FIRM)] = np.nan
    for column in EXTERNAL_REQUIRED_COLUMNS:
        if column not in work:
            work[column] = np.nan
    if "sector" not in work and "sector_bucket" not in work:
        work["sector"] = "unknown"
    # Refuse arbitrary first-ticker selection for conflicting global macro inputs.
    macro = [c for c in EXTERNAL_REQUIRED_COLUMNS if c not in UNSAFE_LEGACY_FIRM]
    if (work.groupby(date_col)[macro].nunique(dropna=True) > 1).any().any():
        raise ValueError("Conflicting market/macro values on the same date.")
    work[macro] = work.groupby(date_col)[macro].transform("first")
    work = compute_technical_features(work, group_col=group_col if group_col in work else None, date_col=date_col)
    work = compute_market_features(work, group_col=group_col, date_col=date_col)
    cutoff = pd.to_datetime(work[date_col], utc=True)
    available = pd.to_datetime(work.get("fund_available_at", pd.Series(pd.NaT, index=work.index)), utc=True, errors="coerce")
    age = (cutoff - available).dt.total_seconds() / 86400
    known = available.notna() & (available < cutoff) & age.between(0, 550)
    values = {}
    for field in FUNDAMENTAL_FIELDS:
        raw = work.get(f"fund_{field}", pd.Series(np.nan, index=work.index))
        values[field] = pd.to_numeric(raw, errors="coerce").where(known)
    price = pd.to_numeric(work["close"], errors="coerce").where(lambda x: x > 0)
    shares = values["shares_outstanding"].where(lambda x: x > 0)
    cap = price * shares
    equity, earnings = values["equity"], values["net_income_ttm"]
    ev = cap + values["debt"] - values["cash"]
    extra = {
        "pe_ratio": _ratio(price, values["eps_ttm"]),
        "pb_ratio": _ratio(cap, equity),
        "ev_ebitda": _ratio(ev.where(ev > 0), values["ebitda_ttm"]),
        "profit_margin": _ratio(earnings, values["revenue_ttm"]),
        "return_on_equity": _ratio(earnings, equity),
        "return_on_assets": _ratio(earnings, values["assets"]),
        "revenue_growth_yoy": _ratio(values["revenue_ttm"] - values["revenue_ttm_prev_year"], values["revenue_ttm_prev_year"]),
        "eps_growth_yoy": _ratio(values["eps_ttm"] - values["eps_ttm_prev_year"], values["eps_ttm_prev_year"].abs()),
        "debt_to_equity": _ratio(values["debt"], equity),
        "fundamental_age_days": age.where(known),
        "market_cap_log": np.log(cap.where(cap > 0)),
        "book_to_market": _ratio(equity, cap),
        "earnings_yield": _ratio(values["eps_ttm"], price),
        "turnover_ratio": _ratio(pd.to_numeric(work["volume"], errors="coerce"), shares),
    }
    sector = work[[date_col, "sector_bucket", "sector_ret_1", "sector_breadth_up"]].drop_duplicates([date_col, "sector_bucket"]).sort_values(["sector_bucket", date_col])
    sector_parts = []
    for _, group in sector.groupby("sector_bucket", sort=False):
        group = group.copy()
        returns = group.sector_ret_1
        index = (1 + returns.fillna(0)).cumprod()
        group["sector_momentum_60"] = index.pct_change(60, fill_method=None)
        prior = returns.shift(1).rolling(20)
        group["sector_surge_z_20"] = _ratio(returns - prior.mean(), prior.std(ddof=0))
        group["sector_surge_breadth"] = ((group.sector_surge_z_20 > 2) & (group.sector_breadth_up >= 0.7)).astype(float).where(group.sector_surge_z_20.notna())
        sector_parts.append(group[[date_col, "sector_bucket", *SECTOR_EXTRA_FEATURES]])
    work = work.drop(columns=list(extra), errors="ignore")
    work = pd.concat([work, pd.DataFrame(extra, index=work.index)], axis=1)
    work = work.merge(pd.concat(sector_parts), on=[date_col, "sector_bucket"], how="left", validate="many_to_one")
    missing_sentiment = [column for column in SENTIMENT_FEATURES if column not in work]
    work = pd.concat([work, pd.DataFrame(np.nan, index=work.index, columns=missing_sentiment)], axis=1).copy()
    columns = feature_columns()
    work[list(columns)] = work[list(columns)].replace([np.inf, -np.inf], np.nan)
    work.attrs.update(deepcopy(frame.attrs))
    return work


class ExpandedFeatureSelector:
    """Freeze coverage/constant-column selection on training data only."""
    def __init__(self, min_coverage=0.5):
        if not 0 < min_coverage <= 1:
            raise ValueError("Feature min coverage must be in (0, 1].")
        self.min_coverage = min_coverage
        self.state = None

    def fit(self, train, columns):
        table = train[list(columns)].replace([np.inf, -np.inf], np.nan)
        coverage = table.notna().mean()
        unique = table.nunique(dropna=True)
        selected = [c for c in columns if coverage[c] >= self.min_coverage and unique[c] > 1]
        if not selected:
            raise ValueError("No expanded features satisfy training coverage/variance checks.")
        self.state = {
            "selected": selected, "min_coverage": self.min_coverage, "train_rows": len(train),
            "coverage": {c: float(coverage[c]) for c in columns},
            "dropped": {c: "missing" if coverage[c] < self.min_coverage else "constant" for c in columns if c not in selected},
            "coverage_by_ticker": {
                str(ticker): {c: float(value) for c, value in group[list(columns)].notna().mean().items()}
                for ticker, group in train.groupby("ticker", sort=False)
            } if "ticker" in train else {},
            "source_note": "Undated Yahoo fundamentals ignored; macro vintage and historical sector membership remain caller responsibilities.",
            "feature_sources": deepcopy(train.attrs.get("feature_sources")),
        }
        return self

    @property
    def columns(self):
        if self.state is None:
            raise ValueError("Feature selection must be fitted on training data.")
        return tuple(self.state["selected"])

    def state_dict(self):
        return deepcopy(self.state)
