"""Fixed-width fractional differentiation, fitted exclusively on training history."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd

FRACDIFF_FEATURE = "fracdiff_log_price"


@dataclass(frozen=True)
class FracDiffConfig:
    # None selects the smallest passing candidate; numeric values freeze the order.
    order: float | None = None
    threshold: float = 1e-3
    max_terms: int = 1000
    candidates: tuple[float, ...] = tuple(i / 10 for i in range(1, 10))
    adf_pvalue: float = 0.05
    min_samples: int = 64

    def __post_init__(self):
        def number(value, lower, upper, name):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")
            if not math.isfinite(value) or not lower < value < upper:
                raise ValueError(f"{name} must be between {lower} and {upper} (exclusive).")

        if self.order is not None:
            number(self.order, 0, 1, "FracDiff order")
        number(self.threshold, 0, 1, "FracDiff threshold")
        number(self.adf_pvalue, 0, 1, "ADF p-value threshold")
        for name, minimum in (("max_terms", 2), ("min_samples", 16)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("FracDiff candidates cannot be empty.")
        for value in candidates:
            number(value, 0, 1, "FracDiff candidate")
        if tuple(sorted(set(candidates))) != candidates:
            raise ValueError("FracDiff candidates must be unique and increasing.")
        object.__setattr__(self, "candidates", candidates)


def fractional_weights(order: float, threshold: float = 1e-3, max_terms: int = 1000):
    """Newest-first binomial coefficients, truncated before |w_k| < threshold.

    Endpoints 0 and 1 are accepted here for identity/difference unit tests.
    Hitting max_terms raises rather than silently changing the truncation rule.
    """
    if isinstance(order, bool) or not isinstance(order, (int, float)) or not 0 <= order <= 1:
        raise ValueError("order must be finite and in [0, 1].")
    FracDiffConfig(threshold=threshold, max_terms=max_terms)
    weights = [1.0]
    for k in range(1, max_terms + 1):
        weight = -weights[-1] * (order - k + 1) / k
        if abs(weight) < threshold:
            return np.asarray(weights, dtype=np.float64)
        if k == max_terms:
            raise ValueError("FracDiff max_terms reached before threshold; increase max_terms or threshold.")
        weights.append(weight)
    raise AssertionError("Unreachable")


def fractional_difference(values, weights) -> np.ndarray:
    """Causal FIR convolution with full-window warmup; never back/forward-fill."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.ndim != 1 or not len(weights):
        raise ValueError("Values and weights must be 1D, with non-empty weights.")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("FracDiff values and weights must be finite.")
    output = np.full(len(values), np.nan)
    if len(values) >= len(weights):
        output[len(weights) - 1:] = np.convolve(values, weights, mode="valid")
    return output


def _diagnostics(values: np.ndarray, original: np.ndarray) -> dict:
    if np.ptp(values) <= np.finfo(float).eps * max(1.0, np.max(np.abs(values))):
        return {"adf_pvalue": None, "correlation": None, "status": "constant"}
    from statsmodels.tsa.stattools import adfuller
    try:
        result = adfuller(values, regression="c", autolag="AIC")
        pvalue = float(result[1])
        correlation = (
            float(np.corrcoef(values, original)[0, 1]) if np.std(original) > 0 else None
        )
        return {
            "adf_pvalue": pvalue if math.isfinite(pvalue) else None,
            "correlation": correlation if correlation is not None and math.isfinite(correlation) else None,
            "status": "tested",
        }
    except (ValueError, np.linalg.LinAlgError) as error:
        return {"adf_pvalue": None, "correlation": None, "status": str(error)}


class FracDiffTransformer:
    """One frozen order per ticker, plus serializable train-only diagnostics.

    transform receives chronological raw history including warmup. It never fits
    or uses rows after the output timestamp. Unseen tickers require a new fit.
    """

    def __init__(self, config: FracDiffConfig, *, price_col="adj_close", date_col="date", group_col=None):
        self.config = config
        self.price_col = price_col
        self.date_col = date_col
        self.group_col = group_col
        self.groups: dict[str, dict] = {}

    def _groups(self, frame):
        required = [self.price_col, self.date_col] + ([self.group_col] if self.group_col else [])
        if frame.empty or any(column not in frame for column in required):
            raise ValueError(f"FracDiff requires a non-empty frame with {required}.")
        if self.group_col and not frame[self.group_col].map(lambda x: isinstance(x, str)).all():
            raise ValueError("FracDiff ticker keys must be non-missing strings.")
        groups = frame.groupby(self.group_col, sort=False) if self.group_col else [("__single__", frame)]
        for key, group in groups:
            dates = pd.to_datetime(group[self.date_col], errors="coerce")
            if dates.isna().any() or dates.duplicated().any():
                raise ValueError("FracDiff dates must be valid and unique per ticker.")
            group = group.assign(**{self.date_col: dates}).sort_values(self.date_col)
            prices = pd.to_numeric(group[self.price_col], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(prices).all() or (prices <= 0).any():
                raise ValueError("FracDiff prices must be finite and positive; no implicit filling.")
            yield key, group, np.log(prices)

    def fit(self, train: pd.DataFrame):
        fitted = {}
        for key, group, values in self._groups(train):
            orders = self.config.candidates if self.config.order is None else (self.config.order,)
            eligible, diagnostics = [], []
            for order in orders:
                try:
                    weights = fractional_weights(order, self.config.threshold, self.config.max_terms)
                    if len(values) - len(weights) + 1 < self.config.min_samples:
                        raise ValueError("Insufficient training history after FracDiff warmup.")
                    eligible.append((order, weights))
                except ValueError as error:
                    diagnostics.append({"order": order, "status": str(error)})
            if not eligible:
                raise ValueError(f"FracDiff {key}: no candidate has sufficient history/converged weights.")
            # Compare all feasible candidates on the exact same training dates.
            start = max(len(weights) - 1 for _, weights in eligible)
            selected = None
            for order, weights in eligible:
                transformed = fractional_difference(values, weights)[start:]
                diagnostic = {
                    "order": order, "terms": len(weights), "n_samples": len(transformed),
                    **_diagnostics(transformed, values[start:]),
                }
                diagnostics.append(diagnostic)
                pvalue = diagnostic["adf_pvalue"]
                if selected is None and (self.config.order is not None or (
                    pvalue is not None and pvalue <= self.config.adf_pvalue
                )):
                    selected = (order, weights)
            if selected is None:
                raise ValueError(f"FracDiff {key}: no candidate passes train-only ADF; use an explicit order or revise candidates.")
            order, weights = selected
            fitted[key] = {
                "order": order, "weights": weights.tolist(), "diagnostics": diagnostics,
                "train_rows": len(group), "train_end": str(group[self.date_col].max()),
                "diagnostic_start": str(group[self.date_col].iloc[start]),
            }
        self.groups = fitted
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.groups:
            raise ValueError("FracDiff must be fitted on training data before transform.")
        output = frame.copy().reset_index(drop=True)
        output[FRACDIFF_FEATURE] = np.nan
        for key, group, values in self._groups(output):
            if key not in self.groups:
                raise ValueError(f"FracDiff has no fitted order for ticker {key}.")
            output.loc[group.index, FRACDIFF_FEATURE] = fractional_difference(values, self.groups[key]["weights"])
        return output

    def state_dict(self):
        return deepcopy({
            "config": asdict(self.config), "price_col": self.price_col,
            "date_col": self.date_col, "group_col": self.group_col, "groups": self.groups,
            "feature": FRACDIFF_FEATURE, "input": "log_price", "version": 1,
        })

    @classmethod
    def from_state_dict(cls, state):
        if state.get("version") != 1 or state.get("feature") != FRACDIFF_FEATURE:
            raise ValueError("Unsupported FracDiff state.")
        result = cls(FracDiffConfig(**state["config"]), price_col=state["price_col"],
                     date_col=state["date_col"], group_col=state["group_col"])
        result.groups = deepcopy(state["groups"])
        for group in result.groups.values():
            expected = fractional_weights(group["order"], result.config.threshold, result.config.max_terms)
            if not np.array_equal(expected, group["weights"]):
                raise ValueError("FracDiff stored weights do not match configuration.")
        return result
