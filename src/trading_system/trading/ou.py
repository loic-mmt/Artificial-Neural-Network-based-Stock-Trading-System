"""Long liquidation/acquisition boundaries for exponential OU with a trailing stop.

Leung & Zhang (2019), arXiv:1701.03960v2, equations 21 and 29--31.
The increasing fundamental solution uses D_nu(-z), the decreasing one D_nu(z).
Prices are normalized by exp(theta); transaction costs remain affine in price.
"""

from dataclasses import dataclass
from functools import lru_cache
import inspect
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar
from scipy.special import pbdv
from statsmodels.tsa.stattools import adfuller

_ADF_OPTIONS = {"result_object": False} if "result_object" in inspect.signature(adfuller).parameters else {}


@dataclass(frozen=True)
class OUConfig:
    mode: str = "off"  # off, exit, entry_exit (longs only)
    window: int = 252
    refit_every: int = 21
    discount_rate: float = 0.05  # subjective annual discount, NOT risk-free rate
    adf_pvalue: float = 0.05
    max_half_life_fraction: float = 0.25
    stability_sigma: float = 2.0

    def __post_init__(self):
        if self.mode not in {"off", "exit", "entry_exit"}:
            raise ValueError("OU mode must be off, exit or entry_exit.")
        for name, minimum in (("window", 40), ("refit_every", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"OU {name} must be an integer >= {minimum}.")
        for name in ("discount_rate", "stability_sigma"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"OU {name} must be finite and positive.")
        for name in ("adf_pvalue", "max_half_life_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"OU {name} must be in (0, 1).")


@dataclass(frozen=True)
class OUParameters:
    mean_reversion: float
    log_mean: float
    volatility: float

    def __post_init__(self):
        if not math.isfinite(self.log_mean):
            raise ValueError("OU log_mean must be finite.")
        for value in (self.mean_reversion, self.volatility):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("OU speed and volatility must be finite and positive.")


@dataclass(frozen=True)
class AffineCosts:
    buy_factor: float = 1.0
    sell_factor: float = 1.0
    buy_fixed: float = 0.0
    sell_fixed: float = 0.0

    def __post_init__(self):
        values = (self.buy_factor, self.sell_factor, self.buy_fixed, self.sell_fixed)
        if not all(math.isfinite(v) for v in values) or not 0 < self.sell_factor <= self.buy_factor or min(values[2:]) < 0:
            raise ValueError("Costs require 0 < sell_factor <= buy_factor and nonnegative fixed costs.")

    @classmethod
    def from_bps(cls, fees, slippage):
        if not all(math.isfinite(v) and 0 <= v < 10000 for v in (fees, slippage)):
            raise ValueError("Fees/slippage must be in [0, 10000) bps.")
        f, s = fees * 1e-4, slippage * 1e-4
        return cls((1 + f) * (1 + s), (1 - f) * (1 - s))


@dataclass(frozen=True)
class OUBoundaries:
    sell_price: float
    buy_price: float | None
    acquisition_value: float
    status: str
    root_residual: float


@lru_cache(maxsize=2048)
def solve_ou_boundaries(parameters, trailing_pct, discount_rate=0.05, *, costs=None,
                        rtol=2e-8, domain_sigma=9.0):
    """Solve a continuous, one-unit long problem. No portfolio optimality claim.

    Acquisition uses the single lower threshold geometry of the OU example.
    Reject unsupported multiple maxima / unconverged domains explicitly.
    No finite acquisition threshold is returned when all computed values <= 0.
    """
    if not 0 < trailing_pct < 1 or not math.isfinite(discount_rate) or discount_rate <= 0:
        raise ValueError("Trailing must be in (0, 1), discount must be positive.")
    if not 0 < rtol < 0.01 or not 6 <= domain_sigma <= 12:
        raise ValueError("Invalid numerical tolerance/domain.")
    costs = costs or AffineCosts()
    lam, sigma = parameters.mean_reversion, parameters.volatility
    scale = math.exp(parameters.log_mean)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("OU price scale is not representable.")
    cb, cs = costs.buy_fixed / scale, costs.sell_fixed / scale
    a, ab = costs.sell_factor, costs.buy_factor
    k, nu = math.sqrt(2 * lam) / sigma, -discount_rate / lam
    log_floor = math.log1p(-trailing_pct)

    def fundamental(l):
        z = k * l
        dm, ddm = pbdv(nu, z)
        dp, ddp = pbdv(nu, -z)
        if min(dm, dp) <= 0 or not np.isfinite([dm, dp, ddm, ddp]).all():
            raise ValueError("OU fundamental function outside numerical domain.")
        return z*z/4 + math.log(dm), z*z/4 + math.log(dp), k*(z/2 + ddm/dm), k*(z/2 - ddp/dp)

    def terms(l):
        lm, lp, sm, sp = fundamental(l)
        fm, fp, _, _ = fundamental(l + log_floor)
        denominator = -math.expm1(fp - fm - lp + lm)
        if denominator <= 0:
            raise ValueError("OU transformed trailing interval collapsed.")
        return sm, (sp - sm) / denominator, math.exp(lm - fm)

    def gamma(l):
        x = math.exp(l)
        sm, coeff, ratio = terms(l)
        return (a*x - (a*x-cs)*sm) / coeff - (a*x-cs) + (a*x*(1-trailing_pct)-cs)*ratio

    # Drift of discounted liquidation reward changes sign at x0.
    def drift(l):
        return a*math.exp(l)*(-lam*l + sigma*sigma/2 - discount_rate) + discount_rate*cs

    width = max(1., 2*sigma / math.sqrt(2*lam), math.log1p(cs))
    x0 = brentq(drift, -width, width + sigma*sigma/(2*lam), xtol=1e-12)
    b = brentq(gamma, x0, x0 - log_floor, xtol=1e-11)
    lower = min(-domain_sigma / k, b - 4/k)

    def ode(l, value):
        sm, coeff, ratio = terms(l)
        return sm*value + coeff*(value - (a*math.exp(l)*(1-trailing_pct)-cs)*ratio)

    solution = solve_ivp(ode, (b, lower), [a*math.exp(b)-cs], dense_output=True,
                         rtol=rtol, atol=rtol*0.1, max_step=min(.05, 1/k))
    if not solution.success:
        raise ValueError("OU liquidation ODE did not converge.")

    def objective(l):
        lm, _, _, _ = fundamental(l)
        return (float(solution.sol(l)[0]) - ab*math.exp(l)-cb) * math.exp(-lm)

    grid = np.linspace(lower, b, 241)
    values = np.array([objective(l) for l in grid])
    index = int(values.argmax())
    sell = scale*math.exp(b)
    residual = abs(gamma(b))
    if values[index] <= max(1e-12, rtol*0.1):
        return OUBoundaries(sell, None, 0., "no_profitable_entry", residual)
    if index in (0, len(grid)-1):
        raise ValueError("OU acquisition maximum touches numerical domain boundary.")
    peaks = (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]) & (values[1:-1] > values[index]*1e-5)
    if peaks.sum() != 1:
        raise ValueError("OU acquisition geometry is not a single lower region.")
    optimum = minimize_scalar(lambda l: -objective(l), bounds=(grid[index-1], grid[index+1]),
                              method="bounded", options={"xatol": 1e-10})
    # A remote lower endpoint must contribute negligible acquisition value.
    if not optimum.success or values[0] > objective(optimum.x)*1e-4:
        raise ValueError("OU acquisition domain did not converge.")
    return OUBoundaries(sell, scale*math.exp(optimum.x), scale*objective(optimum.x), "ok", residual)


def _ar1(y):
    x, following = y[:-1], y[1:]
    centered = x - x.mean()
    denominator = centered @ centered
    if denominator <= 1e-18:
        raise ValueError("constant_prices")
    b = float(centered @ (following-following.mean()) / denominator)
    intercept = float(following.mean()-b*x.mean())
    residual = following - intercept - b*x
    variance = float(residual @ residual / (len(x)-2))
    return intercept, b, variance, math.sqrt(variance / denominator)


def estimate_ou(prices, config=None, *, annualization=252):
    """Exact AR(1) transition fit on completed log prices, with rejection gates.

    Confidence interval is an approximate OLS interval; ADF is a separate unit
    root diagnostic. These checks cannot establish that markets follow OU.
    Caller owns causal sample selection. Window length is in observed bars.
    """
    config = config or OUConfig()
    values = np.asarray(prices, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("OU prices must be a positive finite vector.")
    if not isinstance(annualization, int) or annualization <= 0:
        raise ValueError("annualization must be a positive integer.")
    diagnostics = {"observations": len(values), "valid": False, "reason": "warmup"}
    if len(values) < config.window:
        return None, diagnostics
    y = np.log(values[-config.window:])
    try:
        intercept, b, variance, se = _ar1(y)
    except ValueError as exc:
        diagnostics["reason"] = str(exc)
        return None, diagnostics
    diagnostics.update(ar1=b, ar1_ci_low=b-1.96*se, ar1_ci_high=b+1.96*se)
    if not 0 < b < 1 or variance <= 0:
        diagnostics["reason"] = "non_mean_reverting"
        return None, diagnostics
    lam = -math.log(b)*annualization
    theta = intercept/(1-b)
    sigma = math.sqrt(variance*2*lam/(1-b*b))
    half_life = math.log(2)/(-math.log(b))
    pvalue = float(adfuller(y, maxlag=0, regression="c", autolag=None, **_ADF_OPTIONS)[1])
    stationary_sd = sigma/math.sqrt(2*lam)
    diagnostics.update(mean_reversion=lam, log_mean=theta, volatility=sigma,
                       half_life_bars=half_life, adf_pvalue=pvalue)
    if b+1.96*se >= 1 or pvalue > config.adf_pvalue:
        diagnostics["reason"] = "unit_root_uncertainty"
        return None, diagnostics
    if half_life > config.window*config.max_half_life_fraction:
        diagnostics["reason"] = "half_life_too_long"
        return None, diagnostics
    halves = np.array_split(y, 2)
    try:
        half_fits = [_ar1(part) for part in halves]
    except ValueError:
        diagnostics["reason"] = "unstable_halves"
        return None, diagnostics
    if not all(0 < fit[1] < 1 for fit in half_fits):
        diagnostics["reason"] = "unstable_halves"
        return None, diagnostics
    shift = abs(half_fits[0][0]/(1-half_fits[0][1]) - half_fits[1][0]/(1-half_fits[1][1])) / stationary_sd
    diagnostics["mean_shift_stationary_sigma"] = shift
    if shift > config.stability_sigma:
        diagnostics["reason"] = "unstable_mean"
        return None, diagnostics
    diagnostics.update(valid=True, reason="ok")
    return OUParameters(lam, theta, sigma), diagnostics
