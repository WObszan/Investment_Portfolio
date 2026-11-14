import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize


TRADING_DAYS_PER_YEAR = 252


def portfolio_performance(
    weights: np.ndarray,
    avg_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> tuple[float, float, float]:
    """Annual return, annual risk, sharpe ratio for a given set of weights."""
    annual_return = np.dot(weights, avg_returns) * TRADING_DAYS_PER_YEAR
    annual_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) * TRADING_DAYS_PER_YEAR)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_risk
    return annual_return, annual_risk, sharpe_ratio


def cap_weights(weights: np.ndarray, max_weight: float) -> np.ndarray:
    """Project rows of non-negative weights (summing to 1) onto the cap w_i <= max_weight.

    Excess above the cap is redistributed proportionally to the uncapped assets until
    no weight exceeds the cap. Requires max_weight * n_assets >= 1.
    """
    w = np.atleast_2d(np.asarray(weights, dtype=float)).copy()
    for _ in range(w.shape[1]):
        over = w > max_weight + 1e-12
        if not over.any():
            break
        excess = np.where(over, w - max_weight, 0.0).sum(axis=1, keepdims=True)
        w = np.where(over, max_weight, w)
        free = (w < max_weight - 1e-12)
        free_total = np.where(free, w, 0.0).sum(axis=1, keepdims=True)
        n_free = free.sum(axis=1, keepdims=True)
        # Proportional redistribution; fall back to equal split if free weights are all zero
        share = np.where(free_total > 0, w / np.where(free_total > 0, free_total, 1.0),
                         1.0 / np.maximum(n_free, 1))
        w = np.where(free, w + excess * share, w)
    return w


def run_monte_carlo(
    daily_returns: pd.DataFrame,
    n_simulations: int = 10000,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
) -> pd.DataFrame:
    """Simulate random long-only portfolios whose weights respect `max_weight`."""
    avg_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    num_assets = daily_returns.shape[1]
    _check_max_weight(max_weight, num_assets)

    all_weights = np.random.rand(n_simulations, num_assets)
    all_weights /= all_weights.sum(axis=1, keepdims=True)
    if max_weight < 1.0:
        all_weights = cap_weights(all_weights, max_weight)

    results = []
    for weights in all_weights:
        ret, risk, sharpe = portfolio_performance(weights, avg_returns, cov_matrix, risk_free_rate)
        results.append(
            {
                "weights": weights,
                "Annual Expected Return": ret,
                "Expected Annual Risk": risk,
                "Expected Sharpe Ratio": sharpe,
            }
        )

    return pd.DataFrame(results)

def get_portfolio_var(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    return np.dot(weights.T, np.dot(cov_matrix, weights))


def annual_portfolio_var(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    """Annualized portfolio variance; used as the optimizer objective.

    Raw daily variance (~1e-4) is so small that SLSQP's default tolerance stops at the
    starting point, so the objective is annualized and a tight ftol is used.
    """
    return get_portfolio_var(weights, cov_matrix) * TRADING_DAYS_PER_YEAR


SLSQP_OPTIONS = {"ftol": 1e-12, "maxiter": 1000}


def weights_sum_to_one(weights: np.ndarray) -> float:
    return np.sum(weights) - 1


def portfolio_return_equals_target(weights: np.ndarray, avg_returns: pd.Series, target_return: float) -> float:
    annual_return = np.dot(weights, avg_returns) * TRADING_DAYS_PER_YEAR
    return annual_return - target_return


def _check_max_weight(max_weight: float, num_assets: int) -> None:
    """Raise if no long-only portfolio with w_i <= max_weight can sum to 1."""
    if not 0 < max_weight <= 1:
        raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")
    if max_weight * num_assets < 1 - 1e-9:
        raise ValueError(
            f"max_weight={max_weight} is infeasible for {num_assets} assets "
            f"(needs max_weight >= 1/{num_assets})"
        )


def _weight_bounds(max_weight: float, num_assets: int) -> tuple[tuple[float, float], ...]:
    _check_max_weight(max_weight, num_assets)
    return tuple((0.0, max_weight) for _ in range(num_assets))


def _extreme_feasible_return(avg_returns: pd.Series, max_weight: float, highest: bool) -> float:
    """Highest (or lowest) annual return reachable with weights in [0, max_weight] summing to 1."""
    sorted_returns = np.sort(avg_returns.to_numpy())
    if highest:
        sorted_returns = sorted_returns[::-1]
    remaining, total = 1.0, 0.0
    for r in sorted_returns:
        w = min(max_weight, remaining)
        total += w * r
        remaining -= w
        if remaining <= 1e-12:
            break
    return total * TRADING_DAYS_PER_YEAR


@st.cache_data(show_spinner=False)
def run_efficient_frontier(daily_returns: pd.DataFrame, n_points: int = 100, max_weight: float = 1.0) -> pd.DataFrame:
    """Minimum-variance frontier over the feasible return range for weights in [0, max_weight].

    The `success` column flags points where SLSQP did not converge; callers should drop them.
    """
    avg_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    num_assets = daily_returns.shape[1]

    bounds = _weight_bounds(max_weight, num_assets)
    x0 = np.array([1 / num_assets] * num_assets)
    target_returns = np.linspace(
        _extreme_feasible_return(avg_returns, max_weight, highest=False),
        _extreme_feasible_return(avg_returns, max_weight, highest=True),
        n_points,
    )

    results = []
    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": weights_sum_to_one},
            {"type": "eq", "fun": portfolio_return_equals_target, "args": (avg_returns, target)}
        ]
        result = minimize(
            annual_portfolio_var, x0, args=(cov_matrix, ), method="SLSQP", bounds=bounds, constraints=constraints,
            options=SLSQP_OPTIONS,
        )
        results.append({ "weights": result.x,
                    "Target Return": target,
                    "Risk": np.sqrt(max(result.fun, 0.0)),
                    "success": bool(result.success)})

    return pd.DataFrame(results)


@st.cache_data(show_spinner=False)
def get_min_var_portfolio(daily_returns: pd.DataFrame, max_weight: float = 1.0) -> dict:
    """Exact minimum-variance portfolio with weights in [0, max_weight]."""
    avg_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    num_assets = daily_returns.shape[1]

    bounds = _weight_bounds(max_weight, num_assets)
    x0 = np.array([1 / num_assets] * num_assets)

    constraints = [{"type": "eq", "fun": weights_sum_to_one}]
    result = minimize(annual_portfolio_var, x0, args=(cov_matrix, ), method="SLSQP", bounds=bounds,
                      constraints=constraints, options=SLSQP_OPTIONS)

    return {"weights": result.x,
             "Annual Return": np.dot(result.x, avg_returns) * TRADING_DAYS_PER_YEAR,
             "Annual Risk": np.sqrt(max(result.fun, 0.0)),
             "success": bool(result.success)}


def negative_sharpe_ratio(weights, avg_returns, cov_matrix, risk_free_rate=0.0):
    ret, risk, sharpe = portfolio_performance(weights, avg_returns, cov_matrix, risk_free_rate)
    return -sharpe


@st.cache_data(show_spinner=False)
def get_max_sharpe_portfolio(daily_returns: pd.DataFrame, risk_free_rate: float = 0.0, max_weight: float = 1.0) -> dict:
    """Exact maximum-Sharpe (tangency) portfolio with weights in [0, max_weight]."""
    avg_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    num_assets = daily_returns.shape[1]

    bounds = _weight_bounds(max_weight, num_assets)
    x0 = np.array([1 / num_assets] * num_assets)

    constraints = [{"type": "eq", "fun": weights_sum_to_one}]
    result = minimize(negative_sharpe_ratio, x0, args=(avg_returns, cov_matrix, risk_free_rate), method="SLSQP",
                 bounds=bounds, constraints=constraints, options=SLSQP_OPTIONS)
    ret, risk, sharpe = portfolio_performance(result.x, avg_returns, cov_matrix, risk_free_rate)
    return {"weights": result.x,
             "Annual Return": ret,
             "Annual Risk": risk,
             "Sharpe Ratio": sharpe,
             "success": bool(result.success)}

def calculate_portfolio_growth(
    initial_amount: float,
    periodic_contribution: float,
    frequency: str,
    annual_return: float,
    annual_risk: float,
    years: int,
    n_simulations: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """DCA projection: deterministic mean path plus Monte Carlo 16th/50th/84th percentile paths.

    Each period the contribution is deposited first, then a simulated return is applied.
    Period returns are normal with mean = annual_return / periods_per_year and
    sd = annual_risk / sqrt(periods_per_year); returns are floored at -100%.
    Percentiles are taken across simulations separately for every period, so the band
    widens like sqrt(T) instead of T.
    """
    periods_per_year = 12 if frequency == "Monthly" else 1
    total_periods = years * periods_per_year

    period_return = annual_return / periods_per_year
    period_risk = annual_risk / np.sqrt(periods_per_year)

    rng = np.random.default_rng(seed)
    sim_returns = rng.normal(period_return, period_risk, size=(total_periods, n_simulations))
    sim_returns = np.maximum(sim_returns, -1.0)

    sim_values = np.empty((total_periods + 1, n_simulations))
    sim_values[0] = initial_amount
    deterministic_values = np.empty(total_periods + 1)
    deterministic_values[0] = initial_amount
    for i in range(1, total_periods + 1):
        sim_values[i] = (sim_values[i - 1] + periodic_contribution) * (1 + sim_returns[i - 1])
        deterministic_values[i] = (deterministic_values[i - 1] + periodic_contribution) * (1 + period_return)

    p16, p50, p84 = np.percentile(sim_values, [16, 50, 84], axis=1)
    if annual_risk == 0:
        # No randomness: avoid floating-point noise between identical paths
        p16 = p50 = p84 = deterministic_values

    periods = np.arange(total_periods + 1)
    return pd.DataFrame({
        "Period": periods,
        "Years": periods / periods_per_year,
        "Expected (Mean)": deterministic_values,
        "Median (50th pct)": p50,
        "Optimistic (84th pct)": p84,
        "Pessimistic (16th pct)": p16,
    })


def get_capped_return(avg_returns: pd.Series, max_annual_return: float = 0.20) -> pd.Series:
    """It caps historical rates of return at a reasonable maximum (e.g., 20% annually)
      to avoid distortions resulting from a short-term tech stock rally."""
    annual_returns = avg_returns * TRADING_DAYS_PER_YEAR
    capped_annual = np.clip(annual_returns, -0.90, max_annual_return)
    return capped_annual / TRADING_DAYS_PER_YEAR