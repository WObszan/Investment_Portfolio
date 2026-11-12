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


def run_monte_carlo(daily_returns: pd.DataFrame, n_simulations: int = 10000, risk_free_rate: float = 0.0) -> pd.DataFrame:
    """Simulate random portfolios, same idea as the notebook's Monte Carlo loop."""
    avg_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    num_assets = daily_returns.shape[1]

    results = []
    for _ in range(n_simulations):
        weights = np.random.rand(num_assets)
        weights /= weights.sum()

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


def weights_sum_to_one(weights: np.ndarray) -> float:
    return np.sum(weights) - 1


def portfolio_return_equals_target(weights: np.ndarray, avg_returns: pd.Series, target_return: float) -> float:
    annual_return = np.dot(weights, avg_returns) * TRADING_DAYS_PER_YEAR
    return annual_return - target_return


@st.cache_data(show_spinner=False)
def run_efficient_frontier(daily_returns: pd.DataFrame, n_points: int = 100) -> pd.DataFrame:
    avg_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    num_assets = daily_returns.shape[1]

    bounds = tuple((0, 0.5) for _ in range(num_assets))
    x0 = np.array([1 / num_assets] * num_assets)
    target_returns = np.linspace(avg_returns.min() * TRADING_DAYS_PER_YEAR, avg_returns.max() * TRADING_DAYS_PER_YEAR, n_points)

    results = []
    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": weights_sum_to_one},
            {"type": "eq", "fun": portfolio_return_equals_target, "args": (avg_returns, target)}
        ]
        result = minimize(
            get_portfolio_var, x0, args=(cov_matrix, ), method="SLSQP", bounds=bounds, constraints= constraints
        )
        results.append({ "weights": result.x,
                    "Target Return": target,
                    "Risk": np.sqrt(result.fun * TRADING_DAYS_PER_YEAR)})

    return pd.DataFrame(results)


@st.cache_data(show_spinner=False)
def get_min_var_portfolio(daily_returns: pd.DataFrame) -> dict:
    avg_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    num_assets = daily_returns.shape[1]

    bounds = tuple((0, 1) for _ in range(num_assets))
    x0 = np.array([1 / num_assets] * num_assets)

    constraints = [{"type": "eq", "fun": weights_sum_to_one}]
    result = minimize(get_portfolio_var, x0, args=(cov_matrix, ), method="SLSQP", bounds=bounds, constraints=constraints)

    return {"weights": result.x,
             "Annual Return": np.dot(result.x, avg_returns) * TRADING_DAYS_PER_YEAR,
             "Annual Risk": np.sqrt(result.fun * TRADING_DAYS_PER_YEAR)}


def negative_sharpe_ratio(weights, avg_returns, cov_matrix, risk_free_rate=0.0):
    ret, risk, sharpe = portfolio_performance(weights, avg_returns, cov_matrix, risk_free_rate)
    return -sharpe


@st.cache_data(show_spinner=False)
def get_max_sharpe_portfolio(daily_returns: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    avg_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    num_assets = daily_returns.shape[1]

    bounds = tuple((0, 1) for _ in range(num_assets))
    x0 = np.array([1 / num_assets] * num_assets)

    constraints = [{"type": "eq", "fun": weights_sum_to_one}]
    result = minimize(negative_sharpe_ratio, x0, args=(avg_returns, cov_matrix, risk_free_rate), method="SLSQP",
                 bounds=bounds, constraints=constraints)
    ret, risk, sharpe = portfolio_performance(result.x, avg_returns, cov_matrix, risk_free_rate)
    return {"weights": result.x,
             "Annual Return": ret,
             "Annual Risk": risk,
             "Sharpe Ratio": sharpe}

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