import numpy as np
import pandas as pd
import pytest

from optimization import weights_sum_to_one, get_portfolio_var, portfolio_performance
from risk import portfolio_max_drawdown, compute_cumulative_value, historical_var_cvar


#optimization.py
def test_weights_sum_to_one_when_weights_are_valid():
    weights = np.array([0.5, 0.5])
    assert weights_sum_to_one(weights) == 0


def test_weights_sum_to_one_when_weights_are_invalid():
    weights = np.array([0.5, 0.3])
    assert weights_sum_to_one(weights) == pytest.approx(-0.2)


def test_get_portfolio_var_single_asset():
    # For a single asset with weight 1.0, portfolio variance equals asset variance
    weights = np.array([1.0])
    cov_matrix = pd.DataFrame([[0.0004]])
    assert get_portfolio_var(weights, cov_matrix) == pytest.approx(0.0004)


def test_portfolio_performance_single_asset_known_values():
    #Single asset: annual_risk equals annualized standard deviation sqrt(0.0004) * sqrt(252)
    weights = np.array([1.0])
    avg_returns = pd.Series([0.001], index=["A"])
    cov_matrix = pd.DataFrame([[0.0004]], index=["A"], columns=["A"])

    annual_return, annual_risk, sharpe = portfolio_performance(weights, avg_returns, cov_matrix)

    expected_return = 0.001 * 252
    expected_risk = np.sqrt(0.0004) * np.sqrt(252)

    assert annual_return == pytest.approx(expected_return)
    assert annual_risk == pytest.approx(expected_risk)
    assert sharpe == pytest.approx(expected_return / expected_risk)


def test_portfolio_performance_weights_must_sum_to_one_for_meaningful_result():
    # Ensures the function does not validate weight sums, leaving responsibility to caller
    weights = np.array([0.5, 0.5])
    avg_returns = pd.Series([0.001, 0.002], index=["A", "B"])
    cov_matrix = pd.DataFrame(
        [[0.0001, 0.00005], [0.00005, 0.0002]], index=["A", "B"], columns=["A", "B"]
    )
    annual_return, annual_risk, sharpe = portfolio_performance(weights, avg_returns, cov_matrix)
    assert annual_return == pytest.approx((0.5 * 0.001 + 0.5 * 0.002) * 252)


# risk.py
def test_portfolio_max_drawdown_zero_when_returns_always_positive():
    weights = np.array([1.0])
    daily_returns = pd.DataFrame({"A": [0.01] * 10})
    assert portfolio_max_drawdown(daily_returns, weights) == 0


def test_portfolio_max_drawdown_known_drop():
    # Day 1: +10% (1.10), Day 2: -20% (0.88) -> drawdown = (0.88 - 1.10) / 1.10 = -0.2
    weights = np.array([1.0])
    daily_returns = pd.DataFrame({"A": [0.10, -0.20]})
    assert portfolio_max_drawdown(daily_returns, weights) == pytest.approx(-0.2)


def test_compute_cumulative_value_known_path():
    weights = np.array([1.0])
    daily_returns = pd.DataFrame({"A": [0.10, -0.20]})
    result = compute_cumulative_value(daily_returns, weights)
    assert list(result) == pytest.approx([1.10, 0.88])


def test_historical_var_cvar_cvar_is_at_least_as_extreme_as_var():
    # CVaR is the mean loss beyond the VaR threshold, so CVaR >= VaR must always hold
    np.random.seed(42)
    weights = np.array([1.0])
    daily_returns = pd.DataFrame({"A": np.random.normal(0, 0.02, 500)})

    var, cvar = historical_var_cvar(daily_returns, weights, confidence=0.95)

    assert var >= 0
    assert cvar >= 0
    assert cvar >= var

def test_portfolio_cagr_constant_daily_return():
    from risk import portfolio_cagr

    daily_returns = pd.DataFrame({"A": [0.001] * 252})
    assert portfolio_cagr(daily_returns, np.array([1.0])) == pytest.approx(1.001**252 - 1)


def test_calmar_uses_cagr_not_arithmetic_mean():
    # Alternating +10% / -10%: arithmetic mean is 0 but the portfolio loses money (volatility drag)
    from risk import portfolio_calmar_ratio, portfolio_cagr

    daily_returns = pd.DataFrame({"A": [0.10, -0.10] * 126})
    weights = np.array([1.0])

    cagr = portfolio_cagr(daily_returns, weights)
    mdd = portfolio_max_drawdown(daily_returns, weights)

    assert cagr < 0
    assert portfolio_calmar_ratio(daily_returns, weights) == pytest.approx(cagr / abs(mdd))


def test_buy_and_hold_value_lets_weights_drift():
    # A: +100% then -50%, B: flat. Buy & hold ends at 1.0, daily rebalancing at 1.5 * 0.75 = 1.125
    from risk import compute_buy_and_hold_value

    daily_returns = pd.DataFrame({"A": [1.0, -0.5], "B": [0.0, 0.0]})
    weights = np.array([0.5, 0.5])

    assert list(compute_buy_and_hold_value(daily_returns, weights)) == pytest.approx([1.5, 1.0])
    assert list(compute_cumulative_value(daily_returns, weights)) == pytest.approx([1.5, 1.125])


def test_risk_summary_matches_individual_metrics():
    from risk import portfolio_risk_summary, portfolio_sortino_ratio

    rng = np.random.default_rng(0)
    daily_returns = pd.DataFrame(rng.normal(0.0005, 0.01, (500, 2)), columns=["A", "B"])
    weights = np.array([0.3, 0.7])

    summary = portfolio_risk_summary(daily_returns, weights, risk_free_rate=0.02, confidence=0.95)
    var, cvar = historical_var_cvar(daily_returns, weights, 0.95)

    assert summary["VaR"] == pytest.approx(var)
    assert summary["CVaR"] == pytest.approx(cvar)
    assert summary["Max Drawdown"] == pytest.approx(portfolio_max_drawdown(daily_returns, weights))
    assert summary["Sortino Ratio"] == pytest.approx(portfolio_sortino_ratio(daily_returns, weights, 0.02))


def test_strategy_metrics_reports_arithmetic_and_geometric_return():
    from risk import portfolio_cagr, strategy_metrics

    daily_returns = pd.DataFrame({"A": [0.10, -0.10] * 126})
    weights = np.array([1.0])

    metrics = strategy_metrics(daily_returns, weights, risk_free_rate=0.0)

    assert metrics["Annual Return (arithmetic)"] == pytest.approx(0.0)
    assert metrics["CAGR (geometric)"] == pytest.approx(portfolio_cagr(daily_returns, weights))
    assert metrics["Max Drawdown"] == pytest.approx(portfolio_max_drawdown(daily_returns, weights))
