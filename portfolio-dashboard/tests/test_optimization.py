import numpy as np
import pandas as pd
import pytest

from optimization import (
    get_max_sharpe_portfolio,
    get_min_var_portfolio,
    run_efficient_frontier,
    run_monte_carlo,
)


def _returns_with_dominant_asset(n=1500, seed=3):
    """Three assets; 'STAR' has a much better Sharpe so Max Sharpe puts > 50% in it."""
    rng = np.random.default_rng(seed)
    data = {
        "STAR": rng.normal(0.0012, 0.012, n),
        "MID": rng.normal(0.0004, 0.015, n),
        "LOW": rng.normal(0.0002, 0.010, n),
    }
    return pd.DataFrame(data, index=pd.date_range("2018-01-01", periods=n, freq="B"))


def _frontier_risk_at(frontier: pd.DataFrame, target_return: float) -> float:
    ok = frontier[frontier["success"]].sort_values("Target Return")
    assert ok["Target Return"].min() - 1e-9 <= target_return <= ok["Target Return"].max() + 1e-9, (
        "portfolio return lies outside the plotted frontier"
    )
    return float(np.interp(target_return, ok["Target Return"], ok["Risk"]))


def _assert_on_frontier(returns, **kwargs):
    frontier = run_efficient_frontier(returns, n_points=60, **kwargs)
    max_sharpe = get_max_sharpe_portfolio(returns, **kwargs)
    min_var = get_min_var_portfolio(returns, **kwargs)

    for portfolio in (max_sharpe, min_var):
        frontier_risk = _frontier_risk_at(frontier, portfolio["Annual Return"])
        assert portfolio["Annual Risk"] == pytest.approx(frontier_risk, abs=0.005)


def test_exact_portfolios_lie_on_frontier_with_default_bounds():
    _assert_on_frontier(_returns_with_dominant_asset())


def test_exact_portfolios_lie_on_frontier_with_weight_cap():
    returns = _returns_with_dominant_asset()
    _assert_on_frontier(returns, max_weight=0.5)

    max_sharpe = get_max_sharpe_portfolio(returns, max_weight=0.5)
    assert max_sharpe["weights"].max() <= 0.5 + 1e-6
    assert max_sharpe["success"]


def test_max_weight_below_one_over_n_is_rejected():
    with pytest.raises(ValueError):
        get_min_var_portfolio(_returns_with_dominant_asset(), max_weight=0.2)


def test_monte_carlo_respects_max_weight():
    mc = run_monte_carlo(_returns_with_dominant_asset(), n_simulations=500, max_weight=0.4)
    weights = np.vstack(mc["weights"].to_numpy())
    assert weights.max() <= 0.4 + 1e-9
    assert np.allclose(weights.sum(axis=1), 1.0)
