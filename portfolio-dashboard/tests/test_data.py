import numpy as np
import pandas as pd
import pytest

from data import get_daily_returns


def _correlated_prices_one_listed_halfway(n=4000, true_corr=0.40, daily_vol=0.02, seed=7):
    """Two assets with known return correlation; asset B only has prices for the second half."""
    rng = np.random.default_rng(seed)
    cov = daily_vol**2 * np.array([[1.0, true_corr], [true_corr, 1.0]])
    rets = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    dates = pd.date_range("2005-01-03", periods=n + 1, freq="B")
    prices = pd.DataFrame(
        100 * np.vstack([np.ones(2), np.cumprod(1 + rets, axis=0)]),
        index=dates, columns=["A", "B"],
    )
    prices.iloc[: n // 2, prices.columns.get_loc("B")] = np.nan  # B "IPOs" halfway
    return prices


def test_daily_returns_do_not_treat_pre_listing_as_zero_return():
    prices = _correlated_prices_one_listed_halfway()

    returns = get_daily_returns(prices)

    assert returns.corr().loc["A", "B"] == pytest.approx(0.40, abs=0.05)
    # Volatility of the late-listed asset must not be diluted by fake 0% days
    assert returns["B"].std() * np.sqrt(252) == pytest.approx(0.02 * np.sqrt(252), rel=0.05)


def test_daily_returns_use_common_period_only():
    prices = _correlated_prices_one_listed_halfway()

    returns = get_daily_returns(prices)

    assert not returns.isna().any().any()
    assert returns.index.min() > prices["B"].first_valid_index()
