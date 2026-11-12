import pytest

from optimization import calculate_portfolio_growth


def _col(df, prefix):
    """Find the single column whose name starts with `prefix`."""
    matches = [c for c in df.columns if c.startswith(prefix)]
    assert len(matches) == 1, f"expected one column starting with {prefix!r}, got {matches}"
    return matches[0]


def _final(df, prefix):
    return df[_col(df, prefix)].iloc[-1]


def test_dca_band_grows_like_sqrt_t_not_t():
    # 30y, 15% return, 20% risk, $10k + $500/month: Monte Carlo 16th/84th pct are ~1.17M / ~7.1M
    df = calculate_portfolio_growth(10_000, 500, "Monthly", 0.15, 0.20, 30)

    optimistic = _final(df, "Optimistic")
    pessimistic = _final(df, "Pessimistic")
    median = _final(df, "Median")

    assert optimistic / median < 10
    assert median / pessimistic < 10
    assert optimistic == pytest.approx(7.1e6, rel=0.15)
    assert pessimistic == pytest.approx(1.17e6, rel=0.15)
    assert pessimistic < median < optimistic


def test_dca_zero_risk_all_paths_equal():
    df = calculate_portfolio_growth(10_000, 500, "Monthly", 0.07, 0.0, 10)

    expected = df[_col(df, "Expected")]
    for prefix in ("Median", "Optimistic", "Pessimistic"):
        assert df[_col(df, prefix)].tolist() == pytest.approx(expected.tolist())


@pytest.mark.parametrize("frequency, periods_per_year", [("Monthly", 12), ("Yearly", 1)])
def test_dca_zero_return_final_value_equals_deposits(frequency, periods_per_year):
    years = 5
    df = calculate_portfolio_growth(10_000, 500, frequency, 0.0, 0.0, years)

    total_deposits = 10_000 + 500 * years * periods_per_year
    for prefix in ("Expected", "Median", "Optimistic", "Pessimistic"):
        assert _final(df, prefix) == pytest.approx(total_deposits)


@pytest.mark.parametrize("frequency", ["Monthly", "Yearly"])
def test_dca_years_column_matches_horizon(frequency):
    # The x-axis must end at the chosen horizon for both contribution frequencies
    df = calculate_portfolio_growth(10_000, 500, frequency, 0.05, 0.1, 7)
    assert df["Years"].iloc[-1] == pytest.approx(7)


def test_dca_is_reproducible_with_fixed_seed():
    a = calculate_portfolio_growth(10_000, 500, "Monthly", 0.1, 0.2, 10)
    b = calculate_portfolio_growth(10_000, 500, "Monthly", 0.1, 0.2, 10)
    assert a.equals(b)
