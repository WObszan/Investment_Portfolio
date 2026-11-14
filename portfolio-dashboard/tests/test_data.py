import numpy as np
import pandas as pd
import pytest

from data import get_daily_returns, get_split_date, split_train_test


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


def _fake_irx(values):
    dates = pd.date_range("2020-01-01", periods=len(values), freq="B")
    columns = pd.MultiIndex.from_product([["Close", "Volume"], ["^IRX"]], names=["Price", "Ticker"])
    frame = pd.DataFrame(np.column_stack([values, np.zeros(len(values))]), index=dates, columns=columns)
    return frame, ["^IRX"]


def test_risk_free_rate_is_window_average_not_last_value(monkeypatch):
    import data

    # ^IRX quoted in percent: rises from 1% to 5% over the window
    monkeypatch.setattr(data, "get_clean_data", lambda *args, **kwargs: _fake_irx(np.linspace(1.0, 5.0, 101)))

    assert data.get_risk_free_rate("2020-01-01", "2020-06-01") == pytest.approx(0.03)


def test_risk_free_rate_falls_back_when_download_fails(monkeypatch):
    import data

    monkeypatch.setattr(data, "get_clean_data", lambda *args, **kwargs: (pd.DataFrame(), []))
    assert data.get_risk_free_rate("2020-01-01", "2020-06-01") == 0.0

    def boom(*args, **kwargs):
        raise ConnectionError("no network")

    monkeypatch.setattr(data, "get_clean_data", boom)
    assert data.get_risk_free_rate("2020-01-01", "2020-06-01", fallback=0.01) == 0.01


@pytest.mark.parametrize("train_fraction", [0.5, 0.75, 0.9])
def test_split_train_test_has_no_overlap(train_fraction):
    df = pd.DataFrame({"A": np.arange(100.0)}, index=pd.date_range("2021-01-01", periods=100, freq="B"))

    split_date = get_split_date(df, train_fraction)
    train, test = split_train_test(df, split_date)

    assert train.index.intersection(test.index).empty
    assert len(train) + len(test) == len(df)
    assert train.index.max() < test.index.min()


def test_split_investable_excludes_indices():
    from data import split_investable

    investable, excluded = split_investable(["AAPL", "^GSPC", "SPY", "^VIX", "MSFT"])

    assert investable == ["AAPL", "SPY", "MSFT"]
    assert excluded == ["^GSPC", "^VIX"]


def test_categorized_tickers_defined_once():
    import callbacks
    import data

    assert callbacks.categorized_tickers is data.categorized_tickers


def test_get_clean_data_downloads_once_and_drops_empty_tickers(monkeypatch):
    import data

    calls = []
    dates = pd.date_range("2022-01-03", periods=6, freq="B")
    good = [np.nan, 10.0, 11.0, np.nan, 12.0, 13.0]  # not listed on day 1, gap on day 4

    def fake_download(tickers, **kwargs):
        calls.append(tickers)
        columns = pd.MultiIndex.from_product([["Close", "Volume"], ["GOOD", "BAD"]], names=["Price", "Ticker"])
        values = np.column_stack([good, [np.nan] * 6, [100.0] * 6, [np.nan] * 6])
        return pd.DataFrame(values, index=dates, columns=columns)

    monkeypatch.setattr(data.yf, "download", fake_download)
    data.get_clean_data.clear()

    frame, valid = data.get_clean_data(["GOOD", "BAD"], "2022-01-01", "2022-01-31")

    assert len(calls) == 1
    assert valid == ["GOOD"]
    assert list(frame["Close"].columns) == ["GOOD"]
    close = frame["Close"]["GOOD"]
    assert np.isnan(close.iloc[0])           # pre-listing stays NaN
    assert close.iloc[3] == 11.0             # post-listing gap is forward-filled
