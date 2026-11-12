import numpy as np
import pandas as pd
import pytest

from ml_model import (
    FEATURES,
    baseline_accuracies,
    prepare_features,
    purged_split,
    run_backtest,
    walk_forward_evaluation,
)


class FakeModel:
    """Mock model to avoid training real models during unit testing."""
    def __init__(self, prediction: int):
        self.prediction = prediction

    def predict(self, X):
        return np.full(len(X), self.prediction)


def _make_price_series(n=260, seed=1):
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    price = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))
    volume = np.random.randint(1_000_000, 5_000_000, n)
    return pd.DataFrame({"Close": price, "Volume": volume}, index=dates)


#prepare_features
def test_prepare_features_drops_last_horizon_rows_instead_of_faking_target():
    # Verify that target contains no NaNs and that the last result date is earlier than input
    df = _make_price_series()
    benchmark_df = _make_price_series(seed=2)

    result = prepare_features(df, benchmark_df)

    assert not result["Target"].isna().any()
    assert set(result["Target"].unique()).issubset({0, 1})
    assert result.index.max() < df.index.max()


#run_backtest
def test_run_backtest_zero_signal_gives_zero_strategy_return():
    original_data = _make_price_series(n=40, seed=3)
    X_test = pd.DataFrame(index=original_data.index[:30], data={"dummy": range(30)})
    y_test = pd.Series([0] * 30, index=X_test.index)

    result = run_backtest(FakeModel(prediction=0), X_test, y_test, original_data, horizon=5)

    assert result["Strategy_Cumulative"].iloc[-1] == pytest.approx(0.0)


def test_run_backtest_always_buy_matches_buy_and_hold():
    original_data = _make_price_series(n=40, seed=3)
    X_test = pd.DataFrame(index=original_data.index[:30], data={"dummy": range(30)})
    y_test = pd.Series([0] * 30, index=X_test.index)

    result = run_backtest(FakeModel(prediction=1), X_test, y_test, original_data, horizon=5, cost_bps=0)

    max_diff = (result["Strategy_Cumulative"] - result["Market_Cumulative"]).abs().max()
    assert max_diff < 1e-9


def test_run_backtest_resamples_non_overlapping_every_horizon_days():
    original_data = _make_price_series(n=40, seed=3)
    X_test = pd.DataFrame(index=original_data.index[:30], data={"dummy": range(30)})
    y_test = pd.Series([0] * 30, index=X_test.index)

    result = run_backtest(FakeModel(prediction=1), X_test, y_test, original_data, horizon=5)

    # 30 test days / horizon=5 yields 6 non-overlapping evaluation points
    assert len(result) == 6


class PatternModel:
    """Predicts 1 for the first `horizon` rows, 0 for the next `horizon`, and so on."""
    def __init__(self, horizon: int = 5):
        self.horizon = horizon

    def predict(self, X):
        return ((np.arange(len(X)) // self.horizon) % 2 == 0).astype(int)


def _random_walk(n, seed, drift=0.0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    price = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({"Close": price, "Volume": volume}, index=dates)


# features
def test_features_exclude_non_stationary_and_calendar_columns():
    assert not {"OBV", "Month", "Quarter"} & set(FEATURES)

    result = prepare_features(_make_price_series(), _make_price_series(seed=2))
    assert set(FEATURES).issubset(result.columns)


# purge gap
def test_purged_split_leaves_horizon_gap_between_sets():
    train, val, test = purged_split(n=1000, train_frac=0.6, val_frac=0.2, horizon=5)

    assert train.stop + 5 <= val.start
    assert val.stop + 5 <= test.start
    assert test.stop == 1000


# baselines
def test_baseline_accuracies_always_up_and_majority():
    y_train = pd.Series([0, 0, 0, 1])     # majority class in train = 0
    y_test = pd.Series([1, 1, 1, 0, 0])   # 60% up in test

    baselines = baseline_accuracies(y_train, y_test)

    assert baselines["always_up"] == pytest.approx(0.6)
    assert baselines["majority_class"] == 0
    assert baselines["majority"] == pytest.approx(0.4)


# transaction costs
def test_run_backtest_charges_cost_per_position_change():
    original_data = _make_price_series(n=40, seed=3)
    X_test = pd.DataFrame(index=original_data.index[:30], data={"dummy": range(30)})
    y_test = pd.Series([0] * 30, index=X_test.index)

    free = run_backtest(PatternModel(), X_test, y_test, original_data, horizon=5, cost_bps=0)
    costly = run_backtest(PatternModel(), X_test, y_test, original_data, horizon=5, cost_bps=10)

    # Signals 1,0,1,0,1,0 -> 6 position changes (entry from flat counts as one)
    expected = (1 + free["Strategy_Cumulative"].iloc[-1]) * (1 - 0.001) ** 6 - 1
    assert costly["Strategy_Cumulative"].iloc[-1] == pytest.approx(expected)
    assert costly["Market_Cumulative"].iloc[-1] == pytest.approx(free["Market_Cumulative"].iloc[-1])


def test_run_backtest_default_cost_is_5_bps():
    original_data = _make_price_series(n=40, seed=3)
    X_test = pd.DataFrame(index=original_data.index[:30], data={"dummy": range(30)})
    y_test = pd.Series([0] * 30, index=X_test.index)

    result = run_backtest(FakeModel(prediction=1), X_test, y_test, original_data, horizon=5)

    expected = (1 + result["Market_Cumulative"].iloc[-1]) * (1 - 0.0005) - 1
    assert result["Strategy_Cumulative"].iloc[-1] == pytest.approx(expected)


# walk-forward
def test_walk_forward_returns_one_row_per_fold():
    data = prepare_features(_random_walk(1200, seed=11), _random_walk(1200, seed=12))

    folds = walk_forward_evaluation(data, n_folds=3)

    assert len(folds) == 3
    for col in ["accuracy", "baseline_accuracy", "strategy_return", "buy_hold_return"]:
        assert col in folds.columns
    # Expanding window, purged: every fold trains strictly before it tests
    assert (folds["train_end"] < folds["test_start"]).all()
    assert folds["train_start"].nunique() == 1


def test_walk_forward_on_random_walk_shows_no_skill():
    # Sanity check against leakage: on pure noise the model cannot beat "always up" by much
    data = prepare_features(_random_walk(3000, seed=21), _random_walk(3000, seed=22))

    folds = walk_forward_evaluation(data, n_folds=5)

    edge = folds["accuracy"].mean() - folds["baseline_accuracy"].mean()
    assert abs(edge) <= 0.05
