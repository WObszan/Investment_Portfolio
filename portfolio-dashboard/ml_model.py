import numpy as np
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
import shap
from sklearn.metrics import accuracy_score
import optuna

TRADING_DAYS_PER_YEAR = 252

HORIZON = 5

FEATURES = [
    "log_returns_1d", "log_returns_5d", "log_returns_20d", "log_returns_60d",
    "Volatility_14d", "Volatility_30d", "rsi",
    "MACD_line", "MACD_signal", "MACD_hist",
    "Dist_SMA50",
    "Relative_Return_20d",
    "Volume_Ratio", "OBV_Change_20d",
]

# Fixed (untuned) parameters for walk-forward folds, so no fold is tuned on its own test data
DEFAULT_XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
}


def prepare_features(df: pd.DataFrame, benchmark_df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Create stationary technical features and the `horizon`-day direction target."""
    data = df.copy()
    benchmark_data = benchmark_df.copy()
    
    #logarithmic returns
    data["log_returns_1d"] = np.log(data['Close'] / data['Close'].shift(1))
    data["log_returns_5d"] = np.log(data['Close'] / data['Close'].shift(5))
    data["log_returns_20d"] = np.log(data['Close'] / data['Close'].shift(20))
    data["log_returns_60d"] = np.log(data['Close'] / data['Close'].shift(60))

    # Rolling Volatility
    data['Volatility_14d'] = data['log_returns_1d'].rolling(14).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    data['Volatility_30d'] = data['log_returns_1d'].rolling(30).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # RSI, MACD, SMA
    data['rsi'] = ta.rsi(data['Close'], length=14)

    macd_df = ta.macd(data['Close'], fast=12, slow=26, signal=9)
    data['MACD_line'] = macd_df['MACD_12_26_9']
    data['MACD_signal'] = macd_df['MACDs_12_26_9']
    data['MACD_hist'] = macd_df['MACDh_12_26_9']

    data['SMA_50'] = ta.sma(data['Close'], length=50)
    data['SMA_200'] = ta.sma(data['Close'], length=200)

    # Price distance from the average (percentage or logarithmic)
    data['Dist_SMA50'] = (data['Close'] - data['SMA_50']) / data['SMA_50']

    data['Relative_Return_20d'] = data['log_returns_20d'] - np.log(benchmark_data['Close'] / benchmark_data['Close'].shift(20))

    data['Volume_SMA_20'] = data['Volume'].rolling(20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_20']

    # Stationary On-Balance Volume: 20-day OBV change in units of average daily volume.
    # The cumulative OBV level itself trends with time and is not used as a feature.
    obv = ta.obv(data['Close'], data['Volume'])
    data['OBV_Change_20d'] = (obv - obv.shift(20)) / data['Volume_SMA_20']

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # Target: 1 if the close `horizon` days ahead is higher than today's close
    future_close = df['Close'].shift(-horizon)
    data['Target'] = (future_close > df['Close']).astype(float)
    data.loc[future_close.isna(), 'Target'] = np.nan
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data['Target'] = data['Target'].astype(int)
    return data


def purged_split(n: int, train_frac: float = 0.6, val_frac: float = 0.2, horizon: int = HORIZON) -> tuple[slice, slice, slice]:
    """Chronological train/val/test row slices with a `horizon`-row purge gap between them.

    The target at row t looks `horizon` rows ahead, so without the gap the last training
    labels would overlap the first validation/test prices.
    """
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return slice(0, train_end), slice(train_end + horizon, val_end), slice(val_end + horizon, n)


def baseline_accuracies(y_train: pd.Series, y_test: pd.Series) -> dict:
    """Naive benchmarks: 'always up' and the training-set majority class, both scored on test."""
    majority_class = int(y_train.mean() >= 0.5)
    return {
        "always_up": float((y_test == 1).mean()),
        "majority_class": majority_class,
        "majority": float((y_test == majority_class).mean()),
    }


def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'eval_metric': 'logloss',
        'random_state': 42
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)


def train_xgboost(data: pd.DataFrame, n_trials: int = 30, horizon: int = HORIZON, seed: int = 42) -> dict:
    """Tune XGBoost with Optuna on a purged 60/20/20 split, refit on train+val, score on test.

    Returns the model, the train/test sets, SHAP values, test accuracy and naive baselines.
    """
    X = data[FEATURES]
    y = data['Target']

    train_idx, val_idx, test_idx = purged_split(len(data), 0.6, 0.2, horizon)
    X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
    y_train, y_val, y_test = y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)
    
    best_params = study.best_params
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = 42

    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train_full, y_train_full)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    return {
        "model": model,
        "X_train": X_train_full,
        "y_train": y_train_full,
        "X_test": X_test,
        "y_test": y_test,
        "shap_values": shap_values,
        "accuracy": float(accuracy_score(y_test, model.predict(X_test))),
        "baselines": baseline_accuracies(y_train_full, y_test),
        "best_params": best_params,
    }


def run_backtest(model, X_test, y_test, original_data, horizon: int = HORIZON, cost_bps: float = 5.0):
    """Backtest the ML long/flat strategy vs Buy and Hold on non-overlapping `horizon`-day steps.

    `cost_bps` is charged on every position change (entering from flat counts as one).
    """

    forward_return = np.log(original_data['Close'].shift(-horizon) / original_data['Close'])
    
    test_data = original_data.loc[X_test.index].copy()
    test_data['Forward_Return'] = forward_return.loc[X_test.index]

    #Generate model predictions for the test set
    preds = model.predict(X_test)
    test_data['Model_Signal'] = preds

    test_data = test_data.iloc[::horizon].copy()

    test_data['Strategy_Return'] = test_data['Forward_Return'] * test_data['Model_Signal']
    test_data['Strategy_Return'] = test_data['Strategy_Return'].fillna(0)
    test_data['Forward_Return'] = test_data['Forward_Return'].fillna(0)

    position_change = test_data['Model_Signal'].diff().abs()
    position_change.iloc[0] = abs(test_data['Model_Signal'].iloc[0])
    test_data['Position_Change'] = position_change
    test_data['Strategy_Return'] += position_change * np.log1p(-cost_bps / 10_000)
    
    # Calculate cumulative returns (Equity Curve)
    test_data['Market_Cumulative'] = np.exp(test_data['Forward_Return'].cumsum()) - 1
    test_data['Strategy_Cumulative'] = np.exp(test_data['Strategy_Return'].cumsum()) - 1
    
    return test_data[['Close', 'Market_Cumulative', 'Strategy_Cumulative', 'Model_Signal', 'Position_Change']]


def walk_forward_evaluation(
    data: pd.DataFrame,
    n_folds: int = 5,
    horizon: int = HORIZON,
    cost_bps: float = 5.0,
    params: dict | None = None,
) -> pd.DataFrame:
    """Expanding-window walk-forward evaluation with a `horizon`-row purge before each test block.

    Rows are split into n_folds + 1 equal blocks; fold k trains on everything before block k
    (minus the purge gap) and tests on block k. Uses fixed XGBoost parameters.
    Returns one row per fold with accuracy, 'always up' baseline, strategy and buy & hold return.
    """
    params = {**DEFAULT_XGB_PARAMS, **(params or {})}
    X = data[FEATURES]
    y = data['Target']
    n = len(data)
    block = n // (n_folds + 1)

    rows = []
    for k in range(1, n_folds + 1):
        test_start = k * block
        test_end = (k + 1) * block if k < n_folds else n
        train_end = test_start - horizon

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = X.iloc[test_start:test_end], y.iloc[test_start:test_end]

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        baselines = baseline_accuracies(y_train, y_test)
        bt = run_backtest(model, X_test, y_test, data, horizon=horizon, cost_bps=cost_bps)

        rows.append({
            "fold": k,
            "train_start": X.index[0],
            "train_end": X.index[train_end - 1],
            "test_start": X.index[test_start],
            "test_end": X.index[test_end - 1],
            "accuracy": acc,
            "baseline_accuracy": baselines["always_up"],
            "majority_accuracy": baselines["majority"],
            "edge_vs_always_up": acc - baselines["always_up"],
            "strategy_return": bt['Strategy_Cumulative'].iloc[-1],
            "buy_hold_return": bt['Market_Cumulative'].iloc[-1],
        })

    return pd.DataFrame(rows)


def summarize_walk_forward(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation across folds for the numeric walk-forward metrics."""
    cols = ["accuracy", "baseline_accuracy", "majority_accuracy", "edge_vs_always_up",
            "strategy_return", "buy_hold_return"]
    return folds[cols].agg(["mean", "std"]).T
