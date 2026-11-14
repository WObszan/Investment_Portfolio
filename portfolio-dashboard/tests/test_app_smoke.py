"""End-to-end smoke test: run app.py offline (fake yfinance) and click every tab's action."""
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
LATE_LISTED = "LATE"


def _fake_download(tickers, start=None, end=None, **kwargs):
    """Deterministic synthetic OHLCV in yfinance's (Price, Ticker) MultiIndex layout."""
    if isinstance(tickers, str):
        tickers = [tickers]
    dates = pd.bdate_range(start, pd.Timestamp(end) - pd.Timedelta(days=1))
    frames = {}
    for t in tickers:
        rng = np.random.default_rng(zlib.crc32(t.encode()))
        if t == "^IRX":
            close = 2.0 + 0.1 * rng.standard_normal(len(dates))
        else:
            close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(dates))))
        if t == LATE_LISTED:
            close[: len(dates) // 2] = np.nan
        volume = rng.integers(1_000_000, 5_000_000, len(dates)).astype(float)
        volume[np.isnan(close)] = np.nan
        frames[t] = pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close, "Volume": volume}, index=dates
        )
    data = pd.concat(frames, axis=1).swaplevel(axis=1).sort_index(axis=1)
    data.columns.names = ["Price", "Ticker"]
    return data


def _assert_no_exception(at: AppTest, step: str) -> None:
    assert not at.exception, f"{step}: {[e.value for e in at.exception]}"


def _click(at: AppTest, label: str) -> None:
    next(b for b in at.button if b.label == label).click().run()
    _assert_no_exception(at, label)


@pytest.mark.filterwarnings("ignore")
def test_all_tabs_render_and_actions_run(monkeypatch):
    monkeypatch.setattr("yfinance.download", _fake_download)

    at = AppTest.from_file(str(APP_PATH), default_timeout=300)
    at.run()
    _assert_no_exception(at, "initial render")

    at.multiselect(key="selected_tickers").set_value(["AAPL", "MSFT", LATE_LISTED, "^GSPC"]).run()
    _click(at, "Start analysis")

    assert [t.label for t in at.tabs] == [
        "EDA", "Optimization", "Projection", "Risk", "Clustering", "Backtest", "ML Alpha & Explainability",
    ]
    assert any(LATE_LISTED in w.value for w in at.warning), "EDA should warn about the common period"
    assert any("^GSPC" in i.value and "not investable" in i.value for i in at.info)

    _click(at, "Start Optimization")
    at.checkbox[0].check().run()
    _assert_no_exception(at, "exact frontier")
    _click(at, "Count CAPM")
    _click(at, "Group companies (K-Means)")
    _click(at, "Run backtest")
    _click(at, "Train XGBoost model")
    _click(at, "Run walk-forward evaluation")

    labels = {m.label for m in at.metric}
    assert {"Median Portfolio Value", "'Always up' baseline", "Model − 'always up'", "CAGR"} <= labels
