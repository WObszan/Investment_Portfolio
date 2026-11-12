import pandas as pd
import streamlit as st
import yfinance as yf


categorized_tickers = {
    "Benchmark": [
        "^GSPC",
        "^IXIC",
        "^DJI",
        "^RUT",
        "^VIX",
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "ACWI",
    ],
    "Big Tech": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "PLTR",
        "NFLX",
        "AMD",
    ],
    "Finance": [
        "JPM",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "BLK",
        "AXP",
        "V",
        "MA",
        "BRK-B",
    ],
    "Healthcare": [
        "JNJ",
        "UNH",
        "PFE",
        "ABBV",
        "MRK",
        "LLY",
        "TMO",
        "ABT",
        "AMGN",
        "GILD",
    ],
    "Consumer Goods": [
        "WMT",
        "PG",
        "KO",
        "PEP",
        "COST",
        "MCD",
        "DIS",
        "NKE",
        "SBUX",
        "HD",
    ],
    "Energy & Industry": [
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "EOG",
        "BA",
        "CAT",
        "HON",
        "UPS",
        "GE",
    ],
}

ticker_to_name = {
    # --- Benchmark & Indices ---
    "^GSPC": "S&P 500 Index",
    
    # --- Big Tech & Growth ---
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com, Inc.",
    "NVDA": "NVIDIA Corporation",
    "META": "Meta Platforms, Inc.",
    "TSLA": "Tesla, Inc.",
    "NFLX": "Netflix, Inc.",
    "AMD": "Advanced Micro Devices",
    "INTC": "Intel Corporation",
    "ADBE": "Adobe Inc.",
    "CRM": "Salesforce, Inc.",
    "CSCO": "Cisco Systems, Inc.",
    "QCOM": "QUALCOMM Incorporated",
    "IBM": "International Business Machines",
    "ORCL": "Oracle Corporation",
    "NOW": "ServiceNow, Inc.",
    "TXN": "Texas Instruments Inc.",
    "AVGO": "Broadcom Inc.",
    "INTU": "Intuit Inc.",
    "PYPL": "PayPal Holdings, Inc.",
    "UBER": "Uber Technologies, Inc.",
    "ABNB": "Airbnb, Inc.",
    "SHOP": "Shopify Inc.",
    "SNOW": "Snowflake Inc.",
    "PLTR": "Palantir Technologies",
    "SPOT": "Spotify Technology S.A.",
    
    # --- Finance & Banking ---
    "JPM": "JPMorgan Chase & Co.",
    "BAC": "Bank of America Corp.",
    "WFC": "Wells Fargo & Company",
    "C": "Citigroup Inc.",
    "GS": "Goldman Sachs Group",
    "MS": "Morgan Stanley",
    "BLK": "BlackRock, Inc.",
    "AXP": "American Express Company",
    "V": "Visa Inc.",
    "MA": "Mastercard Incorporated",
    "SCHW": "Charles Schwab Corp.",
    "SPGI": "S&P Global Inc.",
    "BRK-B": "Berkshire Hathaway Inc.",
    
    # --- Healthcare & Pharma ---
    "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth Group",
    "PFE": "Pfizer Inc.",
    "ABBV": "AbbVie Inc.",
    "MRK": "Merck & Co., Inc.",
    "LLY": "Eli Lilly and Company",
    "TMO": "Thermo Fisher Scientific",
    "ABT": "Abbott Laboratories",
    "DHR": "Danaher Corporation",
    "BMY": "Bristol-Myers Squibb",
    "AMGN": "Amgen Inc.",
    "GILD": "Gilead Sciences, Inc.",
    "CVS": "CVS Health Corporation",
    "ISRG": "Intuitive Surgical, Inc.",
    
    # --- Consumer Goods & Retail ---
    "WMT": "Walmart Inc.",
    "PG": "Procter & Gamble Co.",
    "KO": "Coca-Cola Company",
    "PEP": "PepsiCo, Inc.",
    "COST": "Costco Wholesale Corp.",
    "MCD": "McDonald's Corporation",
    "DIS": "Walt Disney Company",
    "NKE": "NIKE, Inc.",
    "SBUX": "Starbucks Corporation",
    "TGT": "Target Corporation",
    "LOW": "Lowe's Companies, Inc.",
    "HD": "Home Depot, Inc.",
    "PM": "Philip Morris International",
    "MO": "Altria Group, Inc.",
    "CL": "Colgate-Palmolive Company",
    "EL": "Estée Lauder Companies",
    
    # --- Industrial & Energy & Defense ---
    "XOM": "Exxon Mobil Corporation",
    "CVX": "Chevron Corporation",
    "COP": "ConocoPhillips",
    "SLB": "Schlumberger N.V.",
    "EOG": "EOG Resources, Inc.",
    "BA": "Boeing Company",
    "CAT": "Caterpillar Inc.",
    "HON": "Honeywell International",
    "UPS": "United Parcel Service",
    "FDX": "FedEx Corporation",
    "GE": "General Electric Company",
    "RTX": "RTX Corporation",
    "LMT": "Lockheed Martin Corp.",
    "NOC": "Northrop Grumman Corp.",
    "DE": "Deere & Company",
    "MMM": "3M Company",
    
    # --- Telecom & Utilities & Others ---
    "T": "AT&T Inc.",
    "VZ": "Verizon Communications",
    "TMUS": "T-Mobile US, Inc.",
    "NEE": "NextEra Energy, Inc.",
    "DUK": "Duke Energy Corporation",
    "SO": "Southern Company",
    "AMAT": "Applied Materials, Inc.",
    "LRCX": "Lam Research Corporation",
    "MU": "Micron Technology, Inc.",
    "BKNG": "Booking Holdings Inc.",
    "MDLZ": "Mondelez International",
    "ZTS": "Zoetis Inc.",
    "BDX": "Becton, Dickinson and Company",
}

@st.cache_data(show_spinner=False)
def get_clean_data(tickers: list[str], start: str, end: str):
  if not tickers:
    return pd.DataFrame(), []

  valid_tickers = []
  
  for t in tickers:
    try:
      test_df = yf.download(t, period="5d", progress=False, threads=False)
      if not test_df.empty and "Close" in test_df.columns:
        valid_tickers.append(t)
    except Exception:
      continue

  if not valid_tickers:
    return pd.DataFrame(), []

  data = yf.download(valid_tickers, start=start, end=end, progress=False, threads=False)

  if data.empty:
    return pd.DataFrame(), []

  if len(valid_tickers) == 1:
    data["Volume"] = data["Volume"].fillna(0)
    data = data.ffill()
    return data, valid_tickers

  try:
    available_tickers = data["Close"].columns.tolist()
  except KeyError:
    return pd.DataFrame(), []

  data["Volume"] = data["Volume"].fillna(0)
  data = data.ffill()

  return data, available_tickers


@st.cache_data(show_spinner=False)
def get_daily_returns(close_prices: pd.DataFrame) -> pd.DataFrame:
    """Daily percentage returns over the common period where every ticker has data.

    Rows before the latest listing date are dropped instead of being filled with 0%,
    which would understate volatility and correlation. Isolated gaps after listing are
    already forward-filled on prices in `get_clean_data`.
    """
    returns = close_prices.pct_change(fill_method=None)
    return returns.dropna(how="any")


def get_listing_dates(close_prices: pd.DataFrame) -> pd.Series:
    """First date with a valid close price for every ticker, latest first."""
    return close_prices.apply(lambda col: col.first_valid_index()).sort_values(ascending=False)


def split_investable(tickers: list[str]) -> tuple[list[str], list[str]]:
    """Split tickers into investable assets and non-investable indices (symbols starting with '^')."""
    investable = [t for t in tickers if not t.startswith("^")]
    excluded = [t for t in tickers if t.startswith("^")]
    return investable, excluded


def get_risk_free_rate(start: str, end: str, fallback: float = 0.0) -> float:
    """Average annualized 13-week T-bill yield (^IRX) over [start, end], as a decimal.

    Using the window average (instead of the latest quote) avoids look-ahead when the
    rate is applied to a historical period. `end` is inclusive. If the download fails or
    returns no data, `fallback` is returned and a Streamlit warning is shown.
    """
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        data, _ = get_clean_data(["^IRX"], start, end_exclusive)
        close = data["Close"].squeeze().dropna() if not data.empty else pd.Series(dtype=float)
    except Exception:
        close = pd.Series(dtype=float)

    if close.empty:
        st.warning(
            f"Could not download the risk-free rate (^IRX) for {start} – {end}; "
            f"using {fallback:.2%} instead."
        )
        return float(fallback)
    return float(close.mean() / 100)


def get_split_date(daily_returns: pd.DataFrame, train_fraction: float = 0.75) -> str:
    daily_returns_len = len(daily_returns)
    split_date_index= int(daily_returns_len * train_fraction)
    split_timestamp = daily_returns.index[split_date_index]
    split_date = split_timestamp.strftime("%Y-%m-%d")
    return split_date
                                  
                               
def split_train_test(daily_returns: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: train includes `split_date`, test starts strictly after it."""
    split_ts = pd.Timestamp(split_date)
    train_daily_returns = daily_returns.loc[daily_returns.index <= split_ts]
    test_daily_returns = daily_returns.loc[daily_returns.index > split_ts]
    return train_daily_returns, test_daily_returns
