import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import shap
import datetime

from data import (get_clean_data, get_daily_returns, get_risk_free_rate, get_split_date, split_train_test,
categorized_tickers, ticker_to_name, get_listing_dates, split_investable)

from optimization import (
    run_monte_carlo, run_efficient_frontier,
    get_min_var_portfolio, get_max_sharpe_portfolio, calculate_portfolio_growth, get_capped_return
)
from risk import (
    compute_capm, build_risk_profile, rolling_beta, compute_cumulative_value,
    compute_buy_and_hold_value, portfolio_risk_summary, strategy_metrics,
)
from clustering import cluster_assets
from viz import (
    plot_correlation_heatmap, plot_efficient_frontier, add_exact_frontier,
    plot_capm_regression, plot_rolling_beta, plot_cluster_scatter, plot_backtest_cumulative,
    plot_portfolio_growth, plot_backtest_results
)
from callbacks import (sync_from_multiselect, sync_from_pills)
from ml_model import (train_xgboost, prepare_futures, run_backtest)


### Dashoard in Streamlit ###

st.title("Portfolio Dashboard")
BENCHMARK = "^GSPC"  # SP500


all_available_tickers = list(
    dict.fromkeys(
        [t for cat_list in categorized_tickers.values() for t in cat_list]
    )
)

if "selected_tickers" not in st.session_state:
    st.session_state["selected_tickers"] = []

for category in categorized_tickers.keys():
    if f"pills_{category}" not in st.session_state:
        st.session_state[f"pills_{category}"] = []

# --- SIDEBAR UI ---
st.sidebar.subheader("Asset Selection")

# Multiselect widget with bound callback
selected = st.sidebar.multiselect(
    "Select or type company tickers:",
    options=all_available_tickers,
    accept_new_options=True,
    placeholder="Type ticker and confirm with Enter...",
    key="selected_tickers",
    on_change=sync_from_multiselect
)

st.sidebar.markdown("---")
st.sidebar.subheader("Browse by Category")

# Category pills widgets with bound callbacks
for category, tickers in categorized_tickers.items():
    st.sidebar.markdown(f"**{category}**")
    st.sidebar.pills(
        f"Select from {category}",
        options=tickers,
        selection_mode="multi",
        key=f"pills_{category}",
        label_visibility="collapsed",
        on_change=sync_from_pills,
        args=(category, tickers)  # Pass arguments to the sync_from_pills function
    )

# ------ Main UI --------
if selected:
    st.markdown("**Selected Tickers:**")
    
    badges = []
    for t in selected:
        if t in ticker_to_name:
            # If the ticker is in the dictionary: show the ticker, break the line (<br>) 
            # and use a smaller, lighter font for the company name in parentheses.
            name_html = f"<br><span style='font-size: 11px; font-weight: 400; opacity: 0.8;'>({ticker_to_name[t]})</span>"
        else:
            # If the user entered a custom ticker (e.g., via keyboard): do not add a name.
            name_html = ""
            
        badge = (
            f"<div style='display: inline-block; background-color: #2e3b4e; color: white; "
            f"padding: 6px 12px; border-radius: 12px; font-weight: 600; "
            f"margin: 0px 6px 8px 0; font-size: 14px; text-align: center; "
            f"line-height: 1.3; vertical-align: top; box-shadow: 1px 1px 3px rgba(0,0,0,0.2);'>"
            f"{t}{name_html}</div>"
        )
        badges.append(badge)

    st.markdown("".join(badges), unsafe_allow_html=True)

if st.button("Start analysis"):
    if len(selected) < 2:
        st.warning("Select minimum two companies to create a portfolio. ")
        st.stop()
    else:
        with st.spinner("Downloading data..."):
            today = datetime.date.today().strftime("%Y-%m-%d")
            data, valid_tickers = get_clean_data(selected, "2015-01-01", today)
            invalid_tickers = [t for t in selected if t not in valid_tickers]
            if invalid_tickers:
                st.error(
                    f"Unavailable or incorrectly entered tickers: "
                    f"{', '.join(invalid_tickers)}"
                )

            if data.empty:
                st.stop()
            investable, _ = split_investable(valid_tickers)
            if len(investable) < 2:
                st.warning("Select minimum two investable companies to create a portfolio "
                           "(indices such as ^GSPC are used only as benchmarks).")
                st.stop()
            close_prices = data["Close"]
            volume_data = data['Volume']
            daily_returns = get_daily_returns(close_prices)
            if len(daily_returns) < 2:
                st.error("The selected tickers have no common trading period with data. "
                         "Remove the most recently listed ticker and try again.")
                st.stop()
            risk_free_rate = get_risk_free_rate(str(daily_returns.index.min().date()),
                                                str(daily_returns.index.max().date()))
 
        # Drop results computed for a previous ticker selection
        for stale_key in ["mc_result", "exact_max_sharpe", "exact_min_var", "opt_max_weight",
                          "backtest", "ml_results", "capm", "clustered"]:
            st.session_state.pop(stale_key, None)
        st.session_state["data"] = data
        st.session_state["close_prices"] = close_prices
        st.session_state['volume_data'] = volume_data
        st.session_state["daily_returns"] = daily_returns
        st.session_state["selected"] = [t for t in selected if t in valid_tickers]
        st.session_state["today"] = today
        st.session_state['risk_free_rate'] = risk_free_rate
 
if "data" in st.session_state:
    data = st.session_state["data"]
    close_prices = st.session_state["close_prices"]
    volume_data = st.session_state['volume_data']
    daily_returns = st.session_state["daily_returns"]
    selected = st.session_state["selected"]
    today = st.session_state["today"]
    risk_free_rate = st.session_state['risk_free_rate']

    # Indices (e.g. ^GSPC) are not investable: keep them out of optimization and backtest
    investable_tickers, excluded_indices = split_investable(list(daily_returns.columns))
    asset_returns = daily_returns[investable_tickers]
    excluded_msg = (
        f"Indices ({', '.join(excluded_indices)}) are not investable and have been excluded "
        f"from the portfolio; they are used only as benchmarks." if excluded_indices else None
    )
 
    tab_eda, tab_opt, tab_proj, tab_risk, tab_cluster, tab_backtest, tab_ml = st.tabs(
        ["EDA", "Optimalization", "Projection", "Risk", "Clustering", "Backtest", "ML Alpha & Explainability"])
 
    with tab_eda:
        st.subheader("Close Prices")
        st.dataframe(data)
        st.line_chart(close_prices)
        listing_dates = get_listing_dates(close_prices)
        if listing_dates.iloc[0] > close_prices.index.min():
            st.warning(
                f"Returns, correlations, optimization and backtest use only the common period where all "
                f"selected tickers have data: from **{daily_returns.index.min().date()}**. "
                f"The latest-listed ticker is **{listing_dates.index[0]}** "
                f"(first price on {listing_dates.iloc[0].date()})."
            )
        st.write("Open market days (common period): ", len(daily_returns))
 
        st.subheader("Daily Return Correlation")
        fig_corr = plot_correlation_heatmap(daily_returns)
        st.plotly_chart(fig_corr, use_container_width=True)


    with tab_opt:
        if excluded_msg:
            st.info(excluded_msg)
        n_sim = st.slider("Select number of Monte Carlo simulations", 1000, 20000, 10000, step=1000)
        min_cap = float(np.ceil(100 / asset_returns.shape[1] / 5) * 5 / 100)
        max_weight = st.slider(
            "Max weight per asset (applies to Monte Carlo, frontier, exact portfolios and backtest)",
            min_cap, 1.0, 1.0, step=0.05, key="max_weight",
        )
 
        if st.button("Start Optimalization"):
            with st.spinner("Portfolio simulating..."):
                mc_result = run_monte_carlo(asset_returns, n_simulations=n_sim, risk_free_rate=risk_free_rate,
                                            max_weight=max_weight)
                exact_max_sharpe = get_max_sharpe_portfolio(asset_returns, risk_free_rate=risk_free_rate,
                                                            max_weight=max_weight)
                exact_min_var = get_min_var_portfolio(asset_returns, max_weight=max_weight)
            st.session_state["mc_result"] = mc_result
            st.session_state["exact_max_sharpe"] = exact_max_sharpe
            st.session_state["exact_min_var"] = exact_min_var
            st.session_state["opt_max_weight"] = max_weight
 
        if "mc_result" in st.session_state:
            mc_result = st.session_state["mc_result"]
            exact_max_sharpe = st.session_state["exact_max_sharpe"]
            exact_min_var = st.session_state["exact_min_var"]
            opt_max_weight = st.session_state["opt_max_weight"]
            asset_names = list(asset_returns.columns)
            max_sharpe_idx = mc_result["Expected Sharpe Ratio"].idxmax()
            min_risk_idx = mc_result["Expected Annual Risk"].idxmin()

            if opt_max_weight != max_weight:
                st.info(f"Results below use max weight {opt_max_weight:.0%}. "
                        "Click **Start Optimization** again to apply the new limit.")
            if not (exact_min_var["success"] and exact_max_sharpe["success"]):
                st.warning("The exact optimizer did not fully converge; treat the scipy portfolios with caution.")
 
            fig = plot_efficient_frontier(mc_result, max_sharpe_idx, min_risk_idx)

            show_exact_frontier = st.checkbox("Show the exact efficient frontier")
            if show_exact_frontier:
                with st.spinner("Calculating the exact efficient frontier..."):
                    frontier = run_efficient_frontier(asset_returns, max_weight=opt_max_weight)

                n_failed = int((~frontier["success"]).sum())
                if n_failed:
                    st.warning(f"{n_failed} of {len(frontier)} frontier points did not converge and are not plotted.")
            else:
                frontier = None
            fig = add_exact_frontier(fig, frontier, exact_max_sharpe, exact_min_var)

            st.plotly_chart(fig, use_container_width=True)
            st.caption("Stars = best Monte Carlo samples; diamonds = exact scipy portfolios. "
                       "All tables and metrics below use the exact scipy weights.")
 
            col1, col2 = st.columns(2)
            with col1:
                st.write("---- Portfolio Max Sharpe (exact) ----")
                st.metric("Sharpe Ratio (in-sample)", f"{exact_max_sharpe['Sharpe Ratio']:.2f}")
                st.dataframe(pd.DataFrame({
                    "Ticker": asset_names,
                    "Weight": exact_max_sharpe["weights"],
                }).set_index("Ticker"))
 
            with col2:
                st.write("---- Portfolio Min Risk (exact) ----")
                st.metric("Annual Risk (in-sample)", f"{exact_min_var['Annual Risk']:.2%}")
                st.dataframe(pd.DataFrame({
                    "Ticker": asset_names,
                    "Weight": exact_min_var["weights"],
                }).set_index("Ticker"))

            st.subheader("Value at Risk ( VAR ) and CVAR — in-sample")
            confidence = st.select_slider("Confidence level", options=[0.90,0.92,0.94, 0.95, 0.97, 0.99], value=0.95)
            risk_max = portfolio_risk_summary(asset_returns, exact_max_sharpe["weights"], risk_free_rate, confidence)
            risk_min = portfolio_risk_summary(asset_returns, exact_min_var["weights"], risk_free_rate, confidence)

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"--- Max Sharpe - daily VaR/CVaR: {confidence:.0%} ---")
                st.metric(f"VaR {confidence:.0%}", f"{risk_max['VaR']:.2%}")
                st.metric(f"CVaR {confidence:.0%}", f"{risk_max['CVaR']:.2%}")
 
            with col2:
                st.write(f"--- Min Risk - daily VaR/CVaR: {confidence:.0%} ---")
                st.metric(f"VaR {confidence:.0%}", f"{risk_min['VaR']:.2%}")
                st.metric(f"CVaR {confidence:.0%}", f"{risk_min['CVaR']:.2%}")
            
            st.caption(
                "VaR = maximum expected loss on a typical day at a selected confidence level,  \n"
                "(e.g., VaR 95% means that only 5% of historical days were worse than this value).  \n"
                "CVaR (Expected Shortfall) = mean loss on those worst days - it shows  \n"
                "how big losses could be if VaR is breached."
                 )

            st.subheader("Additional Risk Metrics — in-sample")
 
            col1, col2 = st.columns(2)
            with col1:
                st.write("--- Max Sharpe ---")
                st.metric("Sortino Ratio", f"{risk_max['Sortino Ratio']:.2f}")
                st.metric("Max Drawdown", f"{risk_max['Max Drawdown']:.2%}")
                st.metric("CAGR", f"{risk_max['CAGR']:.2%}")
                st.metric("Calmar Ratio", f"{risk_max['Calmar Ratio']:.2f}")
            with col2:
                st.write("--- Min Risk ---")
                st.metric("Sortino Ratio", f"{risk_min['Sortino Ratio']:.2f}")
                st.metric("Max Drawdown", f"{risk_min['Max Drawdown']:.2%}")
                st.metric("CAGR", f"{risk_min['CAGR']:.2%}")
                st.metric("Calmar Ratio", f"{risk_min['Calmar Ratio']:.2f}")
    
            st.caption(
                "In-sample: the weights were optimized on the same data these metrics are measured on, "
                "so they overstate what to expect out-of-sample (see the Backtest tab).  \n"
                "Sortino Ratio = like the Sharpe ratio, but penalizes only downside volatility (losses), not gains above the average.  \n"
                "Max Drawdown = the largest historical decline in portfolio value from peak to trough.  \n"
                "CAGR = geometric (compound) annual growth rate of the daily-rebalanced portfolio.  \n"
                "Calmar Ratio = CAGR divided by |Max Drawdown| — shows profit relative to the worst historical scenario."
            )

    with tab_proj:
        st.subheader("DCA Calculator & Portfolio Projection")
        st.write("Check how regular savings can grow your capital based on chosen portfolio strategies.")

        if "mc_result" not in st.session_state:
            st.warning("Please click **Start analysis** at the top and run the simulation in the **Optimalization** tab first!")
        else:
            exact_max_sharpe = st.session_state["exact_max_sharpe"]
            exact_min_var = st.session_state["exact_min_var"]

            ret_max_sharpe = exact_max_sharpe["Annual Return"]
            risk_max_sharpe = exact_max_sharpe["Annual Risk"]

            ret_min_risk = exact_min_var["Annual Return"]
            risk_min_risk = exact_min_var["Annual Risk"]

            strategy_choice = st.selectbox(
                "Choose portfolio strategy for projection", 
                ["Max Sharpe (Highest Return / Optimal)", "Min Risk (Safest)"]
            )

            if "Max Sharpe" in strategy_choice:
                chosen_return = ret_max_sharpe
                chosen_risk = risk_max_sharpe
            else:
                chosen_return = ret_min_risk
                chosen_risk = risk_min_risk

            if chosen_return > 0.20:
                st.warning(
                    f"Warning: The historical annual return for this strategy is exceptionally high at **{chosen_return:.2%}**. "
                    "Such high returns (often driven by past tech sector booms) rarely sustain over long multi-decade horizons."
                )
                apply_cap = st.checkbox("Apply realistic return limit (Max 15% annually)", value=True)
                if apply_cap:
                    chosen_return = min(chosen_return, 0.15)
                    st.info(f"Expected annual return has been capped to a safer long-term benchmark: **{chosen_return:.2%}**.")

            col1, col2 = st.columns(2)
            with col1:
                initial_dep = st.number_input("Initial Deposit ($)", min_value=0.0, value=10000.0, step=1000.0)
                contribution = st.number_input("Periodic Contribution Amount", min_value=0.0, value=500.0, step=100.0)
            with col2:
                frequency = st.selectbox("Contribution Frequency", ["Monthly", "Yearly"], key="proj_freq")
                years = st.slider("Investment Horizon (years)", 1, 50, 10, key="proj_years")

            growth_df = calculate_portfolio_growth(
                initial_dep, contribution, frequency, 
                chosen_return, chosen_risk, years
            )

            final_median = growth_df["Median (50th pct)"].iloc[-1]
            final_low = growth_df["Pessimistic (16th pct)"].iloc[-1]
            final_high = growth_df["Optimistic (84th pct)"].iloc[-1]
            total_invested = initial_dep + (contribution * (years * (12 if frequency == "Monthly" else 1)))
            total_profit = final_median - total_invested

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Personal Deposits", f"{total_invested:,.2f}$")
            m2.metric("Median Portfolio Value", f"{final_median:,.2f}$", delta=f"{total_profit:+,.2f}$")
            m3.metric("16th–84th Percentile Range", f"{final_low:,.0f}$ – {final_high:,.0f}$")
            m4.metric("Expected Annual Return", f"{chosen_return:.2%}")

            fig_growth = plot_portfolio_growth(growth_df, strategy_choice)
            st.plotly_chart(fig_growth, width="stretch")

            st.caption(
                "💡 *The simulation uses the historical return and annual risk of the selected portfolio. "
                "The shaded band is the 16th–84th percentile range of 5,000 Monte Carlo paths "
                "(normally distributed period returns); the solid line is the median path and the "
                "dashed line is the deterministic path at the mean return.* "
            )


    with tab_risk:
        selected_tickers = selected.copy()
        if BENCHMARK in selected_tickers:
            selected_tickers.remove(BENCHMARK)
            st.info(f"The benchmark ({BENCHMARK}) has been removed from the list of portfolio companies to avoid calculation errors.")
        ticker_to_analyze = st.selectbox("Select company for analyze", selected_tickers)
        if st.button("Count CAPM"):
            with st.spinner("Downloading benchmark and calculating regression..."):
                bench_data, _ = get_clean_data([BENCHMARK], "2015-01-01", today)
                bench_close = bench_data["Close"][BENCHMARK]
 
                combined_close = pd.concat(
                    [close_prices[ticker_to_analyze], bench_close],
                    axis=1, keys=[ticker_to_analyze, BENCHMARK]
                ).dropna()
 
                combined_returns = get_daily_returns(combined_close)
                capm = compute_capm(combined_returns, ticker_to_analyze, BENCHMARK)

                st.session_state["capm"] = capm
                st.session_state["capm_returns"] = combined_returns
                st.session_state["capm_ticker"] = ticker_to_analyze
 
        if "capm" in st.session_state and st.session_state.get("capm_ticker") == ticker_to_analyze:
            capm = st.session_state["capm"]
            combined_returns = st.session_state["capm_returns"]
 
            col1, col2 = st.columns(2)
            col1.metric("Beta", f"{capm['beta']:.3f}")
            col2.metric("Alpha (daily)", f"{capm['alpha']:.5f}")
 
            pred = capm["model"].predict(combined_returns[[BENCHMARK]])
            fig = plot_capm_regression(combined_returns, ticker_to_analyze, BENCHMARK, pred)

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Rolling Beta")
            window = st.slider("Window size (days)", 20, 120, 60, step=10)

            beta_series = rolling_beta(combined_returns, ticker_to_analyze, BENCHMARK, window).dropna()

            fig_beta = plot_rolling_beta(beta_series, ticker_to_analyze, BENCHMARK)
            st.plotly_chart(fig_beta, use_container_width=True)

 
    with tab_cluster:
        if len(selected) >= 3:
            n_clusters = st.slider("Select number of clusters", 2, len(selected), 2)
        else: 
            n_clusters = 2
 
        if st.button("Group companies (K-Means)"):
            with st.spinner("Calculating alpha/beta for each company and clustering..."):
                bench_data, _ = get_clean_data([BENCHMARK], "2015-01-01", today)
                bench_close = bench_data["Close"][BENCHMARK]
 
                combined_close = close_prices.copy()
                combined_close[BENCHMARK] = bench_close.reindex(combined_close.index).ffill()
                combined_returns = get_daily_returns(combined_close)
 
                risk_profile = build_risk_profile(combined_returns, selected, benchmark=BENCHMARK)
                clustered = cluster_assets(risk_profile, n_clusters=n_clusters)
                st.session_state["clustered"] = clustered
 
        if "clustered" in st.session_state:
            clustered = st.session_state["clustered"]
            fig = plot_cluster_scatter(clustered)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(clustered.sort_values("Cluster"))

    with tab_backtest:
        if excluded_msg:
            st.info(excluded_msg)
        st.write(
            "We examine how the weights determined during the training period (train)" 
            " would perform on data the model has never seen (test)."
        )

        train_fraction = st.slider("Proportion of training data (train)", 0.5, 0.9, 0.75, step=0.05)
 
        if st.button("Run backtest"):
            with st.spinner("Spliting data and counting portfolios..."):
                split_date = get_split_date(asset_returns, train_fraction)
                train, test = split_train_test(asset_returns, split_date)
 
                num_assets = asset_returns.shape[1]
                equal_weights = np.array([1 / num_assets] * num_assets)
 
                rf_train = get_risk_free_rate(str(train.index.min().date()), str(train.index.max().date()))
                rf_test = get_risk_free_rate(str(test.index.min().date()), str(test.index.max().date()))

                frozen_portfolio = get_max_sharpe_portfolio(train, risk_free_rate=rf_train,
                                                            max_weight=st.session_state.get("max_weight", 1.0))
                frozen_weights = frozen_portfolio["weights"]
 
                bench_data, _ = get_clean_data(
                    [BENCHMARK], split_date,
                    str((test.index.max() + pd.Timedelta(days=1)).date()),
                )
                bench_returns = get_daily_returns(bench_data["Close"])
                bench_returns = bench_returns.loc[bench_returns.index > pd.Timestamp(split_date)]

                strategies = {
                    "Max Sharpe (train weights, daily-rebalanced)": strategy_metrics(test, frozen_weights, rf_test),
                    "Equal Weight (daily-rebalanced)": strategy_metrics(test, equal_weights, rf_test),
                    f"Benchmark ({BENCHMARK})": strategy_metrics(bench_returns, np.array([1.0]), rf_test),
                }
 
                st.session_state["backtest"] = {
                    "train": train, "test": test,
                    "frozen_weights": frozen_weights, "equal_weights": equal_weights,
                    "strategies": strategies,
                    "bench_returns": bench_returns,
                    "rf_train": rf_train, "rf_test": rf_test,
                }
 
        if "backtest" in st.session_state:
            bt = st.session_state["backtest"]
 
            st.write(f"Training period: {bt['train'].index.min().date()} — {bt['train'].index.max().date()} "
                      f"({len(bt['train'])} days)")
            st.write(f"Testing period: {bt['test'].index.min().date()} — {bt['test'].index.max().date()} "
                      f"({len(bt['test'])} days)")
            st.caption(f"Risk-free rate (average ^IRX): train {bt['rf_train']:.2%}, test {bt['rf_test']:.2%}. "
                       "The train rate is used to pick the weights, the test rate for the Sharpe ratios below.")
 
            comparison = pd.DataFrame(bt["strategies"]).T.rename_axis("Strategy").reset_index()
            st.dataframe(
                comparison.style.format({
                    "Annual Return (arithmetic)": "{:.2%}", "CAGR (geometric)": "{:.2%}",
                    "Annual Risk": "{:.2%}", "Sharpe Ratio": "{:.2f}", "Max Drawdown": "{:.2%}",
                }),
                use_container_width=True,
            )
            st.caption(
                "Annual Return (arithmetic) = mean daily return × 252 (used for Sharpe). "
                "CAGR (geometric) = compound annual growth actually realized over the test period. "
                "Table metrics assume the portfolio is rebalanced back to its weights every day."
            )
 
            curves = {
                "Max Sharpe (daily-rebalanced)": compute_cumulative_value(bt["test"], bt["frozen_weights"]),
                "Max Sharpe (buy & hold, weights drift)": compute_buy_and_hold_value(bt["test"], bt["frozen_weights"]),
                "Equal Weight (daily-rebalanced)": compute_cumulative_value(bt["test"], bt["equal_weights"]),
                f"Benchmark ({BENCHMARK})": compute_cumulative_value(bt["bench_returns"], np.array([1.0])),
            }
 
            fig_bt = plot_backtest_cumulative(curves)
            st.plotly_chart(fig_bt, use_container_width=True)

    with tab_ml:
        st.subheader("Machine Learning & Explainable AI (XGBoost + SHAP)")
        st.write("Train a predictive classification model to forecast the next 5 trading days price direction.")

        ml_tickers = [t for t in selected if t != BENCHMARK]
        ticker_for_ml = st.selectbox("Select asset for ML prediction", ml_tickers, key="ml_ticker")

        if st.button('Train XGBoost model'):
            processed_data = pd.DataFrame()
            with st.spinner('Engineering features and training model...'):
                bench_data, _ = get_clean_data([BENCHMARK], "2015-01-01", today)
                bench_close = bench_data["Close"][BENCHMARK]
                bench_volume = bench_data['Volume'][BENCHMARK]

                single_asset_df = pd.DataFrame({"Close": close_prices[ticker_for_ml],
                                                "Volume": volume_data[ticker_for_ml]})
                benchmark_asset_df = pd.DataFrame({"Close": bench_close,
                                                   'Volume': bench_volume})

                processed_data = prepare_futures(single_asset_df, benchmark_asset_df)
                if len(processed_data) < 100:
                    st.error("Not enough data to train the model after dropping NaN values.")
                else:
                    model, X_test, y_test, shap_values = train_xgboost(processed_data)

                    st.session_state["ml_results"] = {
                        "model": model, "X_test": X_test, "y_test": y_test, 
                        "shap_values": shap_values, "ticker": ticker_for_ml
                    }
        if "ml_results" in st.session_state and st.session_state["ml_results"]["ticker"] == ticker_for_ml:
            res = st.session_state["ml_results"]
            model = res["model"]
            X_test = res["X_test"]
            y_test = res["y_test"]
            shap_values = res["shap_values"]

            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            st.metric("Model Test Accuracy", f"{acc:.2%}")

            latest_features = X_test.iloc[[-1]]
            latest_prob = model.predict_proba(latest_features)[0]  # [probability of drop, probability of rise]
            pred_class = model.predict(latest_features)[0]

            st.markdown("---")
            st.subheader("Model Prediction for the Next 5 Trading Days (1 Week)")
            
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                direction = "UPTREND (Bullish)" if pred_class == 1 else "DOWNTREND (Bearish)"
                st.metric("Predicted 5-Day Trend", direction)
            with col_p2:
                confidence = latest_prob[pred_class] * 100
                st.metric("Model Confidence", f"{confidence:.1f}%")

            if pred_class == 1:
                st.success(f"The model suggests that **{ticker_for_ml}**"
                           f" will close **higher** over the next 5 trading days (probability: {confidence:.1f}%).")
            else:
                st.warning(f"The model suggests caution – **{ticker_for_ml}**"
                           f"may trend **lower** over the next 5 trading days (probability: {confidence:.1f}%).")

            st.subheader("Feature Importance (SHAP Summary)")
            st.write("Shows which technical indicators had the most impact on the model's predictions.")
            
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
            st.pyplot(fig)
            plt.clf()

            #BACKTESTING EXECUTION & VISUALIZATION
            st.markdown("---")
            st.subheader("Strategy Backtest (Out-of-Sample)")
            st.write("Performance evaluation of the ML-driven classification strategy compared to standard Buy & Hold.")

            # Run backtest calculation
            backtest_results = run_backtest(model, X_test, y_test, processed_data)

            # Generate and display plot using viz.py
            fig_bt = plot_backtest_results(backtest_results)
            st.plotly_chart(fig_bt, use_container_width=True)

            # Final performance metrics
            final_strat = backtest_results['Strategy_Cumulative'].iloc[-1] * 100
            final_market = backtest_results['Market_Cumulative'].iloc[-1] * 100

            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.metric("ML Strategy Total Return", f"{final_strat:.2f}%")
            with col_b2:
                st.metric("Buy & Hold Total Return", f"{final_market:.2f}%")

    