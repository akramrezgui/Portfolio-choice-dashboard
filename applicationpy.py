import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.title("📈 Portfolio Optimization Dashboard")

# Sidebar inputs
assets_text = st.sidebar.text_input("Enter Assets (comma separated):", "AAPL,MSFT,TSLA,NVDA")
num_portfolios = st.sidebar.slider("Number of Portfolios:", 1000, 20000, 5000, 1000)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%):", value=2.0, step=0.1)
benchmark_ticker = st.sidebar.text_input("Benchmark Ticker:", "^GSPC")

assets = [a.strip() for a in assets_text.split(",")]

# Download data
data = yf.download(assets + [benchmark_ticker], start="2020-01-01", end="2026-01-01")

# ✅ Handle Adj Close vs Close
if "Adj Close" in data.columns:
    prices = data["Adj Close"][assets]
    benchmark_prices = data["Adj Close"][benchmark_ticker]
else:
    prices = data["Close"][assets]
    benchmark_prices = data["Close"][benchmark_ticker]

returns = prices.pct_change().dropna()
benchmark_returns = benchmark_prices.pct_change().dropna()

# Portfolio statistics
mean_returns = returns.mean().to_numpy()
cov_matrix = returns.cov().to_numpy()

results = np.zeros((3, num_portfolios))
weights_record = []
rf = risk_free_rate / 100

for i in range(num_portfolios):
    weights = np.random.dirichlet(np.ones(len(assets)))
    port_return = np.dot(weights, mean_returns)
    port_std = np.sqrt(weights @ cov_matrix @ weights)
    sharpe = (port_return - rf) / port_std
    results[0,i] = port_return
    results[1,i] = port_std
    results[2,i] = sharpe
    weights_record.append(weights)

# Best portfolio
max_sharpe_idx = results[2].argmax()
best_return, best_std, best_sharpe = results[:, max_sharpe_idx]
best_weights = weights_record[max_sharpe_idx]

# Efficient frontier plot
fig_frontier = px.scatter(x=results[1,:], y=results[0,:], color=results[2,:],
                          labels={'x':'Risk (Std Dev)','y':'Return','color':'Sharpe Ratio'},
                          title="Efficient Frontier")
fig_frontier.add_scatter(x=[best_std], y=[best_return], mode='markers',
                         marker=dict(color='red', size=12, symbol='star'),
                         name='Best Portfolio')
st.plotly_chart(fig_frontier)

# Best portfolio weights
st.subheader("🔑 Best Portfolio Weights")
st.dataframe(pd.DataFrame({'Asset': assets, 'Weight': best_weights}))
st.write(f"**Mean Return:** {best_return:.4f}, **Std Dev:** {best_std:.4f}, **Sharpe:** {best_sharpe:.4f}")

# Benchmark comparison
st.subheader("📊 Portfolio vs Benchmark Cumulative Returns")
portfolio_returns = (returns @ best_weights)
portfolio_cum = (1 + portfolio_returns).cumprod()
benchmark_cum = (1 + benchmark_returns).cumprod()

fig_benchmark = go.Figure()
fig_benchmark.add_trace(go.Scatter(y=portfolio_cum, x=portfolio_cum.index,
                                   mode='lines', name='Optimized Portfolio'))
fig_benchmark.add_trace(go.Scatter(y=benchmark_cum, x=benchmark_cum.index,
                                   mode='lines', name=benchmark_ticker))
fig_benchmark.update_layout(title="Portfolio vs Benchmark Cumulative Returns",
                            xaxis_title="Date", yaxis_title="Cumulative Return")
st.plotly_chart(fig_benchmark)

# Rolling Sharpe ratio
st.subheader("📉 Rolling Sharpe Ratio (90-day window)")
rolling_window = 90
rolling_sharpe = (portfolio_returns.rolling(rolling_window).mean() - rf/252) / \
                 portfolio_returns.rolling(rolling_window).std()

fig_sharpe = go.Figure()
fig_sharpe.add_trace(go.Scatter(y=rolling_sharpe, x=rolling_sharpe.index,
                                mode='lines', name='Rolling Sharpe'))
fig_sharpe.update_layout(title="Rolling Sharpe Ratio",
                         xaxis_title="Date", yaxis_title="Sharpe Ratio")
st.plotly_chart(fig_sharpe)
