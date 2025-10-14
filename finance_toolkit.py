# ------------------------------------------------------------
# FINANCE TOOLKIT BY @amiravalles
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Financial Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.title("📈 Financial Portfolio Dashboard")
st.markdown("""
This app allows you to analyze stocks, visualize correlations,
and calculate portfolio risk and return interactively using **Python** and **Streamlit**.
""")

# ------------------------------------------------------------
# SIDEBAR - USER INPUTS
# ------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

tickers_input = st.sidebar.text_input(
    "Enter tickers separated by commas:",
    value="AAPL, MSFT, GOOGL, TSLA"
)
selected_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

start_date = st.sidebar.date_input("Start date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("today"))

# ------------------------------------------------------------
# DOWNLOAD DATA
# ------------------------------------------------------------
if selected_tickers:
    data = yf.download(selected_tickers, start=start_date, end=end_date, auto_adjust=False)["Adj Close"]

    st.subheader("Stock Prices (Adjusted Close)")
    st.plotly_chart(px.line(data, title="Stock Prices Over Time"), use_container_width=True)

    # ------------------------------------------------------------
    # NORMALIZED PRICES
    # ------------------------------------------------------------
    st.subheader("Normalized Prices (Base = 100)")
    norm_data = (data / data.iloc[0]) * 100
    st.plotly_chart(px.line(norm_data, title="Normalized Stock Performance"), use_container_width=True)

    # ------------------------------------------------------------
    # LOG RETURNS
    # ------------------------------------------------------------
    log_returns = np.log(data / data.shift(1)).dropna()

    st.subheader("Annualized Log Returns (%)")
    annual_returns = log_returns.mean() * 250 * 100
    st.write(annual_returns.round(2))

    # ------------------------------------------------------------
    # COVARIANCE & CORRELATION MATRICES
    # ------------------------------------------------------------
    st.subheader("Covariance Matrix (Annualized)")
    st.write((log_returns.cov() * 250).round(4))

    st.subheader("Correlation Matrix (Heatmap)")
    corr = log_returns.corr()
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=[[0, "#0E4C92"], [0.5, "#0D0D0D"], [1, "#E63946"]],
        title="Correlation Matrix",
    )
    fig_corr.update_layout(template="plotly_dark", font=dict(color="white", size=13))
    st.plotly_chart(fig_corr, use_container_width=True)

    # ------------------------------------------------------------
    # PORTFOLIO CALCULATIONS
    # ------------------------------------------------------------
    st.subheader("Portfolio Analysis")

    num_assets = len(selected_tickers)
    default_weights = np.ones(num_assets) / num_assets

    # User weight input
    manual_weights = st.text_input(
        "Enter custom weights (comma-separated, must sum to 1)",
        value=", ".join([f"{w:.2f}" for w in default_weights])
    )

    try:
        weights = np.array([float(w) for w in manual_weights.split(",")])
        if not np.isclose(weights.sum(), 1):
            st.warning("⚠️ The weights must sum to 1. Normalizing automatically.")
            weights = weights / np.sum(weights)
    except:
        st.error("Please enter valid numeric weights.")
        weights = default_weights

    if st.button("🎲 Randomize Weights"):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        st.session_state["weights"] = weights
        st.success(f"New random weights: {weights.round(3)}")

    # Portfolio metrics
    portfolio_return = np.sum(weights * log_returns.mean()) * 250 * 100
    portfolio_var = np.dot(weights.T, np.dot(log_returns.cov() * 250, weights))
    portfolio_vol = np.sqrt(portfolio_var) * 100

    st.metric(label="Expected Annual Return", value=f"{portfolio_return:.2f}%")
    st.metric(label="Expected Volatility", value=f"{portfolio_vol:.2f}%")
    st.metric(label="Expected Variance", value=f"{portfolio_var:.4f}")

    # Pie chart for weights
    fig_weights = go.Figure(data=[go.Pie(labels=selected_tickers, values=weights, hole=0.4)])
    fig_weights.update_layout(title="Portfolio Weights Distribution")
    st.plotly_chart(fig_weights, use_container_width=True)

else:
    st.warning("Please enter at least one valid ticker symbol.")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.markdown("""
**Created by [Agustín Miravalles](https://www.linkedin.com/in/amiravalles/)**  
🔗 [GitHub](https://github.com/amiravalles) | 📧 miravalles.agustin@gmail.com
""")
