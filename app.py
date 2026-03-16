import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Financial Forecasting Dashboard", layout="wide")

st.title("Stock Return & Volatility Forecasting Dashboard")

st.sidebar.header("User Input")

ticker = st.sidebar.text_input("Stock Symbol", "^NSEI")

start = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))

if st.sidebar.button("Load Data"):

    data = yf.download(ticker, start=start, end=end)

    if data.empty:
        st.error("No data found. Please check the stock symbol.")
        st.stop()

    # Candlestick chart
    st.subheader("Candlestick Chart")

    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])

    st.plotly_chart(fig, use_container_width=True)

    # Returns and lag features
    data["Returns"] = np.log(data["Close"] / data["Close"].shift(1))
    data["Lag1"] = data["Returns"].shift(1)
    data["Lag2"] = data["Returns"].shift(2)

    data.dropna(inplace=True)

    split = int(len(data) * 0.8)

    train = data.iloc[:split]
    test = data.iloc[split:]

    st.subheader("Model Comparison")

    results = {}

    # ARIMA
    arima = ARIMA(train["Close"], order=(1,1,1)).fit()

    arima_pred = arima.forecast(steps=len(test))
    arima_pred.index = test.index

    arima_rmse = np.sqrt(mean_squared_error(test["Close"], arima_pred))
    results["ARIMA"] = arima_rmse

    # Random Forest
    X_train = train[["Lag1","Lag2"]]
    y_train = train["Returns"]

    X_test = test[["Lag1","Lag2"]]
    y_test = test["Returns"]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)

    results["Random Forest"] = np.sqrt(mean_squared_error(y_test, rf_pred))

    # XGBoost
    xgb = XGBRegressor(n_estimators=100, random_state=42)

    xgb.fit(X_train, y_train)

    xgb_pred = xgb.predict(X_test)

    results["XGBoost"] = np.sqrt(mean_squared_error(y_test, xgb_pred))

    # Results table
    results_df = pd.DataFrame(list(results.items()), columns=["Model","RMSE"])

    st.table(results_df)

    # Model comparison chart
    st.subheader("Model Performance Comparison")

    fig2 = go.Figure()

    fig2.add_bar(
        x=results_df["Model"],
        y=results_df["RMSE"]
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Volatility heatmap
    st.subheader("Volatility Heatmap")

    data["Volatility"] = data["Returns"].rolling(20).std()

    vol = data["Volatility"].dropna()

    fig3, ax = plt.subplots()

    sns.heatmap(vol.to_frame().T, cmap="coolwarm", cbar=True)

    st.pyplot(fig3)

    # GARCH volatility forecast
    st.subheader("GARCH Volatility Forecast")

    returns = train["Returns"] * 100

    garch = arch_model(returns, vol="GARCH", p=1, q=1)

    garch_fit = garch.fit(disp="off")

    forecast = garch_fit.forecast(horizon=1)

    vol_pred = np.sqrt(forecast.variance.values[-1,0])

    st.metric("Next Day Volatility", round(vol_pred,4))
