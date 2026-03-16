import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------

st.set_page_config(
    page_title="Financial Market Forecasting Dashboard",
    layout="wide",
    page_icon="📈"
)

st.title("📈 Financial Market Forecasting Dashboard")

st.markdown(
"""
This dashboard analyzes **stock returns and volatility** using  
statistical models and machine learning algorithms.

Models Included:

• ARIMA  
• Random Forest  
• XGBoost  
• GARCH Volatility Model
"""
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("Dashboard Controls")

ticker = st.sidebar.text_input("Stock Symbol", "^NSEI")

start = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))

load_data = st.sidebar.button("Load Market Data")

# ---------------------------------------------------------
# Main App
# ---------------------------------------------------------

if load_data:

    with st.spinner("Fetching data and training models..."):

        data = yf.download(ticker, start=start, end=end)

        if data.empty:
            st.error("No data found.")
            st.stop()

        # -----------------------------------------------------
        # Price Chart
        # -----------------------------------------------------

        st.subheader("📊 Market Price Chart")

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price"
        ))

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------------------------------
        # Feature Engineering
        # -----------------------------------------------------

        data["Returns"] = np.log(data["Close"] / data["Close"].shift(1))
        data["Lag1"] = data["Returns"].shift(1)
        data["Lag2"] = data["Returns"].shift(2)

        data.dropna(inplace=True)

        split = int(len(data) * 0.8)

        train = data.iloc[:split]
        test = data.iloc[split:]

        # -----------------------------------------------------
        # Model Training
        # -----------------------------------------------------

        st.subheader("🤖 Model Performance")

        results = {}

        # ARIMA

        arima = ARIMA(train["Returns"], order=(1,0,1)).fit()

        arima_pred = arima.forecast(steps=len(test))
        arima_pred.index = test.index

        arima_rmse = np.sqrt(mean_squared_error(test["Returns"], arima_pred))
        results["ARIMA"] = arima_rmse

        # Random Forest

        X_train = train[["Lag1","Lag2"]]
        y_train = train["Returns"]

        X_test = test[["Lag1","Lag2"]]
        y_test = test["Returns"]

        rf = RandomForestRegressor(n_estimators=100, random_state=42)

        rf.fit(X_train, y_train)

        rf_pred = rf.predict(X_test)

        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

        results["Random Forest"] = rf_rmse

        # XGBoost

        xgb = XGBRegressor(n_estimators=100, random_state=42)

        xgb.fit(X_train, y_train)

        xgb_pred = xgb.predict(X_test)

        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

        results["XGBoost"] = xgb_rmse

        results_df = pd.DataFrame(
            list(results.items()),
            columns=["Model","RMSE"]
        )

        st.dataframe(results_df)

        # -----------------------------------------------------
        # Model Comparison Chart
        # -----------------------------------------------------

        fig3 = px.bar(
            results_df,
            x="Model",
            y="RMSE",
            color="Model",
            title="Model Performance Comparison"
        )

        st.plotly_chart(fig3, use_container_width=True)

        # -----------------------------------------------------
        # Actual vs Predicted (ARIMA)
        # -----------------------------------------------------

        st.subheader("📉 Actual vs Predicted Returns")

        fig_pred = go.Figure()

        fig_pred.add_trace(
            go.Scatter(
                x=test.index,
                y=test["Returns"],
                mode="lines",
                name="Actual Returns"
            )
        )

        fig_pred.add_trace(
            go.Scatter(
                x=arima_pred.index,
                y=arima_pred,
                mode="lines",
                name="ARIMA Prediction"
            )
        )

        st.plotly_chart(fig_pred, use_container_width=True)

        # -----------------------------------------------------
        # Rolling Volatility
        # -----------------------------------------------------

        st.subheader("📊 Rolling Market Volatility")

        data["Volatility"] = data["Returns"].rolling(20).std()

        fig_vol = go.Figure()

        fig_vol.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Volatility"],
                mode="lines",
                name="Rolling Volatility"
            )
        )

        st.plotly_chart(fig_vol, use_container_width=True)

        # -----------------------------------------------------
        # Volatility Heatmap
        # -----------------------------------------------------

        st.subheader("🔥 Monthly Volatility Heatmap")

        vol = data["Volatility"].dropna()

        vol_df = pd.DataFrame({
            "Date": vol.index,
            "Volatility": vol.values
        })

        vol_df["Year"] = vol_df["Date"].dt.year
        vol_df["Month"] = vol_df["Date"].dt.month

        heatmap_data = vol_df.pivot_table(
            values="Volatility",
            index="Year",
            columns="Month",
            aggfunc="mean"
        )

        fig2, ax = plt.subplots(figsize=(10,5))

        sns.heatmap(
            heatmap_data,
            cmap="coolwarm",
            annot=True,
            fmt=".3f"
        )

        ax.set_title("Average Monthly Volatility")

        st.pyplot(fig2)

        # -----------------------------------------------------
        # GARCH Forecast
        # -----------------------------------------------------

        st.subheader("⚡ Next-Day Volatility Forecast")

        returns = train["Returns"] * 100

        garch = arch_model(returns, vol="GARCH", p=1, q=1)

        garch_fit = garch.fit(disp="off")

        forecast = garch_fit.forecast(horizon=1)

        vol_pred = np.sqrt(forecast.variance.values[-1,0])

        st.metric(
            label="Predicted Next-Day Volatility",
            value=round(vol_pred,4)
        )
