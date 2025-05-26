import streamlit as st
import requests
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from datetime import datetime

# Page title
st.title("📈 Forecast Apple Stocks (AAPL)")

# Function to fetch data from Alpha Vantage
def get_data(symbol: str, api_key: str) -> pd.Series:
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "compact",
            "apikey": api_key
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json().get("Time Series (Daily)", {})
        if not data:
            raise ValueError("API response did not contain stock data.")

        df = pd.DataFrame(data).T.astype(float)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        return df["4. close"]

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.Series(dtype=float)

# API Key and symbol
symbol = "AAPL"
api_key = st.secrets["alphavantage"]["api_key"]

# Fetch stock data
series = get_data(symbol, api_key)

# Ensure data is not empty before proceeding
if not series.empty:
    series_diff = series.diff().dropna()

    if st.button("🔮 Predict Next 7 Days Stock Price"):
        with st.spinner("Running ARIMA model and forecasting..."):
            try:
                model = ARIMA(series_diff, order=(3, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=7)

                st.subheader("📊 7-Day Forecast (Differenced)")
                st.line_chart(forecast)

            except Exception as e:
                st.error(f"Error during model fitting or forecasting: {e}")
else:
    st.warning("No data available to display. Please check your API key or network.")
