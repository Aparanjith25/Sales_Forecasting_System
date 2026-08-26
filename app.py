import os
import requests
import streamlit as st
import pandas as pd

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Sales Forecast", page_icon="📈", layout="wide")

st.title("📈 Store Sales Forecaster")
st.markdown("Store 1 daily sales — history and LSTM forecast")

days_history = st.sidebar.slider("Days of history to show", 30, 365, 90)
days_forecast = st.sidebar.slider("Days to forecast", 7, 60, 14)


def api_get(path, params=None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}: {r.text[:200]}"
        return r.json(), None
    except Exception as e:
        return None, f"Request failed: {e}"


history, err1 = api_get("/history", params={"days": days_history})
forecast, err2 = api_get("/forecast", params={"days": days_forecast})

if err1 or err2:
    st.error(f"Something went wrong: {err1 or err2}")
    st.stop()

hist_df = pd.DataFrame({
    "date": pd.to_datetime(history["dates"]),
    "sales": history["sales"],
    "type": "Actual"
})

forecast_df = pd.DataFrame({
    "date": pd.to_datetime(forecast["dates"]),
    "sales": forecast["forecast"],
    "type": "Forecast"
})

combined = pd.concat([hist_df, forecast_df])
combined = combined.set_index("date")

st.line_chart(combined.pivot_table(index=combined.index, columns="type", values="sales"))

st.markdown("### Forecast values")
st.dataframe(forecast_df[["date", "sales"]].rename(columns={"sales": "predicted_sales"}))