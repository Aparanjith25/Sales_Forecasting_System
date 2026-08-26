# Store Sales Forecasting System

A time-series forecasting pipeline that benchmarks a **Stacked LSTM** against classical baselines (naive, moving average, Prophet) on daily retail sales, served through a FastAPI backend and a Streamlit frontend, deployed as two independent cloud services.

**Live demo:** https://salesforecastingsystem-5txzqrdappjnerwqzbegck.streamlit.app

---

## What it does

- Visualizes historical daily sales for a store (Rossmann Store Sales dataset)
- Forecasts the next N days of sales using a trained LSTM model
- Chart shows actual history transitioning into the forecast, plus a table of predicted values

## Model results

Five forecasting approaches were built and benchmarked on the same held-out test period, using **custom-implemented** RMSE/MAE functions :

| Model                   | RMSE     | MAE      |
|--------------------------|----------|----------|
| **LSTM**                 | **601.0** | **483.8** |
| Naive (t-1, "yesterday")  | 806.1    | 570.2    |
| Prophet                  | 851.3    | 688.6    |
| Moving Average (7-day)    | 1027.1   | 816.4    |
| Naive (t-7, "last week")  | 1472.1   | 1209.9   |

## Tech stack

Python · TensorFlow/Keras (Stacked LSTM) · Prophet · pandas · NumPy · FastAPI · Streamlit · Render · Streamlit Cloud
