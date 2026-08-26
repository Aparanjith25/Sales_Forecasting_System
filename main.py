import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
app = FastAPI(title="Sales Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import sys

class MyMinMaxScaler:
    def __init__(self, feature_range=(0,1)):
        self.min_ = None
        self.max_ = None
        self.range_min, self.range_max = feature_range

    def fit(self, data):
        self.min_ = data.min(axis=0)
        self.max_ = data.max(axis=0)
        return self

    def transform(self, data):
        scaled = (data - self.min_) / (self.max_ - self.min_)
        scaled = scaled * (self.range_max - self.range_min) + self.range_min
        return scaled

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        data = (data - self.range_min) / (self.range_max - self.range_min)
        return data * (self.max_ - self.min_) + self.min_

# register the class under __main__ so pickle can find it
sys.modules['__main__'].MyMinMaxScaler = MyMinMaxScaler


# load everything once at startup
model = load_model('sales_lstm_model.h5')
scaler = pickle.load(open('sales_scaler.pkl', 'rb'))

# rebuild the same store 1 sales series used in training
df = pd.read_csv('train.csv', low_memory=False, usecols=['Store', 'Date', 'Sales', 'Open'])
store1 = df[df['Store'] == 1].copy()
store1['Date'] = pd.to_datetime(store1['Date'])
store1 = store1.sort_values('Date')
store1 = store1[store1['Open'] == 1]
store1 = store1.set_index('Date')
sales = store1['Sales']

time_step = 30


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/history")
def history(days: int = Query(90, ge=1, le=len(sales))):
    recent = sales[-days:]
    return {
        "dates": recent.index.strftime('%Y-%m-%d').tolist(),
        "sales": recent.values.tolist()
    }


@app.get("/forecast")
def forecast(days: int = Query(14, ge=1, le=60)):
    # take the last 30 known days as the seed
    last_window = sales.values[-time_step:].reshape(-1, 1)
    scaled_window = scaler.transform(last_window)

    temp_input = scaled_window.flatten().tolist()
    predictions_scaled = []

    for _ in range(days):
        x_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
        pred = model.predict(x_input, verbose=0)
        predictions_scaled.append(pred[0][0])
        temp_input.append(pred[0][0])

    predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
    predictions_actual = scaler.inverse_transform(predictions_scaled).flatten()

    last_date = sales.index[-1]
    future_dates = pd.date_range(start=last_date, periods=days + 1, freq='D')[1:]

    return {
        "dates": future_dates.strftime('%Y-%m-%d').tolist(),
        "forecast": predictions_actual.tolist()
    }