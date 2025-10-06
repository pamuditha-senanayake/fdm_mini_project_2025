# backend/components/stockpredict.py
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fastapi import APIRouter
from pydantic import BaseModel

# Load CSV once
df = pd.read_csv("data.csv")
df['Date'] = pd.to_datetime(df['Date'])

router = APIRouter()


class ForecastRequest(BaseModel):
    category: str
    steps: int = 30


@router.post("/forecast")
def forecast_sales(request: ForecastRequest):
    category = request.category
    steps = int(request.steps)

    df_cat = df[df['Product_Category'] == category]
    daily_sales = df_cat.groupby('Date')['Total_Amount'].sum()
    daily_sales.index = pd.DatetimeIndex(daily_sales.index, freq='D')

    train = daily_sales[:-steps]
    test = daily_sales[-steps:]

    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    fit_model = model.fit(disp=False)
    pred = fit_model.get_forecast(steps=steps)
    pred_values = pred.predicted_mean

    mae = mean_absolute_error(test, pred_values)
    rmse = np.sqrt(mean_squared_error(test, pred_values))
    accuracy_pct = (1 - mae / np.mean(test)) * 100

    trend = np.mean(pred_values) - np.mean(train[-7:])
    if trend > 0:
        trend_text = "upward trend. Stock may need to be increased."
    elif trend < 0:
        trend_text = "downward trend. Consider promotions to avoid overstock."
    else:
        trend_text = "stable. Maintain current stock levels."

    forecast_list = [f"{date.date()}: {round(value, 2)}" for date, value in zip(test.index, pred_values)]
    output_text = {
        "forecast": forecast_list,
        "trend": trend_text,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "accuracy_pct": round(accuracy_pct, 2)
    }

    return output_text
