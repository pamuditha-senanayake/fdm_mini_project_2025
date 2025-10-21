from fastapi import APIRouter, Body
from pydantic import BaseModel
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

    daily_sales_smooth = daily_sales.rolling(window=3, min_periods=1).mean()
    train = daily_sales_smooth[:-steps]
    test = daily_sales_smooth[-steps:]

    train_log = np.log1p(train)
    model = SARIMAX(train_log, order=(1,1,1), seasonal_order=(1,1,0,7),
                    enforce_stationarity=False, enforce_invertibility=False)
    fit_model = model.fit(disp=False)

    pred_log = fit_model.get_forecast(steps=steps).predicted_mean
    pred_values = np.expm1(pred_log)
    pred_values = np.clip(pred_values, 0, None)

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

    # Prepare data for frontend graph
    forecast_dict = {
        "dates": [str(date.date()) for date in test.index],
        "actual": test.tolist(),
        "forecast": pred_values.tolist()
    }

    return {
        "forecast_data": forecast_dict,
        "trend": trend_text,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "accuracy_pct": round(accuracy_pct, 2)
    }
