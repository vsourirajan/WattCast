import pandas as pd
from prophet import Prophet
import numpy as np

def prophet_forecast(train_data, train_timestamps, test_steps):
    df_train = pd.DataFrame({
        'ds': pd.to_datetime(train_timestamps),
        'y': train_data.flatten()
    })
    model = Prophet(daily_seasonality=True)
    model.fit(df_train)

    future = model.make_future_dataframe(periods=test_steps, freq='30min')
    forecast = model.predict(future)
    y_pred = forecast['yhat'].values[-test_steps:]
    return y_pred.reshape(-1, 1)
