import pandas as pd
from prophet import Prophet
import numpy as np

def prophet_forecast(train_data, train_timestamps, test_steps):
    df_train = pd.DataFrame({
        'ds': pd.to_datetime(train_timestamps),
        'y': train_data.flatten()
    })
    model = Prophet(daily_seasonality=True)
    #model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    # model.add_seasonality(name='weekly', period=7, fourier_order=3)
    # model.add_seasonality(name='daily', period=1, fourier_order=3)
    model.add_seasonality(name='half-hourly', period=1, fourier_order=10)
    model.fit(df_train)

    future = model.make_future_dataframe(periods=test_steps, freq='30min')
    forecast = model.predict(future)
    y_pred = forecast['yhat'].values[-test_steps:]
    return y_pred.reshape(-1, 1)
