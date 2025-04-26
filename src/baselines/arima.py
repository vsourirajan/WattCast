from statsmodels.tsa.arima.model import ARIMA
import numpy as np

def arima_forecast(train_data, test_steps, order=(1, 1, 0)):
    train_series = train_data.flatten()
    model = ARIMA(train_series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=test_steps)
    return forecast.reshape(-1, 1)