from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

def arima_forecast(train_data, test_steps, order=(50, 1, 0)):
    train_series = train_data.flatten()
    model = ARIMA(train_series, order=order)
    # model = SARIMAX(train_series, order=order, seasonal_order=(1, 1, 1, 48))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=test_steps)
    return forecast.reshape(-1, 1)