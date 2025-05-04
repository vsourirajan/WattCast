from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

def arima_forecast(train_data, test_steps, order=(24, 0, 0)):
    train_series = train_data.flatten()
    # model = ARIMA(train_series, order=order)
    # model = SARIMAX(train_series, order=order, seasonal_order=(1, 1, 1, 48))
    model = SARIMAX(train_data, order=(1,1,1), seasonal_order=(1,0,1,48))  # 48 = seasonal period
    model_fit = model.fit(disp=False, method="powell")
    # model_fit = model.fit(method='statespace')
    forecast = model_fit.forecast(steps=test_steps)
    return forecast.reshape(-1, 1)