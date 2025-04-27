import lightgbm as lgb
import numpy as np

def boosting_forecast(train_data, test_steps, lags=48):
    X, y = [], []
    for i in range(lags, len(train_data)):
        X.append(train_data[i-lags:i].flatten())
        y.append(train_data[i].flatten())
    
    X = np.array(X)
    y = np.array(y)
    
    model = lgb.LGBMRegressor()
    model.fit(X, y)
    
    last_window = train_data[-lags:].flatten()
    preds = []
    for _ in range(test_steps):
        pred = model.predict(last_window.reshape(1, -1))
        preds.append(pred)
        last_window = np.roll(last_window, -1)
        last_window[-1] = pred
    
    return np.array(preds).reshape(-1, 1)