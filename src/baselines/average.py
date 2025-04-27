import numpy as np

def mean_forecast(train_data, test_data, window_size=1*48):
    if len(train_data) < window_size:
        window_size = len(train_data)
    mean_value = np.mean(train_data[-window_size:], axis=0)
    predictions = np.repeat(mean_value, len(test_data), axis=0)
    return predictions