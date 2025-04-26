import numpy as np

def naive_forecast(train_data, test_data):
    last_value = train_data[-1]
    predictions = np.repeat(last_value, len(test_data), axis=0)
    return predictions