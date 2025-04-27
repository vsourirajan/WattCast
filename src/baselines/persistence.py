import numpy as np

def persistence_forecast(train_data, test_data):
    """
    Persistence forecast that uses the value from the same time yesterday (48 steps back).
    For each point in the test period, use the value from the same time of day in the previous day.
    """
    day_steps = 48  # Number of steps in a day (48 half-hour intervals)
    predictions = np.zeros_like(test_data)
    
    # For each position in test data
    for i in range(len(test_data)):
        # If we have enough training data, use training data value from previous day
        if i < len(predictions) and (len(train_data) >= day_steps):
            # Use value from same time yesterday in training data
            predictions[i] = train_data[-day_steps + (i % day_steps)]
        else:
            # If we've gone past available training data, use values from the test data
            # (This handles multi-day forecasts by cycling through the available data)
            idx = i - day_steps
            if idx >= 0:
                predictions[i] = test_data[idx]
            else:
                # Fallback if we don't have enough data
                predictions[i] = train_data[-1]
    
    return predictions
    