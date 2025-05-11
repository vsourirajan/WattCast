import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from common import load_and_split_data # Assuming this function also returns test timestamps
from utils.metrics import calculate_metrics # Optional: can print metrics for plotted feeders
from baselines.naive import naive_forecast
from baselines.average import mean_forecast
from baselines.arima import arima_forecast
from baselines.persistence import persistence_forecast
from baselines.lgbm import boosting_forecast
from baselines.prophet import prophet_forecast
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import torch

import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
SEQUENCE_LENGTH = 48
TEST_RATIO = 0.2
DATA_DIR = '../data'
PLOT_DIR = '../etc/plots/full_timeline' # Directory to save plots
NUM_FEEDERS_TO_PLOT = 1
# --- End Configuration ---



def plot_feeder_baselines(feeder_id, data):
    """Generates and saves a comparison plot for a single feeder."""
    print(f"  Generating plot for Feeder {feeder_id}...")

    if 'train' not in data or 'test' not in data or 'scaler' not in data or \
       'test_timestamps' not in data or len(data['test']) == 0:
        print(f"    Skipping Feeder {feeder_id} plot due to missing data components.")
        return False # Indicate failure

    train = data['train']
    test = data['test']
    scaler = data['scaler']
    # Ensure timestamps are in datetime format for plotting
    try:
        test_timestamps = pd.to_datetime(data['test_timestamps'])
    except Exception as e:
        print(f"    Error converting test timestamps for Feeder {feeder_id}: {e}")
        return False # Indicate failure

    test_actuals_inv = scaler.inverse_transform(test)

    predictions = {} # Store predictions for plotting

    # --- Run Baselines ---
    # Naive
    try:
        naive_preds = naive_forecast(train, test)
        predictions['Naive'] = scaler.inverse_transform(naive_preds.reshape(-1, 1))
    except Exception as e:
        print(f"    Error running Naive for Feeder {feeder_id}: {e}")
        predictions['Naive'] = None

    # Mean
    try:
        mean_preds = mean_forecast(train, test)
        predictions['Mean (1 day)'] = scaler.inverse_transform(mean_preds.reshape(-1, 1))
    except Exception as e:
         print(f"    Error running Mean for Feeder {feeder_id}: {e}")
         predictions['Mean (1 day)'] = None

    # ARIMA
    try:
        arima_preds = arima_forecast(train, len(test))
        predictions['SARIMA'] = scaler.inverse_transform(arima_preds.reshape(-1, 1))
    except Exception as e:
         print(f"    Error running SARIMA for Feeder {feeder_id}: {e}")
         predictions['SARIMA'] = None

    # Boosting
    try:
        boosting_preds = boosting_forecast(train, len(test))
        predictions['LGBM'] = scaler.inverse_transform(boosting_preds.reshape(-1, 1))
    except Exception as e:
        print(f"    Error running LGBM for Feeder {feeder_id}: {e}")
        predictions['LGBM'] = None

    # Prophet
    if 'train_timestamps' in data:
         try:
            prophet_preds = prophet_forecast(train, data['train_timestamps'], len(test))
            predictions['Prophet'] = scaler.inverse_transform(prophet_preds.reshape(-1, 1))
         except Exception as e:
            print(f"    Error running Prophet for Feeder {feeder_id}: {e}")
            predictions['Prophet'] = None
    else:
        print(f"    Skipping Prophet for Feeder {feeder_id} due to missing 'train_timestamps'.")
        predictions['Prophet'] = None

    # Persistence
    try:
        persistence_preds = persistence_forecast(train, test)
        predictions['Persistence'] = scaler.inverse_transform(persistence_preds.reshape(-1, 1))
    except Exception as e:
        print(f"    Error running Persistence for Feeder {feeder_id}: {e}")
        predictions['Persistence'] = None
    # --- End Baselines ---

    # Calculate and display metrics for each model
    print(f"\n--- Metrics for Feeder {feeder_id} ---")
    metrics_table = {}
    
    for model_name, preds_inv in predictions.items():
        if preds_inv is not None and len(preds_inv) == len(test_actuals_inv):
            model_metrics = calculate_metrics(test_actuals_inv, preds_inv)
            metrics_table[model_name] = model_metrics
            print(f"  {model_name}:")
            print(f"    RMSE: {model_metrics['RMSE']:.2f}")
            print(f"    MAE: {model_metrics['MAE']:.2f}")
            print(f"    MSE: {model_metrics['MSE']:.2f}")
            print(f"    MAPE: {model_metrics['MAPE']:.2f}%")

    # --- Plotting ---
    plt.figure(figsize=(12, 6)) # Match the size in train_timeseries.py
    plt.plot(test_timestamps, test_actuals_inv, label='Actual', linewidth=0.8, color='black')

    for model_name, preds_inv in predictions.items():
        if preds_inv is not None:
            print(model_name)
            # Ensure prediction length matches timestamp length
            if len(preds_inv) == len(test_timestamps):
                 plt.plot(test_timestamps, preds_inv, label=model_name, linewidth=0.8)
            else:
                 print(f"    Warning: Prediction length mismatch for {model_name} on Feeder {feeder_id}. Skipping plot line.")


    plt.title(f'Energy Consumption Forecasting - Feeder {feeder_id}')
    plt.xlabel('Date and Time')
    plt.ylabel('Consumption')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Format x-axis dates - match the style in train_timeseries.py
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.gcf().autofmt_xdate() # Rotate date labels
    plt.tight_layout()

    # Save plot
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
        print(f"Created directory: {PLOT_DIR}")

    plot_filename = os.path.join(PLOT_DIR, f"baseline_comparison_{feeder_id}.png")
    try:
        plt.savefig(plot_filename, bbox_inches='tight')
        print(f"    Plot saved to {plot_filename}")
    except Exception as e:
        print(f"    Error saving plot {plot_filename}: {e}")

    plt.close() # Close the figure to free memory
    return True # Indicate success
    # --- End Plotting ---


def main():
    print("Starting baseline plotting script...")
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    feeders_plotted_count = 0

    for file_path in files:
        if feeders_plotted_count >= NUM_FEEDERS_TO_PLOT:
            break # Stop if we have plotted enough feeders

        print(f"Processing {file_path}")
        try:
            # *** IMPORTANT: Ensure load_and_split_data returns 'test_timestamps' ***
            # You might need to modify 'common.py' if it doesn't.
            # Assuming it returns dict like: {'feeder_id': {'train':..., 'test':..., 'scaler':..., 'test_timestamps':..., 'train_timestamps':...}}
            feeder_data = load_and_split_data(file_path, SEQUENCE_LENGTH, TEST_RATIO)
        except Exception as e:
            print(f"  Error loading or splitting data for {file_path}: {e}")
            continue

        for feeder_id, data in feeder_data.items():
            if feeders_plotted_count >= NUM_FEEDERS_TO_PLOT:
                break # Stop inner loop too

            success = plot_feeder_baselines(feeder_id, data)
            if success:
                feeders_plotted_count += 1

    print(f"Plotting complete. Generated plots for {feeders_plotted_count} feeders.")
    if feeders_plotted_count < NUM_FEEDERS_TO_PLOT:
        print(f"  Note: Fewer than {NUM_FEEDERS_TO_PLOT} feeders were plotted, possibly due to data issues or fewer available feeders.")


if __name__ == "__main__":
    main() 