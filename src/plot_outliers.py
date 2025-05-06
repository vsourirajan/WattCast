import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from common import load_and_split_data
from utils.metrics import calculate_metrics
from baselines.naive import naive_forecast
from baselines.average import mean_forecast
from baselines.arima import arima_forecast
from baselines.persistence import persistence_forecast
from baselines.lgbm import boosting_forecast
from baselines.prophet import prophet_forecast

import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
SEQUENCE_LENGTH = 48
TEST_RATIO = 0.2
DATA_DIR = '../data'
PLOT_DIR = '../etc/plots/outliers'  # Directory to save outlier plots
# --- End Configuration ---

def parse_worst_performers(file_path):
    """Parse the worst_performers.txt file to extract feeder IDs."""
    feeders_to_plot = set()
    current_model = None
    current_metric = None
    metric_count = 0  # Counter for each metric
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check for model line
            if line.startswith('Model:'):
                current_model = line.split('Model:')[1].strip()
                metric_count = 0  # Reset counter for new model
                continue
                
            # Check for metric line
            if 'worst' in line and 'feeders' in line:
                current_metric = line.split('(')[0].strip()
                # Only process R² and MAE metrics
                if current_metric not in ['R2', 'MAE']:
                    current_metric = None
                metric_count = 0  # Reset counter for new metric
                continue
                
            # Parse feeder line
            if line.startswith('Feeder:') and current_metric in ['R2', 'MAE']:
                # Extract feeder ID and value
                parts = line.split(',')
                if len(parts) == 2:
                    feeder_match = re.search(r'Feeder: (SSEN-\d+)', parts[0])
                    if feeder_match and metric_count < 3:  # Only take top 3
                        feeder_id = feeder_match.group(1)
                        value = float(parts[1].split(':')[1].strip())
                        feeders_to_plot.add((feeder_id, current_model, current_metric, value))
                        metric_count += 1
    
    return feeders_to_plot

def plot_feeder_baselines(feeder_id, data, model_name, metric_name, metric_value):
    """Generates and saves a comparison plot for a single feeder."""
    print(f"  Generating plot for Feeder {feeder_id} (Worst in {model_name} for {metric_name}, Value: {metric_value:.2f})...")

    if 'train' not in data or 'test' not in data or 'scaler' not in data or \
       'test_timestamps' not in data or len(data['test']) == 0:
        print(f"    Skipping Feeder {feeder_id} plot due to missing data components.")
        return False

    train = data['train']
    test = data['test']
    scaler = data['scaler']
    
    try:
        test_timestamps = pd.to_datetime(data['test_timestamps'])
    except Exception as e:
        print(f"    Error converting test timestamps for Feeder {feeder_id}: {e}")
        return False

    test_actuals_inv = scaler.inverse_transform(test)
    predictions = {}

    # Run all baselines
    try:
        naive_preds = naive_forecast(train, test)
        predictions['Naive'] = scaler.inverse_transform(naive_preds.reshape(-1, 1))
    except Exception as e:
        print(f"    Error running Naive for Feeder {feeder_id}: {e}")
        predictions['Naive'] = None

    try:
        mean_preds = mean_forecast(train, test)
        predictions['Mean (1 day)'] = scaler.inverse_transform(mean_preds.reshape(-1, 1))
    except Exception as e:
        print(f"    Error running Mean for Feeder {feeder_id}: {e}")
        predictions['Mean (1 day)'] = None

    try:
        arima_preds = arima_forecast(train, len(test))
        predictions['SARIMA'] = scaler.inverse_transform(arima_preds.reshape(-1, 1))
    except Exception as e:
        print(f"    Error running SARIMA for Feeder {feeder_id}: {e}")
        predictions['SARIMA'] = None

    try:
        boosting_preds = boosting_forecast(train, len(test))
        predictions['LGBM'] = scaler.inverse_transform(boosting_preds.reshape(-1, 1))
    except Exception as e:
        print(f"    Error running LGBM for Feeder {feeder_id}: {e}")
        predictions['LGBM'] = None

    if 'train_timestamps' in data:
        try:
            prophet_preds = prophet_forecast(train, data['train_timestamps'], len(test))
            predictions['Prophet'] = scaler.inverse_transform(prophet_preds.reshape(-1, 1))
        except Exception as e:
            print(f"    Error running Prophet for Feeder {feeder_id}: {e}")
            predictions['Prophet'] = None

    try:
        persistence_preds = persistence_forecast(train, test)
        predictions['Persistence'] = scaler.inverse_transform(persistence_preds.reshape(-1, 1))
    except Exception as e:
        print(f"    Error running Persistence for Feeder {feeder_id}: {e}")
        predictions['Persistence'] = None

    # Calculate metrics
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

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(test_timestamps, test_actuals_inv, label='Actual', linewidth=0.8, color='black')

    for model_name, preds_inv in predictions.items():
        if preds_inv is not None and len(preds_inv) == len(test_timestamps):
            plt.plot(test_timestamps, preds_inv, label=model_name, linewidth=0.8)

    plt.title(f'Energy Consumption Forecasting - Feeder {feeder_id}\n(Worst in {model_name} for {metric_name}, Value: {metric_value:.2f})')
    plt.xlabel('Date and Time')
    plt.ylabel('Consumption')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    # Save plot
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
        print(f"Created directory: {PLOT_DIR}")

    plot_filename = os.path.join(PLOT_DIR, f"outlier_{feeder_id}_{metric_name}.png")
    try:
        plt.savefig(plot_filename, bbox_inches='tight')
        print(f"    Plot saved to {plot_filename}")
    except Exception as e:
        print(f"    Error saving plot {plot_filename}: {e}")

    plt.close()
    return True

def main():
    print("Starting outlier plotting script...")
    print("Processing top 3 worst performing feeders for R² and MAE metrics...")
    
    # Read worst performers
    worst_performers_file = "../etc/worst_performers.txt"
    if not os.path.exists(worst_performers_file):
        print(f"Error: Could not find {worst_performers_file}")
        return

    feeders_to_plot = parse_worst_performers(worst_performers_file)
    print(f"Found {len(feeders_to_plot)} feeders to plot")

    # Process each feeder
    for feeder_id, model_name, metric_name, metric_value in feeders_to_plot:
        # Get the substation file path by removing the last 2 digits of the feeder ID
        substation_id = feeder_id[:-2]
        substation_file = os.path.join(DATA_DIR, f"2024-12-{substation_id}.csv")
        print(f"Processing substation file: {substation_file}")
        
        if not os.path.exists(substation_file):
            print(f"Warning: Could not find substation file for {substation_id}")
            continue

        try:
            # Load and split data for the substation
            substation_data = load_and_split_data(substation_file, SEQUENCE_LENGTH, TEST_RATIO)
            
            # Check if our specific feeder exists in the substation data
            if feeder_id in substation_data:
                plot_feeder_baselines(feeder_id, substation_data[feeder_id], model_name, metric_name, metric_value)
            else:
                print(f"Warning: Feeder {feeder_id} not found in substation {substation_id} data")
        except Exception as e:
            print(f"Error processing substation {substation_id} for feeder {feeder_id}: {e}")

    print("Plotting complete.")

if __name__ == "__main__":
    main() 