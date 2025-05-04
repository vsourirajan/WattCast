import os
import numpy as np
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import load_and_split_data
from utils.metrics import calculate_metrics
from baselines.naive import naive_forecast
from baselines.average import mean_forecast
from baselines.arima import arima_forecast
from baselines.persistence import persistence_forecast
from baselines.lgbm import boosting_forecast
from baselines.prophet import prophet_forecast

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(filename='baseline_results.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def evaluate_models(feeder_id, data):
    results = {}

    train = data['train']
    test = data['test']
    scaler = data['scaler']
    test_actuals = scaler.inverse_transform(test)

    def run_forecast(name, func, *args):
        try:
            preds = func(*args)
            preds = scaler.inverse_transform(preds.reshape(-1, 1))
            metrics = calculate_metrics(test_actuals, preds)
            return name, metrics
        except Exception as e:
            logging.warning(f"Feeder {feeder_id} - {name} forecast error: {e}")
            return name, None

    models = [
        ("Naive", naive_forecast, train, test),
        ("Mean (1 day)", mean_forecast, train, test),
        ("Persistence", persistence_forecast, train, test),
        ("ARIMA", arima_forecast, train, len(test)),
        ("Boosting", boosting_forecast, train, len(test))
    ]

    if 'train_timestamps' in data:
        models.append(("Prophet", prophet_forecast, train, data['train_timestamps'], len(test)))
    else:
        logging.info(f"Feeder {feeder_id} - Skipping Prophet due to missing timestamps")

    for name, func, *args in models:
        model_name, metrics = run_forecast(name, func, *args)
        if metrics:
            results[model_name] = metrics
            logging.info(f"Feeder {feeder_id} - {model_name} - {metrics}")

    return results

def main():
    sequence_length = 48
    test_ratio = 0.2
    data_dir = '../data'

    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    all_metrics = {}

    for file_path in files:
        logging.info(f"Processing file: {file_path}")

        try:
            feeder_data = load_and_split_data(file_path, sequence_length, test_ratio)
        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {e}")
            continue

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(evaluate_models, fid, fdata): fid for fid, fdata in feeder_data.items()}

            for future in as_completed(futures):
                fid = futures[future]
                try:
                    model_results = future.result()
                    for model, metrics in model_results.items():
                        if model not in all_metrics:
                            all_metrics[model] = {k: [] for k in metrics.keys()}
                        for metric_name, value in metrics.items():
                            all_metrics[model][metric_name].append(value)
                except Exception as e:
                    logging.error(f"Feeder {fid} evaluation failed: {e}")

    # Print summary
    print("\n--- Average Baseline Metrics Across All Processed Feeders ---")
    for model, metrics in all_metrics.items():
        print(f"\n{model}:")
        for metric_name, values in metrics.items():
            arr = np.array(values)
            avg = np.nanmean(arr)
            std = np.nanstd(arr)
            if metric_name == 'MAPE':
                print(f"  Average {metric_name}: {avg:.2f}% (std: {std:.2f}%)")
            elif metric_name == 'R2':
                print(f"  Average {metric_name}: {avg:.4f} (std: {std:.4f})")
            else:
                print(f"  Average {metric_name}: {avg:.2f} (std: {std:.2f})")

if __name__ == "__main__":
    main()
