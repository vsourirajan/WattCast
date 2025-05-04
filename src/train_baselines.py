import os
import numpy as np
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

def main():
    sequence_length = 48
    test_ratio = 0.2
    data_dir = '../data'  # where all your CSVs live

    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # Initialize dictionary of dictionaries for metrics
    all_metrics = {
        'Naive': {'RMSE': [], 'MAE': [], 'MSE': [], 'MAPE': [], 'R2': []},
        'Mean (1 day)': {'RMSE': [], 'MAE': [], 'MSE': [], 'MAPE': [], 'R2': []},
        'Persistence': {'RMSE': [], 'MAE': [], 'MSE': [], 'MAPE': [], 'R2': []},
        'ARIMA': {'RMSE': [], 'MAE': [], 'MSE': [], 'MAPE': [], 'R2': []},
        'Boosting': {'RMSE': [], 'MAE': [], 'MSE': [], 'MAPE': [], 'R2': []},
        'Prophet': {'RMSE': [], 'MAE': [], 'MSE': [], 'MAPE': [], 'R2': []}
    }

    for file_path in files:
        print(f"\nProcessing {file_path}")
        # Handle potential errors during file loading/processing
        try:
            feeder_data = load_and_split_data(file_path, sequence_length, test_ratio)
        except Exception as e:
            print(f"  Error loading or splitting data for {file_path}: {e}")
            continue # Skip to the next file

        for feeder_id, data in feeder_data.items():
            print(f"  Processing Feeder {feeder_id}...") # Indicate processing

            # Check if data is valid before proceeding
            if 'train' not in data or 'test' not in data or 'scaler' not in data or \
               len(data['train']) == 0 or len(data['test']) == 0:
                print(f"    Skipping Feeder {feeder_id} due to insufficient data.")
                continue

            train = data['train']
            test = data['test']
            scaler = data['scaler']
            test_actuals = scaler.inverse_transform(test) # Inverse transform once

            # Naive
            try:
                naive_preds = naive_forecast(train, test)
                naive_preds = scaler.inverse_transform(naive_preds.reshape(-1, 1))
                naive_metrics = calculate_metrics(test_actuals, naive_preds)
                # Append each metric to its respective list
                all_metrics['Naive']['RMSE'].append(naive_metrics['RMSE'])
                all_metrics['Naive']['MAE'].append(naive_metrics['MAE'])
                all_metrics['Naive']['MSE'].append(naive_metrics['MSE'])
                all_metrics['Naive']['MAPE'].append(naive_metrics['MAPE'])
                all_metrics['Naive']['R2'].append(naive_metrics['R2'])
            except Exception as e:
                print(f"    Error running Naive forecast for Feeder {feeder_id}: {e}")

            # Mean
            try:
                mean_preds = mean_forecast(train, test)
                mean_preds = scaler.inverse_transform(mean_preds.reshape(-1, 1))
                mean_metrics = calculate_metrics(test_actuals, mean_preds)
                # Append each metric to its respective list
                all_metrics['Mean (1 day)']['RMSE'].append(mean_metrics['RMSE'])
                all_metrics['Mean (1 day)']['MAE'].append(mean_metrics['MAE'])
                all_metrics['Mean (1 day)']['MSE'].append(mean_metrics['MSE'])
                all_metrics['Mean (1 day)']['MAPE'].append(mean_metrics['MAPE'])
                all_metrics['Mean (1 day)']['R2'].append(mean_metrics['R2'])
            except Exception as e:
                 print(f"    Error running Mean forecast for Feeder {feeder_id}: {e}")
                 
            # Persistence
            try:
                persistence_preds = persistence_forecast(train, test)
                persistence_preds = scaler.inverse_transform(persistence_preds.reshape(-1, 1))
                persistence_metrics = calculate_metrics(test_actuals, persistence_preds)
                # Append each metric to its respective list
                all_metrics['Persistence']['RMSE'].append(persistence_metrics['RMSE'])
                all_metrics['Persistence']['MAE'].append(persistence_metrics['MAE'])
                all_metrics['Persistence']['MSE'].append(persistence_metrics['MSE'])
                all_metrics['Persistence']['MAPE'].append(persistence_metrics['MAPE'])
                all_metrics['Persistence']['R2'].append(persistence_metrics['R2'])
            except Exception as e:
                 print(f"    Error running Persistence forecast for Feeder {feeder_id}: {e}")

            
            # ARIMA
            try:
                arima_preds = arima_forecast(train, len(test))
                arima_preds = scaler.inverse_transform(arima_preds.reshape(-1, 1))
                arima_metrics = calculate_metrics(test_actuals, arima_preds)
                # Append each metric to its respective list
                all_metrics['ARIMA']['RMSE'].append(arima_metrics['RMSE'])
                all_metrics['ARIMA']['MAE'].append(arima_metrics['MAE'])
                all_metrics['ARIMA']['MSE'].append(arima_metrics['MSE'])
                all_metrics['ARIMA']['MAPE'].append(arima_metrics['MAPE'])
                all_metrics['ARIMA']['R2'].append(arima_metrics['R2'])
            except Exception as e:
                 print(f"    Error running ARIMA forecast for Feeder {feeder_id}: {e}")

                 
            # Boosting (Keep commented out structure)
            try:
                boosting_preds = boosting_forecast(train, len(test))
                boosting_preds = scaler.inverse_transform(boosting_preds.reshape(-1, 1))
                boosting_metrics = calculate_metrics(test_actuals, boosting_preds)
                # Append each metric to its respective list
                all_metrics['Boosting']['RMSE'].append(boosting_metrics['RMSE'])
                all_metrics['Boosting']['MAE'].append(boosting_metrics['MAE'])
                all_metrics['Boosting']['MSE'].append(boosting_metrics['MSE'])
                all_metrics['Boosting']['MAPE'].append(boosting_metrics['MAPE'])
            except Exception as e:
                print(f"    Error running Boosting forecast for Feeder {feeder_id}: {e}")

            # Prophet
            # Ensure 'train_timestamps' exists before calling prophet
            if 'train_timestamps' in data:
                 try:
                    prophet_preds = prophet_forecast(train, data['train_timestamps'], len(test))
                    prophet_preds = scaler.inverse_transform(prophet_preds.reshape(-1, 1))
                    prophet_metrics = calculate_metrics(test_actuals, prophet_preds)
                    # Append each metric to its respective list
                    all_metrics['Prophet']['RMSE'].append(prophet_metrics['RMSE'])
                    all_metrics['Prophet']['MAE'].append(prophet_metrics['MAE'])
                    all_metrics['Prophet']['MSE'].append(prophet_metrics['MSE'])
                    all_metrics['Prophet']['MAPE'].append(prophet_metrics['MAPE'])
                 except Exception as e:
                    print(f"    Error running Prophet forecast for Feeder {feeder_id}: {e}")
            else:
                print(f"    Skipping Prophet for Feeder {feeder_id} due to missing 'train_timestamps'.")

    # Calculate and print average metrics
    print("\n--- Average Baseline Metrics Across All Processed Feeders ---")
    
    for model_name, metrics_dict in all_metrics.items():
        print(f"\n{model_name}:")
        # Count of feeders processed for this model (use the RMSE list length)
        num_feeders = len(metrics_dict['RMSE'])
        
        if num_feeders == 0:
            print(f"\n{model_name}: No data processed.")
            continue
        
        # Debug to check for NaN or infinite values
        rmse_values = np.array(metrics_dict['RMSE'])
        
        # Count NaN or infinite values
        nan_count = np.isnan(rmse_values).sum()
        inf_count = np.isinf(rmse_values).sum()
        print(f"NaN count: {nan_count}, Inf count: {inf_count}")
        
        avg_rmse = np.nanmean(rmse_values) if len(rmse_values) > 0 else float('nan')
        std_rmse = np.nanstd(rmse_values) if len(rmse_values) > 0 else float('nan')
        avg_mae = np.nanmean(metrics_dict['MAE']) if len(metrics_dict['MAE']) > 0 else float('nan')
        std_mae = np.nanstd(metrics_dict['MAE']) if len(metrics_dict['MAE']) > 0 else float('nan')
        avg_mse = np.nanmean(metrics_dict['MSE']) if len(metrics_dict['MSE']) > 0 else float('nan')
        std_mse = np.nanstd(metrics_dict['MSE']) if len(metrics_dict['MSE']) > 0 else float('nan')
        avg_mape = np.nanmean(metrics_dict['MAPE']) if len(metrics_dict['MAPE']) > 0 else float('nan')
        std_mape = np.nanstd(metrics_dict['MAPE']) if len(metrics_dict['MAPE']) > 0 else float('nan')
        avg_r2 = np.nanmean(metrics_dict['R2']) if len(metrics_dict['R2']) > 0 else float('nan')
        std_r2 = np.nanstd(metrics_dict['R2']) if len(metrics_dict['R2']) > 0 else float('nan')

        print(f"\n{model_name} (Processed {num_feeders} feeders):")
        print(f"  Average RMSE: {avg_rmse:.2f} (std: {std_rmse:.2f})")
        print(f"  Average MAE: {avg_mae:.2f} (std: {std_mae:.2f})")
        print(f"  Average MSE: {avg_mse:.2f} (std: {std_mse:.2f})")
        print(f"  Average MAPE: {avg_mape:.2f}% (std: {std_mape:.2f}%)")
        print(f"  Average R²: {avg_r2:.4f} (std: {std_r2:.4f})")

if __name__ == "__main__":
    main()
