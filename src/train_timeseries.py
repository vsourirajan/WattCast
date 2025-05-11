import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import logging
from models.LSTM import LSTM
from utils.metrics import calculate_metrics

# Set up logging
logging.basicConfig(filename='lstm_training_4.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class HistoryAwareTimeSeriesDataset(Dataset):
    def __init__(self, energy_values, timestamps, sequence_length, start_idx, end_idx):
        self.X = []
        self.y = []
        self.timestamps = []
        self.sequence_length = sequence_length

        for i in range(start_idx, end_idx - sequence_length):
            # Get sequence of energy values
            sequence = energy_values[i:i + sequence_length]
            self.X.append(sequence)
            # Get target value and timestamp
            self.y.append(energy_values[i + sequence_length])
            self.timestamps.append(timestamps[i + sequence_length])

        self.X = np.array(self.X)
        self.y = np.array(self.y).reshape(-1, 1)
        self.timestamps = np.array(self.timestamps)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])

def load_and_preprocess_data(file_path, sequence_length=48):
    df = pd.read_csv(file_path)
    df['data_collection_log_timestamp'] = pd.to_datetime(df['data_collection_log_timestamp'])

    feeders = df['lv_feeder_unique_id'].unique()
    feeder_data = {}

    for feeder in feeders:
        feeder_df = df[df['lv_feeder_unique_id'] == feeder]
        timestamps = feeder_df['data_collection_log_timestamp'].values
        consumption = feeder_df['total_consumption_active_import'].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        consumption_scaled = scaler.fit_transform(consumption).flatten()  # Flatten to 1D array

        total_length = len(consumption_scaled)
        train_size = int(total_length * 0.8)
        
        # Create training dataset
        train_dataset = HistoryAwareTimeSeriesDataset(
            consumption_scaled,
            timestamps,
            sequence_length,
            start_idx=0,
            end_idx=train_size
        )
        
        # Create validation dataset that includes necessary history
        val_dataset = HistoryAwareTimeSeriesDataset(
            consumption_scaled,
            timestamps,
            sequence_length,
            start_idx=train_size - sequence_length,
            end_idx=total_length
        )

        feeder_data[feeder] = {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'scaler': scaler,
            'train_timestamps': train_dataset.timestamps,
            'val_timestamps': val_dataset.timestamps,
            'energy_values': consumption_scaled,
            'timestamps': timestamps,
            'train_size': train_size
        }

    return feeder_data

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

def main():
    # Configuration
    sequence_length = 48
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    data_dir = './data'
    results_dir = '../results/lstm'
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Get all CSV files in the data directory
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    all_results = {}

    # Load processed files into a set
    processed_files = set()
    try:
        with open('processed_files.txt', 'r') as f:
            processed_files = set(line.strip() for line in f)
        logging.info(f"Loaded {len(processed_files)} previously processed files")
    except FileNotFoundError:
        logging.info("No processed files record found. Starting fresh.")
        processed_files = set()
    
    for file_name in data_files:
        # Skip if file has already been processed
        if file_name in processed_files:
            logging.info(f"Skipping already processed file: {file_name}")
            continue

        logging.info(f"\nProcessing file: {file_name}")
        file_path = os.path.join(data_dir, file_name)
        
        try:
            feeder_data = load_and_preprocess_data(file_path, sequence_length)
        except Exception as e:
            logging.error(f"Error loading data from {file_name}: {e}")
            continue

        file_results = {}
        
        for feeder_id, data in feeder_data.items():
            logging.info(f"\nProcessing feeder: {feeder_id}")

            try:
                train_loader = DataLoader(data['train_dataset'], batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(data['val_dataset'], batch_size=batch_size)

                model = LSTM().to(device)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                train_model(model, train_loader, criterion, optimizer, num_epochs, device)

                model.eval()
                predictions = []
                actuals = []

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(device)
                        outputs = model(batch_X)
                        predictions.extend(outputs.cpu().numpy())
                        actuals.extend(batch_y.numpy())

                predictions = data['scaler'].inverse_transform(np.array(predictions))
                actuals = data['scaler'].inverse_transform(np.array(actuals))
                timestamps = data['val_timestamps']

                # Calculate metrics
                metrics = calculate_metrics(actuals, predictions)
                
                # Log metrics for this feeder
                logging.info(f"\n--- Metrics for Feeder {feeder_id} ---")
                logging.info(f"  RMSE: {metrics['RMSE']:.2f}")
                logging.info(f"  MAE: {metrics['MAE']:.2f}")
                logging.info(f"  MSE: {metrics['MSE']:.2f}")
                logging.info(f"  MAPE: {metrics['MAPE']:.2f}%")
                logging.info(f"  R²: {metrics['R2']:.4f}")

                file_results[feeder_id] = {
                    'predictions': predictions,
                    'actuals': actuals,
                    'metrics': metrics,
                    'timestamps': timestamps
                }

            except Exception as e:
                logging.error(f"Error processing feeder {feeder_id}: {e}")
                continue

        all_results[file_name] = file_results

    # Log overall results
    logging.info("\nOverall Results:")
    all_metrics = {
        'RMSE': [], 'MAE': [], 'MSE': [], 'MAPE': [], 'R2': []
    }
    
    for file_name, file_results in all_results.items():
        for feeder_id, results in file_results.items():
            for metric_name, value in results['metrics'].items():
                all_metrics[metric_name].append(value)
    
    logging.info("\nAverage Metrics Across All Feeders:")
    for metric_name, values in all_metrics.items():
        if values:  # Check if we have any values
            avg_value = np.mean(values)
            if metric_name == 'R2':
                logging.info(f'Average {metric_name}: {avg_value:.4f}')
            else:
                logging.info(f'Average {metric_name}: {avg_value:.2f}')

    # Save results to file
    results_file = os.path.join(results_dir, 'lstm_results.csv')
    with open(results_file, 'w') as f:
        f.write('file_name,feeder_id,rmse,mae,mse,mape,r2\n')
        for file_name, file_results in all_results.items():
            for feeder_id, results in file_results.items():
                metrics = results['metrics']
                f.write(f'{file_name},{feeder_id},{metrics["RMSE"]:.2f},{metrics["MAE"]:.2f},'
                       f'{metrics["MSE"]:.2f},{metrics["MAPE"]:.2f},{metrics["R2"]:.4f}\n')
    
    logging.info(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()