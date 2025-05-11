import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import logging
from models.LSTM import LSTMWithEmbedding
from utils.metrics import calculate_metrics

# Set up logging
logging.basicConfig(filename = 'lstm_training_embeddings3.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class HistoryAwareTimeSeriesDataset(Dataset):
    def __init__(self, energy_values, timestamps, sequence_length, start_idx, end_idx, embedding):
        self.X = []
        self.y = []
        self.timestamps = []
        self.embedding = torch.FloatTensor(embedding)
        self.sequence_length = sequence_length

        for i in range(start_idx, end_idx - sequence_length):
            sequence = energy_values[i:i + sequence_length]
            self.X.append(sequence)
            self.y.append(energy_values[i + sequence_length])
            self.timestamps.append(timestamps[i + sequence_length])

        self.X = np.array(self.X)
        self.y = np.array(self.y).reshape(-1, 1)
        self.timestamps = np.array(self.timestamps)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx]), self.embedding

def load_and_preprocess_data(file_path, embedding_map, feeder_to_substation, sequence_length=48):
    df = pd.read_csv(file_path)
    df['data_collection_log_timestamp'] = pd.to_datetime(df['data_collection_log_timestamp'])

    feeders = df['lv_feeder_unique_id'].unique()
    feeder_data = {}

    for feeder in feeders:
        if feeder not in feeder_to_substation:
            continue
        substation_id = feeder_to_substation[feeder]
        if substation_id not in embedding_map:
            continue

        substation_emb = embedding_map[substation_id]
        feeder_df = df[df['lv_feeder_unique_id'] == feeder]
        timestamps = feeder_df['data_collection_log_timestamp'].values
        consumption = feeder_df['total_consumption_active_import'].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        consumption_scaled = scaler.fit_transform(consumption).flatten()

        total_length = len(consumption_scaled)
        train_size = int(total_length * 0.8)

        train_dataset = HistoryAwareTimeSeriesDataset(
            consumption_scaled, timestamps, sequence_length, 0, train_size, substation_emb
        )
        val_dataset = HistoryAwareTimeSeriesDataset(
            consumption_scaled, timestamps, sequence_length, train_size - sequence_length, total_length, substation_emb
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
        for batch_X, batch_y, batch_emb in train_loader:
            batch_X = batch_X.unsqueeze(-1).to(device)
            batch_y = batch_y.to(device)
            batch_emb = batch_emb.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X, batch_emb)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

def main():
    sequence_length = 48
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.0003
    data_dir = '../data'
    results_dir = '../results/lstm'

    os.makedirs(results_dir, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Load processed files into a set
    processed_files = set()
    try:
        with open('processed_files_embeddings.txt', 'r') as f:
            processed_files = set(line.strip() for line in f)
        logging.info(f"Loaded {len(processed_files)} previously processed files")
    except FileNotFoundError:
        logging.info("No processed files record found. Starting fresh.")
        processed_files = set()

    # Load embeddings and mapping
    emb_dict = np.load('../etc/substation_embeddings.npy', allow_pickle=True).item()
    logging.info("Original embedding statistics:")
    logging.info(f"Shape: {emb_dict['embeddings'].shape}")
    logging.info(f"Min: {emb_dict['embeddings'].min():.4f}, Max: {emb_dict['embeddings'].max():.4f}, Mean: {emb_dict['embeddings'].mean():.4f}")
    
    # Scale the embeddings
    scaler = MinMaxScaler()
    scaled_embeddings = scaler.fit_transform(emb_dict['embeddings'])
    logging.info("\nScaled embedding statistics:")
    logging.info(f"Min: {scaled_embeddings.min():.4f}, Max: {scaled_embeddings.max():.4f}, Mean: {scaled_embeddings.mean():.4f}")
    
    # Create embedding map with scaled embeddings
    embedding_map = dict(zip(emb_dict['substation_ids'], scaled_embeddings))
    
    feeder_map_df = pd.read_csv('../etc/feeder_to_substation.csv')
    feeder_to_substation = dict(zip(feeder_map_df['lv_feeder_unique_id'], feeder_map_df['substation_id']))

    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    all_results = {}

    for file_name in data_files:
        
        # Skip if file has already been processed
        if file_name in processed_files:
            logging.info(f"Skipping already processed file: {file_name}")
            continue

        logging.info(f"\nProcessing file: {file_name}")
        file_path = os.path.join(data_dir, file_name)

        try:
            feeder_data = load_and_preprocess_data(file_path, embedding_map, feeder_to_substation, sequence_length)
        except Exception as e:
            logging.error(f"Error loading data from {file_name}: {e}")
            continue

        file_results = {}

        for feeder_id, data in feeder_data.items():
            
            logging.info(f"\nProcessing feeder: {feeder_id}")

            try:
                train_loader = DataLoader(data['train_dataset'], batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(data['val_dataset'], batch_size=batch_size)

                model = LSTMWithEmbedding(embedding_dim=16).to(device)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                train_model(model, train_loader, criterion, optimizer, num_epochs, device)

                model.eval()
                predictions = []
                actuals = []

                with torch.no_grad():
                    for batch_X, batch_y, batch_emb in val_loader:
                        batch_X = batch_X.unsqueeze(-1).to(device)
                        batch_emb = batch_emb.to(device)
                        outputs = model(batch_X, batch_emb)
                        predictions.extend(outputs.cpu().numpy())
                        actuals.extend(batch_y.numpy())

                predictions = data['scaler'].inverse_transform(np.array(predictions))
                print(predictions)
                actuals = data['scaler'].inverse_transform(np.array(actuals))
                timestamps = data['val_timestamps']

                metrics = calculate_metrics(actuals, predictions)
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

        # Add successfully processed file to the set and update the file
        processed_files.add(file_name)
        with open('processed_files_embeddings.txt', 'a') as f:
            f.write(f"{file_name}\n")
        logging.info(f"Added {file_name} to processed files list")

    print("\nOverall Results:")
    all_metrics = {'RMSE': [], 'MAE': [], 'MSE': [], 'MAPE': [], 'R2': []}

    for file_name, file_results in all_results.items():
        for feeder_id, results in file_results.items():
            for metric_name, value in results['metrics'].items():
                all_metrics[metric_name].append(value)

    print("\nAverage Metrics Across All Feeders:")
    for metric_name, values in all_metrics.items():
        if values:
            avg_value = np.mean(values)
            print(f'Average {metric_name}: {avg_value:.4f}' if metric_name == 'R2' else f'Average {metric_name}: {avg_value:.2f}')

    results_file = os.path.join(results_dir, 'lstm_results.csv')
    with open(results_file, 'w') as f:
        f.write('file_name,feeder_id,rmse,mae,mse,mape,r2\n')
        for file_name, file_results in all_results.items():
            for feeder_id, results in file_results.items():
                metrics = results['metrics']
                f.write(f'{file_name},{feeder_id},{metrics["RMSE"]:.2f},{metrics["MAE"]:.2f},'
                        f'{metrics["MSE"]:.2f},{metrics["MAPE"]:.2f},{metrics["R2"]:.4f}\n')

    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()