import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from models.LSTM import LSTM
from models.RNN import RNN
from common import TimeSeriesDataset

device = (
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

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
        data_scaled = scaler.fit_transform(consumption)
        
        X, y, y_timestamps = [], [], []
        for i in range(len(data_scaled) - sequence_length):
            X.append(data_scaled[i:(i + sequence_length)])
            y.append(data_scaled[i + sequence_length])
            y_timestamps.append(timestamps[i + sequence_length])
        X = np.array(X)
        y = np.array(y)
        y_timestamps = np.array(y_timestamps)
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        y_timestamps_train, y_timestamps_test = y_timestamps[:train_size], y_timestamps[train_size:]
        
        feeder_data[feeder] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'y_timestamps_train': y_timestamps_train,
            'y_timestamps_test': y_timestamps_test
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
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

def main():
    sequence_length = 48
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    
    feeder_data = load_and_preprocess_data('NGED-110191.csv', sequence_length)
    
    results = {}
    for feeder_id, data in feeder_data.items():
        print(f"\nProcessing feeder: {feeder_id}")
        
        train_dataset = TimeSeriesDataset(data['X_train'], data['y_train'])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        model = LSTM().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        train_model(model, train_loader, criterion, optimizer, num_epochs, device)
        
        model.eval()
        test_dataset = TimeSeriesDataset(data['X_test'], data['y_test'])
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.numpy())
        
        predictions = data['scaler'].inverse_transform(np.array(predictions))
        actuals = data['scaler'].inverse_transform(np.array(actuals))
        timestamps = data['y_timestamps_test']

        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))
        
        results[feeder_id] = {
            'predictions': predictions,
            'actuals': actuals,
            'rmse': rmse,
            'mae': mae,
            'timestamps': timestamps
        }
        
        full_actuals = data['scaler'].inverse_transform(np.vstack([data['y_train'], data['y_test']]))
        full_timestamps = np.concatenate([data['y_timestamps_train'], data['y_timestamps_test']])
        
        if isinstance(full_timestamps[0], str):
            full_timestamps = pd.to_datetime(full_timestamps)

        plt.figure(figsize=(12, 6))

        plt.plot(full_timestamps, full_actuals, label='Actual', linewidth=0.8)
        
        test_start_idx = len(data['y_timestamps_train'])
        plt.plot(full_timestamps[test_start_idx:], predictions, label='Predicted', linewidth=0.8)
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.gcf().autofmt_xdate()
        
        plt.legend()
        plt.title(f'Energy Consumption Forecasting - Feeder {feeder_id}')
        plt.xlabel('Date and Time')
        plt.ylabel('Consumption')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{feeder_id}_dec2024.png')
        plt.clf()
        print(f'Feeder {feeder_id} - Test RMSE: {rmse:.2f}, MAE: {mae:.2f}')
    
    print("\nOverall Results:")
    avg_rmse = np.mean([r['rmse'] for r in results.values()])
    avg_mae = np.mean([r['mae'] for r in results.values()])
    print(f'Average RMSE across all feeders: {avg_rmse:.2f}')
    print(f'Average MAE across all feeders: {avg_mae:.2f}')

if __name__ == "__main__":
    main()
