import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import torch

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_split_data(file_path, test_ratio=0.2):
    df = pd.read_csv(file_path)
    df['data_collection_log_timestamp'] = pd.to_datetime(df['data_collection_log_timestamp'])
    
    feeders = df['lv_feeder_unique_id'].unique()
    feeder_data = {}
    
    for feeder in feeders:
        feeder_df = df[df['lv_feeder_unique_id'] == feeder]
        feeder_df = feeder_df.sort_values('data_collection_log_timestamp')

        timestamps = feeder_df['data_collection_log_timestamp'].values
        consumption = feeder_df['total_consumption_active_import'].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(consumption)

        train_size = int(len(data_scaled) * (1 - test_ratio))
        train_data = data_scaled[:train_size]
        test_data = data_scaled[train_size:]
        train_timestamps = timestamps[:train_size]
        test_timestamps = timestamps[train_size:]
        
        feeder_data[feeder] = {
            'train': train_data,
            'test': test_data,
            'train_timestamps': train_timestamps,
            'test_timestamps': test_timestamps,
            'scaler': scaler
        }
    
    return feeder_data