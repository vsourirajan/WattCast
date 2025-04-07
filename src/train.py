import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size=12, horizon=12):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.window_size - self.horizon + 1

    def __getitem__(self, idx):
        window = self.data[idx:idx+self.window_size]
        future = self.data[idx+self.window_size:idx+self.window_size+self.horizon]
        return torch.tensor(window, dtype=torch.float32), torch.tensor(future, dtype=torch.float32)
    
def create_dataloaders(data, batch_size=32, window_size=12, horizon=12):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    dataset = TimeSeriesDataset(data, window_size, horizon)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    data = pd.read_parquet("london_dec_2024.parquet")
    dataloader = create_dataloaders(data)
    for batch in dataloader:
        print(batch)
        break
