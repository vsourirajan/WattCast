import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from GCN import GCN
import pyarrow as pa

df = pd.read_csv("NGED-110191.csv")

df['longitude'] = df.geometry.str.extract(r'POINT \(([-\d.]+)')
df['latitude'] = df.geometry.str.extract(r'POINT \([-\d.]+ ([-\d.]+)')
df['longitude'] = df['longitude'].astype(float)
df['latitude'] = df['latitude'].astype(float)

feeders = df['lv_feeder_unique_id'].unique()

profile_df = df.groupby(['lv_feeder_unique_id', 'data_collection_log_timestamp'])['total_consumption_active_import'].mean().unstack()

scaler = StandardScaler()
profile_scaled = scaler.fit_transform(profile_df.fillna(0))

consumption_similarity = np.dot(profile_scaled, profile_scaled.T)

coords = df.groupby('lv_feeder_unique_id')[['latitude', 'longitude']].mean()
distance_matrix = cdist(coords, coords, metric='euclidean')

geo_threshold = np.percentile(distance_matrix, 25)
consumption_threshold = np.percentile(consumption_similarity, 75)

adj_matrix = (distance_matrix < geo_threshold) | (consumption_similarity > consumption_threshold)

edges = np.array(np.where(adj_matrix))
edge_index = torch.tensor(edges, dtype=torch.long)

x = torch.tensor(profile_scaled, dtype=torch.float)

in_features = x.shape[1]
hidden_dim = 64
out_features = 16

model = GCN(in_features, hidden_dim, out_features)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.MSELoss()

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out, x)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print("Training complete")