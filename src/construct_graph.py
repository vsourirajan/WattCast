import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import os
import numpy as np

def create_substation_graph(csv_path, output_path, k=5):
    """
    Constructs a substation graph and saves it as a PyTorch Geometric Data object.

    Args:
        csv_path (str): Path to CSV file with substation_id, latitude, longitude
        output_path (str): Where to save the graph (.pt)
        k (int): Number of nearest neighbors for edge construction
    """
    df = pd.read_csv(csv_path)
    required_columns = {'secondary_substation_unique_id', 'latitude', 'longitude'}
    if not required_columns.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain the following columns: {required_columns}")

    substation_ids = df['secondary_substation_unique_id'].values
    coords = df[['latitude', 'longitude']].values
    id_to_idx = {sid: i for i, sid in enumerate(substation_ids)}

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    edge_index = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:
            edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(coords, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    data.substation_ids = substation_ids

    torch.save(data, output_path)
    print(f"Graph saved to {output_path} with {x.shape[0]} nodes and {edge_index.shape[1]} edges.")

if __name__ == "__main__":
    csv_path = "../etc/substation_locations.csv"
    output_path = "../etc/substation_graph.pt"
    k = 5 

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    create_substation_graph(csv_path, output_path, k=k)
