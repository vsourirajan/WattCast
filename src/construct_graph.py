import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import os
import numpy as np

def create_substation_graph(loc_csv_path, profile_csv_path, output_path, k=5):
    """
    Constructs a substation graph using profile features and spatial edges.

    Args:
        loc_csv_path (str): Path to CSV with 'secondary_substation_unique_id', 'latitude', 'longitude'
        profile_csv_path (str): Path to CSV with 'substation_id' and profile features
        output_path (str): Where to save the torch_geometric Data object
        k (int): Number of nearest neighbors for edge construction
    """
    # Load and merge
    loc_df = pd.read_csv(loc_csv_path)
    loc_df = loc_df.drop_duplicates(subset="secondary_substation_unique_id").reset_index(drop=True)

    print(loc_df.head())

    profile_df = pd.read_csv(profile_csv_path)

    print(profile_df.head())

    merged_df = loc_df.merge(
        profile_df,
        left_on="secondary_substation_unique_id",
        right_on="substation_id",
        how="inner"
    )

    if merged_df.shape[0] == 0:
        raise ValueError("No matching substations found between location and profile files.")

    coords = merged_df[['latitude', 'longitude']].values
    profile_features = merged_df[[col for col in profile_df.columns if col.startswith("raw_") or col.startswith("normalized_")]].values
    substation_ids = merged_df['secondary_substation_unique_id'].values

    # Build edge index with kNN
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    edge_index = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:
            edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(profile_features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    data.substation_ids = substation_ids

    torch.save(data, output_path)
    print(f"Graph saved to {output_path} with {x.shape[0]} nodes and {edge_index.shape[1]} edges.")

if __name__ == "__main__":
    loc_csv = "../etc/substation_locations.csv"
    profile_csv = "../etc/substation_profiles.csv"
    out_file = "../etc/substation_graph.pt"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    create_substation_graph(loc_csv, profile_csv, out_file, k=5)
