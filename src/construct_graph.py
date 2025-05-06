import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import numpy as np

def create_substation_graph(substation_metadata, k=5):
    """
    substation_metadata: pd.DataFrame with columns ['substation_id', 'latitude', 'longitude']
    """
    coords = substation_metadata[['latitude', 'longitude']].values
    substation_ids = substation_metadata['substation_id'].values
    id_to_idx = {sid: i for i, sid in enumerate(substation_ids)}

    # KNN for edges
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    # Build edge index
    edge_index = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # skip self-loop
            edge_index.append([i, j])

    edge_index = torch.tensor(edge_index).t().contiguous()

    # Use lat/lon as node features
    x = torch.tensor(coords, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    data.substation_ids = substation_ids  # optional metadata
    return data
