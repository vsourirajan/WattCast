import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.GNN import GCN

def train_gnn_model(model, data, num_epochs=500, lr=0.01, noise_std=0.05, device='cpu'):
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Add Gaussian noise to the input features
        noise = torch.randn_like(data.x) * noise_std
        noisy_x = data.x + noise

        # Forward pass using noisy input
        out = model(noisy_x, data.edge_index)

        # Reconstruction loss vs original clean input
        loss = criterion(out, data.x)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {loss.item():.4f}')

    # Return model and final embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index)

    return model, embeddings.cpu().numpy(), data.substation_ids

def main():
    graph_path = '../etc/substation_graph.pt'
    embedding_out_path = '../etc/substation_embeddings.npy'

    os.makedirs(os.path.dirname(embedding_out_path), exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        graph_data = torch.load(graph_path, weights_only=False)
        print(f"Loaded graph from {graph_path}")
        print(f"Graph has {graph_data.x.shape[0]} nodes and {graph_data.edge_index.shape[1]} edges")
    except Exception as e:
        print(f"Error loading graph: {e}")
        return

    try:
        in_dim = graph_data.x.shape[1]

        model = GCN(in_channels=in_dim, hidden_channels=64, out_channels=in_dim)
        model, embeddings, substation_ids = train_gnn_model(model, graph_data, num_epochs=500, lr=0.01, noise_std=0.05, device=device)

        np.save(embedding_out_path, {'embeddings': embeddings, 'substation_ids': substation_ids})
        print(f"Saved embeddings to {embedding_out_path}")
    except Exception as e:
        print(f"Error training GNN: {e}")

if __name__ == '__main__':
    main()
