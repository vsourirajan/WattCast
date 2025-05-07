import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.GNN import GCN
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

def visualize_graph(graph_data, embeddings=None, title="Graph Visualization"):
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(graph_data.x.shape[0]):
        G.add_node(i)
    
    # Add edges
    edge_index = graph_data.edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot graph structure
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, ax=ax1, font_size=8)
    ax1.set_title("Graph Structure")
    
    # Plot embeddings if provided
    if embeddings is not None:
        # Use t-SNE to reduce dimensionality to 2D
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot embeddings
        scatter = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=np.arange(len(embeddings_2d)), cmap='viridis')
        ax2.set_title("Node Embeddings (t-SNE)")
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax2, label='Node Index')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def train_gnn_model(model, data, num_epochs=500, lr=0.001, noise_std=0.05, device='cpu'):
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

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} - Gradient Max: {param.grad.abs().max().item()}, Min: {param.grad.abs().min().item()}")

    # Return model and final embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index)

    return model, embeddings.cpu().numpy(), data.substation_ids

def main():
    graph_path = '../etc/substation_graph.pt'
    embedding_out_path = '../etc/substation_embeddings.npy'

    os.makedirs(os.path.dirname(embedding_out_path), exist_ok=True)
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    print(f"Using device: {device}")

    try:
        graph_data = torch.load(graph_path, weights_only=False)
        print(f"Loaded graph from {graph_path}")
        print(f"Graph has {graph_data.x.shape[0]} nodes and {graph_data.edge_index.shape[1]} edges")
        
        # Visualize initial graph
        visualize_graph(graph_data, title="Initial Graph Structure")
        
    except Exception as e:
        print(f"Error loading graph: {e}")
        return

    try:
        in_dim = graph_data.x.shape[1]
        print("Node features shape:", graph_data.x.shape)
        print("Edge index shape:", graph_data.edge_index.shape)
        if hasattr(graph_data, 'edge_attr'):
            print("Edge attributes shape:", graph_data.edge_attr.shape)

        print("in_dim:", in_dim)
        model = GCN(in_channels=in_dim, hidden_channels=16, out_channels=in_dim)
        model, embeddings, substation_ids = train_gnn_model(model, graph_data, num_epochs=300, lr=0.001, noise_std=0.05, device=device)

        # Scale the embeddings
        scaler = MinMaxScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)
        print("\nEmbedding statistics before scaling:")
        print(f"Min: {embeddings.min():.4f}, Max: {embeddings.max():.4f}, Mean: {embeddings.mean():.4f}")
        print("\nEmbedding statistics after scaling:")
        print(f"Min: {scaled_embeddings.min():.4f}, Max: {scaled_embeddings.max():.4f}, Mean: {scaled_embeddings.mean():.4f}")

        # Visualize final graph with scaled embeddings
        visualize_graph(graph_data, scaled_embeddings, title="Graph with Scaled Learned Embeddings")

        np.save(embedding_out_path, {'embeddings': scaled_embeddings, 'substation_ids': substation_ids})
        print(f"Saved scaled embeddings to {embedding_out_path}")
    except Exception as e:
        print(f"Error training GNN: {e}")

if __name__ == '__main__':
    main()
