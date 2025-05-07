import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    

class LSTMWithEmbedding(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, embedding_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size + embedding_dim, 1)  # dynamically match sizes

    def forward(self, x_seq, x_emb):
        out, _ = self.lstm(x_seq)  # [B, T, hidden]
        lstm_out = out[:, -1, :]   # [B, hidden]
        combined = torch.cat([lstm_out, x_emb], dim=-1)  # [B, hidden + embedding_dim]
        return self.fc(combined)

    
    
        
        
