import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_dim, dropout=0.2):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.fc = nn.Linear((hidden_dim // 2) + num_features, 1)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        
        combined = torch.cat([h, x], dim=1) # Tabular Skip Connection
        out = self.fc(combined)
        return out

class GraphSAGEModel(nn.Module):
    def __init__(self, num_features, hidden_dim, dropout=0.2):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim // 2)
        self.fc = nn.Linear((hidden_dim // 2) + num_features, 1)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        
        combined = torch.cat([h, x], dim=1)
        out = self.fc(combined)
        return out

class GATModel(nn.Module):
    def __init__(self, num_features, hidden_dim, dropout=0.2):
        super(GATModel, self).__init__()
        # GAT needs careful dimension handling with heads
        self.conv1 = GATConv(num_features, hidden_dim // 2, heads=2)
        self.conv2 = GATConv(hidden_dim, hidden_dim // 4, heads=2)
        self.fc = nn.Linear((hidden_dim // 2) + num_features, 1)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        
        combined = torch.cat([h, x], dim=1)
        out = self.fc(combined)
        return out
