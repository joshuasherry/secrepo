import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from itertools import combinations

# Load data
nodes_df = pd.read_csv('nodes.csv')  # nodeId, nodeType, createdTimestamp, updatedTimestamp
edges_df = pd.read_csv('edges.csv')  # sourceId, targetId, edgeName, timeStamp

# Encode categorical nodeType
node_type_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
nodes_encoded = node_type_encoder.fit_transform(nodes_df[['nodeType']])

# Normalize timestamps
nodes_df['createdTimestamp'] = (nodes_df['createdTimestamp'] - nodes_df['createdTimestamp'].min()) / (nodes_df['createdTimestamp'].max() - nodes_df['createdTimestamp'].min())
nodes_df['updatedTimestamp'] = (nodes_df['updatedTimestamp'] - nodes_df['updatedTimestamp'].min()) / (nodes_df['updatedTimestamp'].max() - nodes_df['updatedTimestamp'].min())

# Create node feature matrix
node_features = np.hstack([nodes_encoded, nodes_df[['createdTimestamp', 'updatedTimestamp']].values])
node_features = torch.tensor(node_features, dtype=torch.float)

# Create edge index (Graph structure)
edge_index = torch.tensor(edges_df[['sourceId', 'targetId']].values.T, dtype=torch.long)

# Define GNN model
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Create model
model = GNN(in_channels=node_features.shape[1], hidden_channels=64, out_channels=32)

# Train-test split (time-based)
train_edges = edges_df[edges_df['timeStamp'] < edges_df['timeStamp'].quantile(0.8)]
test_edges = edges_df[edges_df['timeStamp'] >= edges_df['timeStamp'].quantile(0.8)]

# Generate negative samples (non-existent edges)
all_possible_edges = set(combinations(nodes_df['nodeId'], 2))
existing_edges = set(zip(edges_df['sourceId'], edges_df['targetId']))
negative_edges = list(all_possible_edges - existing_edges)
negative_edges = np.random.choice(len(negative_edges), len(edges_df), replace=False)

def create_edge_tensor(df):
    return torch.tensor(df[['sourceId', 'targetId']].values.T, dtype=torch.long)

train_edge_index = create_edge_tensor(train_edges)
test_edge_index = create_edge_tensor(test_edges)

# Training loop
def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    optimizer.zero_grad()
    z = model(node_features, edge_index)
    pos_pred = (z[train_edge_index[0]] * z[train_edge_index[1]]).sum(dim=-1)
    neg_pred = (z[test_edge_index[0]] * z[test_edge_index[1]]).sum(dim=-1)
    loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred)) + \
           F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
    loss.backward()
    optimizer.step()
    return loss.item()

# Run training
for epoch in range(100):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# Link prediction evaluation
def evaluate():
    model.eval()
    with torch.no_grad():
        z = model(node_features, edge_index)
        pos_pred = (z[test_edge_index[0]] * z[test_edge_index[1]]).sum(dim=-1).sigmoid()
        neg_pred = (z[test_edge_index[0]] * z[test_edge_index[1]]).sum(dim=-1).sigmoid()
        y_true = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])
        y_pred = torch.cat([pos_pred, neg_pred])
        auc = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
    return auc

print('AUC Score:', evaluate())
