import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling, train_test_split_edges
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

# Load data
nodes_df = pd.read_csv('nodes.csv')  # nodeId, nodeType, createdTimestamp, updatedTimestamp
edges_df = pd.read_csv('edges.csv')  # sourceId, targetId, edgeName, timeStamp

# Normalize timestamps
scaler = MinMaxScaler()
nodes_df[['createdTimestamp', 'updatedTimestamp']] = scaler.fit_transform(
    nodes_df[['createdTimestamp', 'updatedTimestamp']]
)
edges_df['timeStamp'] = scaler.transform(edges_df[['timeStamp']])

# Create mappings
node_mapping = {id: i for i, id in enumerate(nodes_df['nodeId'])}
edges_df[['sourceId', 'targetId']] = edges_df[['sourceId', 'targetId']].applymap(node_mapping.get)

# Convert to PyTorch tensors
edge_index = torch.tensor(edges_df[['sourceId', 'targetId']].values.T, dtype=torch.long)
x = torch.tensor(nodes_df[['createdTimestamp', 'updatedTimestamp']].values, dtype=torch.float)

# Create PyTorch Geometric data object
data = Data(x=x, edge_index=edge_index)
data = train_test_split_edges(data)

# Define GraphSAGE Model
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        row, col = edge_label_index
        return (z[row] * z[col]).sum(dim=1)

# Train model
model = GNN(in_channels=2, hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    pos_pred = model.decode(z, data.train_pos_edge_index)
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1)
    )
    neg_pred = model.decode(z, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(
        torch.cat([pos_pred, neg_pred]),
        torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))])
    )
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Evaluate model
model.eval()
z = model.encode(data.x, data.train_pos_edge_index)
pos_pred = model.decode(z, data.test_pos_edge_index).sigmoid().detach().cpu().numpy()
neg_pred = model.decode(z, data.test_neg_edge_index).sigmoid().detach().cpu().numpy()
y_true = torch.cat([torch.ones(len(pos_pred)), torch.zeros(len(neg_pred))]).numpy()
y_score = torch.cat([torch.tensor(pos_pred), torch.tensor(neg_pred)]).numpy()
auc = roc_auc_score(y_true, y_score)
print(f'AUC Score: {auc:.4f}')
