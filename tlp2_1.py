import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score
import numpy as np
import random
from sklearn.linear_model import LogisticRegression

# ---------------------------
# 1. Load Data
# ---------------------------
def load_data(nodes_file, edges_file):
    # Load the label-encoded node features
    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)
    
    # Prepare node features and IDs
    node_features = nodes_df.drop(['nodeId'], axis=1).values
    node_ids = nodes_df['nodeId'].values
    
    # Create a mapping of nodeId to index
    node_index_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    # Convert source and target IDs in edges to corresponding indices
    edges_df['sourceId'] = edges_df['sourceId'].map(node_index_map)
    edges_df['targetId'] = edges_df['targetId'].map(node_index_map)
    
    # Create edge index (PyTorch Geometric format)
    edge_index = torch.tensor(edges_df[['sourceId', 'targetId']].values.T, dtype=torch.long)
    
    # Convert node features to tensor
    node_features = torch.tensor(node_features, dtype=torch.float)
    
    return node_features, edge_index, edges_df, node_index_map

# ---------------------------
# 2. GraphSAGE Model
# ---------------------------
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# ---------------------------
# 3. Sampling Negative Edges
# ---------------------------
def sample_negative_edges(num_samples, node_ids, positive_edges_set):
    negatives = set()
    while len(negatives) < num_samples:
        u, v = random.sample(node_ids, 2)
        if u != v and (u, v) not in positive_edges_set and (v, u) not in positive_edges_set:
            negatives.add((u, v))
    return list(negatives)

# ---------------------------
# 4. Prepare Training Data
# ---------------------------
def prepare_training_data(node_embeddings, pos_edges, neg_edges):
    X, y = [], []
    for u, v in pos_edges:
        X.append(np.concatenate([node_embeddings[u], node_embeddings[v]]))
        y.append(1)
    for u, v in neg_edges:
        X.append(np.concatenate([node_embeddings[u], node_embeddings[v]]))
        y.append(0)
    return np.array(X), np.array(y)

# ---------------------------
# 5. Train GraphSAGE
# ---------------------------
def train_graphsage(node_features, edge_index, epochs=100, lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGE(node_features.size(1), 64, 32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    node_features = node_features.to(device)
    edge_index = edge_index.to(device)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings = model(node_features, edge_index)
        loss = torch.norm(embeddings)  # L2 regularization (can be modified for supervised tasks)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    model.eval()
    with torch.no_grad():
        final_embeddings = model(node_features, edge_index).cpu().numpy()
    return final_embeddings

# ---------------------------
# 6. Main
# ---------------------------
def main():
    # File paths
    nodes_file = 'featureListencoded.csv'
    edges_file = 'edges.csv'
    
    print("Loading data...")
    node_features, edge_index, edges_df, node_index_map = load_data(nodes_file, edges_file)
    
    print("Training GraphSAGE to learn node embeddings...")
    embeddings = train_graphsage(node_features, edge_index)
    
    # Set a cutoff timestamp for training/test split
    cutoff_timestamp = edges_df['timestamp'].quantile(0.8)
    
    # Positive edges for training (edges before cutoff)
    training_edges = edges_df[edges_df['timestamp'] <= cutoff_timestamp][['sourceId', 'targetId']].values
    pos_edges = [(row[0], row[1]) for row in training_edges]
    
    # Negative sampling
    positive_edges_set = set(pos_edges)
    neg_edges = sample_negative_edges(len(pos_edges), list(node_index_map.values()), positive_edges_set)
    
    # Prepare training data (features and labels)
    print("Preparing training data...")
    X_train, y_train = prepare_training_data(embeddings, pos_edges, neg_edges)
    
    # Train a logistic regression classifier
    print("Training classifier...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    train_probs = clf.predict_proba(X_train)[:, 1]
    auc = roc_auc_score(y_train, train_probs)
    print(f"Training AUC: {auc:.4f}")
    
    # Predict future links (edges after the cutoff timestamp)
    test_edges = edges_df[edges_df['timestamp'] > cutoff_timestamp][['sourceId', 'targetId']].values
    test_pos_edges = [(row[0], row[1]) for row in test_edges]
    
    # Prepare test data
    neg_test_edges = sample_negative_edges(len(test_pos_edges), list(node_index_map.values()), positive_edges_set)
    X_test, y_test = prepare_training_data(embeddings, test_pos_edges, neg_test_edges)
    
    print("Predicting on test data...")
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)
    print(f"Test AUC: {test_auc:.4f}")
    
    # Output top predictions
    top_k = 50
    top_indices = np.argsort(test_probs)[-top_k:][::-1]
    print(f"\nTop {top_k} predicted future links:")
    for idx in top_indices:
        u, v = test_pos_edges[idx]
        score = test_probs[idx]
        print(f"Edge {u} - {v} with predicted score {score:.4f}")

if __name__ == '__main__':
    main()
