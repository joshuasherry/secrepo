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
def train_graphsage(node_features, edge_index, pos_edges, neg_edges, epochs=100, lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGE(node_features.size(1), 64, 32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    node_features = node_features.to(device)
    edge_index = edge_index.to(device)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Get embeddings from the model
        embeddings = model(node_features, edge_index)
        
        # Prepare positive and negative edges for link prediction
        pos_edge_tensors = torch.tensor(pos_edges, dtype=torch.long, device=device).T
        neg_edge_tensors = torch.tensor(neg_edges, dtype=torch.long, device=device).T

        # Compute positive and negative similarities (dot product)
        pos_scores = (embeddings[pos_edge_tensors[0]] * embeddings[pos_edge_tensors[1]]).sum(dim=1)
        neg_scores = (embeddings[neg_edge_tensors[0]] * embeddings[neg_edge_tensors[1]]).sum(dim=1)
        
        pos_probs = torch.sigmoid(pos_scores)
        neg_probs = torch.sigmoid(neg_scores)

        # Labels: 1 for positive edges, 0 for negative edges
        labels = torch.cat([torch.ones(pos_probs.size(0)), torch.zeros(neg_probs.size(0))]).to(device)
        probs = torch.cat([pos_probs, neg_probs])

        # **Fix: Use BCE Loss Directly on Probabilities**
        loss = F.binary_cross_entropy(probs, labels)
        
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
    
    # Set a cutoff timestamp for training/test split
    cutoff_timestamp = edges_df['timestamp'].quantile(0.8)
    
    # Training set: edges before the cutoff
    training_edges = edges_df[edges_df['timestamp'] <= cutoff_timestamp][['sourceId', 'targetId']].values
    pos_edges_train = [(row[0], row[1]) for row in training_edges]
    
    # Testing set: edges after the cutoff
    test_edges = edges_df[edges_df['timestamp'] > cutoff_timestamp][['sourceId', 'targetId']].values
    pos_edges_test = [(row[0], row[1]) for row in test_edges]
    
    # Generate negative edges
    positive_edges_set = set(pos_edges_train)  # Use training edges only to prevent data leakage
    neg_edges_train = sample_negative_edges(len(pos_edges_train), list(node_index_map.values()), positive_edges_set)
    neg_edges_test = sample_negative_edges(len(pos_edges_test), list(node_index_map.values()), positive_edges_set)
    
    print("Training GraphSAGE to learn node embeddings...")
    embeddings = train_graphsage(node_features, edge_index, pos_edges_train, neg_edges_train)

    # Prepare training data for logistic regression
    print("Preparing training data...")
    X_train, y_train = prepare_training_data(embeddings, pos_edges_train, neg_edges_train)

    # Train a logistic regression classifier
    print("Training classifier...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Evaluate on training data
    train_probs = clf.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_probs)
    print(f"Training AUC: {train_auc:.4f}")
    
    # Prepare test data
    print("Preparing test data...")
    X_test, y_test = prepare_training_data(embeddings, pos_edges_test, neg_edges_test)

    print("Predicting on test data...")
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)
    print(f"Test AUC: {test_auc:.4f}")

    # Output top predictions
    top_k = 50
    top_indices = np.argsort(test_probs)[-top_k:][::-1]
    print(f"\nTop {top_k} predicted future links:")
    for idx in top_indices:
        u, v = pos_edges_test[idx]
        score = test_probs[idx]
        print(f"Edge {u} - {v} with predicted score {score:.4f}")
        
if __name__ == '__main__':
    main()
