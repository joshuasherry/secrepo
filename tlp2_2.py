import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import xgboost as xgb

# ---------------------------
# 1. Load Data
# ---------------------------
def load_data(nodes_file, edges_file):
    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)
    
    node_features = nodes_df.drop(['nodeId'], axis=1).values
    node_ids = nodes_df['nodeId'].values
    node_index_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    edges_df['sourceId'] = edges_df['sourceId'].map(node_index_map)
    edges_df['targetId'] = edges_df['targetId'].map(node_index_map)
    
    edge_index = torch.tensor(edges_df[['sourceId', 'targetId']].values.T, dtype=torch.long)
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
# 3. Improved Negative Sampling
# ---------------------------
def sample_negative_edges(num_samples, node_ids, pos_edges_set):
    negatives = set()
    while len(negatives) < num_samples:
        u, v = random.sample(node_ids, 2)
        if (u, v) not in pos_edges_set:
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
        embeddings = model(node_features, edge_index)
        
        pos_edge_tensors = torch.tensor(pos_edges, dtype=torch.long, device=device).T
        neg_edge_tensors = torch.tensor(neg_edges, dtype=torch.long, device=device).T

        pos_scores = (embeddings[pos_edge_tensors[0]] * embeddings[pos_edge_tensors[1]]).sum(dim=1)
        neg_scores = (embeddings[neg_edge_tensors[0]] * embeddings[neg_edge_tensors[1]]).sum(dim=1)
        
        pos_probs = torch.sigmoid(pos_scores)
        neg_probs = torch.sigmoid(neg_scores)
        
        labels = torch.cat([torch.ones(pos_probs.size(0)), torch.zeros(neg_probs.size(0))]).to(device)
        probs = torch.cat([pos_probs, neg_probs])
        
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
    nodes_file = 'featureListencoded.csv'
    edges_file = 'edges.csv'
    
    print("Loading data...")
    node_features, edge_index, edges_df, node_index_map = load_data(nodes_file, edges_file)
    
    cutoff_timestamp = edges_df['timestamp'].quantile(0.8)
    
    train_edges = edges_df[edges_df['timestamp'] <= cutoff_timestamp][['sourceId', 'targetId']].values
    test_edges = edges_df[edges_df['timestamp'] > cutoff_timestamp][['sourceId', 'targetId']].values
    
    pos_edges_train = [(row[0], row[1]) for row in train_edges]
    pos_edges_test = [(row[0], row[1]) for row in test_edges]
    
    train_edge_set = set(pos_edges_train)
    test_edge_set = set(pos_edges_test)
    
    neg_edges_train = sample_negative_edges(len(pos_edges_train), list(node_index_map.values()), train_edge_set)
    neg_edges_test = sample_negative_edges(len(pos_edges_test), list(node_index_map.values()), test_edge_set)
    
    print("Training GraphSAGE to learn node embeddings...")
    embeddings = train_graphsage(node_features, edge_index, pos_edges_train, neg_edges_train)
    
    print("Preparing training data...")
    X_train, y_train = prepare_training_data(embeddings, pos_edges_train, neg_edges_train)
    X_test, y_test = prepare_training_data(embeddings, pos_edges_test, neg_edges_test)
    
    print("Training classifier...")
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train, y_train)
    
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)
    print(f"Test AUC: {test_auc:.4f}")
    
if __name__ == '__main__':
    main()
