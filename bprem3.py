import pandas as pd
import numpy as np
import networkx as nx
import igraph as ig
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from tqdm import tqdm

# Load node and edge data
nodes = pd.read_csv("nodes.csv")
edges = pd.read_csv("edges.csv")

# Create graph using iGraph for efficiency
G = ig.Graph.TupleList(edges[['sourceId', 'targetId']].itertuples(index=False), directed=False)

# Convert to NetworkX for feature engineering
nx_G = nx.Graph()
nx_G.add_edges_from(edges[['sourceId', 'targetId']].values)

# Generate feature set for link prediction
def generate_features(nx_G, node_pairs):
    """
    Generate link prediction features for given node pairs.
    """
    features = []
    for u, v in tqdm(node_pairs):
        # Common neighbors
        cn = len(list(nx.common_neighbors(nx_G, u, v))) if nx_G.has_node(u) and nx_G.has_node(v) else 0
        # Jaccard coefficient
        jaccard = list(nx.jaccard_coefficient(nx_G, [(u, v)]))[0][2] if nx_G.has_node(u) and nx_G.has_node(v) else 0
        # Adamic/Adar index
        adamic_adar = list(nx.adamic_adar_index(nx_G, [(u, v)]))[0][2] if nx_G.has_node(u) and nx_G.has_node(v) else 0
        # Resource allocation index
        resource_alloc = list(nx.resource_allocation_index(nx_G, [(u, v)]))[0][2] if nx_G.has_node(u) and nx_G.has_node(v) else 0
        features.append([u, v, cn, jaccard, adamic_adar, resource_alloc])
    
    return pd.DataFrame(features, columns=["sourceId", "targetId", "common_neighbors", "jaccard", "adamic_adar", "resource_allocation"])

# Generate positive samples (existing edges)
pos_samples = edges[['sourceId', 'targetId']].values.tolist()

# Generate negative samples (non-existent edges)
all_nodes = list(nx_G.nodes)
neg_samples = set()
while len(neg_samples) < len(pos_samples):
    u, v = np.random.choice(all_nodes, 2, replace=False)
    if not nx_G.has_edge(u, v):
        neg_samples.add((u, v))

neg_samples = list(neg_samples)

# Label data
pos_labels = np.ones(len(pos_samples))
neg_labels = np.zeros(len(neg_samples))

# Generate features for both sets
print("Generating features for positive samples...")
pos_features = generate_features(nx_G, pos_samples)
print("Generating features for negative samples...")
neg_features = generate_features(nx_G, neg_samples)

# Combine datasets
X = pd.concat([pos_features, neg_features], axis=0).reset_index(drop=True)
y = np.concatenate([pos_labels, neg_labels])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X.iloc[:, 2:], y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
model = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=7)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred)
print(f"ROC AUC Score: {auc_score:.4f}")

# Predict new future links
def predict_future_links(nx_G, model, top_k=100):
    """
    Predict the top K most likely future links.
    """
    potential_links = set()
    nodes = list(nx_G.nodes)
    
    # Generate random pairs of unconnected nodes
    while len(potential_links) < 10 * top_k:  # More pairs to get top K later
        u, v = np.random.choice(nodes, 2, replace=False)
        if not nx_G.has_edge(u, v):
            potential_links.add((u, v))
    
    potential_links = list(potential_links)
    print("Generating features for future links...")
    future_features = generate_features(nx_G, potential_links)
    
    future_predictions = model.predict_proba(future_features.iloc[:, 2:])[:, 1]
    future_features["score"] = future_predictions
    
    # Sort by highest probability
    future_features = future_features.sort_values(by="score", ascending=False)
    
    return future_features.head(top_k)

# Get top 100 predicted future links
predicted_links = predict_future_links(nx_G, model, top_k=100)

# Save results
predicted_links.to_csv("predicted_future_links.csv", index=False)
print("Predicted future links saved.")
