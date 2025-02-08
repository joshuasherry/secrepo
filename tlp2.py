import pandas as pd
import networkx as nx
import numpy as np
import random
import time
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from node2vec import Node2Vec
from joblib import Parallel, delayed

# ---------------------------
# 1. Data Loading & Preprocessing
# ---------------------------

def load_data(nodes_file, edges_file):
    """Load and preprocess data, encoding categorical features."""
    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)
    
    # Encode categorical columns
    label_encoders = {}
    for col in ['nodeType', 'anotherCategory']:  # Adjust based on actual columns
        le = LabelEncoder()
        nodes_df[col] = le.fit_transform(nodes_df[col])
        label_encoders[col] = le
    
    return nodes_df, edges_df

# ---------------------------
# 2. Graph Building
# ---------------------------

def build_graph(nodes_df, edges_df, cutoff_timestamp):
    """Build a graph using edges up to cutoff_timestamp."""
    G = nx.Graph()
    node_feature_cols = [col for col in nodes_df.columns if col not in ['nodeId', 'createdTimestamp', 'updatedTimestamp']]
    
    for _, row in nodes_df.iterrows():
        G.add_node(row['nodeId'], features={col: row[col] for col in node_feature_cols},
                   created=row['createdTimestamp'], updated=row['updatedTimestamp'])
    
    training_edges = edges_df[edges_df['timeStamp'] <= cutoff_timestamp]
    for _, row in training_edges.iterrows():
        G.add_edge(row['sourceId'], row['targetId'], timestamp=row['timeStamp'])
    
    return G

# ---------------------------
# 3. Feature Computation
# ---------------------------

def compute_network_features(G, u, v):
    """Compute advanced network-based features."""
    cn = len(list(nx.common_neighbors(G, u, v)))
    jc = next(nx.jaccard_coefficient(G, [(u, v)]), (None, None, 0))[2]
    pa = G.degree[u] * G.degree[v]
    aa = sum(1 / np.log(G.degree[w]) for w in nx.common_neighbors(G, u, v) if G.degree[w] > 1)
    katz = nx.katz_centrality(G)[u] + nx.katz_centrality(G)[v]
    ppr = nx.pagerank(G)[u] + nx.pagerank(G)[v]
    return [cn, jc, pa, aa, katz, ppr]

def compute_time_features(G, u, v, current_time):
    """Compute time-aware features like edge recency and node activity trends."""
    edge_times = [G[u][v]['timestamp'] for u, v in G.edges if G.has_edge(u, v)]
    if edge_times:
        last_interaction = max(edge_times)
        recency = np.exp(-(current_time - last_interaction) / (24 * 3600))
    else:
        recency = 0
    return [recency]

def compute_features(G, u, v, current_time):
    """Compute all features (network, time, node attributes)."""
    network_features = compute_network_features(G, u, v)
    time_features = compute_time_features(G, u, v, current_time)
    u_attrs, v_attrs = G.nodes[u]['features'], G.nodes[v]['features']
    attr_features = [1 if u_attrs[k] == v_attrs[k] else 0 for k in u_attrs]
    return network_features + time_features + attr_features

# ---------------------------
# 4. Negative Sampling
# ---------------------------

def sample_negative_edges(G, num_samples):
    """Hard negative sampling by selecting unconnected node pairs with common neighbors."""
    negatives = set()
    nodes = list(G.nodes())
    while len(negatives) < num_samples:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v) and len(set(G.neighbors(u)) & set(G.neighbors(v))) > 0:
            negatives.add((u, v))
    return list(negatives)

# ---------------------------
# 5. Training & Prediction
# ---------------------------

def train_and_evaluate(G, pos_edges, neg_edges, current_time):
    """Train model and evaluate on training set."""
    X_train = [compute_features(G, u, v, current_time) for u, v in pos_edges + neg_edges]
    y_train = [1] * len(pos_edges) + [0] * len(neg_edges)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
    return clf, auc

# ---------------------------
# 6. Candidate Generation & Prediction
# ---------------------------

def generate_candidate_edges(G):
    """Generate candidate edges based on shared neighbors."""
    candidates = set()
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        for u, v in combinations(neighbors, 2):
            if not G.has_edge(u, v):
                candidates.add((u, v))
    return list(candidates)

def predict_future_links(G, clf, candidates, current_time):
    """Predict future links based on learned model."""
    X_test = [compute_features(G, u, v, current_time) for u, v in candidates]
    scores = clf.predict_proba(X_test)[:, 1]
    return sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

# ---------------------------
# 7. Main Execution
# ---------------------------

def main():
    nodes_file, edges_file = 'nodes.csv', 'edges.csv'
    nodes_df, edges_df = load_data(nodes_file, edges_file)
    cutoff_timestamp = edges_df['timeStamp'].quantile(0.8)
    G_train = build_graph(nodes_df, edges_df, cutoff_timestamp)
    
    pos_edges = [(u, v) for u, v in zip(edges_df['sourceId'], edges_df['targetId']) if edges_df['timeStamp'].iloc[0] > cutoff_timestamp]
    neg_edges = sample_negative_edges(G_train, len(pos_edges))
    
    clf, auc = train_and_evaluate(G_train, pos_edges, neg_edges, cutoff_timestamp)
    print(f"Training AUC: {auc:.4f}")
    
    candidates = generate_candidate_edges(G_train)
    predictions = predict_future_links(G_train, clf, candidates, cutoff_timestamp)
    
    print("Top predicted future links:")
    for (u, v), score in predictions[:50]:
        print(f"{u} - {v}: {score:.4f}")

if __name__ == '__main__':
    main()
