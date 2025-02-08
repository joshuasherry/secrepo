import pandas as pd
import networkx as nx
import numpy as np
import random
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import time

# ---------------------------
# 1. Data Loading Functions
# ---------------------------

def load_data(nodes_file, edges_file):
    """Load nodes and edges data from CSV files."""
    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)
    return nodes_df, edges_df

# ---------------------------
# 2. Graph Building
# ---------------------------

def build_training_graph(nodes_df, edges_df, cutoff_timestamp):
    """
    Build a graph from the nodes and training edges.
    Only include edges with timeStamp <= cutoff_timestamp.
    """
    G = nx.Graph()
    
    # Add nodes with attributes (e.g. type, creation time)
    for _, row in nodes_df.iterrows():
        G.add_node(row['nodeId'],
                   nodeType=row['nodeType'],
                   createdTimestamp=row['createdTimestamp'],
                   updatedTimestamp=row['updatedTimestamp'])
    
    # Add training edges (edges up to cutoff)
    training_edges = edges_df[edges_df['timeStamp'] <= cutoff_timestamp]
    for _, row in training_edges.iterrows():
        # (If nodes might be missing, you can add them here.)
        G.add_edge(row['sourceId'], row['targetId'],
                   edgeName=row['edgeName'],
                   timeStamp=row['timeStamp'])
    return G

# ---------------------------
# 3. Splitting Data & Sampling
# ---------------------------

def get_future_positive_edges(G_train, edges_df, cutoff_timestamp):
    """
    From the full edge list, extract positive (future) edges:
    those edges with timeStamp > cutoff and that are not in the training graph.
    """
    test_edges = edges_df[edges_df['timeStamp'] > cutoff_timestamp]
    pos_edges = []
    for _, row in test_edges.iterrows():
        u, v = row['sourceId'], row['targetId']
        # Make sure we have a unique (sorted) tuple and that it wasn’t in training.
        if not G_train.has_edge(u, v):
            pos_edges.append(tuple(sorted((u, v))))
    return list(set(pos_edges))  # remove duplicates

def sample_negative_edges(G, num_samples):
    """
    Randomly sample negative node pairs (i.e. pairs not connected in G).
    For efficiency, we use random sampling. (For a very large graph, consider more advanced methods.)
    """
    negatives = set()
    nodes = list(G.nodes())
    while len(negatives) < num_samples:
        u = random.choice(nodes)
        v = random.choice(nodes)
        if u == v:
            continue
        if G.has_edge(u, v):
            continue
        negatives.add(tuple(sorted((u, v))))
    return list(negatives)

# ---------------------------
# 4. Feature Computation
# ---------------------------

def compute_features(G, edge):
    """
    Compute several link-prediction features for a given node pair (u, v):
      - Common Neighbors count
      - Jaccard Coefficient
      - Preferential Attachment score
      - Adamic–Adar index
    """
    u, v = edge

    # Common Neighbors: count the number of common neighbors
    common_neigh = list(nx.common_neighbors(G, u, v))
    cn = len(common_neigh)

    # Jaccard Coefficient: common / union of neighbors
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    union_size = len(neighbors_u.union(neighbors_v))
    jc = cn / union_size if union_size > 0 else 0

    # Preferential Attachment: product of degrees
    pa = G.degree(u) * G.degree(v)

    # Adamic-Adar Index: sum of inverse log-degrees over common neighbors
    aa = 0
    for w in common_neigh:
        deg_w = G.degree(w)
        if deg_w > 1:
            aa += 1 / np.log(deg_w)
    return [cn, jc, pa, aa]

def prepare_training_set(G_train, pos_edges, neg_edges):
    """
    For each candidate edge (both positive and negative), compute feature vector.
    Returns X (features) and y (labels).
    """
    X, y = [], []
    for edge in pos_edges:
        X.append(compute_features(G_train, edge))
        y.append(1)
    for edge in neg_edges:
        X.append(compute_features(G_train, edge))
        y.append(0)
    return np.array(X), np.array(y)

# ---------------------------
# 5. Candidate Generation for Prediction
# ---------------------------

def generate_candidate_edges(G):
    """
    Generate candidate node pairs for link prediction.
    Here we use a heuristic: for each node, we look at pairs of its neighbors
    (i.e. if two nodes share a neighbor but are not already connected, they become candidates).
    This reduces the total number of pairs considered.
    """
    candidates = set()
    for node in G.nodes():
        neigh = list(G.neighbors(node))
        for u, v in combinations(neigh, 2):
            pair = tuple(sorted((u, v)))
            if not G.has_edge(pair[0], pair[1]):
                candidates.add(pair)
    return candidates

# ---------------------------
# 6. Main Script
# ---------------------------

def main():
    # File names (assumes files are in the same folder)
    nodes_file = 'nodes.csv'
    edges_file = 'edges.csv'
    
    print("Loading data...")
    nodes_df, edges_df = load_data(nodes_file, edges_file)
    
    # Choose a cutoff timestamp.
    # For example, we use the 80th percentile of all edge timestamps.
    cutoff_timestamp = edges_df['timeStamp'].quantile(0.8)
    print(f"Cutoff timestamp (80th percentile): {cutoff_timestamp}")
    
    print("Building training graph (using edges with timeStamp <= cutoff)...")
    G_train = build_training_graph(nodes_df, edges_df, cutoff_timestamp)
    print(f"Training graph has {G_train.number_of_nodes()} nodes and {G_train.number_of_edges()} edges.")
    
    print("Extracting positive (future) edges from test period...")
    pos_edges = get_future_positive_edges(G_train, edges_df, cutoff_timestamp)
    print(f"Found {len(pos_edges)} positive future edges.")
    
    # For training the classifier, sample as many negative examples as positives.
    print("Sampling negative edges from training graph...")
    neg_edges = sample_negative_edges(G_train, len(pos_edges))
    
    print("Computing features for training examples...")
    X_train, y_train = prepare_training_set(G_train, pos_edges, neg_edges)
    
    # ---------------------------
    # 7. Model Training
    # ---------------------------
    
    print("Training logistic regression classifier...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    train_probs = clf.predict_proba(X_train)[:, 1]
    auc = roc_auc_score(y_train, train_probs)
    print(f"Training AUC: {auc:.4f}")
    
    # ---------------------------
    # 8. Predicting Likely Future Links
    # ---------------------------
    
    print("Generating candidate node pairs (based on shared neighbors)...")
    candidate_edges = generate_candidate_edges(G_train)
    print(f"Number of candidate pairs: {len(candidate_edges)}")
    
    print("Computing features for candidate edges and predicting scores...")
    candidate_list = []
    candidate_features = []
    start_time = time.time()
    for edge in candidate_edges:
        candidate_list.append(edge)
        candidate_features.append(compute_features(G_train, edge))
    candidate_features = np.array(candidate_features)
    candidate_scores = clf.predict_proba(candidate_features)[:, 1]
    elapsed = time.time() - start_time
    print(f"Feature computation and scoring for candidates took {elapsed:.2f} seconds.")
    
    # Get the top-K predicted edges (highest likelihood of forming in the future)
    top_k = 50
    top_indices = np.argsort(candidate_scores)[-top_k:][::-1]
    
    print(f"\nTop {top_k} predicted future links:")
    for idx in top_indices:
        u, v = candidate_list[idx]
        score = candidate_scores[idx]
        print(f"Edge {u} - {v} with predicted score {score:.4f}")
        
if __name__ == '__main__':
    main()
