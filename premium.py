import pandas as pd
import networkx as nx
import numpy as np
import random

# Load the data
nodes_df = pd.read_csv('nodes.csv')
edges_df = pd.read_csv('edges.csv')

# Define a time threshold (here using the 80th percentile)
time_threshold = edges_df['timeStamp'].quantile(0.8)

# Split edges: training (historical) and future (for link prediction)
train_edges = edges_df[edges_df['timeStamp'] <= time_threshold]
future_edges = edges_df[edges_df['timeStamp'] > time_threshold]

# Build a training graph from historical data
# (Here we assume an undirected graph – if your edges are directional, you can adapt accordingly)
G_train = nx.Graph()
# Add nodes (ensuring we include nodes that may not have any edge)
G_train.add_nodes_from(nodes_df['nodeId'].unique())
for _, row in train_edges.iterrows():
    G_train.add_edge(row['sourceId'], row['targetId'])



# Build a set of edges already present in the training graph (store as frozensets for order independence)
existing_edges = set(frozenset((u, v)) for u, v in G_train.edges())

# Collect positive candidate pairs: future edges that are new (i.e. not already in the training graph)
positive_pairs = set()
for _, row in future_edges.iterrows():
    pair = frozenset((row['sourceId'], row['targetId']))
    if pair not in existing_edges:
        positive_pairs.add(tuple(sorted(pair)))
positive_pairs = list(positive_pairs)

# For negative pairs, sample from pairs that are not connected in training or in the future positives.
node_list = list(G_train.nodes())
negative_pairs = set()
# For balance, sample as many negatives as positives (you can adjust the ratio)
while len(negative_pairs) < len(positive_pairs):
    u, v = random.sample(node_list, 2)
    pair = frozenset((u, v))
    if pair in existing_edges:
        continue
    if pair in (frozenset(p) for p in positive_pairs):
        continue
    negative_pairs.add(tuple(sorted((u, v))))
negative_pairs = list(negative_pairs)


# Create a mapping from nodeId to its attributes (if a node is missing attributes, default values are used)
node_attr = nodes_df.set_index('nodeId').to_dict('index')


def get_features(u, v, G, node_attr):
    # Graph-based features:
    # 1. Common neighbors
    common_neighbors = len(list(nx.common_neighbors(G, u, v)))
    
    # 2. Jaccard Coefficient
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    union_neighbors = neighbors_u.union(neighbors_v)
    jaccard = common_neighbors / len(union_neighbors) if union_neighbors else 0.0

    # 3. Adamic–Adar Index
    aa = 0.0
    for w in neighbors_u.intersection(neighbors_v):
        deg = G.degree(w)
        if deg > 1:
            aa += 1 / np.log(deg)

    # 4. Preferential Attachment
    pa = G.degree(u) * G.degree(v)
    
    # Node attribute features:
    attr_u = node_attr.get(u, {})
    attr_v = node_attr.get(v, {})
    # If not available, default to 0
    created_u = attr_u.get('createdTimestamp', 0)
    created_v = attr_v.get('createdTimestamp', 0)
    updated_u = attr_u.get('updatedTimestamp', 0)
    updated_v = attr_v.get('updatedTimestamp', 0)
    created_diff = abs(created_u - created_v)
    updated_diff = abs(updated_u - updated_v)
    # A binary feature: 1 if node types are identical, else 0.
    same_type = 1 if attr_u.get('nodeType') == attr_v.get('nodeType') else 0

    return [common_neighbors, jaccard, aa, pa, created_diff, updated_diff, same_type]


# Build feature matrix X and label vector y
X = []
y = []

# Label 1 for positive examples (future links)
for u, v in positive_pairs:
    X.append(get_features(u, v, G_train, node_attr))
    y.append(1)

# Label 0 for negative examples
for u, v in negative_pairs:
    X.append(get_features(u, v, G_train, node_attr))
    y.append(0)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# Split the candidate dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Validate the model
y_val_pred = clf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_val_pred)
print(f'Validation AUC: {auc:.4f}')


def predict_future_links(G, clf, node_attr, score_threshold=0.7):
    predictions = []
    nodes = list(G.nodes())
    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            u, v = nodes[i], nodes[j]
            if G.has_edge(u, v):
                continue  # skip already connected pairs
            features = get_features(u, v, G, node_attr)
            prob = clf.predict_proba([features])[0][1]
            if prob >= score_threshold:
                predictions.append((u, v, prob))
    # Rank the candidate pairs by predicted probability
    predictions.sort(key=lambda x: x[2], reverse=True)
    return predictions

# Predict and list the top candidate links
predicted_links = predict_future_links(G_train, clf, node_attr, score_threshold=0.7)
print("Predicted future links (u, v, probability):")
for u, v, prob in predicted_links:
    print(u, v, f"{prob:.4f}")
