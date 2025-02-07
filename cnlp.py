import pandas as pd
import networkx as nx
import numpy as np
from cnlp import CNLP
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Load edges (sourceId, targetId, edgeName, timeStamp)
edges_df = pd.read_csv('edges.csv')

# Create a NetworkX graph
G = nx.Graph()
G.add_edges_from(zip(edges_df['sourceId'], edges_df['targetId']))

# Initialize CNLP
cnlp = CNLP(G)

# Get positive edges (existing edges)
positive_edges = list(G.edges())

# Generate negative edges (randomly selected non-existing edges)
nodes = list(G.nodes())
negative_edges = []
while len(negative_edges) < len(positive_edges):  
    src, tgt = np.random.choice(nodes, 2, replace=False)  
    if not G.has_edge(src, tgt):  
        negative_edges.append((src, tgt))

# Convert edges to DataFrame
positive_df = pd.DataFrame(positive_edges, columns=['sourceId', 'targetId'])
negative_df = pd.DataFrame(negative_edges, columns=['sourceId', 'targetId'])

# Label positive (1) and negative (0) edges
positive_df['label'] = 1
negative_df['label'] = 0

# Combine positive and negative edges
all_edges = pd.concat([positive_df, negative_df])

# Compute CNLP-based features for link prediction
methods = ['common_neighbors', 'adamic_adar', 'jaccard', 'resource_allocation', 'preferential_attachment']
for method in methods:
    all_edges[method] = all_edges.apply(lambda row: cnlp.predict(row['sourceId'], row['targetId'], method), axis=1)

# Train-test split
X = all_edges[methods]
y = all_edges['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
