import pandas as pd
from igraph import Graph
import numpy as np

# Load nodes and edges
nodes_df = pd.read_csv('nodes.csv')
edges_df = pd.read_csv('edges.csv')

# Optionally, convert timestamps (if needed)
nodes_df['createdTimestamp'] = pd.to_datetime(nodes_df['createdTimestamp'], unit='s')
nodes_df['updatedTimestamp'] = pd.to_datetime(nodes_df['updatedTimestamp'], unit='s')
edges_df['timeStamp'] = pd.to_datetime(edges_df['timeStamp'], unit='s')

# Create a mapping from nodeId to index for igraph
node_id_to_idx = {node_id: idx for idx, node_id in enumerate(nodes_df['nodeId'])}

# Build edge list (assuming directed edges; adjust if undirected)
edge_list = [(node_id_to_idx[row['sourceId']], node_id_to_idx[row['targetId']]) for _, row in edges_df.iterrows()]

# Create igraph graph
g = Graph(directed=True)
g.add_vertices(len(nodes_df))
g.add_edges(edge_list)

# Store additional node/edge attributes as needed
g.vs['nodeType'] = nodes_df['nodeType'].tolist()
g.vs['createdTimestamp'] = nodes_df['createdTimestamp'].tolist()
g.vs['updatedTimestamp'] = nodes_df['updatedTimestamp'].tolist()
g.es['edgeName'] = edges_df['edgeName'].tolist()
g.es['timeStamp'] = edges_df['timeStamp'].tolist()


###############

from node2vec import Node2Vec

# Prepare data for node2vec; extract edge list as list of tuples (source, target)
edges = [(str(row['sourceId']), str(row['targetId'])) for _, row in edges_df.iterrows()]

# Create node2vec model instance (adjust parameters for large graphs)
node2vec = Node2Vec(edges, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1)

# Get embeddings dictionary mapping node id to vector
embeddings = {node: model.wv[node] for node in model.wv.index_to_key}



#####################

def generate_candidates(g, candidate_ratio=0.1):
    # For demonstration, consider neighbors-of-neighbors as candidates
    candidates = set()
    for node in range(len(g.vs)):
        neighbors = set(g.neighbors(node, mode="ALL"))
        # Get neighbors-of-neighbors
        for neigh in neighbors:
            candidates.update(g.neighbors(neigh, mode="ALL"))
        # Remove self and direct neighbors
        candidates.discard(node)
        candidates -= set(neighbors)
        # Optionally, sample a subset
        sampled_candidates = np.random.choice(list(candidates), size=int(len(candidates) * candidate_ratio), replace=False)
        for cand in sampled_candidates:
            yield (node, cand)



from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

def compute_features(u, v, g, embeddings, lambda_decay=0.001, current_time=None):
    # Topological features
    neighbors_u = set(g.neighbors(u, mode="ALL"))
    neighbors_v = set(g.neighbors(v, mode="ALL"))
    common_neighbors = neighbors_u.intersection(neighbors_v)
    
    cn = len(common_neighbors)
    union_size = len(neighbors_u.union(neighbors_v))
    jaccard = cn / union_size if union_size > 0 else 0
    
    # Adamic-Adar score
    aa = 0
    for w in common_neighbors:
        deg_w = len(g.neighbors(w, mode="ALL"))
        if deg_w > 1:
            aa += 1 / np.log(deg_w)
    
    # Preferential attachment
    pa = len(neighbors_u) * len(neighbors_v)
    
    # Embedding similarity
    emb_sim = cosine_similarity([embeddings[str(u)]], [embeddings[str(v)]])[0][0]
    
    # Temporal feature (if available): e.g., most recent edge time between u and any common neighbor
    # This is a placeholder; actual implementation may require indexing edges by node.
    recency = 0
    if current_time is not None and common_neighbors:
        # Assume we can retrieve the max timestamp among edges involving u and v's common neighbors
        last_interaction = max([max_edge_timestamp(u, w, g) for w in common_neighbors])
        recency = (current_time - last_interaction).total_seconds()
    
    return [cn, jaccard, aa, pa, emb_sim, recency]

# Placeholder function: Implement efficient retrieval of the last interaction timestamp between two nodes.
def max_edge_timestamp(u, w, g):
    # This requires iterating over the edges incident on u or using an index.
    # For demonstration, we return a fixed value.
    return pd.Timestamp.now()

# Build dataset
X, y = [], []

# Assume we have a list of candidate pairs and a function is_positive(u, v) that checks if the link appears in the future window.
for u, v in generate_candidates(g):
    features = compute_features(u, v, g, embeddings, current_time=pd.Timestamp.now())
    X.append(features)
    label = 1 if is_positive(u, v) else 0
    y.append(label)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
clf.fit(X_train, y_train)

# Evaluate
pred_probs = clf.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, pred_probs)
print(f"AUC: {auc:.4f}")




