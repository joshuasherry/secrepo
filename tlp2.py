import pandas as pd
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score
from stellargraph import StellarGraph
from stellargraph.mapper import LinkGenerator, FullBatchLinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# ---------------------------
# 1. Data Loading
# ---------------------------

def load_data(nodes_file, edges_file):
    """
    Load nodes and edges data from CSV files with label-encoded features.
    """
    nodes_df = pd.read_csv(nodes_file, index_col="nodeId")
    edges_df = pd.read_csv(edges_file)
    return nodes_df, edges_df

# ---------------------------
# 2. Graph Creation with StellarGraph
# ---------------------------

def create_stellar_graph(nodes_df, edges_df):
    """
    Create a StellarGraph instance from nodes and edges.
    """
    return StellarGraph(nodes=nodes_df, edges=edges_df, node_type_default="node", edge_type_default="edge")

# ---------------------------
# 3. Splitting Data & Sampling
# ---------------------------

def get_training_test_edges(edges_df, cutoff_timestamp):
    """
    Split edges into training and test sets based on the timestamp.
    """
    train_edges = edges_df[edges_df['timestamp'] <= cutoff_timestamp]
    test_edges = edges_df[edges_df['timestamp'] > cutoff_timestamp]
    return train_edges, test_edges

# ---------------------------
# 4. Building and Training GraphSAGE Model
# ---------------------------

def train_graphsage(graph, train_edges, feature_size):
    """
    Train a GraphSAGE model for link prediction.
    """
    generator = LinkGenerator(graph, batch_size=128, num_samples=[10, 5])
    train_gen = generator.flow(train_edges[['sourceId', 'targetId']].values, targets=np.ones(len(train_edges)))

    # Define GraphSAGE model
    graphsage = GraphSAGE(
        layer_sizes=[32, 32], generator=generator, bias=True, dropout=0.5
    )

    x_inp, x_out = graphsage.in_out_tensors()
    prediction = link_classification(edge_embedding_method="ip", output_dim=1)(x_out)

    model = Model(inputs=x_inp, outputs=prediction)
    model.compile(optimizer=Adam(learning_rate=0.01), loss="binary_crossentropy", metrics=["AUC"])

    print("Training GraphSAGE model...")
    model.fit(train_gen, epochs=20, verbose=1)
    return model, generator

# ---------------------------
# 5. Feature Computation and Prediction
# ---------------------------

def predict_links(graph, model, generator, candidate_edges):
    """
    Predict likelihood of future links using GraphSAGE.
    """
    candidates_gen = generator.flow(candidate_edges)
    predictions = model.predict(candidates_gen)
    return predictions.flatten()

# ---------------------------
# 6. Candidate Edge Generation
# ---------------------------

def generate_candidate_edges(G):
    """
    Generate candidate node pairs for link prediction.
    For each node, consider pairs among its neighbors (if they are not already connected).
    """
    candidates = set()
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        for u, v in combinations(neighbors, 2):
            pair = tuple(sorted((u, v)))
            if not G.has_edge(pair[0], pair[1]):
                candidates.add(pair)
    return candidates

# ---------------------------
# 7. Main Script
# ---------------------------

def main():
    # File names (assumes files are in the same folder)
    nodes_file = "featureListencoded.csv"
    edges_file = "edges.csv"

    print("Loading data...")
    nodes_df, edges_df = load_data(nodes_file, edges_file)

    print("Creating graph...")
    G_nx = nx.from_pandas_edgelist(edges_df, source="sourceId", target="targetId")
    cutoff_timestamp = edges_df['timestamp'].quantile(0.8)
    train_edges, test_edges = get_training_test_edges(edges_df, cutoff_timestamp)
    graph = create_stellar_graph(nodes_df, edges_df)

    print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    model, generator = train_graphsage(graph, train_edges, feature_size=nodes_df.shape[1])

    print("Generating candidate node pairs...")
    candidate_edges = generate_candidate_edges(G_nx)
    candidate_array = np.array(list(candidate_edges))

    print("Predicting future links...")
    predictions = predict_links(graph, model, generator, candidate_array)

    # Get top-K predictions
    top_k = 50
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    print(f"\nTop {top_k} predicted future links:")
    for idx in top_indices:
        u, v = candidate_array[idx]
        score = predictions[idx]
        print(f"Edge {u} - {v} with predicted score {score:.4f}")

if __name__ == "__main__":
    main()
