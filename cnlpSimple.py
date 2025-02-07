import pandas as pd
import networkx as nx
from cnlp.similarity_methods.local_similarity import common_neighbors
from cnlp.utils import nodes_to_indexes, get_top_predicted_link

# Load edges data
edges_df = pd.read_csv("edges.csv")  # Contains sourceId, targetId, edgeName, timeStamp
nodes_df = pd.read_csv("nodes.csv")  # Contains nodeId, nodeType, createdTimestamp, updatedTimestamp

# Create a directed graph (or undirected if appropriate)
G = nx.Graph()
G.add_edges_from(zip(edges_df["sourceId"], edges_df["targetId"]))

# Convert node IDs to indexes
name_index_map = list(nodes_to_indexes(G).items())

# Compute link prediction scores using common neighbors
predicted_adj_matrix_common_neighbors = common_neighbors(G)

# Get the top predicted links (5% new links)
new_links = get_top_predicted_link(
    predicted_adj_matrix_common_neighbors, 
    G.number_of_nodes(), 
    pct_new_link=5, 
    name_index_map=name_index_map, 
    verbose=True
)

# Print the top predicted new links
print(new_links)
