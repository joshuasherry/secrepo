def sample_hard_negative_edges(num_samples, node_df, edges_df, positive_edges_set):
    negatives = set()
    node_types = node_df.set_index("nodeId")["nodeType"].to_dict()

    while len(negatives) < num_samples:
        # Pick a real edge and modify it slightly
        u, v = random.choice(list(positive_edges_set))

        # Get node types
        u_type, v_type = node_types[u], node_types[v]

        if u_type == 0 and v_type == 2:
            # Swap `v` (type 2) for another type 2 node
            possible_negatives = [n for n, t in node_types.items() if t == 2 and n != v]
            if possible_negatives:
                v_new = random.choice(possible_negatives)
                if (u, v_new) not in positive_edges_set and (v_new, u) not in positive_edges_set:
                    negatives.add((u, v_new))

        elif u_type == 2 and v_type == 1:
            # Swap `u` (type 2) for another type 2 node
            possible_negatives = [n for n, t in node_types.items() if t == 2 and n != u]
            if possible_negatives:
                u_new = random.choice(possible_negatives)
                if (u_new, v) not in positive_edges_set and (v, u_new) not in positive_edges_set:
                    negatives.add((u_new, v))

    return list(negatives)

import random

def sample_hard_negative_edges(num_samples, node_df, edges_df, positive_edges_set):
    negatives = set()
    node_types = node_df.set_index("nodeId")["nodeType"].to_dict()

    # Precompute possible negative candidates by node type
    node_type_dict = {t: [n for n, t_ in node_types.items() if t_ == t] for t in set(node_types.values())}

    while len(negatives) < num_samples:
        # Pick a random positive edge
        u, v = random.choice(list(positive_edges_set))

        # Get node types
        u_type, v_type = node_types[u], node_types[v]

        # Generate negative samples based on the types
        if u_type == 0 and v_type == 2:
            # Find a new type-2 node for `v`
            possible_negatives = node_type_dict[2]
            possible_negatives.remove(v)  # Remove the original node `v`
        elif u_type == 2 and v_type == 1:
            # Find a new type-2 node for `u`
            possible_negatives = node_type_dict[2]
            possible_negatives.remove(u)  # Remove the original node `u`
        else:
            continue

        # Pick a random negative node and check for the edge condition
        while possible_negatives:
            v_new = random.choice(possible_negatives)
            if (u, v_new) not in positive_edges_set and (v_new, u) not in positive_edges_set:
                negatives.add((u, v_new))
                break
            possible_negatives.remove(v_new)

    return list(negatives)





neg_edges_train = sample_hard_negative_edges(len(pos_edges_train), nodes_df, edges_df, positive_edges_set)
neg_edges_test = sample_hard_negative_edges(len(pos_edges_test), nodes_df, edges_df, positive_edges_set)
