import dgl
import torch
import numpy as np
import random
import networkx as nx
from dgl.data import CoraGraphDataset
from sklearn.metrics import roc_auc_score

# -----------------------------
# 1. Load Cora dataset
# -----------------------------
dataset = CoraGraphDataset()
g = dataset[0]
print(g)

# -----------------------------
# 2. Convert to NetworkX and clean
# -----------------------------
# Convert to undirected simple graph
G = nx.Graph(g.to_networkx())
G.remove_edges_from(nx.selfloop_edges(G))

# Take largest connected component
largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()

# Safety check
assert nx.is_connected(G), "Graph is not connected after taking largest CC"
print(f"Number of nodes in largest CC: {G.number_of_nodes()}")
print(f"Number of edges in largest CC: {G.number_of_edges()}")

# -----------------------------
# 3. Train/test split using spanning tree (ensures connectedness)
# -----------------------------
# Create spanning tree
spanning_tree = nx.minimum_spanning_tree(G)
tree_edges = set(tuple(sorted(e)) for e in spanning_tree.edges())

# Candidate edges for test set = edges not in spanning tree
candidate_edges = [tuple(sorted(e)) for e in G.edges() if tuple(sorted(e)) not in tree_edges]
num_test = int(0.1 * len(G.edges()))
test_edges = random.sample(candidate_edges, num_test)

# Remove test edges from training graph
train_graph = G.copy()
for u, v in test_edges:
    train_graph.remove_edge(u, v)

# Verify connectedness
assert nx.is_connected(train_graph), "Training graph is disconnected!"
print(f"Training graph edges: {train_graph.number_of_edges()}")
print(f"Positive test edges: {len(test_edges)}")

# -----------------------------
# 4. Negative sampling for test set
# -----------------------------
nodes = list(G.nodes())
test_neg_edges = set()

while len(test_neg_edges) < len(test_edges):
    u, v = random.sample(nodes, 2)
    if G.has_edge(u, v) or (u, v) in test_neg_edges or (v, u) in test_neg_edges:
        continue
    test_neg_edges.add((u, v))

test_neg_edges = list(test_neg_edges)
print(f"Negative test edges: {len(test_neg_edges)}")

# -----------------------------
# 5. Convert edges to tensors for PyTorch/DGL
# -----------------------------
test_pos_u = torch.tensor([u for u, v in test_edges])
test_pos_v = torch.tensor([v for u, v in test_edges])
test_neg_u = torch.tensor([u for u, v in test_neg_edges])
test_neg_v = torch.tensor([v for u, v in test_neg_edges])

# -----------------------------
# 6. AUC utility function
# -----------------------------
def compute_auc(pos_scores, neg_scores):
    scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
    labels = np.concatenate([
        np.ones(len(pos_scores)),
        np.zeros(len(neg_scores))
    ])
    return roc_auc_score(labels, scores)

# -----------------------------
# 7. Final safety checks
# -----------------------------
assert nx.is_connected(train_graph), "Training graph disconnected after all removals!"
assert not train_graph.is_multigraph(), "Training graph has multi-edges!"
for u, v in train_graph.edges():
    assert u != v, "Self-loop detected in training graph!"
assert len(test_edges) == len(test_neg_edges), "Mismatch positive/negative test edges!"

print("Dataset preparation successful.")
