import random
import networkx as nx
import numpy as np
import torch
from dgl.data import CoraGraphDataset
from sklearn.metrics import roc_auc_score

# Load dataset
dataset = CoraGraphDataset()
graph = dataset[0]
print(graph)

# Convert graph to undirected
g = nx.Graph(graph.to_networkx())
g.remove_edges_from(nx.selfloop_edges(g))

# Keep largest connected component
largest_connected_component = max(nx.connected_components(g), key=len)
g = g.subgraph(largest_connected_component).copy()

print("Nodes: ", g.number_of_nodes())
print("Edges: ", g.number_of_edges())
print("Connected: ", nx.is_connected(g))

# Train test
edges = list(g.edges())
random.shuffle(edges)

num_test = int(0.1 * len(edges))

training_graph = g.copy()
test_pos_edges = []

for (u, v) in edges:
    if len(test_pos_edges) == num_test:
        break

    training_graph.remove_edge(u, v)

    if nx.is_connected(training_graph):
        test_pos_edges.append((u, v))
    else:
        training_graph.add_edge(u, v)

print("Training edges: ", training_graph.number_of_edges())
print("Positive test edges: ", len(test_pos_edges))
print("Training graph connected: ", nx.is_connected(training_graph))

# Negative sampling
nodes = list(g.nodes())
test_neg_edges = set()

while len(test_neg_edges) < len(test_pos_edges):
    u, v = random.sample(nodes, 2)
    if g.has_edge(u, v):
        continue
    test_neg_edges.add((u, v))

test_neg_edges = list(test_neg_edges)

print("Negative test edges: ", len(test_neg_edges))

# Make to tensors
test_pos_u = torch.tensor([u for u, v in test_pos_edges])
test_pos_v = torch.tensor([v for u, v in test_pos_edges])

test_neg_u = torch.tensor([u for u, v in test_neg_edges])
test_neg_v = torch.tensor([v for u, v in test_neg_edges])

# Auc
def compute_auc(pos_scores, neg_scores):
    scores = torch.cat([pos_scores, neg_scores]).numpy()
    labels = np.concatenate([
        np.ones(len(pos_scores)), np.zeros(len(neg_scores))
    ])
    return roc_auc_score(labels, scores)

# Final check
assert nx.is_connected(training_graph)
assert len(test_neg_edges) == len(test_pos_edges)