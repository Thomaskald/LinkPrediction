import random
from typing import Tuple, List

import dgl
import torch
import networkx as nx
from dgl.data import CoraGraphDataset
from sklearn.metrics import roc_auc_score

def load_cora_dataset():
    dataset = CoraGraphDataset()
    graph = dataset[0]

    # Convert graph to undirected
    graph = dgl.to_bidirected(graph)
    graph = dgl.to_simple_graph(graph)
    # Remove self-loop
    graph = dgl.remove_self_loop(graph)

    return graph

def train_test_split_edges(graph: dgl.DGLGraph, test_ratio: float = 0.1, seed: int = 42) -> Tuple[dgl.DGLGraph, List[Tuple[int, int]]]:
    random.seed(seed)

    nx_graph = graph.to_networkx().to_undirected()

    edges = list(nx_graph.edges())
    num_test = int(len(edges) * test_ratio)

    random.shuffle(edges)

    test_pos_edges = []
    removed = 0

    for u, v in edges:
        if removed >= num_test:
            break

        nx_graph.remove_edge(u, v)

        if nx.is_connected(nx_graph):
            test_pos_edges.append((u, v))
            removed += 1
        else:
            nx_graph.add_edge(u, v)

    train_graph = dgl.from_networkx(nx_graph)
    train_graph = dgl.to_simple_graph(train_graph)
    train_graph = dgl.remove_self_loop(train_graph)

    return train_graph, test_pos_edges

def negative_sampling(full_graph: dgl.DGLGraph, positive_edges: List[Tuple[int, int]], seed: int = 42) -> List[Tuple[int, int]]:
    random.seed(seed)

    num_nodes = full_graph.number_of_nodes()
    existing_edges = set((u.item(), v.item()) for u, v in zip(*full_graph.edges()))

    negative_edges = []

    while len(negative_edges) < len(positive_edges):
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)

        if u == v:
            continue

        if (u, v) in existing_edges or (v, u) in existing_edges:
            continue

        negative_edges.append((u, v))

    return negative_edges

def compute_auc(y_true: List[int], y_scores: List[float]) -> float:
    return roc_auc_score(y_true, y_scores)

def load_data(test_ratio: float = 0.1, seed: int = 42):
    full_graph = load_cora_dataset()

    train_graph, test_pos_edges = train_test_split_edges(full_graph, test_ratio, seed)

    test_neg_edges = negative_sampling(full_graph, test_pos_edges, seed)

    return train_graph, test_pos_edges, test_neg_edges