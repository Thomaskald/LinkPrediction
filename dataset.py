import random

import dgl
from dgl.data import CoraGraphDataset
from sklearn.metrics import roc_auc_score

def load_cora_dataset():
    dataset = CoraGraphDataset()
    graph = dataset[0]

    # Convert graph to undirected
    graph = dgl.to_bidirected(graph)
    # Remove self-loop
    graph = dgl.remove_self_loop(graph)

    return graph

