import torch

import dgl
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from dgl import AddSelfLoop

def load_data(dataset):
    transform = (AddSelfLoop())
    if dataset == 'cora':
        data = CoraGraphDataset(transform=transform)
        g = data[0]
        return g
    elif dataset=='citeseer':
        data = CiteseerGraphDataset(transform=transform)
        g = data[0]
        return g
    elif dataset == 'pubmed':
        data = PubmedGraphDataset(transform=transform)
        g = data[0]
        return g

# data = CiteseerGraphDataset()
# print(data[0])