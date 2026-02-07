import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
import torch
from scipy import stats

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj = adj.coalesce()
    dim = adj.size()[0]
    rows = adj.indices()[0].cpu().numpy()
    cols = adj.indices()[1].cpu().numpy()
    values = adj.values().cpu().numpy()

    adj = sp.coo_matrix((values, (rows, cols)), shape=(dim, dim))

    adj_normalized = normalize_adj(adj)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_bbgcn(output, labels):
    output = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    preds = stats.mode(np.argmax(output, axis=2), axis=1, keepdims=True)[0].reshape(-1).astype(np.int32)
    correct = np.equal(labels, preds).astype(np.float32)
    correct = correct.sum()
    return correct