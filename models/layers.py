import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class BBGraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True, residual=False):
        super(BBGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.residual = residual
        # if residual:
        #     self.skip_wt = Parameter(torch.ones(in_features))
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, mask=None):
        res = input
        B, S, D = input.shape
        base_features = torch.einsum('nsf, fg -> nsg', input, self.weight)  # [n * s * f] x [f * g] => [n * s * g]
        base_features = base_features.view(B, -1)
        # print(base_features.shape)
        # print(adj.shape)
        base_features = torch.spmm(adj, base_features)
        output = base_features.view(B, S, self.out_features)
        # exit(0)

        if self.bias is not None:
            output = output + self.bias

        if mask is not None:
            output = F.relu(output)
            output = output * mask.unsqueeze(0)
        if self.residual:
            output = output + res   #*self.skip_wt
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'