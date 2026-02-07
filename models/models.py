import torch.nn as nn
import torch.nn.functional as F
from models.layers import BBGraphConvolution
from models.ArchitectureSampler import SampleNetworkArchitecture

class BBGCN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass,  dropout=0.1, device=None):
        super(BBGCN, self).__init__()

        self.args = args
        self.loss = 0.0
        self.device = device
        self.dropout = dropout
        self.architecture_sampler = SampleNetworkArchitecture(args, self.device)

        self.layers = nn.ModuleList([BBGraphConvolution(nfeat, nhid).to(self.device)])

        # we simply add K layers at initialization
        # Note that we can also dynamically add new layers based on the inferred depth
        for i in range(self.args.truncation-1):
            self.layers.append(BBGraphConvolution(nhid, nhid, residual=True).to(self.device))

        self.out_layer = nn.Linear(nhid, nclass).to(self.device)

    def forward(self, x, adj, num_samples, feats_out=False):
        x = F.dropout(x, self.dropout, training=self.training)
        Z, n_layers, pi = self.architecture_sampler(num_samples)
        x = x.unsqueeze(1).expand(-1,num_samples,-1)

        for i in range(n_layers):
            x = self.layers[i](x,adj,mask=Z[:, :, i])
        feats = x
        x = self.out_layer(x)
        x = F.log_softmax(x, dim=-1)
        if feats_out:
            return feats, x, n_layers
        else:
            return x, n_layers

    def get_arch_info(self):
        mask_matrix, n_layers, percentile_25, percentile_75, pi = self.architecture_sampler(num_samples=20, get_pi=True)
        return n_layers, pi, percentile_25, percentile_75

    def representations(self, x, adj, num_samples):
        x = F.dropout(x, self.dropout, training=self.training)
        Z, n_layers, pi = self.architecture_sampler(num_samples)
        x = x.unsqueeze(1).expand(-1, num_samples, -1)

        for i in range(n_layers):
            x = self.layers[i](x, adj, mask=Z[:, :, i])

        feats = x.mean(dim=1)
        x = self.out_layer(x)
        x = F.log_softmax(x, dim=-1)

        return feats, x


