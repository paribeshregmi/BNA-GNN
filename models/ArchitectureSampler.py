import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Beta, RelaxedBernoulli, Bernoulli
from torch.distributions.kl import kl_divergence

########## adopted from https://github.com/kckishan/Depth_and_Dropout/
class SampleNetworkArchitecture(nn.Module):
    """
    Samples an architecture from Beta-Bernoulli prior
    """

    def __init__(self, args, device):
        super(SampleNetworkArchitecture, self).__init__()
        self.args = args
        self.device = device
        self.num_neurons = args.hidden

        # Temperature for Concrete Bernoulli
        self.temperature = torch.tensor(args.temp)
        # Number of samples from Beta-Bernoulli prior to estimate expectations
        self.num_samples = args.num_samples

        # Hyper-parameters for Prior probabilities
        self.a_prior = torch.tensor(args.a_prior).float().to(self.device)
        self.b_prior = torch.tensor(args.b_prior).float().to(self.device)

        # Define a prior beta distribution
        self.beta_prior = Beta(self.a_prior, self.b_prior)

        # inverse softplus to avoid parameter underflow
        a_val = np.log(np.exp(args.a_prior) - 1)
        b_val = np.log(np.exp(args.b_prior) - 1)

        # Define variational parameters for posterior distribution
        self.a_variational = nn.Parameter(torch.Tensor(args.truncation).zero_() + a_val)
        self.b_variational = nn.Parameter(torch.Tensor(args.truncation).zero_() + b_val)

    def get_var_params(self):

        beta_a = F.softplus(self.a_variational) + 0.01
        beta_b = F.softplus(self.b_variational) + 0.01
        return beta_a, beta_b

    def get_kl(self):
        """
        Computes the KL divergence between posterior and prior
        """
        beta_a, beta_b = self.get_var_params()
        beta_posterior = Beta(beta_a, beta_b)
        kl_beta = kl_divergence(beta_posterior, self.beta_prior)
        return kl_beta.sum()

    def forward(self, num_samples=5, get_pi=False):

        # Define variational beta distribution
        beta_a, beta_b = self.get_var_params()
        beta_posterior = Beta(beta_a, beta_b)

        # sample from variational beta distribution
        v = beta_posterior.rsample((num_samples, )).view(num_samples, self.args.truncation)

        # Convert v -> pi i.e. activation level of layer
        pi = torch.cumsum(v.log(), dim=1).exp()
        pi_d = pi
        keep_probs = pi.detach().mean(0)
        pi = torch.mean(pi, dim=0).expand(num_samples, -1)
        pi = pi.unsqueeze(1).expand(-1, self.num_neurons, -1)
        if self.training:
            # sample active neurons given the activation probability of that layer
            bernoulli_dist = RelaxedBernoulli(temperature=self.temperature, probs=pi)
            Z = bernoulli_dist.rsample()
        else:
            # sample active neurons given the activation probability of that layer
            bernoulli_dist = Bernoulli(probs=pi)
            Z = bernoulli_dist.sample()

        # compute threshold
        threshold_Z = (Z > 0.01).sum(1)
        threshold_array = (threshold_Z > 0).sum(dim=1).cpu().numpy()
        threshold = max(threshold_array)

        # Use single hidden layer (input -> hidden -> output)
        if threshold == 0:
            threshold = torch.tensor(1)

        self.n_layers = threshold

        if get_pi:
            return Z, np.median(threshold_array), np.percentile(threshold_array, 25), \
                   np.percentile(threshold_array, 75), pi.mean(0).mean(0)

        # return Z, threshold, threshold_array
        return Z, threshold, pi_d.detach().mean(0)