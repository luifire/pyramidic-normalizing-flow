import numpy as np
import torch
import torch.nn as nn
from torch import distributions

from misc.misc import *
from misc.constants import *
# copied from https://github.com/PolinaKirichenko/flows_ood

class FlowLoss(nn.Module):
    """Get the NLL loss for a RealNVP model.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """

    def __init__(self, rv_size, k=256):
        super().__init__()
        self.k = k
        self.rv_size = rv_size
        self.prior = distributions.MultivariateNormal(torch.zeros(rv_size).to(DEVICE),
                                                 torch.eye(rv_size).to(DEVICE))

    def forward(self, z, summed_nl_det): #, sldj, y=None, mean=True):
        z = z.reshape((z.shape[0], -1))
        prior_ll = self.prior.log_prob(z)
        #if y is not None:
        #    prior_ll = self.prior.log_prob(z, y)
        #else:
        #    prior_ll = self.prior.log_prob(z)
        #warn("if this is actually correct, you gonna see if you check!")

        # -log(k) * rv_size um die Discretisierung weg zu rechnen
        corrected_prior_ll = prior_ll - np.log(self.k) * self.rv_size # np.prod(z.size()[1:])

        ll = corrected_prior_ll + summed_nl_det
        #nll = -ll.mean() if mean else -ll
        nll = -ll.mean()
        return nll


