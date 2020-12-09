import numpy as np
import torch
import torch.nn as nn
from torch import distributions

from misc.misc import *
from misc.constants import *


class PyramidLoss(nn.Module):

    def __init__(self, rv_size, k=256, weight_decrease=2):
        super().__init__()
        self.k = k
        self.rv_size = rv_size
        self.weight_decrease = weight_decrease
        self.prior = []
        self.discretisation_offset = np.log(self.k) * self.rv_size

    """ we create the prior dynamically """
    def get_prior(self, i, size):
        if len(self.prior) <= i:
            self.prior.append(distributions.MultivariateNormal(torch.zeros(size).to(DEVICE),
                                         torch.eye(size).to(DEVICE)))

        return self.prior[i]

    def forward(self, z_steps, summed_logd_det):
        loss = torch.zeros(1, device=DEVICE)
        # most important loss first
        z_steps = reversed(z_steps)
        for i, z in enumerate(z_steps):
            z = z.reshape((z.shape[0], -1))
            prior = self.get_prior(i, z.shape[1])
            prior_ll = prior.log_prob(z)

            # -log(k) * rv_size um die Discretisierung weg zu rechnen
            corrected_prior_ll = prior_ll - self.discretisation_offset

            ll = corrected_prior_ll + summed_logd_det
            #ll = summed_logd_det
            nll = -ll.mean()

            # pyramid weighting
            weight = 1 / self.weight_decrease**i
            loss += nll * weight
            #if loss.isnan() or loss.isinf():
            #    warn("loss is " + loss.item())
            #    exit(1)

        return loss, corrected_prior_ll


