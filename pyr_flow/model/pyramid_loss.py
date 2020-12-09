import numpy as np
import torch
import torch.nn as nn
from torch import distributions

from misc.misc import *
from misc.constants import *


class PyramidLoss(nn.Module):

    def __init__(self, rv_size, k=256, weight_decrease=2):
        super().__init__()
        self.log_k = np.log(k)
        self.rv_size = rv_size
        self.weight_decrease = weight_decrease
        self.prior = dict()
        #self.discretisation_offset = np.log(self.k) * self.rv_size

    """ we create the prior dynamically """
    def _get_prior(self, size):
        if size not in self.prior:
            self.prior[size] = distributions.MultivariateNormal(torch.zeros(size).to(DEVICE),
                                                         torch.eye(size).to(DEVICE))
        return self.prior[size]

    def forward(self, pyramid_steps, pyramid_steps_lnorm):
        loss = torch.zeros(1, device=DEVICE)
        #unweighted_prior = torch.zeros(1, device=DEVICE)
        #unweighted_lnorm = torch.zeros(1, device=DEVICE)

        batch_size = pyramid_steps[0].shape[0]
        unweighted_loss = torch.ones(batch_size, device=DEVICE)
        unweighted_lnorm = torch.zeros(batch_size, device=DEVICE)
        # most important loss first
        pyramid_steps = reversed(pyramid_steps)
        pyramid_steps_lnorm = list(reversed(pyramid_steps_lnorm))
        for i, step in enumerate(pyramid_steps):
            lnorm_step = pyramid_steps_lnorm[i]
            ll = self._compute_log_prob(step, lnorm_step)

            # pyramid weighting
            weight = 1 / self.weight_decrease**i
            loss += weight * ll.mean()

            unweighted_loss *= ll
            unweighted_lnorm += lnorm_step.reshape((batch_size, -1)).sum(1)

        return -loss, -unweighted_loss.mean(), -unweighted_lnorm.mean()
        """
            step = step.reshape((batch_size, -1))
            n_rv = step.shape[1]
            prior = self._get_prior(n_rv)
            prior_ll = prior.log_prob(step)

            lnorm_map = pyramid_steps_lnorm[i]
            lnorm = lnorm_map.reshape((batch_size, -1)).sum(1)
            # -log(k) * rv_size um die Discretisierung weg zu rechnen
            discretisation_offset = self.log_k * n_rv

            corrected_prior_ll = prior_ll - discretisation_offset

            ll = corrected_prior_ll + lnorm
            #ll = corrected_prior_ll #+ lnorm
            #ll = summed_logd_det
            nll = -ll.mean()

            # pyramid weighting
            weight = 1 / self.weight_decrease**i
            loss += nll * weight

            unweighted_prior += -corrected_prior_ll
            unweighted_lnorm += -lnorm
            #if loss.isnan() or loss.isinf():
            #    warn("loss is " + loss.item())
            #    exit(1)

        return loss, unweighted_prior.mean(), unweighted_lnorm.mean()
        """

    def _compute_log_prob(self, x, lnorm_map):
        batch_size = x.shape[0]

        x = x.reshape((batch_size, -1))
        n_rv = x.shape[1]

        prior = self._get_prior(n_rv)
        prior_ll = prior.log_prob(x)

        discretisation_offset = self.log_k * n_rv

        lnorm = lnorm_map.reshape((batch_size, -1)).sum(1)

        corrected_prior_ll = prior_ll - discretisation_offset

        ll = corrected_prior_ll + lnorm
        return ll#.mean()
