import torch.nn as nn
from torch import distributions

from pyr_flow.model.computation_layers.s_log_gate import SLogGate
from pyr_flow.model.computation_layers.tanh_gate import TanhGate

from pyr_flow.constants import *
from pyr_flow.misc.misc import *

class PyramidLoss(nn.Module):

    def __init__(self, rv_size=-1, k=256):
        super().__init__()
        self.log_k = np.log(k)
        #self.rv_size = rv_size
        self.prior = dict()
        #self.discretisation_offset = np.log(self.k) * self.rv_size

    """ we create the prior dynamically """
    def _get_prior(self, size):
        if size not in self.prior:
            self.prior[size] = distributions.MultivariateNormal(torch.zeros(size).to(DEVICE),
                                                         torch.eye(size).to(DEVICE))
        return self.prior[size]

    def forward(self, pyramid_steps, pyramid_steps_lnorm, use_norm=True):
        loss = torch.zeros(1, device=DEVICE)
        #unweighted_prior = torch.zeros(1, device=DEVICE)
        #unweighted_lnorm = torch.zeros(1, device=DEVICE)
        top_nll = None
        batch_size = pyramid_steps[0].shape[0]
        unweighted_loss = torch.ones(batch_size, device=DEVICE)
        unweighted_lnorm = torch.zeros(batch_size, device=DEVICE)
        # most important loss first
        pyramid_steps = reversed(pyramid_steps)
        pyramid_steps_lnorm = list(reversed(pyramid_steps_lnorm))
        for i, step in enumerate(pyramid_steps):
            lnorm_step = pyramid_steps_lnorm[i]

            step = step.reshape((batch_size, -1))
            lnorm_step = lnorm_step.reshape((batch_size, -1))
            ll = self._compute_log_prob(step, lnorm_step, use_norm)

            if i == 0: top_nll = -ll
            # pyramid weighting
            weight = 1 / PYRAMID_STEP_WEIGHTING ** i
            loss += weight * ll.mean() / step.shape[1]

            unweighted_loss += ll # isotropic Gauss -> we can simply multiply (or add in log Space)
            unweighted_lnorm += lnorm_step.sum(1)

        return -loss, -unweighted_loss, -unweighted_lnorm, top_nll

    def _compute_log_prob(self, x, lnorm_map, use_norm=True):
        batch_size, n_rv = x.shape

        prior = self._get_prior(n_rv)
        prior_ll = prior.log_prob(x)

        discretisation_offset = self.log_k * n_rv

        lnorm = lnorm_map.sum(1)

        corrected_prior_ll = prior_ll - discretisation_offset

        if use_norm:
            ll = corrected_prior_ll + lnorm
        else:
            ll = corrected_prior_ll
        return ll

    @staticmethod
    def print_loss_size(pyramid_steps):
        print_separator()
        print('Pyramid Loss Shape')
        print('0 is most important\n')
        print('Note: that we do: weight * log(p(Size))/Size = weight (log(p(1)) +...+log(p(size))/size')
        pyramid_steps = reversed(pyramid_steps)
        size_sum = 0
        for i, step in enumerate(pyramid_steps):
            step = step[0] # ignore batch
            step = step.reshape(-1)
            weight = 1 / (PYRAMID_STEP_WEIGHTING ** i)
            size = step.shape[0]
            print(f'step {i} size: {size} weight: {weight:.4f}')
            size_sum += size

        print(f'Sum(size) = {size_sum} total image dimension {TOTAL_IMAGE_DIMENSION}')

        print(f'Amount of Log Gates: {SLogGate.log_gate_count}')
        print(f'Amount of Tanh Gates: {TanhGate.tanh_gate_count}')
        print(f'Amount of Bill Gates: 1')

        print_separator()
