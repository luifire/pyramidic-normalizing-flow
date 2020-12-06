import torch.nn as nn
import torch
import torch.nn.functional as F

from model.shift_channel_layer import ChannelShifter

from misc.misc import *
from misc.constants import *
from misc.helper import channel_to_last_dim, channel_normal_position
# BIAS


class DepthConv(nn.Module):

    """  """
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel_size_sq = kernel_size**2

        total_kernel_size = self.kernel_size_sq*CHANNEL_COUNT
        # 1.5 kind of suggested by IFA-VAE
        weights = torch.normal(mean=1.5, std=0.5, size=[total_kernel_size, total_kernel_size], device=DEVICE)
        #printt("init weights", weights)
        self.weights = nn.Parameter(weights, requires_grad=True)  # nn.Parameter is a Tensor that's a module parameter.

        bias = torch.normal(mean=1.5, std=0.5, size=[total_kernel_size], device=DEVICE)
        self.bias = nn.Parameter(bias, requires_grad=True)  # nn.Parameter is a Tensor that's a module parameter.

    def _prepare_weight_matrix_and_norm(self):
        # make triangular matrix
        # printt("forward 1. weights", self.weights)
        #printt("device", self.weights.device)

        weights = torch.triu(self.weights)

        diag = torch.diagonal(weights)

        det = torch.prod(diag)
        return weights, det

    def forward(self, x : torch.Tensor):

        conv_matrix, det = self._prepare_weight_matrix_and_norm()

        _, _, height, width = x.shape

        x = channel_to_last_dim(x)

        warn("achtung beim splitten! [0] bekommt die wenigste info und [8] die meiste!!!")
        x = x.matmul(conv_matrix)

        x = x + self.bias

        x = channel_normal_position(x)

        # TODO speedup
        amount_of_convolutions = (height // 3) * (width // 3)
        # |det| == |det(Kernel)^amount_of_convolutions|
        # Note that the power part ccan be done like this
        # log(|a^b|) = log(|a|^b) = b log(|a|)
        det_logd = det.abs().log()
        inverted_det_for_all = -amount_of_convolutions * det_logd

        return x, inverted_det_for_all


class DepthConvBundle(nn.Module):

    def __init__(self, kernel_size=KERNEL_SIZE, bundle_count=KERNEL_SIZE_SQ):
        super().__init__()
        self.bundle = nn.ModuleList()
        #self.channel_shifter = ChannelShifter(kernel_size ** 2 * CHANNEL_COUNT)

        for _ in range(bundle_count):
            self.bundle.append(DepthConv(kernel_size))

    def __call__(self, x):
        logd_norm_sums = torch.zeros(1, device=DEVICE)
        for i, layer in enumerate(self.bundle):
            x, log_norm = layer(x)
            logd_norm_sums = logd_norm_sums.add(log_norm)

            #x = self.channel_shifter(x)
        #logd_norm_sum = self.logd_norms.sum()
        #self.logd_norms
        return x, logd_norm_sums


