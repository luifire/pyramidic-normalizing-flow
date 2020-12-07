import torch.nn as nn
import torch

from model.shift_channel_layer import ChannelShifter
from model.layer_module import LayerModule

from misc.misc import *
from misc.constants import *


class DepthConv(LayerModule):

    """  """
    def __init__(self, name, channel_count):
        super().__init__()
        self.channel_count = channel_count
        #self.kernel_size_sq = channel_count ** 2
        self.name = name

        #total_kernel_size = self.kernel_size_sq*CHANNEL_COUNT
        # 1.5 kind of suggested by IFA-VAE
        weights = torch.normal(mean=1.5, std=0.5, size=[channel_count, channel_count], device=DEVICE)
        #printt("init weights", weights)
        self.weights = nn.Parameter(weights, requires_grad=True)  # nn.Parameter is a Tensor that's a module parameter.

        #bias = torch.normal(mean=1.5, std=0.5, size=[total_kernel_size], device=DEVICE)
        #self.bias = nn.Parameter(bias, requires_grad=True)  # nn.Parameter is a Tensor that's a module parameter.

    def _prepare_weight_matrix_and_norm(self):
        # make triangular matrix
        # printt("forward 1. weights", self.weights)
        #printt("device", self.weights.device)

        weights = torch.triu(self.weights)

        diag = torch.diagonal(weights)

        logd_det = diag.abs().log().sum()

        return weights, logd_det

    def forward(self, x : torch.Tensor):
        _, height, width, _ = x.shape

        conv_matrix, logd_det = self._prepare_weight_matrix_and_norm()
        #x = channel_to_last_dim(x)
        #warn("achtung beim splitten! [0] bekommt die wenigste info und [8] die meiste!!!")
        x = x.matmul(conv_matrix)

        # add bias at last
        #x = x + self.bias
        #x = channel_normal_position(x)

        # TODO speedup
        amount_of_convolutions = (height // self.channel_count) * (width // self.channel_count)
        # |det| == |det(Kernel)^amount_of_convolutions|
        # Note that the power part ccan be done like this
        # log(|a^b|) = log(|a|^b) = b log(|a|)
        inverted_det_for_all = amount_of_convolutions * logd_det

        return x, inverted_det_for_all

    def print_parameter(self):
        w = self.weights
        dia = w.diag().abs()
        print(f"{self.name} diag min: {dia.min():.3e} max: {dia.max():.3e} prod: {dia.prod():.3e} "
              f"avg total {w.mean():.3e}")

    def get_parameter_count(self):
        dim = self.weights.shape[0]
        return (dim ** 2 + dim) / 2
