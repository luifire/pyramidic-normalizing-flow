import torch.nn as nn

from pyr_flow.model.layer_module import LayerModule

from pyr_flow.constants import *
from pyr_flow.misc.misc import *


class DepthConv2(LayerModule):
    """
    initial matrix
    1 b c d e <- trainable part
    0 1 0 0 0 <- not trainable part
    0 0 1 0 0
    ...
    """

    def __init__(self, name, total_pixel_depth, internal_pixel_depth):
        super().__init__()
        self.total_pixel_depth = total_pixel_depth
        # self.internal_pixel_depth = internal_pixel_depth
        self.name = name

        warn('wanna use XAVIER?')

        weights = torch.normal(mean=0, std=0.0001, size=[total_pixel_depth], device=DEVICE)
        weights[0] = 1.0  # identity part
        self.weights = nn.Parameter(weights, requires_grad=True)

        conv_matrix = torch.eye(total_pixel_depth, device=DEVICE)
        conv_matrix[0] = weights
        self.conv_matrix = conv_matrix

    def forward(self, x: torch.Tensor, lnorm_map):
        x_2 = x.clone()
        x_2[:,:,:,0] = x[:,:,:].matmul(self.weights)

        #x = x.matmul(self.conv_matrix)

        # as all other identity parts are 1, log(1) = 0, so we only need to look at weights[0]
        lnorm_map += self.weights[0].abs().log()

        return x_2, lnorm_map

    def print_parameter(self):
        w = self.weights
        print(f"{self.name} #all weights: identity {w[0]:.2e} avg total {w.mean():.2e} "
              f"total min: {w.min():.2e} total max: {w.max():.2e} first 10 : {w[:9]}")

    def get_parameter_count(self):
        return self.weights.shape[0]
