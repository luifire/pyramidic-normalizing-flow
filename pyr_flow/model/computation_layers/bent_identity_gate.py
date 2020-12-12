import torch.nn as nn

from pyr_flow.constants import *
from pyr_flow.model.layer_module import LayerModule


class BentIdentity(LayerModule):
    benting = 1.2

    def __init__(self):
        super().__init__()

        print('Bent Identity')

        self.dummy = nn.Parameter(torch.zeros(1, device=DEVICE))

    def forward(self, x: torch.Tensor, lnorm_map):
        x_sq = x ** 2

        derivative = x / (BentIdentity.benting * (x_sq + 1).sqrt()) + 1
        lnorm_map += derivative.log()

        x = ((x_sq + 1).sqrt() - 1) / BentIdentity.benting + x

        return x, lnorm_map

    def get_parameter_count(self):
        return 0
