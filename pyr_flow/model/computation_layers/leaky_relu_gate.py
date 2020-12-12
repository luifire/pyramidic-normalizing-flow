import torch.nn as nn

from pyr_flow.constants import *
from pyr_flow.model.layer_module import LayerModule


class LeakyRelu(LayerModule):
    leaky_relu_gate_count = 0
    slope = 0.01
    logd_slope = np.log(slope)

    def __init__(self):
        super().__init__()

        # see: "Initialization of the parameter" in "Invertible Convolutional Flow"
        LeakyRelu.leaky_relu_gate_count += 1
        print('leaky_reluGate')

        self.dummy = nn.Parameter(torch.zeros(1, device=DEVICE))

        self.l_relu = torch.nn.LeakyReLU(negative_slope = LeakyRelu.slope)

    def forward(self, x: torch.Tensor, lnorm_map):
        lnorm_map[(x < LeakyRelu.slope)] += LeakyRelu.logd_slope
        x = self.l_relu(x)
        return x, lnorm_map

    def get_parameter_count(self):
        return 0
