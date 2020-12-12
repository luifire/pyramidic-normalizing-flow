import torch.nn as nn

from pyr_flow.constants import *
from pyr_flow.model.layer_module import LayerModule


"""
x = tanh(x)
lnorm = log( (1 + tanh(x)) (1 - tanh(x)) ) 
"""
class TanhGate(LayerModule):
    tanh_gate_count = 0

    def __init__(self):
        super().__init__()

        # see: "Initialization of the parameter" in "Invertible Convolutional Flow"
        TanhGate.tanh_gate_count += 1
        print('tanhGate')

        self.dummy = nn.Parameter(torch.zeros(1, device=DEVICE))

    def forward(self, x: torch.Tensor, lnorm_map):
        tanh = (x).tanh()

        # (1 + tanh) * (1 - tanh)
        logd_derivative = (1 + tanh).log() + (1 - tanh).log()
        # will never be negative
        lnorm_map += logd_derivative
        x = tanh
        return x, lnorm_map

    def get_parameter_count(self):
        return 0
