import torch.nn as nn

from pyr_flow.constants import *
from pyr_flow.model.layer_module import LayerModule

class SLogGate(LayerModule):
    log_gate_count = 0

    def __init__(self):
        super().__init__()

        # see: "Initialization of the parameter" in "Invertible Convolutional Flow"
        #self.alpha = nn.Parameter(torch.tensor(0.01, device=DEVICE))
        self.alpha = nn.Parameter(torch.tensor(S_LOG_GATE_ALPHA_INIT, device=DEVICE))
        SLogGate.log_gate_count += 1
        print('S-Log Gate')

    def forward(self, x: torch.Tensor, lnorm_map):
        # alpha should be positive
        #positive_alpha = self.alpha.exp()
        #warn("if nan should occure, it could be because of this alpha")
        positive_alpha = self.alpha
        x_abs = x.abs()
        #sign(x)/a * (exp(a |x|) -1)
        bracket = (positive_alpha * x_abs).exp() - 1
        x = x.sign() / positive_alpha * bracket

        lnorm_map += positive_alpha * x_abs
        return x, lnorm_map

        #normalization = positive_alpha * x_abs
        #total_norm = normalization.sum(1).sum(1).sum(1)

        #return y, total_norm

    # I think this was the inverse ^^"
    def forward_(self, x: torch.Tensor):
        # why this activation:
        # our transformation tries to transform towards a Gauss
        # but with relu we are constantly cutting of negative values...
        # also I am afraid, that i.e. with a sigmoid, the precision of our memory gets
        # to bad s.t. we values that have small likelihood can not be distinguished
        # from one with higher precision.
        # this transformation comes from: Invertible Convolutional Flow by Karami

        parameterized_alpha = self.alpha.exp() # it is only positive now
        sign = x.sign() / parameterized_alpha
        log_part = torch.log(parameterized_alpha * x.abs() + 1)

        y = sign * log_part

        #warn("check, if this actually needs to be -1")
        # TODO I think this is correct, checked
        normalization = -(parameterized_alpha * y.abs())
        normalization = normalization.sum(1).sum(1).sum(1)
        return y, normalization

    def print_parameter(self):
        print(f"Alpha: {self.alpha:.3e}")

    def get_parameter_count(self):
        return 1
