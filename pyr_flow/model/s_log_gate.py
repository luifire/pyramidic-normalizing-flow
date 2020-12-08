import numpy as np
import torch
import torch.nn as nn
from torch import distributions

from misc.misc import *
from misc.constants import *
from model.layer_module import LayerModule

class SLogGate(LayerModule):

    def __init__(self):
        super().__init__()

        # see: "Initialization of the parameter" in "Invertible Convolutional Flow"
        #self.alpha = nn.Parameter(torch.tensor(0.01, device=DEVICE))
        self.alpha = nn.Parameter(torch.tensor(-4.605, device=DEVICE)) # exp(alpha) ~= 0.01

    def forward(self, x: torch.Tensor):
        # alpha should be positive
        #positive_alpha = self.alpha.exp()
        warn("if nan should occure, it could be because of this alpha")
        positive_alpha = self.alpha
        #|x|/a * (exp(a |x|) -1)
        bracket = (positive_alpha * x.abs()).exp() - 1
        y = x.sign() / positive_alpha * bracket

        normalization = positive_alpha * x.abs()
        total_norm = normalization.sum(1).sum(1).sum(1)

        return y, total_norm

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
        print(f"Alpha: {self.alpha.exp():.3e}")

    def get_parameter_count(self):
        return 1
