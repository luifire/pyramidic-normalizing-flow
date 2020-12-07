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
        self.alpha = nn.Parameter(torch.tensor(0.01, device=DEVICE))

    def forward(self, x: torch.Tensor):
        # why this activation:
        # our transformation tries to transform towards a Gauss
        # but with relu we are constantly cutting of negative values...
        # also I am afraid, that i.e. with a sigmoid, the precision of our memory gets
        # to bad s.t. we values that have small likelihood can not be distinguished
        # from one with higher precision.
        # this transformation comes from: Invertible Convolutional Flow by Karami

        sign = x.sign() / self.alpha
        log_part = torch.log(self.alpha * x.abs() + 1)

        y = sign * log_part

        #warn("check, if this actually needs to be -1")
        # TODO I think this is correct, checked
        normalization = -(self.alpha * y.abs())
        normalization = normalization.sum(1).sum(1).sum(1)
        return y, normalization

    def print_parameter(self):
        print(f"Alpha: {self.alpha:.3e}")

    def get_parameter_count(self):
        return 1