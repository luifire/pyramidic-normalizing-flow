import torch.nn as nn
import torch
import torch.nn.functional as Function

from misc.misc import *
from misc.constants import *
# BIAS

import torch

ALPHA = 1

class SLogGate(torch.autograd.Function):


    @staticmethod
    def forward(ctx, x):
        y = None

        x_abs = x.abs()
        norm_matrix = x.sign() / ALPHA * ((ALPHA * x_abs).exp() - 1)

        normed = y / x.abs()

        return normed

    @staticmethod
    def backward(ctx, x):
        pass