import torch.nn as nn
import torch
import torch.nn.functional as Function

from misc.misc import *
from misc.constants import *
# BIAS

import torch


class NormedLeakyRelu():

    def __init__(self, alpha=0.1):
        # pick alpha == 0.1 ** 0.5 | that way relu kind of stays the same
        # i.e. we do norm * alpha * neg_x , with norm = alpha ~ 0.31
        self.alpha = alpha
        self.relu = nn.LeakyReLU(alpha)

    def __call__(self, x):
        b,c,h,w = x.shape

        bigger_zero = x > 0
        pos_counts = bigger_zero.sum(3).sum(2)
        neg_count = h*w - pos_counts
        print("alpha to tensor")
        norm = self.alpha ** neg_count

        print("dat klappt so auch nicht")
        return x * norm






"""
LEAKY_RELU = nn.LeakyReLU(0.9)

class NormedLeakyRelu2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x : torch.Tensor):
        y = None

        # rather have more values bigger then, s.t. det == 1
        bigger_zero = x > 0
        bigger_zero.sum()
        x = LEAKY_RELU(x)

        return

    @staticmethod
    def backward(ctx, x):
        pass
"""