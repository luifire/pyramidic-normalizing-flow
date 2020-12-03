import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model.nf_conv_layer import NFConvLayer
from misc.constants import *

class PyramidFlow(nn.Module):

    def __init__(self):
        super().__init__()
        self.start = NFConvLayer(KERNEL_SIZE)

    def forward(self, x):
        return self.start(x)