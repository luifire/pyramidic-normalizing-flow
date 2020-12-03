import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model.depth_conv_layer import *
from model.initial_reshaping import initialReshaping
from misc.constants import *



class PyramidFlowModel(nn.Module):

    def __init__(self):
        super().__init__()
        DepthConv(KERNEL_SIZE)
        self.conv_1 = DepthConvBundl(KERNEL_SIZE, KERNEL_SIZE_SQ)

    def forward(self, x):
        x = initialReshaping(x)
        x = self.conv_1(x)


        return x