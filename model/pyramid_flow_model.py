import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model.depth_conv_layer import *
from model.initial_reshaping import initialReshaping
from model.pyramid_loss import PyramidLoss
from model.s_log_gate import SLogGate
from model.layer_module import LayerModule
from model.cut_off_layer import CutOff
from model.conv_bundle import DepthConvBundle

from misc.constants import *


class PyramidFlowModel(LayerModule):

    def __init__(self):
        super().__init__()

        # TODO make this prettier
        self.layer_list = nn.ModuleList()
        channel_size = KERNEL_SIZE_SQ * CHANNEL_COUNT
        bundle_size = channel_size # s.t. every pixel consists of all others
        cut_off = 3
        #channel_size = 1 # warn
        self.layer_list.append(DepthConvBundle("1", channel_count=channel_size, bundle_size=bundle_size))
        self.layer_list.append(SLogGate())
        self.layer_list.append(DepthConvBundle("2", channel_count=channel_size, bundle_size=3))
        #self.layer_list.append(CutOff(cut_off))
        channel_size -= cut_off * CHANNEL_COUNT

    def forward(self, x):
        pyramid_steps = []
        x = initialReshaping(x)
        log_norm = torch.zeros(x.shape[0], device=DEVICE)
        for layer in self.layer_list:
            if type(layer) is CutOff:
                x, cut_off = layer(x)
                pyramid_steps.append(cut_off)
            else: # normal procedure
                x, x_log_norm = layer(x)
                log_norm += x_log_norm

        pyramid_steps.append(x)
        return pyramid_steps, log_norm

    def print_parameter(self):
        for layer in self.layer_list:
            layer.print_parameter()

    def get_parameter_count(self):
        return sum([layer.get_parameter_count() for layer in self.layer_list])
