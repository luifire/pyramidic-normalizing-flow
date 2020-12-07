import torch.nn as nn
import torch

from model.shift_channel_layer import ChannelShifter
from model.layer_module import LayerModule

from misc.misc import *
from misc.constants import *
from model.depth_conv_layer import DepthConv


class DepthConvBundle(LayerModule):

    def __init__(self, name, channel_count, bundle_size):
        super().__init__()
        self.bundle = nn.ModuleList()
        self.name = name
        self.channel_shifter = ChannelShifter(channel_count)
        bias = torch.normal(mean=1.5, std=0.5, size=[channel_count], device=DEVICE)
        self.bias = nn.Parameter(bias, requires_grad=True)  # nn.Parameter is a Tensor that's a module parameter.

        for i in range(bundle_size):
            conv_name = "Bundle " + name + " Conv " + str(i)
            self.bundle.append(DepthConv(conv_name, channel_count))

    def __call__(self, x):
        logd_norm_sums = torch.zeros(1, device=DEVICE)
        for i, layer in enumerate(self.bundle):
            x, log_norm = layer(x)
            logd_norm_sums = logd_norm_sums.add(log_norm)
            x = self.channel_shifter(x)

            #x = self.channel_shifter(x)
        #logd_norm_sum = self.logd_norms.sum()
        #self.logd_norms
        return x, logd_norm_sums

    def print_parameter(self):
        for layer in self.bundle:
            layer.print_parameter()
        print(f"bias mean {self.bias.mean(): .3e}")

    def get_parameter_count(self):
        return sum([layer.get_parameter_count() for layer in self.bundle]) + self.bias.shape[0]
