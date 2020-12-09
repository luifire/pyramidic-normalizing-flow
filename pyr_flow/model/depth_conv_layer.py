import torch.nn as nn
import torch

from model.shift_channel_layer import ChannelShifter
from model.layer_module import LayerModule

from misc.misc import *
from misc.constants import *


class DepthConv(LayerModule):

    """  so we ignore all pixels from 1...pixel_idx (pixel 0 was just
    swapped from the bottom and thus needs to be filled) """
    def __init__(self, name, channel_count, jump_over_pixels, pixel_idx=-1):
        super().__init__()
        self.channel_count = channel_count
        self.jump_over_pixels = jump_over_pixels
        #self.kernel_size_sq = channel_count ** 2
        self.name = name
        self.kernel_size = channel_count // PIXEL_DEPTH
        warn("this will not be correct, when we go deeper!")

        #total_kernel_size = self.kernel_size_sq*CHANNEL_COUNT
        # NOT: 1.5 kind of suggested by IFA-VAE
        # Weight init kind of suggested by invertible conv flow
        #weights = torch.normal(mean=1.5, std=0.5, size=[channel_count, channel_count], device=DEVICE)
        weights = torch.normal(mean=0, std=0.1, size=[channel_count, channel_count])
        weights += torch.eye(channel_count)# * torch.normal(mean=1, std=0.1, size=[channel_count])
        weights = weights.to(DEVICE)

        #printt("init weights", weights)
        self.weights = nn.Parameter(weights, requires_grad=True)  # nn.Parameter is a Tensor that's a module parameter.

        self.create_identity_part(pixel_idx)
        #bias = torch.normal(mean=1.5, std=0.5, size=[total_kernel_size], device=DEVICE)
        #self.bias = nn.Parameter(bias, requires_grad=True)  # nn.Parameter is a Tensor that's a module parameter.

    def create_identity_part(self, pixel_idx):
        # not relevant here
        if pixel_idx == 0:
            self.identity_start = 0
            return

        pixel_start = 1
        pixel_end = pixel_idx + 1
        if self.jump_over_pixels:
            pixel_start *= PIXEL_DEPTH
            pixel_end *= PIXEL_DEPTH

        self.identity_start = pixel_start
        self.identity_end = pixel_end

        identity = torch.eye(self.channel_count, device=DEVICE)
        self.identity_keeper_sub_matrix = identity[pixel_start:pixel_end]
        #printt("keep_identity", self.identity_keeper_sub_matrix)

    def _prepare_weight_matrix_and_norm(self):
        # make triangular matrix
        # printt("forward 1. weights", self.weights)
        #printt("device", self.weights.device)

        # keep the values that have already been changed.
        # this is to hinder the network of making things more complicated
        if True and self.identity_start > 0:
            with torch.no_grad():
                #printt("weights before", self.weights)
                self.weights[self.identity_start:self.identity_end] = self.identity_keeper_sub_matrix
                #printt("weights after", self.weights)

        weights = torch.triu(self.weights)

        diag = torch.diagonal(weights)

        if False and diag.prod().item() == 0:
            print(self.name)
            print(diag)

            print("diag has zeros")

        logd_det = diag.abs().log().sum()

        return weights, logd_det

    def forward(self, x : torch.Tensor):
        _, height, width, _ = x.shape

        conv_matrix, logd_det = self._prepare_weight_matrix_and_norm()
        #x = channel_to_last_dim(x)
        #warn("achtung beim splitten! [0] bekommt die wenigste info und [8] die meiste!!!")
        x = x.matmul(conv_matrix)

        # add bias at last
        #x = x + self.bias
        #x = channel_normal_position(x)

        # TODO speedup
        amount_of_convolutions = height * width
        # |det| == |det(Kernel)^amount_of_convolutions|
        # Note that the power part ccan be done like this
        # log(|a^b|) = log(|a|^b) = b log(|a|)
        total_logd_det = logd_det * amount_of_convolutions

        return x, total_logd_det

    def print_parameter(self):
        w = self.weights
        dia = w.diag().abs()
        print(f"{self.name} diag min: {dia.min():.3e} max: {dia.max():.3e} prod: {dia.prod():.3e} "
              f"avg total {w.mean():.3e}")

    def get_parameter_count(self):
        dim = self.weights.shape[0]
        return (dim ** 2 + dim) / 2
