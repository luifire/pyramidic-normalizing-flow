import torch
import torch.nn as nn

from misc.constants import *
from misc.misc import *

from model.layer_module import LayerModule
from utils.functional_utils import get_shift_matrix

class ChannelShifter(LayerModule):

    """ dim 1 -> dim 2, dim2 -> dim3 ... (we are doing this on a patch level) """
    def __init__(self, image_depth, jump_over_pixel):
        super().__init__()

        """
        shift_matrix = torch.zeros(image_depth, image_depth, device=DEVICE)
        for i in range(image_depth):
            shift_matrix[i, (i+1) % image_depth] = 1

        if jump_over_pixel:
            # for images with more channels we want to change the entire pixel
            for _ in range(PIXEL_DEPTH - 1): # ** CHANNEL_COUNT didn't work
                shift_matrix = shift_matrix.matmul(shift_matrix)
        """
        shift_by = 1 if jump_over_pixel is False else PIXEL_DEPTH
        self.shift_matrix = get_shift_matrix(image_depth, shift_by)

        self.dummy = nn.Parameter(torch.zeros(1, device=DEVICE))

    def forward(self, x : torch.Tensor):
        #x = channel_to_last_dim(x)
        warn("there is probably sth wrong")
        x = x.matmul(self.shift_matrix)
        #x = channel_normal_position(x)

        return x
