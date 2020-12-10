import torch.nn as nn

from misc.misc import *

from model.layer_module import LayerModule
from utils.functional_utils import get_shift_matrix

class ChannelShifter(LayerModule):

    """ dim 1 -> dim 2, dim2 -> dim3 ... (we are doing this on a patch level) """
    """if jump_over_pixel (let internal_pixel_depth=2): dim1 -> dim3, dim2 -> dim4"""
    def __init__(self, internal_pixel_depth, total_pixel_depth, jump_over_pixels):
        super().__init__()

        shift_by = 1 if jump_over_pixels is False else internal_pixel_depth
        self.shift_matrix = get_shift_matrix(total_pixel_depth, shift_by)

        self.dummy = nn.Parameter(torch.zeros(1, device=DEVICE), requires_grad=False)

        print(f'Channel Shifter - Shift {total_pixel_depth} by {shift_by}')

    def forward(self, x : torch.Tensor):
        x = x.matmul(self.shift_matrix)

        return x
