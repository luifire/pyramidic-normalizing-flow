import numpy as np
import torch
import torch.nn as nn
from torch import distributions

from misc.misc import *
from misc.constants import *
from model.layer_module import LayerModule


class CutOff(LayerModule):

    """
    remove_pixel_count - if a pixel has 3 values (RGB) we remove 3 values!
    """
    def __init__(self, remove_pixel_count):
        super().__init__()

        self.remove_dimension_count = remove_pixel_count * PIXEL_DEPTH
        # dummy registration
        self.dummy = nn.Parameter(torch.zeros(1, device=DEVICE))

    def forward(self, x):
        channel_count = x.shape[CHANNEL_DIM]
        remains = x[:,:,:,0:channel_count - self.remove_dimension_count]
        cut_off = x[:,:,:,self.remove_dimension_count:]

        return remains, cut_off
