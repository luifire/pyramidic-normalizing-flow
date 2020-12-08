import numpy as np
import torch
import torch.nn as nn
from torch import distributions

from misc.misc import *
from misc.constants import *
from model.layer_module import LayerModule

from torch_helper.functional_stuff import get_shift_matrix


class InvertiblePolynome(LayerModule):

    def __init__(self):
        super().__init__()

        self.hole_filler_vector = None
        self.holed_shift_matrix = None
        self.dummy = nn.Parameter(torch.zeros(1, device=DEVICE))

    def _get_holed_shift_matrix_and_filler_vector(self, total_dim):
        if self.holed_shift_matrix is not None and self.hole_filler_vector is not None:
            return self.holed_shift_matrix, self.hole_filler_vector

        zeros = torch.zeros(total_dim, device=DEVICE)
        hole_filler = torch.zeros(total_dim, device=DEVICE)

        shift_by = total_dim // 2 + (total_dim % 2)
        # round up
        shift_matrix = get_shift_matrix(total_dim, shift_by)
        for i in range(shift_by):
            shift_matrix[total_dim-i-1] = zeros
            hole_filler[i] = 1
        #printt("shift", shift_matrix)
        #printt("hole_filler", hole_filler)

        self.holed_shift_matrix = shift_matrix
        self.hole_filler_vector = hole_filler

        #printt("shift_matrix", shift_matrix)
        #printt("hole_filler", hole_filler)
        return self.holed_shift_matrix, self.hole_filler_vector

    def forward(self, x: torch.Tensor):
        # an example for why this is correct is in misc.test_polynomes
        original_shape = x.shape
        #flat = x.flatten(1, -1)
        #total_dim = flat.shape[1]
        total_dim = original_shape[CHANNEL_DIM]

        holed_shift_matrix, hole_filler_vector = self._get_holed_shift_matrix_and_filler_vector(total_dim)

        holed = x.matmul(holed_shift_matrix)
        multiplier = holed + hole_filler_vector
        multiplier[multiplier==0] = 1
        #printt("x", x[0, 1:2,4])
        x = multiplier * x

        #printt("holed", holed[0,1:2,4])
        #printt("result", multiplier[0, 1:2,4])
        #printt("x danach", x[0, 1:2,4])

        # sum log |x|
        logd_det = multiplier.abs().log().sum(1).sum(1).sum(1)

        #x = x.reshape(original_shape)

        return x, logd_det
