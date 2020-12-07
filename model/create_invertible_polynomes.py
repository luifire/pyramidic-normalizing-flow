import numpy as np
import torch
import torch.nn as nn
from torch import distributions

from misc.misc import *
from misc.constants import *
from model.layer_module import LayerModule

class InvertiblePolynome(LayerModule):

    def __init__(self):
        super().__init__()

        self.hole_filler_vector = None
        self.holed_shift_matrix = None

    def _get_holed_shift_matrix_and_filler_vector(self, total_dim):
        if self.holed_shift_matrix is not None and self.hole_filler_vector is not None:
            return self.holed_shift_matrix, self.hole_filler_vector

        shift_matrix = torch.zeros(total_dim, total_dim)
        hole_filler_matrix = torch.zeros(total_dim, total_dim)

        for i in range(total_dim):
            shift_matrix[i, (i + 1) % total_dim] = 1 * ((i + 1) % 2)
            hole_filler_matrix[i, i] = 1 * ((i + 1) % 2)

        zero_hole_filler = torch.ones(total_dim)
        hole_filler = zero_hole_filler.matmul(hole_filler_matrix)

        # shift_matrix is like
        # [[0., 1., 0., 0., 0., 0.,
        # [0., 0., 0., 0., 0., 0.,
        # [0., 0., 0., 1., 0., 0.,
        # [0., 0., 0., 0., 0., 0.,

        # hole filler is like
        # [1, 0, 1, 0, ...]
        self.holed_shift_matrix = shift_matrix
        self.hole_filler_vector = hole_filler

        return self.holed_shift_matrix, self.hole_filler_vector

    def forward(self, x: torch.Tensor):
        # an example for why this is correct is in misc.test_polynomes
        original_shape = x.shape
        flat = x.flatten(1, -1)
        total_dim = flat.shape[1]

        holed_shift_matrix, hole_filler_vector = self._get_holed_shift_matrix_and_filler_vector(total_dim)

        holed = flat.matmul(holed_shift_matrix)
        result = holed + hole_filler_vector
        x = result * flat

        logd_det = holed.abs().log().sum(1)

        x = x.reshape(original_shape)

        return x, logd_det
