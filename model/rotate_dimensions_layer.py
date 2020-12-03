import torch
import torch.nn as nn

class RotateDimensions(nn.Module):
    """ dim 1 -> dim 2, dim2 -> dim3 ... (we are doing this on a patch level) """
    def __init__(self, kernel_size):

        self.rotation_matrix = None

    def forward(self, x):
        pass