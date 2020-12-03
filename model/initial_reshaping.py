import torch
import torch.nn as nn

from misc.constants import *
from misc.misc import *

def initialReshaping(x):
    batch_size, channel_count, height, width = x.shape

    assert width % KERNEL_SIZE == 0
    assert height % KERNEL_SIZE == 0

    with torch.no_grad():
        new_x = torch.empty((batch_size, height // KERNEL_SIZE, width // KERNEL_SIZE, KERNEL_SIZE_SQ), device=DEVICE)
        print_shape("new_x", new_x)
        for h in range(height // KERNEL_SIZE):
            row = x.narrow(HEIGHT_DIM, h * KERNEL_SIZE, KERNEL_SIZE)
            for w in range(width // KERNEL_SIZE):
                patch = row.narrow(WIDTH_DIM, w * KERNEL_SIZE, KERNEL_SIZE)
                flattened = patch.reshape((batch_size, channel_count, KERNEL_SIZE_SQ))
                printt("flattened", flattened)

                new_x[:, h * KERNEL_SIZE:(h + 1) * KERNEL_SIZE, w * KERNEL_SIZE:(w + 1) * KERNEL_SIZE] = flattened

        return new_x.permute(0, 3, 1, 2)

    raise Exception("This shouldn't happen")