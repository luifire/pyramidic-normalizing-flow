import torch
import torch.nn as nn

from misc.constants import *
from misc.misc import *


def initialReshaping(x):
    batch_size, channel_count, height, width = x.shape

    assert width % KERNEL_SIZE == 0
    assert height % KERNEL_SIZE == 0

    #printt("x start", x)
    with torch.no_grad():
        new_x = torch.empty((batch_size, height // KERNEL_SIZE, width // KERNEL_SIZE, KERNEL_SIZE_SQ * PIXEL_DEPTH),
                            device=DEVICE)
        #summer = torch.zeros(1, device=DEVICE)
        #print_shape("new_x", new_x)
        #row = x.narrow(HEIGHT_DIM, h * KERNEL_SIZE, KERNEL_SIZE)
        #patch = row.narrow(WIDTH_DIM, w * KERNEL_SIZE, KERNEL_SIZE)

        for h in range(height // KERNEL_SIZE):
            for w in range(width // KERNEL_SIZE):
                h_start = h*KERNEL_SIZE
                w_start = w*KERNEL_SIZE
                patch = x[:,:,h_start:h_start+KERNEL_SIZE, w_start:w_start+KERNEL_SIZE]
                # TODO: sollte man hier erst alle blauen, alle roten und alle grünen listen?
                flattened = patch.reshape((batch_size, KERNEL_SIZE_SQ * PIXEL_DEPTH))
                #if h == 7 and w == 3:
                #    print(str(h) + "   " + str(w))
                #    printt("patch", patch)
                #    printt("flattened", flattened)
                #    printt("new_x", new_x)

                #summer = summer.add(flattened.sum())
                #print_shape("new_x[:, h, w]", new_x[:, h, w])
                #print_shape("flattened", flattened)
                new_x[:, h, w] = flattened
                #if h == 7 and w == 3:
                #    printt("directly", new_x[:, h, w, :])
                #    printt("newer_x", new_x)

        #print(new_x.permute(0, 3, 1, 2))
        #printt("sum new", new_x.sum())
        #printt("summer", summer)
        #printt("x", x.sum())
        #exit(1)
        return new_x # .permute(0, 3, 1, 2)

    raise Exception("This shouldn't happen")