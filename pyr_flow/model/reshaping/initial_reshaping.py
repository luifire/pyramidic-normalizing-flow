from constants import *


def merge_patches(x, kernel_size):
    batch_size, height, width, pixel_depth = x.shape

    assert width % kernel_size == 0
    assert height % kernel_size == 0

    squared_kernel_size = kernel_size ** 2
    #with torch.no_grad():
    new_x = torch.empty((batch_size, height // kernel_size, width // kernel_size, squared_kernel_size * pixel_depth),
                        device=DEVICE)

    for h in range(height // kernel_size):
        for w in range(width // kernel_size):
            h_start = h * kernel_size
            w_start = w * kernel_size
            patch = x[:,h_start:h_start+kernel_size, w_start:w_start+kernel_size]
            # TODO: sollte man hier erst alle blauen, alle roten und alle grünen listen?
            flattened = patch.reshape((batch_size, squared_kernel_size * pixel_depth))

            new_x[:, h, w] = flattened

    return new_x

    #raise Exception("This shouldn't happen")