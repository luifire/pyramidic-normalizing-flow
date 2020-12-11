import torch.nn as nn

from pyr_flow.constants import *
from pyr_flow.model.layer_module import LayerModule
from pyr_flow.utils.functional_utils import get_shift_matrix


class CutOff(LayerModule):

    """
    remove_pixel_count - if a pixel has 3 values (RGB) we remove 3 values!
    """
    def __init__(self, remaining_depth):
        super().__init__()

        self.remaining_depth = remaining_depth
        # dummy registration
        self.dummy = nn.Parameter(torch.zeros(1, device=DEVICE), requires_grad=False)

        print(f'Cut Off - Remaining Depth {remaining_depth}')

    def forward(self, x):
        remains = x[:,:,:,0:self.remaining_depth]
        cut_off = x[:,:,:,self.remaining_depth:]

        return remains, cut_off


###########################################################################
###########################################################################
###########################################################################

class ChannelShifter__(LayerModule):

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
        x = x.matmul(self.shift_matrix)
        #x = channel_normal_position(x)

        return x
