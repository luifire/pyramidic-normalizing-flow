import torch.nn as nn

from pyr_flow.model.layer_module import LayerModule

from pyr_flow.constants import *


class DepthConv(LayerModule):

    """  so we ignore all pixels from 1...pixel_idx (pixel 0 was just
    swapped from the bottom and thus needs to be filled) """
    def __init__(self, name, total_pixel_depth, internal_pixel_depth, jump_over_pixels, pixel_idx=-1):
        super().__init__()
        self.total_pixel_depth = total_pixel_depth
        self.internal_pixel_depth = internal_pixel_depth
        self.jump_over_pixels = jump_over_pixels
        self.name = name



        # Weight init kind of suggested by invertible conv flow
        weights = torch.normal(mean=0, std=0.0001, size=[total_pixel_depth, total_pixel_depth], device=DEVICE)
        weights += torch.eye(total_pixel_depth, device=DEVICE)

        self.weights = nn.Parameter(weights, requires_grad=True)

        self._create_identity_part(pixel_idx)

    def _create_identity_part(self, pixel_idx):
        # not relevant here
        if pixel_idx == 0:
            self.non_updateable_parameters = 0
            self.identity_start = 0
            return

        pixel_start = 1
        pixel_end = pixel_idx + 1
        if self.jump_over_pixels:
            pixel_start *= self.internal_pixel_depth
            pixel_end *= self.internal_pixel_depth

        self.identity_start = pixel_start
        self.identity_end = pixel_end

        identity = torch.eye(self.total_pixel_depth, device=DEVICE)
        self.identity_keeper_sub_matrix = identity[pixel_start:pixel_end]

        n = pixel_end - pixel_start
        self.non_updateable_parameters = 0.5 * (n ** 2 + n)

    def _prepare_weight_matrix_and_norm(self):

        # keep the values that have already been changed.
        # this is to hinder the network of making things more complicated
        if True and self.identity_start > 0:
            with torch.no_grad():
                self.weights[self.identity_start:self.identity_end] = self.identity_keeper_sub_matrix

        # make triangular matrix
        weights = torch.triu(self.weights)

        diag = torch.diagonal(weights)

        #logd_det = diag.abs().log().sum()
        logd_det = diag.abs().log()

        return weights, logd_det

    def forward(self, x : torch.Tensor, lnorm_map):
        _, height, width, _ = x.shape

        """
        if torch.isnan(self.weights).any():
            print("NAAAAAAAAAAAAAAAAAAAAAAAAAAAAN")
            print(self.name)
        if torch.isinf(self.weights).any():
            print("IIIIIIIIINF")
            print(self.name)
            
        print(x.shape)
        print(conv_matrix.shape)
        print(self.internal_pixel_depth)
        print(self.total_pixel_depth)
        print(self.name)
        """
        conv_matrix, logd_det = self._prepare_weight_matrix_and_norm()

        x = x.matmul(conv_matrix)

        # |det| == |det(Kernel)^amount_of_convolutions|
        # Note that the power part ccan be done like this
        # log(|a^b|) = log(|a|^b) = b log(|a|)
        #total_logd_det = logd_det * amount_of_convolutions

        lnorm_map += logd_det
        return x, lnorm_map

    def print_parameter(self):
        w = self.weights
        dia = w.diag()
        print(f"{self.name} diag min: {dia.min():.2e} max: {dia.max():.2e} prod: {dia.prod():.2e} "
              f"avg total {w.mean():.2e} inf min : {dia.abs().min():2e}")

    def get_parameter_count(self):
        dim = self.weights.shape[0]
        return (dim ** 2 + dim) / 2 - self.non_updateable_parameters
